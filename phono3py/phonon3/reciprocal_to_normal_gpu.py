# phonopy_codes/phono3py_einsum/phono3py/phonon3/reciprocal_to_normal_gpu.py
"""CuPy-based reciprocal-to-normal transform (|V|^2) for fc3.

GPU implementation of the v3 C-equivalent algorithm: mass-weighting of
eigenvectors, full contraction in band space, and division by
(freq0 * freq1 * freq2) for each (band0, band1, band2) triple.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError as exc:
    raise ImportError(
        "reciprocal_to_normal_gpu requires CuPy. "
        "Install cupy-cudaXX matching your CUDA version."
    ) from exc


class ReciprocalToNormalSquaredGPU:
    """GPU implementation of reciprocal-to-normal transform.

    Parameters
    ----------
    primitive : Primitive
        Phonopy primitive cell (for masses).
    frequencies : ndarray
        Phonon frequencies on the BZ grid, shape=(num_bz_grid, num_band).
    eigenvectors : ndarray
        Phonon eigenvectors on the BZ grid, shape=(num_bz_grid, num_band, num_band).
    band_indices : array_like
        Band indices at q0 whose interaction strength we keep as the
        leading index in |V|^2.
    cutoff_frequency : float
        Modes below this frequency are ignored.
    """

    def __init__(
        self,
        primitive,
        frequencies: np.ndarray,
        eigenvectors: np.ndarray,
        band_indices,
        cutoff_frequency: float = 0.0,
    ) -> None:
        self._primitive = primitive
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._band_indices = np.array(band_indices, dtype=np.int64)
        self._cutoff_frequency = float(cutoff_frequency)
        self._fc3_normal_squared: np.ndarray | None = None
        self._on_gpu = False

    def run(
        self,
        fc3_reciprocal: np.ndarray,
        grid_triplet: np.ndarray,
        method: str = "super_fast",
        return_gpu: bool = False,
    ) -> None:
        """Compute |V|^2 for a given triplet.

        Parameters
        ----------
        fc3_reciprocal : ndarray
            Third-order force constants in reciprocal space for this triplet.
            Shape may be (num_atom, num_atom, num_atom, 3, 3, 3) or
            (num_band, num_band, num_band); it will be reshaped internally.
        grid_triplet : ndarray
            Grid-point indices [q0, q1, q2] in BZGrid indexing.
        method : {'super_fast', 'einsum'}
            'super_fast' uses a single large einsum; 'einsum' uses a more
            conservative algorithm. Both run on GPU.
        """
        q0, q1, q2 = map(int, grid_triplet)

        freqs0 = self._frequencies[q0]
        freqs1 = self._frequencies[q1]
        freqs2 = self._frequencies[q2]

        eig0 = self._eigenvectors[q0]
        eig1 = self._eigenvectors[q1]
        eig2 = self._eigenvectors[q2]

        masses = np.array(self._primitive.masses, dtype=np.float64)

        if method.lower() == "super_fast":
            fc3_ns = reciprocal_to_normal_squared_gpu_super_fast(
                fc3_reciprocal,
                freqs0,
                freqs1,
                freqs2,
                eig0,
                eig1,
                eig2,
                masses,
                self._band_indices,
                self._cutoff_frequency,
                return_gpu=return_gpu,
            )
        elif method.lower() in ("einsum", "optimized"):
            fc3_ns = reciprocal_to_normal_squared_gpu_einsum(
                fc3_reciprocal,
                freqs0,
                freqs1,
                freqs2,
                eig0,
                eig1,
                eig2,
                masses,
                self._band_indices,
                self._cutoff_frequency,
                return_gpu=return_gpu,
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'super_fast' or 'einsum'."
            )

        self._fc3_normal_squared = fc3_ns
        self._on_gpu = return_gpu

    def get_reciprocal_to_normal_squared(self) -> np.ndarray | None:
        """Return |V|^2 array, shape=(num_band0, num_band, num_band)."""
        return self._fc3_normal_squared


def _as_fc3_6d(fc3_reciprocal_cp: cp.ndarray, num_atom: int) -> cp.ndarray:
    """Convert fc3 to (na, na, na, 3, 3, 3) format.
    
    Handles two input formats:
    - (nb, nb, nb) from RealToReciprocalExact: needs reshape + transpose
    - (na, na, na, 3, 3, 3) already in correct format: pass through
    
    The (nb, nb, nb) format has fc3[atom0*3+cart0, atom1*3+cart1, atom2*3+cart2].
    We need to convert to fc3[atom0, atom1, atom2, cart0, cart1, cart2].
    """
    num_band = num_atom * 3
    if fc3_reciprocal_cp.shape == (num_band, num_band, num_band):
        # Input is (nb, nb, nb) - need reshape and transpose
        # Reshape to (na, 3, na, 3, na, 3) then transpose to (na, na, na, 3, 3, 3)
        return fc3_reciprocal_cp.reshape(
            num_atom, 3, num_atom, 3, num_atom, 3
        ).transpose(0, 2, 4, 1, 3, 5)
    elif fc3_reciprocal_cp.shape == (num_atom, num_atom, num_atom, 3, 3, 3):
        # Already in correct format
        return fc3_reciprocal_cp
    else:
        raise ValueError(
            f"Unexpected fc3_reciprocal shape: {fc3_reciprocal_cp.shape}. "
            f"Expected ({num_band}, {num_band}, {num_band}) or "
            f"({num_atom}, {num_atom}, {num_atom}, 3, 3, 3)."
        )


def _as_fc3_3d(fc3_reciprocal_cp: cp.ndarray, num_atom: int) -> cp.ndarray:
    """Convert fc3 to (nb, nb, nb) mode-indexed format.
    
    Handles two input formats:
    - (na, na, na, 3, 3, 3): needs transpose + reshape  
    - (nb, nb, nb) already in correct mode-indexed format: pass through
    
    Output has fc3[mode0, mode1, mode2] where mode = atom*3 + cart.
    """
    num_band = num_atom * 3
    if fc3_reciprocal_cp.shape == (num_atom, num_atom, num_atom, 3, 3, 3):
        # Input is 6D - need transpose and reshape
        # From (na, na, na, 3, 3, 3) to (na, 3, na, 3, na, 3) to (nb, nb, nb)
        return fc3_reciprocal_cp.transpose(0, 3, 1, 4, 2, 5).reshape(
            num_band, num_band, num_band
        )
    elif fc3_reciprocal_cp.shape == (num_band, num_band, num_band):
        # Already in correct format
        return fc3_reciprocal_cp
    else:
        raise ValueError(
            f"Unexpected fc3_reciprocal shape: {fc3_reciprocal_cp.shape}. "
            f"Expected ({num_atom}, {num_atom}, {num_atom}, 3, 3, 3) or "
            f"({num_band}, {num_band}, {num_band})."
        )


def reciprocal_to_normal_squared_gpu_super_fast(
    fc3_reciprocal: np.ndarray,
    freqs0: np.ndarray,
    freqs1: np.ndarray,
    freqs2: np.ndarray,
    eigvecs0: np.ndarray,
    eigvecs1: np.ndarray,
    eigvecs2: np.ndarray,
    masses: np.ndarray,
    band_indices: np.ndarray,
    cutoff_frequency: float = 0.0,
    return_gpu: bool = False,
) -> np.ndarray:
    """Fully vectorized CuPy einsum implementation.

    - Mass weight eigenvectors: 1/sqrt(m_i).
    - Contract over atoms and Cartesian indices to get complex amplitudes
      A[I,J,K] for all bands.
    - Return |A|^2 / (ω0_I * ω1_J * ω2_K), restricted to band_indices on the
      first index.
    """
    num_atom = len(masses)
    num_band = num_atom * 3
    band_indices = np.array(band_indices, dtype=np.int64)

    # Mass weighting
    m_cp = cp.asarray(masses, dtype=cp.float64)  # (na,)
    inv_sqrt_m = 1.0 / cp.sqrt(m_cp)            # (na,)

    # Eigenvectors (nb, nb) -> (na,3,nb)
    e0 = cp.asarray(eigvecs0, dtype=cp.complex128).reshape(num_atom, 3, num_band)
    e1 = cp.asarray(eigvecs1, dtype=cp.complex128).reshape(num_atom, 3, num_band)
    e2 = cp.asarray(eigvecs2, dtype=cp.complex128).reshape(num_atom, 3, num_band)

    inv_sqrt_m = inv_sqrt_m[:, None, None]  # (na,1,1)
    e0_m = e0 * inv_sqrt_m
    e1_m = e1 * inv_sqrt_m
    e2_m = e2 * inv_sqrt_m

    # fc3: (na,na,na,3,3,3)
    fc3_cp = cp.asarray(fc3_reciprocal, dtype=cp.complex128)
    fc3_cp = _as_fc3_6d(fc3_cp, num_atom)

    # Contract: iaI, jbJ, kcK, ijkabc -> IJK
    A = cp.einsum(
        "iaI,jbJ,kcK,ijkabc->IJK",
        e0_m,
        e1_m,
        e2_m,
        fc3_cp,
        optimize=True,
    )  # (nb, nb, nb)

    # Frequency denominators
    f0 = cp.asarray(freqs0, dtype=cp.float64)
    f1 = cp.asarray(freqs1, dtype=cp.float64)
    f2 = cp.asarray(freqs2, dtype=cp.float64)

    # Apply cutoff
    valid0 = f0 > cutoff_frequency
    valid1 = f1 > cutoff_frequency
    valid2 = f2 > cutoff_frequency

    denom = (
        f0[:, None, None] * f1[None, :, None] * f2[None, None, :]
    )  # (nb,nb,nb)
    denom = cp.where(
        (valid0[:, None, None] & valid1[None, :, None] & valid2[None, None, :]),
        denom,
        0.0,
    )

    V2 = cp.abs(A) ** 2  # (nb,nb,nb)

    # Avoid div-by-zero
    out = cp.zeros_like(V2, dtype=cp.float64)
    mask = denom > 0.0
    out[mask] = (V2[mask] / denom[mask]).real

    # Restrict to band_indices on the first axis
    band_idx_cp = cp.asarray(band_indices, dtype=cp.int64)
    out_sel = out[band_idx_cp, :, :]  # (nb0, nb, nb)

    if return_gpu:
        return out_sel
    return cp.asnumpy(out_sel)


def reciprocal_to_normal_squared_gpu_einsum(
    fc3_reciprocal: np.ndarray,
    freqs0: np.ndarray,
    freqs1: np.ndarray,
    freqs2: np.ndarray,
    eigvecs0: np.ndarray,
    eigvecs1: np.ndarray,
    eigvecs2: np.ndarray,
    masses: np.ndarray,
    band_indices: np.ndarray,
    cutoff_frequency: float = 0.0,
    return_gpu: bool = False,
) -> np.ndarray:
    """CuPy implementation that mirrors the v3 'einsum' algorithm.

    This is a bit more conservative than `super_fast` but uses the same
    mass weighting and division by ω0 ω1 ω2.
    
    Accepts fc3 in either:
    - (na, na, na, 3, 3, 3) format from RealToReciprocal
    - (nb, nb, nb) mode-indexed format from RealToReciprocalExact
    """
    num_atom = len(masses)
    num_band = num_atom * 3
    band_indices = np.array(band_indices, dtype=np.int64)
    num_band0 = len(band_indices)

    m_cp = cp.asarray(masses, dtype=cp.float64)
    inv_sqrt_m = cp.repeat(1.0 / cp.sqrt(m_cp), 3)  # (nb,)

    def _transpose_mass_weight(ev_np: np.ndarray) -> cp.ndarray:
        ev = cp.asarray(ev_np, dtype=cp.complex128)  # (nb, nb)
        return ev.T * inv_sqrt_m[None, :]

    e0 = _transpose_mass_weight(eigvecs0)
    e1 = _transpose_mass_weight(eigvecs1)
    e2 = _transpose_mass_weight(eigvecs2)

    # Convert fc3 to 3D mode-indexed format (nb, nb, nb)
    fc3_cp = cp.asarray(fc3_reciprocal, dtype=cp.complex128)
    fc3_band = _as_fc3_3d(fc3_cp, num_atom)  # (nb, nb, nb)

    fc3_e0 = cp.empty((num_band0, num_band, num_band), dtype=cp.complex128)
    for i, bi in enumerate(band_indices):
        fc3_e0[i] = cp.einsum("j,jkl->kl", e0[bi], fc3_band, optimize=True)

    f0 = cp.asarray(freqs0, dtype=cp.float64)
    f1 = cp.asarray(freqs1, dtype=cp.float64)
    f2 = cp.asarray(freqs2, dtype=cp.float64)

    valid0 = f0 > cutoff_frequency
    valid1 = f1 > cutoff_frequency
    valid2 = f2 > cutoff_frequency

    out = cp.zeros((num_band0, num_band, num_band), dtype=cp.float64)

    for i0 in range(num_band0):
        bi0 = band_indices[i0]
        if not valid0[bi0]:
            continue
        fc3_e0_i = fc3_e0[i0]  # (nb, nb)
        for j in range(num_band):
            if not valid1[j]:
                continue
            fc3_e0_e1 = cp.einsum("ij,i->j", fc3_e0_i, e1[j], optimize=True)
            for k in range(num_band):
                if not valid2[k]:
                    continue
                total = cp.einsum("i,i->", fc3_e0_e1, e2[k], optimize=True)
                num = cp.abs(total) ** 2
                denom = f0[bi0] * f1[j] * f2[k]
                if denom > 0.0:
                    out[i0, j, k] = float(num / denom)

    if return_gpu:
        return out
    return cp.asnumpy(out)