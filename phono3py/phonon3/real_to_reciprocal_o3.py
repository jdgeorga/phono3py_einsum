"""Fast real→reciprocal‑space transform of FC3 (exact phonon‑3py re‑implementation).

This is a single‑file drop‑in replacement for the original Phono3py C routine.  It
vectorises *all* inner loops with NumPy and collapses the j–k summations with
``np.einsum``; only pure‑Python that remains is very small book‑keeping.

The public API matches the reference implementation:

>>> r2r = RealToReciprocalExact(fc3, primitive, mesh)
>>> r2r.run(triplet)               # triplet is (3,3) int array of grid IDs
>>> fc3q = r2r.get_fc3_reciprocal()  # (3N,3N,3N) complex128

The speed‑up comes mainly from
  • vectorised phase‑factor tables  (no satom×patom Python loops)
  • a single ``einsum`` that accumulates the weighted FC3
  • avoiding per‑Cartesian‑component slicing – the 27 components ride an extra
    axis so they cost nothing extra.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ------------------------------------------------------------ helpers
PI2 = 2.0 * np.pi


# ------------------------------------------------------------ main class
class RealToReciprocalO3:
    """Transform *fc3* from real to reciprocal space (NumPy, exact formula)."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        fc3: NDArray,           # (S, S, S, 3, 3, 3) real  (super‑cell indices)
        primitive,              # phonopy Primitive object (has p2s_map, etc.)
        mesh: NDArray,          # (3,) int  – Γ‑centred Monkhorst–Pack mesh
        *,
        symprec: float = 1e-5,
        make_r0_average: bool = True,
    ) -> None:
        self._primitive = primitive
        self._mesh = np.asarray(mesh, dtype=int)
        self._symprec = symprec
        self._make_r0_average = bool(make_r0_average)

        self._fc3: NDArray = fc3.reshape((*fc3.shape[:3], 27))  # (S,S,S,27)
        self._p2s_map: NDArray = primitive.p2s_map               # (P,)
        self._s2p_map: NDArray = primitive.s2p_map               # (S,)
        self._svecs, self._multi = primitive.get_smallest_vectors()

        # _svecs        → (N_multi, 3)
        # _multi[s,p,0] → multiplicity count (int)
        # _multi[s,p,1] → starting index into _svecs (int)

        # optional arrays (Phono3py sets them at runtime)
        self._all_shortest = getattr(primitive, "all_shortest", None)
        if self._all_shortest is None:
            P, S = len(self._p2s_map), len(self._s2p_map)
            self._all_shortest = np.zeros((P, S, S), dtype=bool)

        # treat fc3 as fully dense – if a sparse mask exists, pass it in later
        self._nonzero_indices = np.ones(self._fc3.shape[:3], dtype=bool)

        # place‑holders filled in run()
        self._fc3_reciprocal: NDArray | None = None

    # -------------------------------------------------------------- public
    def run(self, triplet: NDArray) -> None:  # (3,3) int  – grid triplet
        """Build ``self._fc3_reciprocal`` for the supplied *triplet* of q IDs."""
        q_vecs = triplet.astype(float) / self._mesh[None, :]   # (3,3)

        # 1) phase factors & pre‑phases -------------------------------------------------
        phase_tbl = self._build_phase_tables(q_vecs)           # (P,S,3) complex128
        pre_phase = self._get_pre_phase_factors(q_vecs)        # (P,)   complex128

        # 2) core contraction -----------------------------------------------------------
        self._fc3_reciprocal = self._contract_fc3(phase_tbl, pre_phase)

    def get_fc3_reciprocal(self) -> NDArray:                    # (3P,3P,3P)
        if self._fc3_reciprocal is None:
            raise RuntimeError("run() has not been called yet.")
        return self._fc3_reciprocal

    # ---------------------------------------------------------------- private helpers
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ vectorised phase factors ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _build_phase_tables(self, q_vecs: NDArray) -> NDArray:  # (P,S,3)
        """Return *phase_factor[patom, satom, leg]*.

        *leg* index: 0,1,2 correspond to q¹, q², q³ in the triplet.
        """
        svecs, multi = self._svecs, self._multi
        P, S = len(self._p2s_map), len(self._s2p_map)

        # flatten (s,p) → 1‑D so we can do a single matmul & exp
        counts = multi[..., 0].astype(int).ravel()   # (S*P,)
        starts = multi[..., 1].astype(int).ravel()   # (S*P,)

        # build the block index that slices svecs for every replica vector
        replica_idx = np.repeat(starts, counts) 
        phases = PI2 * (svecs[replica_idx] @ q_vecs.T)
        eiphi = np.exp(1j * phases)                  # (..,3)

        # average over multiplicities with bincount (real & imag separately)
        seg = np.repeat(np.arange(starts.size), counts)
        tbl_real = np.empty((starts.size, 3))
        tbl_imag = np.empty_like(tbl_real)
        for leg in range(3):
            tbl_real[:, leg] = np.bincount(seg, eiphi[:, leg].real) / counts
            tbl_imag[:, leg] = np.bincount(seg, eiphi[:, leg].imag) / counts

        phase_tbl = (tbl_real + 1j * tbl_imag).reshape(S, P, 3).swapaxes(0, 1)
        # now  (P,S,3)
        return phase_tbl.astype(np.complex128)

    def _get_pre_phase_factors(self, q_vecs: NDArray) -> NDArray:  # (P,)
        """Vectorised form of the tiny loop in the C reference code.

        Uses **one** shortest vector for each primitive atom, matching the C
        implementation (multiplicity[*,1] → first entry).
        """
        P = len(self._p2s_map)
        # locate the *first* shortest vector for each primitive atom
        satom_idx = self._p2s_map                      # (P,)
        start_idx = self._multi[satom_idx, 0, 1].astype(int)  # (P,)
        svec = self._svecs[start_idx]                  # (P,3)
        q_sum = q_vecs.sum(axis=0)                     # (3,)
        phase = PI2 * (svec @ q_sum)                  # (P,)
        return np.exp(1j * phase, dtype=np.complex128)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ core contraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _contract_fc3(self, phase_tbl: NDArray, pre_phase: NDArray) -> NDArray:
        """Perform the ∑_{j,k} contraction and leg permutations in one shot."""
        P, S = phase_tbl.shape[:2]
        fc3_flat = self._fc3                          # (S,S,S,27)
     
        i_mesh = np.arange(P).repeat(P*P)        
        # build the selection mask once:  M[p,s] == 1  ⇔ s2p_map[s] == p2s_map[p]
        M = (self._s2p_map[None, :] == self._p2s_map[:, None])  # (P,S)

        # masks & phase factors per leg ------------------------------------------------
        phi0 = phase_tbl[..., 0]                      # (P,S)
        phi1 = phase_tbl[..., 1]
        phi2 = phase_tbl[..., 2]

        # broadcast & mask so shape becomes (P,P,S) and (P,P,S)
        phi1M = phi1[:, None, :] * M[None, :, :]      # φ1M[i,j,s]
        phi2M = phi2[:, None, :] * M[None, :, :]      # φ2M[i,k,s] later

        # compact weight tensor --------------------------------------------------------
        weights = np.ones_like(P*P*P,S,S,27)      # (S,S,S)
  
        if self._make_r0_average: 
            weights += 2.0 * self._all_shortest.repeat(P*P, axis=0)      # +2 → net×3 as in C code
        weights *= self._nonzero_indices
        fc3w = fc3_flat * weights[..., None]          # (S,S,S,27)

        # first leg (i fixed) -----------------------------------------------------------
        # einsum indices:  i j k p
        tmp = np.einsum(
            "ijkp,ai,iBj,ck->aBcp",
            fc3w,
            np.conj(pre_phase),        # a  (P,)
            phi1M,                    # iBj (P,P,S)
            phi2M,                    # ck  (P,P,S)
            optimize="greedy",
        )                               # (P,P,P,27)

        # add the other two permutations (leg‑2 and leg‑3)
        rec = tmp + np.transpose(tmp, (1, 0, 2, 0)) + np.transpose(tmp, (2, 1, 0, 0))
        rec /= 3.0                                     # final r0‑average factor

        # reshape (P,3) → 3P
        rec = rec.reshape(P, 1, P, 1, P, 1, 27)        # insert axis for cart comps
        rec = rec.repeat(3, axis=1).repeat(3, axis=3).repeat(3, axis=5)
        return rec.reshape(3 * P, 3 * P, 3 * P)
