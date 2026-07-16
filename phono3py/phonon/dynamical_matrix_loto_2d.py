"""QE ``loto_2d`` dynamical matrix for q2r short-range force constants.

The input IFCs used here are in QE Ry/bohr**2 units and have already had
QE's reciprocal 2D term removed by ``q2r.x``.  This module deliberately does
not derive from :class:`DynamicalMatrixNAC`: phonopy's NAC classes implement a
different, three-dimensional long-range reconstruction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    get_dynamical_matrices_at_qpoints,
    get_dynamical_matrix,
)
from phonopy.harmonic.force_constants import symmetrize_force_constants
from phonopy.physical_units import get_physical_units


_BOHR_TO_ANGSTROM = 0.529177210903
_QE_FREQUENCY_FACTOR_THZ = 108.97077184367376
_GMAX = 14.0


class DynamicalMatrixQELoto2D(DynamicalMatrix):
    """Literal NumPy port of QE ``rigid:rgd_blk(..., loto_2d=.true.)``.

    Only an aligned two-dimensional cell is currently supported.  The
    correction uses QE's literal pair phase
    ``exp(2pi i (q+G).(tau_k-tau_k'))``.  The additional positive-sign
    atom-position factor was selected empirically by the A/B/C 152-point
    QE-matdyn comparison before the correction is added to the short-range
    matrix.
    """

    is_qe_loto_2d = True

    def __init__(
        self,
        supercell: Any,
        primitive: Any,
        force_constants: NDArray[np.double],
        nac_params: dict[str, Any],
        decimals: int | None = None,
        hermitianize: bool = True,
        log_level: int = 0,
        use_openmp: bool = False,
        lang: str = "Rust",
        **_: Any,
    ) -> None:
        if nac_params.get("method") != "qe_loto_2d":
            raise ValueError("DynamicalMatrixQELoto2D requires method='qe_loto_2d'.")
        self._nac_params_qe = nac_params
        self._born_qe = np.asarray(nac_params["born"], dtype="double")
        self._dielectric_qe = np.asarray(nac_params["dielectric"], dtype="double")
        self._alpha_qe = float(nac_params["alpha_ewald"])
        self._area_qe = float(nac_params["area"])
        self._c_qe = float(nac_params["loto_2d_c"])
        self._axes = tuple(int(i) for i in nac_params["loto_2d_periodic_axes"])
        self._normal_axis = int(nac_params.get("loto_2d_normal_axis", 2))
        self._factor_qe = float(nac_params["factor"])
        if len(self._axes) != 2 or self._normal_axis in self._axes:
            raise ValueError("qe_loto_2d needs exactly two periodic axes and one normal axis.")
        if self._born_qe.shape != (len(primitive), 3, 3):
            raise ValueError("born must have shape (n_primitive_atoms, 3, 3).")

        # QE matdyn's ``asr='crystal'`` is applied to the q2r short-range IFC
        # before rgd_blk is added.  Iterating phonopy's translational plus
        # permutation projection is the corresponding supercell operation.
        fc_qe = np.array(force_constants, dtype="double", copy=True, order="C")
        if fc_qe.shape[0] == fc_qe.shape[1]:
            symmetrize_force_constants(fc_qe, level=10, lang="C")
        else:
            raise ValueError("qe_loto_2d currently requires full supercell force constants.")

        # Phonopy's default force-unit convention is eV/Angstrom**2/amu.
        # Keep the public solver convention by converting QE's Ry/bohr**2
        # dynamical-matrix eigenvalues to that convention internally.
        self._qe_to_phonopy_fc = (
            _QE_FREQUENCY_FACTOR_THZ / get_physical_units().DefaultToTHz
        ) ** 2
        super().__init__(
            supercell,
            primitive,
            fc_qe * self._qe_to_phonopy_fc,
            decimals=decimals,
            hermitianize=hermitianize,
            use_openmp=use_openmp,
            lang=lang,  # type: ignore[arg-type]
        )
        self._tau = np.asarray(primitive.scaled_positions, dtype="double")
        # QE's alpha is expressed in (2pi/alat)^2.  The q2r convention uses
        # alat=|a1|, so convert it to bohr^-2 alongside our physical K vectors.
        alat_bohr = np.linalg.norm(np.asarray(primitive.cell[0])) / _BOHR_TO_ANGSTROM
        self._alpha_bohr2 = self._alpha_qe * (2 * np.pi / alat_bohr) ** 2
        # phonopy stores direct lattice vectors by *row*.  Consequently the
        # reciprocal lattice vectors are rows too and a crystallographic row
        # vector is transformed as ``q_red @ reciprocal``.  Keeping that
        # convention explicit avoids silently exchanging the off-diagonal
        # components in non-orthogonal (e.g. hexagonal) cells.
        self._reciprocal_bohr = (
            2
            * np.pi
            * np.linalg.inv(np.asarray(primitive.cell) / _BOHR_TO_ANGSTROM).T
        )
        self._onsite_qe: NDArray[np.cdouble] | None = None

    def run(self, q: Sequence[float] | NDArray[np.double]) -> None:
        """Build the QE-corrected dynamical matrix at one crystallographic q."""
        self._dynamical_matrix = self.get_dynamical_matrices(np.asarray(q, dtype="double"))

    def get_dynamical_matrices(self, qpoints: NDArray[np.double]) -> NDArray[np.cdouble]:
        """Return short-range plus QE 2D long-range matrices for qpoints."""
        qpts = np.asarray(qpoints, dtype="double")
        scalar = qpts.ndim == 1
        if scalar:
            qpts = qpts.reshape(1, 3)
        if qpts.ndim != 2 or qpts.shape[1] != 3:
            raise ValueError("qpoints must have shape (nq, 3).")
        if np.any(np.abs(qpts[:, self._normal_axis]) > 1e-10):
            raise ValueError("qe_loto_2d currently supports only q_normal = 0.")
        short = get_dynamical_matrices_at_qpoints(
            self, qpts, is_nac=False, hermitianize=True
        )
        if short.ndim == 2:
            short = short.reshape(1, *short.shape)
        correction_qe = self._get_qe_loto_2d_force_constant_blocks(qpts)
        masses = np.asarray(self.normalization_masses, dtype="double")
        inv_sqrt_mm = 1.0 / np.sqrt(masses[:, None] * masses[None, :])
        correction = correction_qe * inv_sqrt_mm[None, :, :, None, None]
        nat = len(masses)
        correction = correction.transpose(0, 1, 3, 2, 4).reshape(len(qpts), 3 * nat, 3 * nat)
        result = short + correction * self._qe_to_phonopy_fc
        result = 0.5 * (result + result.swapaxes(-1, -2).conj())
        return result[0] if scalar else result

    def _get_qe_loto_2d_force_constant_blocks(
        self, qpoints: NDArray[np.double]
    ) -> NDArray[np.cdouble]:
        """Return raw QE-Ry/bohr**2 LR blocks in the selected Bloch gauge.

        This is a vectorized transcription of the two loops in ``rgd_blk``:
        the first loop is the q-independent, atom-diagonal on-site drift
        subtraction and the second is the G+q pair term.
        """
        qpts = np.asarray(qpoints, dtype="double")
        if qpts.ndim == 1:
            qpts = qpts.reshape(1, 3)
        if self._onsite_qe is None:
            self._onsite_qe = self._build_onsite_subtraction()
        nat = len(self._tau)
        out = np.broadcast_to(-self._onsite_qe, (len(qpts), nat, nat, 3, 3)).copy()
        g_red = self._reciprocal_vectors_for(qpts)
        delta_tau = self._tau[:, None, :] - self._tau[None, :, :]
        for iq, q in enumerate(qpts):
            for g in g_red:
                reduced_k = q + g
                k = reduced_k @ self._reciprocal_bohr
                fac = self._kernel(k)
                if fac == 0.0:
                    continue
                z = np.einsum("i,kia->ka", k, self._born_qe)
                phase = np.exp(2j * np.pi * np.einsum("i,kli->kl", reduced_k, delta_tau))
                out[iq] += fac * phase[:, :, None, None] * z[:, None, :, None] * z[None, :, None, :]

            # The positive sign was selected by the A/B/C 152-point QE-matdyn
            # comparison: it gave the lowest max and RMS frequency residual.
            u = np.exp(2j * np.pi * (q @ self._tau.T))
            out[iq] *= u[:, None, None, None] * u[None, :, None, None].conj()
        return out

    def _build_onsite_subtraction(self) -> NDArray[np.cdouble]:
        nat = len(self._tau)
        onsite = np.zeros((nat, nat, 3, 3), dtype="cdouble")
        zero = np.zeros((1, 3), dtype="double")
        delta_tau = self._tau[:, None, :] - self._tau[None, :, :]
        for g in self._reciprocal_vectors_for(zero):
            k = g @ self._reciprocal_bohr
            fac = self._kernel(k)
            if fac == 0.0:
                continue
            z = np.einsum("i,kia->ka", k, self._born_qe)
            phase_cos = np.cos(2 * np.pi * np.einsum("i,kli->kl", g, delta_tau))
            for ka in range(nat):
                fnat = np.einsum("l,la->a", phase_cos[ka], z)
                block = np.outer(z[ka], fnat)
                onsite[ka, ka] += fac * 0.5 * (block + block.T)
        return onsite

    def _reciprocal_vectors_for(self, qpoints: NDArray[np.double]) -> NDArray[np.double]:
        # A deliberately generous rectangular enumeration; _kernel supplies
        # QE's exact strict exp-cutoff test for each q+G.
        shortest = min(np.linalg.norm(self._reciprocal_bohr[a]) for a in self._axes)
        nmax = int(np.ceil(np.sqrt(4 * _GMAX * self._alpha_bohr2) / shortest)) + 2
        integers = np.arange(-nmax, nmax + 1)
        g = np.zeros((len(integers) ** 2, 3), dtype="double")
        ii, jj = np.meshgrid(integers, integers, indexing="ij")
        g[:, self._axes[0]] = ii.ravel()
        g[:, self._axes[1]] = jj.ravel()
        return g

    def _kernel(self, k_bohr: NDArray[np.double]) -> float:
        k2 = float(np.dot(k_bohr, k_bohr))
        if k2 <= 1e-28 or k2 / (4 * self._alpha_bohr2) >= _GMAX:
            return 0.0
        kp = k_bohr[list(self._axes)]
        kp2 = float(np.dot(kp, kp))
        if kp2 <= 1e-28:
            return 0.0
        reff = 0.5 * self._c_qe * (
            self._dielectric_qe[np.ix_(self._axes, self._axes)] - np.eye(2)
        )
        grg = float(kp @ reff @ kp / kp2)
        return (
            self._factor_qe
            * 2 * np.pi
            / self._area_qe
            * np.exp(-k2 / (4 * self._alpha_bohr2))
            / (np.sqrt(k2) * (1.0 + grg * np.sqrt(k2)))
        )


def get_phph_dynamical_matrix(
    fc2: NDArray[np.double], supercell: Any, primitive: Any, nac_params: dict | None = None,
    **kwargs: Any,
) -> DynamicalMatrix:
    """Fork-local factory which leaves all non-QE-2D calls to phonopy."""
    if nac_params is not None and nac_params.get("method") == "qe_loto_2d":
        # frequency_scale_factor belongs to phonopy's factory; applying it
        # before this class's native-unit conversion preserves its semantics.
        scale = kwargs.pop("frequency_scale_factor", None)
        if scale is not None:
            fc2 = np.asarray(fc2) * float(scale) ** 2
        return DynamicalMatrixQELoto2D(supercell, primitive, fc2, nac_params, **kwargs)
    return get_dynamical_matrix(fc2, supercell, primitive, nac_params=nac_params, **kwargs)
