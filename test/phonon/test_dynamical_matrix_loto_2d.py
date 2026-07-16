"""Regression checks for the fork-local QE 2D long-range matrix."""

from pathlib import Path

import numpy as np
import phono3py
from phonopy.harmonic.dynamical_matrix import diagonalize_dynamical_matrices
from phonopy.physical_units import get_physical_units

from mlip_phonon_scattering.linewidth.matdyn_modes import load_matdyn_path, load_phband_freq

from phono3py.phonon.dynamical_matrix_loto_2d import DynamicalMatrixQELoto2D


_RUN = Path("/pscratch/sd/j/jdgeorga/ued/split_mlip_generators/runs/mose2_wse2_bilayer_dfpt_36x36x1_T50")
_COLLECT = Path(
    "/pscratch/sd/j/jdgeorga/ued/tdbe_paper_prod_speed_density_fine/"
    "2-mose2_wse2_6atoms/1-mf/ph_perq_d3fix/collect"
)


def _dm():
    ph3 = phono3py.load(_RUN / "phono3py_disp.yaml")
    nac_npz = np.load(_RUN / "nac_2d.npz")
    nac = {
        "method": "qe_loto_2d", "born": nac_npz["born"],
        "dielectric": nac_npz["dielectric"],
        "alpha_ewald": float(nac_npz["alpha_ewald"]),
        "loto_2d_c": float(nac_npz["c"]), "area": float(nac_npz["area"]),
        "loto_2d_periodic_axes": tuple(int(i) for i in nac_npz["periodic_axes"]),
        "loto_2d_normal_axis": 2, "factor": float(nac_npz["factor"]),
    }
    return DynamicalMatrixQELoto2D(
        ph3.phonon_supercell, ph3.phonon_primitive, np.load(_RUN / "dfpt_fc2.npy"), nac
    )


def _literal_rgd_blk_loto_2d(dm, q):
    """Scalar-loop transcription of QE rigid.f90's ``rgd_blk`` 2D branch.

    This deliberately avoids array products and broadcasting: it is a small,
    independent indexing/gauge oracle for the production vectorized kernel.
    All Cartesian quantities are in bohr-based physical units.
    """
    nat = len(dm._tau)
    result = np.zeros((nat, nat, 3, 3), dtype="cdouble")
    onsite = np.zeros_like(result)
    zero_g = dm._reciprocal_vectors_for(np.zeros((1, 3)))

    for g in zero_g:
        k = np.zeros(3)
        for cart in range(3):
            for reduced in range(3):
                k[cart] += g[reduced] * dm._reciprocal_bohr[reduced, cart]
        fac = dm._kernel(k)
        if fac == 0.0:
            continue
        z = np.zeros((nat, 3))
        for atom in range(nat):
            for cart in range(3):
                for direction in range(3):
                    z[atom, cart] += k[direction] * dm._born_qe[atom, direction, cart]
        for atom in range(nat):
            fnat = np.zeros(3)
            for other in range(nat):
                dot = 0.0
                for direction in range(3):
                    dot += g[direction] * (dm._tau[atom, direction] - dm._tau[other, direction])
                phase_cos = np.cos(2 * np.pi * dot)
                for cart in range(3):
                    fnat[cart] += phase_cos * z[other, cart]
            for cart_i in range(3):
                for cart_j in range(3):
                    onsite[atom, atom, cart_i, cart_j] += fac * 0.5 * (
                        z[atom, cart_i] * fnat[cart_j] + fnat[cart_i] * z[atom, cart_j]
                    )

    for atom in range(nat):
        for other in range(nat):
            for cart_i in range(3):
                for cart_j in range(3):
                    result[atom, other, cart_i, cart_j] = -onsite[atom, other, cart_i, cart_j]

    for g in zero_g:
        reduced_k = np.zeros(3)
        for direction in range(3):
            reduced_k[direction] = q[direction] + g[direction]
        k = np.zeros(3)
        for cart in range(3):
            for reduced in range(3):
                k[cart] += reduced_k[reduced] * dm._reciprocal_bohr[reduced, cart]
        fac = dm._kernel(k)
        if fac == 0.0:
            continue
        z = np.zeros((nat, 3))
        for atom in range(nat):
            for cart in range(3):
                for direction in range(3):
                    z[atom, cart] += k[direction] * dm._born_qe[atom, direction, cart]
        for atom in range(nat):
            for other in range(nat):
                dot = 0.0
                for direction in range(3):
                    dot += reduced_k[direction] * (
                        dm._tau[atom, direction] - dm._tau[other, direction]
                    )
                phase = np.exp(2j * np.pi * dot)
                for cart_i in range(3):
                    for cart_j in range(3):
                        result[atom, other, cart_i, cart_j] += (
                            fac * phase * z[atom, cart_i] * z[other, cart_j]
                        )

    u = np.zeros(nat, dtype="cdouble")
    for atom in range(nat):
        q_dot_tau = 0.0
        for direction in range(3):
            q_dot_tau += q[direction] * dm._tau[atom, direction]
        u[atom] = np.exp(2j * np.pi * q_dot_tau)
    for atom in range(nat):
        for other in range(nat):
            for cart_i in range(3):
                for cart_j in range(3):
                    result[atom, other, cart_i, cart_j] *= u[atom] * u[other].conjugate()
    return result


def test_qe_loto_2d_is_hermitian_and_time_reversal_symmetric():
    dm = _dm()
    q = np.array([0.137, 0.071, 0.0])
    dq = dm.get_dynamical_matrices(q)
    dminus = dm.get_dynamical_matrices(-q)
    assert np.max(abs(dq - dq.T.conj())) < 1e-12
    # The selected gauge preserves the real-space time-reversal relation.
    assert np.max(abs(dminus - dq.conj())) < 1e-11


def test_qe_loto_2d_gamma_acoustic_and_no_directional_nac():
    dm = _dm()
    d0 = dm.get_dynamical_matrices(np.zeros(3))
    eigvals, _ = diagonalize_dynamical_matrices(d0[None], with_eigenvectors=True)
    assert np.max(abs(eigvals[0, :3])) < 1e-10
    # Exact Gamma evaluates the G-only part and has no q-direction argument.
    assert np.max(abs(d0 - d0.T.conj())) < 1e-12


def test_qe_loto_2d_vectorized_kernel_matches_literal_rgd_blk_loops():
    dm = _dm()
    for q in (
        np.array([0.137, 0.071, 0.0]),
        np.array([-0.219, 0.164, 0.0]),
        np.array([0.333, -0.121, 0.0]),
        np.array([0.041, 0.287, 0.0]),
    ):
        literal = _literal_rgd_blk_loto_2d(dm, q)
        vectorized = dm._get_qe_loto_2d_force_constant_blocks(q)[0]
        np.testing.assert_allclose(vectorized, literal, rtol=0.0, atol=1e-10)


def test_qe_loto_2d_matches_qe_matdyn_phband():
    """The complete QE ``matdyn`` Gamma-M-K-Gamma path is an accuracy gate."""
    qpoints = load_matdyn_path(_COLLECT / "matdyn.in")
    reference = load_phband_freq(_COLLECT / "phband.freq")
    eigvals, _ = diagonalize_dynamical_matrices(
        _dm().get_dynamical_matrices(qpoints), with_eigenvectors=False
    )
    units = get_physical_units()
    frequencies = (
        np.sign(eigvals)
        * np.sqrt(np.abs(eigvals))
        * units.DefaultToTHz
        * units.THzToCm
    )
    residual = frequencies - reference
    max_residual = float(np.max(np.abs(residual)))
    rms_residual = float(np.sqrt(np.mean(residual**2)))
    assert max_residual < 0.5, f"max |Delta w| = {max_residual:.6f} cm^-1"
    assert rms_residual < 0.1, f"RMS |Delta w| = {rms_residual:.6f} cm^-1"
