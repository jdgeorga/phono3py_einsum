"""Calculate ph-ph interaction and phonons on grid."""

# Copyright (C) 2020 Atsushi Togo
# All rights reserved.
#
# This file is part of phono3py.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
import time
import numpy as np
import multiprocessing as mp
from functools import partial
from numpy.typing import NDArray
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, get_dynamical_matrix
from phonopy.physical_units import get_physical_units
from phonopy.structure.cells import Primitive, compute_all_sg_permutations
from phonopy.structure.symmetry import Symmetry

from phonopy.phonon.grid import (
    BZGrid,
    get_grid_points_by_rotations,
    get_ir_grid_points,
)
from phono3py.phonon.solver import run_phonon_solver_c, run_phonon_solver_py
from phono3py.phonon3.real_to_reciprocal_fast import RealToReciprocalFast
from phono3py.phonon3.real_to_reciprocal_o3 import RealToReciprocalO3
from phono3py.phonon3.r2r_chunk import RealToReciprocalChunked
from phono3py.phonon3.reciprocal_to_normal_fast import ReciprocalToNormalFast
from phono3py.phonon3.reciprocal_to_normal import ReciprocalToNormal
from phono3py.phonon3.real_to_reciprocal import RealToReciprocal
from phono3py.phonon3.real_to_reciprocal_v3 import RealToReciprocalV3
from phono3py.phonon3.real_to_reciprocal_exact import RealToReciprocalExact
from phono3py.phonon3.reciprocal_to_normal_v3 import ReciprocalToNormalSquaredV3
from phono3py.phonon3.triplets import get_nosym_triplets_at_q, get_triplets_at_q
from phono3py.phonon3.real_to_reciprocal_gpu import RealToReciprocalExactGPU
from phono3py.phonon3.reciprocal_to_normal_gpu import ReciprocalToNormalSquaredGPU


def _process_triplet_worker(args):
    """Worker function to process a single triplet."""
    try:
        (i, grid_triplet, fc3, primitive, mesh_numbers, symprec, make_r0_average, 
         frequencies, eigenvectors, band_indices, cutoff_frequency, unit_conversion,
         bz_grid_addresses, existing_fc3_reciprocal, phonon_done, all_shortest) = args
        
   
        r2r = RealToReciprocalExact(
            fc3, primitive, mesh_numbers, symprec=symprec, make_r0_average=make_r0_average,
            all_shortest=all_shortest
        )
        
        # Create ReciprocalToNormalSquaredV3 instance for this worker
        r2n = ReciprocalToNormalSquaredV3(
            primitive,
            frequencies,
            eigenvectors,
            band_indices,
            cutoff_frequency=cutoff_frequency,
        )
        
        print(f"Processing triplet {i + 1}", flush=True)
        start_time = time.time()
        
        # Run r2r transformation
        r2r.run(bz_grid_addresses[grid_triplet])
        r2r_time = time.time() - start_time
        
        # Get fc3_reciprocal
        if np.sum(np.abs(existing_fc3_reciprocal)) < 1e-5:
            fc3_reciprocal = r2r.get_fc3_reciprocal()
            if fc3_reciprocal is not None:
                fc3_reciprocal = fc3_reciprocal.reshape(existing_fc3_reciprocal.shape)
            else:
                fc3_reciprocal = existing_fc3_reciprocal
        else:
            fc3_reciprocal = existing_fc3_reciprocal
        
        # Run r2n transformation
        start_time = time.time()
        r2n.run(fc3_reciprocal, grid_triplet, method='super_fast')
        r2n_time = time.time() - start_time
        
        fc3_normal_squared = r2n.get_reciprocal_to_normal_squared()
        
        if fc3_normal_squared is not None:
            interaction_strength = fc3_normal_squared * unit_conversion
        else:
            interaction_strength = None
        
        print(f"Triplet {i + 1} completed - r2r: {r2r_time:.4f}s, r2n: {r2n_time:.4f}s", flush=True)
        
        return i, fc3_reciprocal, interaction_strength, grid_triplet
        
    except Exception as e:
        print(f"Error processing triplet {i + 1}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return i, None, None, grid_triplet


class Interaction:
    """Calculate ph-ph interaction and phonons on grid.

    This class instance is the heart of phono3py calculation.
    Many data are stored.

    The following three steps have to be done manually.
    1) init_dynamical_matrix
    2) set_grid_point
    3) run

    Attributes
    ----------
    interaction_strength
    mesh_numbers
    is_mesh_symmetry
    fc3
    dynamical_matrix
    primitive
    primitive_symmetry
    bz_grid
    band_indices
    nac_params
    nac_q_direction
    zero_value_positions
    frequency_factor_to_THz
    lapack_zheev_uplo
    cutoff_frequency
    symmetrize_fc3q
    make_r0_average

    """

    def __init__(
        self,
        primitive: Primitive,
        bz_grid: BZGrid,
        primitive_symmetry: Symmetry,
        fc3: NDArray | None = None,
        fc3_nonzero_indices: NDArray | None = None,
        band_indices: NDArray | Sequence | None = None,
        constant_averaged_interaction: float | None = None,
        frequency_factor_to_THz: float | None = None,
        frequency_scale_factor: float | None = None,
        unit_conversion: float | None = None,
        is_mesh_symmetry: bool = True,
        symmetrize_fc3q: bool = False,
        make_r0_average: bool = False,
        cutoff_frequency: float | None = None,
        lapack_zheev_uplo: Literal["L", "U"] = "L",
        openmp_per_triplets: bool | None = None,
        lang=None,
    ):
        """Init method."""
        self._primitive = primitive
        self._bz_grid = bz_grid
        self._primitive_symmetry = primitive_symmetry
        self._lang = lang

        self._band_indices = self._get_band_indices(band_indices)
        self._constant_averaged_interaction = constant_averaged_interaction
        if frequency_factor_to_THz is None:
            self._frequency_factor_to_THz = get_physical_units().DefaultToTHz
        else:
            self._frequency_factor_to_THz = frequency_factor_to_THz
        self._frequency_scale_factor = frequency_scale_factor

        if fc3 is not None:
            self._set_fc3(fc3, fc3_nonzero_indices=fc3_nonzero_indices)

        # Unit to eV^2
        if unit_conversion is None:
            num_grid = np.prod(self.mesh_numbers)
            self._unit_conversion = float(
                (get_physical_units().Hbar * get_physical_units().EV) ** 3
                / 36
                / 8
                * get_physical_units().EV ** 2
                / get_physical_units().Angstrom ** 6
                / (2 * np.pi * get_physical_units().THz) ** 3
                / get_physical_units().AMU ** 3
                / num_grid
                / get_physical_units().EV ** 2
            )
        else:
            self._unit_conversion = unit_conversion
        if cutoff_frequency is None:
            self._cutoff_frequency = 0
        else:
            self._cutoff_frequency = cutoff_frequency
        self._is_mesh_symmetry = is_mesh_symmetry
        self._symmetrize_fc3q = symmetrize_fc3q
        self._make_r0_average = make_r0_average
        self._lapack_zheev_uplo: Literal["L", "U"] = lapack_zheev_uplo
        self._openmp_per_triplets = openmp_per_triplets

        self._symprec = self._primitive_symmetry.tolerance

        self._triplets_at_q = None
        self._weights_at_q = None
        self._triplets_map_at_q = None
        self._ir_map_at_q = None
        self._interaction_strength = None
        self._interaction_strength_gpu = None  # CuPy array, kept on GPU for efficiency
        self._g_zero = None

        self._phonon_done = None
        self._phonon_all_done = False
        self._done_nac_at_gamma = False  # Phonon at Gamma is calculated with NAC.
        self._frequencies = None
        self._eigenvectors = None
        self._frequencies_at_gamma = None
        self._eigenvectors_at_gamma = None
        self._dm = None
        self._nac_params = None
        self._nac_q_direction = None

        self._band_index_count = 0

        self._svecs, self._multi = self._primitive.get_smallest_vectors()
        self._masses = np.array(self._primitive.masses, dtype="double")
        self._p2s = np.array(self._primitive.p2s_map, dtype="int64")
        self._s2p = np.array(self._primitive.s2p_map, dtype="int64")
        n_satom, n_patom, _ = self._multi.shape
        self._all_shortest = np.zeros(
            (n_patom, n_satom, n_satom), dtype="byte", order="C"
        )
        self._get_all_shortest()
        self._fc3_reciprocal = None

    def run(self,
            lang: Literal["C", "Python", "Fast", "V3", "GPU", "GPU_phase", "Hybrid"] = "C",
            g_zero: NDArray | None = None,
            num_processes: int | None = None
    ):
        """Run ph-ph interaction calculation."""

        if self._phonon_all_done:
            self.run_phonon_solver()

        if self._triplets_at_q is None:
            raise RuntimeError("Set grid point first by set_grid_point().")

        # This fork's Interaction implements the backends enumerated below.
        # Upstream v4.2.0 introduced a "Rust" default (and a "Python" option)
        # that this fork never implemented; normalize any such value to the
        # standard "C" path so ph-ph setup never crashes on a non-GPU default.
        if lang not in ("C", "Fast", "V3", "GPU", "GPU_phase", "Hybrid"):
            lang = "C"

        num_band = len(self._primitive) * 3
        num_triplets = len(self._triplets_at_q)

        # Allocate fc3(q) storage only for CPU-style backends.
        # The GPU and Hybrid paths never require the full
        # (num_triplets, natom, natom, natom, 3, 3, 3) array and
        # preallocating it can cost tens of GB for large systems.
        #
        # For GPU / Hybrid, fc3 is kept and processed entirely on
        # device via RealToReciprocalExactGPU, so we skip this
        # allocation to avoid huge host RAM usage.
        if self._fc3_reciprocal is None and lang in ("C", "Fast", "V3"):
            self._fc3_reciprocal = np.zeros(
                (num_triplets,
                 len(self._primitive),
                 len(self._primitive),
                 len(self._primitive),
                 3, 3, 3),
                dtype="complex128",
            )

        self._interaction_strength = np.empty(
            (num_triplets, len(self._band_indices), num_band, num_band), dtype="double"
        )
        
        if self._constant_averaged_interaction is None:
            self._interaction_strength[:] = 0
            if lang == "C":
                self._run_c(g_zero)
            elif lang == "Fast":
                self._run_fast(g_zero, num_processes)
            elif lang == "V3":
                self._run_v3(g_zero)
            elif lang == "GPU":
                self._run_gpu(g_zero)
            elif lang == "GPU_phase":
                self._run_gpu(g_zero, phase_vectorized=True)
            elif lang == "Hybrid":
                self._run_hybrid(g_zero)
            else:
                # Unreachable: lang is normalized to a supported backend above.
                # (The old default here, _run_py_test(), was dev-only
                # Python-vs-GPU reference scaffolding assuming the 3.15.1 fc3
                # layout — not a production path.)
                raise NotImplementedError(f"Unsupported interaction lang: {lang!r}")
        else:
            num_grid = np.prod(self.mesh_numbers)
            self._interaction_strength[:] = (
                self._constant_averaged_interaction / num_grid
            )

    @property
    def interaction_strength(self) -> NDArray | None:
        """Return ph-ph interaction strength.

        Returns
        -------
        ndarray
            shape=(num_ir_grid_points, num_specified_band, num_band, num_band),
            dtype='double', order='C'

        """
        return self._interaction_strength

    @property
    def interaction_strength_gpu(self):
        """Return ph-ph interaction strength as CuPy array (if available).
        
        Returns None if GPU computation wasn't used or data isn't on GPU.
        """
        return self._interaction_strength_gpu

    @property
    def mesh_numbers(self) -> NDArray:
        """Return mesh numbers.

        Returns
        -------
        ndarray
           shape=(3, ), dtype='int64'

        """
        return self._bz_grid.D_diag

    @property
    def is_mesh_symmetry(self) -> bool:
        """Whether symmetry of grid is utilized or not."""
        return self._is_mesh_symmetry

    @property
    def fc3(self) -> NDArray:
        """Return fc3."""
        return self._fc3

    @property
    def fc3_nonzero_indices(self) -> NDArray:
        """Return fc3_nonzero_indices."""
        return self._fc3_nonzero_indices

    @property
    def fc3_reciprocal(self) -> NDArray | None:
        """Return fc3_reciprocal.
        
        Returns
        -------
        ndarray or None
            shape=(num_triplets, num_atoms, num_atoms, num_atoms, 3, 3, 3),
            dtype='complex128', order='C'
            
        """
        return self._fc3_reciprocal

    @fc3_reciprocal.setter
    def fc3_reciprocal(self, fc3_reciprocal):
        self._fc3_reciprocal = fc3_reciprocal

    @property
    def dynamical_matrix(self) -> DynamicalMatrix | None:
        """Return DynamicalMatrix class instance."""
        return self._dm

    @property
    def primitive(self) -> Primitive:
        """Return Primitive class instance."""
        return self._primitive

    @property
    def primitive_symmetry(self) -> Symmetry:
        """Return Symmetry class instance of primitive cell."""
        return self._primitive_symmetry

    @property
    def phonon_all_done(self) -> bool:
        """Return whether phonon calculation is done at all grid points.

        True value is set in run_phonon_solver(). Even when False, it is
        possible that all phonons are already calculated, but it is safer to
        think some of phonons are not calculated yet. To be sure, check
        (self._phonon_done == 0).any().

        """
        return self._phonon_all_done

    def get_triplets_at_q(
        self,
    ) -> tuple[
        NDArray | None,
        NDArray | None,
        NDArray | None,
        NDArray | None,
    ]:
        """Return grid point triplets information.

        triplets_at_q is in BZ-grid.
        triplets_map_at_q is in GR-grid.
        ir_map_at_q is in GR-grid.
        See details at ``get_triplets_at_q``.

        """
        return (
            self._triplets_at_q,
            self._weights_at_q,
            self._triplets_map_at_q,
            self._ir_map_at_q,
        )

    @property
    def bz_grid(self) -> BZGrid:
        """Return BZGrid class instance."""
        return self._bz_grid

    @property
    def band_indices(self) -> NDArray:
        """Return band indices.

        Returns
        -------
        ndarray
            shape=(num_specified_bands, ), dtype='int64'

        """
        return self._band_indices

    @property
    def nac_params(self) -> dict | None:
        """Return NAC params."""
        return self._nac_params

    @property
    def nac_q_direction(self) -> NDArray | None:
        """Return q-direction used for NAC at q->0.

        Direction of q-vector watching from Gamma point used for
        non-analytical term correction. This is effective only at q=0
        (physically q->0). The direction is given in crystallographic
        (fractional) coordinates.
        shape=(3,), dtype='double'.
        Default value is None, which means this feature is not used.

        """
        return self._nac_q_direction

    @nac_q_direction.setter
    def nac_q_direction(self, nac_q_direction):
        if nac_q_direction is None:
            self._nac_q_direction = None
        else:
            self._nac_q_direction = np.array(nac_q_direction, copy=True, dtype="double")

    @property
    def zero_value_positions(self) -> NDArray | None:
        """Return zero ph-ph interaction elements information.

        Returns
        -------
        shape is same as that of interaction_strength, dtype='byte', order='C'

        """
        return self._g_zero

    def get_phonons(
        self,
    ) -> tuple[NDArray | None, NDArray | None, NDArray | None]:
        """Return phonons on grid.

        Returns
        -------
        tuple
            frequencies : ndarray
                Phonon frequencies on grid.
                shape=(num_bz_grid, num_band), dtype='double', order='C'
            eigenvectors : ndarray
                Phonon eigenvectors on grid.
                shape=(num_bz_grid, num_band, num_band),
                dtype="c%d" % (np.dtype('double').itemsize * 2), order='C'
            phonon_done : ndarray
                1 if phonon at a grid point is calculated, otherwise 0.
                shape=(num_bz_grid, ), dtype='byte'

        """
        return self._frequencies, self._eigenvectors, self._phonon_done

    @property
    def frequency_factor_to_THz(self) -> float:
        """Return phonon frequency conversion factor to THz."""
        return self._frequency_factor_to_THz

    @property
    def lapack_zheev_uplo(self) -> Literal["L", "U"]:
        """Return U or L for lapack zheev solver."""
        return self._lapack_zheev_uplo

    @property
    def cutoff_frequency(self) -> float:
        """Return cutoff phonon frequency to judge imaginary phonon."""
        return self._cutoff_frequency

    @property
    def openmp_per_triplets(self) -> bool | None:
        """Return whether OpenMP distribution over triplets or bands."""
        return self._openmp_per_triplets

    @property
    def symmetrize_fc3q(self) -> bool:
        """Return boolean of symmetrize_fc3q."""
        return self._symmetrize_fc3q

    @property
    def make_r0_average(self) -> bool:
        """Return boolean of make_r0_average.

        This flag is used to activate averaging of fc3 transformation
        from real space to reciprocal space around three atoms. With False,
        it is done at the first atom. With True, it is done at three atoms
        and averaged.

        """
        return self._make_r0_average

    @property
    def all_shortest(self) -> NDArray:
        """Return boolean of make_r0_average.

        This flag is used to activate averaging of fc3 transformation
        from real space to reciprocal space around three atoms. With False,
        it is done at the first atom. With True, it is done at three atoms
        and averaged.

        """
        return self._all_shortest

    @property
    def averaged_interaction(self) -> NDArray:
        """Return sum over phonon triplets of interaction strength.

        See Eq.(21) of PRB 91, 094306 (2015)

        """
        if self._interaction_strength is None or self._weights_at_q is None:
            raise RuntimeError(
                "Interaction strength and weights at q are not set. "
                "Run Interaction.run() first."
            )

        # v[triplet, band0, band, band]
        v = self._interaction_strength
        w = self._weights_at_q
        v_sum = np.dot(w, v.sum(axis=2).sum(axis=2))
        return v_sum / np.prod(v.shape[2:])

    def get_primitive_and_supercell_correspondence(
        self,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Return atomic pair information."""
        return (self._svecs, self._multi, self._p2s, self._s2p, self._masses)

    @property
    def unit_conversion_factor(self) -> float:
        """Return unit conversion factor."""
        return self._unit_conversion

    @property
    def constant_averaged_interaction(self) -> float | None:
        """Return constant averaged interaction."""
        return self._constant_averaged_interaction

    def set_interaction_strength(self, pp_strength, g_zero=None):
        """Set interaction strength."""
        self._interaction_strength = pp_strength
        self._g_zero = g_zero
        
    def set_grid_point(self, grid_point, store_triplets_map=False):
        """Set grid point and prepare grid point triplets."""
        if not self._is_mesh_symmetry:
            (
                triplets_at_q,
                weights_at_q,
                triplets_map_at_q,
                ir_map_at_q,
            ) = get_nosym_triplets_at_q(grid_point, self._bz_grid)
        else:
            # Special treatment of symmetry is applied when q_direction is used
            # at Gamma point = (0 0 0).
            if (
                self._bz_grid.addresses[grid_point] == 0
            ).all() and self._nac_q_direction is not None:
                rotations = []
                for i, r in enumerate(self._bz_grid.reciprocal_operations):
                    dq = self._nac_q_direction
                    dq /= np.linalg.norm(dq)
                    diff = np.dot(r, dq) - dq
                    if (abs(diff) < 1e-5).all():
                        rotations.append(self._bz_grid.rotations[i])
                (
                    triplets_at_q,
                    weights_at_q,
                    triplets_map_at_q,
                    ir_map_at_q,
                ) = get_triplets_at_q(
                    grid_point,
                    self._bz_grid,
                    reciprocal_rotations=rotations,
                    is_time_reversal=False,
                )
            else:
                (
                    triplets_at_q,
                    weights_at_q,
                    triplets_map_at_q,
                    ir_map_at_q,
                ) = get_triplets_at_q(grid_point, self._bz_grid)

            # Re-calculate phonon at Gamma-point when q-direction is given.
            if (self._bz_grid.addresses[grid_point] == 0).all():
                self.run_phonon_solver_at_gamma(is_nac=True)
            elif self._done_nac_at_gamma:
                if self._nac_q_direction is None:
                    self.run_phonon_solver_at_gamma()
                else:
                    msg = (
                        "Phonons at Gamma has been calculated with NAC, "
                        "but ph-ph interaction is expected to calculate at "
                        "non-Gamma point. Setting Interaction.nac_q_direction = "
                        "None, can avoid raising this exception to re-run phonon "
                        "calculation at Gamma without NAC."
                    )
                    raise RuntimeError(msg)

        reciprocal_lattice = np.linalg.inv(self._primitive.cell)
        for triplet in triplets_at_q:
            sum_q = (self._bz_grid.addresses[triplet]).sum(axis=0)
            if (sum_q % self.mesh_numbers != 0).any():
                print("============= Warning ==================")
                print("%s" % triplet)
                for tp in triplet:
                    print(
                        "%s %s"
                        % (
                            self._bz_grid.addresses[tp],
                            np.linalg.norm(
                                np.dot(
                                    reciprocal_lattice,
                                    self._bz_grid.addresses[tp]
                                    / self.mesh_numbers.astype("double"),
                                )
                            ),
                        )
                    )
                print("%s" % sum_q)
                print("============= Warning ==================")

        self._triplets_at_q = triplets_at_q
        self._weights_at_q = weights_at_q
        self._fc3_reciprocal = None  # must be reallocated for new triplet count

        if store_triplets_map:
            self._triplets_map_at_q = triplets_map_at_q
            self._ir_map_at_q = ir_map_at_q

    def init_dynamical_matrix(
        self,
        fc2,
        supercell,
        primitive,
        nac_params=None,
        decimals=None,
    ):
        """Prepare for phonon calculation on grid.

        solve_dynamical_matrices : bool
           When False, phonon calculation will be postponed.

        """
        self._nac_params = nac_params
        self._dm = get_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=self._frequency_scale_factor,
            decimals=decimals,
        )
        self._allocate_phonon()

    def set_phonon_data(self, frequencies, eigenvectors, bz_grid_addresses):
        """Set phonons on grid."""
        if bz_grid_addresses.shape != self._bz_grid.addresses.shape:
            raise RuntimeError(
                "Input grid address size is inconsistent. Setting phonons failed."
            )

        if (self._bz_grid.addresses - bz_grid_addresses).all():
            raise RuntimeError(
                "Input grid addresses are inconsistent. Setting phonons failed."
            )
        else:
            if (
                self._phonon_done is None
                or self._frequencies is None
                or self._eigenvectors is None
            ):
                raise RuntimeError(
                    "Phonons are not initialized. Call init_dynamical_matrix() first."
                )
            self._phonon_done[:] = 1
            self._frequencies[:] = frequencies
            self._eigenvectors[:] = eigenvectors
            gp_Gamma = self._bz_grid.gp_Gamma
            self._frequencies_at_gamma = self._frequencies[gp_Gamma].copy()
            self._eigenvectors_at_gamma = self._eigenvectors[gp_Gamma].copy()

    def run_phonon_solver(self, grid_points=None, solve_by_rotation=False):
        """Run phonon solver at BZ-grid points."""
        if grid_points is None:
            if solve_by_rotation:
                self.run_phonon_solver_with_eigvec_rotation()
            else:
                self._run_phonon_solver_c(
                    np.arange(len(self._bz_grid.addresses), dtype="int64")
                )
            self._phonon_all_done = True
        else:
            self._run_phonon_solver_c(grid_points)

    def run_phonon_solver_at_gamma(self, is_nac=False):
        """Run phonon solver at Gamma point.

        Run phonon solver at Gamma point with/without NAC. When `self._nac_q_direction`
        is None, always without NAC. `self._nac_q_direction` will be unchanged in any
        case.

        Parameters
        ----------
        is_nac : bool, optional
            With NAC when is_nac is True and `self._nac_q_direction` is not None,
            otherwise without NAC. Default is False.

        """
        if (
            self._phonon_done is None
            or self._frequencies is None
            or self._eigenvectors is None
        ):
            raise RuntimeError(
                "Phonons are not initialized. Call init_dynamical_matrix() first."
            )

        if not is_nac and self._frequencies_at_gamma is not None:
            gp_Gamma = self._bz_grid.gp_Gamma
            self._frequencies[gp_Gamma] = self._frequencies_at_gamma
            self._eigenvectors[gp_Gamma] = self._eigenvectors_at_gamma
            return

        self._phonon_done[self._bz_grid.gp_Gamma] = 0
        if is_nac:
            self._done_nac_at_gamma = True
            self.run_phonon_solver(np.array([self._bz_grid.gp_Gamma], dtype="int64"))
        else:
            self._done_nac_at_gamma = False
            _nac_q_direction = self._nac_q_direction
            self._nac_q_direction = None
            self.run_phonon_solver(np.array([self._bz_grid.gp_Gamma], dtype="int64"))
            self._nac_q_direction = _nac_q_direction

    def run_phonon_solver_with_eigvec_rotation(self):
        """Phonons at ir-grid-points are copied by proper rotations.

        Some phonons that are not covered by rotations are solved.

        The following data are updated.
            self._frequencies
            self._eigenvectors
            self._phonon_done

        """
        if self._phonon_done is None:
            raise RuntimeError(
                "Phonons are not initialized. Call init_dynamical_matrix() first."
            )

        self._phonon_done[:] = 0
        ir_grid_points, _, _ = get_ir_grid_points(self._bz_grid)
        ir_bz_grid_points = self._bz_grid.grg2bzg[ir_grid_points]
        self.run_phonon_solver(grid_points=ir_bz_grid_points)

        d2r_map = self._get_reciprocal_rotations_in_space_group_operations()

        # perms.shape = (len(spg_ops), len(primitive)), dtype='intc'
        perms = compute_all_sg_permutations(
            self._primitive.scaled_positions,
            self._bz_grid.grid_symmetry_dataset.rotations,  # type: ignore
            self._bz_grid.grid_symmetry_dataset.translations,  # type: ignore
            np.array(self._primitive.cell.T, dtype="double", order="C"),
            symprec=self._symprec,
            # phonopy 4.3 added a required `types` arg (atom permutation types);
            # matches upstream phono3py fc3.py's supercell.permutation_types.
            types=self._primitive.permutation_types,
        )

        for d_i, r_i in enumerate(d2r_map):
            r = self._bz_grid.rotations[r_i]
            r_cart = self._bz_grid.rotations_cartesian[r_i]
            for irgp in ir_bz_grid_points:
                bzgp = get_grid_points_by_rotations(
                    irgp,
                    self._bz_grid,
                    reciprocal_rotations=[
                        r,
                    ],
                    with_surface=True,
                )[0]
                if self._phonon_done[bzgp]:
                    continue

                self._rotate_eigvecs(irgp, bzgp, r_cart, perms[d_i], d_i)

        bz_grid_points_solved = self._get_phonons_at_minus_q()
        if bz_grid_points_solved:
            print("DEBUG: BZ-grid points additionally solved than ir-grid-points.")
            qpoints = np.dot(
                self._bz_grid.addresses[bz_grid_points_solved]
                / self._bz_grid.D_diag.astype("double"),
                self._bz_grid.Q.T,
            )
            distances = np.linalg.norm(
                np.dot(qpoints, np.linalg.inv(self._primitive.cell).T), axis=1
            )
            for qpt, dist in zip(qpoints, distances):
                print(qpt, dist)

    def _get_reciprocal_rotations_in_space_group_operations(self):
        """Collect reciprocal rotations that belong to space group operations.

        Exclude reciprocal rotations that are made by time reversal symmetry.

        Returns
        -------
        d2r_map : list
            Indices of reciprocal rotations.

        """
        d2r_map = []
        for r in self._bz_grid.grid_symmetry_dataset.rotations:  # type: ignore
            for i, rec_r in enumerate(self._bz_grid.reciprocal_operations):
                if (rec_r.T == r).all():
                    d2r_map.append(i)
                    break

        assert len(d2r_map) == len(self._bz_grid.grid_symmetry_dataset.rotations)  # type: ignore

        return d2r_map

    def _rotate_eigvecs(self, orig_gp, bzgp, r_cart, perm, t_i):
        r"""Rotate eigenvectors at q to those Rq.

        e_j'(Rq) = R e_j(q) exp(-iRq.\tau)

        """
        assert self._phonon_done is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None

        Rq = np.dot(self._bz_grid.QDinv, self._bz_grid.addresses[bzgp])
        tau = self._bz_grid.grid_symmetry_dataset.translations[t_i]  # type: ignore
        phase_factor = np.exp(-2j * np.pi * np.dot(Rq, tau))
        self._phonon_done[bzgp] = 1
        self._frequencies[bzgp, :] = self._frequencies[orig_gp, :]
        eigvecs = self._eigenvectors[orig_gp, :, :] * phase_factor
        for i, vec in enumerate(eigvecs.T):
            vec_perm = vec.reshape(-1, 3)[perm, :].T
            vec_rot = np.dot(r_cart, vec_perm).T.ravel()
            self._eigenvectors[bzgp, :, i] = vec_rot

    def _get_phonons_at_minus_q(self):
        """Phonons at -q are given by phonons at q.

        A few points may be uncovered by rotations. Those points are counted.

        Returns
        -------
        bz_grid_points_solved : list of int
            BZ-grid points where phonons that were additionally solved
            in this method.

        """
        assert self._phonon_done is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None

        r_inv = -np.eye(3, dtype="int64")
        bz_grid_points_solved = []
        for bzgp, done in enumerate(self._phonon_done):
            if done:
                continue

            # Get grid point at -q.
            bzgp_mq = get_grid_points_by_rotations(
                bzgp,
                self._bz_grid,
                reciprocal_rotations=[
                    r_inv,
                ],
                with_surface=True,
            )[0]

            if self._phonon_done[bzgp_mq] == 0:
                self.run_phonon_solver(
                    grid_points=np.array(
                        [
                            bzgp_mq,
                        ],
                        dtype="int64",
                    )
                )
                bz_grid_points_solved.append(bzgp_mq)

            self._phonon_done[bzgp] = 1
            self._frequencies[bzgp, :] = self._frequencies[bzgp_mq, :]
            self._eigenvectors[bzgp, :, :] = np.conj(self._eigenvectors[bzgp_mq, :, :])

        assert (self._phonon_done == 1).all()

        return bz_grid_points_solved

    def delete_interaction_strength(self):
        """Delete large arrays loosely.

        Memory deallocation would rely on garbage collector of python.
        So this may not work as expected.

        """
        self._interaction_strength = None
        self._g_zero = None
        self._fc3_reciprocal = None

    def _set_fc3(self, fc3: NDArray, fc3_nonzero_indices: NDArray | None = None):
        if (
            isinstance(fc3, np.ndarray)
            and fc3.dtype == np.dtype("double")
            and fc3.flags.aligned
            and fc3.flags.owndata
            and fc3.flags.c_contiguous
            and self._frequency_scale_factor is None
        ):
            self._fc3 = fc3
        elif self._frequency_scale_factor is None:
            self._fc3 = np.array(fc3, dtype="double", order="C")
        else:
            self._fc3 = np.array(
                fc3 * self._frequency_scale_factor**2, dtype="double", order="C"
            )

        if fc3_nonzero_indices is None:
            self._fc3_nonzero_indices = np.ones(
                self._fc3.shape[:3], dtype="byte", order="C"
            )
        elif (
            isinstance(fc3_nonzero_indices, np.ndarray)
            and fc3_nonzero_indices.dtype == np.dtype("byte")
            and fc3_nonzero_indices.flags.aligned
            and fc3_nonzero_indices.flags.owndata
            and fc3_nonzero_indices.flags.c_contiguous
        ):
            self._fc3_nonzero_indices = fc3_nonzero_indices
        else:
            self._fc3_nonzero_indices = np.array(
                fc3_nonzero_indices, dtype="byte", order="C"
            )

    def _get_band_indices(self, band_indices) -> NDArray:
        num_band = len(self._primitive) * 3
        if band_indices is None:
            return np.arange(num_band, dtype="int64")
        else:
            return np.array(band_indices, dtype="int64")

    def _run_c(self, g_zero):
        import phono3py._phono3py as phono3c

        assert self._interaction_strength is not None
        assert self._triplets_at_q is not None
        assert self._fc3_reciprocal is not None
        
        num_band = len(self._primitive) * 3
        if g_zero is None or self._symmetrize_fc3q:
            _g_zero = np.zeros(
                self._interaction_strength.shape,
                dtype="byte",
                order="C",
            )
        else:
            _g_zero = g_zero

        # True: OpenMP over triplets
        # False: OpenMP over bands
        if self._openmp_per_triplets is None:
            if len(self._triplets_at_q) > num_band:
                openmp_per_triplets = True
            else:
                openmp_per_triplets = False
        else:
            openmp_per_triplets = self._openmp_per_triplets

        phono3c.interaction(
            self._interaction_strength,
            self._fc3_reciprocal,
            _g_zero,
            self._frequencies,
            self._eigenvectors,
            self._triplets_at_q,
            self._bz_grid.addresses,
            self._bz_grid.D_diag,
            self._bz_grid.Q,
            self._fc3,
            self._fc3_nonzero_indices,
            self._svecs,
            self._multi,
            self._masses,
            self._p2s,
            self._s2p,
            self._band_indices,
            self._symmetrize_fc3q * 1,
            self._make_r0_average * 1,
            self._all_shortest,
            self._cutoff_frequency,
            openmp_per_triplets * 1,
        )

        self._interaction_strength *= self._unit_conversion
        # for i in range(len(self._triplets_at_q)):
        #     print("DEBUG CAPTURE: post C interaction_strength %d = " % i, self._interaction_strength[i].flatten()[:3])

        self._g_zero = g_zero

    def _run_phonon_solver_c(self, grid_points):
        run_phonon_solver_c(
            self._dm,
            self._frequencies,
            self._eigenvectors,
            self._phonon_done,
            grid_points,
            self._bz_grid.addresses,
            self._bz_grid.QDinv,
            frequency_conversion_factor=self._frequency_factor_to_THz,
            nac_q_direction=self._nac_q_direction,
            lapack_zheev_uplo=self._lapack_zheev_uplo,
        )

    def _run_fast(self, g_zero, num_processes=None):
        """Parallelized version using multiprocessing.
        
        This method parallelizes the processing of grid triplets using multiprocessing.
        Each triplet is processed independently in a separate process, which can
        significantly speed up computation on multi-core systems.
        
        Parameters
        ----------
        g_zero : ndarray or None
            Zero value positions (not used in current implementation).
        num_processes : int or None, optional
            Number of processes to use for parallel computation.
            If None (default), uses min(cpu_count(), num_triplets).
            If 1, runs sequentially for debugging purposes.
            
        Notes
        -----
        - All phonon calculations must be completed before calling this method
        - Each worker process creates its own RealToReciprocalFast and 
          ReciprocalToNormalSquaredV3 instances
        - Results are collected and merged back into the main instance arrays
        - Error handling is included to prevent process crashes from stopping
          the entire calculation
          
        Examples
        --------
        # Auto-detect number of processes
        interaction._run_fast(g_zero=None)
        
        # Use specific number of processes
        interaction._run_fast(g_zero=None, num_processes=4)
        
        # Sequential execution for debugging
        interaction._run_fast(g_zero=None, num_processes=1)
        """
        assert self._interaction_strength is not None
        assert self._triplets_at_q is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None
        assert self._make_r0_average is not None
        assert self._fc3_reciprocal is not None
        
        if num_processes is None:
            num_processes = min(mp.cpu_count(), len(self._triplets_at_q))
        
        print(f"Using {num_processes} processes for {len(self._triplets_at_q)} triplets")
        
        # Ensure all phonons are solved before parallel processing
        for i, grid_triplet in enumerate(self._triplets_at_q):
            for gp in grid_triplet:
                self._run_phonon_solver_py(gp)
        
        # Prepare arguments for parallel processing
        args_list = []
        for i, grid_triplet in enumerate(self._triplets_at_q):
            args = (
                i, grid_triplet, self._fc3, self._primitive, self.mesh_numbers, 
                self._symprec, self._make_r0_average, self._frequencies, 
                self._eigenvectors, self._band_indices, self._cutoff_frequency,
                self._unit_conversion, self._bz_grid.addresses, 
                self._fc3_reciprocal[i], self._phonon_done, self._all_shortest
            )
            args_list.append(args)
        
        # Process triplets in parallel
        start_time = time.time()
        if num_processes == 1:
            # Sequential processing for debugging
            results = [_process_triplet_worker(args) for args in args_list]
        else:
            # Parallel processing
            with mp.Pool(processes=num_processes) as pool:
                results = pool.map(_process_triplet_worker, args_list)
        
        total_time = time.time() - start_time
        print(f"Parallel processing completed in {total_time:.4f} seconds")
        
        # Collect results
        successful_count = 0
        for result in results:
            if result is None:
                continue
            i, fc3_reciprocal, interaction_strength, grid_triplet = result
            
            if fc3_reciprocal is not None:
                self._fc3_reciprocal[i] = fc3_reciprocal
            
            if interaction_strength is not None:
                self._interaction_strength[i] = interaction_strength
                successful_count += 1
        
        print(f"Parallel processing completed: {successful_count}/{len(self._triplets_at_q)} triplets processed successfully")

    def _run_v3(self, g_zero):
        assert self._interaction_strength is not None
        assert self._triplets_at_q is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None
        assert self._make_r0_average is not None
        assert self._fc3_reciprocal is not None
        
        r2r = RealToReciprocalExact(
            self._fc3, self._primitive, self.mesh_numbers, symprec=self._symprec, 
            make_r0_average=self._make_r0_average, all_shortest=self._all_shortest
        )
        
        # r2r = RealToReciprocal(
        #     self._fc3, self._primitive, self.mesh_numbers, symprec=self._symprec
        # )

        r2n = ReciprocalToNormalSquaredV3(
            self._primitive,
            self._frequencies,
            self._eigenvectors,
            self._band_indices,
            cutoff_frequency=self._cutoff_frequency,
        )

        for i, grid_triplet in enumerate(self._triplets_at_q):
            print("%d / %d" % (i + 1, len(self._triplets_at_q)),flush=True)
            r2r.run(self._bz_grid.addresses[grid_triplet])
            # print("DEBUG CAPTURE: r2r.get_fc3_reciprocal() shape:", r2r.get_fc3_reciprocal().shape)
            # print("DEBUG CAPTURE: r2r.get_fc3_reciprocal() first 5 elements:", r2r.get_fc3_reciprocal().flatten()[:5])
         
            # r2r_slow.run(self._bz_grid.addresses[grid_triplet])
            # print("r2r - r2r_slow", np.abs(r2r.get_fc3_reciprocal() - r2r_slow.get_fc3_reciprocal()).sum())
            # print("compare r2r and r2r_slow", np.allclose(r2r.get_fc3_reciprocal(), r2r_slow.get_fc3_reciprocal(),rtol=0, atol=1e-10))
   
            if np.sum(np.abs(self._fc3_reciprocal[i])) < 1e-5:
           
                fc3_reciprocal = r2r.get_fc3_reciprocal()
                if self._fc3_reciprocal is not None:
                    self._fc3_reciprocal[i] = fc3_reciprocal.reshape(self._fc3_reciprocal[i].shape)
            else:
                fc3_reciprocal = self._fc3_reciprocal[i]
                # print("fc3_reciprocal[i] first 5 elements:", fc3_reciprocal.flatten()[:5])
            for gp in grid_triplet:
                self._run_phonon_solver_py(gp)
                
            print("grid_triplet:", grid_triplet)

            start_time = time.time()
            r2n.run(self._fc3_reciprocal[i], grid_triplet, method='super_fast')
            end_time = time.time()
            print(f"r2n.run() took {end_time - start_time:.4f} seconds")
            fc3_normal_squared = r2n.get_reciprocal_to_normal_squared()

            assert fc3_normal_squared is not None
            # fc3_normal_squared = np.abs(fc3_normal) ** 2
            self._interaction_strength[i] = (
                fc3_normal_squared * self._unit_conversion
            )
            # print("DEBUG CAPTURE: post py interaction_strength = ", self._interaction_strength[i].flatten()[:3])
            
    def _run_gpu(self, g_zero, phase_vectorized=False):
        """GPU implementation for ph-ph interaction calculation.

        Uses RealToReciprocalExactGPU for real->reciprocal transform and
        ReciprocalToNormalSquaredGPU for the band-space contraction.
        
        Optimized to:
        1. Pre-run all phonon solvers first
        2. Use batched r2r.run_batch() to process ALL triplets at once
        3. Keep data on GPU between transformations
        """
        import cupy as cp
        import time
        
        # Timing dictionary to collect results
        timing = {}
        
        assert self._interaction_strength is not None
        assert self._triplets_at_q is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None
        assert self._fc3 is not None

        num_atom = len(self._primitive)
        num_band = num_atom * 3
        num_triplets = len(self._triplets_at_q)

        # Step 1: Pre-run phonon solver for ALL grid points needed
        t_start = time.perf_counter()
        all_grid_points = set()
        for grid_triplet in self._triplets_at_q:
            for gp in grid_triplet:
                all_grid_points.add(gp)
        
        for gp in all_grid_points:
            self._run_phonon_solver_py(gp)
        timing['phonon_solver'] = time.perf_counter() - t_start

        # Step 2: Prepare all triplet addresses for batched computation
        t_start = time.perf_counter()
        all_triplet_addresses = np.array([
            self._bz_grid.addresses[grid_triplet] 
            for grid_triplet in self._triplets_at_q
        ])  # Shape: (num_triplets, 3, 3)
        timing['prepare_addresses'] = time.perf_counter() - t_start

        # Step 3: Create GPU objects
        t_start = time.perf_counter()
        r2r = RealToReciprocalExactGPU(
            self._fc3,
            self._primitive,
            self.mesh_numbers,
            symprec=self._symprec,
            make_r0_average=self._make_r0_average,
            all_shortest=self._all_shortest,
            phase_vectorized=phase_vectorized,
        )
        cp.cuda.Stream.null.synchronize()
        timing['r2r_init'] = time.perf_counter() - t_start

        t_start = time.perf_counter()
        r2n_gpu = ReciprocalToNormalSquaredGPU(
            self._primitive,
            self._frequencies,
            self._eigenvectors,
            self._band_indices,
            cutoff_frequency=self._cutoff_frequency,
        )
        cp.cuda.Stream.null.synchronize()
        timing['r2n_init'] = time.perf_counter() - t_start

        # Estimate batch size for triplets based on GPU memory
        # Each triplet needs: fc3_reciprocal (P*P*P*3*3*3*16 bytes) + interaction_strength (B0*B*B*8 bytes)
        num_band0 = len(self._band_indices)
        bytes_per_triplet_fc3 = num_atom * num_atom * num_atom * 3 * 3 * 3 * 16  # complex128
        bytes_per_triplet_pp = num_band0 * num_band * num_band * 8  # float64
        bytes_per_triplet = bytes_per_triplet_fc3 + bytes_per_triplet_pp
        
        try:
            free_mem = cp.cuda.Device().mem_info[0]
            # Use at most 30% of free memory for triplet batches (leave room for intermediates)
            triplet_batch_size = max(1, int(0.6 * free_mem / bytes_per_triplet))
        except Exception:
            triplet_batch_size = 10  # Conservative default

        triplet_batch_size = min(triplet_batch_size, num_triplets)
        num_triplet_batches = (num_triplets + triplet_batch_size - 1) // triplet_batch_size
        
        print(f"  [DEBUG] Triplet batching: {num_triplets} triplets in {num_triplet_batches} batches of ~{triplet_batch_size}")
        print(f"  [DEBUG] Per-triplet memory: fc3={bytes_per_triplet_fc3/1e6:.1f}MB, pp={bytes_per_triplet_pp/1e6:.1f}MB")
        
        # Step 4 & 5: Process triplets in batches
        t_start = time.perf_counter()
        timing['r2r_batch'] = 0.0
        timing['r2n_loop_run'] = 0.0
        timing['r2n_loop_get'] = 0.0
        
        # Store results on CPU (will be transferred in batches during ISE)
        # This avoids the 16+ GB GPU allocation
        for batch_idx in range(num_triplet_batches):
            batch_start = batch_idx * triplet_batch_size
            batch_end = min(batch_start + triplet_batch_size, num_triplets)
            batch_triplet_addresses = all_triplet_addresses[batch_start:batch_end]
            
            # R2R for this batch of triplets
            t_r2r = time.perf_counter()
            fc3_reciprocal_batch_gpu = r2r.run_batch(batch_triplet_addresses, return_gpu=True)
            cp.cuda.Stream.null.synchronize()
            timing['r2r_batch'] += time.perf_counter() - t_r2r
            
            if fc3_reciprocal_batch_gpu is None:
                raise RuntimeError("GPU real-to-reciprocal returned None.")
            
            # R2N for each triplet in this batch
            for i_local, i_global in enumerate(range(batch_start, batch_end)):
                grid_triplet = self._triplets_at_q[i_global]
                fc3q_gpu = fc3_reciprocal_batch_gpu[i_local]
                
                t_run = time.perf_counter()
                r2n_gpu.run(fc3q_gpu, grid_triplet, method="super_fast", return_gpu=False)
                timing['r2n_loop_run'] += time.perf_counter() - t_run
                
                t_get = time.perf_counter()
                fc3_normal_squared = r2n_gpu.get_reciprocal_to_normal_squared()
                timing['r2n_loop_get'] += time.perf_counter() - t_get
                
                if fc3_normal_squared is None:
                    raise RuntimeError("GPU reciprocal-to-normal returned None.")
                
                # Store on CPU (the pre-allocated array)
                self._interaction_strength[i_global] = fc3_normal_squared * self._unit_conversion
            
            # Free batch GPU memory
            del fc3_reciprocal_batch_gpu
            cp.get_default_memory_pool().free_all_blocks()
        
        # No GPU array for all triplets (too large) - ISE will process in batches from CPU
        self._interaction_strength_gpu = None
        
        timing['r2n_loop_total'] = time.perf_counter() - t_start
        
        # Print timing summary
        # r2n_loop_total includes r2r_batch, r2n_loop_run, r2n_loop_get
        total_time = (timing['phonon_solver'] + timing['prepare_addresses'] + 
                      timing['r2r_init'] + timing['r2n_init'] + timing['r2n_loop_total'])
        
        print(f"\n[GPU TIMING] _run_gpu breakdown ({num_triplets} triplets in {num_triplet_batches} batches, {num_atom} atoms):")
        print(f"  Step 1 - Phonon solver:      {timing['phonon_solver']:8.4f} s")
        print(f"  Step 2 - Prepare addresses:  {timing['prepare_addresses']:8.4f} s")
        print(f"  Step 3 - R2R GPU init:       {timing['r2r_init']:8.4f} s")
        print(f"  Step 3 - R2N GPU init:       {timing['r2n_init']:8.4f} s")
        print(f"  Step 4+5 - Batched R2R+R2N:  {timing['r2n_loop_total']:8.4f} s")
        print(f"           - R2R batch total:  {timing['r2r_batch']:8.4f} s")
        print(f"           - R2N run total:    {timing['r2n_loop_run']:8.4f} s")
        print(f"           - R2N get result:   {timing['r2n_loop_get']:8.4f} s")
        print(f"  ----------------------------------------")
        print(f"  TOTAL:                       {total_time:8.4f} s\n", flush=True)

    def _run_hybrid(self, g_zero):
        """Hybrid C+GPU implementation for ph-ph interaction calculation.

        Uses fast C implementation (RealToReciprocalExact) for real->reciprocal
        transform, and GPU (ReciprocalToNormalSquaredGPU) for the band-space 
        contraction where matrix operations benefit from GPU acceleration.
        
        This is the best of both worlds:
        - C is highly optimized for R2R with OpenMP parallelization
        - GPU excels at the large matrix contractions in R2N
        """
        assert self._interaction_strength is not None
        assert self._triplets_at_q is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None
        assert self._fc3 is not None

        num_atom = len(self._primitive)
        num_band = num_atom * 3

        # Use fast C implementation for real->reciprocal
        r2r = RealToReciprocalExact(
            self._fc3,
            self._primitive,
            self.mesh_numbers,
            symprec=self._symprec,
            make_r0_average=self._make_r0_average,
            all_shortest=self._all_shortest,
        )

        # Use GPU for reciprocal->normal (large matrix ops benefit from GPU)
        r2n_gpu = ReciprocalToNormalSquaredGPU(
            self._primitive,
            self._frequencies,
            self._eigenvectors,
            self._band_indices,
            cutoff_frequency=self._cutoff_frequency,
        )

        for i, grid_triplet in enumerate(self._triplets_at_q):
            # Ensure phonons for q0, q1, q2 exist
            for gp in grid_triplet:
                self._run_phonon_solver_py(gp)

            # Real -> reciprocal using FAST C implementation
            r2r.run(self._bz_grid.addresses[grid_triplet])
            fc3q = r2r.get_fc3_reciprocal()
            
            if fc3q is None:
                raise RuntimeError("C real-to-reciprocal returned None.")
            
            # Handle different output shapes from r2r
            if fc3q.shape == (num_atom, num_atom, num_atom, 3, 3, 3):
                fc3q_6d = fc3q
            elif fc3q.shape == (num_band, num_band, num_band):
                fc3q_6d = fc3q.reshape(
                    num_atom, 3, num_atom, 3, num_atom, 3
                ).transpose(0, 2, 4, 1, 3, 5)
            else:
                raise ValueError(f"Unexpected fc3q shape: {fc3q.shape}")
            
            # Store fc3_reciprocal if allocated
            if self._fc3_reciprocal is not None:
                self._fc3_reciprocal[i] = fc3q_6d

            # Reciprocal -> normal on GPU (returns |V|^2 / (ω0 ω1 ω2))
            r2n_gpu.run(fc3q_6d, grid_triplet, method="super_fast")
            fc3_normal_squared = r2n_gpu.get_reciprocal_to_normal_squared()
            
            if fc3_normal_squared is None:
                raise RuntimeError("GPU reciprocal-to-normal returned None.")
            
            self._interaction_strength[i] = fc3_normal_squared * self._unit_conversion
        
    def _run_py(self):
        """Python implementation matching C behavior."""
        assert self._interaction_strength is not None
        assert self._triplets_at_q is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None

        num_atom = len(self._primitive)
        num_band = num_atom * 3

        # Use RealToReciprocalExact with all_shortest to match C
        r2r = RealToReciprocalExact(
            self._fc3, self._primitive, self.mesh_numbers, 
            symprec=self._symprec, make_r0_average=self._make_r0_average,
            all_shortest=self._all_shortest
        )
        r2n = ReciprocalToNormal(
            self._primitive,
            self._frequencies,
            self._eigenvectors,
            self._band_indices,
            cutoff_frequency=self._cutoff_frequency,
        )

        for i, grid_triplet in enumerate(self._triplets_at_q):
            print("%d / %d" % (i + 1, len(self._triplets_at_q)))
            
            for gp in grid_triplet:
                self._run_phonon_solver_py(gp)
            
            r2r.run(self._bz_grid.addresses[grid_triplet])
            fc3_reciprocal = r2r.get_fc3_reciprocal()
            
            # RealToReciprocalExact returns (nb, nb, nb), convert to 6D for r2n
            if fc3_reciprocal.shape == (num_band, num_band, num_band):
                fc3_reciprocal_6d = fc3_reciprocal.reshape(
                    num_atom, 3, num_atom, 3, num_atom, 3
                ).transpose(0, 2, 4, 1, 3, 5)
            else:
                fc3_reciprocal_6d = fc3_reciprocal
            
            r2n.run(fc3_reciprocal_6d, grid_triplet)
            fc3_normal = r2n.get_reciprocal_to_normal()
            assert fc3_normal is not None
            self._interaction_strength[i] = (
                np.abs(fc3_normal) ** 2 * self._unit_conversion
            )
            
    def _run_py_test(self):
        """Test to compare Python and GPU implementations step by step."""
        assert self._interaction_strength is not None
        assert self._triplets_at_q is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None

        num_atom = len(self._primitive)
        num_band = num_atom * 3
        
        print("=" * 60)
        print("DEBUG TEST: Comparing Python vs GPU implementations")
        print(f"num_atom = {num_atom}, num_band = {num_band}")
        print("=" * 60)

        # Python real-to-reciprocal (using RealToReciprocalExact to match C)
        r2r_py = RealToReciprocalExact(
            self._fc3, self._primitive, self.mesh_numbers, 
            symprec=self._symprec, make_r0_average=self._make_r0_average,
            all_shortest=self._all_shortest,
        )
        
        # GPU real-to-reciprocal (uses RealToReciprocalExact, returns 3D)
        r2r_gpu = RealToReciprocalExactGPU(
            self._fc3,
            self._primitive,
            self.mesh_numbers,
            symprec=self._symprec,
            make_r0_average=self._make_r0_average,
            all_shortest=self._all_shortest,
        )
        
        # Python reciprocal-to-normal (returns complex fc3_normal)
        r2n_py = ReciprocalToNormal(
            self._primitive,
            self._frequencies,
            self._eigenvectors,
            self._band_indices,
            cutoff_frequency=self._cutoff_frequency,
        )
        
        # GPU reciprocal-to-normal (returns real |V|^2)
        r2n_gpu = ReciprocalToNormalSquaredGPU(
            self._primitive,
            self._frequencies,
            self._eigenvectors,
            self._band_indices,
            cutoff_frequency=self._cutoff_frequency,
        )

        for i, grid_triplet in enumerate(self._triplets_at_q):
            print(f"\n{'='*60}")
            print(f"Triplet {i + 1} / {len(self._triplets_at_q)}: {grid_triplet}")
            print(f"{'='*60}")
            
            # Ensure phonons are computed
            for gp in grid_triplet:
                self._run_phonon_solver_py(gp)
            
            # ============================================================
            # STEP 1: Compare Real-to-Reciprocal outputs
            # ============================================================
            print("\n--- STEP 1: Real-to-Reciprocal comparison ---")
            
            r2r_py.run(self._bz_grid.addresses[grid_triplet])
            fc3_recip_py = r2r_py.get_fc3_reciprocal()
            
            r2r_gpu.run(self._bz_grid.addresses[grid_triplet])
            fc3_recip_gpu = r2r_gpu.get_fc3_reciprocal()
            
            print(f"Python fc3_reciprocal shape: {fc3_recip_py.shape}")
            print(f"GPU fc3_reciprocal shape:    {fc3_recip_gpu.shape}")
            
            # The Python RealToReciprocal returns (na, na, na, 3, 3, 3)
            # The RealToReciprocalExact returns (nb, nb, nb)
            # We need to compare them in a common format
            
            if fc3_recip_py.shape == (num_atom, num_atom, num_atom, 3, 3, 3):
                # Convert Python 6D to 3D for comparison
                # fc3_py_3d[a0*3+c0, a1*3+c1, a2*3+c2] = fc3_py_6d[a0, a1, a2, c0, c1, c2]
                fc3_py_3d = fc3_recip_py.transpose(0, 3, 1, 4, 2, 5).reshape(num_band, num_band, num_band)
                print(f"Python fc3 (converted to 3D): {fc3_py_3d.shape}")
            else:
                fc3_py_3d = fc3_recip_py.reshape(num_band, num_band, num_band)
            
            if fc3_recip_gpu.shape == (num_band, num_band, num_band):
                fc3_gpu_3d = fc3_recip_gpu
            else:
                # If GPU returns 6D, convert to 3D
                fc3_gpu_3d = fc3_recip_gpu.transpose(0, 3, 1, 4, 2, 5).reshape(num_band, num_band, num_band)
            
            print(f"Python fc3_3d first 5 elements: {fc3_py_3d.flatten()[:5]}")
            print(f"GPU fc3_3d first 5 elements:    {fc3_gpu_3d.flatten()[:5]}")
            
            fc3_diff = np.abs(fc3_py_3d - fc3_gpu_3d)
            print(f"Max |fc3_py - fc3_gpu|: {fc3_diff.max():.6e}")
            print(f"Mean |fc3_py - fc3_gpu|: {fc3_diff.mean():.6e}")
            
            if fc3_diff.max() > 1e-8:
                print("WARNING: fc3_reciprocal differs significantly!")
                # Find where max difference is
                max_idx = np.unravel_index(np.argmax(fc3_diff), fc3_diff.shape)
                print(f"Max diff at index {max_idx}")
                print(f"  Python: {fc3_py_3d[max_idx]}")
                print(f"  GPU:    {fc3_gpu_3d[max_idx]}")
            else:
                print("OK: fc3_reciprocal matches within tolerance")
            
            # ============================================================
            # STEP 2: Compare Reciprocal-to-Normal outputs
            # ============================================================
            print("\n--- STEP 2: Reciprocal-to-Normal comparison ---")
            
            # Python r2n expects 6D input - convert fc3_py_3d to 6D
            fc3_py_6d = fc3_py_3d.reshape(num_atom, 3, num_atom, 3, num_atom, 3).transpose(0, 2, 4, 1, 3, 5)
            r2n_py.run(fc3_py_6d, grid_triplet)
            fc3_normal_py = r2n_py.get_reciprocal_to_normal()
            
            # GPU r2n - convert GPU fc3 to 6D format
            fc3_gpu_6d = fc3_gpu_3d.reshape(num_atom, 3, num_atom, 3, num_atom, 3).transpose(0, 2, 4, 1, 3, 5)
            
            print(f"Python fc3_normal shape: {fc3_normal_py.shape}")
            print(f"Python fc3_normal dtype: {fc3_normal_py.dtype}")
            
            # Compute |V|^2 from Python
            # Python returns complex fc3_normal, normalized by sqrt(f1*f2*f3)
            # So |fc3_normal|^2 gives us the interaction strength (before unit_conversion)
            fc3_normal_squared_py = np.abs(fc3_normal_py) ** 2
            print(f"Python |fc3_normal|^2 shape: {fc3_normal_squared_py.shape}")
            
            # GPU r2n returns |V|^2 / (f1*f2*f3), which is the same as |fc3_normal|^2
            # since fc3_normal = V / sqrt(f1*f2*f3)
            r2n_gpu.run(fc3_gpu_6d, grid_triplet, method="super_fast")
            fc3_normal_squared_gpu = r2n_gpu.get_reciprocal_to_normal_squared()
            
            print(f"GPU fc3_normal_squared shape: {fc3_normal_squared_gpu.shape}")
            print(f"GPU fc3_normal_squared dtype: {fc3_normal_squared_gpu.dtype}")
            
            print(f"\nPython |fc3_normal|^2 first 5 elements: {fc3_normal_squared_py.flatten()[:5]}")
            print(f"GPU fc3_normal_squared first 5 elements: {fc3_normal_squared_gpu.flatten()[:5]}")
            
            # Find non-zero elements for better comparison
            py_nonzero = fc3_normal_squared_py[fc3_normal_squared_py > 1e-20]
            gpu_nonzero = fc3_normal_squared_gpu[fc3_normal_squared_gpu > 1e-20]
            print(f"\nPython non-zero count: {len(py_nonzero)}")
            print(f"GPU non-zero count: {len(gpu_nonzero)}")
            
            if len(py_nonzero) > 0:
                print(f"Python non-zero range: [{py_nonzero.min():.6e}, {py_nonzero.max():.6e}]")
            if len(gpu_nonzero) > 0:
                print(f"GPU non-zero range: [{gpu_nonzero.min():.6e}, {gpu_nonzero.max():.6e}]")
            
            # Compare
            v2_diff = np.abs(fc3_normal_squared_py - fc3_normal_squared_gpu)
            print(f"\nMax |py - gpu|: {v2_diff.max():.6e}")
            print(f"Mean |py - gpu|: {v2_diff.mean():.6e}")
            
            # Relative error where Python is non-zero
            mask = fc3_normal_squared_py > 1e-20
            if mask.any():
                rel_err = v2_diff[mask] / fc3_normal_squared_py[mask]
                print(f"Max relative error (where py > 1e-20): {rel_err.max():.6e}")
                print(f"Mean relative error: {rel_err.mean():.6e}")
            
            if v2_diff.max() > 1e-8:
                print("\nWARNING: fc3_normal_squared differs significantly!")
                max_idx = np.unravel_index(np.argmax(v2_diff), v2_diff.shape)
                print(f"Max diff at index {max_idx}")
                print(f"  Python: {fc3_normal_squared_py[max_idx]:.10e}")
                print(f"  GPU:    {fc3_normal_squared_gpu[max_idx]:.10e}")
            else:
                print("\nOK: fc3_normal_squared matches within tolerance")
            
            # Store in interaction_strength (using GPU result to validate full GPU path)
            self._interaction_strength[i] = fc3_normal_squared_gpu * self._unit_conversion
            
            # # Only test first triplet for debugging
            # print("\n" + "=" * 60)
            # print("Stopping after first triplet for debugging")
            # print("=" * 60)
            # break

    def _run_phonon_solver_py(self, grid_point):
        run_phonon_solver_py(
            grid_point,
            self._phonon_done,
            self._frequencies,
            self._eigenvectors,
            self._bz_grid.addresses,
            self._bz_grid.QDinv,
            self._dm,
            self._frequency_factor_to_THz,
            self._lapack_zheev_uplo,
        )

    def _allocate_phonon(self):
        """Allocate phonon arrays.

        Phonons at Gamma point without NAC are stored in `self._frequencies_at_gamma`
        and `self._eigenvectors_at_gamma`.

        """
        num_band = len(self._primitive) * 3
        num_grid = len(self._bz_grid.addresses)
        self._phonon_done = np.zeros(num_grid, dtype="byte")
        self._phonon_all_done = False
        self._frequencies = np.zeros((num_grid, num_band), dtype="double", order="C")
        complex_dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._eigenvectors = np.zeros(
            (num_grid, num_band, num_band), dtype=complex_dtype, order="C"
        )
        gp_Gamma = self._bz_grid.gp_Gamma
        self.run_phonon_solver_at_gamma()
        self._frequencies_at_gamma = self._frequencies[gp_Gamma].copy()
        self._eigenvectors_at_gamma = self._eigenvectors[gp_Gamma].copy()
        self._phonon_done[gp_Gamma] = 0

    def _get_all_shortest(self):
        """Return array indicating distances among three atoms are all shortest.

        multi.shape = (n_satom, n_patom)
        svecs : distance with respect to primitive cell basis
        perms.shape = (n_pure_trans, n_satom)

        """
        svecs = self._svecs
        multi = self._multi
        n_satom, n_patom, _ = multi.shape
        perms = self._primitive.atomic_permutations
        s2pp_map = [self._primitive.p2p_map[i] for i in self._s2p]
        lattice = self._primitive.cell

        for i_patom, j_atom in np.ndindex((n_patom, n_satom)):
            if multi[j_atom, i_patom, 0] > 1:
                continue
            j_patom = s2pp_map[j_atom]
            i_perm = np.where(perms[:, j_atom] == self._p2s[j_patom])[0]
            assert len(i_perm) == 1
            for k_atom in range(n_satom):
                if multi[k_atom, i_patom, 0] > 1:
                    continue
                k_atom_mapped = perms[i_perm[0], k_atom]
                if multi[k_atom_mapped, j_patom, 0] > 1:
                    continue
                vec_jk = (
                    svecs[multi[k_atom, i_patom, 1]] - svecs[multi[j_atom, i_patom, 1]]
                )
                d_jk = np.linalg.norm(vec_jk @ lattice)
                d_jk_mapped = np.linalg.norm(
                    svecs[multi[k_atom_mapped, j_patom, 1]] @ lattice
                )
                if abs(d_jk_mapped - d_jk) < self._symprec:
                    self._all_shortest[i_patom, j_atom, k_atom] = 1


def all_bands_exist(interaction: Interaction):
    """Return if all bands are selected or not."""
    band_indices = interaction.band_indices
    num_band = len(interaction.primitive) * 3
    if len(band_indices) == num_band:
        if (band_indices - np.arange(num_band) == 0).all():
            return True
    return False
