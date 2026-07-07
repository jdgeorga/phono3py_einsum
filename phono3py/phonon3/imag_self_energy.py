"""Calculation of imaginary-part of self-energy of bubble diagram."""

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

import sys
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.physical_units import get_physical_units

from phono3py._lang import log_dispatch, resolve_lang
from phono3py.file_IO import (
    write_gamma_detail_to_hdf5,
    write_imag_self_energy_at_grid_point,
)
from phono3py.phonon.func import bose_einstein
from phono3py.phonon3.interaction_fast import Interaction
from phono3py.phonon3.triplets import get_triplets_integration_weights


def run_imag_self_energy_with_g_rust(
    imag_self_energy: NDArray[np.double],
    pp_strength: NDArray[np.double],
    triplets_at_q: NDArray[np.int64],
    weights_at_q: NDArray[np.int64],
    frequencies: NDArray[np.double],
    temperature_thz: float,
    g: NDArray[np.double],
    g_zero: NDArray[np.byte],
    cutoff_frequency: float,
    frequency_point_index: int = -1,
) -> None:
    """Compute the imag. self-energy with integration weights using the Rust backend.

    Drop-in replacement for ``phono3c.imag_self_energy_with_g`` at a
    single temperature.

    - ``imag_self_energy`` (out, ``(num_band0,)``): accumulated in place.
    - ``pp_strength``: ``(num_triplets, num_band0, num_band, num_band)``.
    - ``g``: ``(2, num_triplets, num_band0_or_freqpts, num_band, num_band)``.
    - ``g_zero``: ``(num_triplets, num_band0_or_freqpts, num_band, num_band)``
      ``byte``; non-zero entries mark zero-weight positions to skip.
    - ``frequency_point_index``: ``-1`` for band-index mode, otherwise the
      frequency-point index to sample.

    """
    import phonors

    phonors.imag_self_energy_with_g(
        imag_self_energy,
        np.ascontiguousarray(pp_strength, dtype="double"),
        np.ascontiguousarray(triplets_at_q, dtype="int64"),
        np.ascontiguousarray(weights_at_q, dtype="int64"),
        np.ascontiguousarray(frequencies, dtype="double"),
        float(temperature_thz),
        np.ascontiguousarray(g, dtype="double"),
        np.ascontiguousarray(g_zero, dtype="byte"),
        float(cutoff_frequency),
        int(frequency_point_index),
    )


def run_detailed_imag_self_energy_with_g_rust(
    detailed_imag_self_energy: NDArray[np.double],
    imag_self_energy_N: NDArray[np.double],
    imag_self_energy_U: NDArray[np.double],
    pp_strength: NDArray[np.double],
    triplets_at_q: NDArray[np.int64],
    weights_at_q: NDArray[np.int64],
    bz_grid_addresses: NDArray[np.int64],
    frequencies: NDArray[np.double],
    temperature_thz: float,
    g: NDArray[np.double],
    g_zero: NDArray[np.byte],
    cutoff_frequency: float,
) -> None:
    """Compute the detailed imag. self-energy via the Rust backend.

    Drop-in replacement for ``phono3c.detailed_imag_self_energy_with_g``
    at a single temperature.  Writes into the three output arrays in
    place:

    - ``detailed_imag_self_energy``:
      ``(num_triplets, num_band0, num_band, num_band)``.
    - ``imag_self_energy_N`` / ``imag_self_energy_U``: ``(num_band0,)``
      Normal and Umklapp contributions, respectively.

    """
    import phonors

    phonors.detailed_imag_self_energy_with_g(
        detailed_imag_self_energy,
        imag_self_energy_N,
        imag_self_energy_U,
        np.ascontiguousarray(pp_strength, dtype="double"),
        np.ascontiguousarray(triplets_at_q, dtype="int64"),
        np.ascontiguousarray(weights_at_q, dtype="int64"),
        np.ascontiguousarray(bz_grid_addresses, dtype="int64"),
        np.ascontiguousarray(frequencies, dtype="double"),
        float(temperature_thz),
        np.ascontiguousarray(g, dtype="double"),
        np.ascontiguousarray(g_zero, dtype="byte"),
        float(cutoff_frequency),
    )


class ImagSelfEnergy:
    """Class for imaginary-part of self-energy of bubble diagram."""

    def __init__(
        self,
        interaction: Interaction,
        with_detail: bool = False,
        lang: Literal["C", "Python", "Rust", "GPU", "GPU_phase", "Hybrid"] = "Rust",
    ) -> None:
        """Init method.

        Band indices to be calculated at are kept in Interaction instance.

        Parameters
        ----------
        interaction : Interaction
            Class instance of ph-ph interaction.
        with_detail : bool, optional
            Contributions to gammas for each triplets are computed. Default is
            False.
        lang : str, optional
            Backend selection.  ``"C"`` (default) uses the C extension,
            ``"Python"`` the slow reference implementation, and ``"Rust"``
            the Rust backend.  For ``"Rust"``, the ``detailed`` +
            ``frequency_points`` path is not implemented and falls back
            to C.

        """
        self._pp = interaction
        self._sigma: float | None = None
        self._sigma_cutoff: float | None = None
        self._temperature: float | None = None
        self._frequency_points: NDArray[np.double] | None = None
        self._grid_point: int | None = None

        if lang in ("C", "Rust"):
            lang = resolve_lang(lang)
        self._lang: Literal[
            "C", "Python", "Rust", "GPU", "GPU_phase", "Hybrid"
        ] = lang
        log_dispatch(lang, "ImagSelfEnergy.__init__")
        self._imag_self_energy: NDArray[np.double] | None = None
        self._detailed_imag_self_energy: NDArray[np.double] | None = None
        self._pp_strength: NDArray[np.double] | None = None
        self._pp_strength_gpu: NDArray[np.double] | None = None
        self._frequencies: NDArray[np.double] | None = None
        self._triplets_at_q: NDArray[np.int64] | None = None
        self._weights_at_q: NDArray[np.int64] | None = None
        self._with_detail: bool = with_detail
        self._cutoff_frequency: float = interaction.cutoff_frequency

        self._g: NDArray[np.double] | None = None  # integration weights
        self._g_zero: NDArray[np.byte] | None = (
            None  # Necessary elements of interaction strength
        )
        self._ise_N: NDArray[np.double] | None = None
        self._ise_U: NDArray[np.double] | None = None

        # Unit to THz of Gamma
        self._unit_conversion: float = (
            18
            * np.pi
            / (get_physical_units().Hbar * get_physical_units().EV) ** 2
            / (2 * np.pi * get_physical_units().THz) ** 2
            * get_physical_units().EV ** 2
        )

    def run(self) -> None:
        """Calculate imaginary-part of self-energies."""
        if self._pp_strength is None:
            self.run_interaction()
        assert self._pp_strength is not None

        num_band0 = self._pp_strength.shape[1]

        if self._frequency_points is None:
            self._imag_self_energy = np.zeros(num_band0, dtype="double")
            if self._with_detail:
                self._detailed_imag_self_energy = np.empty_like(self._pp_strength)
                self._detailed_imag_self_energy[:] = 0
                self._ise_N = np.zeros_like(self._imag_self_energy)
                self._ise_U = np.zeros_like(self._imag_self_energy)
            self._run_with_band_indices()
        else:
            self._imag_self_energy = np.zeros(
                (len(self._frequency_points), num_band0), order="C", dtype="double"
            )
            if self._with_detail:
                self._detailed_imag_self_energy = np.zeros(
                    (len(self._frequency_points),) + self._pp_strength.shape,
                    order="C",
                    dtype="double",
                )
                self._ise_N = np.zeros_like(self._imag_self_energy)
                self._ise_U = np.zeros_like(self._imag_self_energy)
            self._run_with_frequency_points()

    def run_interaction(self, is_full_pp: bool = True) -> None:
        """Calculate ph-ph interaction."""
        if is_full_pp or self._frequency_points is not None:
            self._pp.run(lang=self._lang)
        else:
            self._pp.run(lang=self._lang, g_zero=self._g_zero)
        self._pp_strength = self._pp.interaction_strength
        # Also get GPU array if available (for efficient GPU ISE calculation)
        self._pp_strength_gpu = getattr(self._pp, "interaction_strength_gpu", None)

    def run_integration_weights(
        self, scattering_event_class: Literal[1, 2] | None = None
    ) -> None:
        """Compute integration weights at grid points."""
        if self._frequency_points is None:
            assert self._frequencies is not None
            assert self._grid_point is not None
            bi = self._pp.band_indices
            f_points = self._frequencies[self._grid_point][bi]
        else:
            f_points = self._frequency_points

        from phono3py.phonon3.collision_matrix import CollisionMatrix

        self._g, self._g_zero = get_triplets_integration_weights(
            self._pp,
            np.array(f_points, dtype="double"),
            self._sigma,
            self._sigma_cutoff,
            is_collision_matrix=isinstance(self, CollisionMatrix),
            lang=self._lang,
        )

        if scattering_event_class == 1 or scattering_event_class == 2:
            self._g[scattering_event_class - 1] = 0

    @property
    def imag_self_energy(self) -> NDArray[np.double] | None:
        """Return calculated imaginary-part of self-energies."""
        if self._cutoff_frequency is None:
            return self._imag_self_energy
        else:
            assert self._imag_self_energy is not None
            return self._average_by_degeneracy(self._imag_self_energy)

    def get_imag_self_energy(self) -> NDArray[np.double] | None:
        """Return calculated imaginary-part of self-energies."""
        warnings.warn(
            "Use attribute, ImagSelfEnergy.imag_self_energy "
            "instead of ImagSelfEnergy.get_imag_self_energy().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.imag_self_energy

    def get_imag_self_energy_N_and_U(
        self,
    ) -> tuple[NDArray[np.double] | None, NDArray[np.double] | None]:
        """Return normal and Umklapp contributions.

        Three-phonon scatterings are categorized into normal and Umklapp and
        the contributions of the triplets to imaginary-part of self-energies
        are returned.

        """
        if self._cutoff_frequency is None:
            return self._ise_N, self._ise_U
        else:
            assert self._ise_N is not None
            assert self._ise_U is not None
            return (
                self._average_by_degeneracy(self._ise_N),
                self._average_by_degeneracy(self._ise_U),
            )

    @property
    def detailed_imag_self_energy(self) -> NDArray[np.double] | None:
        """Return triplets contributions to imaginary-part of self-energies."""
        return self._detailed_imag_self_energy

    @property
    def reduced_imag_self_energy(
        self,
    ) -> tuple[NDArray[np.double], NDArray[np.double]] | None:
        """Return band-summed reduced views of the detailed self-energy.

        Returns ``(gamma_q1_sum, gamma_q2_sum)`` derived from the detailed
        (triplet-resolved) gamma without materializing the full dense tensor in
        the caller. For the band-indices path the detailed array has shape
        ``(triplets, band0, band1, band2)`` and the reduced views are::

            gamma_q1_sum = sum(detailed, axis=band2)  -> (triplets, band0, band1)
            gamma_q2_sum = sum(detailed, axis=band1)  -> (triplets, band0, band2)

        This matches the reductions the W-assembly driver expects (see
        ``compute_reduced_gamma`` in the GPU scattering driver). Returns None if
        the detailed gamma was not computed (``with_detail=False``).
        """
        d = self._detailed_imag_self_energy
        if d is None:
            return None
        # band2 is the last axis, band1 the second-to-last.
        gamma_q1_sum = np.sum(d, axis=-1)
        gamma_q2_sum = np.sum(d, axis=-2)
        return gamma_q1_sum, gamma_q2_sum

    def get_detailed_imag_self_energy(self) -> NDArray[np.double] | None:
        """Return triplets contributions to imaginary-part of self-energies."""
        warnings.warn(
            "Use attribute, ImagSelfEnergy.detailed_imag_self_energy "
            "instead of ImagSelfEnergy.get_detailed_imag_self_energy().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.detailed_imag_self_energy

    def get_integration_weights(
        self,
    ) -> tuple[NDArray[np.double] | None, NDArray[np.byte] | None]:
        """Return integration weights.

        See the details of returns at ``get_triplets_integration_weights``.

        """
        return self._g, self._g_zero

    @property
    def unit_conversion_factor(self) -> float:
        """Return unit conversion factor of gamma."""
        return self._unit_conversion

    def get_unit_conversion_factor(self) -> float:
        """Return unit conversion factor of gamma."""
        warnings.warn(
            "Use attribute, ImagSelfEnergy.unit_conversion_factor "
            "instead of ImagSelfEnergy.get_unit_conversion_factor().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.unit_conversion_factor

    def set_grid_point(self, grid_point: int | None = None) -> None:
        """Set a grid point at which calculation will be performed."""
        if grid_point is None:
            self._grid_point = None
        else:
            self._pp.set_grid_point(grid_point)
            self._pp_strength = None
            self._pp_strength_gpu = None
            self._triplets_at_q, self._weights_at_q = self._pp.get_triplets_at_q()[:2]
            self._grid_point = grid_point
            self._frequencies, self._eigenvectors, _ = self._pp.get_phonons()

    def set_sigma(self, sigma: float | None, sigma_cutoff: float | None = None) -> None:
        """Set sigma value. None means tetrahedron method."""
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)

        if sigma_cutoff is None:
            self._sigma_cutoff = None
        else:
            self._sigma_cutoff = float(sigma_cutoff)

        self.delete_integration_weights()

    @property
    def frequency_points(self) -> NDArray[np.double] | None:
        """Getter and setter of sampling frequency points."""
        return self._frequency_points

    @frequency_points.setter
    def frequency_points(self, frequency_points: NDArray[np.double] | None) -> None:
        if frequency_points is None:
            self._frequency_points = None
        else:
            self._frequency_points = np.array(frequency_points, dtype="double")

    def set_frequency_points(self, frequency_points: NDArray[np.double] | None) -> None:
        """Set frequency points where spectrum calculation will be performed."""
        warnings.warn(
            "Use attribute, ImagSelfEnergy.frequency_points "
            "instead of ImagSelfEnergy.set_frequency_points().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.frequency_points = frequency_points

    @property
    def temperature(self) -> float | None:
        """Getter and setter of temperature."""
        return self._temperature

    @temperature.setter
    def temperature(self, temperature: float | None) -> None:
        if temperature is None:
            self._temperature = None
        else:
            self._temperature = float(temperature)

    def set_temperature(self, temperature: float | None) -> None:
        """Set temperature where calculation will be performed."""
        warnings.warn(
            "Use attribute, ImagSelfEnergy.temperature "
            "instead of ImagSelfEnergy.set_temperature().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.temperature = temperature

    def set_averaged_pp_interaction(self, ave_pp: NDArray[np.double]) -> None:
        """Set averaged ph-ph interactions.

        This is used for analysis of the calculated results by introducing
        averaged value as an approximation.
        Setting this, ph-ph interaction calculation will not be executed.

        """
        assert self._triplets_at_q is not None
        num_triplets = len(self._triplets_at_q)
        num_band = len(self._pp.primitive) * 3
        num_grid = np.prod(self._pp.mesh_numbers)
        bi = self._pp.band_indices
        self._pp_strength = np.zeros(
            (num_triplets, len(bi), num_band, num_band), dtype="double"
        )

        for i, v_ave in enumerate(ave_pp):
            self._pp_strength[:, i, :, :] = v_ave / num_grid

    def set_interaction_strength(
        self,
        pp_strength: NDArray[np.double],
        g_zero: NDArray[np.byte] | None = None,
    ) -> None:
        """Set ph-ph interaction strengths."""
        self._pp_strength = pp_strength
        if g_zero is not None:
            self._g_zero = g_zero
        self._pp.set_interaction_strength(pp_strength, g_zero=g_zero)

    def delete_integration_weights(self) -> None:
        """Delete large ndarray's."""
        self._g = None
        self._g_zero = None
        if self._lang != "GPU_phase":
            self._pp_strength = None
            self._pp_strength_gpu = None

    def _run_with_band_indices(self) -> None:
        if self._g is not None:
            if self._lang == "C" or self._lang == "Fast":
                if self._with_detail:
                    # self._detailed_imag_self_energy.shape =
                    #    (num_triplets, num_band0, num_band, num_band)
                    # self._imag_self_energy is also set.
                    self._run_c_detailed_with_band_indices_with_g()
                else:
                    # self._imag_self_energy.shape = (num_band0,)
                    self._run_c_with_band_indices_with_g()
            elif self._lang == "Rust":
                if self._with_detail:
                    self._run_rust_detailed_with_band_indices_with_g()
                else:
                    self._run_rust_with_band_indices_with_g()
            elif self._lang in ("GPU", "GPU_phase", "Hybrid"):
                self._run_gpu_with_band_indices_with_g()
            else:
                print("Running into _run_py_with_band_indices_with_g()")
                print("This routine is super slow and only for the test.")
                self._run_py_with_band_indices_with_g()
        else:
            print(
                "get_triplets_integration_weights must be executed "
                "before calling this method."
            )
            import sys

            sys.exit(1)

    def _run_with_frequency_points(self) -> None:
        if self._g is not None:
            if self._lang == "C" or self._lang == "Fast":
                if self._with_detail:
                    self._run_c_detailed_with_frequency_points_with_g()
                else:
                    self._run_c_with_frequency_points_with_g()
            elif self._lang == "Rust":
                if self._with_detail:
                    # detailed + frequency_points is not yet implemented
                    # in the Rust backend; fall back to C.
                    self._run_c_detailed_with_frequency_points_with_g()
                else:
                    self._run_rust_with_frequency_points_with_g()
            elif self._lang in ("GPU", "GPU_phase", "Hybrid") and self._with_detail:
                raise NotImplementedError(
                    "Detailed gamma with frequency-point sampling is not implemented "
                    "for lang='GPU'/'GPU_phase'/'Hybrid'. "
                    "Use frequency_points_at_bands=True."
                )
            else:
                print("Running into _run_py_with_frequency_points_with_g()")
                print("This routine is super slow and only for the test.")
                self._run_py_with_frequency_points_with_g()
        else:
            print(
                "get_triplets_integration_weights must be executed "
                "before calling this method."
            )
            import sys

            sys.exit(1)

    def _run_c_with_band_indices_with_g(self) -> None:
        import phono3py._phono3py as phono3c  # type: ignore[import-untyped]

        assert self._pp_strength is not None
        assert self._temperature is not None
        assert self._g is not None
        assert self._imag_self_energy is not None
        assert self._triplets_at_q is not None
        assert self._weights_at_q is not None
        assert self._frequencies is not None
        if self._g_zero is None:
            _g_zero = np.zeros(self._pp_strength.shape, dtype="byte", order="C")
        else:
            _g_zero = self._g_zero

        phono3c.imag_self_energy_with_g(
            self._imag_self_energy,
            self._pp_strength,
            self._triplets_at_q,
            self._weights_at_q,
            self._frequencies,
            self._temperature * get_physical_units().KB / get_physical_units().THzToEv,
            self._g,
            _g_zero,
            self._cutoff_frequency,
            -1,
        )
        self._imag_self_energy *= self._unit_conversion

    def _run_c_detailed_with_band_indices_with_g(self) -> None:
        import phono3py._phono3py as phono3c  # type: ignore[import-untyped]

        assert self._pp_strength is not None
        assert self._temperature is not None
        assert self._g is not None
        assert self._detailed_imag_self_energy is not None
        assert self._ise_N is not None
        assert self._ise_U is not None
        assert self._imag_self_energy is not None
        assert self._triplets_at_q is not None
        assert self._weights_at_q is not None
        assert self._frequencies is not None
        if self._g_zero is None:
            _g_zero = np.zeros(self._pp_strength.shape, dtype="byte", order="C")
        else:
            _g_zero = self._g_zero

        phono3c.detailed_imag_self_energy_with_g(
            self._detailed_imag_self_energy,
            self._ise_N,  # Normal
            self._ise_U,  # Umklapp
            self._pp_strength,
            self._triplets_at_q,
            self._weights_at_q,
            self._pp.bz_grid.addresses,
            self._frequencies,
            self._temperature * get_physical_units().KB / get_physical_units().THzToEv,
            self._g,
            _g_zero,
            self._cutoff_frequency,
        )

        self._detailed_imag_self_energy *= self._unit_conversion
        self._ise_N *= self._unit_conversion
        self._ise_U *= self._unit_conversion
        self._imag_self_energy = self._ise_N + self._ise_U

    def _run_c_with_frequency_points_with_g(self) -> None:
        import phono3py._phono3py as phono3c  # type: ignore[import-untyped]

        assert self._pp_strength is not None
        assert self._temperature is not None
        assert self._frequency_points is not None
        assert self._imag_self_energy is not None
        assert self._triplets_at_q is not None
        assert self._weights_at_q is not None
        assert self._frequencies is not None
        assert self._g is not None
        num_band0 = self._pp_strength.shape[1]
        ise_at_f = np.zeros(num_band0, dtype="double")

        for i in range(len(self._frequency_points)):
            phono3c.imag_self_energy_with_g(
                ise_at_f,
                self._pp_strength,
                self._triplets_at_q,
                self._weights_at_q,
                self._frequencies,
                self._temperature
                * get_physical_units().KB
                / get_physical_units().THzToEv,
                self._g,
                self._g_zero,
                self._cutoff_frequency,
                i,
            )
            self._imag_self_energy[i] = ise_at_f
        self._imag_self_energy *= self._unit_conversion

    def _run_c_detailed_with_frequency_points_with_g(self) -> None:
        import phono3py._phono3py as phono3c  # type: ignore[import-untyped]

        assert self._pp_strength is not None
        assert self._temperature is not None
        assert self._frequency_points is not None
        assert self._detailed_imag_self_energy is not None
        assert self._ise_N is not None
        assert self._ise_U is not None
        assert self._imag_self_energy is not None
        assert self._triplets_at_q is not None
        assert self._weights_at_q is not None
        assert self._frequencies is not None
        assert self._g is not None
        num_band0 = self._pp_strength.shape[1]
        g_shape = list(self._g.shape)
        g_shape[2] = num_band0
        g = np.zeros((2,) + self._pp_strength.shape, order="C", dtype="double")
        detailed_ise_at_f = np.zeros(
            self._detailed_imag_self_energy.shape[1:5], order="C", dtype="double"
        )
        ise_at_f_N = np.zeros(num_band0, dtype="double")
        ise_at_f_U = np.zeros(num_band0, dtype="double")
        _g_zero = np.zeros(g_shape, dtype="byte", order="C")

        for i in range(len(self._frequency_points)):
            for j in range(g.shape[2]):  # type: ignore
                g[:, :, j, :, :] = self._g[:, :, i, :, :]
                phono3c.detailed_imag_self_energy_with_g(
                    detailed_ise_at_f,
                    ise_at_f_N,
                    ise_at_f_U,
                    self._pp_strength,
                    self._triplets_at_q,
                    self._weights_at_q,
                    self._pp.bz_grid.addresses,
                    self._frequencies,
                    self._temperature
                    * get_physical_units().KB
                    / get_physical_units().THzToEv,
                    g,
                    _g_zero,
                    self._cutoff_frequency,
                )
            self._detailed_imag_self_energy[i] = (
                detailed_ise_at_f * self._unit_conversion
            )
            self._ise_N[i] = ise_at_f_N * self._unit_conversion
            self._ise_U[i] = ise_at_f_U * self._unit_conversion
            self._imag_self_energy[i] = self._ise_N[i] + self._ise_U[i]

    def _run_rust_with_band_indices_with_g(self) -> None:
        assert self._pp_strength is not None
        assert self._temperature is not None
        assert self._g is not None
        assert self._imag_self_energy is not None
        assert self._triplets_at_q is not None
        assert self._weights_at_q is not None
        assert self._frequencies is not None
        if self._g_zero is None:
            _g_zero = np.zeros(self._pp_strength.shape, dtype="byte", order="C")
        else:
            _g_zero = self._g_zero

        run_imag_self_energy_with_g_rust(
            self._imag_self_energy,
            self._pp_strength,
            self._triplets_at_q,
            self._weights_at_q,
            self._frequencies,
            self._temperature * get_physical_units().KB / get_physical_units().THzToEv,
            self._g,
            _g_zero,
            self._cutoff_frequency,
            -1,
        )
        self._imag_self_energy *= self._unit_conversion

    def _run_rust_detailed_with_band_indices_with_g(self) -> None:
        assert self._pp_strength is not None
        assert self._temperature is not None
        assert self._g is not None
        assert self._detailed_imag_self_energy is not None
        assert self._ise_N is not None
        assert self._ise_U is not None
        assert self._imag_self_energy is not None
        assert self._triplets_at_q is not None
        assert self._weights_at_q is not None
        assert self._frequencies is not None
        if self._g_zero is None:
            _g_zero = np.zeros(self._pp_strength.shape, dtype="byte", order="C")
        else:
            _g_zero = self._g_zero

        run_detailed_imag_self_energy_with_g_rust(
            self._detailed_imag_self_energy,
            self._ise_N,  # Normal
            self._ise_U,  # Umklapp
            self._pp_strength,
            self._triplets_at_q,
            self._weights_at_q,
            self._pp.bz_grid.addresses,
            self._frequencies,
            self._temperature * get_physical_units().KB / get_physical_units().THzToEv,
            self._g,
            _g_zero,
            self._cutoff_frequency,
        )

        self._detailed_imag_self_energy *= self._unit_conversion
        self._ise_N *= self._unit_conversion
        self._ise_U *= self._unit_conversion
        self._imag_self_energy = self._ise_N + self._ise_U

    def _run_rust_with_frequency_points_with_g(self) -> None:
        assert self._pp_strength is not None
        assert self._temperature is not None
        assert self._frequency_points is not None
        assert self._imag_self_energy is not None
        assert self._triplets_at_q is not None
        assert self._weights_at_q is not None
        assert self._frequencies is not None
        assert self._g is not None
        num_band0 = self._pp_strength.shape[1]
        ise_at_f = np.zeros(num_band0, dtype="double")
        if self._g_zero is None:
            _g_zero = np.zeros(self._g.shape[1:], dtype="byte", order="C")
        else:
            _g_zero = self._g_zero

        for i in range(len(self._frequency_points)):
            run_imag_self_energy_with_g_rust(
                ise_at_f,
                self._pp_strength,
                self._triplets_at_q,
                self._weights_at_q,
                self._frequencies,
                self._temperature
                * get_physical_units().KB
                / get_physical_units().THzToEv,
                self._g,
                _g_zero,
                self._cutoff_frequency,
                i,
            )
            self._imag_self_energy[i] = ise_at_f
        self._imag_self_energy *= self._unit_conversion

    def _run_py_with_band_indices_with_g(self) -> None:
        assert self._temperature is not None
        if self._temperature > 0:
            self._ise_thm_with_band_indices()
        else:
            self._ise_thm_with_band_indices_0K()

    def _run_gpu_with_band_indices_with_g(self):
        """CuPy implementation of _ise_thm_with_band_indices.

        Vectorizes the sum over triplets and bands on GPU using batched
        processing to avoid out-of-memory errors for large systems.

        If pp_strength is already on GPU (from GPU interaction calculation),
        uses it directly to avoid CPU->GPU transfer overhead.
        """
        import time
        timing = {}
        t_total_start = time.perf_counter()

        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError(
                "GPU imag_self_energy requires CuPy. "
                "Install cupy-cudaXX or use lang='C'."
            ) from exc

        if self._pp_strength is None or self._g is None:
            raise RuntimeError("pp_strength or integration weights are not set.")

        # Check if pp_strength is already on GPU (avoids CPU->GPU transfer)
        pp_on_gpu = self._pp_strength_gpu is not None

        t_start = time.perf_counter()
        cutoff = self._cutoff_frequency
        triplets = self._triplets_at_q
        weights = self._weights_at_q
        freqs_all = self._frequencies  # (num_bz, num_band)

        num_triplets = len(triplets)
        num_band0 = self._pp_strength.shape[1]
        num_band = self._pp_strength.shape[2]

        # Estimate batch size based on available GPU memory
        # Each triplet needs: g(2*B0*B*B) + pp(B0*B*B) + intermediates
        # Use conservative estimate: ~10x the raw data size for intermediates
        # If pp is already on GPU, we need less transfer bandwidth
        detail_factor = 16 if self._with_detail else 10
        bytes_per_triplet = 8 * (
            2 * num_band0 * num_band * num_band
            + num_band0 * num_band * num_band
            + detail_factor * num_band0 * num_band * num_band
        )

        try:
            free_mem = cp.cuda.Device().mem_info[0]  # Free memory in bytes
            # Use at most 50% of free memory, with minimum batch size of 10
            max_batch = max(1, int(0.6 * free_mem / bytes_per_triplet))
        except Exception:
            max_batch = 10  # Fallback default

        batch_size = min(max_batch, num_triplets)
        timing['setup'] = time.perf_counter() - t_start

        # Pre-compute Bose-Einstein factors on CPU (cheap, avoids repeated GPU transfers)
        t_start = time.perf_counter()
        f2 = freqs_all[triplets[:, 1]]  # (T,B)
        f3 = freqs_all[triplets[:, 2]]  # (T,B)
        f_stack = np.stack((f2, f3), axis=1)  # (T,2,B)
        f_stack = np.where(f_stack > cutoff, f_stack, 1.0)

        from phono3py.phonon.func import bose_einstein
        n_all = bose_einstein(f_stack, self._temperature)  # (T,2,B)

        # Pre-compute frequency masks on CPU
        mask2_all = f2 > cutoff  # (T,B)
        mask3_all = f3 > cutoff  # (T,B)
        timing['cpu_precompute'] = time.perf_counter() - t_start

        # Accumulate result on GPU
        result_gpu = cp.zeros(num_band0, dtype=cp.float64)

        # Process triplets in batches
        t_batch_start = time.perf_counter()
        timing['batch_transfer'] = 0.0
        timing['batch_compute'] = 0.0
        num_batches = 0

        # Print array sizes for debugging OOM
        g_shape = self._g.shape
        g_size_per_triplet = 2 * num_band0 * num_band * num_band * 8 / 1e9
        print(f"  [DEBUG] g array: shape={g_shape}, size_per_triplet={g_size_per_triplet:.2f} GB")
        print(f"  [DEBUG] batch_size={batch_size}, total g_batch size={batch_size * g_size_per_triplet:.2f} GB")

        for batch_start in range(0, num_triplets, batch_size):
            batch_end = min(batch_start + batch_size, num_triplets)
            bs = batch_end - batch_start  # Current batch size
            num_batches += 1

            # Slice data for this batch
            t_xfer = time.perf_counter()
            n_batch = n_all[batch_start:batch_end]  # (bs, 2, B)
            mask2_batch = mask2_all[batch_start:batch_end]  # (bs, B)
            mask3_batch = mask3_all[batch_start:batch_end]  # (bs, B)
            g_batch = self._g[:, batch_start:batch_end]  # (2, bs, B0, B, B)
            w_batch = weights[batch_start:batch_end]  # (bs,)

            # Debug: print sizes for first batch or when OOM might occur
            if num_batches == 1:
                free_mem, total_mem = cp.cuda.Device().mem_info
                print(f"  [DEBUG] GPU memory: {free_mem/1e9:.2f} GB free / {total_mem/1e9:.2f} GB total")
                print(f"  [DEBUG] First batch sizes:")
                print(f"    - g_batch: shape={g_batch.shape}, size={g_batch.nbytes/1e9:.2f} GB")
                print(f"    - n_batch: shape={n_batch.shape}, size={n_batch.nbytes/1e9:.2f} GB")
                if not pp_on_gpu:
                    pp_batch_temp = self._pp_strength[batch_start:batch_end]
                    print(f"    - pp_batch: shape={pp_batch_temp.shape}, size={pp_batch_temp.nbytes/1e9:.2f} GB")

            # Transfer batch to GPU with error handling
            try:
                n_cp = cp.asarray(n_batch, dtype=cp.float64)
            except cp.cuda.memory.OutOfMemoryError as e:
                print(f"  [OOM] Failed to allocate n_batch: shape={n_batch.shape}, size={n_batch.nbytes/1e9:.2f} GB")
                raise

            try:
                g_cp = cp.asarray(g_batch, dtype=cp.float64)
            except cp.cuda.memory.OutOfMemoryError as e:
                print(f"  [OOM] Failed to allocate g_batch: shape={g_batch.shape}, size={g_batch.nbytes/1e9:.2f} GB")
                raise

            try:
                w_cp = cp.asarray(w_batch, dtype=cp.float64)[:, None, None, None]
            except cp.cuda.memory.OutOfMemoryError as e:
                print(f"  [OOM] Failed to allocate w_batch: shape={w_batch.shape}, size={w_batch.nbytes/1e9:.2f} GB")
                raise

            # Get pp_strength: use GPU array directly if available (no transfer!)
            if pp_on_gpu:
                pp_cp = self._pp_strength_gpu[batch_start:batch_end]  # Already on GPU
            else:
                try:
                    pp_batch = self._pp_strength[batch_start:batch_end]  # (bs, B0, B, B)
                    pp_cp = cp.asarray(pp_batch, dtype=cp.float64)
                except cp.cuda.memory.OutOfMemoryError as e:
                    print(f"  [OOM] Failed to allocate pp_batch: shape={pp_batch.shape}, size={pp_batch.nbytes/1e9:.2f} GB")
                    raise

            # Build frequency mask on GPU
            mask2_cp = cp.asarray(mask2_batch)
            mask3_cp = cp.asarray(mask3_batch)
            cp.cuda.Stream.null.synchronize()
            timing['batch_transfer'] += time.perf_counter() - t_xfer

            t_compute = time.perf_counter()
            try:
                freq_mask = (mask2_cp[:, None, :, None] & mask3_cp[:, None, None, :]).astype(cp.float64)
                del mask2_cp, mask3_cp  # Free immediately

                # Extract components
                n2 = n_cp[:, 0, :]  # (bs, B)
                n3 = n_cp[:, 1, :]  # (bs, B)
                g0 = g_cp[0]  # (bs, B0, B, B)
                g1 = g_cp[1]  # (bs, B0, B, B)
                del n_cp, g_cp  # Free intermediate references

                # Broadcast Bose factors: (bs, 1, B, B)
                n2_b = n2[:, None, :, None]
                n3_b = n3[:, None, None, :]
                del n2, n3

                # Compute contribution for this batch
                # Use in-place operations where possible to reduce peak memory
                term_plus = n2_b + n3_b
                term_plus += 1.0  # in-place
                term_minus = n2_b - n3_b
                del n2_b, n3_b

                # coeff = (term_plus * g0 + term_minus * g1) * w * freq_mask
                coeff = term_plus * g0
                del term_plus
                coeff += term_minus * g1
                del term_minus, g0, g1
                coeff *= freq_mask
                del freq_mask

                if self._with_detail:
                    # Keep triplet-resolved contribution for write_gamma_detail path.
                    # This is intentionally unweighted by triplet multiplicity so that
                    # readers can apply `weight` exactly as done in CPU workflows.
                    # Shape: (batch_triplets, band0, band1, band2)
                    detailed_batch = coeff * pp_cp
                    weighted_batch = detailed_batch * w_cp
                    batch_contrib = cp.sum(weighted_batch, axis=(0, 2, 3))
                    result_gpu += batch_contrib
                    self._detailed_imag_self_energy[batch_start:batch_end] = cp.asnumpy(
                        detailed_batch
                    )
                    del detailed_batch, weighted_batch, batch_contrib, w_cp, pp_cp, coeff
                else:
                    # Sum over triplets and band indices: (B0,)
                    coeff *= w_cp
                    del w_cp
                    coeff *= pp_cp
                    del pp_cp
                    batch_contrib = cp.sum(coeff, axis=(0, 2, 3))
                    del coeff

                    result_gpu += batch_contrib
                    del batch_contrib
            except cp.cuda.memory.OutOfMemoryError as e:
                free_mem, total_mem = cp.cuda.Device().mem_info
                print(f"  [OOM] Failed during compute phase!")
                print(f"  [OOM] GPU memory: {free_mem/1e9:.2f} GB free / {total_mem/1e9:.2f} GB total")
                print(f"  [OOM] Batch size was: {bs}, g0 shape: ({bs}, {num_band0}, {num_band}, {num_band})")
                print(f"  [OOM] Each g0/g1 array: {bs * num_band0 * num_band * num_band * 8 / 1e9:.2f} GB")
                print(f"  [OOM] Total for intermediate arrays (coeff, term_plus, etc): ~{4 * bs * num_band0 * num_band * num_band * 8 / 1e9:.2f} GB")
                raise

            cp.cuda.Stream.null.synchronize()
            timing['batch_compute'] += time.perf_counter() - t_compute

            # Aggressively free GPU memory between batches
            cp.get_default_memory_pool().free_all_blocks()

        timing['batch_loop_total'] = time.perf_counter() - t_batch_start

        t_start = time.perf_counter()
        imag = cp.asnumpy(result_gpu) * self._unit_conversion
        del result_gpu
        cp.get_default_memory_pool().free_all_blocks()
        timing['result_transfer'] = time.perf_counter() - t_start

        self._imag_self_energy[:] = imag
        if self._with_detail:
            self._detailed_imag_self_energy *= self._unit_conversion

        timing['total'] = time.perf_counter() - t_total_start

        # Print timing summary
        pp_status = "ON GPU (no transfer)" if pp_on_gpu else "CPU->GPU transfer"
        print(
            f"\n[ISE GPU TIMING] _run_gpu_with_band_indices_with_g "
            f"({num_triplets} triplets, {num_band} bands):"
        )
        print(f"  Batch size: {batch_size}, Num batches: {num_batches}")
        print(f"  pp_strength: {pp_status}")
        print(f"  with_detail: {self._with_detail}")
        print(f"  Setup:              {timing['setup']:8.4f} s")
        print(f"  CPU precompute:     {timing['cpu_precompute']:8.4f} s")
        print(f"  Batch loop total:   {timing['batch_loop_total']:8.4f} s")
        print(f"    - Transfer:       {timing['batch_transfer']:8.4f} s")
        print(f"    - Compute:        {timing['batch_compute']:8.4f} s")
        print(f"  Result transfer:    {timing['result_transfer']:8.4f} s")
        print(f"  ----------------------------------------")
        print(f"  TOTAL:              {timing['total']:8.4f} s\n", flush=True)

    def _ise_thm_with_band_indices(self) -> None:
        assert self._frequencies is not None
        assert self._triplets_at_q is not None
        assert self._temperature is not None
        assert self._weights_at_q is not None
        assert self._pp_strength is not None
        assert self._g is not None
        assert self._imag_self_energy is not None
        freqs = self._frequencies[self._triplets_at_q[:, [1, 2]]]
        freqs = np.where(freqs > self._cutoff_frequency, freqs, 1)
        n = bose_einstein(freqs, self._temperature)
        for i, (tp, w, interaction) in enumerate(
            zip(self._triplets_at_q, self._weights_at_q, self._pp_strength, strict=True)
        ):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                f1 = self._frequencies[tp[1]][j]
                f2 = self._frequencies[tp[2]][k]
                if f1 > self._cutoff_frequency and f2 > self._cutoff_frequency:
                    n2 = n[i, 0, j]
                    n3 = n[i, 1, k]
                    g1 = self._g[0, i, :, j, k]
                    g2_g3 = self._g[1, i, :, j, k]  # g2 - g3
                    self._imag_self_energy[:] += (
                        ((n2 + n3 + 1) * g1 + (n2 - n3) * (g2_g3))
                        * interaction[:, j, k]
                        * w
                    )

        self._imag_self_energy *= self._unit_conversion

    def _ise_thm_with_band_indices_0K(self) -> None:
        assert self._weights_at_q is not None
        assert self._pp_strength is not None
        assert self._g is not None
        assert self._imag_self_energy is not None
        for i, (w, interaction) in enumerate(
            zip(self._weights_at_q, self._pp_strength, strict=True)
        ):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[0, i, :, j, k]
                self._imag_self_energy[:] += g1 * interaction[:, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _run_py_with_frequency_points_with_g(self) -> None:
        assert self._temperature is not None
        if self._temperature > 0:
            self._ise_thm_with_frequency_points()
        else:
            self._ise_thm_with_frequency_points_0K()

    def _ise_thm_with_frequency_points(self) -> None:
        assert self._triplets_at_q is not None
        assert self._weights_at_q is not None
        assert self._pp_strength is not None
        assert self._frequencies is not None
        assert self._temperature is not None
        assert self._g is not None
        assert self._imag_self_energy is not None
        for i, (tp, w, interaction) in enumerate(
            zip(self._triplets_at_q, self._weights_at_q, self._pp_strength, strict=True)
        ):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                f1 = self._frequencies[tp[1]][j]
                f2 = self._frequencies[tp[2]][k]
                if f1 > self._cutoff_frequency and f2 > self._cutoff_frequency:
                    n2 = bose_einstein(f1, self._temperature)
                    n3 = bose_einstein(f2, self._temperature)
                    g1 = self._g[0, i, :, j, k]
                    g2_g3 = self._g[1, i, :, j, k]  # g2 - g3
                    for ll in range(len(interaction)):
                        self._imag_self_energy[:, ll] += (
                            ((n2 + n3 + 1) * g1 + (n2 - n3) * (g2_g3))
                            * interaction[ll, j, k]
                            * w
                        )

        self._imag_self_energy *= self._unit_conversion

    def _ise_thm_with_frequency_points_0K(self) -> None:
        assert self._weights_at_q is not None
        assert self._pp_strength is not None
        assert self._g is not None
        assert self._imag_self_energy is not None
        for i, (w, interaction) in enumerate(
            zip(self._weights_at_q, self._pp_strength, strict=True)
        ):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[0, i, :, j, k]
                for ll in range(len(interaction)):
                    self._imag_self_energy[:, ll] += g1 * interaction[ll, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _average_by_degeneracy(
        self, imag_self_energy: NDArray[np.double]
    ) -> NDArray[np.double]:
        assert self._frequencies is not None
        assert self._grid_point is not None
        return average_by_degeneracy(
            imag_self_energy, self._pp.band_indices, self._frequencies[self._grid_point]
        )


def get_detailed_imag_self_energy_from_g(
    interaction: Interaction,
    temperature: float,
    g: np.ndarray,
    g_zero: Optional[np.ndarray] = None,
    pp_strength: Optional[np.ndarray] = None,
    unit_conversion_factor: Optional[float] = None,
) -> np.ndarray:
    """Return unweighted per-triplet gamma_detail from integration weights.

    This mirrors the formula used by ``ImagSelfEnergy`` and by
    ``write_gamma_detail`` for frequency_points_at_bands=True:

    ``((n1 + n2 + 1) * g[0] + (n1 - n2) * g[1]) * |V3|^2 * unit``.

    The returned array is intentionally not multiplied by triplet weights,
    matching the HDF5 ``gamma_detail`` convention. Callers that assemble row
    sums or W-like matrices should apply ``interaction.get_triplets_at_q()[1]``
    exactly once.
    """
    if pp_strength is None:
        pp_strength = interaction.interaction_strength
    if pp_strength is None:
        raise RuntimeError("pp_strength is required; run interaction first.")
    if g.shape[0] != 2:
        raise ValueError(f"ISE gamma_detail requires two integration weights, got {g.shape[0]}.")

    frequencies = interaction.get_phonons()[0]
    triplets = interaction.get_triplets_at_q()[0]
    cutoff = interaction.cutoff_frequency
    if unit_conversion_factor is None:
        unit_conversion_factor = ImagSelfEnergy(interaction).unit_conversion_factor

    f1 = frequencies[triplets[:, 1]]
    f2 = frequencies[triplets[:, 2]]
    valid = (f1 > cutoff) & (f2 > cutoff)
    safe_f1 = np.where(f1 > cutoff, f1, 1.0)
    safe_f2 = np.where(f2 > cutoff, f2, 1.0)
    n1 = bose_einstein(safe_f1, temperature)
    n2 = bose_einstein(safe_f2, temperature)

    coeff = (
        (n1[:, None, :, None] + n2[:, None, None, :] + 1.0) * g[0]
        + (n1[:, None, :, None] - n2[:, None, None, :]) * g[1]
    )
    coeff *= valid[:, None, :, None]
    if g_zero is not None:
        coeff = np.where(np.asarray(g_zero, dtype=bool), 0.0, coeff)
    return coeff * pp_strength * float(unit_conversion_factor)


def get_imag_self_energy(
    interaction: Interaction,
    grid_points: NDArray[np.int64] | Sequence[int],
    temperatures: NDArray[np.double] | Sequence[float],
    sigmas: Sequence[float | None] | None = None,
    frequency_points: NDArray[np.double] | Sequence[float] | None = None,
    frequency_step: float | None = None,
    num_frequency_points: int | None = None,
    frequency_points_at_bands: bool = False,
    num_points_in_batch: int | None = None,
    scattering_event_class: Literal[1, 2] | None = None,
    write_gamma_detail: bool = False,
    return_gamma_detail: bool = False,
    return_reduced_gamma_detail: bool = False,
    output_filename: str | None = None,
    log_level: int = 0,
    lang: Literal["C", "Python", "Rust", "GPU", "GPU_phase", "Hybrid"] = "Rust",
) -> (
    tuple[NDArray[np.double] | None, NDArray[np.double]]
    | tuple[
        NDArray[np.double] | None,
        NDArray[np.double],
        list[NDArray[np.double]]
        | list[tuple[NDArray[np.double], NDArray[np.double]]],
    ]
):
    """Imaginary-part of self-energy at frequency points.

    Band indices to be calculated at are found in Interaction instance.

    Parameters
    ----------
    interaction : Interaction
        Ph-ph interaction.
    grid_points : array_like
        Grid-point indices where imag-self-energies are calculated.
        dtype=int, shape=(grid_points,)
    temperatures : array_like
        Temperatures where imag-self-energies are calculated.
        dtype=float, shape=(temperatures,)
    sigmas : Sequence, optional
        A set of sigmas. simgas=[None, ] means to use tetrahedron method,
        otherwise smearing method with real positive value of sigma.
        For example, sigmas=[None, 0.01, 0.03] is possible. Default is None,
        which results in [None, ].
        dtype=float, shape=(sigmas,)
    frequency_points : array_like, optional
        Frequency sampling points. Default is None. With
        frequency_points_at_bands=False and frequency_points is None,
        num_frequency_points or frequency_step is used to generate uniform
        frequency sampling points.
        dtype=float, shape=(frequency_points,)
    frequency_step : float, optional
        Uniform pitch of frequency sampling points. Default is None. This
        results in using num_frequency_points.
    num_frequency_points: int, optional
        Number of sampling sampling points to be used instead of
        frequency_step. This number includes end points. Default is None,
        which gives 201.
    frequency_points_at_bands : bool, optional
        Phonon band frequencies are used as frequency points when True.
        Default is False.
    num_points_in_batch: int, optional
        Number of sampling points in one batch. This is for the frequency
        sampling mode and the sampling points are divided into batches.
        Lager number provides efficient use of multi-cores but more
        memory demanding. Default is None, which give the number of 10.
    scattering_event_class : int, optional
        Specific choice of scattering event class, 1 or 2 that is specified
        1 or 2, respectively. The result is stored in gammas. Therefore
        usual gammas are not stored in the variable. Default is None, which
        doesn't specify scattering_event_class.
    write_gamma_detail : bool, optional
        Detailed gammas are written into a file in hdf5. Default is False.
    return_gamma_detail : bool, optional
        With True, detailed gammas are returned. Default is False.
    log_level: int
        Log level. Default is 0.
    lang: str, optional
        Language to use for the calculation. Default is "C".
    Returns
    -------
    tuple :
        (frequency_points, gammas) are returned. With return_gamma_detail=True,
        (frequency_points, gammas, detailed_gamma) are returned.
        detailed_gamma is a list of detailed_gamma_at_gp's.

        When frequency_points_at_bands is True,

            gamma.shape = (sigmas, temperatures, grid_points, band_indices)
            detailed_gamma_at_gp.shape = (sigmas, temperatures, triplets,
                                          band_indices, num_band, num_band)
        else:
            detailed_gamma_at_gp = np.zeros(
                (len(_sigmas), len(temperatures), _num_frequency_points,
                 len(weights), num_band0, num_band, num_band),
                dtype='double')

        otherwise

            gamma.shape = (sigmas, temperatures, grid_points,
                           band_indices, frequency_points)
            detailed_gamma_at_gp.shape = (sigmas, temperatures, triplets,
                                          frequency_points,
                                          band_indices, num_band, num_band)

    """
    if sigmas is None:
        _sigmas: Sequence[float | None] = [None]
    else:
        _sigmas = sigmas

    _temperatures = np.asarray(temperatures, dtype="double")

    if not interaction.phonon_all_done:
        if log_level:
            print("Running harmonic phonon calculations...")
        interaction.run_phonon_solver()

    # Set phonon at Gamma without NAC for finding max_phonon_freq.
    interaction.run_phonon_solver_at_gamma()
    max_phonon_freq = float(np.amax(interaction.get_phonons()[0]))  # type: ignore[arg-type]
    interaction.run_phonon_solver_at_gamma(is_nac=True)

    num_band0 = len(interaction.band_indices)

    if frequency_points_at_bands:
        _frequency_points = None
        _num_frequency_points = num_band0
        gamma = np.zeros(  # type: ignore[call-overload, assignment]
            (len(_sigmas), len(_temperatures), len(grid_points), _num_frequency_points),
            dtype="double",
            order="C",
        )
    else:
        _frequency_points = get_frequency_points(
            max_phonon_freq=max_phonon_freq,
            sigmas=_sigmas,
            frequency_points=frequency_points,
            frequency_step=frequency_step,
            num_frequency_points=num_frequency_points,
        )
        _num_frequency_points = len(_frequency_points)
        gamma = np.zeros(  # type: ignore[call-overload, assignment]
            (
                len(_sigmas),
                len(_temperatures),
                len(grid_points),
                num_band0,
                _num_frequency_points,
            ),
            dtype="double",
            order="C",
        )

    detailed_gamma: list[NDArray[np.double]] = []
    reduced_gamma: list[
        tuple[NDArray[np.double], NDArray[np.double]]
        | list[tuple[NDArray[np.double], NDArray[np.double]]]
    ] = []

    ise = ImagSelfEnergy(
        interaction,
        with_detail=(write_gamma_detail or return_gamma_detail or return_reduced_gamma_detail),
        lang=lang,
    )
    for i, gp in enumerate(grid_points):
        ise.set_grid_point(gp)

        if log_level:
            bz_grid = interaction.bz_grid
            weights = interaction.get_triplets_at_q()[1]
            assert weights is not None
            if len(grid_points) > 1:
                print(
                    "---------------- Imaginary part of self energy -o- (%d/%d) "
                    "----------------" % (i + 1, len(grid_points))
                )
            else:
                print(
                    "-------------------- Imaginary part of self energy -o- "
                    "--------------------"
                )
            print("Grid point: %d" % gp)
            print("Number of ir-triplets: %d / %d" % (len(weights), weights.sum()))
        ise.run_interaction()
        frequencies = interaction.get_phonons()[0][gp]  # type: ignore[index]

        if log_level:
            qpoint = np.dot(bz_grid.QDinv, bz_grid.addresses[gp])
            print("Phonon frequencies at (%4.2f, %4.2f, %4.2f):" % tuple(qpoint))
            for bi, freq in enumerate(frequencies):
                print("%3d  %f" % (bi + 1, freq))
            sys.stdout.flush()

        _get_imag_self_energy_at_gp(
            gamma,
            detailed_gamma,
            reduced_gamma,
            i,
            gp,
            _sigmas,
            _temperatures,
            _frequency_points,
            _num_frequency_points,
            scattering_event_class,
            num_points_in_batch,
            interaction,
            ise,
            write_gamma_detail,
            return_gamma_detail,
            return_reduced_gamma_detail,
            output_filename,
            log_level,
        )

    if return_reduced_gamma_detail:
        return _frequency_points, gamma, reduced_gamma
    if return_gamma_detail:
        return _frequency_points, gamma, detailed_gamma
    return _frequency_points, gamma


def _get_imag_self_energy_at_gp(
    gamma: NDArray[np.double],
    detailed_gamma: list[NDArray[np.double]],
    reduced_gamma: list[
        tuple[NDArray[np.double], NDArray[np.double]]
        | list[tuple[NDArray[np.double], NDArray[np.double]]]
    ],
    i: int,
    gp: int,
    sigmas: Sequence[float | None],
    temperatures: NDArray[np.double],
    frequency_points: NDArray[np.double] | None,
    num_frequency_points: int,
    scattering_event_class: Literal[1, 2] | None,
    num_points_in_batch: int | None,
    interaction: Interaction,
    ise: ImagSelfEnergy,
    write_gamma_detail: bool,
    return_gamma_detail: bool,
    return_reduced_gamma_detail: bool,
    output_filename: str | None,
    log_level: int,
) -> None:
    if return_reduced_gamma_detail and frequency_points is not None:
        raise NotImplementedError(
            "return_reduced_gamma_detail is only supported with "
            "frequency_points_at_bands=True (band-indices mode)."
        )
    num_band0 = len(interaction.band_indices)
    frequencies = interaction.get_phonons()[0]
    assert frequencies is not None
    mesh = interaction.mesh_numbers
    bz_grid = interaction.bz_grid

    if write_gamma_detail or return_gamma_detail:
        triplets, weights, _, _ = interaction.get_triplets_at_q()
        assert weights is not None
        num_band = frequencies.shape[1]
        if frequency_points is None:
            detailed_gamma_at_gp = np.zeros(
                (
                    len(sigmas),
                    len(temperatures),
                    len(weights),
                    num_band0,
                    num_band,
                    num_band,
                ),
                dtype="double",
            )
        else:
            detailed_gamma_at_gp = np.zeros(  # type: ignore[call-overload, assignment]
                (
                    len(sigmas),
                    len(temperatures),
                    num_frequency_points,
                    len(weights),
                    num_band0,
                    num_band,
                    num_band,
                ),
                dtype="double",
            )
    else:
        detailed_gamma_at_gp = None

    reduced_gamma_at_gp: (
        list[tuple[NDArray[np.double], NDArray[np.double]]] | None
    ) = [] if return_reduced_gamma_detail else None

    for j, sigma in enumerate(sigmas):
        if log_level:
            if sigma:
                print("Sigma: %s" % sigma)
            else:
                print("Tetrahedron method is used for BZ integration.")

        ise.set_sigma(sigma)
        _get_imag_self_energy_at_sigma(
            gamma,
            detailed_gamma_at_gp,
            reduced_gamma_at_gp,
            i,
            j,
            temperatures,
            frequency_points,
            scattering_event_class,
            num_points_in_batch,
            ise,
            write_gamma_detail,
            return_gamma_detail,
            return_reduced_gamma_detail,
            log_level,
        )

        if write_gamma_detail:
            assert detailed_gamma_at_gp is not None
            full_filename = write_gamma_detail_to_hdf5(
                temperatures,
                mesh,
                bz_grid=bz_grid,
                gamma_detail=detailed_gamma_at_gp[j],
                grid_point=gp,
                triplet=triplets,
                weight=weights,
                sigma=sigma,
                frequency_points=frequency_points,
                filename=output_filename,
            )

            if log_level:
                print(
                    "Contribution of each triplet to imaginary part of "
                    'self energy is written in\n"%s".' % full_filename
                )

        if return_gamma_detail:
            detailed_gamma.append(detailed_gamma_at_gp)

    if return_reduced_gamma_detail:
        assert reduced_gamma_at_gp is not None
        # The driver uses a single sigma and a single temperature, in which case
        # reduced_gamma_at_gp holds exactly one (gamma_q1_sum, gamma_q2_sum)
        # tuple; expose that tuple directly. For multiple (sigma, temperature)
        # combinations, keep the full list of tuples.
        if len(reduced_gamma_at_gp) == 1:
            reduced_gamma.append(reduced_gamma_at_gp[0])
        else:
            reduced_gamma.append(reduced_gamma_at_gp)


def _get_imag_self_energy_at_sigma(
    gamma: NDArray[np.double],
    detailed_gamma_at_gp: NDArray[np.double] | None,
    reduced_gamma_at_gp: list[tuple[NDArray[np.double], NDArray[np.double]]] | None,
    i: int,
    j: int,
    temperatures: NDArray[np.double],
    _frequency_points: NDArray[np.double] | None,
    scattering_event_class: Literal[1, 2] | None,
    num_points_in_batch: int | None,
    ise: ImagSelfEnergy,
    write_gamma_detail: bool,
    return_gamma_detail: bool,
    return_reduced_gamma_detail: bool,
    log_level: int,
) -> None:
    # Run one by one at frequency points
    if detailed_gamma_at_gp is None:
        detailed_gamma_at_gp_at_j = None
    else:
        detailed_gamma_at_gp_at_j = detailed_gamma_at_gp[j]

    if _frequency_points is None:
        ise.run_integration_weights(scattering_event_class=scattering_event_class)
        for k, t in enumerate(temperatures):
            ise.temperature = t
            ise.run()
            gamma[j, k, i] = ise.imag_self_energy
            if write_gamma_detail or return_gamma_detail:
                assert detailed_gamma_at_gp_at_j is not None
                detailed_gamma_at_gp_at_j[k] = ise.detailed_imag_self_energy
            if return_reduced_gamma_detail:
                # Reduce immediately and keep only the band-summed views, so the
                # full per-(sigma, temperature) detailed gamma is never retained.
                reduced_ise = ise.reduced_imag_self_energy
                assert reduced_ise is not None
                assert reduced_gamma_at_gp is not None
                q1, q2 = reduced_ise
                reduced_gamma_at_gp.append(
                    (np.array(q1, copy=True), np.array(q2, copy=True))
                )
    else:
        run_ise_at_frequency_points_batch(
            i,
            j,
            _frequency_points,
            ise,
            temperatures,
            gamma,
            write_gamma_detail=write_gamma_detail,
            return_gamma_detail=return_gamma_detail,
            detailed_gamma_at_gp=detailed_gamma_at_gp_at_j,
            scattering_event_class=scattering_event_class,
            nelems_in_batch=num_points_in_batch,
            log_level=log_level,
        )


def get_frequency_points(
    max_phonon_freq: float | None = None,
    sigmas: Sequence[float | None] | None = None,
    frequency_points: Sequence[float] | NDArray[np.double] | None = None,
    frequency_step: float | None = None,
    num_frequency_points: int | None = None,
) -> NDArray[np.double]:
    """Generate frequency points.

    This function may be mostly used for the phonon frequency axis of
    spectrum-like calculations.

    """
    if frequency_points is None:
        assert max_phonon_freq is not None
        if sigmas is not None:
            sigma_vals = [sigma for sigma in sigmas if sigma is not None]
        else:
            sigma_vals = []
        if sigma_vals:
            fmax = max_phonon_freq * 2 + np.max(sigma_vals) * 4
        else:
            fmax = max_phonon_freq * 2
        fmax *= 1.005
        fmin = 0
        _frequency_points = _sample_frequency_points(
            fmin,
            fmax,
            frequency_step=frequency_step,
            num_frequency_points=num_frequency_points,
        )
    else:
        _frequency_points = np.array(frequency_points, dtype="double")

    return _frequency_points


def _sample_frequency_points(
    f_min: float,
    f_max: float,
    frequency_step: float | None = None,
    num_frequency_points: int | None = None,
) -> NDArray[np.double]:
    frequency_points: NDArray[np.double]
    if num_frequency_points is None:
        if frequency_step is not None:
            frequency_points = np.arange(f_min, f_max, frequency_step, dtype="double")
        else:
            frequency_points = np.array(np.linspace(f_min, f_max, 201), dtype="double")
    else:
        frequency_points = np.array(
            np.linspace(f_min, f_max, num_frequency_points), dtype="double"
        )

    return frequency_points


def write_imag_self_energy(
    imag_self_energy: NDArray[np.double],
    mesh: NDArray[np.int64],
    grid_points: Sequence[int] | NDArray[np.int64],
    band_indices: Sequence[NDArray[np.int64]],
    frequency_points: NDArray[np.double] | None,
    temperatures: Sequence[float] | NDArray[np.double],
    sigmas: Sequence[float | None],
    scattering_event_class: Literal[1, 2] | None = None,
    output_filename: str | None = None,
    is_mesh_symmetry: bool = True,
    log_level: int = 0,
) -> None:
    """Write imaginary-part of self-energies into text files."""
    for sigma, ise_temps in zip(sigmas, imag_self_energy, strict=True):
        for t, ise_gps in zip(temperatures, ise_temps, strict=True):
            for gp, ise in zip(grid_points, ise_gps, strict=True):
                for i, bi in enumerate(band_indices):
                    pos = 0
                    for j in range(i):
                        pos += len(band_indices[j])
                    filename = write_imag_self_energy_at_grid_point(
                        gp,
                        bi,
                        mesh,
                        frequency_points,
                        ise[pos : (pos + len(bi))].sum(axis=0) / len(bi),
                        sigma=sigma,
                        temperature=t,
                        scattering_event_class=scattering_event_class,
                        filename=output_filename,
                        is_mesh_symmetry=is_mesh_symmetry,
                    )
                    if log_level:
                        print(
                            "Imaginary part of self-energies were "
                            'written to "%s".' % filename
                        )


def average_by_degeneracy(
    imag_self_energy: NDArray[np.double],
    band_indices: NDArray[np.int64],
    freqs_at_gp: NDArray[np.double],
) -> NDArray[np.double]:
    """Take averages of values of energetically degenerated bands."""
    deg_sets = degenerate_sets(freqs_at_gp)
    imag_se = np.zeros_like(imag_self_energy)
    for dset in deg_sets:
        dset_s = set(dset)
        bi_set = [i for i, bi in enumerate(band_indices) if bi in dset_s]
        for i in bi_set:
            if imag_self_energy.ndim == 1:
                imag_se[i] = imag_self_energy[bi_set].sum() / len(bi_set)
            else:
                imag_se[:, i] = imag_self_energy[:, bi_set].sum(axis=1) / len(bi_set)
    return imag_se


def run_ise_at_frequency_points_batch(
    i: int,
    j: int,
    _frequency_points: NDArray[np.double],
    ise: ImagSelfEnergy,
    temperatures: NDArray[np.double],
    gamma: NDArray[np.double],
    write_gamma_detail: bool = False,
    return_gamma_detail: bool = False,
    detailed_gamma_at_gp: NDArray[np.double] | None = None,
    scattering_event_class: Literal[1, 2] | None = None,
    nelems_in_batch: int | None = 50,
    log_level: int = 0,
) -> None:
    """Run calculations at frequency points batch by batch.

    See the details about batch in docstring of ``get_imag_self_energy``.

    """
    if nelems_in_batch is None:
        _nelems_in_batch = 10
    else:
        _nelems_in_batch = nelems_in_batch

    batches = get_freq_points_batches(len(_frequency_points), _nelems_in_batch)

    if log_level:
        print(
            "Calculations at %d frequency points are divided into "
            "%d batches." % (len(_frequency_points), len(batches))
        )

    for bi, fpts_batch in enumerate(batches):
        if log_level:
            print("%d/%d: %s" % (bi + 1, len(batches), fpts_batch + 1))
            sys.stdout.flush()

        ise.frequency_points = _frequency_points[fpts_batch]
        ise.run_integration_weights(scattering_event_class=scattering_event_class)
        for ll, t in enumerate(temperatures):
            ise.temperature = t
            ise.run()
            gamma[j, ll, i, :, fpts_batch] = ise.imag_self_energy
            if write_gamma_detail or return_gamma_detail:
                assert detailed_gamma_at_gp is not None
                detailed_gamma_at_gp[ll, fpts_batch] = ise.detailed_imag_self_energy


def get_freq_points_batches(
    tot_nelems: int, nelems: int | None = None
) -> list[NDArray[np.int64]]:
    """Divide frequency points into batches."""
    if nelems is None:
        _nelems = 10
    else:
        _nelems = nelems
    nbatch = tot_nelems // _nelems
    batches = [
        np.arange(i * _nelems, (i + 1) * _nelems, dtype="int64") for i in range(nbatch)
    ]
    if tot_nelems % _nelems > 0:
        batches.append(np.arange(_nelems * nbatch, tot_nelems, dtype="int64"))
    return batches
