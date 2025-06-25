"""Transform fc3 in reciprocal space to phonon space."""

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

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms


class ReciprocalToNormalFast:
    """Class to transform fc3 in reciprocal space to phonon space.

    This is an implementation in python for prototyping and the test.
    Equivalent routine is implemented in C, and this is what usually
    we use.

    """

    def __init__(
        self,
        primitive: PhonopyAtoms,
        frequencies: np.ndarray,
        eigenvectors: np.ndarray,
        band_indices: Union[Sequence[int], np.ndarray],
        cutoff_frequency: float = 0,
    ):
        """Init method."""
        self._primitive = primitive
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._band_indices = band_indices
        self._cutoff_frequency = cutoff_frequency

        self._fc3_normal: np.ndarray
        self._fc3_reciprocal: np.ndarray

    def run(self, fc3_reciprocal, grid_triplet):
        """Calculate fc3 in phonon coordinates."""
        num_band = len(self._primitive) * 3
        self._fc3_reciprocal = fc3_reciprocal
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._fc3_normal = np.zeros(
            (len(self._band_indices), num_band, num_band), dtype=dtype
        )
        self._reciprocal_to_normal(grid_triplet)

    def get_reciprocal_to_normal(self) -> Optional[np.ndarray]:
        """Return fc3 in phonon coordinates."""
        return self._fc3_normal

    def _reciprocal_to_normal(self, grid_triplet):
        e1, e2, e3 = self._eigenvectors[grid_triplet]
        f1, f2, f3 = self._frequencies[grid_triplet]
        num_band = len(f1)
        cutoff = self._cutoff_frequency
        m = self._primitive.masses
        
        # Create masks for bands with frequencies above cutoff
        band_indices = self._band_indices
        mask1 = f1[band_indices] > cutoff
        mask2 = f2 > cutoff
        mask3 = f3 > cutoff

        valid_i_indices = np.where(mask1)[0]
        valid_j_indices = np.where(mask2)[0]
        valid_k_indices = np.where(mask3)[0]

        if (
            len(valid_i_indices) == 0
            or len(valid_j_indices) == 0
            or len(valid_k_indices) == 0
        ):
            return

        b1_indices = band_indices[valid_i_indices]
        b2_indices = valid_j_indices
        b3_indices = valid_k_indices

        # Reshape eigenvectors for valid bands for einsum
        e1_valid = e1[:, b1_indices].reshape(num_atom, 3, -1)
        e2_valid = e2[:, b2_indices].reshape(num_atom, 3, -1)
        e3_valid = e3[:, b3_indices].reshape(num_atom, 3, -1)

        # Sum over cartesian coordinates for all atom and band triplets
        # i, j, k -> atom indices κ κ' κ''
        # l, m, n -> cartesian indices α β γ
        # I, J, K -> band indices
        # This computes the sum over l,m,n for each i,j,k,I,J,K combination
        # The output shape is (num_atom, num_atom, num_atom,
        #                       n_valid_i, n_valid_j, n_valid_k)
        
        # Create a grid of masses for division
        m = self._masses
        
        # Get frequency arrays for valid bands
        f1_valid = f1[b1_indices]
        f2_valid = f2[b2_indices]
        f3_valid = f3[b3_indices]
        
        # Combined einsum including frequency normalization
        fc3_normal_valid = np.einsum(
            "ilI,jmJ,knK,ijklmn,i,j,k,I,J,K -> IJK",
            e1_valid,
            e2_valid,
            e3_valid,
            self._fc3_reciprocal,
            1/np.sqrt(m),
            1/np.sqrt(m),
            1/np.sqrt(m),
            1/np.sqrt(f1_valid),
            1/np.sqrt(f2_valid),
            1/np.sqrt(f3_valid),
            optimize=True,
        )

        # Place the calculated values into the final output array
        ix_ = np.ix_(valid_i_indices, valid_j_indices, valid_k_indices)
        self._fc3_normal[ix_] = fc3_normal_valid 