"""Transform fc3 in real space to reciprocal space - Version 3."""

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

import numpy as np


class RealToReciprocalV3:
    """Transform fc3 in real space to reciprocal space with legacy and r0_average options."""

    def __init__(self, fc3, primitive, mesh, symprec=1e-5, make_r0_average=True):
        """Init method."""
        self._fc3 = fc3
        self._primitive = primitive
        self._mesh = mesh
        self._symprec = symprec
        self._make_r0_average = make_r0_average

        self._p2s_map = primitive.p2s_map
        self._s2p_map = primitive.s2p_map
        # Reduce supercell atom index to primitive index
        self._svecs, self._multi = self._primitive.get_smallest_vectors()
        self._fc3_reciprocal = None

    def run(self, triplet):
        """Run at each triplet of q-vectors."""
        self._triplet = triplet
        num_patom = len(self._primitive)
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._fc3_reciprocal = np.zeros(
            (num_patom, num_patom, num_patom, 3, 3, 3), dtype=dtype
        )

        if self._make_r0_average:
            self._real_to_reciprocal_r0_average()
        else:
            self._real_to_reciprocal_legacy()

    def get_fc3_reciprocal(self):
        """Return fc3 in reciprocal space."""
        return self._fc3_reciprocal

    def _real_to_reciprocal_legacy(self):
        """Legacy implementation similar to the original Python code."""
        if self._fc3_reciprocal is None:
            raise RuntimeError("fc3_reciprocal not initialized. Call run() first.")
            
        num_patom = len(self._primitive)
        sum_triplets = np.where(
            np.all(self._triplet != 0, axis=0), self._triplet.sum(axis=0), 0
        )
        sum_q = sum_triplets.astype("double") / self._mesh
        
        for i in range(num_patom):
            for j in range(num_patom):
                for k in range(num_patom):
                    self._fc3_reciprocal[i, j, k] = self._real_to_reciprocal_elements(
                        (i, j, k), phase_refs=(1, 2), ref_patom=i
                    )
            prephase = self._get_prephase(sum_q, i)
            print("DEBUG CAPTURE: prephase", prephase,)
            self._fc3_reciprocal[i] *= prephase

    def _real_to_reciprocal_r0_average(self):
        """R0 average implementation with three different lattice reference points."""
        if self._fc3_reciprocal is None:
            raise RuntimeError("fc3_reciprocal not initialized. Call run() first.")
            
        num_patom = len(self._primitive)
        
        # Compute all pre-phase factors
        sum_triplets = np.where(
            np.all(self._triplet != 0, axis=0), self._triplet.sum(axis=0), 0
        )
        sum_q = sum_triplets.astype("double") / self._mesh
        
        pre_phase_factors = np.zeros(num_patom, dtype=complex)
        for i in range(num_patom):
            pre_phase_factors[i] = self._get_prephase(sum_q, i)
        
        for i in range(num_patom):
            for j in range(num_patom):
                for k in range(num_patom):
                    # First contribution: normal order (i, j, k) with phase factors 1, 2
                    fc3_elem_1 = self._real_to_reciprocal_elements(
                        (i, j, k), phase_refs=(1, 2), ref_patom=i
                    )
                    contribution_1 = fc3_elem_1 * pre_phase_factors[i]
                    
                    # Second contribution: reorder (j, i, k) with phase factors 0, 2
                    # Result gets transposed: jm <-> il
                    fc3_elem_2 = self._real_to_reciprocal_elements(
                        (j, i, k), phase_refs=(0, 2), ref_patom=j
                    )
                    # Transpose axes: (l, m, n) -> (m, l, n) to swap jm <-> il
                    fc3_elem_2_swapped = np.transpose(fc3_elem_2, (1, 0, 2))
                    contribution_2 = fc3_elem_2_swapped * pre_phase_factors[j]
                    
                    # Third contribution: reorder (k, j, i) with phase factors 1, 0
                    # Result gets transposed: kn <-> il
                    fc3_elem_3 = self._real_to_reciprocal_elements(
                        (k, j, i), phase_refs=(1, 0), ref_patom=k
                    )
                    # Transpose axes: (l, m, n) -> (n, m, l) to swap kn <-> il
                    fc3_elem_3_swapped = np.transpose(fc3_elem_3, (2, 1, 0))
                    contribution_3 = fc3_elem_3_swapped * pre_phase_factors[k]
                    
                    # Sum all three contributions
                    self._fc3_reciprocal[i, j, k] = (
                        contribution_1 + contribution_2 + contribution_3
                    )
        
        # Divide by 3 as done in the C code
        self._fc3_reciprocal /= 3.0

    def _real_to_reciprocal_elements(self, patom_indices, phase_refs=(1, 2), ref_patom=0):
        """Calculate reciprocal space elements for given primitive atom indices."""
        num_satom = len(self._s2p_map)
        pi = patom_indices
        i = self._p2s_map[pi[0]]
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        fc3_reciprocal = np.zeros((3, 3, 3), dtype=dtype)
        
        for j in range(num_satom):
            if self._s2p_map[j] != self._p2s_map[pi[1]]:
                continue
            for k in range(num_satom):
                if self._s2p_map[k] != self._p2s_map[pi[2]]:
                    continue
                
                # Get phase using the specified phase references
                phase = self._get_phase_with_refs((j, k), ref_patom, phase_refs)
                fc3_reciprocal += self._fc3[i, j, k] * phase
                    
        return fc3_reciprocal

    def _get_prephase(self, sum_q, patom_index):
        """Calculate pre-phase factor."""
        r = self._primitive.scaled_positions[patom_index]
        return np.exp(2j * np.pi * np.dot(sum_q, r))

    def _get_phase_with_refs(self, satom_indices, patom0_index, phase_refs):
        """Calculate phase factor using specified phase reference indices."""
        si = satom_indices
        p0 = patom0_index
        phase = 1 + 0j
        
        # Use the specified phase reference indices
        q_indices = list(phase_refs)
        
        for i, q_idx in enumerate(q_indices):
            svecs_adrs = self._multi[si[i], p0, 1]
            multi = self._multi[si[i], p0, 0]
            vs = self._svecs[svecs_adrs : (svecs_adrs + multi)]
            phase *= (
                np.exp(
                    2j
                    * np.pi
                    * np.dot(vs, self._triplet[q_idx].astype("double") / self._mesh)
                ).sum()
                / multi
            )
        return phase

    def _get_phase(self, satom_indices, patom0_index, ref_index=1):
        """Calculate phase factor for different reference points."""
        si = satom_indices
        p0 = patom0_index
        phase = 1 + 0j
        
        # Choose which q-vectors to use based on reference index
        if ref_index == 1:
            # Original: use q1 and q2
            q_indices = [1, 2]
        elif ref_index == 2:
            # Second: use q0 and q2
            q_indices = [0, 2]
        elif ref_index == 3:
            # Third: use q1 and q0
            q_indices = [1, 0]
        else:
            # Default: use q1 and q2
            q_indices = [1, 2]
        
        for i, q_idx in enumerate(q_indices):
            svecs_adrs = self._multi[si[i], p0, 1]
            multi = self._multi[si[i], p0, 0]
            vs = self._svecs[svecs_adrs : (svecs_adrs + multi)]
            phase *= (
                np.exp(
                    2j
                    * np.pi
                    * np.dot(vs, self._triplet[q_idx].astype("double") / self._mesh)
                ).sum()
                / multi
            )
        return phase 