"""Transform fc3 in real space to reciprocal space - Exact C reproduction."""

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


class RealToReciprocalFast:
    """Transform fc3 in real space to reciprocal space - Exact C reproduction."""

    def __init__(self, fc3, primitive, mesh, symprec=1e-5, make_r0_average=True):
        """Init method."""
        self._fc3 = fc3
        self._primitive = primitive
        self._mesh = mesh
        self._symprec = symprec
        self._make_r0_average = make_r0_average

        self._p2s_map = primitive.p2s_map
        self._s2p_map = primitive.s2p_map
        self._svecs, self._multi = self._primitive.get_smallest_vectors()
        
        # Get all_shortest from primitive if available
        if hasattr(primitive, 'all_shortest'):
            self._all_shortest = primitive.all_shortest
        else:
            # Create a default all_shortest array (all False)
            num_patom = len(self._primitive)
            num_satom = len(self._s2p_map)
            self._all_shortest = np.zeros((num_patom, num_satom, num_satom), dtype=bool)
        
        # Create nonzero_indices (all True for non-compact fc3)
        self._nonzero_indices = np.ones(self._fc3.shape[:3], dtype=bool)
        
        self._fc3_reciprocal = None

    def run(self, triplet):
        """Run at each triplet of q-vectors."""
        self._triplet = triplet
        q_vecs = triplet.astype('double') / self._mesh
        
        # Main C function equivalent
        self._r2r_real_to_reciprocal(q_vecs)

    def get_fc3_reciprocal(self):
        """Return fc3 in reciprocal space."""
        return self._fc3_reciprocal

    def _r2r_real_to_reciprocal(self, q_vecs):
        """Main function equivalent to r2r_real_to_reciprocal in C."""
        num_patom = len(self._primitive)
        num_satom = len(self._s2p_map)
        num_band = num_patom * 3

        # Initialize fc3_reciprocal with shape (num_band, num_band, num_band)
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._fc3_reciprocal = np.zeros((num_band, num_band, num_band), dtype=dtype)
        
        # Compute pre_phase_factors
        pre_phase_factors = self._get_pre_phase_factors_vectorized(q_vecs)
        
        print("debug pre_phase_factors: ", pre_phase_factors)
        
        # Compute phase_factors for all combinations
        phase_factor0 = self._get_phase_factors_vectorized(q_vecs[0])
        phase_factor1 = self._get_phase_factors_vectorized(q_vecs[1])
        phase_factor2 = self._get_phase_factors_vectorized(q_vecs[2])
        
        # Choose algorithm based on make_r0_average
        if self._make_r0_average:
            self._real_to_reciprocal_r0_average_optimized(
                pre_phase_factors, phase_factor0, phase_factor1, phase_factor2
            )
            # Divide by 3 as done in C code
            self._fc3_reciprocal /= 3.0
        else:
            self._real_to_reciprocal_legacy_optimized(
                pre_phase_factors, phase_factor1, phase_factor2
            )

    def _get_pre_phase_factors_vectorized(self, q_vecs):
        """Vectorized computation of pre-phase factors."""
        num_patom = len(self._primitive)
        
        # Get starting indices for all primitive atoms
        satom_indices = self._p2s_map  # Shape: (num_patom,)
        multi_start_indices = self._multi[satom_indices, 0, 1].astype(int)  # Shape: (num_patom,)
        
        # Get svecs for all primitive atoms
        svecs_for_patoms = self._svecs[multi_start_indices]  # Shape: (num_patom, 3)
        
        # Compute q_sum for all three directions
        q_sum = np.sum(q_vecs, axis=0)  # Shape: (3,)
        
        # Vectorized dot product
        pre_phase = np.dot(svecs_for_patoms, q_sum) * 2 * np.pi  # Shape: (num_patom,)
        
        return np.exp(1j * pre_phase)

    def _get_phase_factors_vectorized(self, q):
        """Vectorized computation of phase factors for all (patom, satom) pairs."""
        num_patom = len(self._primitive)
        num_satom = len(self._s2p_map)
        
        # Initialize result array
        phase_factors = np.zeros((num_patom, num_satom), dtype=complex)
        
        # Pre-compute 2π * q for efficiency
        q_2pi = 2 * np.pi * q
        
        for i in range(num_patom):
            for j in range(num_satom):
                multi_count = int(self._multi[j, i, 0])
                multi_start = int(self._multi[j, i, 1])
                
                # Vectorized computation for all vectors in this multiplet
                if multi_count > 0:
                    svecs_slice = self._svecs[multi_start:multi_start + multi_count]  # Shape: (multi_count, 3)
                    phases = np.dot(svecs_slice, q_2pi)  # Shape: (multi_count,)
                    
                    # Sum complex exponentials and average
                    sum_exp = np.sum(np.exp(1j * phases))
                    phase_factors[i, j] = sum_exp / multi_count
        
        return phase_factors

    def _real_to_reciprocal_legacy_optimized(self, pre_phase_factors, phase_factor1, phase_factor2):
        """Optimized legacy implementation using broadcasting and einsum."""
        num_patom = len(self._primitive)
        
        # Create index arrays without meshgrid - use broadcasting instead
        i_indices = np.repeat(np.arange(num_patom), num_patom * num_patom)
        j_indices = np.tile(np.repeat(np.arange(num_patom), num_patom), num_patom)
        k_indices = np.tile(np.arange(num_patom), num_patom * num_patom)
        
        # Compute all fc3_rec_elements at once
        fc3_rec_elements = self._real_to_reciprocal_elements_optimized(
            phase_factor1, phase_factor2, i_indices, j_indices, k_indices, leg_index=0
        )  # Shape: (num_patom^3, 3, 3, 3)
        
        # Apply pre-phase factors using broadcasting
        pre_factors = pre_phase_factors[i_indices]  # Shape: (num_patom^3,)
        fc3_rec_elements = fc3_rec_elements * pre_factors[:, None, None, None]
        
        # Reshape and assign to fc3_reciprocal using einsum
        fc3_rec_elements = fc3_rec_elements.reshape(num_patom, num_patom, num_patom, 3, 3, 3)
        self._fc3_reciprocal = np.einsum('ijklmn->ilmjnk', fc3_rec_elements)

    def _real_to_reciprocal_r0_average_optimized(self, pre_phase_factors, phase_factor0, phase_factor1, phase_factor2):
        """Optimized R0 average implementation using broadcasting and einsum."""
        num_patom = len(self._primitive)
        
        # Create index arrays without meshgrid - use broadcasting instead
        i_indices = np.repeat(np.arange(num_patom), num_patom * num_patom)
        j_indices = np.tile(np.repeat(np.arange(num_patom), num_patom), num_patom)
        k_indices = np.tile(np.arange(num_patom), num_patom * num_patom)
        
        # Compute all three contributions using vectorized operations
        contributions = []
        
        # First contribution: leg_index=1
        fc3_rec_elem1 = self._real_to_reciprocal_elements_optimized(
            phase_factor1, phase_factor2, i_indices, j_indices, k_indices, leg_index=1
        )
        pre_factors1 = pre_phase_factors[i_indices]
        contrib1 = fc3_rec_elem1 * pre_factors1[:, None, None, None]
        contrib1 = contrib1.reshape(num_patom, num_patom, num_patom, 3, 3, 3)
        contributions.append(contrib1)
        
        # Second contribution: leg_index=2, swap jm <-> il
        fc3_rec_elem2 = self._real_to_reciprocal_elements_optimized(
            phase_factor0, phase_factor2, j_indices, i_indices, k_indices, leg_index=2
        )
        pre_factors2 = pre_phase_factors[j_indices]
        contrib2 = fc3_rec_elem2 * pre_factors2[:, None, None, None]
        contrib2 = contrib2.reshape(num_patom, num_patom, num_patom, 3, 3, 3)
        # Swap indices: jilmnk -> ijlmnk, then swap l<->m: ijmnlk
        contrib2 = np.transpose(contrib2, (0, 1, 2, 4, 3, 5))
        contributions.append(contrib2)

        # Third contribution: leg_index=3, swap kn <-> il  
        fc3_rec_elem3 = self._real_to_reciprocal_elements_optimized(
            phase_factor1, phase_factor0, k_indices, j_indices, i_indices, leg_index=3
        )
        pre_factors3 = pre_phase_factors[k_indices]
        contrib3 = fc3_rec_elem3 * pre_factors3[:, None, None, None]
        contrib3 = contrib3.reshape(num_patom, num_patom, num_patom, 3, 3, 3)
        # Swap indices: kjilmn -> ijklmn, then swap l<->n: ijkmnl
        contrib3 = np.transpose(contrib3, (0, 1, 2, 5, 4, 3))
        contributions.append(contrib3)
        
        # Sum all contributions and reshape to final form using einsum
        total_contrib = np.sum(contributions, axis=0)
        self._fc3_reciprocal = np.einsum('ijklmn->iljmkn', total_contrib)

    def _real_to_reciprocal_elements_optimized(self, phase_factor1, phase_factor2, pi0_array, pi1_array, pi2_array, leg_index):
        """Highly optimized calculation using broadcasting and masks."""
        num_satom = len(self._s2p_map)
        num_combinations = len(pi0_array)
        
        # Pre-compute all necessary arrays
        i_satom_array = self._p2s_map[pi0_array]
        
        # Create full satom index arrays for broadcasting
        j_all = np.arange(num_satom)
        k_all = np.arange(num_satom)
        
        # Use broadcasting to create all combinations
        # Shape: (num_combinations, num_satom, num_satom)
        i_bc = i_satom_array[:, None, None]
        j_bc = j_all[None, :, None] 
        k_bc = k_all[None, None, :]
        
        # Create validity masks using broadcasting
        pi1_bc = pi1_array[:, None, None]
        pi2_bc = pi2_array[:, None, None]
        
        j_valid_mask = (self._s2p_map[j_bc] == self._p2s_map[pi1_bc])
        k_valid_mask = (self._s2p_map[k_bc] == self._p2s_map[pi2_bc])
        valid_mask = j_valid_mask & k_valid_mask
        
        # Apply additional masks for leg_index > 1
        if leg_index > 1:
            pi0_bc = pi0_array[:, None, None]
            all_shortest_mask = self._all_shortest[pi0_bc, j_bc, k_bc]
            valid_mask = valid_mask & (~all_shortest_mask)
        
        # Apply nonzero_indices mask
        flat_indices = i_bc * num_satom * num_satom + j_bc * num_satom + k_bc
        nonzero_mask = self._nonzero_indices.flat[flat_indices]
        valid_mask = valid_mask & nonzero_mask
        
        # Compute phase factors using broadcasting
        pi0_bc_2d = pi0_array[:, None, None]
        phase_factors_j = phase_factor1[pi0_bc_2d, j_bc]
        phase_factors_k = phase_factor2[pi0_bc_2d, k_bc]
        phase_factors = phase_factors_j * phase_factors_k
        
        # Get fc3 slices for all combinations
        fc3_slices = self._fc3[i_bc, j_bc, k_bc]  # Shape: (num_combinations, num_satom, num_satom, 3, 3, 3)
        
        # Apply leg_index == 1 weights using broadcasting
        if leg_index == 1:
            pi0_bc_weight = pi0_array[:, None, None]
            all_shortest_weights = np.where(
                self._all_shortest[pi0_bc_weight, j_bc, k_bc], 3.0, 1.0
            )
            fc3_slices = fc3_slices * all_shortest_weights[..., None, None, None]
        
        # Apply masks to zero out invalid entries
        valid_mask_expanded = valid_mask[..., None, None, None]
        phase_factors_masked = np.where(valid_mask, phase_factors, 0.0)
        fc3_slices_masked = np.where(valid_mask_expanded, fc3_slices, 0.0)
        
        # Compute weighted fc3 using broadcasting
        weighted_fc3 = fc3_slices_masked * phase_factors_masked[..., None, None, None]
        
        # Sum over j and k dimensions using einsum for speed
        result = np.einsum('ijklmn->ilmn', weighted_fc3)  # Shape: (num_combinations, 3, 3, 3)
        
        return result

    def _get_pre_phase_factor(self, i_patom, q_vecs):
        """Calculate pre-phase factor - equivalent to C function."""
        # From C code: svecs_adrs = atom_triplets->p2s_map[i_patom] * atom_triplets->multi_dims[1] + i_patom
        # Then uses: atom_triplets->multiplicity[svecs_adrs][1]
        # This is the same pattern as get_phase_factor: (satom_index, patom_index) -> multiplicity
        num_patom = len(self._primitive)
        satom_index = self._p2s_map[i_patom]  # Get supercell atom index for this primitive atom
        
        # Use the same indexing as get_phase_factor: _multi[satom_index, patom_index, :]
        multi_start_idx = int(self._multi[satom_index, 0, 1])  # Get start_index
        
        # print("DEBUG get_pre_phase: Entry for i_patom=%d" % i_patom)
        # print("DEBUG get_pre_phase: svecs_adrs=%d, p2s_map[%d]=%d" % (satom_index * num_patom, i_patom, satom_index))
        # print("DEBUG get_pre_phase: multiplicity[%d][1]=%d" % (satom_index * num_patom, multi_start_idx))
        
        pre_phase = 0.0
        for j in range(3):
            svec_component = self._svecs[multi_start_idx, j]
            q_sum = q_vecs[0, j] + q_vecs[1, j] + q_vecs[2, j]
            pre_phase += svec_component * q_sum
            # print("DEBUG get_pre_phase: j=%d, multiplicity[%d][1]=%d, svecs[%d][%d]=%f, q_sum=%f" % (j, satom_index * num_patom, multi_start_idx, multi_start_idx, j, svec_component, q_sum))
            # print("DEBUG get_pre_phase: j=%d completed, pre_phase=%f" % (j, pre_phase))
        
        pre_phase *= 2 * np.pi
        # print("DEBUG get_pre_phase: Exit for i_patom=%d" % i_patom)
        return np.exp(1j * pre_phase)

    def _get_phase_factor(self, q, satom_index, patom_index):
        """Calculate phase factor - equivalent to C function."""
        # In C code: get_phase_factor(q, svecs, multiplicity[j * multi_dims[1] + i])
        # where j=satom_index, i=patom_index, multi_dims[1]=num_patom
        # In our Python structure, this corresponds to _multi[satom_index, patom_index, :]
        
        multi_count = int(self._multi[satom_index, patom_index, 0])
        multi_start = int(self._multi[satom_index, patom_index, 1])
                
        sum_real = 0.0
        sum_imag = 0.0
        
        for i in range(multi_count):
            phase = 0.0
            for j in range(3):
                phase += q[j] * self._svecs[multi_start + i, j]
            phase *= 2 * np.pi
            sum_real += np.cos(phase)
            sum_imag += np.sin(phase)
        
        sum_real /= multi_count
        sum_imag /= multi_count
                
        return sum_real + 1j * sum_imag 