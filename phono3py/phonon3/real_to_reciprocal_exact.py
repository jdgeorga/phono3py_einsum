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


class RealToReciprocalExact:
    """Transform fc3 in real space to reciprocal space - Exact C reproduction."""

    def __init__(self, fc3, primitive, mesh, symprec=1e-5, make_r0_average=True, all_shortest=None):
        """Init method."""
        self._fc3 = fc3
        self._primitive = primitive
        self._mesh = mesh
        self._symprec = symprec
        self._make_r0_average = make_r0_average

        self._p2s_map = primitive.p2s_map
        self._s2p_map = primitive.s2p_map
        self._svecs, self._multi = self._primitive.get_smallest_vectors()
        
        # Get all_shortest from parameter, primitive, or create default
        if all_shortest is not None:
            self._all_shortest = all_shortest
        elif hasattr(primitive, 'all_shortest'):
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
        pre_phase_factors = np.zeros(num_patom, dtype=complex)
        for i in range(num_patom):
            pre_phase_factors[i] = self._get_pre_phase_factor(i, q_vecs)
            
        # print("debug pre_phase_factors: ", pre_phase_factors)
        # Compute phase_factors for all combinations
        phase_factor0 = np.zeros((num_patom, num_satom), dtype=complex)
        phase_factor1 = np.zeros((num_patom, num_satom), dtype=complex)
        phase_factor2 = np.zeros((num_patom, num_satom), dtype=complex)
        
        for i in range(num_patom):
            for j in range(num_satom):
                # In C code: get_phase_factor(q_vecs[1], svecs, multiplicity[j * multi_dims[1] + i])
                # The multiplicity depends on both j (satom index) and i (patom index)
                phase_factor0[i, j] = self._get_phase_factor(q_vecs[0], j, i)
                phase_factor1[i, j] = self._get_phase_factor(q_vecs[1], j, i)
                phase_factor2[i, j] = self._get_phase_factor(q_vecs[2], j, i)
        
        # for i in range(num_patom):
        #     for j in range(num_satom):
        #         print("DEBUG r2r: phase_factor0[%d] = %f + %fi" % (i * num_satom + j, phase_factor0[i,j].real, phase_factor0[i,j].imag))
                
        # for i in range(num_patom):
        #     for j in range(num_satom):       
        #         print("DEBUG r2r: phase_factor1[%d] = %f + %fi" % (i * num_satom + j, phase_factor1[i,j].real, phase_factor1[i,j].imag))
                
        # for i in range(num_patom):
        #     for j in range(num_satom):       
        #         print("DEBUG r2r: phase_factor2[%d] = %f + %fi" % (i * num_satom + j, phase_factor2[i,j].real, phase_factor2[i,j].imag))
        
        # Choose algorithm based on make_r0_average
        if self._make_r0_average:
            self._real_to_reciprocal_r0_average(
                pre_phase_factors, phase_factor0, phase_factor1, phase_factor2
            )
            # Divide by 3 as done in C code
            self._fc3_reciprocal /= 3.0
        else:
            self._real_to_reciprocal_legacy(
                pre_phase_factors, phase_factor1, phase_factor2
            )

    def _real_to_reciprocal_legacy(self, pre_phase_factors, phase_factor1, phase_factor2):
        """Legacy implementation equivalent to C code."""
        num_patom = len(self._primitive)
        num_satom = len(self._s2p_map)
        
        for ijk in range(num_patom * num_patom * num_patom):
            i = ijk // (num_patom * num_patom)
            j = (ijk - (i * num_patom * num_patom)) // num_patom
            k = ijk % num_patom
            
            fc3_rec_elem = self._real_to_reciprocal_elements(
                phase_factor1[i], phase_factor2[i], i, j, k, leg_index=0
            )
            
            for l in range(3):
                for m in range(3):
                    for n in range(3):
                        # C indexing: fc3_reciprocal[(i * 3 + l) * num_band * num_band + (j * 3 + m) * num_band + k * 3 + n]
                        self._fc3_reciprocal[i * 3 + l, j * 3 + m, k * 3 + n] = (
                            fc3_rec_elem[l, m, n] * pre_phase_factors[i]
                        )

    def _real_to_reciprocal_r0_average(self, pre_phase_factors, phase_factor0, phase_factor1, phase_factor2):
        """R0 average implementation equivalent to C code."""
        num_patom = len(self._primitive)
        num_satom = len(self._s2p_map)
        
        fc3_rec_elem1 = np.zeros((num_patom*3, num_patom*3, num_patom*3), dtype=complex)
        fc3_rec_elem2 = np.zeros((num_patom*3, num_patom*3, num_patom*3), dtype=complex)
        fc3_rec_elem3 = np.zeros((num_patom*3, num_patom*3, num_patom*3), dtype=complex)
        # print("DEBUG: self._all_shortest.sum(): ", self._all_shortest.sum())
        # print("DEBUG: self._nonzero_indices.sum(): ", self._nonzero_indices.sum())
        for ijk in range(num_patom * num_patom * num_patom):
            i = ijk // (num_patom * num_patom)
            j = (ijk - (i * num_patom * num_patom)) // num_patom
            k = ijk % num_patom
            # print("DEBUG: ijk,i,j,k: ", ijk,i,j,k)
            # print("DEBUG: self._fc3_reciprocal.shape: ", self._fc3_reciprocal.shape)
            # print("DEBUG: j: ", j)
            # print("DEBUG: k: ", k)
            
            # First contribution: leg_index=1
            fc3_rec_elem = self._real_to_reciprocal_elements(
                phase_factor1[i], phase_factor2[i], i, j, k, leg_index=1
            )
            
            for l in range(3):
                for m in range(3):
                    for n in range(3):
                        fc3_rec = fc3_rec_elem[l, m, n] * pre_phase_factors[i]
                        self._fc3_reciprocal[i * 3 + l, j * 3 + m, k * 3 + n] += fc3_rec
                        fc3_rec_elem1[i * 3 + l, j * 3 + m, k * 3 + n] += fc3_rec

            
            # Second contribution: leg_index=2, swap jm <-> il
            fc3_rec_elem = self._real_to_reciprocal_elements(
                phase_factor0[j], phase_factor2[j], j, i, k, leg_index=2
            )
            
            for l in range(3):
                for m in range(3):
                    for n in range(3):
                        fc3_rec = fc3_rec_elem[m, l, n] * pre_phase_factors[j]
                        self._fc3_reciprocal[i * 3 + l, j * 3 + m, k * 3 + n] += fc3_rec
                        fc3_rec_elem2[i * 3 + l, j * 3 + m, k * 3 + n] += fc3_rec
            
            # Third contribution: leg_index=3, swap kn <-> il
            fc3_rec_elem = self._real_to_reciprocal_elements(
                phase_factor1[k], phase_factor0[k], k, j, i, leg_index=3
            )
            for l in range(3):
                for m in range(3):
                    for n in range(3):
                        fc3_rec = fc3_rec_elem[n, m, l] * pre_phase_factors[k]
                        self._fc3_reciprocal[i * 3 + l, j * 3 + m, k * 3 + n] += fc3_rec
                        fc3_rec_elem3[i * 3 + l, j * 3 + m, k * 3 + n] += fc3_rec
        # quit()  
        # print("DEBUG r2r: fc3_rec_elem1: leg_index = 1")
        # print("DEBUG shape: ", fc3_rec_elem1.shape)
        # print(fc3_rec_elem1)
        # np.save("fc3_rec_elem1.npy", fc3_rec_elem1)
        # print("DEBUG r2r: fc3_rec_elem2: leg_index = 2")
        # print("DEBUG shape: ", fc3_rec_elem2.shape)
        # print(fc3_rec_elem2)
        # np.save("fc3_rec_elem2.npy", fc3_rec_elem2)
        # print("DEBUG r2r: fc3_rec_elem3: leg_index = 3")
        # print("DEBUG shape: ", fc3_rec_elem3.shape)
        # print(fc3_rec_elem3)
        # np.save("fc3_rec_elem3.npy", fc3_rec_elem3)

        # print("DEBUG r2r: fc3_recirpcoal: leg_index = 2")
        # print("DEBUG shape: ", self._fc3_reciprocal.shape)
        # print(self._fc3_reciprocal)

    def _real_to_reciprocal_elements(self, phase_factor1, phase_factor2, pi0, pi1, pi2, leg_index):
        """Calculate reciprocal space elements - equivalent to C function."""
        num_satom = len(self._s2p_map)
        fc3_rec_real = np.zeros((27,), dtype=float)
        fc3_rec_imag = np.zeros((27,), dtype=float)

        # Determine i based on compact fc3 flag (assuming non-compact)
        i = self._p2s_map[pi0]
        
        for j in range(num_satom):
            if self._s2p_map[j] != self._p2s_map[pi1]:
                continue
            
            for k in range(num_satom):
                if self._s2p_map[k] != self._p2s_map[pi2]:
                    continue
                
                # Check all_shortest condition for leg_index > 1
                if leg_index > 1:
                    if self._all_shortest[pi0, j, k]:
                        continue
                
                # Check nonzero_indices
                adrs_shift_atoms = i * num_satom * num_satom + j * num_satom + k
                if not self._nonzero_indices.flat[adrs_shift_atoms]:
                    continue
                
                # Calculate phase factor
                phase_factor = phase_factor1[j] * phase_factor2[k]
                
                # Add fc3 contribution
                # print(i,j,k)
                fc3_slice = self._fc3[i, j, k].flatten()

                
                # Special handling for leg_index == 1 and all_shortest
                if (leg_index == 1 and self._all_shortest[pi0, j, k]):
                    weight = 3.0
                else:
                    weight = 1.0
                
                for l in range(27):
                    fc3_rec_real[l] += phase_factor.real * fc3_slice[l] * weight
                    fc3_rec_imag[l] += phase_factor.imag * fc3_slice[l] * weight
        
        # Convert back to complex and reshape
        fc3_rec_elem = fc3_rec_real + 1j * fc3_rec_imag
        return fc3_rec_elem.reshape((3, 3, 3))

    def _get_pre_phase_factor(self, i_patom, q_vecs):
        """Calculate pre-phase factor - equivalent to C function."""
        # From C code: svecs_adrs = atom_triplets->p2s_map[i_patom] * atom_triplets->multi_dims[1] + i_patom
        # Then uses: atom_triplets->multiplicity[svecs_adrs][1]
        # This is the same pattern as get_phase_factor: (satom_index, patom_index) -> multiplicity
        num_patom = len(self._primitive)
        satom_index = self._p2s_map[i_patom]  # Get supercell atom index for this primitive atom
        
        # Use the same indexing as get_phase_factor: _multi[satom_index, patom_index, :]
        multi_start_idx = int(self._multi[satom_index, 0, 1])  # Get start_index
        # print("DEBUG get_pre_phase: satom_index=%d, multi_start_idx=%d" % (satom_index, multi_start_idx))
        
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