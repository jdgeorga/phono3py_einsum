# phonopy_codes/phono3py_einsum/phono3py/phonon3/real_to_reciprocal_gpu.py
"""CuPy-based real-to-reciprocal fc3 transform - BATCHED GPU implementation.

Key optimization: Process ALL triplets in a single GPU call to amortize overhead.
The naive approach of processing one triplet at a time is slower than C due to
GPU kernel launch overhead and CPU-GPU memory transfers.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

PI2 = 2.0 * np.pi


class RealToReciprocalExactGPU:
    """Batched CuPy implementation of RealToReciprocal.

    This class can process triplets one at a time (for API compatibility) or
    all at once via run_batch() for maximum performance.
    
    For best GPU utilization, use run_batch() to process all triplets together.
    """

    def __init__(
        self,
        fc3,
        primitive,
        mesh,
        symprec=1e-5,
        make_r0_average=False,
        all_shortest=None,
        phase_vectorized=False,
    ):
        if cp is None:
            raise ImportError("CuPy is required for RealToReciprocalExactGPU")
        
        self._fc3 = fc3
        self._primitive = primitive
        self._mesh = np.array(mesh, dtype=np.int64)
        self._symprec = symprec
        self._make_r0_average = make_r0_average
        self._phase_vectorized = bool(phase_vectorized)

        self._p2s_map = np.array(primitive.p2s_map, dtype=np.int64)
        self._s2p_map = np.array(primitive.s2p_map, dtype=np.int64)
        
        svecs, multi = primitive.get_smallest_vectors()
        self._svecs = np.array(svecs, dtype=np.float64)
        self._multi = np.array(multi, dtype=np.int32)

        self._num_patom = len(self._p2s_map)
        self._num_satom = len(self._s2p_map)
        self._num_band = self._num_patom * 3

        self._fc3_reciprocal = None
        self._fc3_reciprocal_gpu = None  # Keep on GPU to avoid transfers
        
        if all_shortest is not None:
            self._all_shortest = np.array(all_shortest, dtype=np.int8)
        else:
            self._all_shortest = self._compute_all_shortest()
        
        self._init_gpu_data()

    def _compute_all_shortest(self):
        """Compute all_shortest array matching C implementation."""
        num_patom = self._num_patom
        num_satom = self._num_satom
        all_shortest = np.zeros((num_patom, num_satom, num_satom), dtype=np.int8)
        
        svecs = self._svecs
        multi = self._multi
        perms = self._primitive.atomic_permutations
        s2pp_map = [self._primitive.p2p_map[i] for i in self._s2p_map]
        lattice = self._primitive.cell
        
        for i_patom in range(num_patom):
            for j_atom in range(num_satom):
                if multi[j_atom, i_patom, 0] > 1:
                    continue
                j_patom = s2pp_map[j_atom]
                i_perm = np.where(perms[:, j_atom] == self._p2s_map[j_patom])[0]
                if len(i_perm) != 1:
                    continue
                for k_atom in range(num_satom):
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
                        all_shortest[i_patom, j_atom, k_atom] = 1
        
        return all_shortest

    def _init_gpu_data(self):
        """Transfer constant data to GPU once."""
        P, S = self._num_patom, self._num_satom
        
        self._svecs_gpu = cp.asarray(self._svecs, dtype=cp.float64)
        self._multi_gpu = cp.asarray(self._multi, dtype=cp.int32)
        self._all_shortest_gpu = cp.asarray(self._all_shortest, dtype=cp.float64)
        self._p2s_gpu = cp.asarray(self._p2s_map, dtype=cp.int64)
        self._s2p_gpu = cp.asarray(self._s2p_map, dtype=cp.int64)
        self._scaled_positions_gpu = cp.asarray(
            self._primitive.scaled_positions, dtype=cp.float64
        )
        
        # FC3 on GPU
        fc3 = self._fc3
        if fc3.shape[0] == S:
            self._fc3_gpu = cp.asarray(fc3.reshape(S, S, S, 27), dtype=cp.complex128)
            self._is_compact = False
        elif fc3.shape[0] == P:
            self._fc3_gpu = cp.asarray(fc3.reshape(P, S, S, 27), dtype=cp.complex128)
            self._is_compact = True
        else:
            raise ValueError(f"Unknown fc3 shape: {fc3.shape}")
        
        # Selection mask M: M[p, s] = 1 iff s2p_map[s] == p2s_map[p]
        self._M = cp.asarray(
            (self._s2p_map[None, :] == self._p2s_map[:, None]).astype(np.float64)
        )
        
        # Precompute weight tensors (constant across triplets)
        self._weight_leg1 = cp.where(self._all_shortest_gpu > 0, 3.0, 1.0)
        self._weight_leg23 = cp.where(self._all_shortest_gpu > 0, 0.0, 1.0)
        
        # Get fc3 base indexed by primitive atoms
        if self._is_compact:
            self._fc3_base = self._fc3_gpu  # (P, S, S, 27)
        else:
            self._fc3_base = self._fc3_gpu[self._p2s_gpu]  # (P, S, S, 27)

    def run(self, triplet):
        """Compute fc3(q) for a single triplet. Keeps result on GPU."""
        if self._make_r0_average:
            self._run_r0_average_single(triplet)
        else:
            self._run_legacy_single(triplet)

    def get_fc3_reciprocal(self):
        """Return fc3 in reciprocal space (NumPy array on CPU)."""
        if self._fc3_reciprocal_gpu is not None:
            return cp.asnumpy(self._fc3_reciprocal_gpu)
        return self._fc3_reciprocal

    def get_fc3_reciprocal_gpu(self):
        """Return fc3 in reciprocal space (CuPy array on GPU)."""
        return self._fc3_reciprocal_gpu

    def _compute_phase_single(self, q_vecs):
        """Compute phase factors for a single triplet - optimized."""
        P, S = self._num_patom, self._num_satom
        q_gpu = cp.asarray(q_vecs, dtype=cp.float64)  # (3, 3)
        
        # Phase computation using direct formula (avoiding segment reduction overhead)
        phase_table = cp.zeros((P, S, 3), dtype=cp.complex128)
        
        multi_count = self._multi_gpu[:, :, 0]  # (S, P)
        multi_start = self._multi_gpu[:, :, 1]  # (S, P)
        
        # For each (s, p) pair, compute average phase
        for p in range(P):
            for s in range(S):
                count = int(multi_count[s, p])
                start = int(multi_start[s, p])
                if count == 0:
                    continue
                vecs = self._svecs_gpu[start:start + count]  # (count, 3)
                for qi in range(3):
                    phases = PI2 * cp.dot(vecs, q_gpu[qi])
                    phase_table[p, s, qi] = cp.exp(1j * phases).mean()
        
        return phase_table

    def _run_legacy_single(self, triplet):
        """Single-triplet legacy algorithm."""
        P, S = self._num_patom, self._num_satom
        q_vecs = triplet.astype(np.float64) / self._mesh
        
        phase_table = self._compute_phase_single(q_vecs)
        phase1 = phase_table[:, :, 1]
        phase2 = phase_table[:, :, 2]
        
        q_sum = q_vecs.sum(axis=0)
        prephase = cp.exp(1j * PI2 * cp.dot(self._scaled_positions_gpu, cp.asarray(q_sum)))
        
        phase_weight = phase1[:, :, None] * phase2[:, None, :]
        weighted_fc3 = self._fc3_base * phase_weight[:, :, :, None]
        
        temp = cp.einsum('ijkc,lk->ijlc', weighted_fc3, self._M)
        fc3_rec = cp.einsum('ijkc,lj->ilkc', temp, self._M)
        fc3_rec = fc3_rec * prephase[:, None, None, None]
        
        self._fc3_reciprocal_gpu = fc3_rec.reshape(P, P, P, 3, 3, 3)

    def _run_r0_average_single(self, triplet):
        """Single-triplet R0-average algorithm - optimized."""
        P, S = self._num_patom, self._num_satom
        q_vecs = triplet.astype(np.float64) / self._mesh
        
        phase_table = self._compute_phase_single(q_vecs)
        phase0 = phase_table[:, :, 0]
        phase1 = phase_table[:, :, 1]
        phase2 = phase_table[:, :, 2]
        
        q_sum = q_vecs.sum(axis=0)
        prephase = cp.exp(1j * PI2 * cp.dot(self._scaled_positions_gpu, cp.asarray(q_sum)))
        
        M = self._M
        fc3_rec = cp.zeros((P, P, P, 27), dtype=cp.complex128)
        
        # LEG 1
        phase_jk = phase1[:, :, None] * phase2[:, None, :]
        weighted = self._fc3_base * (phase_jk * self._weight_leg1)[:, :, :, None]
        temp = cp.einsum('ijkc,lk->ijlc', weighted, M)
        leg1 = cp.einsum('ijkc,lj->ilkc', temp, M) * prephase[:, None, None, None]
        fc3_rec += leg1
        
        # LEG 2
        phase_ik = phase0[:, :, None] * phase2[:, None, :]
        weighted = self._fc3_base * (phase_ik * self._weight_leg23)[:, :, :, None]
        temp = cp.einsum('jikc,li->jlkc', weighted, M)
        leg2 = cp.einsum('jikc,lk->jilc', temp, M) * prephase[:, None, None, None]
        leg2 = leg2.transpose(1, 0, 2, 3).reshape(P, P, P, 3, 3, 3)
        leg2 = leg2.transpose(0, 1, 2, 4, 3, 5).reshape(P, P, P, 27)
        fc3_rec += leg2
        
        # LEG 3
        phase_ji = phase1[:, :, None] * phase0[:, None, :]
        weighted = self._fc3_base * (phase_ji * self._weight_leg23)[:, :, :, None]
        temp = cp.einsum('kjic,lj->klic', weighted, M)
        leg3 = cp.einsum('kjic,li->kjlc', temp, M) * prephase[:, None, None, None]
        leg3 = leg3.transpose(2, 1, 0, 3).reshape(P, P, P, 3, 3, 3)
        leg3 = leg3.transpose(0, 1, 2, 5, 4, 3).reshape(P, P, P, 27)
        fc3_rec += leg3
        
        fc3_rec /= 3.0
        self._fc3_reciprocal_gpu = fc3_rec.reshape(P, P, P, 3, 3, 3)

    # =========================================================================
    # BATCHED API - Process ALL triplets at once for maximum GPU efficiency
    # =========================================================================
    
    def run_batch(self, triplet_addresses, return_gpu=True):
        """Process ALL triplets at once on GPU.
        
        This is MUCH faster than calling run() in a loop because:
        1. Single GPU memory allocation for all results
        2. Batched phase factor computation
        3. No CPU-GPU transfers between triplets
        
        Parameters
        ----------
        triplet_addresses : ndarray
            Shape (num_triplets, 3, 3) - q-vector addresses for all triplets
        return_gpu : bool
            If True, return CuPy array (stays on GPU). If False, return NumPy.
            
        Returns
        -------
        fc3_reciprocal : ndarray
            Shape (num_triplets, P, P, P, 3, 3, 3)
        """
        if self._make_r0_average:
            return self._run_r0_average_batch(triplet_addresses, return_gpu)
        else:
            return self._run_legacy_batch(triplet_addresses, return_gpu)

    def _compute_phase_batch(self, all_q_vecs):
        """Compute phase factors for ALL triplets at once."""
        if self._phase_vectorized:
            return self._compute_phase_batch_vectorized(all_q_vecs)
        return self._compute_phase_batch_loop(all_q_vecs)

    def _compute_phase_batch_loop(self, all_q_vecs):
        """Compute phase factors for ALL triplets with the original loop path.
        
        Parameters
        ----------
        all_q_vecs : ndarray
            Shape (num_triplets, 3, 3) - fractional q-vectors
            
        Returns
        -------
        phase_tables : cp.ndarray
            Shape (num_triplets, P, S, 3) - phases for q0, q1, q2
        """
        T = len(all_q_vecs)
        P, S = self._num_patom, self._num_satom
        
        all_q_gpu = cp.asarray(all_q_vecs, dtype=cp.float64)  # (T, 3, 3)
        
        # For each (s, p) pair, we need to average exp(2πi q·R) over replica vectors
        multi_count = cp.asnumpy(self._multi_gpu[:, :, 0])  # (S, P)
        multi_start = cp.asnumpy(self._multi_gpu[:, :, 1])  # (S, P)
        
        phase_tables = cp.zeros((T, P, S, 3), dtype=cp.complex128)
        
        # Batch over (p, s) pairs that have vectors
        for p in range(P):
            for s in range(S):
                count = int(multi_count[s, p])
                start = int(multi_start[s, p])
                if count == 0:
                    continue
                    
                vecs = self._svecs_gpu[start:start + count]  # (count, 3)
                
                # Compute phases for all triplets and all q's at once
                # vecs: (count, 3), all_q_gpu: (T, 3, 3)
                # We want: exp(2πi vecs @ q.T) for each triplet and each q
                for qi in range(3):
                    # q vectors for this component: (T, 3)
                    q_comp = all_q_gpu[:, qi, :]  # (T, 3)
                    # phases: vecs @ q_comp.T -> (count, T)
                    dot_prod = cp.dot(vecs, q_comp.T)  # (count, T)
                    exp_phase = cp.exp(1j * PI2 * dot_prod)  # (count, T)
                    phase_tables[:, p, s, qi] = exp_phase.mean(axis=0)  # (T,)
        
        return phase_tables

    def _init_phase_vectorized_cache(self):
        """Prepare padded replica vectors used by the vectorized phase path."""
        P, S = self._num_patom, self._num_satom

        multi_count = cp.asnumpy(self._multi_gpu[:, :, 0])  # (S, P)
        multi_start = cp.asnumpy(self._multi_gpu[:, :, 1])  # (S, P)
        max_count = int(multi_count.max())
        vecs_padded = np.zeros((P, S, max_count, 3), dtype=np.float64)
        norm = np.zeros((P, S), dtype=np.float64)
        pad_count = np.full((P, S), float(max_count), dtype=np.float64)
        svecs = cp.asnumpy(self._svecs_gpu)

        for p in range(P):
            for s in range(S):
                count = int(multi_count[s, p])
                start = int(multi_start[s, p])
                if count > 0:
                    vecs_padded[p, s, :count] = svecs[start:start + count]
                    norm[p, s] = 1.0 / count
                    pad_count[p, s] = float(max_count - count)

        self._phase_vec_cache = cp.asarray(vecs_padded)
        self._phase_norm_cache = cp.asarray(norm)
        self._phase_pad_cache = cp.asarray(pad_count)
        self._phase_max_count = max_count

    def _compute_phase_batch_vectorized(self, all_q_vecs):
        """Compute all q-leg phase factors through one dense CuPy matmul."""
        T = len(all_q_vecs)
        P, S = self._num_patom, self._num_satom

        if not hasattr(self, "_phase_vec_cache"):
            self._init_phase_vectorized_cache()

        max_count = self._phase_max_count
        vecs = self._phase_vec_cache
        norm = self._phase_norm_cache
        pad_count = self._phase_pad_cache
        q_all = cp.asarray(all_q_vecs, dtype=cp.float64)

        dots = (
            vecs.reshape(P * S * max_count, 3) @ q_all.reshape(T * 3, 3).T
        ).reshape(P, S, max_count, T, 3)

        phase_sum = cp.exp(1j * PI2 * dots).sum(axis=2)
        phase_sum -= pad_count[:, :, None, None]
        return (phase_sum * norm[:, :, None, None]).transpose(2, 0, 1, 3)

    def _run_r0_average_batch(self, triplet_addresses, return_gpu=True):
        """Batched R0-average for all triplets."""
        T = len(triplet_addresses)
        P, S = self._num_patom, self._num_satom
        
        # Convert addresses to fractional q-vectors
        all_q_vecs = triplet_addresses.astype(np.float64) / self._mesh[None, None, :]
        
        # Compute all phase factors at once: (T, P, S, 3)
        phase_tables = self._compute_phase_batch(all_q_vecs)
        phase0 = phase_tables[:, :, :, 0]  # (T, P, S)
        phase1 = phase_tables[:, :, :, 1]  # (T, P, S)
        phase2 = phase_tables[:, :, :, 2]  # (T, P, S)
        
        # Prephase for all triplets: (T, P)
        q_sums = all_q_vecs.sum(axis=1)  # (T, 3)
        q_sums_gpu = cp.asarray(q_sums, dtype=cp.float64)
        prephase = cp.exp(1j * PI2 * cp.dot(q_sums_gpu, self._scaled_positions_gpu.T))  # (T, P)
        
        M = self._M  # (P, S)
        
        # Output: (T, P, P, P, 27)
        fc3_rec = cp.zeros((T, P, P, P, 27), dtype=cp.complex128)
                    
        # Process each triplet (still looping, but with minimal overhead)
        # TODO: Further optimize with custom CUDA kernel for true batching
        for t in range(T):
            # LEG 1
            phase_jk = phase1[t, :, :, None] * phase2[t, :, None, :]  # (P, S, S)
            weighted = self._fc3_base * (phase_jk * self._weight_leg1)[:, :, :, None]
            temp = cp.einsum('ijkc,lk->ijlc', weighted, M)
            leg1 = cp.einsum('ijkc,lj->ilkc', temp, M) * prephase[t, :, None, None, None]
            fc3_rec[t] += leg1
            
            # LEG 2
            phase_ik = phase0[t, :, :, None] * phase2[t, :, None, :]
            weighted = self._fc3_base * (phase_ik * self._weight_leg23)[:, :, :, None]
            temp = cp.einsum('jikc,li->jlkc', weighted, M)
            leg2 = cp.einsum('jikc,lk->jilc', temp, M) * prephase[t, :, None, None, None]
            leg2 = leg2.transpose(1, 0, 2, 3).reshape(P, P, P, 3, 3, 3)
            leg2 = leg2.transpose(0, 1, 2, 4, 3, 5).reshape(P, P, P, 27)
            fc3_rec[t] += leg2
            
            # LEG 3
            phase_ji = phase1[t, :, :, None] * phase0[t, :, None, :]
            weighted = self._fc3_base * (phase_ji * self._weight_leg23)[:, :, :, None]
            temp = cp.einsum('kjic,lj->klic', weighted, M)
            leg3 = cp.einsum('kjic,li->kjlc', temp, M) * prephase[t, :, None, None, None]
            leg3 = leg3.transpose(2, 1, 0, 3).reshape(P, P, P, 3, 3, 3)
            leg3 = leg3.transpose(0, 1, 2, 5, 4, 3).reshape(P, P, P, 27)
            fc3_rec[t] += leg3
        
        fc3_rec /= 3.0
        fc3_rec = fc3_rec.reshape(T, P, P, P, 3, 3, 3)
        
        if return_gpu:
            return fc3_rec
        else:
            return cp.asnumpy(fc3_rec)

    def _run_legacy_batch(self, triplet_addresses, return_gpu=True):
        """Batched legacy algorithm for all triplets."""
        T = len(triplet_addresses)
        P, S = self._num_patom, self._num_satom
        
        all_q_vecs = triplet_addresses.astype(np.float64) / self._mesh[None, None, :]
        phase_tables = self._compute_phase_batch(all_q_vecs)
        phase1 = phase_tables[:, :, :, 1]
        phase2 = phase_tables[:, :, :, 2]
        
        q_sums = all_q_vecs.sum(axis=1)
        q_sums_gpu = cp.asarray(q_sums, dtype=cp.float64)
        prephase = cp.exp(1j * PI2 * cp.dot(q_sums_gpu, self._scaled_positions_gpu.T))
        
        M = self._M
        fc3_rec = cp.zeros((T, P, P, P, 27), dtype=cp.complex128)
        
        for t in range(T):
            phase_weight = phase1[t, :, :, None] * phase2[t, :, None, :]
            weighted = self._fc3_base * phase_weight[:, :, :, None]
            temp = cp.einsum('ijkc,lk->ijlc', weighted, M)
            result = cp.einsum('ijkc,lj->ilkc', temp, M)
            fc3_rec[t] = result * prephase[t, :, None, None, None]
        
        fc3_rec = fc3_rec.reshape(T, P, P, P, 3, 3, 3)
        
        if return_gpu:
            return fc3_rec
        else:
            return cp.asnumpy(fc3_rec)
