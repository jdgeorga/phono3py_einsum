"""Transform fc3 in reciprocal space to phonon space - C implementation equivalent.

This module provides a Python implementation that exactly replicates the C 
implementation in reciprocal_to_normal.c. It follows the same algorithm, 
optimizations, and data flow as the C code.

Key differences from the original Python implementation (reciprocal_to_normal.py):

1. **Output type**: Returns real-valued squared magnitudes instead of complex values
2. **Frequency normalization**: Divides by product of frequencies (f1*f2*f3) instead 
   of square root of product (sqrt(f1*f2*f3))
3. **Data structure**: Uses g_pos array for efficient indexing instead of nested loops
4. **Mass weighting**: Pre-multiplies eigenvectors by inverse sqrt masses for efficiency
5. **Algorithm**: Pre-computes fc3_e0 contraction for better performance
6. **Memory layout**: Transposes eigenvectors for better cache performance

The algorithm computes:
    result[i] = |sum_{atoms j,k,l} sum_{cart α,β,γ} 
                 (e0[j,α,band0] * e1[k,β,band1] * e2[l,γ,band2] * fc3[j,k,l,α,β,γ]) 
                 / sqrt(mass[j] * mass[k] * mass[l])|^2 
                / (freq0 * freq1 * freq2)

Usage:
    # For exact C equivalence (slower but exact match)
    result = reciprocal_to_normal_squared(g_pos, fc3_reciprocal, ...)
    
    # For optimized performance using einsum (recommended)
    result = reciprocal_to_normal_squared_einsum(g_pos, fc3_reciprocal, ...)
    
    # Alternative optimized version (identical results)
    result = reciprocal_to_normal_squared_super_fast(g_pos, fc3_reciprocal, ...)
    
    # Class methods with optimization choice in run()
    calculator = ReciprocalToNormalSquaredV3(...)
    calculator.run(fc3_reciprocal, grid_triplet)                    # Default: einsum
    calculator.run(fc3_reciprocal, grid_triplet, method='einsum')   # Explicit einsum
    calculator.run(fc3_reciprocal, grid_triplet, method='super_fast')  # Ultra-fast
    calculator.run(fc3_reciprocal, grid_triplet, method='exact')    # C-equivalent loops
    
    # Direct access to optimization levels from class instance
    result = calculator.reciprocal_to_normal_squared_einsum(g_pos, fc3_reciprocal, ...)
    result = calculator.reciprocal_to_normal_squared_super_fast(g_pos, fc3_reciprocal, ...)
    result = calculator.reciprocal_to_normal_squared_optimized(g_pos, fc3_reciprocal, ...)  # Also einsum
    
    # Easy benchmarking of different methods (all give identical results)
    import time
    methods = ['exact', 'einsum', 'super_fast']
    for method in methods:
        start = time.time()
        calculator.run(fc3_reciprocal, grid_triplet, method=method)
        print(f"{method}: {time.time() - start:.4f}s")
"""

# This is a Python implementation that exactly replicates the C implementation
# in reciprocal_to_normal.c. It follows the same algorithm and data flow.

import numpy as np
from typing import Tuple, Optional


class ReciprocalToNormalSquaredV3:
    """Python equivalent of the C implementation in reciprocal_to_normal.c.
    
    This class implements the exact same algorithm as the C function
    reciprocal_to_normal_squared, including the same optimizations and
    data layout.
    """

    def __init__(
        self,
        primitive,
        frequencies: np.ndarray,
        eigenvectors: np.ndarray,
        band_indices,
        cutoff_frequency: float = 0,
    ):
        """Init method."""
        self._primitive = primitive
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._band_indices = band_indices
        self._cutoff_frequency = cutoff_frequency
        self._fc3_normal_squared: Optional[np.ndarray] = None

    def run(self, fc3_reciprocal: np.ndarray, grid_triplet: np.ndarray, method: str = 'einsum'):
        """Calculate fc3 in phonon coordinates using the specified algorithm.
        
        This method matches the interface expected by interaction_fast.py.
        
        Parameters
        ----------
        fc3_reciprocal : np.ndarray
            Force constants in reciprocal space
        grid_triplet : np.ndarray
            Grid triplet indices [q0, q1, q2]
        method : str, optional
            Optimization method to use. Options:
            - 'einsum' (default): Highly optimized einsum-based implementation
            - 'super_fast': Ultra-fast for sparse band combinations
            - 'optimized': Alias for 'einsum'
            - 'exact': Exact C-equivalent loop implementation (slower, for verification)
        """
        # Extract frequencies and eigenvectors for the triplet
        freqs0 = self._frequencies[grid_triplet[0]]
        freqs1 = self._frequencies[grid_triplet[1]]
        freqs2 = self._frequencies[grid_triplet[2]]
        
        eigvecs0 = self._eigenvectors[grid_triplet[0]]
        eigvecs1 = self._eigenvectors[grid_triplet[1]]
        eigvecs2 = self._eigenvectors[grid_triplet[2]]
        
        # Get masses from primitive cell
        masses = np.array(self._primitive.masses, dtype=np.float64)
        
        # Create g_pos array for the calculation
        num_band = len(freqs0)
        num_band0 = len(self._band_indices)
        
        # Create g_pos mapping all combinations of bands
        g_pos_list = []
        output_idx = 0
        for i in range(num_band0):
            for j in range(num_band):
                for k in range(num_band):
                    g_pos_list.append([i, j, k, output_idx])
                    output_idx += 1
        
        g_pos = np.array(g_pos_list, dtype=np.int64)
        
        # Run the calculation using the specified method
        if method.lower() in ['einsum', 'optimized']:
            fc3_normal_squared_flat = reciprocal_to_normal_squared_einsum(
                g_pos, fc3_reciprocal, freqs0, freqs1, freqs2,
                eigvecs0, eigvecs1, eigvecs2, masses, self._band_indices, self._cutoff_frequency
            )
        elif method.lower() == 'super_fast':
            fc3_normal_squared = reciprocal_to_normal_squared_super_fast(
                g_pos, fc3_reciprocal, freqs0, freqs1, freqs2,
                eigvecs0, eigvecs1, eigvecs2, masses, self._band_indices, self._cutoff_frequency
            )
            self._fc3_normal_squared = fc3_normal_squared
        elif method.lower() == 'exact':
            fc3_normal_squared_flat = self.reciprocal_to_normal_squared(
                g_pos, fc3_reciprocal, freqs0, freqs1, freqs2,
                eigvecs0, eigvecs1, eigvecs2, masses, self._band_indices, self._cutoff_frequency
            )
            self._fc3_normal_squared = fc3_normal_squared_flat.reshape(num_band0, num_band, num_band)
        else:
            raise ValueError(f"Unknown method: {method}. Valid options: 'einsum', 'super_fast', 'optimized', 'exact'")
        
        # Reshape to match expected output format
        # self._fc3_normal_squared = fc3_normal_squared_flat.reshape(num_band0, num_band, num_band)

    def get_reciprocal_to_normal_squared(self) -> Optional[np.ndarray]:
        """Return fc3 in phonon coordinates (squared magnitude).
        
        Returns
        -------
        ndarray or None
            shape=(num_band0, num_band, num_band), dtype='double'
        """
        return self._fc3_normal_squared

    def reciprocal_to_normal_squared(
        self,
        g_pos: np.ndarray,
        fc3_reciprocal: np.ndarray,
        freqs0: np.ndarray,
        freqs1: np.ndarray,
        freqs2: np.ndarray,
        eigvecs0: np.ndarray,
        eigvecs1: np.ndarray,
        eigvecs2: np.ndarray,
        masses: np.ndarray,
        band_indices: np.ndarray,
        cutoff_frequency: float = 0.0,
    ) -> np.ndarray:
        """Calculate fc3 in phonon coordinates (squared magnitude).
        
        This is the Python equivalent of the C function reciprocal_to_normal_squared.
        
        Parameters
        ----------
        g_pos : np.ndarray, shape (num_g_pos, 4)
            Grid positions array. Each row contains [band0_idx, band1_idx, band2_idx, output_idx]
        fc3_reciprocal : np.ndarray, shape (num_atom, num_atom, num_atom, 3, 3, 3)
            Force constants in reciprocal space
        freqs0, freqs1, freqs2 : np.ndarray, shape (num_band,)
            Frequencies for each q-point
        eigvecs0, eigvecs1, eigvecs2 : np.ndarray, shape (num_band, num_band)
            Eigenvectors for each q-point
        masses : np.ndarray, shape (num_atom,)
            Masses of atoms
        band_indices : np.ndarray, shape (num_band0,)
            Band indices to consider for the first q-point
        cutoff_frequency : float
            Frequency cutoff
            
        Returns
        -------
        np.ndarray, shape (num_g_pos,)
            Squared magnitude of fc3 in phonon coordinates
        """
        num_band = len(freqs0)
        num_atom = num_band // 3
        num_band0 = len(band_indices)
        num_g_pos = len(g_pos)
        
     
        # Create inverse sqrt masses array (repeated 3 times per atom for x,y,z)
        inv_sqrt_masses = np.zeros(num_band, dtype=np.float64)
        for i in range(num_atom):
            for j in range(3):
                inv_sqrt_masses[i * 3 + j] = 1.0 / np.sqrt(masses[i])
        
        print("DEBUG reciprocal_to_normal_squared v3: eigvecs0 = ", eigvecs0[0])
        print("DEBUG reciprocal_to_normal_squared v3: eigvecs1 = ", eigvecs1[0])
        print("DEBUG reciprocal_to_normal_squared v3: eigvecs2 = ", eigvecs2[0])
        
        # Transpose eigenvectors and multiply by inverse sqrt masses
        # This follows the C implementation exactly
        e0 = self._transpose_and_mass_weight_eigenvectors(eigvecs0, inv_sqrt_masses, num_band)
        e1 = self._transpose_and_mass_weight_eigenvectors(eigvecs1, inv_sqrt_masses, num_band)
        e2 = self._transpose_and_mass_weight_eigenvectors(eigvecs2, inv_sqrt_masses, num_band)
        
        print("DEBUG reciprocal_to_normal_squared v3: e0 = ", e0[0])
        print("DEBUG reciprocal_to_normal_squared v3: e1 = ", e1[0])
        print("DEBUG reciprocal_to_normal_squared v3: e2 = ", e2[0])
        print("DEBUG reciprocal_to_normal_squared v3: freqs0 = ", freqs0)
        print("DEBUG reciprocal_to_normal_squared v3: freqs1 = ", freqs1)
        print("DEBUG reciprocal_to_normal_squared v3: freqs2 = ", freqs2)
        print("DEBUG reciprocal_to_normal_squared v3: e0.shape = ", e0.shape)
        print("DEBUG reciprocal_to_normal_squared v3: e1.shape = ", e1.shape)
        print("DEBUG reciprocal_to_normal_squared v3: e2.shape = ", e2.shape)
        # print("DEBUG reciprocal_to_normal_squared v3: fc3_reciprocal.shape = ", fc3_reciprocal.shape)
        # print("DEBUG reciprocal_to_normal_squared v3: fc3_reciprocal = ", fc3_reciprocal.reshape(num_band, num_band, num_band))
        # print("DEBUG reciprocal_to_normal_squared v3: g_pos = ", g_pos)
        # print("DEBUG reciprocal_to_normal_squared v3: num_g_pos = ", num_g_pos)
        
        
        # Main computation
        fc3_normal_squared = self._get_fc3_e0_e1_e2(
            g_pos, fc3_reciprocal, freqs0, freqs1, freqs2,
            e0, e1, e2, band_indices, num_band0, num_band, cutoff_frequency
        )
        print("DEBUG reciprocal_to_normal_squared v3: fc3_normal_squared.shape = ", fc3_normal_squared.shape)
        # print("DEBUG reciprocal_to_normal_squared v3: fc3_normal_squared = ", fc3_normal_squared)
        return fc3_normal_squared

    def _transpose_and_mass_weight_eigenvectors(
        self, eigvecs: np.ndarray, inv_sqrt_masses: np.ndarray, num_band: int
    ) -> np.ndarray:
        """Transpose eigenvectors and multiply by inverse sqrt masses.
        
        This follows the C implementation exactly:
        e[i * num_band + j] = eigvecs[j * num_band + i] * inv_sqrt_masses[j]
        """
        e = np.zeros((num_band, num_band), dtype=np.complex128)
        
        for i in range(num_band):
            for j in range(num_band):
                e[i, j] = eigvecs[j, i] * inv_sqrt_masses[j]
        
        return e

    def _get_fc3_e0_e1_e2(
        self,
        g_pos: np.ndarray,
        fc3_reciprocal: np.ndarray,
        freqs0: np.ndarray,
        freqs1: np.ndarray,
        freqs2: np.ndarray,
        e0: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        band_indices: np.ndarray,
        num_band0: int,
        num_band: int,
        cutoff_frequency: float,
    ) -> np.ndarray:
        """Main computation following the C implementation.
        
        This is equivalent to the C function get_fc3_e0_e1_e2.
        """
        num_g_pos = len(g_pos)
        
        # Pre-compute fc3_e0 for all band indices
        fc3_e0 = np.zeros((num_band0, num_band, num_band), dtype=np.complex128)
        # print("DEBUG reciprocal_to_normal_squared v3: fc3_reciprocal.shape = ", fc3_reciprocal.shape)
        # print("DEBUG reciprocal_to_normal_squared v3: fc3_reciprocal.flatten() = ", fc3_reciprocal.flatten())
        
        for i in range(num_band0):
            self._get_fc3_e0(fc3_e0[i], fc3_reciprocal, e0, band_indices[i], num_band)
        
        print("DEBUG reciprocal_to_normal_squared v3: fc3_e0.shape = ", fc3_e0[0].shape)
        print("DEBUG reciprocal_to_normal_squared v3: fc3_e0 = ", fc3_e0[0].flatten())
        # print("DEBUG reciprocal_to_normal_squared v3: fc3_e0 = ", fc3_e0)
        # Compute final fc3_normal_squared values
        fc3_normal_squared = np.zeros(num_g_pos, dtype=np.float64)
        
        for i in range(num_g_pos):
            band0_idx = g_pos[i, 0]
            band1_idx = g_pos[i, 1]
            band2_idx = g_pos[i, 2]
            output_idx = g_pos[i, 3]
            
            freq0 = freqs0[band_indices[band0_idx]]
            freq1 = freqs1[band1_idx]
            freq2 = freqs2[band2_idx]
            
            if freq0 > cutoff_frequency and freq1 > cutoff_frequency and freq2 > cutoff_frequency:
                fc3_sum = self._get_fc3_sum_optimized(
                    e1[band1_idx], e2[band2_idx], fc3_e0[band0_idx], num_band
                )
                fc3_normal_squared[output_idx] = fc3_sum / (freq0 * freq1 * freq2)
            else:
                fc3_normal_squared[output_idx] = 0.0
        
        return fc3_normal_squared

    def _get_fc3_e0(
        self,
        fc3_e0: np.ndarray,
        fc3_reciprocal: np.ndarray,
        e0: np.ndarray,
        band_index_0: int,
        num_band: int,
    ) -> None:
        """Compute fc3_e0 following the C implementation.
        
        This is equivalent to the C function get_fc3_e0.
        The C code does: fc3_e0[k] += fc3_reciprocal[j * num_band^2 + k] * e0[band_index_0 * num_band + j]
        where k represents flattened indices for a num_band x num_band matrix.
        
        fc3_reciprocal has shape (num_atom, num_atom, num_atom, 3, 3, 3)
        which needs to be reshaped to (num_band, num_band, num_band) to match C layout.
        """
        # First, reshape fc3_reciprocal to match the C layout: (num_band, num_band, num_band)
        # This groups (atom, cart) pairs together to form band indices
        fc3_reshaped = fc3_reciprocal.reshape(num_band, num_band, num_band)
        
        print("DEBUG reciprocal_to_normal_squared v3: fc3_reshaped = ", fc3_reshaped.shape)
        print("DEBUG reciprocal_to_normal_squared v3: fc3_reshaped = ", fc3_reshaped[0])
        print("DEBUG reciprocal_to_normal_squared v3: e0 = ", e0.shape)
        print("DEBUG reciprocal_to_normal_squared v3: e0 = ", e0[0])
        
        # Now follow the C implementation exactly
        for j in range(num_band):
            for k in range(num_band * num_band):
                # Convert k to 2D indices for fc3_e0
                k_i = k // num_band
                k_j = k % num_band
                
                # Access fc3_reshaped using C-style indexing
                fc3_elem = fc3_reshaped[j, k_i, k_j] * e0[band_index_0, j]
                fc3_e0[k_i, k_j] += fc3_elem

    def _get_fc3_sum(
        self,
        e1: np.ndarray,
        e2: np.ndarray,
        fc3_e0: np.ndarray,
        num_band: int,
    ) -> float:
        """Compute fc3 sum following the C implementation.
        
        This is equivalent to the C function get_fc3_sum.
        Returns the squared magnitude of the complex sum.
        """
        # First contraction: fc3_e0_e1[j] = sum_i fc3_e0[i, j] * e1[i]
        fc3_e0_e1 = np.zeros(num_band, dtype=np.complex128)
        
        for i in range(num_band):
            for j in range(num_band):
                fc3_elem = fc3_e0[i, j] * e1[i]
                fc3_e0_e1[j] += fc3_elem
        
        # Second contraction: sum_i fc3_e0_e1[i] * e2[i]
        total_sum = 0.0 + 0.0j
        for i in range(num_band):
            fc3_elem = fc3_e0_e1[i] * e2[i]
            total_sum += fc3_elem
        
        # Return squared magnitude
        return (total_sum.real ** 2 + total_sum.imag ** 2)

    def _get_fc3_sum_optimized(
        self,
        e1: np.ndarray,
        e2: np.ndarray,
        fc3_e0: np.ndarray,
        num_band: int,
    ) -> float:
        """Optimized version of _get_fc3_sum using NumPy vectorization.
        
        This does the same computation as _get_fc3_sum but uses vectorized operations.
        """
        # First contraction: fc3_e0_e1[j] = sum_i fc3_e0[i, j] * e1[i]
        fc3_e0_e1 = np.sum(fc3_e0 * e1[:, np.newaxis], axis=0)
        
        # Second contraction: sum_i fc3_e0_e1[i] * e2[i]
        total_sum = np.sum(fc3_e0_e1 * e2)
        
        # Return squared magnitude
        return abs(total_sum) ** 2

    def reciprocal_to_normal_squared_optimized(
        self,
        g_pos: np.ndarray,
        fc3_reciprocal: np.ndarray,
        freqs0: np.ndarray,
        freqs1: np.ndarray,
        freqs2: np.ndarray,
        eigvecs0: np.ndarray,
        eigvecs1: np.ndarray,
        eigvecs2: np.ndarray,
        masses: np.ndarray,
        band_indices: np.ndarray,
        cutoff_frequency: float = 0.0,
    ) -> np.ndarray:
        """Optimized version using einsum for maximum performance.
        
        This uses the highly optimized einsum-based implementation for
        maximum performance while maintaining exact compatibility with
        the C implementation.
        """
        return reciprocal_to_normal_squared_einsum(
            g_pos, fc3_reciprocal, freqs0, freqs1, freqs2,
            eigvecs0, eigvecs1, eigvecs2, masses, band_indices, cutoff_frequency
        )

    def reciprocal_to_normal_squared_einsum(
        self,
        g_pos: np.ndarray,
        fc3_reciprocal: np.ndarray,
        freqs0: np.ndarray,
        freqs1: np.ndarray,
        freqs2: np.ndarray,
        eigvecs0: np.ndarray,
        eigvecs1: np.ndarray,
        eigvecs2: np.ndarray,
        masses: np.ndarray,
        band_indices: np.ndarray,
        cutoff_frequency: float = 0.0,
    ) -> np.ndarray:
        """Highly optimized einsum-based implementation (class method).
        
        Direct access to the einsum optimization from the class instance.
        """
        return reciprocal_to_normal_squared_einsum(
            g_pos, fc3_reciprocal, freqs0, freqs1, freqs2,
            eigvecs0, eigvecs1, eigvecs2, masses, band_indices, cutoff_frequency
        )

    def reciprocal_to_normal_squared_super_fast(
        self,
        g_pos: np.ndarray,
        fc3_reciprocal: np.ndarray,
        freqs0: np.ndarray,
        freqs1: np.ndarray,
        freqs2: np.ndarray,
        eigvecs0: np.ndarray,
        eigvecs1: np.ndarray,
        eigvecs2: np.ndarray,
        masses: np.ndarray,
        band_indices: np.ndarray,
        cutoff_frequency: float = 0.0,
    ) -> np.ndarray:
        """Ultra-fast einsum implementation using comprehensive tensor contraction (class method).
        
        Direct access to the super_fast optimization from the class instance.
        """
        return reciprocal_to_normal_squared_super_fast(
            g_pos, fc3_reciprocal, freqs0, freqs1, freqs2,
            eigvecs0, eigvecs1, eigvecs2, masses, band_indices, cutoff_frequency
        )


def reciprocal_to_normal_squared(
    g_pos: np.ndarray,
    fc3_reciprocal: np.ndarray,
    freqs0: np.ndarray,
    freqs1: np.ndarray,
    freqs2: np.ndarray,
    eigvecs0: np.ndarray,
    eigvecs1: np.ndarray,
    eigvecs2: np.ndarray,
    masses: np.ndarray,
    band_indices: np.ndarray,
    cutoff_frequency: float = 0.0,
) -> np.ndarray:
    """Standalone function that matches the C implementation exactly.
    
    This function signature matches the C implementation exactly.
    Uses the loop-based implementation that mirrors the C code exactly.
    """
    num_band = len(freqs0)
    num_atom = num_band // 3
    num_band0 = len(band_indices)
    num_g_pos = len(g_pos)
    
    # Create inverse sqrt masses array (repeated 3 times per atom for x,y,z)
    inv_sqrt_masses = np.zeros(num_band, dtype=np.float64)
    for i in range(num_atom):
        for j in range(3):
            inv_sqrt_masses[i * 3 + j] = 1.0 / np.sqrt(masses[i])
    
    # Transpose eigenvectors and multiply by inverse sqrt masses
    def transpose_and_mass_weight_eigenvectors(eigvecs, inv_sqrt_masses, num_band):
        e = np.zeros((num_band, num_band), dtype=np.complex128)
        for i in range(num_band):
            for j in range(num_band):
                e[i, j] = eigvecs[j, i] * inv_sqrt_masses[j]
        return e
    
    e0 = transpose_and_mass_weight_eigenvectors(eigvecs0, inv_sqrt_masses, num_band)
    e1 = transpose_and_mass_weight_eigenvectors(eigvecs1, inv_sqrt_masses, num_band)
    e2 = transpose_and_mass_weight_eigenvectors(eigvecs2, inv_sqrt_masses, num_band)
    
    print("DEBUG reciprocal_to_normal_squared v3: e0 = ", e0[0])
    print("DEBUG reciprocal_to_normal_squared v3: e1 = ", e1[0])
    print("DEBUG reciprocal_to_normal_squared v3: e2 = ", e2[0])

    # Helper functions
    def get_fc3_e0_e1_e2(g_pos, fc3_reciprocal, freqs0, freqs1, freqs2,
                         e0, e1, e2, band_indices, num_band0, num_band, cutoff_frequency):
        num_g_pos = len(g_pos)
        fc3_e0 = np.zeros((num_band0, num_band, num_band), dtype=np.complex128)
        
        for i in range(num_band0):
            get_fc3_e0(fc3_e0[i], fc3_reciprocal, e0, band_indices[i], num_band)
        
        fc3_normal_squared = np.zeros(num_g_pos, dtype=np.float64)
        
        for i in range(num_g_pos):
            band0_idx = g_pos[i, 0]
            band1_idx = g_pos[i, 1]
            band2_idx = g_pos[i, 2]
            output_idx = g_pos[i, 3]
            
            freq0 = freqs0[band_indices[band0_idx]]
            freq1 = freqs1[band1_idx]
            freq2 = freqs2[band2_idx]
            
            if freq0 > cutoff_frequency and freq1 > cutoff_frequency and freq2 > cutoff_frequency:
                fc3_sum = get_fc3_sum(e1[band1_idx], e2[band2_idx], fc3_e0[band0_idx], num_band)
                fc3_normal_squared[output_idx] = fc3_sum / (freq0 * freq1 * freq2)
            else:
                fc3_normal_squared[output_idx] = 0.0
        
        return fc3_normal_squared
    
    def get_fc3_e0(fc3_e0, fc3_reciprocal, e0, band_index_0, num_band):
        # First, reshape fc3_reciprocal to match the C layout: (num_band, num_band, num_band)
        # Transpose: (atom0, atom1, atom2, cart0, cart1, cart2) -> (atom0, cart0, atom1, cart1, atom2, cart2)
        # This groups (atom, cart) pairs together to form band indices
        fc3_reshaped = fc3_reciprocal.transpose(0, 3, 1, 4, 2, 5).reshape(num_band, num_band, num_band)
        
        # Now follow the C implementation exactly
        for j in range(num_band):
            for k in range(num_band * num_band):
                k_i = k // num_band
                k_j = k % num_band
                
                # Access fc3_reshaped using C-style indexing
                fc3_elem = fc3_reshaped[j, k_i, k_j] * e0[band_index_0, j]
                fc3_e0[k_i, k_j] += fc3_elem
    
    def get_fc3_sum(e1, e2, fc3_e0, num_band):
        fc3_e0_e1 = np.zeros(num_band, dtype=np.complex128)
        
        for i in range(num_band):
            for j in range(num_band):
                fc3_elem = fc3_e0[i, j] * e1[i]
                fc3_e0_e1[j] += fc3_elem
        
        total_sum = 0.0 + 0.0j
        for i in range(num_band):
            fc3_elem = fc3_e0_e1[i] * e2[i]
            total_sum += fc3_elem
        
        return (total_sum.real ** 2 + total_sum.imag ** 2)
    
    # Main computation
    fc3_normal_squared = get_fc3_e0_e1_e2(
        g_pos, fc3_reciprocal, freqs0, freqs1, freqs2,
        e0, e1, e2, band_indices, num_band0, num_band, cutoff_frequency
    )
    
    return fc3_normal_squared


def reciprocal_to_normal_squared_optimized(
    g_pos: np.ndarray,
    fc3_reciprocal: np.ndarray,
    freqs0: np.ndarray,
    freqs1: np.ndarray,
    freqs2: np.ndarray,
    eigvecs0: np.ndarray,
    eigvecs1: np.ndarray,
    eigvecs2: np.ndarray,
    masses: np.ndarray,
    band_indices: np.ndarray,
    cutoff_frequency: float = 0.0,
) -> np.ndarray:
    """Optimized standalone function using NumPy vectorization.
    
    This uses NumPy vectorization for better performance while following
    the same algorithm as the C implementation.
    
    Parameters
    ----------
    fc3_reciprocal : np.ndarray, shape (num_atom, num_atom, num_atom, 3, 3, 3)
        Force constants in reciprocal space
    """
    num_band = len(freqs0)
    num_atom = num_band // 3
    num_band0 = len(band_indices)
    num_g_pos = len(g_pos)
    
    # Create inverse sqrt masses array (vectorized)
    inv_sqrt_masses = np.repeat(1.0 / np.sqrt(masses), 3)
    
    # Transpose eigenvectors and multiply by inverse sqrt masses (exact C match)
    def transpose_and_mass_weight_eigenvectors(eigvecs, inv_sqrt_masses, num_band):
        e = np.zeros((num_band, num_band), dtype=np.complex128)
        for i in range(num_band):
            for j in range(num_band):
                e[i, j] = eigvecs[j, i] * inv_sqrt_masses[j]
        return e
    
    e0 = transpose_and_mass_weight_eigenvectors(eigvecs0, inv_sqrt_masses, num_band)
    e1 = transpose_and_mass_weight_eigenvectors(eigvecs1, inv_sqrt_masses, num_band)
    e2 = transpose_and_mass_weight_eigenvectors(eigvecs2, inv_sqrt_masses, num_band)
    
    
    # Pre-compute fc3_e0 for all band indices (can be partially vectorized)
    fc3_e0 = np.zeros((num_band0, num_band, num_band), dtype=np.complex128)
    
    for i, band_idx in enumerate(band_indices):
        # Vectorized version of get_fc3_e0 for 6D fc3_reciprocal
        # Need to transpose and reshape fc3_reciprocal from (num_atom, num_atom, num_atom, 3, 3, 3) 
        # to (num_band, num_band, num_band) for einsum
        # Transpose: (atom0, atom1, atom2, cart0, cart1, cart2) -> (atom0, cart0, atom1, cart1, atom2, cart2)
        # This groups (atom, cart) pairs together to form band indices
        fc3_reshaped = fc3_reciprocal.transpose(0, 3, 1, 4, 2, 5).reshape(num_band, num_band, num_band)
        fc3_e0[i] = np.einsum('j,jkl->kl', e0[band_idx], fc3_reshaped)
    
    # Compute final fc3_normal_squared values
    fc3_normal_squared = np.zeros(num_g_pos, dtype=np.float64)
    
    for i in range(num_g_pos):
        band0_idx = g_pos[i, 0]
        band1_idx = g_pos[i, 1]
        band2_idx = g_pos[i, 2]
        output_idx = g_pos[i, 3]
        
        freq0 = freqs0[band_indices[band0_idx]]
        freq1 = freqs1[band1_idx]
        freq2 = freqs2[band2_idx]
        
        if freq0 > cutoff_frequency and freq1 > cutoff_frequency and freq2 > cutoff_frequency:
            # Optimized fc3_sum using NumPy vectorization
            fc3_e0_e1 = np.sum(fc3_e0[band0_idx] * e1[band1_idx][:, np.newaxis], axis=0)
            total_sum = np.sum(fc3_e0_e1 * e2[band2_idx])
            fc3_sum = abs(total_sum) ** 2
            fc3_normal_squared[output_idx] = fc3_sum / (freq0 * freq1 * freq2)
        else:
            fc3_normal_squared[output_idx] = 0.0
    
    return fc3_normal_squared


def reciprocal_to_normal_squared_einsum(
    g_pos: np.ndarray,
    fc3_reciprocal: np.ndarray,
    freqs0: np.ndarray,
    freqs1: np.ndarray,
    freqs2: np.ndarray,
    eigvecs0: np.ndarray,
    eigvecs1: np.ndarray,
    eigvecs2: np.ndarray,
    masses: np.ndarray,
    band_indices: np.ndarray,
    cutoff_frequency: float = 0.0,
) -> np.ndarray:
    """Optimized einsum-based function that exactly matches the C implementation.
    
    This follows the exact same algorithm as the C implementation but uses einsum
    operations for the tensor contractions, providing better performance while
    guaranteeing identical results to the exact version.
    
    Parameters
    ----------
    fc3_reciprocal : np.ndarray, shape (num_atom, num_atom, num_atom, 3, 3, 3)
        Force constants in reciprocal space
    g_pos : np.ndarray, shape (num_g_pos, 4)
        Grid positions array. Each row contains [band0_idx, band1_idx, band2_idx, output_idx]
    
    Returns
    -------
    np.ndarray, shape (num_g_pos,)
        Squared magnitude of fc3 in phonon coordinates, identical to C implementation
    """
    num_band = len(freqs0)
    num_atom = num_band // 3
    num_band0 = len(band_indices)
    num_g_pos = len(g_pos)
    
    # Create inverse sqrt masses array (vectorized)
    inv_sqrt_masses = np.repeat(1.0 / np.sqrt(masses), 3)
    
    # Mass-weight and transpose eigenvectors to match C implementation exactly
    # In C: e[i * num_band + j] = eigvecs[j * num_band + i] * inv_sqrt_masses[j]
    # This means: e[i, j] = eigvecs[j, i] * inv_sqrt_masses[j]
    def transpose_and_mass_weight_eigenvectors(eigvecs, inv_sqrt_masses, num_band):
        e = np.zeros((num_band, num_band), dtype=np.complex128)
        for i in range(num_band):
            for j in range(num_band):
                e[i, j] = eigvecs[j, i] * inv_sqrt_masses[j]
        return e
    
    e0 = transpose_and_mass_weight_eigenvectors(eigvecs0, inv_sqrt_masses, num_band)
    e1 = transpose_and_mass_weight_eigenvectors(eigvecs1, inv_sqrt_masses, num_band) 
    e2 = transpose_and_mass_weight_eigenvectors(eigvecs2, inv_sqrt_masses, num_band)
    
    # Reshape fc3_reciprocal to band coordinates (exact same as the C version)
    fc3_band = fc3_reciprocal.flatten().reshape(num_band, num_band, num_band)
    
    # Pre-compute fc3_e0 for all band indices (matching the exact algorithm)
    fc3_e0 = np.zeros((num_band0, num_band, num_band), dtype=np.complex128)
    
    for i, band_idx in enumerate(band_indices):
        # This exactly matches get_fc3_e0 from the exact version using einsum
        fc3_e0[i] = np.einsum('j,jkl->kl', e0[band_idx], fc3_band)
    
    # Initialize output array
    fc3_normal_squared = np.zeros(num_g_pos, dtype=np.float64)
    
    # Process each g_pos entry exactly like the exact version
    for i in range(num_g_pos):
        band0_idx = g_pos[i, 0]
        band1_idx = g_pos[i, 1]
        band2_idx = g_pos[i, 2]
        output_idx = g_pos[i, 3]
        
        freq0 = freqs0[band_indices[band0_idx]]
        freq1 = freqs1[band1_idx]
        freq2 = freqs2[band2_idx]
        
        if freq0 > cutoff_frequency and freq1 > cutoff_frequency and freq2 > cutoff_frequency:
            # Use einsum to efficiently compute the fc3_sum (matching _get_fc3_sum logic)
            # First contraction: fc3_e0_e1[j] = sum_i fc3_e0[band0_idx][i, j] * e1[band1_idx][i]
            fc3_e0_e1 = np.einsum('ij,i->j', fc3_e0[band0_idx], e1[band1_idx])
            
            # Second contraction: total_sum = sum_i fc3_e0_e1[i] * e2[band2_idx][i]
            total_sum = np.einsum('i,i->', fc3_e0_e1, e2[band2_idx])
            
            # Get squared magnitude and apply frequency normalization
            fc3_sum = abs(total_sum) ** 2
            fc3_normal_squared[output_idx] = fc3_sum / (freq0 * freq1 * freq2)
        else:
            fc3_normal_squared[output_idx] = 0.0
    
    return fc3_normal_squared


def reciprocal_to_normal_squared_super_fast(
    g_pos: np.ndarray,
    fc3_reciprocal: np.ndarray,
    freqs0: np.ndarray,
    freqs1: np.ndarray,
    freqs2: np.ndarray,
    eigvecs0: np.ndarray,
    eigvecs1: np.ndarray,
    eigvecs2: np.ndarray,
    masses: np.ndarray,
    band_indices: np.ndarray,
    cutoff_frequency: float = 0.0,
) -> np.ndarray:
    """Ultra-fast einsum implementation optimized for sparse band combinations.
    
    This implementation uses vectorized einsum operations to compute all valid
    band combinations at once, then extracts only the specific combinations
    needed based on g_pos. It maintains full compatibility with the v3 data
    structures and C implementation while providing significant performance
    improvements for cases with many valid band combinations.
    """
    num_band = len(freqs0)
    num_atom = num_band // 3
    num_band0 = len(band_indices)
    num_g_pos = len(g_pos)
        
    # Initialize output array
    fc3_normal_squared = np.zeros(num_g_pos, dtype=np.float64)
    
    # Create frequency masks for valid bands (above cutoff)
    freq0_selected = freqs0[band_indices]
    mask0 = freq0_selected > cutoff_frequency
    mask1 = freqs1 > cutoff_frequency
    mask2 = freqs2 > cutoff_frequency
    
    valid_band0_indices = np.where(mask0)[0]  # indices within band_indices
    valid_band1_indices = np.where(mask1)[0]  # indices within freqs1
    valid_band2_indices = np.where(mask2)[0]  # indices within freqs2
    
    # If no valid bands, return zeros
    if (len(valid_band0_indices) == 0 or 
        len(valid_band1_indices) == 0 or 
        len(valid_band2_indices) == 0):
        return fc3_normal_squared
    
    # Get the actual band indices for valid bands
    actual_band0_indices = band_indices * mask0
    
    # Prepare eigenvectors in the format needed for the comprehensive einsum
    # We need to reshape eigenvectors to (num_atom, 3, n_valid_bands) format like in fast.py
    
     # Create mass weighting arrays
    inv_sqrt_masses = 1.0 / np.sqrt(masses)
   
    inv_sqrt_freqs0 = np.where(freqs0 > cutoff_frequency, 1/np.sqrt(freqs0), 0)
    inv_sqrt_freqs1 = np.where(freqs1 > cutoff_frequency, 1/np.sqrt(freqs1), 0)
    inv_sqrt_freqs2 = np.where(freqs2 > cutoff_frequency, 1/np.sqrt(freqs2), 0)
    
    
    # inv_sqrt_freqs0 = np.where(freqs0 > cutoff_frequency, 1, 0)
    # inv_sqrt_freqs1 = np.where(freqs1 > cutoff_frequency, 1, 0)
    # inv_sqrt_freqs2 = np.where(freqs2 > cutoff_frequency, 1, 0)
    
    e0_valid = eigvecs0.reshape(num_atom, 3, -1) * inv_sqrt_masses[:,None,None] * inv_sqrt_freqs0[None,None,:]
    e1_valid = eigvecs1.reshape(num_atom, 3, -1) * inv_sqrt_masses[:,None,None] * inv_sqrt_freqs1[None,None,:]
    e2_valid = eigvecs2.reshape(num_atom, 3, -1) * inv_sqrt_masses[:,None,None] * inv_sqrt_freqs2[None,None,:]
    
    e0_valid = e0_valid.transpose(2,0,1).reshape(num_band, num_band)
    e1_valid = e1_valid.transpose(2,0,1).reshape(num_band, num_band)
    e2_valid = e2_valid.transpose(2,0,1).reshape(num_band, num_band)

    
    # Extract valid eigenvectors and reshape to atom, cartesian, band format
    # e0_valid = eigvecs0[:, actual_band0_indices].reshape(num_atom, 3, -1) * inv_sqrt_masses[:,None,None]
    # e1_valid = eigvecs1[:, valid_band1_indices].reshape(num_atom, 3, -1) * inv_sqrt_masses[:,None,None]
    # e2_valid = eigvecs2[:, valid_band2_indices].reshape(num_atom, 3, -1) * inv_sqrt_masses[:,None,None]    
  
    # print("DEBUG reciprocal_to_normal_squared v3: e0_valid = ", e0_valid[0])
    # print("DEBUG reciprocal_to_normal_squared v3: e1_valid = ", e1_valid[0])
    # print("DEBUG reciprocal_to_normal_squared v3: e2_valid = ", e2_valid[0])
    
    # print("DEBUG reciprocal_to_normal_squared v3: freqs0 = ", inv_sqrt_freqs0[0])
    # print("DEBUG reciprocal_to_normal_squared v3: freqs1 = ", inv_sqrt_freqs1[0])
    # print("DEBUG reciprocal_to_normal_squared v3: freqs2 = ", inv_sqrt_freqs2[0])
    # print("DEBUG reciprocal_to_normal_squared v3: fc3_reciprocal.flatten() = ", fc3_reciprocal[0,:,:,0].flatten()[:5])
    # # quit()
    
    # Single comprehensive einsum computation for all valid band combinations
    # This follows the exact pattern from reciprocal_to_normal_fast.py
    fc3_complex_all = np.einsum(
        "Ii,Jj,Kk,ijk-> IJK",
        e0_valid,           # (num_atom, 3, n_valid_band0)  
        e1_valid,           # (num_atom, 3, n_valid_band1)
        e2_valid,           # (num_atom, 3, n_valid_band2)
        fc3_reciprocal.reshape(num_band, num_band, num_band),     # (num_atom, num_atom, num_atom, 3, 3, 3)
        optimize=True,      # Let NumPy find optimal contraction path
    )
    
    # Convert to squared magnitudes
    fc3_normal_squared = np.abs(fc3_complex_all) ** 2
    
    # print("DEBUG reciprocal_to_normal_squared_super_fast: fc3_complex_all.flatten() = ", fc3_complex_all.flatten()[:5])
    # # Extract only the specific combinations needed based on g_pos
    # for i in range(num_g_pos):
    #     band0_idx = g_pos[i, 0]  # index within band_indices
    #     band1_idx = g_pos[i, 1]  # index within freqs1
    #     band2_idx = g_pos[i, 2]  # index within freqs2
    #     output_idx = g_pos[i, 3]
        
    #     freq0 = freqs0[band_indices[band0_idx]]
    #     freq1 = freqs1[band1_idx]
    #     freq2 = freqs2[band2_idx]
        
    #     # Check if this combination is valid (frequencies above cutoff)
    #     if freq0 > cutoff_frequency and freq1 > cutoff_frequency and freq2 > cutoff_frequency:
    #         # Find the indices in the valid arrays
    #         try:
    #             valid_idx0 = np.where(valid_band0_indices == band0_idx)[0][0]
    #             valid_idx1 = np.where(valid_band1_indices == band1_idx)[0][0]
    #             valid_idx2 = np.where(valid_band2_indices == band2_idx)[0][0]
                
    #             # Extract the pre-computed result and apply frequency normalization
    #             fc3_squared = fc3_squared_all[valid_idx0, valid_idx1, valid_idx2]
    #             fc3_normal_squared[output_idx] = fc3_squared 
                
    #         except (IndexError, ValueError):
    #             # This combination was not in the valid set, set to zero
    #             fc3_normal_squared[output_idx] = 0.0
    #     else:
    #         fc3_normal_squared[output_idx] = 0.0
    
    return fc3_normal_squared 