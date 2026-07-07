#!/usr/bin/env python3
"""
Test the direct C binding for reciprocal_to_normal_squared.

This test uses the newly added py_reciprocal_to_normal_squared function
that directly calls the C implementation and compares it with the fast
Python implementation to verify correctness.
"""

import numpy as np
import sys
import time
from typing import Optional, Tuple

# Add the path to the phono3py module  
sys.path.insert(0, '../phono3py')

from phonon3.reciprocal_to_normal_fast import ReciprocalToNormalFast
from phonopy.structure.atoms import PhonopyAtoms

# Import the C extension with our new binding
try:
    import phono3py._phono3py as phono3c
    HAS_C_EXTENSION = True
    print("✓ Successfully imported phono3py._phono3py")
except ImportError as e:
    print(f"✗ Failed to import phono3py._phono3py: {e}")
    HAS_C_EXTENSION = False


def create_test_system(num_atom: int = 2) -> dict:
    """Create test data for comparison.
    
    Args:
        num_atom: Number of atoms in the primitive cell
    """
    
    # Create a system with the specified number of atoms
    if num_atom == 2:
        symbols = ['Si', 'Si'] 
        positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    elif num_atom == 4:
        symbols = ['Si', 'Si', 'Si', 'Si'] 
        positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.0], [0.75, 0.75, 0.25]]
    elif num_atom == 6:
        symbols = ['Si', 'Si', 'Si', 'Si', 'Si', 'Si'] 
        positions = [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.0], 
                    [0.6, 0.6, 0.2], [0.8, 0.8, 0.0], [0.0, 0.2, 0.4]]
    elif num_atom == 8:
        symbols = ['Si'] * 8
        positions = [[i*0.125, j*0.125, k*0.125] for i in range(2) for j in range(2) for k in range(2)]
    else:
        # General case: create a linear chain of atoms
        symbols = ['Si'] * num_atom
        positions = [[i/num_atom, 0.0, 0.0] for i in range(num_atom)]
    
    lattice = [[5.4, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 5.4]]
    
    primitive = PhonopyAtoms(symbols=symbols, positions=positions, cell=lattice)
    
    num_atom = len(primitive)
    num_band = num_atom * 3
    
    # Fixed seed for reproducible results
    np.random.seed(42)
    
    # Realistic frequencies (all above cutoff)
    num_grid_points = 3
    freqs = np.random.random((num_grid_points, num_band)) * 10.0 + 1.0
    freqs = freqs.astype('float64', order='C')
    
    # Realistic eigenvectors (orthonormal)
    eigvecs = np.zeros((num_grid_points, num_band, num_band), dtype='complex128', order='C')
    for i in range(num_grid_points):
        h = np.random.random((num_band, num_band)) + 1j * np.random.random((num_band, num_band))
        h = h + h.conj().T  # Make hermitian
        _, evecs = np.linalg.eigh(h)
        eigvecs[i] = evecs
    
    # Create reciprocal-space FC3 in the format Python expects: (num_atom, num_atom, num_atom, 3, 3, 3)
    fc3_reciprocal_6d = (np.random.random((num_atom, num_atom, num_atom, 3, 3, 3)) + 
                        1j * np.random.random((num_atom, num_atom, num_atom, 3, 3, 3))) * 0.01
    fc3_reciprocal_6d = fc3_reciprocal_6d.astype('complex128', order='C')
    
    # Convert to band format for C function: (num_band, num_band, num_band)
    fc3_reciprocal_3d = np.zeros((num_band, num_band, num_band), dtype='complex128', order='C')
    for i1 in range(num_atom):
        for i2 in range(num_atom):
            for i3 in range(num_atom):
                for c1 in range(3):
                    for c2 in range(3):
                        for c3 in range(3):
                            band1 = i1 * 3 + c1
                            band2 = i2 * 3 + c2  
                            band3 = i3 * 3 + c3
                            fc3_reciprocal_3d[band1, band2, band3] = fc3_reciprocal_6d[i1, i2, i3, c1, c2, c3]
    
    # Band indices and triplets (multiple triplets to test)
    band_indices = np.arange(num_band, dtype='int64')
    grid_triplets = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 2]]
    cutoff_frequency = 0.1
    
    return {
        'primitive': primitive,
        'frequencies': freqs,
        'eigenvectors': eigvecs,
        'fc3_reciprocal_6d': fc3_reciprocal_6d,  # For Python implementations
        'fc3_reciprocal_3d': fc3_reciprocal_3d,  # For C implementation
        'band_indices': band_indices,
        'grid_triplets': grid_triplets,
        'cutoff_frequency': cutoff_frequency,
        'num_atom': num_atom,
        'num_band': num_band,
        'num_grid_points': num_grid_points
    }


def call_direct_c_binding(test_data: dict) -> Tuple[Optional[dict], float]:
    """Call the new direct C binding for reciprocal_to_normal_squared."""
    
    if not HAS_C_EXTENSION:
        print("  C extension not available")
        return None, 0.0
    
    print("  Calling direct C reciprocal_to_normal_squared binding...")
    
    # Prepare output arrays for all triplets
    num_band0 = len(test_data['band_indices'])
    num_band = test_data['num_band']
    
    # Convert masses to proper format
    masses = np.array(test_data['primitive'].masses, dtype='float64', order='C')
    
    results = {}
    total_time = 0.0
    
    for i, grid_triplet in enumerate(test_data['grid_triplets']):
        print(f"    Processing triplet {i+1}/4: {grid_triplet}")
        
        fc3_normal_squared = np.zeros((num_band0, num_band, num_band), 
                                      dtype='float64', order='C')
        
        start_time = time.time()
        
        try:
            # Call the new direct C binding
            phono3c.reciprocal_to_normal_squared(
                fc3_normal_squared,                   # output array
                test_data['fc3_reciprocal_3d'],       # reciprocal-space FC3 (3D format)
                test_data['frequencies'],             # frequencies for all grid points
                test_data['eigenvectors'],            # eigenvectors for all grid points  
                masses,                               # atomic masses
                test_data['band_indices'],            # band indices
                grid_triplet[0],                      # grid point 0
                grid_triplet[1],                      # grid point 1
                grid_triplet[2],                      # grid point 2
                test_data['cutoff_frequency'],        # cutoff frequency
                0                                     # openmp_per_triplets
            )
            
            triplet_time = time.time() - start_time
            total_time += triplet_time
            results[tuple(grid_triplet)] = fc3_normal_squared.copy()
            print(f"    ✓ Triplet {grid_triplet} completed in {triplet_time:.4f} seconds")
            
        except Exception as e:
            print(f"    ✗ Triplet {grid_triplet} failed: {e}")
            results[tuple(grid_triplet)] = None
    
    print(f"  ✓ All direct C calls completed in {total_time:.4f} seconds")
    return results, total_time


def run_python_implementations(test_data: dict) -> Tuple[Optional[dict], float]:
    """Run the fast Python implementation."""
    
    print("  Running fast Python implementation...")
    
    results = {}
    total_time = 0.0
    
    for i, grid_triplet in enumerate(test_data['grid_triplets']):
        print(f"    Processing triplet {i+1}/4: {grid_triplet}")
        
        # Fast Python implementation
        start_time = time.time()
        calc_fast = ReciprocalToNormalFast(
            test_data['primitive'], 
            test_data['frequencies'], 
            test_data['eigenvectors'],
            test_data['band_indices'], 
            test_data['cutoff_frequency']
        )
        calc_fast.run(test_data['fc3_reciprocal_6d'], grid_triplet)
        result_fast = calc_fast.get_reciprocal_to_normal()
        triplet_time = time.time() - start_time
        total_time += triplet_time
        
        results[tuple(grid_triplet)] = result_fast
        print(f"    ✓ Triplet {grid_triplet} completed in {triplet_time:.4f} seconds")
    
    print(f"  ✓ All Python calls completed in {total_time:.4f} seconds")
    return results, total_time


def analyze_direct_c_results(result_orig: Optional[np.ndarray], 
                            result_fast: Optional[dict],
                            result_c: Optional[dict]) -> bool:
    """Analyze results from the direct C binding."""
    
    print(f"\n  Results Analysis (Direct C Binding):")
    print("  " + "=" * 50)
    
    overall_success = True
    
    if result_fast is None or result_c is None:
        print("  ✗ Missing results from Python or C implementations")
        return False
    
    # Analyze each triplet
    for triplet_key in result_fast.keys():
        triplet = list(triplet_key)
        print(f"\n  Analyzing triplet {triplet}:")
        print("  " + "-" * 30)
        
        fast_result = result_fast[triplet_key]
        c_result = result_c.get(triplet_key)
        
        success = True
        
        # Check if both implementations succeeded for this triplet
        if fast_result is not None and c_result is not None:
            print(f"    ✓ Both implementations succeeded for triplet {triplet}")
            
            # Calculate |Python|² for comparison with C
            python_squared = np.abs(fast_result) ** 2
            
            c_python_diff = np.max(np.abs(c_result - python_squared))
            relative_diff = c_python_diff / (np.max(np.abs(python_squared)) + 1e-16)
            
            print(f"    Result shape: {c_result.shape}")
            print(f"    Max |Python|² vs C difference: {c_python_diff:.2e}")
            print(f"    Relative difference: {relative_diff:.2e}")
            
            # Check if they match within reasonable tolerance
            if c_python_diff < 1e-12:
                print(f"    ✓ Direct C gives |Python|² exactly!")
            elif relative_diff < 1e-10:
                print(f"    ✓ Direct C gives |Python|² within numerical precision")
            else:
                print(f"    ✗ Direct C differs significantly from |Python|²")
                success = False
                
            # Show some sample values
            print(f"    Sample values comparison:")
            print(f"      |Python|²[0,0,0] = {python_squared[0,0,0]:.6e}")
            print(f"      C result[0,0,0] = {c_result[0,0,0]:.6e}")
            if python_squared.shape[1] > 1 and python_squared.shape[2] > 1:
                print(f"      |Python|²[0,1,1] = {python_squared[0,1,1]:.6e}") 
                print(f"      C result[0,1,1] = {c_result[0,1,1]:.6e}")
                
        elif fast_result is None:
            print(f"    ✗ Python implementation failed for triplet {triplet}")
            success = False
        elif c_result is None:
            print(f"    ✗ C implementation failed for triplet {triplet}")
            success = False
        else:
            print(f"    ✗ Both implementations failed for triplet {triplet}")
            success = False
        
        if not success:
            overall_success = False
    
    print("\n  " + "=" * 50)
    if overall_success:
        print("  🎉 All triplets: Direct C binding works correctly!")
    else:
        print("  ❌ Some triplets: Issues found with direct C binding")
    
    return overall_success


def test_direct_c_binding(num_atom: int = 2):
    """Test the new direct C binding.
    
    Args:
        num_atom: Number of atoms in the test system
    """
    
    print(f"Testing Direct C Binding for reciprocal_to_normal_squared ({num_atom} atoms)")
    print("=" * 75)
    print("This test calls the C function directly and compares with fast Python")
    print("Testing 4 different grid triplets: [0,0,0], [0,1,0], [1,0,0], [0,1,2]")
    print("=" * 75)
    
    if not HAS_C_EXTENSION:
        print("❌ Cannot run test - C extension not available")
        return False, 0, 0
    
    # Create test system
    test_data = create_test_system(num_atom)
    print(f"System: {test_data['num_atom']} atoms, {test_data['num_band']} bands")
    print(f"Grid triplets: {test_data['grid_triplets']}")
    print(f"FC3 array sizes: 6D={test_data['fc3_reciprocal_6d'].shape}, 3D={test_data['fc3_reciprocal_3d'].shape}")
    
    # Run implementations
    result_fast, time_fast = run_python_implementations(test_data)
    result_c, time_c = call_direct_c_binding(test_data)
    
    # Analyze results
    success = analyze_direct_c_results(None, result_fast, result_c)
    
    # Performance comparison
    print(f"\n  Performance Comparison (Total for all 4 triplets):")
    print(f"  Fast Python:     {time_fast:.4f} seconds")
    print(f"  Direct C:        {time_c:.4f} seconds")
    
    if time_fast > 0 and time_c > 0:
        speedup_c_vs_fast = time_fast / time_c if time_c > 0 else float('inf')
        print(f"  C vs Fast speedup: {speedup_c_vs_fast:.4f}x")
        print(f"  Average per triplet - Python: {time_fast/4:.4f}s, C: {time_c/4:.4f}s")
    
    # Final summary
    print("\n" + "=" * 75)
    if success:
        print("🎉 SUCCESS: Direct C binding works correctly for all 4 triplets!")
        print("✓ C function gives exact |Fast Python|² results for all triplets")
        print("✓ The relationship |Fast_Python_result|² = C_result is confirmed")
        print(f"✓ Tested triplets: {test_data['grid_triplets']}")
    else:
        print("❌ FAILURE: Issues found with direct C binding for some triplets")
        
    print("=" * 75)
    return success, time_fast, time_c


def test_multiple_system_sizes():
    """Test multiple system sizes to see scaling behavior."""
    
    sizes = [2, 4, 6, 12, 24, 50]
    results = {}
    
    print("🔬 SCALING TEST: Multiple System Sizes")
    print("=" * 75)
    
    for num_atom in sizes:
        print(f"\n🧪 Testing {num_atom}-atom system:")
        print("-" * 40)
        
        try:
            success, time_fast, time_c = test_direct_c_binding(num_atom)
            results[num_atom] = {
                'success': success,
                'time_fast': time_fast,
                'time_c': time_c,
                'num_band': num_atom * 3
            }
        except Exception as e:
            print(f"❌ Failed for {num_atom} atoms: {e}")
            results[num_atom] = {'success': False, 'error': str(e)}
    
    # Summary table
    print(f"\n📊 SCALING SUMMARY (Total time for all 4 triplets):")
    print("=" * 75)
    print(f"{'Atoms':<6} {'Bands':<6} {'Fast':<10} {'C':<10} {'Fast/C':<10} {'C/Fast':<10}")
    print("-" * 75)
    
    for num_atom in sizes:
        if results[num_atom].get('success', False):
            r = results[num_atom]
            speedup_c = r['time_fast'] / r['time_c'] if r['time_c'] > 0 else float('inf')
            speedup_fast_vs_c = r['time_c'] / r['time_fast'] if r['time_fast'] > 0 else float('inf')
            print(f"{num_atom:<6} {r['num_band']:<6} {r['time_fast']:<10.4f} {r['time_c']:<10.4f} {speedup_c:<10.4f} {speedup_fast_vs_c:<10.4f}")
        else:
            print(f"{num_atom:<6} {'FAIL':<6}")
    
    print("=" * 75)
    return results


if __name__ == "__main__":
    # Test with a larger system (8 atoms)
    test_multiple_system_sizes() 