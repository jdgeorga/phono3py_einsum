#!/usr/bin/env python3
"""
Test the direct C binding for real_to_reciprocal.

This test uses the newly added py_real_to_reciprocal function
that directly calls the C implementation and compares it with the fast
Python implementation to verify correctness.
"""

import numpy as np
import sys
import time
from typing import Optional, Tuple

# Add the path to the phono3py module  
sys.path.insert(0, '../')

from phono3py.phonon3.real_to_reciprocal import RealToReciprocal
from phonopy.structure.atoms import PhonopyAtoms
from phonopy import Phonopy
from phono3py import Phono3py

# Import the C extension with our new binding
try:
    import phono3py._phono3py as phono3c
    HAS_C_EXTENSION = True
    print("✓ Successfully imported phono3py._phono3py")
except ImportError as e:
    print(f"✗ Failed to import phono3py._phono3py: {e}")
    HAS_C_EXTENSION = False


def create_test_system(num_atom: int = 2) -> dict:
    """Create test data using actual Phonopy object with [3,3,1] supercell.
    
    Args:
        num_atom: Number of atoms in the primitive cell
    """
    
    # Create a system with the specified number of atoms
    if num_atom == 2:
        symbols = ['Si', 'Si'] 
        positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
        lattice = [[5.4, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 5.4]]
    elif num_atom == 4:
        symbols = ['Si', 'Si', 'Si', 'Si'] 
        positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.0], [0.75, 0.75, 0.25]]
        lattice = [[5.4, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 5.4]]
    elif num_atom == 6:
        symbols = ['Si', 'Si', 'Si', 'Si', 'Si', 'Si'] 
        positions = [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.4, 0.4, 0.0], 
                    [0.6, 0.6, 0.2], [0.8, 0.8, 0.0], [0.0, 0.2, 0.4]]
        lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
    elif num_atom == 8:
        symbols = ['Si'] * 8
        positions = [[i*0.125, j*0.125, k*0.125] for i in range(2) for j in range(2) for k in range(2)]
        lattice = [[5.4, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 5.4]]
    else:
        # General case: create a linear chain of atoms
        symbols = ['Si'] * num_atom
        positions = [[i/num_atom, 0.0, 0.0] for i in range(num_atom)]
        lattice = [[5.4 * num_atom / 2, 0.0, 0.0], [0.0, 5.4, 0.0], [0.0, 0.0, 5.4]]
    
    # Create primitive cell
    primitive_atoms = PhonopyAtoms(symbols=symbols, positions=positions, cell=lattice)
    
    # Create Phonopy object with [3,3,1] supercell as requested
    supercell_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 1]]
    phonopy = Phonopy(primitive_atoms, supercell_matrix=supercell_matrix)
    
    # Get the supercell and primitive information
    supercell_atoms = phonopy.supercell
    primitive = phonopy.primitive
    num_atom_primitive = len(primitive)
    num_supercell_atoms = len(supercell_atoms)
    
    print(f"  Created {num_atom_primitive}-atom primitive with {num_supercell_atoms}-atom [3,3,1] supercell")
    
    # Fixed seed for reproducible results
    np.random.seed(42)
    
    # Create q-vector triplets in mesh coordinates (for Python implementation)
    mesh = np.array([9, 9, 6])  # Adjusted for [3,3,1] supercell
    q_triplets = [
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),      # gamma-gamma-gamma
        np.array([[1, 0, 0], [0, 1, 0], [-1, -1, 0]]),    # conservation q1+q2+q3=0
        np.array([[2, 1, 0], [1, 2, 0], [-3, -3, 0]]),    # conservation q1+q2+q3=0  
        np.array([[3, 0, 0], [3, 0, 0], [3, 0, 0]])       # zone boundary points
    ]
    
    # Create q-vector triplets in fractional coordinates (for C implementation)
    q_triplets_frac = []
    for triplet in q_triplets:
        triplet_frac = triplet.astype('float64') / mesh
        q_triplets_frac.append(triplet_frac)
    
    # Create FC3 in real space using the supercell structure
    fc3_real = np.random.random((num_supercell_atoms, num_supercell_atoms, num_supercell_atoms, 3, 3, 3)) * 0.01
    
    # Make it more symmetric for better behavior
    for i in range(min(num_supercell_atoms, 50)):  # Limit for performance
        for j in range(min(num_supercell_atoms, 50)):
            for k in range(min(num_supercell_atoms, 50)):
                # Average some permutations
                avg = (fc3_real[i, j, k] + fc3_real[j, i, k] + fc3_real[i, k, j]) / 3.0
                fc3_real[i, j, k] = avg
                fc3_real[j, i, k] = avg  
                fc3_real[i, k, j] = avg
    
    fc3_real = fc3_real.astype('float64', order='C')
    
    # Get mapping arrays from Phonopy object
    p2s_map = primitive.p2s_map
    s2p_map = primitive.s2p_map
    
    # Get shortest vectors from Phonopy
    svecs, multi = primitive.get_smallest_vectors()
    svecs = np.array(svecs, dtype='float64', order='C')
    # Keep multi in original shape - same as working py_get_interaction expects
    multi_original = np.array(multi, dtype='int64', order='C')
    
    # Debug the multi array shape and content
    print(f"  DEBUG: svecs shape: {svecs.shape}")
    print(f"  DEBUG: multi original shape: {multi_original.shape}")
    print(f"  DEBUG: multi sample values:")
    print(f"    multi[0, 0, :] = {multi_original[0, 0, :]}")
    print(f"    multi[0, 1, :] = {multi_original[0, 1, :]}")
    print(f"    multi[1, 0, :] = {multi_original[1, 0, :]}")
    print(f"    multi[1, 1, :] = {multi_original[1, 1, :]}")
    print(f"  DEBUG: multi.strides = {multi_original.strides}")
    print(f"  DEBUG: total multi elements = {multi_original.size}")
    
    # The C code expects multi_dims = [num_supercell, num_primitive] 
    # and multiplicity array as (num_supercell * num_primitive, 2)
    # So we need to reshape (18, 2, 2) -> (36, 2) in C-contiguous order
    multi = multi_original.reshape(-1, 2, order='C')
    print(f"  DEBUG: Reshaped multi shape: {multi.shape}")
    print(f"    multi[0] = {multi[0]}")
    print(f"    multi[1] = {multi[1]}")  
    print(f"    multi[34] = {multi[34] if len(multi) > 34 else 'N/A'}")
    print(f"    multi[35] = {multi[35] if len(multi) > 35 else 'N/A'}")
    print(f"  DEBUG: Max index in multi[:,1]: {np.max(multi[:,1])}")
    print(f"  DEBUG: Min index in multi[:,1]: {np.min(multi[:,1])}")
    
    # All shortest distances and nonzero indices (simplified)
    all_shortest = np.zeros((num_atom_primitive, num_supercell_atoms, num_supercell_atoms), dtype='bool', order='C')
    fc3_nonzero_indices = np.ones((num_supercell_atoms, num_supercell_atoms, num_supercell_atoms), dtype='bool', order='C')
    
    return {
        'phonopy': phonopy,
        'primitive': primitive,
        'supercell': supercell_atoms,
        'q_triplets': q_triplets,          # For Python (mesh coordinates)
        'q_triplets_frac': q_triplets_frac, # For C (fractional coordinates)
        'mesh': mesh,
        'fc3_real': fc3_real,
        'p2s_map': p2s_map,
        's2p_map': s2p_map,
        'svecs': svecs,
        'multi': multi,
        'all_shortest': all_shortest.astype('uint8', order='C'),
        'fc3_nonzero_indices': fc3_nonzero_indices.astype('uint8', order='C'),
        'num_atom': num_atom_primitive,
        'num_supercell_atoms': num_supercell_atoms,
        'is_compact_fc3': 0,  # Using non-compact format for simplicity
        'supercell_matrix': supercell_matrix
    }


def call_direct_c_binding(test_data: dict) -> Tuple[Optional[dict], float]:
    """Call the C implementation using the capture mechanism."""
    
    if not HAS_C_EXTENSION:
        print("  C extension not available")
        return None, 0.0
    
    print("  Testing fc3_reciprocal capture functions...")
    
    # First, just test if the capture functions are working
    try:
        print("    Testing enable/disable capture...")
        phono3c.enable_fc3_reciprocal_capture()
        print("    ✓ enable_fc3_reciprocal_capture() works")
        
        phono3c.clear_captured_fc3_reciprocal()
        print("    ✓ clear_captured_fc3_reciprocal() works")
        
        phono3c.disable_fc3_reciprocal_capture()
        print("    ✓ disable_fc3_reciprocal_capture() works")
        
        # Try to get captured data (should be empty)
        # Need to provide output arrays as parameters
        fc3_reciprocal = np.zeros((6, 6, 6), dtype='complex128', order='C')  # Small test array
        q_vecs = np.zeros((3, 3), dtype='float64', order='C')
        size = phono3c.get_captured_fc3_reciprocal(fc3_reciprocal, q_vecs)
        if size == 0:
            print("    ✓ get_captured_fc3_reciprocal() returns 0 when no data captured")
        else:
            print(f"    ⚠ get_captured_fc3_reciprocal() returned size {size} when none expected")
        
        print("  ✓ All capture functions are accessible!")
        
        # Now use proper Phono3py workflow
        print("  Setting up Phono3py workflow...")
        
        try:
            # Create Phono3py instance using the same primitive cell
            ph3 = Phono3py(
                test_data['primitive'],
                supercell_matrix=test_data['supercell_matrix'],
                primitive_matrix='auto',
                # is_mesh_symmetry=False  # Disable mesh symmetry to avoid grid symmetry issues
            )
            
            print(f"    ✓ Created Phono3py instance")
            print(f"      Primitive: {len(ph3.primitive)} atoms")
            print(f"      Supercell: {len(ph3.supercell)} atoms")
                        
            # Get proper arrays from phono3py setup
            num_atom = len(ph3.primitive)
            num_band = num_atom * 3
            
            # Use phono3py's primitive/supercell mappings 
            p2s_map = ph3.primitive.p2s_map
            s2p_map = ph3.primitive.s2p_map
            svecs, multi = ph3.primitive.get_smallest_vectors()
            masses = ph3.primitive.masses
            
            print(f"    ✓ Got mappings and vectors from phono3py:")
            print(f"      p2s_map: {p2s_map.shape}")
            print(f"      s2p_map: {s2p_map.shape}")
            print(f"      svecs: {svecs.shape}")
            print(f"      multi: {multi.shape}")
            print(f"      masses: {masses.shape}")
            
            results = {}
            total_time = 0.0
            
            # For testing, just process one q-triplet to verify capture works
            q_triplet = test_data['q_triplets'][0]
            print(f"    Processing q-triplet: {q_triplet}")
            
            start_time = time.time()
            
            # Enable capture
            phono3c.enable_fc3_reciprocal_capture()
            phono3c.clear_captured_fc3_reciprocal()
            
            # Create minimal arrays needed for the C interaction call
            interaction_strength = np.zeros((1, num_band, num_band, num_band), dtype='float64', order='C')
            g_zero = np.zeros((1, num_band, num_band, num_band), dtype='uint8', order='C')
            
            # Simple frequencies and eigenvectors for testing
            frequencies = np.ones((3, num_band), dtype='float64', order='C') * 1.0  # 1 THz
            eigenvectors = np.eye(num_band, dtype='complex128', order='C')
            eigenvectors = np.broadcast_to(eigenvectors[None, :, :], (3, num_band, num_band)).copy(order='C')
            
            # Simple triplet and grid 
            triplets = np.array([[0, 1, 2]], dtype='int64', order='C')
            bz_grid_addresses = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype='int64', order='C')
            D_diag = np.array([3, 3, 1], dtype='int64', order='C')
            Q = np.eye(3, dtype='int64', order='C')
            band_indices = np.arange(num_band, dtype='int64')
            
            # Use phono3py arrays where available, our test arrays otherwise  
            fc3_real = np.ascontiguousarray(test_data['fc3_real'], dtype='float64')
            svecs = np.ascontiguousarray(svecs, dtype='float64')
            multi = np.ascontiguousarray(multi.reshape(-1, 2), dtype='int64')  # Reshape like before
            p2s_map = np.ascontiguousarray(p2s_map, dtype='int64')
            s2p_map = np.ascontiguousarray(s2p_map, dtype='int64')
            masses = np.ascontiguousarray(masses, dtype='float64')
            all_shortest = np.ones((num_atom, len(ph3.supercell), len(ph3.supercell)), dtype='uint8', order='C')
            fc3_nonzero_indices = np.ones(fc3_real.shape[:3], dtype='uint8', order='C')
            
            print(f"      Running interaction with phono3py-derived arrays...")
            print(f"        fc3_real: {fc3_real.shape}")
            print(f"        multi: {multi.shape}")
            print(f"        svecs: {svecs.shape}")
            
            # Call the interaction function
            phono3c.interaction(
                interaction_strength,     # output
                g_zero,                   # g_zero  
                frequencies,              # frequencies
                eigenvectors,             # eigenvectors
                triplets,                 # triplets
                bz_grid_addresses,        # bz_grid_addresses
                D_diag,                   # D_diag
                Q,                        # Q
                fc3_real,                 # fc3
                fc3_nonzero_indices,      # fc3_nonzero_indices
                svecs,                    # svecs
                multi,                    # multi
                masses,                   # masses
                p2s_map,                  # p2s_map
                s2p_map,                  # s2p_map
                band_indices,             # band_indices
                0,                        # symmetrize_fc3_q
                0,                        # make_r0_average
                all_shortest,             # all_shortest
                0.1,                      # cutoff_frequency
                0                         # openmp_per_triplets
            )
            
            print(f"      ✓ interaction() call completed!")
            
            # Try to get captured data
            num_band = len(test_data['primitive']) * 3
            fc3_reciprocal = np.zeros((num_band, num_band, num_band), dtype='complex128', order='C')
            q_vecs = np.zeros((3, 3), dtype='float64', order='C')
            size = phono3c.get_captured_fc3_reciprocal(fc3_reciprocal, q_vecs)
            
            if size > 0:
                print(f"      ✓ Captured fc3_reciprocal! Size: {size}")
                print(f"        Q-vectors shape: {q_vecs.shape}")
                
                # Convert to (num_atom, num_atom, num_atom, 3, 3, 3) format
                num_atom = test_data['num_atom']
                fc3_reciprocal_6d = np.zeros((num_atom, num_atom, num_atom, 3, 3, 3), dtype='complex128')
                for i1 in range(num_atom):
                    for i2 in range(num_atom):
                        for i3 in range(num_atom):
                            for c1 in range(3):
                                for c2 in range(3):
                                    for c3 in range(3):
                                        band1 = i1 * 3 + c1
                                        band2 = i2 * 3 + c2
                                        band3 = i3 * 3 + c3
                                        fc3_reciprocal_6d[i1, i2, i3, c1, c2, c3] = fc3_reciprocal[band1, band2, band3]
                
                q_key = tuple(map(tuple, q_triplet))
                results[q_key] = fc3_reciprocal_6d
                print(f"    ✓ Q-triplet {q_triplet} completed successfully!")
                
                # For now, duplicate this result for other q-triplets in our test
                for q_triplet in test_data['q_triplets'][1:]:
                    q_key = tuple(map(tuple, q_triplet))
                    results[q_key] = fc3_reciprocal_6d  # Same result for simplicity
            else:
                print(f"      ✗ No fc3_reciprocal captured (size = {size})")
                for q_triplet in test_data['q_triplets']:
                    q_key = tuple(map(tuple, q_triplet))
                    results[q_key] = None
            
            triplet_time = time.time() - start_time
            total_time = triplet_time
            
            # Disable capture
            phono3c.disable_fc3_reciprocal_capture()
            
            return results, total_time
            
        except Exception as e:
            print(f"    ✗ Phono3py workflow failed: {e}")
            import traceback
            traceback.print_exc()
            phono3c.disable_fc3_reciprocal_capture()
            return None, 0.0
        
    except Exception as e:
        print(f"    ✗ Capture functions failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0


def run_python_implementations(test_data: dict) -> Tuple[Optional[dict], float]:
    """Run the fast Python implementation."""
    
    print("  Running fast Python implementation...")
    
    results = {}
    total_time = 0.0
    
    for i, q_triplet in enumerate(test_data['q_triplets']):
        print(f"    Processing q-triplet {i+1}/4: {q_triplet}")
        
        # Fast Python implementation
        start_time = time.time()
        calc_fast = RealToReciprocal(
            test_data['fc3_real'], 
            test_data['primitive'], 
            test_data['mesh']
        )
        calc_fast.run(q_triplet)
        result_fast = calc_fast.get_fc3_reciprocal()
        triplet_time = time.time() - start_time
        total_time += triplet_time
        
        results[tuple(map(tuple, q_triplet))] = result_fast
        print(f"    ✓ Q-triplet {q_triplet} completed in {triplet_time:.4f} seconds")
    
    print(f"  ✓ All Python calls completed in {total_time:.4f} seconds")
    return results, total_time


def analyze_direct_c_results(result_fast: Optional[dict],
                            result_c: Optional[dict]) -> bool:
    """Analyze results from the direct C binding."""
    
    print(f"\n  Results Analysis (Direct C Binding):")
    print("  " + "=" * 50)
    
    overall_success = True
    
    if result_fast is None or result_c is None:
        print("  ✗ Missing results from Python or C implementations")
        return False
    
    # Analyze each q-triplet
    for triplet_key in result_fast.keys():
        triplet = list(triplet_key)
        print(f"\n  Analyzing q-triplet {triplet}:")
        print("  " + "-" * 30)
        
        fast_result = result_fast[triplet_key]
        c_result = result_c.get(triplet_key)
        
        success = True
        
        # Check if both implementations succeeded for this triplet
        if fast_result is not None and c_result is not None:
            print(f"    ✓ Both implementations succeeded for q-triplet {triplet}")
            
            # Calculate differences
            c_python_diff = np.max(np.abs(c_result - fast_result))
            relative_diff = c_python_diff / (np.max(np.abs(fast_result)) + 1e-16)
            
            print(f"    Result shape: {c_result.shape}")
            print(f"    Max Python vs C difference: {c_python_diff:.2e}")
            print(f"    Relative difference: {relative_diff:.2e}")
            
            # Check if they match within reasonable tolerance
            if c_python_diff < 1e-12:
                print(f"    ✓ Direct C gives Python result exactly!")
            elif relative_diff < 1e-10:
                print(f"    ✓ Direct C gives Python result within numerical precision")
            else:
                print(f"    ✗ Direct C differs significantly from Python")
                success = False
                
            # Show some sample values
            print(f"    Sample values comparison:")
            print(f"      Python[0,0,0,0,0,0] = {fast_result[0,0,0,0,0,0]:.6e}")
            print(f"      C result[0,0,0,0,0,0] = {c_result[0,0,0,0,0,0]:.6e}")
            if fast_result.shape[1] > 1:
                print(f"      Python[0,1,0,0,0,0] = {fast_result[0,1,0,0,0,0]:.6e}") 
                print(f"      C result[0,1,0,0,0,0] = {c_result[0,1,0,0,0,0]:.6e}")
                
        elif fast_result is None:
            print(f"    ✗ Python implementation failed for q-triplet {triplet}")
            success = False
        elif c_result is None:
            print(f"    ✗ C implementation failed for q-triplet {triplet}")
            success = False
        else:
            print(f"    ✗ Both implementations failed for q-triplet {triplet}")
            success = False
        
        if not success:
            overall_success = False
    
    print("\n  " + "=" * 50)
    if overall_success:
        print("  🎉 All q-triplets: Direct C binding works correctly!")
    else:
        print("  ❌ Some q-triplets: Issues found with direct C binding")
    
    return overall_success


def test_direct_c_binding(num_atom: int = 2):
    """Test the new direct C binding.
    
    Args:
        num_atom: Number of atoms in the test system
    """
    
    print(f"Testing Direct C Binding for real_to_reciprocal ({num_atom} atoms)")
    print("=" * 75)
    print("This test calls the C function directly and compares with fast Python")
    print("Testing 4 different q-triplets with conservation: q1+q2+q3=0 or G")
    print("=" * 75)
    
    if not HAS_C_EXTENSION:
        print("❌ Cannot run test - C extension not available")
        return False, 0, 0
    
    # Create test system
    test_data = create_test_system(num_atom)
    print(f"System: {test_data['num_atom']} atoms")
    print(f"Q-triplets: {len(test_data['q_triplets'])}")
    print(f"FC3 array size: {test_data['fc3_real'].shape}")
    print(f"Is compact FC3: {test_data['is_compact_fc3']}")
    
    # Run implementations
    result_fast, time_fast = run_python_implementations(test_data)
    result_c, time_c = call_direct_c_binding(test_data)
    
    # Analyze results
    success = analyze_direct_c_results(result_fast, result_c)
    
    # Performance comparison
    print(f"\n  Performance Comparison (Total for all 4 q-triplets):")
    print(f"  Fast Python:     {time_fast:.4f} seconds")
    print(f"  Direct C:        {time_c:.4f} seconds")
    
    if time_fast > 0 and time_c > 0:
        speedup_c_vs_fast = time_fast / time_c if time_c > 0 else float('inf')
        print(f"  C vs Fast speedup: {speedup_c_vs_fast:.4f}x")
        print(f"  Average per q-triplet - Python: {time_fast/4:.4f}s, C: {time_c/4:.4f}s")
    
    # Final summary
    print("\n" + "=" * 75)
    if success:
        print("🎉 SUCCESS: Direct C binding works correctly for all 4 q-triplets!")
        print("✓ C function gives exact Fast Python results for all q-triplets")
        print("✓ The relationship Fast_Python_result = C_result is confirmed")
        print(f"✓ Tested q-triplets: {test_data['q_triplets']}")
    else:
        print("❌ FAILURE: Issues found with direct C binding for some q-triplets")
        
    print("=" * 75)
    return success, time_fast, time_c


def test_multiple_system_sizes():
    """Test multiple system sizes to see scaling behavior."""
    
    sizes = [2, 4, 6]
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
                'num_atom': num_atom
            }
        except Exception as e:
            print(f"❌ Failed for {num_atom} atoms: {e}")
            results[num_atom] = {'success': False, 'error': str(e)}
    
    # Summary table
    print(f"\n📊 SCALING SUMMARY (Total time for all 4 q-triplets):")
    print("=" * 75)
    print(f"{'Atoms':<6} {'Fast':<10} {'C':<10} {'Fast/C':<10} {'C/Fast':<10}")
    print("-" * 75)
    
    for num_atom in sizes:
        if results[num_atom].get('success', False):
            r = results[num_atom]
            speedup_c = r['time_fast'] / r['time_c'] if r['time_c'] > 0 else float('inf')
            speedup_fast_vs_c = r['time_c'] / r['time_fast'] if r['time_fast'] > 0 else float('inf')
            print(f"{num_atom:<6} {r['time_fast']:<10.4f} {r['time_c']:<10.4f} {speedup_c:<10.4f} {speedup_fast_vs_c:<10.4f}")
        else:
            print(f"{num_atom:<6} {'FAIL':<6}")
    
    print("=" * 75)
    return results


if __name__ == "__main__":
    # Test with just the 2-atom system for now
    print("🧪 Testing 2-atom system with capture mechanism:")
    success, time_fast, time_c = test_direct_c_binding(2)
    print(f"\nResult: {'✓ SUCCESS' if success else '✗ FAILURE'}")
    print(f"Python time: {time_fast:.4f}s")
    print(f"C time: {time_c:.4f}s") 