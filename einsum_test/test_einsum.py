import numpy as np
import opt_einsum as oe
# i, j, k -> atom indices κ κ' κ''
# l, m, n -> cartesian indices α β γ
# I, J, K -> band indices
einsum_string = "Ii,Jj,Kk,ijk-> IJK"
unique_inds = set(einsum_string) - {',', '-', '>'}
list_of_indices = ["i", "j", "k", "I", "J", "K"]
index_size = [42, 42, 42, 42, 42, 42]
sizes_dict = dict(zip(list_of_indices, index_size))
print(sizes_dict)
# Create dummy arrays manually instead of using the non-existent build_views
# Parse the einsum string to determine the shape of each array
operands = einsum_string.split('->')[0].split(',')
views = []

for operand in operands:
    shape = tuple(sizes_dict[idx] for idx in operand.strip())
    views.append(np.random.rand(*shape))

# Default path (no memory limit)
path, path_info = oe.contract_path(einsum_string, *views)
print("\nDEFAULT PATH (no memory limit):")
print(path)
print(path_info)

# Calculate memory usage of largest intermediate
largest_intermediate = path_info.largest_intermediate
memory_mb = largest_intermediate * 8 / (1024**2)  # 8 bytes per float64, convert to MB
print(f"\nLargest intermediate: {largest_intermediate:.3e} elements = {memory_mb:.2f} MB")

# Now let's try with different memory limits
print("\n" + "="*80)
print("TESTING DIFFERENT MEMORY LIMITS:")
print("="*80)

# Memory limit: 1 MB (very restrictive)
memory_limit_mb = 1
memory_limit_elements = memory_limit_mb * (1024**2) / 8  # Convert MB to number of elements
print(f"\nTrying with memory limit: {memory_limit_mb} MB ({memory_limit_elements:.0f} elements)")

try:
    path_limited, path_info_limited = oe.contract_path(einsum_string, *views, memory_limit=memory_limit_elements)
    print("SUCCESS with 1 MB limit:")
    print(f"Largest intermediate: {path_info_limited.largest_intermediate:.3e} elements = {path_info_limited.largest_intermediate * 8 / (1024**2):.2f} MB")
    print(f"FLOP count: {path_info_limited.opt_cost:.3e}")
except Exception as e:
    print(f"FAILED with 1 MB limit: {e}")

# Memory limit: 10 MB
memory_limit_mb = 10
memory_limit_elements = memory_limit_mb * (1024**2) / 8
print(f"\nTrying with memory limit: {memory_limit_mb} MB ({memory_limit_elements:.0f} elements)")

try:
    path_limited, path_info_limited = oe.contract_path(einsum_string, *views, memory_limit=memory_limit_elements)
    print("SUCCESS with 10 MB limit:")
    print(f"Largest intermediate: {path_info_limited.largest_intermediate:.3e} elements = {path_info_limited.largest_intermediate * 8 / (1024**2):.2f} MB")
    print(f"FLOP count: {path_info_limited.opt_cost:.3e}")
except Exception as e:
    print(f"FAILED with 10 MB limit: {e}")

# Memory limit: -1 (no limit)
print(f"\nTrying with memory limit: -1 (no limit)")
path_unlimited, path_info_unlimited = oe.contract_path(einsum_string, *views, memory_limit=-1)
print("SUCCESS with no memory limit:")
print(f"Largest intermediate: {path_info_unlimited.largest_intermediate:.3e} elements = {path_info_unlimited.largest_intermediate * 8 / (1024**2):.2f} MB")
print(f"FLOP count: {path_info_unlimited.opt_cost:.3e}")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print("Memory limits in opt_einsum:")
print("- Set memory_limit parameter in contract_path() or contract()")
print("- Units are in number of elements (not bytes)")
print("- memory_limit=-1 means no limit")
print("- Tighter limits may increase FLOP count but reduce memory usage")
print("- Very tight limits may cause the optimizer to fail")

