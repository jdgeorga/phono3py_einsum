import numpy as np
import phono3py as Phono3py
import os

phono3py = Phono3py.load("phono3py_disp.yaml", log_level=2)
phono3py.log_level = 2
fc2_forces = "disp_forces_2nd_from_3rd_mace.npy"
fc3_forces = "disp_forces_3rd_mace.npy"
forces3 = np.load(fc3_forces)
forces2 = np.load(fc2_forces)
phono3py.forces = forces3
phono3py.phonon_forces = forces2
fc3_file = "fc3.npy"
fc2_file = "fc2.npy"
if not (os.path.exists(fc3_file) and os.path.exists(fc2_file)):
    phono3py.produce_fc3(symmetrize_fc3r=True)
    phono3py.produce_fc2(symmetrize_fc2=True)
    np.save("fc3.npy", phono3py.fc3, allow_pickle=True)
    np.save("fc2.npy", phono3py.fc2, allow_pickle=True)

phono3py.fc3 = np.load("fc3.npy", allow_pickle=True)
phono3py.fc2 = np.load("fc2.npy", allow_pickle=True)

phono3py.mesh_numbers = [36, 36, 1]  # You might want to adjust this

phono3py.init_phph_interaction()
interaction = phono3py.phph_interaction
interaction.run_phonon_solver_at_gamma()
max_phonon_freq = np.amax(interaction.get_phonons()[0])
interaction.run_phonon_solver_at_gamma(is_nac=True)
num_band0 = len(interaction.band_indices)
# print(phono3py._grid_points)
# quit()
grid_points = [0,3]
interaction.set_grid_point(grid_points[1])
interaction.run_phonon_solver()
# interaction._make_r0_average = False


import time
# interaction.fc3_reciprocal = None
# start_time = time.time()
# interaction.run(lang = "C")
# print(f"C interaction.run() took {time.time() - start_time:.4f} seconds")

# c_interaction_strength = interaction.interaction_strength
# c_fc3_reciprocal = interaction.fc3_reciprocal
# print("fc3_reciprocal shape:", c_fc3_reciprocal.shape if c_fc3_reciprocal is not None else None)
# print("fc3_reciprocal access test:", "SUCCESS" if c_fc3_reciprocal is not None else "FAILED")
# print("fc3_reciprocal first 5 elements:", c_fc3_reciprocal[0].flatten()[:5])

# interaction.fc3_reciprocal = None
# start_time = time.time()
# interaction.run(lang = "V3")
# v3_interaction_strength = interaction.interaction_strength
# v3_fc3_reciprocal = interaction.fc3_reciprocal
# print(f"V3 interaction.run() took {time.time() - start_time:.4f} seconds")
# print("fc3_reciprocal shape:", v3_fc3_reciprocal.shape if v3_fc3_reciprocal is not None else None)
# print("fc3_reciprocal access test:", "SUCCESS" if v3_fc3_reciprocal is not None else "FAILED")
# print("fc3_reciprocal first 5 elements:", v3_fc3_reciprocal[0].flatten()[:5])
# # print("c_interaction_strength:", c_interaction_strength)
# # print("fast_interaction_strength:", fast_interaction_strength)
# print("interaction_strength difference:", np.abs(c_interaction_strength - v3_interaction_strength).sum())
# print("fc3_reciprocal difference:", np.abs(c_fc3_reciprocal - v3_fc3_reciprocal).sum())

interaction.fc3_reciprocal = None
start_time = time.time() 
interaction.run(lang = "Fast", num_processes=64)
fast_interaction_strength = interaction.interaction_strength
fast_fc3_reciprocal = interaction.fc3_reciprocal
print(f"Fast interaction.run() took {time.time() - start_time:.4f} seconds")
print("fc3_reciprocal shape:", fast_fc3_reciprocal.shape if fast_fc3_reciprocal is not None else None)
print("fc3_reciprocal access test:", "SUCCESS" if fast_fc3_reciprocal is not None else "FAILED")
# print("fc3_reciprocal first 5 elements:", v3_fc3_reciprocal[0].flatten()[:5])
# print("c_interaction_strength:", c_interaction_strength)
# print("fast_interaction_strength:", fast_interaction_strength)
# print("interaction_strength difference:", np.abs(c_interaction_strength - fast_interaction_strength).sum())
# print("fc3_reciprocal difference:", np.abs(c_fc3_reciprocal - fast_fc3_reciprocal).sum())

# print("python compare: fc3_reciprocal difference:", np.abs(v3_fc3_reciprocal - fast_fc3_reciprocal).sum())
# print("python compare: interaction_strength difference:", np.abs(v3_interaction_strength - fast_interaction_strength).sum())

# interaction.run(lang = "Fast")
# fast_interaction_strength = interaction.interaction_strength
# fast_fc3_reciprocal = interaction.fc3_reciprocal
# print("fc3_reciprocal shape:", fast_fc3_reciprocal.shape if fast_fc3_reciprocal is not None else None)
# print("fc3_reciprocal access test:", "SUCCESS" if fast_fc3_reciprocal is not None else "FAILED")
# # print("c_interaction_strength:", c_interaction_strength)
# # print("python_interaction_strength:", python_interaction_strength)
# print("interaction_strength difference:", np.abs(c_interaction_strength - fast_interaction_strength).sum())
# print("interaction_strength difference:", np.abs(v3_interaction_strength - fast_interaction_strength).sum())
# print("fc3_reciprocal difference:", np.abs(c_fc3_reciprocal - fast_fc3_reciprocal).sum())
# print("fc3_reciprocal difference:", np.abs(v3_fc3_reciprocal - fast_fc3_reciprocal).sum())
# _, gamma = phono3py.run_imag_self_energy(
#     grid_points=[0,3],
#     temperatures=[300],
#     frequency_points_at_bands=True,
#     lang="C"
# )

# print(f"Gamma shape after parallel calculation: {gamma.shape}", flush=True)
# # Convert gamma to lifetimes
# gamma = gamma.squeeze()  # Remove singleton dimensions
# print(f"Gamma shape after squeeze: {gamma.shape}", flush=True)
# gamma = 2 * np.pi * gamma  # Convert to angular frequency
# gamma = np.where(gamma > 0, gamma, -1)  # Handle zero/negative values
# lifetime = np.where(gamma > 0, 1.0 / (2 * gamma), 0)  # Calculate lifetimes
# print(f"Lifetime shape: {lifetime.shape}", flush=True)
# print(lifetime)