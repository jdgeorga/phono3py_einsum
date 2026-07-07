import numpy as np

contrib1 = np.load("contrib1.npy")
contrib2 = np.load("contrib2.npy")
contrib3 = np.load("contrib3.npy")

fc3_rec_elem1 = np.load("fc3_rec_elem1.npy")
fc3_rec_elem2 = np.load("fc3_rec_elem2.npy")
fc3_rec_elem3 = np.load("fc3_rec_elem3.npy")

# Compare
print("Compare contrib1 and fc3_rec_elem1")
print(np.abs(contrib1 - fc3_rec_elem1).sum())

print("Compare contrib2 and fc3_rec_elem2")
print(np.abs(contrib2 - fc3_rec_elem2).sum())

print("Compare contrib3 and fc3_rec_elem3")
print(np.abs(contrib3 - fc3_rec_elem3).sum())

quit()