import numpy as np
import time

size=32

basis = []
for k1 in [-1, 0, 1]:
    for k2 in [-1, 0, 1]:
        b = np.zeros((size**2, size**2))
        for i in range(size):
            for j in range(size):
                if i+k1>= 0 and i+k1<size and j+k2 >=0 and j+k2<size:
                    b[i + size*j][(i + k1) + size*(j + k2)] = 1.
        basis.append(b)

# Precompute B matrix for many depths
now = time.time()
max_depth = 20000
depths = [d for d in range(max_depth) if (d < 1000 and d%10 == 0) or(d < 5000 and d % 100 == 0) or d % 500 == 0]
matrices = []
B_new = np.ones((size**2, size**2))
# B_new = np.eye(size**2)
for depth in range(max_depth):
    B_new = B_new/np.linalg.norm(B_new)
    # save B_new
    if depth in depths:
        print(depth, time.time() - now)
        matrices.append(B_new)
    B = B_new
    B_new = np.zeros((size**2, size**2))
    for b in basis:
        B_new += b @ B @ b.T
matrices = np.array(matrices)
depths = np.array(depths)

file_name = 'precomputed_vals_{}.npy'.format(size)
with open(file_name, 'wb') as f:
    np.save(f, depths)
    np.save(f, matrices)
    