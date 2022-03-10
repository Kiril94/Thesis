
import numpy as np
from numpy import linalg
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
# distance matrix
D = np.array([
                [0, 2, 2, 2],
                [2, 0, np.sqrt(8), np.sqrt(8)],
                [2, np.sqrt(8), 0, 4],
                [2, np.sqrt(8), 4, 0]
                ])
d = len(D)
B = np.zeros_like(D)
for j in np.arange(d):
    for k in np.arange(j, d):
        B[j,k] = 1/2*(D[0,j]**2 + D[0,k]**2 - D[j,k]**2 )


i_lower = np.tril_indices(d, -1)
B[i_lower] = B.T[i_lower]

print(linalg.matrix_rank(B))
w, U = linalg.eig(B)                
V = np.eye(d)*w
A = np.round(np.sqrt(V)@U.T, 8)
print(A)


v0, v1, v2, v3 = A[:,0], A[:,1], A[:,2], A[:,3]

print('d01=',linalg.norm(v0-v1), 'true', 2)
print('d02=',linalg.norm(v0-v2), 'true', 2)
print('d12=',linalg.norm(v1-v2), 'true', np.sqrt(8))
print('d13=',linalg.norm(v2-v3), 'true', 4)