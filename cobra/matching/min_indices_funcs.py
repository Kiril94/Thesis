#%%
import matplotlib.pyplot as plt
import numpy as np
#%%
def compute_sum(D, A):
    return np.sum(D*A)
def create_A(D:np.array, indices:tuple):
    A_shape = D.shape
    A = np.zeros(A_shape)
    A[indices] = 1
    return A
 
D = np.array([[1.01, 1],
    [1, 2],
    [3, 2.1],
    [2.1, 3],
    [1, 4],
    ])
A = np.zeros((5,2))
A_indices_rows = [0,1,3,4]
A_indices_cols = [1,1,0,0]
A[A_indices_rows, A_indices_cols] = 1

def get_row_sort_inds(D:np.array):
    return np.argsort(D, axis=0)
def get_col_sort_inds(D:np.array):
    return np.argsort(D, axis=1)
def sort_rows(D:np.array):
    return np.sort(D, axis=0)
def sort_cols(D:np.array):
    return np.sort(D, axis=1)    
def get_Ainds():
    pass
k = 2
num_cols = 2
num_rows = 5
#print(compute_sum(D, A))
#print(D)
#print(get_row_ordering_indeces(D).shape)
#row_inds = get_row_sort_inds(D)
#print(np.take_along_axis(D, row_inds, axis=0))
#print(get_row_sort_inds(D).shape)
D_row_sort = np.sort(D, axis=0)
D_col_sort = np.sort(D, axis=1)

D_diff_row = D_row_sort - np.roll(D_row_sort, 1, axis=0)
D_diff_col = D_col_sort - np.roll(D_col_sort, 1, axis=1)

def create_A_inds(D: np.array):
    row_sort_inds = get_row_sort_inds(D).flatten('F')
    num_rows, num_cols = D.shape
    column_inds_list = np.repeat(np.arange(num_cols), num_rows)
    return row_sort_inds, column_inds_list

# create initial matrix
A0 = np.zeros(D.shape)
A_row_inds, A_col_inds = create_A_inds(D)

#print(A_row_inds)

#print(A_row_inds.reshape(-1,num_rows)[:,:k].reshape(-1))
def get_first_k_every_n(arr: np.array, n: int, k: int):
    return arr.reshape(-1,n)[:,:k].reshape(-1)
#print(get_first_k_every_n(A_row_inds, 5, 2))
#print(get_first_k_every_n(A_col_inds, 5, 2))
A0[get_first_k_every_n(A_row_inds, num_rows, k), 
    get_first_k_every_n(A_col_inds, num_rows, k)] = 1
print(A0)
#print(np.arange(0,2))
#print(np.repeat(np.arange(num_cols), num_rows))
column_indices_arr = np.repeat(np.arange(0,2)[:,np.newaxis], 5, axis=1).T
#print(row_inds)
#print(np.ndarray.flatten(row_inds))

#print(column_indices_arr.shape)
full_ind_arr = np.stack((row_inds, column_indices_arr), axis=0)
#print(full_ind_arr.shape)
#print(full_ind_arr)