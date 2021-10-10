import matplotlib.pyplot as plt
import math
import numpy as np
from accaccess_sif_data import load_data_tools as ld
import glob


def display3d(arr3d, figsize=(15, 15), start_slice=0, num_slices=None, axis=0):
    """Input: 3D array of shape (num_slices, h, w)
    Returns: fig, ax
    Displays all the slices of a 3d array. 
    You can also display the slices along axis 1 or 2 by setting the axis argument."""
    
    if num_slices==None:
        num_slices = len(arr3d)
    if num_slices>50:
        num_slices = 50
    data_shape = arr3d.shape
    if data_shape[axis]<(start_slice+num_slices):
        print("The data has only {data_shape[axis]} slices.")
        start_slice = 0
        num_slices = data_shape[axis]
    cols = 5
    rows = math.floor(num_slices/cols)
    figsize = (figsize[0], math.floor(rows/cols*figsize[0]))
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    ax = ax.flatten()
    plt.axis('off')
    if axis==1:
        arr3d = np.transpose(arr3d, [1,0,2])
    elif axis==2:
        arr3d = np.transpose(arr3d, [2,0,1])
    else:
        pass
    for i, a in enumerate(ax):
        a.imshow(arr3d[i+start_slice], cmap='gray')
        a.axis('off')
        a.set_title(f"slice = {i+start_slice}")
    fig.tight_layout()
    return fig, ax

def show_series(patient_id, series_id, base_data_dir='Y:'):
    """Shows the series for a series_id"""
    counter = 0
    for series_dir in glob.iglob(f"{base_data_dir}/*/{patient_id}/*/*/{series_id}"):
        if counter>0:
            break
        arr3d = ld.reconstruct3d(series_dir)
        display3d(arr3d)
        counter += 1
        

        
        
        
        
        
        
        
        
        
        
        