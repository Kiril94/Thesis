import matplotlib.pyplot as plt
import math
import numpy as np


def display3d(arr3d, figsize=(15, 15), start_slice=0, num_slices=None, axis=0):
    """Input: 3D array of shape (num_slices, h, w)
    Returns: fig, ax
    Displays all the slices of a 3d array. 
    You can also display the slices along axis 1 or 2 by setting the axis argument."""
    
    if num_slices==None:
        num_slices = len(arr3d)
    if num_slices>50:
        num_slices = 50
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
        a.imshow(arr3d[i+start_slice])
        a.axis('off')
        a.set_title(f"slice = {i+start_slice}")
    fig.tight_layout()
    return fig, ax
