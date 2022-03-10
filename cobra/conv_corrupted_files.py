import os
from os.path import join
from pathlib import Path
import pandas as pd
from utilities import download
import pickle

script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
disk_dir = "F:"
dst_data_dir = f"{disk_dir}/CoBra/Data/dcm"
data_dir = join(base_dir, 'data')