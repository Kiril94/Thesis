from pathlib import Path
from tools import utils
from tools.utils import dotdict
import numpy as np
BASE_DIR = Path(__file__).resolve().parent.parent

num_networks = 1
classifiers = ['cnn']*num_networks

for i, classifier in zip(np.arange(num_networks), classifiers):
    epochs = 100
    batch_size = 1000
    new_model = True
    gpu_on = True
    data_dir = f"{BASE_DIR}/equivariance/data"
    output_dir = 'trained_models/simple'
    model_name = str(classifier) + f"_run_{i}"
    description = f""
    args_dict = {
        'verbose': True, 'description': description, 'build': new_model,
    }

    settings = dotdict({
        'args_dict': args_dict, 'classifier': classifier,
        'model_name': model_name, 'epochs': epochs,
        'batch_size': batch_size, 'output_dir': output_dir,
        'data_dir': data_dir, 'gpu_on': gpu_on})

    utils.train_pred(settings)