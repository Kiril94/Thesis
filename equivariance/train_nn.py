from pathlib import Path
from tools import utils
from tools.utils import dotdict
import numpy as np
BASE_DIR = Path(__file__).resolve().parent.parent

num_networks = 1
classifiers = ['cnn']*num_networks

for i, classifier in zip(np.arange(num_networks), classifiers):
    epochs = 10
    batch_size = 100
    new_model = True
    gpu_on = False
    data_dir = f"{BASE_DIR}/data"
    output_dir = 'trained_models/simple'
    model_name = str(classifier) + f"_run_{i}"
    description = f""
    args_dict = {
        'verbose': True, 'description': description, 'build': new_model,
        'gpu': gpu_on
    }
    settings = dotdict({
        'args_dict': args_dict, 'classifier': classifier,
        'model_name': model_name, 'epochs': epochs,
        'batch_size': batch_size, 'output_dir': output_dir,
        'data_dir': data_dir})

    utils.train_pred(settings)