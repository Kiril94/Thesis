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
    output_dir = 'trained_models/simple'
    model_name = str(classifier) + f"_run_{i}"
    description = f""
    args_dict = {
        'verbose':True, 'description':description, 'build':new_model,
    }
    settings = dotdict({
        'args_dict':args_dict, 'classifier':classifier,
        'data_dir':data_dir, 'model_name':model_name,
        'epochs':epochs, 'batch_size':batch_size,
        'weight':weight, 'output_dir':output_dir})

    utils.train_pred(settings)