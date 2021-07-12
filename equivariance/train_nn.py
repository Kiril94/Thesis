from pathlib import Path
from tools import utils
from tools.utils import dotdict
import numpy as np
BASE_DIR = Path(__file__).resolve().parent.parent

classifiers = ['cnn_new']*100

for i, classifier in zip(np.arange(100), classifiers):
    for data_id in [7]:
        epochs = 100
        batch_size = 100
        new_model = True
        output_dir = 'trained_models/feature_importance'
        model_name = str(classifier) + f"_{data_id}_" + f"bs_{batch_size}_run{i}"
        data_dir = f"{BASE_DIR}/Data/Kiril_tf_ready/d0{data_id}"
        weight = True
        description = f"{classifier}\n"\
                      + f"data_dir: {data_dir}\n"\
                      + '1 min time window\n'\
                      + f"batch size = {batch_size}\n"\
                      + f"weighted: {weight}\n"\
                     # + "filters: 3,6,12,24\n"\
                     # + "kernel size: 9\n"\
                     # + "pool size: 4,3,4,5\n"

        args_dict = {
            'verbose':True, 'description':description, 'build':new_model,
        }
        settings = dotdict({
            'args_dict':args_dict, 'classifier':classifier,
            'data_dir':data_dir, 'model_name':model_name,
            'epochs':epochs, 'batch_size':batch_size,
            'weight':weight, 'output_dir':output_dir})

        utils.train_pred(settings)