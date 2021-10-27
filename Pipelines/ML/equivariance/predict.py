from pathlib import Path
from tools.utils import predict
import os
import numpy as np

if __name__=='__main__':
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_name = 'cnn_new_7_bs_100_run20'
    #model_name = 'tcn_multihead_7_bs_100_run42'
    model_dir = f"{BASE_DIR}/NN/trained_models/final_cnn_new_bootstrap/{model_name}"
    #model_dir = f"{BASE_DIR}/NN/trained_models/final_tcn_multihead_small_kernel_bootstrap/{model_name}"
    output_dir = f"{BASE_DIR}/NN/results/nastja/{model_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = f"{BASE_DIR}/Data/Nastja_tf_ready/d02"
    f1, y_true, y_pred = predict(model_dir, data_dir, output_dir)
    np.savetxt(f"{output_dir}/true_pred.txt", np.stack((y_true, y_pred), axis=-1).astype(int),
                fmt="%i", header='true predicted')
    np.savetxt(f"{output_dir}/f1.txt", np.array([f1]))

    #print(np.load(f"{data_dir}/X_test.npy").shape)
    #print(len(np.load(f"{data_dir}/Y_test.npy").shape))