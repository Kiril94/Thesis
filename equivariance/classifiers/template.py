# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from NN.tools.utils import save_logs
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tensorflow.keras.utils import plot_model
from NN.tools.utils import VarImpVIANN


class Classifier_Template:

    def __init__(self, output_directory, input_shape, nb_classes,
                 verbose=False, build=True, description = ''):
        self.output_directory = output_directory
        self.build = build
        self.description = description
        if self.build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
            with open(f"{output_directory}/description.txt", 'w') as f:
                f.write(description + '\n')
        else:
            self.verbose = verbose
            self.model = keras.models.load_model(
                self.output_directory + 'best_model.hdf5')
            if verbose==True:
                self.model.summary()

    def fit(self, x_train, y_train, x_val, y_val, epochs=120,
            d_class_weights=None, batch_size=64):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # x_val and y_val are only used to monitor the test loss and NOT for training
        start_time = time.time()
        if self.build==False:
            callbacks = None
        else:
            callbacks = self.callbacks
        VIANN = VarImpVIANN(verbose=1)
        callbacks.append(VIANN)
        hist = self.model.fit(
            x_train, y_train, batch_size=batch_size, epochs=epochs,
            verbose=self.verbose, validation_data=(x_val, y_val),
            callbacks=callbacks, class_weight=d_class_weights)
        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')
        save_logs(self.output_directory, hist, duration,
                  new_model=self.build)
        keras.backend.clear_session()
        with open(self.output_directory+'feature_importance.txt', 'w') as f:
            print(VIANN.varScores, file=f)
        with open(self.output_directory+'weights.txt', 'w') as f:
            print(d_class_weights, file=f)
        return self.model

    def predict(self, x_test, y_true):
        best_model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(best_model_path)
        model_dir = self.output_directory
        y_pred = model.predict(x_test, batch_size=1000, verbose=1)
        print('y_pred.shape ',y_pred.shape)
        binary = False
        if y_pred.shape[1]==1: #check if binary
            binary = True
        if binary:
            mask = y_pred>0.5
            y_pred[mask] = 1
            y_pred[~mask] = 0
            y_true = np.expand_dims(y_true, axis=1)
        else:
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)
        #avg_accuracy = accuracy_score(y_pred, y_true)
        f1 = f1_score(y_pred, y_true, average='micro')

        fig, ax = plt.subplots()
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        labels = ['coding', 'reading', 'powerpoint', 'writing', 'kb mashing', 'web surfing']
        if len(np.unique(y_true))==2: #if only two classes: 0 non-coding, 1 coding
            labels = ['non-coding', 'coding']
        ax = sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, cmap='Reds',
                         ax=ax, annot=True, vmin=0, vmax=1)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_ylabel('true', fontsize=20)
        ax.set_xlabel('pred', fontsize=20)
        ax.set_title('f1 = ' + f"{f1:.3f}", fontsize=22)
        fig.tight_layout()
        fig.savefig(f"{model_dir}/heatmap.png", dpi=100)
        plot_model(model, to_file=f"{model_dir}/architecture.png",
                   show_shapes=True, show_layer_names=True)
        # get accuracy for each class
        cm_diag = cm_norm.diagonal()
        if binary:
            np.savetxt(
                f"{model_dir}/true_vs_prediction.txt",
                np.concatenate((y_true, y_pred), axis=-1).astype(int),
                fmt="%i", header='true predicted'
            )
        else:
            np.savetxt(
                f"{model_dir}/true_vs_prediction.txt",
                np.stack((y_true, y_pred), axis=-1).astype(int),
                fmt="%i", header='true predicted'
            )
        np.savetxt(
            f"{model_dir}/accuracies.txt",
            cm_diag.flatten(), newline = " ",
            fmt="%.3f", header=', '.join([label for label in labels])+'\n'
        )
        with open(f"{model_dir}/f1.txt", 'w') as f:
            f.write(str(f1))
