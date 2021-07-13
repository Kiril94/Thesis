from builtins import print
from tensorflow.keras import models
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.utils import class_weight

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.use('agg')

def plot_heatmap(y_true, y_pred, save_dir, figsize=(6,5)):

    fig, ax = plt.subplots(figsize=figsize)
    cm = confusion_matrix(y_true, y_pred)

    cm_norm = cm.astype('float') #/ cm.sum(axis=1)[:, np.newaxis]
    labels = ['coding', 'reading', 'powerpoint', 'writing', 'kb mashing', 'web surfing']
    if len(np.unique(y_pred))==2 or len(np.unique(y_pred))==1: #if only two classes: 0 non-coding, 1 coding
        labels = ['non-coding', 'coding']
    ax = sns.heatmap(cm_norm, xticklabels=labels, yticklabels=labels, cmap='Reds',
                     ax=ax, annot=True, vmin=0, vmax=1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('true', fontsize=20)
    ax.set_xlabel('pred', fontsize=20)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/heatmap.png", dpi=100)

def plot_heatmap1d(y_true, y_pred, save_dir, figsize=(7,5)):
    size = 4
    data = np.arange(size * size).reshape((size, size))

    # Limits for the extent
    x_start = 3.0
    x_end = 9.0
    y_start = 6.0
    y_end = 12.0

    extent = [x_start, x_end, y_start, y_end]

    # The normal figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, extent=extent, origin='lower',
                   interpolation='None', cmap='Reds')

    # Add the text
    jump_x = (x_end - x_start) / (2.0 * size)
    jump_y = (y_end - y_start) / (2.0 * size)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = data[y_index, x_index]
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(text_x, text_y, label, color='black', ha='center', va='center')

    fig.colorbar(im)
    plt.show()

def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize=20)
    plt.xlabel('epoch', fontsize=20 )
    plt.legend(['train', 'val'], loc='upper right', fontsize=20)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def save_logs(output_directory, hist, duration, new_model):
    if new_model:
        hist_df = pd.DataFrame(hist.history)
        with open(output_directory + 'training_duration.txt', 'w') as f:
            f.write(str(duration)+'\n')
        hist_df.to_csv(output_directory + 'history.csv', index=False)
        # plot losses
        plot_epochs_metric(hist, output_directory + 'epochs_loss.png')
    else:
        hist_df_old = pd.read_csv(output_directory + 'history.csv')
        hist_df_new = pd.DataFrame(hist.history)
        hist_df = pd.concat([hist_df_old, hist_df_new])
        hist_df.to_csv(output_directory + 'history.csv', index=False)
        with open(output_directory + 'training_duration.txt', 'a') as f:
            f.write(str(duration)+'\n')

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_data(data_dir):
    data_dict = {}
    data_dict['X_train'] = np.load(f"{data_dir}/X_train.npy")
    data_dict['y_train'] = np.load(f"{data_dir}/Y_train.npy")
    data_dict['X_test'] = np.load(f"{data_dir}/X_test.npy")
    data_dict['y_test'] = np.load(f"{data_dir}/Y_test.npy")
    data_dict['X_val'] = np.load(f"{data_dir}/X_val.npy")
    data_dict['y_val'] = np.load(f"{data_dir}/Y_val.npy")
    return data_dict

def train_pred(settings):
    # load data
    (trainX, trainy), (testX, testy) = cifar10.load_data()

    # set if samples should be weighted,
    # this will be removed later as we will use only weighted samples
    if settings.weight:  # imbalanced dataset
        class_weights = class_weight.compute_class_weight(
            class_weight= 'balanced',
            classes = np.unique(np.argmax(y_train, axis=1)),
            y = np.argmax(y_train, axis=1))
    else:
        class_weights = np.ones(len(np.unique(np.argmax(y_train, axis=1))))
    d_class_weights = dict(enumerate(class_weights))

    model_dir = f"{settings.output_dir}/{settings.model_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #set missing options
    args_dict = dotdict(settings.args_dict)
    args_dict['output_directory'] = model_dir
    args_dict['input_shape'] = X_train.shape[1:]
    args_dict['nb_classes'] = y_train.shape[1]
    if y_train.shape[1] == 1: #check if binary
        datad.y_train = np.argmax(datad.y_train, axis = 1)
        datad.y_test = np.argmax(datad.y_test, axis=1)
        datad.y_val = np.argmax(datad.y_val, axis=1)

    classifier = settings.classifier
    if classifier == 'cnn':
        import classifiers.cnn_optimize
        model = classifiers.cnn_optimize.Classifier_CNN(**args_dict)
    else:
        print(classifier, ' is not a valid classifier name!')


    model.fit(datad.X_train, datad.y_train, datad.X_val, datad.y_val,
              epochs=settings.epochs, d_class_weights=d_class_weights,
              batch_size=settings.batch_size)
    model.predict(datad.X_test, datad.y_test)

def predict(model_dir, data_dir, output_dir):
    """Make predictions"""
    #TODO: adjust this function to both multiclass and binary
    X_test = np.load(f"{data_dir}/X_test.npy")
    input_list = np.split(X_test, 5, axis=-1)
    y_test = np.load(f"{data_dir}/Y_test.npy")
    filepath_best = f"{model_dir}/best_model.hdf5"
    model = models.load_model(filepath_best)
    y_pred = model.predict(input_list, verbose=1)
    y_test = np.argmax(y_test, axis=1).flatten()
    y_pred = np.argmax(y_pred, axis=1).flatten()
    fig, ax = plt.subplots()
    ax.hist(y_pred)
    fig.savefig(f"{output_dir}/distr")
    mask = y_pred>0.5
    y_pred[mask] = 1
    y_pred[~mask] = 0
    #if not len(y_test.shape)==1: # check if multiclass
    #    y_pred = np.argmax(y_pred, axis=1)
    #    y_test = np.argmax(y_test, axis=1)
    f1 = f1_score(y_pred, y_test, average='macro')
    plot_heatmap(y_test, y_pred, output_dir)

    return f1, y_test, y_pred


class VarImpVIANN(keras.callbacks.Callback):
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.n = 0
        self.M2 = 0.0

    def on_train_begin(self, logs={}, verbose=1):
        if self.verbose:
            print("VIANN version 1.0 (Wellford + Mean) update per epoch")
        self.diff = self.model.layers[-1].get_weights()[0]
        # last layer is interesting

    def on_epoch_end(self, batch, logs={}):
        """
        for i, l in enumerate(self.model.layers):
            if len(l.get_weights())==0:
                print(f"layer {i}: empty")
                continue
            else:
                print(f"layer {i}")
                print(l.get_weights()[0].shape)
        """
        currentWeights = self.model.layers[-1].get_weights()[0]

        self.n += 1
        delta = np.subtract(currentWeights, self.diff)
        self.diff += delta / self.n
        delta2 = np.subtract(currentWeights, self.diff)
        self.M2 += delta * delta2

        self.lastweights = self.model.layers[-1].get_weights()[0]

    def on_train_end(self, batch, logs={}):
        if self.n < 2:
            self.s2 = float('nan')
        else:
            self.s2 = self.M2 / (self.n - 1)

        scores = np.sum(np.multiply(self.s2, np.abs(self.lastweights)), axis=1)

        self.varScores = (scores - min(scores)) / (max(scores) - min(scores))
        if self.verbose:
            print("Most important variables: ",
                  np.array(self.varScores).argsort()[-10:][::-1])