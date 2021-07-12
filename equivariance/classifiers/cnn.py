import tensorflow.keras as keras
from .template import Classifier_Template
from tensorflow.keras.layers import LeakyReLU

class Classifier_CNN(Classifier_Template):

    def build_model(self, input_shape, nb_classes):
        padding = 'same'
        input_layer = keras.layers.Input(input_shape)
        activation = 'sigmoid'
        output_activation = 'softmax'

        kernel_size = 9
        if activation == 'leaky_relu':
            activation=None
        conv1 = keras.layers.Conv1D(filters=6, kernel_size=kernel_size,padding=padding,
                                    activation=activation)(input_layer)
        if activation == None:
            conv1 = LeakyReLU()(conv1)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)
        conv2 = keras.layers.Conv1D(filters=6,kernel_size=kernel_size,padding=padding,
                                    activation=activation)(conv1)
        if activation == None:
            conv2 = LeakyReLU()(conv2)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        if nb_classes==2:
            nb_classes=1 # only 1 neuron is needed for binary classification
            output_activation = 'sigmoid'
        output_layer = keras.layers.Dense(units=nb_classes,
                                          activation=output_activation)(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        if nb_classes==1:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        model.compile(loss=loss,
                      optimizer=keras.optimizers.Adam(amsgrad=True),
                      metrics=['accuracy'],)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [model_checkpoint]

        return model



