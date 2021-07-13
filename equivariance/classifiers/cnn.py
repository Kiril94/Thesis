import tensorflow.keras as keras
from .template import Classifier_Template

class Classifier_CNN(Classifier_Template):

    def build_model(self, input_shape, nb_classes, args):
        padding = 'same'
        input_layer = keras.layers.Input(input_shape)
        activation = 'ReLu'
        output_activation = 'softmax'

        kernel_size = (3, 3)
        conv1 = keras.layers.Conv2D(filters=1, kernel_size=kernel_size, padding=padding,
                                    activation=activation,
                                    kernel_initializer='he_uniform')(input_layer)
        conv1 = keras.layers.MaxPooling2D(pool_size=3)(conv1)
        conv2 = keras.layers.Conv2D(filters=1, kernel_size=kernel_size, padding=padding,
                                    activation=activation,
                                    kernel_initializer='he_uniform')(conv1)
        conv2 = keras.layers.MaxPooling2D(pool_size=3)(conv2)
        flatten_layer = keras.layers.Flatten()(conv2)
        output_layer = keras.layers.Dense(units=nb_classes,
                                          activation=output_activation)(flatten_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=loss,
                      optimizer=keras.optimizers.Adam(amsgrad=True),
                      metrics=['accuracy'],)
        file_path = self.output_directory + 'best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)
        self.callbacks = [model_checkpoint]

        return model



