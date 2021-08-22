import tensorflow.keras as keras
from classifiers.template import Classifier_Template

class Classifier_CNN(Classifier_Template):

    def build_model(self, input_shape, nb_classes, args={}):
        padding = 'same'
        input_layer = keras.layers.Input(input_shape)
        activation = 'relu'
        output_activation = 'softmax'
        kernel_size = (3, 3)

        conv1 = keras.layers.Conv2D(filters=32, kernel_size=kernel_size, padding=padding,
                                    activation=activation,
                                    kernel_initializer='he_uniform')(input_layer)
        conv2 = keras.layers.Conv2D(filters=32, kernel_size=kernel_size, padding=padding,
                                    activation=activation, kernel_initializer='he_uniform')(conv1)
        conv2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = keras.layers.Conv2D(filters=64, kernel_size=kernel_size, padding=padding,
                                    activation=activation, kernel_initializer='he_uniform')(conv2)
        conv4 = keras.layers.Conv2D(filters=64, kernel_size=kernel_size, padding=padding,
                                    activation=activation, kernel_initializer='he_uniform')(conv3)
        conv4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = keras.layers.Conv2D(filters=128, kernel_size=kernel_size, padding=padding,
                                    activation=activation, kernel_initializer='he_uniform')(conv4)
        conv6 = keras.layers.Conv2D(filters=128, kernel_size=kernel_size, padding=padding,
                                    activation=activation, kernel_initializer='he_uniform')(conv5)
        conv6 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv6)

        flatten_layer = keras.layers.Flatten()(conv6)
        dense_layer = keras.layers.Dense(128, activation=activation,
                                           kernel_initializer='he_uniform')(flatten_layer)
        output_layer = keras.layers.Dense(units=nb_classes,
                                          activation=output_activation)(dense_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss="categorical_crossentropy",
                      optimizer=keras.optimizers.Adam(amsgrad=True),
                      metrics=['accuracy'],)
        file_path = self.output_directory + 'best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)
        self.callbacks = [model_checkpoint]

        return model



