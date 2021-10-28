import tensorflow as tf
import tensorflow.keras as keras
from classifiers.template import Classifier_Template

class Classifier_CNN(Classifier_Template):

    def build_model(self, input_shape, nb_classes, padding="SAME",
                    activation="relu", Filters=[18, 36, 72], kernel_size=(3,3),
                    args={},):
        
        input_layer = keras.layers.Input(input_shape)
        x1 = keras.layers.Conv2D(filters=Filters[0], kernel_size=kernel_size,
                                    padding=padding, activation=activation)(input_layer)
        x2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(x1)
        x3 = keras.layers.Conv2D(filters=Filters[1], kernel_size=kernel_size,
                                    padding=padding, activation=activation)(x2)
        x4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(x3)
        x5 = keras.layers.Conv2D(filters=Filters[2], kernel_size=kernel_size,
                                    padding=padding, activation=activation)(x4)
        x6 = keras.layers.AveragePooling2D(pool_size=(7, 7))(x5)

        flatten_layer = keras.layers.Flatten()(x6)
        x7 = keras.layers.Dense(units=50, activation=activation)(flatten_layer)
        outputs = keras.layers.Dense(units=nb_classes, activation='sigmoid')(x7)
        model = keras.models.Model(inputs=input_layer, outputs=outputs)
        file_path = self.output_directory + 'best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                           monitor='loss',
                                                           save_best_only=True)
        self.callbacks = [model_checkpoint]
        model.compile(loss=keras.losses.CategoricalCrossentropy(),
                      optimizer=keras.optimizers.Adam(amsgrad=True),
                      metrics=['accuracy'], callbacks=self.callbacks )

        return model



