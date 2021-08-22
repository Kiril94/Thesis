import tensorflow as tf
import tensorflow.keras as keras


class GConv2D(keras.layers.Layer):
    """GConvolutional layer for the group p4m"""

    def __init__(self, kernel_size=(3, 3), out_channels=1,
                 kernel_initializer="glorot_uniform", stride=1, padding="SAME",
                 activation='relu', group='p4m'):
        super(GConv2D, self).__init__()
        self.out_channels = out_channels
        self.kernel_initializer = kernel_initializer
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.kernel_size = kernel_size
        self.group = group

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.kernel_size[0],
                   self.kernel_size[1], self.out_channels),
            initializer=self.kernel_initializer, trainable=True)

    def call(self, inputs, name=""):
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=-1)

        # rot90 and flip_left_right take shape [batch, height, width, channels]
        kernels_p4 = [tf.image.rot90(self.kernel, k=i) for i in range(1, 4)]
        kernels_p4.insert(0, self.kernel)
        if self.group == 'p4m':
            kernel_m = tf.image.flip_left_right(self.kernel)
            kernels_m = [tf.image.rot90(kernel_m, k=i) for i in range(1, 4)]
            kernels_m.insert(0, kernel_m)
            kernels_p4m = kernels_p4 + kernels_m
        elif self.group == 'p4':
            kernels_p4m = kernels_p4
        else:
            raise ValueError(f"The group {self.group} is not implemented.\n"
                             f" Choose group \"p4\" or \"p4m\" ")
        kernels_p4m = [tf.transpose(ker, perm=[1, 2, 0, 3]) for ker in kernels_p4m]
        # filter takes shape [filter_height, filter_width, in_channels, out_channels]
        outputs = [tf.nn.conv2d(
            input=inputs, filters=ker, padding=self.padding,
            strides=(self.stride, self.stride, 1, 1)) for i, ker in enumerate(kernels_p4m)]
        if self.activation == None:
            pass
        elif self.activation == 'relu':
            outputs = [tf.keras.activations.relu(output) for output in outputs]
        else:
            raise ValueError(f"{self.activation} activation is not implemented")
        outputs = tf.concat(outputs, axis=-1)
        return outputs