import tensorflow as tf
from train.batchNormalization import BatchNormalization


class Operations:

    @staticmethod
    def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
        if downsample:
            input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                                      padding=padding,
                                      use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                      bias_initializer=tf.constant_initializer(0.))(input_layer)

        if bn:
            conv = BatchNormalization()(conv)

        if activate == True:
            if activate_type == "leaky":
                conv = tf.nn.leaky_relu(conv, alpha=0.1)
            elif activate_type == "mish":
                conv = Operations.mish(conv)
        return conv

    @staticmethod
    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))

    @staticmethod
    def upsample(input_layer):
        return tf.image.resize(input_layer,
                               (input_layer.shape[1] * 2, input_layer.shape[2] * 2),
                               method='bilinear')

    @staticmethod
    def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
        short_cut = input_layer
        conv = Operations.convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
        conv = Operations.convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2), activate_type=activate_type)

        residual_output = short_cut + conv
        return residual_output