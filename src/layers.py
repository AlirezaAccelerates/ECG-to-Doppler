import keras

weights_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=2024)


def Conv1D(filters, kernel_size, strides=1, padding='same', activation=None, use_bias = True):

    layer = keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=None,
        dilation_rate=1, groups=1, activation=activation, use_bias=use_bias, kernel_initializer=weights_initializer,
        bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None
    )

    return layer


def Dense(units):

    layer = keras.layers.Dense(
        units=units, activation=None, use_bias=True, kernel_initializer="glorot_uniform",
        bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
    )

    return layer

def BatchNormalization(momentum=0.99, epsilon=0.001, trainable=True, virtual_batch_size=None):

    layer = keras.layers.BatchNormalization(
        axis=-1, momentum=momentum, epsilon=epsilon, center=True, scale=True,
        beta_initializer='zeros', gamma_initializer='ones',
         moving_mean_initializer='zeros', moving_variance_initializer='ones',
        beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
         gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
        fused=None, trainable=trainable, virtual_batch_size=virtual_batch_size, adjustment=None, name=None
    )
    
    return layer


def Activation(x, activation):
    
    if activation == 'relu':
        return keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)
    elif activation == 'leaky_relu':
        return keras.activations.relu(x, alpha=0.2, max_value=None, threshold=0)
    elif activation == 'silu':
        return keras.activations.silu(x)
    elif activation == 'sigmoid':
        return keras.activations.sigmoid(x)
    elif activation == 'softmax':
        return keras.activations.softmax(x, axis=-1)
    elif activation == 'tanh':
        return keras.activations.tanh(x)
    else:
        raise ValueError('please check the name of the activation function!')
    

def MaxPooling1D(pool_size=2, strides=None, padding="valid"):

    layer = keras.layers.MaxPooling1D(
    pool_size=pool_size, strides=strides, padding=padding, data_format=None, name=None
    )

    return layer


def Dropout(rate=0.2):

    layer = keras.layers.Dropout(
        rate=rate, noise_shape=None, seed=2024
    )

    return layer

