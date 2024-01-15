import layers
from keras.layers import Input, Add, MaxPooling1D, UpSampling1D, concatenate, Multiply, GlobalAveragePooling1D

def recurrent_residual_conv1d(input_layer, filters, kernel_size=3):
    # First Convolution
    conv = conv1d(filters, kernel_size, padding='same', stride=1)(input_layer)
    conv = batchNormalization()(conv)
    conv = activation('relu')(conv)

    # Recurrent Layer
    recurrent = conv1d(filters, kernel_size, padding='same', activation='relu')(conv)
    
    # Residual Connection
    residual = Add()([input_layer, recurrent])

    return residual