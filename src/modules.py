import layers
from keras.layers import Input, Add, UpSampling1D, concatenate, Multiply, GlobalAveragePooling1D


def recurrent_residual_conv1d(input_layer, filters, kernel_size=3):
    # First Convolution
    conv = Covid1D(filters, kernel_size, padding='same', stride=1)(input_layer)
    conv = BatchNormalization()(conv)
    conv = Activation(conv, 'relu')

    # Recurrent Layer
    recurrent = Covid1D(filters, kernel_size, padding='same', activation='relu')(conv)
    
    # Residual Connection
    residual = Add()([input_layer, recurrent])

    return residual


def attention_block(x, g, inter_channel):
    # Attention Block
    theta_x = Covid1D(inter_channel, kernel_size=1, padding='same')(x)
    phi_g = Covid1D(inter_channel, kernel_size=1, padding='same')(g)

    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Covid1D(1, kernel_size=1, padding='same', activation='sigmoid')(f)

    return Multiply()([x, psi_f])


def R2U_Net(input_size, depth, initial_features, kernel_size):
    inputs = Input(input_size)

    # Encoder path
    encoders = []
    for i in range(depth):
        filters = initial_features * (2 ** i)
        if filters >= 512: filters = 512
        if i == 0:
            x = recurrent_residual_conv1d(inputs, filters, kernel_size)
        else:
            x = recurrent_residual_conv1d(encoders[-1][0], filters, kernel_size)
        p = MaxPooling1D(pool_size=2)(x)
        encoders.append((x, p))

    # Bottleneck
    bottleneck = recurrent_residual_conv1d(encoders[-1][1], 512, kernel_size)

    # Decoder path
    for i in reversed(range(depth)):
        filters = initial_features * (2 ** i)
        if filters >= 512: filters = 512
        if filters <= 64: filters = 64
        g = UpSampling1D(size=2)(bottleneck if i == depth - 1 else x)
        attn = attention_block(encoders[i][0], g, filters)
        x = concatenate([g, attn])
        x = recurrent_residual_conv1d(x, filters, kernel_size)

    # Output layer
    output = Conv1D(1, kernel_size, strides=2, padding='same', activation='tanh')(x)

    model = Model(inputs=inputs, outputs=output)
    return model