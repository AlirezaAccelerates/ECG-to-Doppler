import layers
from keras.layers import Input, Add, UpSampling1D, concatenate, Multiply
from keras.models import Model, Sequential



## Recurrent Residual 1DCNN
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



## Attention Block
def attention_block(x, g, inter_channel):
    
    theta_x = Covid1D(inter_channel, kernel_size=1, padding='same')(x)
    phi_g = Covid1D(inter_channel, kernel_size=1, padding='same')(g)

    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Covid1D(1, kernel_size=1, padding='same', activation='sigmoid')(f)

    return Multiply()([x, psi_f])




## Attention Recurrent Residual Convolutional Neural Network based on U-Net
def AR2U_Net(input_size, depth, initial_filters, kernel_size):
    inputs = Input(input_size)

    # Encoder path
    encoders = []
    for i in range(depth):
        if i == 0:
            x = recurrent_residual_conv1d(inputs, filters, kernel_size)
        else:
            x = recurrent_residual_conv1d(encoders[-1][0], filters, kernel_size)
        p = MaxPooling1D(pool_size=2)(x)
        encoders.append((x, p))

        filters = initial_filters * (2 ** i)
        if filters >= 512: filters = 512

    # Bottleneck
    bottleneck = recurrent_residual_conv1d(encoders[-1][1], 512, kernel_size)

    # Decoder path
    for i in reversed(range(depth)):
        filters = initial_filters * (2 ** i)
        if filters >= 512: filters = 512
        if filters <= 64: filters = 64
        u = UpSampling1D(size=2)(bottleneck if i == depth - 1 else x)
        attn = attention_block(encoders[i][0], u, filters)
        x = concatenate([u, attn])
        x = recurrent_residual_conv1d(x, filters, kernel_size)

    # Output layer
    output = Conv1D(1, kernel_size, padding='same', activation='tanh')(x)

    model = Model(inputs=inputs, outputs=output)
    return model



## Attention Convolutional Neural Network based on U-Net
def A_Net(input_size, depth, initial_filters, kernel_size):
    inputs = Input(input_size)

    # Encoder path
    encoders = []
    for i in range(depth):
        if i == 0:
            x = Conv1D(filters, kernel_size, padding='same', activation='relu')(inputs)
        else:
            x = Conv1D(filters, kernel_size, padding='same', activation='relu')(encoders[-1][1])
        x = Conv1D(filters, kernel_size, padding='same', activation='relu')(encoders[-1][1])
        r = Add()([input_layer, recurrent])

        p = MaxPooling1D(pool_size=2)(r)
        encoders.append((r, p))

        filters = initial_filters * (2 ** i)
        if filters > 512: filters = 512

    # Bottleneck
    bottleneck = Conv1D(512, kernel_size, activation='relu', padding='same')(encoders[-1][1])
    bottleneck = Conv1D(512, kernel_size, activation='relu', padding='same')(bottleneck)

    # Decoder path
    for i in reversed(range(depth)):
        filters = initial_filters * (2 ** i)
        if filters > 512: filters = 512
        if filters <= 64: filters = 64
        u = UpSampling1D(size=2)(bottleneck if i == depth - 1 else x)
        attn = attention_block(encoders[i][0], u, filters)
        x = Conv1D(filters, kernel_size, padding='same', activation='relu')(concatenate([u, attn]))
        x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)

    # Output layer
    output = Conv1D(1, kernel_size, padding='same', activation='tanh')(x)

    model = Model(inputs=inputs, outputs=output)
    return model


## Discriminator's Architecture
def discriminator(shape, kernel_size, strides=2):
    model = Sequential()
    
    model.add(Conv1D(64, kernel_size=kernel_size, strides=strides, input_shape=shape, padding="same", activation='leaky_relu'))
    model.add(Conv1D(64, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout())
    
    model.add(Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu'))
    model.add(Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu'))
    model.add(BatchNormalization())
    model.add(Dropout())
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    input = Input(shape=shape)
    validity = model(input)

    return Model(input, validity)