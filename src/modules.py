import layers
from keras.layers import Input, Add, UpSampling1D, concatenate, Multiply, Flatten
from keras.models import Model, Sequential



## Recurrent Residual 1DCNN
def recurrent_residual_conv1d(input_layer, filters, kernel_size):
    # First Convolution
    conv = layers.Conv1D(filters, kernel_size, padding='same', strides=1)(input_layer)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation(conv, 'relu')
    print(conv)

    # Recurrent Layer
    recurrent = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(conv)
    
    # Residual Connection
    residual = Add()([input_layer, recurrent])

    return residual



## Attention Block
def attention_block(x, g, inter_channel):
    
    theta_x = layers.Conv1D(inter_channel, kernel_size=1, padding='same')(x)
    phi_g = layers.Conv1D(inter_channel, kernel_size=1, padding='same')(g)

    f = layers.Activation(Add()([theta_x, phi_g]),'relu')
    psi_f = layers.Conv1D(1, kernel_size=1, padding='same', activation='sigmoid')(f)

    return Multiply()([x, psi_f])




## Attention Recurrent Residual Convolutional Neural Network based on U-Net
def AR2U_Net(input_size, depth=6, initial_filters=64, kernel_size=16):
    inputs = Input(input_size)

    encoders = []
    for i in range(depth):

        if i == 0:
            x = recurrent_residual_conv1d(inputs, initial_filters, kernel_size)
            print('HI')
        else:
            filters = initial_filters * (2 ** i)
            if filters >= 512: filters = 512
            x = recurrent_residual_conv1d(encoders[-1][0], filters, kernel_size)
            print('HI2')
        p = layers.MaxPooling1D(pool_size=2)(x)
        encoders.append((x, p))



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
    output = layers.Conv1D(1, kernel_size, padding='same', activation='tanh')(x)

    model = Model(inputs=inputs, outputs=output)
    return model



## Attention Convolutional Neural Network based on U-Net
def AU_Net(input_size, depth=4, initial_filters=64, kernel_size=16):
    inputs = Input(input_size)

    # Encoder path
    encoders = []
    for i in range(depth):
        if i == 0:
            x = layers.Conv1D(initial_filters, kernel_size, padding='same', activation='relu')(inputs)
            x = layers.Conv1D(initial_filters, kernel_size, padding='same', activation='relu')(x)
        else:
            filters = initial_filters * (2 ** i)
            if filters > 512: filters = 512
            x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(encoders[-1][1])
            x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(encoders[-1][1])

        p = layers.MaxPooling1D(pool_size=2)(x)
        encoders.append((x, p))



    # Bottleneck
    bottleneck = layers.Conv1D(512, kernel_size, activation='relu', padding='same')(encoders[-1][1])
    bottleneck = layers.Conv1D(512, kernel_size, activation='relu', padding='same')(bottleneck)

    # Decoder path
    for i in reversed(range(depth)):
        filters = initial_filters * (2 ** i)
        if filters > 512: filters = 512
        if filters <= 64: filters = 64

        u = UpSampling1D(size=2)(bottleneck if i == depth - 1 else x)
        #attn = attention_block(encoders[i][0], u, filters)
        x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(u)#(concatenate([u, attn]))
        x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)


    # Output layer
    output = layers.Conv1D(1, 1, padding='same', activation='tanh')(x)

    model = Model(inputs=inputs, outputs=output)
    return model


## Discriminator's Architecture
def discriminator(shape, kernel_size=16, strides=2):
    model = Sequential()
    
    model.add(layers.Conv1D(32, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu'))
    model.add(layers.Conv1D(32, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout())
    
    model.add(layers.Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu'))
    model.add(layers.Conv1D(128, kernel_size=kernel_size, strides=strides, padding="same", activation='leaky_relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout())
    
    model.add(Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    input = Input(shape=shape)
    validity = model(input)

    return Model(input, validity)