import layers
import modules
from keras.models import Model

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes