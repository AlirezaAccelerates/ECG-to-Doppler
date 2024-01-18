import layers
import modules
from keras.models import Model


batch_size = 64
num_channels = 1
num_classes = 0

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes