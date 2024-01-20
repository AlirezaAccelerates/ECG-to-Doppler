import layers
import modules
import tensorflow as tf
import keras
import numpy as np


batch_size = 64
num_channels = 1
num_classes = 0
data_size = 2000


discriminator_in_channels = num_channels + num_classes


class CGAN(keras.Model):
    def __init__(self, discriminator, generator):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_ecg = data[0:10,:,:]
        real_doppler = data[10:20,:,:]

        # Determine the batch size from the real data
        batch_size = tf.shape(real_ecg)[0]

        # Generate fake Doppler signals using the generator
        generated_doppler = self.generator(real_ecg)


        # Combine real and fake Doppler signals
        combined_doppler = tf.concat([generated_doppler, real_doppler], axis=0)

        # Labels for real and fake data
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )

        # Train the discriminator
        with tf.GradientTape() as tape:
            print('t')
            predictions = self.discriminator(combined_doppler)
            d_loss = self.d_loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Train the generator
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            print('g')
            fake_doppler = self.generator(real_ecg)
            predictions = self.discriminator(fake_doppler)
            g_loss = self.g_loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }



def discriminator_loss(real_img, fake_img):
    return keras.losses.BinaryCrossentropy(from_logits=True)(real_img, fake_img)


def generator_loss(real_img, fake_img):
    return keras.losses.BinaryCrossentropy(from_logits=True)(real_img, fake_img)

ecg = [np.float32(np.random.uniform(-1, 1, (20000, 1))) for _ in range(10)]
doppler = [np.float32(np.random.uniform(-1, 1, (20000, 1))) for _ in range(10)]

all_data = np.concatenate([ecg, doppler])
dataset = tf.data.Dataset.from_tensor_slices((all_data))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)


input_size = (20000,1)
c_gan = CGAN(
    discriminator=modules.discriminator(input_size,), generator=modules.AU_Net(input_size))
c_gan.compile(
    d_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0003),
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)




c_gan.fit(dataset, epochs=20)