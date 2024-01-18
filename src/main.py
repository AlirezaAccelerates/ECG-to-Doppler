import layers
import modules
import tensorflow as tf
from keras.models import Model


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

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data_ecg, data_doppler):
        # Unpack the data.
        #real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        #image_one_hot_labels = one_hot_labels[:, :, None, None]
        #image_one_hot_labels = ops.repeat(
        #    image_one_hot_labels, repeats=[image_size * image_size]
        #)
        #image_one_hot_labels = ops.reshape(
        #    image_one_hot_labels, (-1, image_size, image_size, num_classes)
        #)

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        #batch_size = ops.shape(real_images)[0]
        #random_latent_vectors = keras.random.normal(
        #    shape=(batch_size, self.latent_dim), seed=self.seed_generator
        #)
        #random_vector_labels = ops.concatenate(
        #    [random_latent_vectors, one_hot_labels], axis=1
        #)

        
        generated_images = self.generator(data_ecg)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        #fake_image_and_labels = ops.concatenate(
        #    [generated_images, image_one_hot_labels], -1
        #)
        #real_image_and_labels = ops.concatenate([real_images, image_one_hot_labels], -1)
        combined_data = ops.concatenate(
            [data_ecg, data_doppler], axis=0
        )

        # Assemble labels discriminating real from fake images.
        #labels = ops.concatenate(
        #    [ops.ones((batch_size, 1)), ops.zeros((batch_size, 1))], axis=0
        #)

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_data)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        random_vector_labels = ops.concatenate(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = ops.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = ops.concatenate(
                [fake_images, image_one_hot_labels], -1
            )
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }



c_gan = CGAN(
    discriminator=discriminator, generator=generator)
c_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

c_gan.fit(dataset, epochs=20)