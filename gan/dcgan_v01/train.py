#!/usr/bin/env python3
from dcgan_v01.gan import GAN
from dcgan_v01.generator import Generator
from dcgan_v01.discriminator import Discriminator
from keras.datasets import mnist
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import logging

class Trainer:
    def __init__(self, width=28, height=28, channels=1, latent_size=100, epochs=50000, batch=32, checkpoint=50, model_type=-1, data_path=''):
        self.logger = logging.getLogger(__name__)
        self.W = width
        self.H = height
        self.C = channels
        self.EPOCHS = epochs
        self.BATCH = batch
        self.CHECKPOINT = checkpoint
        self.model_type = model_type
        self.LATENT_SPACE_SIZE = latent_size
        self.generator = Generator(height=self.H, width=self.W, channels=self.C, latent_size=self.LATENT_SPACE_SIZE, model_type='DCGAN')
        self.discriminator = Discriminator(height=self.H, width=self.W, channels=self.C, model_type='DCGAN')
        self.gan = GAN(generator=self.generator.Generator, discriminator=self.discriminator.Discriminator)
        self.load_npy(data_path)

    def load_npy(self, npy_path):
        self.X_train = np.load(npy_path)
        self.X_train = self.X_train[:int(0.25 * float(len(self.X_train)))]
        self.X_train = (self.X_train.astype(np.float32) - 127.5) / 127.5
        self.X_train = np.expand_dims(self.X_train, axis=3)
        return

    def train(self):
        for e in range(self.EPOCHS):
            b = 0
            X_train_temp = deepcopy(self.X_train)
            while len(X_train_temp) > self.BATCH:
                self.logger.info("X_train_temp length is {}".format(len(X_train_temp)))
                # Keep track of Batches
                b = b + 1
                # Train Discriminator
                # Make the training batch for this model be half real, half noise
                # Grab Real Images for this training batch
                if self.flipCoin():
                    self.logger.info("Flip Coin is 1")
                    count_real_images = int(self.BATCH)
                    starting_index = randint(0, (len(X_train_temp) - count_real_images))
                    real_images_raw = X_train_temp[ starting_index : (starting_index + count_real_images)]
                    self.logger.info("{} shape of real_images_raw".format(real_images_raw.shape))
                    #self.logger.info("width {}, height {}, channels {}".format(self.W, self.H, self.C))
                    # Delete the images used until we have none left
                    X_train_temp = np.delete(X_train_temp, range(starting_index, (starting_index + count_real_images)),0)
                    x_batch = real_images_raw.reshape(count_real_images, self.W, self.H, self.C)
                    y_batch = np.zeros([self.BATCH,1])
                else:
                    self.logger.info("Flip Coin is 0")
                    # Grab Generated Images for this training batch
                    latent_space_samples = self.sample_latent_space(self.BATCH)
                    x_batch = self.generator.Generator.predict(latent_space_samples)
                    y_batch = np.zeros([self.BATCH,1])

                # Now, train the discriminator with this batch
                discriminator_loss = self.discriminator.Discriminator.train_on_batch(x_batch, y_batch)[0]

                # In practice, flipping the label when training the generator improves convergence
                if self.flipCoin(chance=0.9):
                    y_generated_labels = np.ones([self.BATCH,1])
                else:
                    y_generated_labels = np.zeros([self.BATCH,1])
                x_latent_space_samples = self.sample_latent_space(self.BATCH)
                generator_loss = self.gan.gan_model.train_on_batch(x_latent_space_samples, y_generated_labels)

                print('Batch: ' + str(int(b)) + ', [Discriminator :: Loss: ' + str(discriminator_loss) +
                      '], [ Generator :: Loss : ' + str(generator_loss) + ']')
                if b % self.CHECKPOINT == 0:
                    label = str(e) + '_' + str(b)
                    self.plot_checkpoint(label)

            print ('Epoch: ' + str(int(e)) + ', [Discriminator :: Loss: ' + str(discriminator_loss) +
                   '], [ Generator :: Loss: ' + str(generator_loss) + ']')

            if e % self.CHECKPOINT == 0:
                self.plot_checkpoint(e)

        return

    def flipCoin(selfs, chance=0.5):
        return np.random.binomial(1, chance)

    def sample_latent_space(self, instances):
        return np.random.normal(0,1, (instances, self.LATENT_SPACE_SIZE))

    def plot_checkpoint(self, e):
        filename = 'data/sample_' + str(e) + ".png"

        noise = self.sample_latent_space(16)
        images = self.generator.Generator.predict(noise)

        plt.figure(figsize = (10,10))
        for i in range(images.shape[0]):
            plt.subplot(4,4,i+1)
            if self.C ==1:
                image = images[i, :, :]
                image = np.reshape(image, [self.H, self.W])
                image = (255 * (image - np.min(image)) / np.ptp(image)).astype(int)
                plt.imshow(image, cmap='gray')
            elif self.C == 3:
                image = images[i, :, :, :]
                image = np.reshape(image, [self.H, self.W, self.C])
                image = (255 * (image - np.min(image)) / np.ptp(image)).astype(int)
                plt.imshow(image)

            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
        return
