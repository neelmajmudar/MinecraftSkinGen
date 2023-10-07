import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, classification_report
import PIL 
from PIL import Image
import os 
import numpy as np
import seaborn as sns
import time 

def process_data(folder):
    image_paths = [os.path.join(data_folder, filename) for filename in os.listdir(data_folder)[:25000]]

    train_images = [np.array(load_img(path)) for path in tqdm(image_paths)]
    train_images = np.array(train_images)

    train_images = train_images.reshape((-1, 64, 64, 3)).astype('float32')

    train_images = (train_images - 127.5) / 127.5

    # If save file exists
    if os.path.exists('processed_images.npy'):
        processed_images = np.load('processed_images.npy')
    else:
        # Save images to a file
        np.save('processed_images.npy', train_images)

data_folder = 'C:\\Users\\Administrator\\Desktop\\Skins'

loaded_images = np.load('processed_images.npy')
num_images = len(loaded_images)
input_shape = (64, 64, 3)
latent_dim = 100
weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=42)
channels = 3

def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(8 * 8 * 512, input_dim=latent_dim))
    model.add(ReLU())
    model.add(Reshape((8, 8, 512)))
    
    # Upsample to 16x16
    model.add(Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init))
    model.add(ReLU())
    
    # Upsample to 32x32
    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init))
    model.add(ReLU())
    
    # Upsample to 64x64
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init))
    model.add(ReLU())
    
    # Final convolutional layer
    model.add(Conv2D(channels, (3, 3), padding='same', activation='tanh'))

    return model

def build_discriminator():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # Add more convolutional layers here for further depth
    model.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(layers.Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    return model


generator = build_generator(latent_dim)
discriminator = build_discriminator()

cross_entropy = BinaryCrossentropy(from_logits=False)

generator_optimizer = Adam(learning_rate=0.0003, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)

checkpoint_dir = 'ModelCP'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradient_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradient_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradient_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss  # Return the losses

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        gen_losses = []
        disc_losses = []

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        avg_gen_loss = tf.reduce_mean(gen_losses)
        avg_disc_loss = tf.reduce_mean(disc_losses)

        print(f"Epoch {epoch+1}/{epochs}, Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}")

        generate_and_save_images(generator, epoch+1)

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    generate_and_save_images(generator, epochs)
    on_train_end()


def generate_and_save_images(generator, epoch):
    noise = tf.random.normal([25, latent_dim])
    image_gen = generator(noise)
    image_gen = (image_gen * 127.5) + 127.5 
    image_gen.numpy()

    fig = plt.figure(figsize=(8,8))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        image = array_to_img(image_gen[i])
        plt.imshow(image)
        plt.axis('off')
    plt.savefig(os.path.join('GeneratedSkins', 'epoch_{:03d}.png'.format(epoch)))
    plt.close(fig)

def on_train_end():
    generator.save('generator.h5')

def generate_single_image():
    model = keras.models.load_model('generator.h5')
    noise = tf.random.normal([1, latent_dim])
    image_gen = model(noise)
    image_gen = (image_gen * 127.5) + 127.5
    image_gen = tf.image.resize(image_gen, (64, 64, 4)) 
    image_gen.numpy()
    image = array_to_img(image_gen[0])
    image.save('C:\\Users\\Administrator\\Desktop\\generated_image.png', 'PNG') 
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def generate_multiple_images():
    noise = tf.random.normal([25, latent_dim])
    model = keras.models.load_model('generator.h5')
    image_gen = model(noise)
    image_gen = (image_gen * 127.5) + 127.5 
    image_gen = tf.image.resize(image_gen, (64, 64)) 
    image_gen.numpy()

    fig = plt.figure(figsize=(8,8))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        image = array_to_img(image_gen[i])
        image.save(os.path.join('C:\\Users\\Administrator\\Desktop\\Test Skins', f'Skin_{i}.png'), 'PNG')  
        plt.imshow(image)
        plt.axis('off')
    plt.show()


num_generated = 12
noise_dim = 100
batch_size = 64
epochs = 50

dataset = tf.data.Dataset.from_tensor_slices(loaded_images)
dataset = dataset.shuffle(buffer_size=num_images)
dataset = dataset.batch(batch_size, drop_remainder=True)

#generate_single_image()
generate_multiple_images()

#train(dataset, epochs)
#generate_and_save_images(generator, epochs)
#on_train_end()