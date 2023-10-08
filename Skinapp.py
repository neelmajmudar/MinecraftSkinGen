import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os
import numpy as np
import seaborn as sns
import time

def generate_single_image():
    latent_dim = 100
    model = keras.models.load_model('generator.h5')
    noise = tf.random.normal([1, latent_dim])
    image_gen = model(noise)
    image_gen = (image_gen * 127.5) + 127.5
    image_gen = tf.image.resize(image_gen, (64, 64))
    image_gen.numpy()
    image = array_to_img(image_gen[0])
    return image

# Function to generate multiple Minecraft skins
def generate_multiple_images(num_images):
    latent_dim = 100
    noise = tf.random.normal([num_images, latent_dim])
    model = keras.models.load_model('generator.h5')
    image_gen = model(noise)
    image_gen = (image_gen * 127.5) + 127.5
    image_gen = tf.image.resize(image_gen, (64, 64))
    image_gen = image_gen.numpy()

    generated_images = []
    for i in range(num_images):
        image = array_to_img(image_gen[i])
        generated_images.append(image)

    return generated_images

# Streamlit app code
def main():
    st.title("Minecraft Skin Generator")

    st.sidebar.header("Generator Options")
    generate_single = st.sidebar.button("Generate Single Skin")
    generate_multiple = st.sidebar.button("Generate Multiple Skins")
    
    if generate_single:
        st.subheader("Generated Skin")
        generated_image = generate_single_image()
        st.image(generated_image, use_column_width=True)

        # Convert PIL Image to bytes
        image_bytes = generated_image.tobytes()

        # Add a download button for the generated image
        download_button = st.download_button(
            label="Download Image",
            data=image_bytes,
            key="download_button",
            file_name="generated_skin.png",
        )

    if generate_multiple:
        st.subheader("Generated Skins")
        num_images = 25  # You can adjust the number of images to generate
        generated_images = generate_multiple_images(num_images)

        for i, image in enumerate(generated_images):
            st.image(image, caption=f"Generated Skin {i + 1}", use_column_width=True)

if __name__ == "__main__":
    main()
