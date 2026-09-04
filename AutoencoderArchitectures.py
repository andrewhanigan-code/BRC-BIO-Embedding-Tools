import tensorflow as tf
from keras import layers, models
from PIL import Image
import numpy as np
import os

# =========================
# CONFIG
# =========================
IMAGE_DIR = "" 
IMG_HEIGHT = 128
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 50

# =========================
# DATA PIPELINE
# =========================
def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)

    # Ensure shape is known
    img.set_shape([None, None, 3])

    # Resize
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    img = tf.cast(img, tf.float32) / 255.0

    return img, img  # autoencoder (input, target)


def augment(img, target):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, target


def build_dataset(image_dir):
    file_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # 🔥 Toggle augmentation here
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(500)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def build_orig_conv_model():
    inputs = tf.keras.Input(shape=(128, 64, 3))

    # ===== Encoder =====
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # ===== Decoder =====
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs, outputs)
    return model

def build_fc_auto_model():
    inputs = layers.Input(shape=(128, 64, 3))

    # =========================
    # FLATTEN
    # =========================
    x = layers.Flatten()(inputs)  # 24576

    # =========================
    # ENCODER (gradual compression)
    # =========================
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    # =========================
    # BOTTLENECK (embedding space)
    # =========================
    latent = layers.Dense(64, activation='relu', name="latent_vector")(x)

    # =========================
    # DECODER (gradual expansion)
    # =========================
    x = layers.Dense(128, activation='relu')(latent)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(4096, activation='relu')(x)

    # =========================
    # OUTPUT
    # =========================
    x = layers.Dense(24576, activation='sigmoid')(x)
    outputs = layers.Reshape((128, 64, 3))(x)

    model = models.Model(inputs, outputs)
    return model

# =========================
# BUILD + TRAIN
# =========================
dataset = build_dataset(IMAGE_DIR)

fc_autoencoder = build_fc_auto_model()

fc_autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='mse'
)


fc_autoencoder.fit(
    dataset,
    epochs=EPOCHS
)
fc_autoencoder.save("")

conv_autoencoder = build_orig_conv_model()
conv_autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='mse'
)

conv_autoencoder.fit(
    dataset,
    epochs=EPOCHS
)
conv_autoencoder.save("")