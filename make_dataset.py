import keras
import tensorflow as tf
import os
import numpy as np
import re
import pandas as pd

IMAGES_PATH = "Flickr8k/Images"
CAPTION_PATH = "Flickr8k/captions.txt"
IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 8800


SEQ_LENGTH = 25
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE



df = pd.read_csv(CAPTION_PATH)
image_ids = df['image'].values
captions = df['caption'].values

dataset = []

for image_id, caption in zip(image_ids, captions):
    path = os.path.join(IMAGES_PATH, image_id)
    dataset.append([path, caption])

caption_mapping = {}
text_data = []
image_paths = [] # X_en_data
left_pad_captions = [] # X_de_data
right_pad_captions = [] # Y_data

for img_name, caption in dataset:
    if img_name.endswith("jpg"):
        left_pad_captions.append("startseq " + caption.strip().replace(".", ""))
        right_pad_captions.append(caption.strip().replace(".", "") + " endseq")
        text_data.append(
            "startseq " + caption.strip().replace(".", "") + " endseq")
        image_paths.append(img_name)

        if img_name in caption_mapping:
            caption_mapping[img_name].append(caption)
        else:
            caption_mapping[img_name] = [caption]

np.save("Save/text_data.npy", text_data)

train_size = 0.8
shuffle = True
np.random.seed(42)

zipped = list(zip(image_paths, left_pad_captions, right_pad_captions))
np.random.shuffle(zipped)
image_paths, left_pad_captions, right_pad_captions = zip(*zipped)

train_size = int(len(image_paths)*train_size)
train_image_paths = list(image_paths[:train_size])
train_left_pad_captions = list(left_pad_captions[:train_size])
train_right_pad_captions = list(right_pad_captions[:train_size])

val_image_paths = list(image_paths[train_size:])
val_left_pad_captions = list(left_pad_captions[train_size:])
val_right_pad_captions = list(right_pad_captions[train_size:])


strip_chars = "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~"

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f'{re.escape(strip_chars)}', '')


vectorization = keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)

vectorization.adapt(text_data)

vocab = np.array(vectorization.get_vocabulary())
np.save('Save/vocabulary.npy', vocab)


def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(img_cap, y_captions):
    img_path, x_captions = img_cap
    return ((decode_and_resize(img_path), vectorization(x_captions)), vectorization(y_captions))


def make_dataset(images, x_captions, y_captions):
    dataset = tf.data.Dataset.from_tensor_slices(
        ((images, x_captions), y_captions))
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset


train_dataset = make_dataset(train_image_paths, train_left_pad_captions, train_right_pad_captions)
valid_dataset = make_dataset(val_image_paths, val_left_pad_captions, val_right_pad_captions)
print(len(train_dataset))
print(len(valid_dataset))