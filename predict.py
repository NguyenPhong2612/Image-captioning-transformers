import keras
import numpy as np
import tensorflow as tf
import re
from model import get_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel, image_augmentation

SEQ_LENGTH = 25
VOCAB_SIZE = 8800
IMAGE_SIZE = (299, 299)
EMBED_DIM = 512
FF_DIM = 512

print("model loading ...")
cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, 
                                  dense_dim=FF_DIM, 
                                  num_heads=1)

decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, 
                                  ff_dim=FF_DIM, 
                                  num_heads=2)

loaded_model = ImageCaptioningModel(
    cnn_model=cnn_model,
    encoder=encoder,
    decoder=decoder,
    image_aug=image_augmentation)

loaded_model.compile(optimizer=keras.optimizers.Adam(learning_rate = 3e-4), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

loaded_model.load_weights("Save/Weight/model.npy")
print("model loaded...")

vocab = np.load("Save/vocabulary.npy")
print("vocab loaded...")

data_txt = np.load("Save/text_data.npy").tolist()
print("vectorization data loaded...")

index_lookup = dict(zip(range(len(vocab)), vocab))
print("index lookup loaded...")
max_decoded_sentence_length = SEQ_LENGTH - 1
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

vectorization.adapt(data_txt)
print("vectorization adapted...")


def generate_caption(image):
    
    if isinstance(image, np.ndarray):
        img = tf.constant(image)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)
        
    img = tf.expand_dims(img, 0)
    img = loaded_model.cnn_model(img)
    encoded_img = loaded_model.encoder(img, training=False)

    decoded_caption = "startseq "
    for i in range(SEQ_LENGTH - 1):
        tokenized_caption = vectorization([decoded_caption])
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = loaded_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "endseq":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("startseq ", "")
    decoded_caption = decoded_caption.replace(" endseq", "").strip()
    return decoded_caption
