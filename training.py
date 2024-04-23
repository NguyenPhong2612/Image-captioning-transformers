import keras
import tensorflow as tf
from make_dataset import train_dataset, valid_dataset
from model import get_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel, image_augmentation, LRSchedule


EMBED_DIM = 512
FF_DIM = 512
EPOCHS = 30

cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(
    embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
decoder = TransformerDecoderBlock(
    embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model,
    encoder=encoder,
    decoder=decoder,
    image_aug=image_augmentation,
)


early_stopping = keras.callbacks.EarlyStopping(
    patience=3, restore_best_weights=True)

caption_model.compile(optimizer=keras.optimizers.Adam(learning_rate = 3e-4), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
)

caption_model.save_weights("Save/Weight/model.npy")
