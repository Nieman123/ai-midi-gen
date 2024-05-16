from transformer_model import build_transformer_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard, ModelCheckpoint
import tensorflow as tf
import math
import os
import datetime

def train_model(X_train, y_train, X_val, y_val, vocab_size, sequence_length=128, batch_size=32):
    # Reshape the data to match the expected input shape of the transformer model
    X_train = X_train.reshape((X_train.shape[0], sequence_length))
    X_val = X_val.reshape((X_val.shape[0], sequence_length))
    y_train = y_train.reshape((y_train.shape[0], sequence_length))
    y_val = y_val.reshape((y_val.shape[0], sequence_length))

    train_dataset = create_dataset(X_train, y_train, batch_size)
    val_dataset = create_dataset(X_val, y_val, batch_size)

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    lrate = LearningRateScheduler(step_decay)

    # TensorBoard
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    #Checkpoints
    checkpoint_filepath = 'checkpoints/checkpoint'
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    model = build_transformer_model(input_shape=(sequence_length,), vocab_size=vocab_size)

    # Load the latest checkpoint if it exists
    if os.path.exists(checkpoint_filepath):
        model.load_weights(checkpoint_filepath)
        print("Loaded model from checkpoint")

    model.fit(train_dataset, epochs=100, batch_size=32, validation_data=val_dataset, callbacks=[early_stopping, lrate, tensorboard_callback, checkpoint_callback])
    model.save("transformer_midi_gen.keras")
    return model

def create_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate