from build_lstm_model import build_lstm_model
from keras.callbacks import EarlyStopping, LearningRateScheduler
import math

def train_model(X_train, y_train, X_val, y_val, input_shape):
    # Reshape the data to add the time_steps dimension (1 in this case)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
    y_val = y_val.reshape((y_val.shape[0], 1, y_val.shape[1]))

    input_shape = (1, X_train.shape[2])  # (time_steps, features)
    
    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Learning Rate Scheduler
    def step_decay(epoch):
        initial_lrate = 0.001
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    lrate = LearningRateScheduler(step_decay)

    model = build_lstm_model(input_shape)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, lrate])
    model.save("crappy_midi_gen.keras")
    return model
