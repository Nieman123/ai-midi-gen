from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import math

def build_lstm_model(input_shape, num_units=64, dropout_rate=0.3):
    model = Sequential([
        Bidirectional(LSTM(num_units, return_sequences=True, input_shape=(1, input_shape[1]))),
        BatchNormalization(),
        Dropout(dropout_rate),
        Bidirectional(LSTM(num_units, return_sequences=True)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(num_units, activation='relu'),
        Dense(64, activation='sigmoid')  # Outputting 4 features: start, duration, pitch, velocity
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)
