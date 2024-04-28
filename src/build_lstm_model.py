import tensorflow as tf
from tensorflow import Sequential
from tensorflow import LSTM, Dense, Dropout

def build_lstm_model(input_shape, num_units=64, dropout_rate=0.3, output_units=1):
    model = Sequential([
        LSTM(num_units, input_shape=input_shape, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(num_units),
        Dropout(dropout_rate),
        Dense(num_units, activation='relu'),
        Dense(output_units, activation='sigmoid')  # Assume binary classification for simplicity
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model