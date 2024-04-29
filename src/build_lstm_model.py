import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout  

def build_lstm_model(input_shape, num_units=64, dropout_rate=0.3, output_units=206):
    model = Sequential([
        LSTM(num_units, input_shape=input_shape, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(num_units, return_sequences=True),
        Dropout(dropout_rate),
        Dense(num_units, activation='relu'),
        Dense(1, activation='sigmoid')  # Assume binary classification for simplicity
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model