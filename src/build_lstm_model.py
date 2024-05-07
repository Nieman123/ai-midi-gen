from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout  

def build_lstm_model(input_shape, num_units=64, dropout_rate=0.3):
    model = Sequential([
        LSTM(num_units, input_shape=(1, input_shape[1]), return_sequences=True),
        Dropout(dropout_rate),
        LSTM(num_units, return_sequences=True),
        Dropout(dropout_rate),
        Dense(num_units, activation='relu'),
        Dense(64, activation='sigmoid')  # Outputting 4 features: start, duration, pitch, velocity
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    return model