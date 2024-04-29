from build_lstm_model import build_lstm_model

def train_model(X_train, y_train, X_val, y_val, input_shape):
    model = build_lstm_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    model.save("crappy_midi_gen.keras")
    return model