from build_lstm_model import build_lstm_model

def train_model(features, labels, input_shape):
    model = build_lstm_model(input_shape)
    model.fit(features, labels, epochs=10, batch_size=32, validation_split=0.2)
    return model
