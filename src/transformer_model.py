import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam

def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def build_transformer_model(input_shape, vocab_size, num_heads=8, num_layers=6, d_model=512, dff=2048, rate=0.1):
    inputs = Input(shape=input_shape)

    # Embedding and Positional Encoding
    embedding = Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)
    positional_encoding = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
    positional_encoding = Embedding(input_dim=512, output_dim=d_model)(positional_encoding)
    
    x = embedding + positional_encoding

    # Transformer Blocks
    for _ in range(num_layers):
        x = transformer_block(x, head_size=d_model, num_heads=num_heads, ff_dim=dff, dropout=rate)

    # Ensure correct output shape
    outputs = Dense(vocab_size, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
