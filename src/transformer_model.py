import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    attn_output = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(out1.shape[-1])(ffn_output)  # Ensure the output dimension matches input for residual connection
    ffn_output = Dropout(dropout)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def build_transformer_model(input_shape, total_tokens, embed_dim=128, num_heads=8, ff_dim=128, num_layers=4):
    inputs = Input(shape=input_shape)
    embedding_layer = Embedding(input_dim=total_tokens, output_dim=embed_dim)(inputs)

    x = embedding_layer
    for _ in range(num_layers):
        x = transformer_block(x, head_size=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=0.1)

    outputs = Dense(total_tokens, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model