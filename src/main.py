import argparse
import logging
import coloredlogs
import pickle
import numpy as np
from keras.models import load_model
from midi_gen import generate_midi, play_midi, sequence_to_midi
from preprocess_data import preprocess_data, explore_data
from feature_engineering import find_max_duration
from train_model import train_model
from split_data import split_data

def main(args):
    
    logging.basicConfig(level=logging.INFO)

    features, labels = None, None

    try:
        if args.preprocess:
            
            features, labels, vocab_size = preprocess_data('dataset/dataset.pkl')

            sample_count = min(len(features), 5)  # Ensure not to sample more than exists
            if sample_count > 0:
                # Generate random indices
                indices = np.random.choice(len(features), sample_count, replace=False)

                # Log information about the randomly selected samples
                for idx in indices:
                    logging.info(f"Random Sample features at index {idx}: {features[idx]}")
                    logging.info(f"Random Sample labels at index {idx}: {labels[idx]}")
                    logging.info(f"Shape of features at index {idx}: {features[idx].shape}")
                    logging.info(f"Shape of labels at index {idx}: {labels[idx].shape}")

                # Optional: Log statistical summary
                logging.info(f"Features mean: {np.mean(features, axis=0)}")
                logging.info(f"Features std dev: {np.std(features, axis=0)}")
                logging.info(f"Labels mean: {np.mean(labels, axis=0)}")
                logging.info(f"Labels std dev: {np.std(labels, axis=0)}")

        if args.train:
            if features is None or labels is None:
                logging.error("Training requires preprocessed data. Run with --preprocess first.")
                return
            logging.info("Starting model training...")
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels)
            model = train_model(X_train, y_train, X_val, y_val, vocab_size=vocab_size)
            logging.info("Model training complete.")

        if args.evaluate:
            # evaluate_model(model)
            pass

        if args.generate:
            model = tf.keras.models.load_model("transformer_midi_gen.keras")
            logger.info("Model loaded successfully.")
            
            # Retrieve the embedding layer
            embedding_layer = model.layers[1]

            if isinstance(embedding_layer, tf.keras.layers.Embedding):
                total_tokens = embedding_layer.input_dim
                logger.info(f"Total Tokens: {total_tokens}")
            else:
                logger.error("The retrieved layer is not an Embedding layer")
                return

            seed_sequence = [np.random.randint(0, total_tokens) for _ in range(128)]

            generated_sequence = generate_midi(model, seed_sequence, sequence_length=128, num_notes=100, total_tokens=total_tokens)
            sequence_to_midi(generated_sequence, output_file='generated_midi.mid', tempo=120)
            play_midi('generated_midi.mid')

            logger.info("Main process finished.")

        
        if args.explore:
            explore_data('dataset/dataset.pkl')


    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MIDI Generation Script")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the data')
    parser.add_argument('--generate', action='store_true', help='Generate MIDI')
    parser.add_argument('--data_path', type=str, help='Path to the dataset file')
    args = parser.parse_args()

    main(args)
