import argparse
import logging
import preprocess_data
from feature_engineering import create_features
from train_model import train_model
# from train_model import train_model
# from evaluate_model import evaluate_model
# from generate_progressions import generate

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="AI MIDI Generator")
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing')
    parser.add_argument('--feature_engineering', action='store_true', help='Run feature engineering')
    parser.add_argument('--train', action='store_true', help='Run model training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--generate', action='store_true', help='Generate new MIDI progressions')


    args = parser.parse_args()

    try:
        if args.preprocess:
            logging.info("Starting preprocessing...")
            logging.info(f"Preprocessed data: {data[:1]}")
            data = preprocess_data('dataset\dataset.pkl')

        if args.feature_engineering:
            logging.info("Starting feature creation...")
            features = create_features(data)

        if args.train:
            logging.info("Starting model training...")
            # You would need to have your labels defined somewhere, or adjust how you handle data
            model = train_model(features, labels, input_shape=features.shape[1:])  # Assuming features is an array

        if args.evaluate:
            # evaluate_model(model)
            pass

        if args.generate:
            # generate_progressions(model)
            pass

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == '__main__':
    main()