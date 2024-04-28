import argparse
import logging
from preprocess_data import preprocess_data
from feature_engineering import create_features
from train_model import train_model
from split_data import split_data

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="AI MIDI Generator")
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing')
    parser.add_argument('--feature_engineering', action='store_true', help='Run feature engineering')
    parser.add_argument('--train', action='store_true', help='Run model training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--generate', action='store_true', help='Generate new MIDI progressions')

    args = parser.parse_args()

    features, labels = None, None

    try:
        if args.preprocess:
            logging.info("Starting preprocessing...")
            features, labels = preprocess_data('dataset\dataset.pkl')
            logging.info(f"Sample preprocessed data: {features[:1]}")

        if args.train:
            if features is None or labels is None:
                logging.error("Training requires preprocessed data. Run with --preprocess first.")
                return
            logging.info("Starting model training...")
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, labels)
            model = train_model(X_train, y_train, X_val, y_val, input_shape=X_train.shape[1:])
            logging.info("Model training complete.")

        if args.evaluate:
            # evaluate_model(model)
            pass

        if args.generate:
            # generate_progressions(model)
            pass

    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == '__main__':
    main()
