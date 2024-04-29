import logging
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
import numpy as np

def split_data(features, labels, test_size=0.2, val_size=0.1):
    logging.info(f"Total number of samples: {len(features)}")

    logging.info(f"First label: {labels[0]}, Type: {type(labels[0])}")
    logging.info(f"Sample labels: {labels[:5]}")

    max_length_features = get_max_length(features)  # Find the max length from the features
    max_length_labels = get_max_length(labels) if isinstance(labels[0], np.ndarray) else None 

    logging.info("Max Features Length is " + str(max_length_features))
    logging.info("Max Features Labels is " + str(max_length_labels))

    """
    Split data into training, validation, and test sets.
    Args:
        features: The input features to the model.
        labels: The target labels corresponding to the features.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Split dataset.
    """
    # First split to carve out the final test set
    X_temp, X_test, y_temp, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    
    # Calculate new validation set size relative to the reduced dataset size (X_temp)
    relative_val_size = val_size / (1 - test_size)
    
    # Second split to separate out the validation set
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=relative_val_size, random_state=42)

    # Pad sequences
    X_train = pad_sequences(X_train, maxlen=max_length_features, padding='post', dtype='float32')
    X_val = pad_sequences(X_val, maxlen=max_length_features, padding='post', dtype='float32')
    X_test = pad_sequences(X_test, maxlen=max_length_features, padding='post', dtype='float32')
    y_train = pad_sequences(y_train, maxlen=max_length_labels, padding='post', dtype='float32')
    y_val = pad_sequences(y_val, maxlen=max_length_labels, padding='post', dtype='float32')
    y_test = pad_sequences(y_test, maxlen=max_length_labels, padding='post', dtype='float32')

    logging.info(f"Shape of X_train after padding: {X_train.shape}")
    logging.info(f"Shape of X_val: {X_val.shape}")
    logging.info(f"Shape of X_test: {X_test.shape}")
    logging.info(f"Shape of y_train after padding: {y_train.shape}")
    logging.info(f"Shape of y_val: {y_val.shape}")
    logging.info(f"Shape of y_test: {y_test.shape}")

    # Convert labels to numpy arrays if not already
    X_train = np.array(X_train, dtype='float32')
    X_val = np.array(X_val, dtype='float32')
    X_test = np.array(X_test, dtype='float32')
    y_train = np.array(y_train, dtype='float32')
    y_val = np.array(y_val, dtype='float32')
    y_test = np.array(y_test, dtype='float32')

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_max_length(features):
    """Calculate the maximum sequence length from the feature list."""
    return max(len(feature) for feature in features)