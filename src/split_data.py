from sklearn.model_selection import train_test_split

def split_data(features, labels, test_size=0.2, val_size=0.1):
    """
    Split data into training, validation, and test sets.
    Args:
        features (np.array): The input features to the model.
        labels (np.array): The target labels corresponding to the features.
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
    
    return X_train, X_val, X_test, y_train, y_val, y_test