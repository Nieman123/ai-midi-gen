import pickle
import logging
from feature_engineering import create_features
import numpy as np

def preprocess_data(file_path):
    # Load the dataset
    with open('dataset\dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)

        features = []
        labels = []

    # Loop through each piece in the dataset
    for piece_name, piece_data in dataset.items():
        nmat = np.array(piece_data['nmat'])  # Note matrix
        if len(nmat) > 1:
            feature_seq = create_features(nmat[:-1])
            label_seq = create_features(nmat[1:])
            features.append(feature_seq)
            labels.append(label_seq[:, 2])
    
    return features, labels