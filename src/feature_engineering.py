import logging
import numpy as np 

def create_features(notes, sequence_length=16, num_features_per_note=4):
    log = True  # Set this to True if you want detailed logging

    if log:
        logging.info(f"Notes shape: {notes.shape} with data: {notes}")

    # Initialize a feature matrix for all notes in the sequence
    if notes.size == 0:
        # Return a zero-filled vector if there are no notes
        return np.zeros((sequence_length * num_features_per_note,))

    # Ensure notes are at least 2D (even if it's a single note)
    if notes.ndim == 1:
        notes = np.expand_dims(notes, axis=0)

    # Normalization steps
    pitches = normalize_pitch(notes[:, 2])
    velocities = normalize_velocity(notes[:, 3])
    normalized_starts = normalize_quantized_starts(notes[:, 0], sequence_length)
    durations = normalize_durations(notes[:, 0], notes[:, 1], max_duration=500)

    # Create a feature matrix for this sequence
    features = np.stack([normalized_starts, durations, pitches, velocities], axis=1)

    # Flatten the feature matrix to create a single feature vector
    feature_vector = features.flatten()

    if log:
        logging.info(f"Final feature vector shape: {feature_vector.shape} with data: {feature_vector}")

    return feature_vector



def normalize_pitch(pitches):
    """ Normalize MIDI pitches to a 0-1 range based on MIDI standards (0-127). """
    return pitches / 127.0

def normalize_velocity(velocities):
    """ Normalize MIDI velocities to a 0-1 range based on MIDI standards (0-127). """
    return velocities / 127.0

def normalize_durations(starts, ends, max_duration):
    """ Normalize durations by a predetermined maximum duration to ensure consistency across the dataset. """
    durations = ends - starts
    return durations / max_duration

def normalize_quantized_starts(starts, total_positions):
    """ Normalize quantized start times to a range between 0 and 1. """
    return starts / (total_positions - 1) 

def find_max_duration(dataset):
    max_duration = 0
    for piece_data in dataset.values():
        nmat = np.array(piece_data['nmat'])
        durations = nmat[:, 1] - nmat[:, 0]
        local_max = np.max(durations)
        if local_max > max_duration:
            max_duration = local_max
    return max_duration