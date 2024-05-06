import logging
import numpy as np 

def create_features(notes, total_positions=8, num_features=4):
    log = False

    if log: logging.info(f"Notes shape: {notes.shape} with data: {notes}")

    """ Create a feature matrix from notes data. Ensure the input is not empty and has the expected dimensions. """
    
    if len(notes) == 0:
        return np.zeros((0, num_features))

    # Ensure notes are at least 2D (even if it's a single note)
    if notes.ndim == 1:
        notes = np.expand_dims(notes, axis=0)

    if log: logging.info(f"Notes shape: {notes.shape} with data: {notes}")

    pitches = normalize_pitch(notes[:, 2])
    velocities = normalize_velocity(notes[:, 3])
    normalized_starts = normalize_quantized_starts(notes[:, 0], total_positions)
    durations = normalize_durations(notes[:, 0], notes[:, 1], max_duration=500)

    features = np.stack([normalized_starts, durations, pitches, velocities], axis=1)

    # Log the final feature matrix
    if log: logging.info(f"Final feature matrix shape: {features.shape} with data: {features}")
    
    return features



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