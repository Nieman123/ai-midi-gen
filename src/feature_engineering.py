import numpy as np 

def normalize_pitch(pitches):
    """ Normalize MIDI pitches to a 0-1 range based on MIDI standards (0-127). """
    return pitches / 127.0

def encode_durations(starts, ends):
    """ Convert start and end times to durations and normalize. """
    durations = ends - starts
    max_duration = durations.max()
    return durations / max_duration

def create_features(notes):
    """ Create a feature matrix from notes data. """
    pitches = normalize_pitch(notes[:, 2])
    velocities = notes[:, 3] / 127.0  # Normalize velocities in the same way as pitches
    durations = encode_durations(notes[:, 0], notes[:, 1])
    return np.stack([pitches, velocities, durations], axis=1)