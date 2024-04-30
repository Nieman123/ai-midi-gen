import numpy as np 

def create_features(notes, total_positions=8):  # Example max_duration, adjust based on data exploration
    """ Create a feature matrix from notes data. """
    pitches = normalize_pitch(notes[:, 2])
    velocities = normalize_velocity(notes[:, 3])
    normalized_starts = normalize_quantized_starts(notes[:, 0], total_positions)
    durations = normalize_durations(notes[:, 0], notes[:, 1], max_duration=500)
    return np.stack([normalized_starts, durations, pitches, velocities], axis=1)

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