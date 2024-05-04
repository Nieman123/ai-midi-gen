import pickle
import logging
import random
from feature_engineering import create_features, find_max_duration
import numpy as np

def preprocess_data(file_path):
    # Load the dataset
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)

    features = []
    labels = []

    # Iterate through each piece in the dataset
    for piece_name, piece_data in dataset.items():
        nmat = np.array(piece_data['nmat'])  # Convert to numpy array for easier manipulation
        grouped_notes = group_notes_by_chord(nmat)  # Group notes into chords and melody

        for i in range(len(grouped_notes) - 1):
            current_group = grouped_notes[i]
            next_group = grouped_notes[i + 1]

            # Separate handling for chords and melody
            chord_features, melody_features = separate_chords_melody(current_group)
            chord_labels, melody_labels = separate_chords_melody(next_group)

            # Combine chord and melody features into one vector
            feature_vector = np.concatenate([chord_features, melody_features])
            label_vector = np.concatenate([chord_labels, melody_labels])

            features.append(feature_vector)
            labels.append(label_vector)

            # Logging the processing information
            logging.info(f"Processed piece: {piece_name}, Group index: {i}")
            logging.info(f"Feature vector shape: {feature_vector.shape}")
            logging.info(f"Label vector shape: {label_vector.shape}")

    return np.array(features), np.array(labels)

def group_notes_by_chord(nmat):
    """ Group notes by their start times and durations into chords and melodies. """
    note_groups = {}
    for note in nmat:
        start_time = note[0]
        duration = note[1] - note[0]
        group_key = (start_time, duration)  # Group by both start and duration

        if group_key not in note_groups:
            note_groups[group_key] = []
        note_groups[group_key].append(note)

    # Collect groups in order of their start times
    return [note_groups[key] for key in sorted(note_groups, key=lambda x: x[0])]

def separate_chords_melody(notes):
    # Assuming chords are notes with the same start time and similar durations
    # Melody notes may have unique timings or extended durations
    chords = []
    melody = []
    if not notes:
        return np.array([]), np.array([])
    # Example logic, needs refining based on actual data characteristics
    duration_threshold = np.median([note[1] - note[0] for note in notes])  # median duration
    for note in notes:
        if note[1] - note[0] < duration_threshold:
            chords.append(note)
        else:
            melody.append(note)
    return create_features(chords), create_features(melody)

def explore_data(dataset):
    import matplotlib.pyplot as plt

    starts = []
    durations = []
    pitches = []

    sample_keys = random.sample(list(dataset.keys()), 5)
    examples = [(key, np.array(dataset[key]['nmat'][:5])) for key in sample_keys]

    for piece_name, piece_data in dataset.items():
        nmat = np.array(piece_data['nmat'])
        starts.extend(nmat[:, 0])
        durations.extend(nmat[:, 1] - nmat[:, 0])
        pitches.extend(nmat[:, 2])

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.hist(starts, bins=50, color='blue')
    plt.title('Note Starts')
    plt.xlabel('Start Time')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(durations, bins=50, color='red')
    plt.title('Note Durations')
    plt.xlabel('Duration')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.hist(pitches, bins=50, color='green')
    plt.title('Pitch Usage')
    plt.xlabel('MIDI Pitch Number')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Output some examples from the dataset
    for name, example in examples:
        print(f"Example from {name}:")
        print("Starts, Ends, Pitches, Velocities")
        for note in example:
            print(f"{note[0]}, {note[1]}, {note[2]}, {note[3]}")

    # Optionally, output some numeric statistics
    print(f"Average note start: {np.mean(starts):.2f}, Average duration: {np.mean(durations):.2f}, Average pitch: {np.mean(pitches):.2f}")
    print(f"Total pieces analyzed: {len(dataset)}")