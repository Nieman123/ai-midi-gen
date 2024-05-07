import pickle
import logging
import random
from feature_engineering import create_features, find_max_duration
import numpy as np

def preprocess_data(file_path, sequence_length=16):
    log = False  # Set to True for detailed debugging
    # Load the dataset
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)

    max_duration = find_max_duration(dataset)
    
    logging.info(f"Max duration: {max_duration}")

    features = []
    labels = []

    # Iterate through each piece in the dataset
    for piece_name, piece_data in dataset.items():
        notes = np.array(piece_data['nmat'])  # Convert to numpy array for easier manipulation

        indices = np.argsort(notes[:, 0])
        notes = notes[indices]

        if log:
            logging.info(f"Processing piece: {piece_name}")
            logging.info(f"Combined note data: {notes}")

        for i in range(len(notes) - sequence_length):
            window = notes[i:i+sequence_length]
            next_window = notes[i+1:i+1+sequence_length]

            feature_vector = create_features(window, sequence_length, max_duration)
            label_vector = create_features(next_window, sequence_length, max_duration)

            features.append(feature_vector)
            labels.append(label_vector)

            if log:
                logging.info(f"Feature vector shape: {feature_vector.shape}, Label vector shape: {label_vector.shape}")

    return np.array(features), np.array(labels)


def group_notes_by_roles(nmat):
    log = False
    # Group notes initially by start time and duration
    grouped_notes = group_notes_by_chord(nmat)
    
    chords = []
    melodies = []
    
    # Further classify the grouped notes
    for index, group in enumerate(grouped_notes):
        if log: logging.info(f"Processing group {index} with {len(group)} notes")
        
        if len(group) > 1:
            # Typically, a chord is expected to have multiple notes played together
            chords.extend(group)  # Add all notes in this group to chords
            if log: logging.info(f"Added to chords: {group}")
        else:
            # Single notes or uniquely timed notes can be considered part of the melody
            melodies.extend(group)
            if log: logging.info(f"Added to melodies: {group}")
    
    if log: logging.info(f"Total chords collected: {len(chords)}")
    if log: logging.info(f"Total melodies collected: {len(melodies)}")
    if log: logging.info(f"Grouped notes: {grouped_notes}")
    if log: logging.info(f"Melody notes: {melodies}")
    if log: logging.info(f"Chord notes: {chords}")
     # Ensure the output is always a 2D array
    chords = np.array(chords).reshape(-1, 4) if chords else np.empty((0, 4))
    melodies = np.array(melodies).reshape(-1, 4) if melodies else np.empty((0, 4))
    
    return chords, melodies

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