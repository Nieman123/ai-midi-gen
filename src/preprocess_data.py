import logging
import random
import numpy as np
import pickle

vocab = {
    'pitch': {pitch: idx for idx, pitch in enumerate(range(21, 109), 1)},  # MIDI pitches range from 21 to 108
    'velocity': {vel: idx+89 for idx, vel in enumerate(range(128))},  # MIDI velocities range from 0 to 127, index starts after pitch
    'duration': {dur: idx+217 for idx, dur in enumerate(range(16))},  # 16 possible durations, index starts after velocity
    'start_time': {start: idx+233 for idx, start in enumerate(range(16))},  # 16 possible start times, index starts after duration
    'root': {root: idx+249 for idx, root in enumerate(range(12))},  # 12 roots, index starts after start_time
    'mode': {'M': 261, 'm': 262},  # Two modes, index starts after root
    'style': {'pop_standard': 263, 'pop_complex': 264, 'dark': 265, 'r&b': 266, 'unknown': 267},  # Five styles, index starts after mode
    'tonic': {note: idx+268 for idx, note in enumerate(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])}  # 12 tonics, index starts after style
}

def preprocess_data(file_path, sequence_length=16):
    log = True  # Set to True for detailed debugging
    # Load the dataset
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)

    input_sequences = []
    target_sequences = []

    # Iterate through each piece in the dataset
    for piece_name, piece_data in dataset.items():
        notes = np.array(piece_data['nmat'])  # Convert to numpy array for easier manipulation
        indices = np.argsort(notes[:, 0])     # Sort notes by start time
        notes = notes[indices]
        root_data = piece_data['root']  # This is a 2D matrix where each row corresponds to a bar

        if log:
            logging.info(f"Processing piece: {piece_name}")
            logging.info(f"Combined note data: {notes}")
            logging.info(f"Root data: {root_data}")

        for i in range(len(notes) - sequence_length):
            sequence = notes[i:i+sequence_length + 1]  # Plus one to include the next note in the sequence
            tokenized_sequence = [tokenize_note(note, root_data, piece_data, vocab) for note in sequence]

            input_seq = np.concatenate(tokenized_sequence[:-1])  # All but last note tokens
            target_seq = np.concatenate(tokenized_sequence[1:])  # All but first note tokens

            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

            if log:
                logging.info(f"Input sequence length: {len(input_seq)}, Target sequence length: {len(target_seq)}")

    return np.array(input_sequences), np.array(target_sequences)

def tokenize_note(note, root_data, piece_data, vocab):
    logging.info("Tokenizing Note")
    start, end, pitch, velocity = note
    duration = end - start
    logging.info("Calculating Bar")
    bar = start // 8  # Each bar contains 8 eighth notes
    position = start % 8
    logging.info(f"Bar: {bar} Position: {position}")
    # Safeguard for index out of range
    if bar >= len(root_data) or position >= len(root_data[bar]):
        logging.warning(f"Root data index out of range: bar={bar}, position={position}")
        root_note = 0
    else:
        root_note = root_data[bar][position]
    logging.info(f"Root note: {root_note}")
    tokens = [
        vocab['start_time'].get(start, 0),
        vocab['duration'].get(duration, 0),
        vocab['pitch'].get(pitch, 0),
        vocab['velocity'].get(velocity, 0),
        vocab['root'].get(root_note, 0),
        vocab['mode'][piece_data['mode']],
        vocab['style'][piece_data['style']],
        vocab['tonic'][piece_data['tonic']]
    ]
    return tokens

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
        print(f"Note data: {example}")
        print("Starts, Ends, Pitches, Velocities")
        for note in example:
            print(f"{note[0]}, {note[1]}, {note[2]}, {note[3]}")

    # Optionally, output some numeric statistics
    print(f"Average note start: {np.mean(starts):.2f}, Average duration: {np.mean(durations):.2f}, Average pitch: {np.mean(pitches):.2f}")
    print(f"Total pieces analyzed: {len(dataset)}")