import logging
import random
import numpy as np
import pickle

def preprocess_data(file_path, sequence_length=16, log = False):
    # Load the dataset
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    
    # Calculate max_start and max_duration
    max_start, max_duration = find_max_values(file_path);

    # Create vocabulary and tonic map
    vocab, tonic_map = create_vocab(max_start, max_duration)
    
    logging.info(f"Max duration: {max_duration}")

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

        for i in range(len(notes) - sequence_length):
            sequence = notes[i:i+sequence_length + 1]  # Plus one to include the next note in the sequence
            tokenized_sequence = [tokenize_note(note, root_data, piece_data, vocab, tonic_map) for note in sequence]

            input_seq = np.concatenate(tokenized_sequence[:-1])  # All but last note tokens
            target_seq = np.concatenate(tokenized_sequence[1:])  # All but first note tokens

            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

            if log:
                logging.info(f"Input sequence length: {len(input_seq)}, Target sequence length: {len(target_seq)}")

    return np.array(input_sequences), np.array(target_sequences)

def tokenize_note(note, root_data, piece_data, vocab, tonic_map, log=False):
    if log: logging.info("Tokenizing Note")
    start, end, pitch, velocity = note
    duration = end - start
    if log: logging.info("Calculating Bar")
    bar = start // 8  # Each bar contains 8 eighth notes
    position = start % 8
    if log: logging.info(f"Bar: {bar} Position: {position}")
    
    # Safeguard for index out of range
    if bar >= len(root_data) or position >= len(root_data[bar]):
        if log: logging.warning(f"Root data index out of range: bar={bar}, position={position}")
        root_note = 0
    else:
        root_note = root_data[bar][position]
    
    if log: logging.info(f"Root note: {root_note}")
    
    try:
        if log: logging.info(f"Start: {start}")
        start_token = vocab['start_time'].get(start, 0)
        if log: logging.info(f"Duration: {duration}")
        duration_token = vocab['duration'].get(duration, 0)
        if log: logging.info(f"Pitch: {pitch}")
        pitch_token = vocab['pitch'].get(pitch, 0)
        if log: logging.info(f"Velocity: {velocity}")
        velocity_token = vocab['velocity'].get(velocity, 0)
        if log: logging.info(f"Root note token: {root_note}")
        root_token = vocab['root'].get(root_note, 0)
        if log: logging.info(f"Mode: {piece_data['mode']}")
        mode_token = vocab['mode'][piece_data['mode']]
        if log: logging.info(f"Style: {piece_data['style']}")
        style_token = vocab['style'][piece_data['style']]
        if log: logging.info(f"Tonic: {piece_data['tonic']}")
        tonic_token = vocab['tonic'].get(tonic_map[int(piece_data['tonic'])], 0)
        
        tokens = [
            start_token,
            duration_token,
            pitch_token,
            velocity_token,
            root_token,
            mode_token,
            style_token,
            tonic_token
        ]
        
        if log: logging.info(f"Tokens: {tokens}")
        return tokens
    
    except KeyError as e:
        logging.error(f"KeyError: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

    
    except KeyError as e:
        logging.error(f"KeyError: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise
    
    except KeyError as e:
        logging.error(f"KeyError: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

def create_vocab(max_start, max_duration):
    note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    tonic_map = {i: note_names[i] for i in range(12)}

    vocab = {
        'pitch': {pitch: idx for idx, pitch in enumerate(range(21, 109), 1)},  # MIDI pitches range from 21 to 108
        'velocity': {vel: idx+89 for idx, vel in enumerate(range(128))},  # MIDI velocities range from 0 to 127, index starts after pitch
        'duration': {dur: idx+217 for idx, dur in enumerate(range(max_duration + 1))},  # Duration based on max_duration
        'start_time': {start: idx+217+max_duration for idx, start in enumerate(range(max_start + 1))},  # Start time based on max_start
        'root': {root: idx+217+max_duration+max_start for idx, root in enumerate(range(12))},  # 12 roots
        'mode': {'M': 261, 'm': 262},  # Two modes
        'style': {'pop_standard': 263, 'pop_complex': 264, 'dark': 265, 'r&b': 266, 'unknown': 267},  # Five styles
        'tonic': {note: idx+268 for idx, note in enumerate(note_names)}  # 12 tonics
    }

    return vocab, tonic_map


def find_max_values(file_path):
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    max_start = 0
    max_duration = 0

    for piece_data in dataset.values():
        notes = np.array(piece_data['nmat'])
        max_start = max(max_start, np.max(notes[:, 0]))
        durations = notes[:, 1] - notes[:, 0]
        max_duration = max(max_duration, np.max(durations))

    logging.info(f"Max Duration: {max_duration}")
    return max_start, max_duration

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