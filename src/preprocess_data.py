import logging
import numpy as np
import pickle

def preprocess_data(file_path, sequence_length=16, log=False):
    # Load the dataset
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    
    # Calculate max_start and max_duration
    max_start, max_duration = find_max_values(file_path)

    # Create vocabulary and tonic map
    vocab, tonic_map, total_tokens = create_vocab(max_start, max_duration)
    
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

    return np.array(input_sequences), np.array(target_sequences), total_tokens

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

    # Validate ranges
    max_start = 64  # Example value, adjust based on your dataset analysis
    max_duration = 26  # Example value, adjust based on your dataset analysis
    min_pitch, max_pitch = 12, 99
    min_velocity, max_velocity = 0, 127
    valid_roots = set(range(12))
    valid_modes = {'M', 'm'}
    valid_styles = {'pop_standard', 'pop_complex', 'dark', 'r&b', 'unknown'}
    valid_tonics = set(range(12))

    if not (0 <= start <= max_start):
        logging.error(f"Start time out of range: {start}")
    if not (0 <= duration <= max_duration):
        logging.error(f"Duration out of range: {duration}")
    if not (min_pitch <= pitch <= max_pitch):
        logging.error(f"Pitch out of range: {pitch}")
    if not (min_velocity <= velocity <= max_velocity):
        logging.error(f"Velocity out of range: {velocity}")
    if root_note not in valid_roots:
        logging.error(f"Root note out of range: {root_note}")
    if piece_data['mode'] not in valid_modes:
        logging.error(f"Mode out of range: {piece_data['mode']}")
    if piece_data['style'] not in valid_styles:
        logging.error(f"Style out of range: {piece_data['style']}")
    if int(piece_data['tonic']) not in valid_tonics:
        logging.error(f"Tonic out of range: {piece_data['tonic']}")
    
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

def create_vocab(max_start, max_duration):
    note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    tonic_map = {i: note_names[i] for i in range(12)}

    vocab = {
        'pitch': {pitch: idx for idx, pitch in enumerate(range(12, 100), 1)},  # MIDI pitches range from 12 to 99
        'velocity': {vel: idx+88 for idx, vel in enumerate(range(128))},  # MIDI velocities range from 0 to 127, index starts after pitch
        'duration': {dur: idx+216 for idx, dur in enumerate(range(max_duration + 1))},  # Duration based on max_duration
        'start_time': {start: idx+216+max_duration for idx, start in enumerate(range(max_start + 1))},  # Start time based on max_start
        'root': {root: idx+216+max_duration+max_start for idx, root in enumerate(range(12))},  # 12 roots
        'mode': {'M': 260, 'm': 261},  # Two modes
        'style': {'pop_standard': 262, 'pop_complex': 263, 'dark': 264, 'r&b': 265, 'unknown': 266},  # Five styles
        'tonic': {note: idx+267 for idx, note in enumerate(note_names)}  # 12 tonics
    }

    total_tokens = max(
        max(vocab['pitch'].values()),
        max(vocab['velocity'].values()),
        max(vocab['duration'].values()),
        max(vocab['start_time'].values()),
        max(vocab['root'].values()),
        max(vocab['mode'].values()),
        max(vocab['style'].values()),
        max(vocab['tonic'].values())
    ) + 1 

    return vocab, tonic_map, total_tokens

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

def explore_data(file_path):
    # Load the dataset
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)

    # Initialize variables to store information about notes
    start_times = []
    durations = []
    pitches = []
    velocities = []

    for piece_name, piece_data in dataset.items():
        notes = np.array(piece_data['nmat'])
        
        start_times.extend(notes[:, 0])
        durations.extend(notes[:, 1] - notes[:, 0])
        pitches.extend(notes[:, 2])
        velocities.extend(notes[:, 3])

    # Calculate statistics
    avg_start = np.mean(start_times)
    avg_duration = np.mean(durations)
    avg_pitch = np.mean(pitches)
    avg_velocity = np.mean(velocities)

    logging.info(f"Average start time: {avg_start}")
    logging.info(f"Average duration: {avg_duration}")
    logging.info(f"Average pitch: {avg_pitch}")
    logging.info(f"Average velocity: {avg_velocity}")

    # Additional logging for out-of-range pitches
    pitch_min = np.min(pitches)
    pitch_max = np.max(pitches)
    logging.info(f"Pitch range: {pitch_min} to {pitch_max}")

    out_of_range_pitches = [pitch for pitch in pitches if pitch < 12 or pitch > 99]
    if out_of_range_pitches:
        logging.warning(f"Out-of-range pitches detected: {out_of_range_pitches}")
