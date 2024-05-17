from mido import MidiFile, MidiTrack, Message
import logging
import numpy as np
import pretty_midi  # type: ignore
import pygame
import tensorflow as tf
import coloredlogs

# Configure logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s - %(levelname)s - %(message)s')

def generate_midi(model, seed_sequence, sequence_length=128, num_notes=100, total_tokens=128):
    note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    tonic_map = {i: note_names[i] for i in range(12)}

    vocab = {
        'pitch': {pitch: idx for idx, pitch in enumerate(range(12, 100), 1)},  # MIDI pitches range from 12 to 99
        'velocity': {vel: idx + 88 for idx, vel in enumerate(range(128))},  # MIDI velocities range from 0 to 127, index starts after pitch
        'duration': {dur: idx + 216 for idx, dur in enumerate(range(26 + 1))},  # Duration based on max_duration
        'start_time': {start: idx + 216 + 26 for idx, start in enumerate(range(63 + 1))},  # Start time based on max_start
        'root': {root: idx + 216 + 26 + 63 for idx, root in enumerate(range(12))},  # 12 roots
        'mode': {'M': 260, 'm': 261},  # Two modes
        'style': {'pop_standard': 262, 'pop_complex': 263, 'dark': 264, 'r&b': 265, 'unknown': 266},  # Five styles
        'tonic': {note: idx + 267 for idx, note in enumerate(note_names)}  # 12 tonics
    }

    reverse_vocab = build_reverse_vocab(vocab)

    logger.info("Starting MIDI generation...")
    generated_sequence = seed_sequence.copy()
    
    for i in range(num_notes):
        input_sequence = [note['pitch'] if isinstance(note, dict) else note for note in generated_sequence[-sequence_length:]]
        input_sequence = np.array(input_sequence)
        input_sequence = np.expand_dims(input_sequence, axis=0)
        
        # Debugging: Print shapes
        logger.debug(f"Input sequence shape: {input_sequence.shape}")
        
        predicted_probs = model.predict(input_sequence)
        
        # Debugging: Print shapes and predicted probabilities
        logger.debug(f"Predicted probabilities shape: {predicted_probs.shape}")

        # Get the index of the token with the highest probability for the last position in the sequence
        predicted_token_index = np.argmax(predicted_probs[0, -1])
        
        new_note = decode_token(predicted_token_index, reverse_vocab, log=True)
        
        # Re-encode the new_note to match the input format required by the model
        new_note_token = vocab['pitch'][new_note['pitch']]  # This is a simplified example, adjust based on your encoding logic
        
        generated_sequence.append({'pitch': new_note_token})  # Append the re-encoded token
        
        logger.debug(f"Generated note {i+1}/{num_notes}: {new_note}")
    
    logger.info("MIDI generation complete.")
    return generated_sequence


def sequence_to_midi(sequence, output_file='generated_midi.mid', tempo=120):
    logger.info(f"Converting sequence to MIDI file: {output_file} with tempo: {tempo} BPM")
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)
    
    # Constants for quantization
    eighth_note_duration = 0.5 * 60 / tempo  # Duration of an eighth note in seconds

    for note_info in sequence:
        pitch = int(round(note_info['pitch']))
        start = note_info['start'] * eighth_note_duration
        duration = note_info['duration'] * eighth_note_duration
        velocity = note_info['velocity']
        
        # Ensure the pitch is within the MIDI range
        if pitch < 0 or pitch > 127:
            logger.warning(f"Pitch {pitch} out of range. Clamping to valid MIDI range.")
            pitch = max(0, min(127, pitch))
        
        note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=start + duration)
        instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    midi.write(output_file)
    logger.info(f"MIDI file {output_file} created successfully.")

def decode_token(token_index, reverse_vocab, log=True):
    if log:
        logging.info(f"Decoding token index: {token_index}")
    
    start = reverse_vocab['start_time'].get(token_index, "Unknown")
    duration = reverse_vocab['duration'].get(token_index, "Unknown")
    pitch = reverse_vocab['pitch'].get(token_index, "Unknown")
    velocity = reverse_vocab['velocity'].get(token_index, "Unknown")
    root = reverse_vocab['root'].get(token_index, "Unknown")
    mode = reverse_vocab['mode'].get(token_index, "Unknown")
    style = reverse_vocab['style'].get(token_index, "Unknown")
    tonic = reverse_vocab['tonic'].get(token_index, "Unknown")
    
    decoded_note = {
        'start': start,
        'duration': duration,
        'pitch': pitch,
        'velocity': velocity,
        'root': root,
        'mode': mode,
        'style': style,
        'tonic': tonic
    }

    if log:
        logging.info(f"Decoded note: {decoded_note}")
        logging.info(f"Reverse vocab: {reverse_vocab}")
        logging.info(f"Reverse vocab start_time: {reverse_vocab['start_time']}")
        logging.info(f"Reverse vocab duration: {reverse_vocab['duration']}")
        logging.info(f"Reverse vocab pitch: {reverse_vocab['pitch']}")
        logging.info(f"Reverse vocab velocity: {reverse_vocab['velocity']}")
        logging.info(f"Reverse vocab root: {reverse_vocab['root']}")
        logging.info(f"Reverse vocab mode: {reverse_vocab['mode']}")
        logging.info(f"Reverse vocab style: {reverse_vocab['style']}")
        logging.info(f"Reverse vocab tonic: {reverse_vocab['tonic']}")
        logging.info(f"Decoded note: {decoded_note}")

    return decoded_note

def build_reverse_vocab(vocab):
    reverse_vocab = {}
    for key, sub_vocab in vocab.items():
        reverse_vocab[key] = {v: k for k, v in sub_vocab.items()}
        logging.info(f"Reverse mapping for {key}: {reverse_vocab[key]}")
    return reverse_vocab

def note_to_index(note, vocab_size):
    # A simple encoding function, adjust according to your actual encoding logic
    return note['pitch'] % vocab_size

def index_to_note(index, start, duration):
    return {
        'pitch': index,  # Adjust according to your decoding logic
        'start': start,
        'duration': duration
    }

def play_midi(midi_file):
    logger.info(f"Attempting to play MIDI file: {midi_file}")
    freq = 44100    # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2    # 1 is mono, 2 is stereo
    buffer = 1024   # number of samples (experiment to get right sound)
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(1.0)
    clock = pygame.time.Clock()

    try:
        pygame.mixer.music.load(midi_file)
        logger.info(f"Music file {midi_file} loaded successfully.")
    except pygame.error as e:
        logger.error(f"File {midi_file} not found! ({e})")
        return

    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)
    logger.info("Playback finished.")
