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

def generate_midi(model, seed_sequence, sequence_length=128, num_notes=100, vocab_size=128):
    logger.info("Starting MIDI generation...")
    generated_sequence = seed_sequence.copy()
    
    for i in range(num_notes):
        input_sequence = [note_to_index(note, vocab_size) for note in generated_sequence[-sequence_length:]]
        input_sequence = np.array(input_sequence)
        input_sequence = np.expand_dims(input_sequence, axis=0)
        
        predicted_probs = model.predict(input_sequence)
        logger.info(f"Perdicted Probs: {predicted_probs}")
        predicted_note_index = np.argmax(predicted_probs, axis=-1)[0, -1]
        
        new_note = index_to_note(predicted_note_index, start=len(generated_sequence), duration=1)
        logger.info(f"New Note Data: {new_note}")
        generated_sequence.append(new_note)
        
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
        
        # Ensure the pitch is within the MIDI range
        if pitch < 0 or pitch > 127:
            logger.warning(f"Pitch {pitch} out of range. Clamping to valid MIDI range.")
            pitch = max(0, min(127, pitch))
        
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + duration)
        instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    midi.write(output_file)
    logger.info(f"MIDI file {output_file} created successfully.")

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

# Example usage
def main():
    logger.info("Starting main process...")
    
    # Load the trained model
    model = tf.keras.models.load_model("transformer_midi_gen.keras")
    logger.info("Model loaded successfully.")
    
    # Retrieve the embedding layer
    embedding_layer = model.layers[1]

    # Confirm the embedding layer
    if isinstance(embedding_layer, tf.keras.layers.Embedding):
        vocab_size = embedding_layer.input_dim
        logger.info(f"Vocabulary Size: {vocab_size}")
    else:
        logger.error("The retrieved layer is not an Embedding layer")
        return

    # Example seed sequence with all note attributes
    seed_sequence = [{
        'pitch': np.random.randint(60, 72),  # Random pitch between C4 and B4
        'start': i,
        'duration': 1
    } for i in range(sequence_length)]

    generated_sequence = generate_midi(model, seed_sequence, sequence_length=sequence_length, num_notes=100, vocab_size=vocab_size)
    sequence_to_midi(generated_sequence, output_file='generated_midi.mid', tempo=120)
    play_midi('generated_midi.mid')

    logger.info("Main process finished.")

if __name__ == "__main__":
    main()
