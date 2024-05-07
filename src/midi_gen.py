from mido import MidiFile, MidiTrack, Message
import logging
import numpy as np
import pygame
import time

def generate_midi(predictions, output_file='generated_midi.mid', ticks_per_beat=480, max_duration=13):
    # Assume each eighth note is half of a quarter note
    eighth_note_ticks = ticks_per_beat // 2
    total_ticks_for_max_duration = eighth_note_ticks * max_duration  # Total ticks that max_duration corresponds to

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    # Ensure predictions are in shape (n_steps, 4)
    predictions = predictions.reshape(-1, 4)

    # Sort predictions by the start time (first column)
    predictions = predictions[predictions[:, 0].argsort()]

    last_start_time = 0  # Initialize the last start time

    # Iterate over each set of predictions
    for prediction in predictions:
        start_time = int(prediction[0] * eighth_note_ticks)  # Assuming the first feature is start time normalized
        pitch = int(prediction[2] * 127)  # Assuming the third feature is pitch
        velocity = int(prediction[3] * 127)  # Assuming the fourth feature is velocity
        normalized_duration = prediction[1]
        duration_ticks = int(normalized_duration * total_ticks_for_max_duration)  # Reverse the normalization

        # Calculate delta time as the difference from the last note's start time
        delta_time = start_time - last_start_time
        if delta_time < 0:
            logging.error(f"Negative delta time detected: {delta_time}. Adjusting to 0.")
            delta_time = 0

        logging.info(f"Start Time: {start_time}, Pitch: {pitch}, Velocity: {velocity}, Normalized Duration: {normalized_duration}, Duration Ticks: {duration_ticks}, Delta Time: {delta_time}")

        # Append a note on and note off message
        track.append(Message('note_on', note=pitch, velocity=velocity, time=delta_time))
        track.append(Message('note_off', note=pitch, velocity=0, time=duration_ticks))

        # Update last start time to this note's start time for the next iteration
        last_start_time = start_time

    # Save the MIDI file
    mid.save(output_file)
    print(f"MIDI file saved as {output_file}")
    play_midi(output_file)

def play_midi(midi_file):
    freq = 44100    # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2    # 1 is mono, 2 is stereo
    buffer = 1024   # number of samples (experiment to get right sound)
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(1.0)
    clock = pygame.time.Clock()

    try:
        pygame.mixer.music.load(midi_file)
        print("Music file {} loaded!".format(midi_file))
    except pygame.error:
        print("File {} not found! ({})".format(midi_file, pygame.get_error()))
        return

    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)
