from mido import MidiFile, MidiTrack, Message
import logging
import numpy as np

def generate_midi(predictions, output_file='generated_midi.mid', ticks_per_beat=480):
    # Assume each eighth note is half of a quarter note
    eighth_note_ticks = ticks_per_beat // 2

    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    # Ensure predictions are in shape (n_steps, 4)
    predictions = predictions.reshape(-1, 4)

    # Iterate over each set of predictions
    for prediction in predictions:
        pitch = int(prediction[2] * 127)  # Assuming the third feature is pitch
        velocity = int(prediction[3] * 127)  # Assuming the fourth feature is velocity
        normalized_duration = prediction[1]
        duration_ticks = int(prediction[1] * eighth_note_ticks)  # Assuming the second feature is duration

        logging.info(f"Pitch: {pitch}, Velocity: {velocity}, Normalized Duration: {normalized_duration}, Duration Ticks: {duration_ticks}")

        # Append a note on and note off message
        # Note on with delta time 0 (right after the previous event)
        track.append(Message('note_on', note=pitch, velocity=velocity, time=0))
        # Note off with delta time calculated based on the duration
        track.append(Message('note_off', note=pitch, velocity=0, time=duration_ticks))

    # Save the MIDI file
    mid.save(output_file)
    print(f"MIDI file saved as {output_file}")
