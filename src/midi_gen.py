from mido import MidiFile, MidiTrack, Message
import logging
import numpy as np

def generate_midi(predictions, output_file='generated_midi.mid'):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Ensure predictions are in shape (n_steps, 4)
    predictions = predictions.reshape(-1, 4)

    # Iterate over each set of predictions
    for prediction in predictions:
        pitch = int(prediction[2] * 127)  # Assuming the third feature is pitch
        velocity = int(prediction[3] * 127)  # Assuming the fourth feature is velocity
        duration = int(prediction[1] * 500)  # Assuming the second feature is duration

        # Append a note on and note off message
        track.append(Message('note_on', note=pitch, velocity=velocity, time=0))
        track.append(Message('note_off', note=pitch, velocity=velocity, time=duration))

    # Save the MIDI file
    mid.save(output_file)
    print(f"MIDI file saved as {output_file}")
