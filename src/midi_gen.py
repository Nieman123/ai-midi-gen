from mido import MidiFile, MidiTrack, Message
import logging
import numpy as np

def generate_midi(predictions, output_file='generated_midi.mid'):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    predictions = predictions.reshape(-1, 3)  # Ensure predictions are in shape (n_steps, 3)

    for prediction in predictions[0]:  # Accessing the first element of the batch
            pitch = int(prediction[0] * 127)
            velocity = int(prediction[1] * 127)
            duration = int(prediction[2] * 500)

            track.append(Message('note_on', note=pitch, velocity=velocity, time=0))
            track.append(Message('note_off', note=pitch, velocity=velocity, time=duration))


    mid.save(output_file)
    print(f"MIDI file saved as {output_file}")