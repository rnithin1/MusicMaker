import numpy as np
from mido import MidiFile
import librosa as lr
from math import ceil, floor

def to_piano_roll(midi):
    """Convert MIDI file to a 2D NumPy ndarray (notes, timesteps)."""
    notes = 127
    tempo = 500000  # Assume same default tempo and 4/4 for all MIDI files.
    seconds_per_beat = tempo / 1000000.0
    seconds_per_tick = seconds_per_beat / midi.ticks_per_beat
    velocities = np.zeros(notes)
    sequence = []
    for m in midi:
        ticks = int(np.round(m.time / seconds_per_tick))
        ls = [velocities.copy()] * ticks
        sequence.extend(ls)
        if m.type == 'note_on':
            velocities[m.note] = m.velocity
        elif m.type == 'note_off':
            velocities[m.note] = 0
        else:
            continue
    piano_roll = np.array(sequence).T
    return piano_roll.reshape(piano_roll.shape[1], piano_roll.shape[0])

def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]
