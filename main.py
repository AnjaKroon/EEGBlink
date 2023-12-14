# Objective: Removing blink artifacts from eye
# Part a: filter to remove blink eye artifacts
# Part b: filter eval

# General Idea:
# y = x + v
# x: blink component
# v: clean brain signal
# estimate x (x_hat), y-x_hat = v_clean

# Input Data:
# 2 EEG recordings, 19 nodes, 400 Hz F_sam

# Output:
# Clean versions of 2 EEG recordings

import scipy.io
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

x = 2
print(x)


# import data

mat = scipy.io.loadmat('drive/MyDrive/ED/Training_EEG.mat')
blinks= mat["blinks"]
data= mat["train_eeg"]

print(mat.keys())

# Plotting Raw EEG
sfreq = 400
info = mne.create_info(ch_names=['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5', 'EEG 6','EEG 7', 'EEG 8', 'EEG 9', 'EEG 10','EEG 11', 'EEG 12', 'EEG 13', 'EEG 14', 'EEG 15', 'EEG 16','EEG 17', 'EEG 18', 'EEG 19'],
                       ch_types=['eeg'] * 19, sfreq=sfreq)

raw = mne.io.RawArray(data, info)

mne.viz.plot_raw(raw, duration=25.0, start=0.0, scalings=100, n_channels=19)