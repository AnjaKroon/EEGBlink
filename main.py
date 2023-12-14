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


