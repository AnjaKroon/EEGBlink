import scipy.io
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

def load_data():
  mat = scipy.io.loadmat('Training_EEG.mat')
  blinks= mat["blinks"]
  data = mat["train_eeg"]
  return data, blinks

def plot_blinks(blinks):
  l=np.zeros(9600)
  for i in range(9600):
    if i in blinks:
      l[i]=1
  plt.plot(l)
  plt.xlabel('Blink instances')
  plt.show()

def remove_blinks(data, when_blink):
  clean = data
  return clean

def plot_EEG(data):
  sfreq = 400
  info = mne.create_info(ch_names=['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5',
                                    'EEG 6','EEG 7', 'EEG 8', 'EEG 9', 'EEG 10','EEG 11',
                                      'EEG 12', 'EEG 13', 'EEG 14', 'EEG 15', 'EEG 16','EEG 17',
                                        'EEG 18', 'EEG 19'], ch_types=['eeg'] * 19, sfreq=sfreq)
  raw = mne.io.RawArray(data, info)
  mne.viz.plot_raw(raw, duration=25.0, start=0.0, scalings=100, n_channels=19)
  plt.show()

def main():
  data, blinks = load_data()
  plot_blinks(blinks)
  plot_EEG(data)
  clean = remove_blinks(data, blinks)
  plot_EEG(clean)

if __name__ == "__main__":
    main()