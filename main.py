'''
Required: Training_EEG.mat
'''
import scipy.io
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import math

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

def short_stationary_frame_length(data):
  def autocorrelation(signal):
    auto_corr = np.correlate(signal, signal, mode='full')
    return auto_corr[len(auto_corr)//2:]

  def plot_autocorr(ac):
    plt.plot(ac)
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()

  channels, data_length = data.shape
  frame_length = 300           
  amt_overlap_frames = 200
  shift_param = frame_length - amt_overlap_frames             # how many points before start of next frame
  last_frame_start = data_length - frame_length               # starting point of last frame
  amount_frames = math.floor(((last_frame_start) / (shift_param))) + 1

  # for every channel in data, reshape into frames, frames are rows, time is col, stack is diff. channels
  result_matrix = np.zeros((amount_frames, frame_length, channels))

  # split the data into frames of a certain length and channels go along the z axis now
  for i in range(channels):   # For every channel
    for j in range(amount_frames):    # for every frame
        if j == 0:
          start_idx = 0
        else:
          start_idx = j * shift_param
        end_idx = start_idx + frame_length
        result_matrix[j, :, i] = data[i, start_idx:end_idx]

  # print("Data Matrix Shape:", data.shape)
  # print("Result Matrix Shape:", result_matrix.shape)    # rows, columns, channels
  
  # compute the autocorrelation over the frame length of 300
  # testing computation of the autocorrelation matrix
  # ac = autocorrelation(result_matrix[0, :, 0])
  # plot_autocorr(ac)
  # ac_ = autocorrelation(result_matrix[4, :, 0])
  # plot_autocorr(ac_)
  # TODO: The autocorrelation is not showing SMALL SENSE STATIONARY properties
  # could probably just assume small sense stationary but I don't want to do that
  # Perhaps it is because the blink is throwing it off? Or is there additional noise?
  # for extra, rather than checking all 32, propose to randomly select 4 (4/32 = 1/8 of the samples)
        
  return result_matrix

def main():
  data, blinks = load_data()
  short_stationary_frame_length(data)
  plot_blinks(blinks)
  plot_EEG(data)

  # calculating H for each frame within each channel and applying it to produce a matrix of the same dimensions
  
  clean = remove_blinks(data, blinks)
  plot_EEG(clean)

if __name__ == "__main__":
  main()