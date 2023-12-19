'''
Required: Training_EEG.mat
'''
import scipy.io
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import math

def load_data(path):
  mat = scipy.io.loadmat(path)
  blinks= mat["blinks"]
  data = mat["train_eeg"]
  return data, blinks

def plot_blinks(blinks):
  plt.plot(blinks)
  plt.xlabel('Blink instances')
  plt.show()

def plot_EEG(data):
  sfreq = 400
  info = mne.create_info(ch_names=['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5',
                                    'EEG 6','EEG 7', 'EEG 8', 'EEG 9', 'EEG 10','EEG 11',
                                      'EEG 12', 'EEG 13', 'EEG 14', 'EEG 15', 'EEG 16','EEG 17',
                                        'EEG 18', 'EEG 19'], ch_types=['eeg'] * 19, sfreq=sfreq)
  raw = mne.io.RawArray(data, info)
  mne.viz.plot_raw(raw, duration=25.0, start=0.0, scalings=100, n_channels=19)
  plt.show()

def autocorrelation(signal):
    auto_corr = np.correlate(signal, signal, mode='full')

    # from scipy.linalg import toeplitz
    # x = data[1,:] #x is complex
    # acf = np.convolve(x,np.conj(x)[::-1]) # using Method 2 to compute Auto-correlation sequence
    # Rxx=acf[2:]; # R_xx(0) is the center element
    # Rx = toeplitz(Rxx,np.hstack((Rxx[0], np.conj(Rxx[1:]))))

    return auto_corr[len(auto_corr)//2:]

def plot_autocorr(ac):
    plt.plot(ac)
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()

def divide_into_frames(data, fl):
  channels, data_length = data.shape
  frame_length = fl        # data_length   
  amt_overlap_frames = 0
  shift_param = frame_length - amt_overlap_frames             # how many points before start of next frame
  last_frame_start = data_length - frame_length               # starting point of last frame
  amount_frames = math.floor(((last_frame_start) / (shift_param))) + 1

  result_matrix = np.zeros((amount_frames, channels, frame_length))

  for i in range(channels):   # For every channel
    for j in range(amount_frames):    # for every frame
        if j == 0:
          start_idx = 0
        else:
          start_idx = j * shift_param
        end_idx = start_idx + frame_length
        result_matrix[j, i, :] = data[i, start_idx:end_idx]
        

  print("Data Matrix Shape:", data.shape)
  print("Result Matrix Shape:", result_matrix.shape)    # rows, columns, channels
  return result_matrix

def find_blink_frames(blinks, fl):
  l=np.zeros(9600)
  for i in range(9600):
    if i in blinks:
      l[i]=1
  start_idx=0
  blink_frames=[]
  n_frames= math.floor((len(l) / fl))
  for i in range(n_frames):
    start_idx = i*fl
    end_idx = start_idx + fl
    if np.sum(l[start_idx:end_idx])>= (fl/2):
      blink_frames.append(1)
    else:
      blink_frames.append(0)
  return blink_frames

def remove_blinks(data, blinks):
  # calculating H for each frame within each channel and applying it to produce a matrix of the same dimensions
  # blinks will not be in the correct form factor -- make zeros matrix and frame-ify it, any extras fill with zero
  
  clean = data
  return clean
  
def gen_y_seq(result_matrix):
  n_frames,channels,frame_len=result_matrix.shape
  y=np.zeros((n_frames,channels*frame_len))
  for i in range(n_frames):
    dump=[]
    for j in range(channels):
      dump=np.append(dump,result_matrix[i,j,:])
    y[i,:]=dump
  return y

def gen_corr(y,check):
  n_frames,l=y.shape  
  Ryy=np.zeros((n_frames,l,l))
  for i in range(n_frames):
    seq=np.outer(y[i,:], y[i,:].transpose())
    if check[i]==0:
      Ryy[i,:,:]=seq
      prev=seq
    else:
      Ryy[i,:,:]=prev
  return Ryy

def Umat(n,N,L):
  #Un= O(Lx(n-1)L)  I(LxL)  O(Lx(N-n)L)
  o1=np.zeros((L,(n-1)*L))
  o2=np.zeros((L,(N-n)*L))
  i=np.identity(L)
  Un=np.concatenate((o1,i,o2), axis=1)
  return Un

if __name__ == "__main__":
  data, blinks = load_data('EEG_data\Training_EEG.mat')
  fl=50

  result_matrix= divide_into_frames(data, fl)
  n_frames,channels,frame_len=result_matrix.shape
  bf=find_blink_frames(blinks,fl)
  y=gen_y_seq(result_matrix)            #(192, 950)
  Ryy=gen_corr(y,np.zeros(n_frames))    #(192, 950, 950)
  Rvv=gen_corr(y,bf)






  # for the first frame:
  u1=Umat(1,channels, fl)               #(50, 950)
  #H1=U1(I-Rvv.Ryy^-1)

  b=np.identity(channels*fl)- np.matmul(Rvv[0,:,:],np.linalg.inv(Ryy[0,:,:]))
  H1=np.matmul(u1, b)
  # print('H1: ',H1.shape)
  # print('y: ',y[0,:].shape)
  # print('yn: ',data[0,:50].shape)
  vn=  data[0,:50] -np.matmul(H1, y[0,:])


  fig, axs = plt.subplots(2)
  axs[0].plot(vn)
  axs[0].set_title("clean?")
  axs[1].plot(data[0,:50])
  axs[1].set_title("original")
  plt.show()


  # for i in range(len(bf)): 
  #   if bf[i]==1:
  #     c=i
  #     break
  # print(c)