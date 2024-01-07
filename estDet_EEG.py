'''
Required: Training_EEG.mat
'''

import scipy.io
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import math
import timeit
import operator

'''
@load_data
Used to load data for the training set.
'''
def load_data(path):
  mat = scipy.io.loadmat(path)
  blinks= mat["blinks"]
  data = mat["train_eeg"]
  return data, blinks

'''
@plot_blinks
Plots the blink instances given per sample in the training set.
'''
def plot_blinks(blinks):
  plt.plot(blinks)
  plt.xlabel('Blink instances')
  plt.show()

'''
@plot_EEG
Plots the EEG using MNE library for a given dataset, given number of channels, and a given title.
'''
def plot_EEG(d, arr, title):
  channels, l = d.shape
  print("total num data points ", l)
  sfreq = 400
  dur = l/sfreq
  namelist= ['EEG 1', 'EEG 2', 'EEG 3', 'EEG 4', 'EEG 5','EEG 6','EEG 7', 'EEG 8', 'EEG 9', 'EEG 10','EEG 11',
            'EEG 12', 'EEG 13', 'EEG 14', 'EEG 15', 'EEG 16','EEG 17','EEG 18', 'EEG 19']
  ch_names = [namelist[i] for i in arr]
  info = mne.create_info(ch_names=ch_names, ch_types=['eeg'] * len(arr), sfreq=sfreq)
  raw = mne.io.RawArray(d, info)
  fig = mne.viz.plot_raw(raw, duration=dur, start=0.0, scalings=100, n_channels=channels, title=title, show=False, show_scrollbars=False)
  fig.savefig(title)
  #return fig

'''
@divide_into_frames
Reshaping the incoming data in the form of (channels, datapoints) into (frames, channels, datapoints).
Channels and datapoints are the (row x column) dim convention for a matrix.
Frames refer to the "number of matricies". 
'''
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
  print("Reshaped for Processing Matrix Shape:", result_matrix.shape)    #  sheet, rows, columns,
  return result_matrix

'''
@find_blink_frames
If frame as 'significant' blink instances, then flag frame as a blink frame.
Significance is based on frame length (fl) and sensitivity parameter.
'''
def find_blink_frames(l, fl):

  start_idx = 0
  blink_frames = []
  n_frames = math.floor((len(l) / fl))
  sensitivity = 0.20                            # Number of points in a frame that must be of blink = 1 to be a blink frame
  for i in range(n_frames):
    start_idx = i * fl
    end_idx = start_idx + fl
    if np.sum(l[start_idx:end_idx]) >= (fl*sensitivity):          #make this more sensitive by increasing tolerance, more tolerance for longer frames needed
      blink_frames.append(1)
    else:
      blink_frames.append(0)
  return blink_frames

'''
@remove_blinks
Equation 11 in paper:
Hn = Un [I - RvvRyy^-1]
Equation 12 in paper:
vn(k) = yn(k) - Hn * yn(k) --> produced clean EEG signals across all channels
'''
def remove_blinks(Rvv, Ryy, d, y, fl, bf):
  channels, l = d.shape
  n_frames= math.floor((l/fl))
  vn = np.zeros((channels, l))
  for n in range(channels):
    print(n)
    un = Umat( n+1, channels, fl)               #(50, 950)
    for i in range(n_frames):
      if bf[i]==1:
        b = np.identity(channels * fl) - np.matmul( Rvv[i,:,:] , np.linalg.pinv( Ryy[i,:,:] ))    # Eq. 11 from paper
        Hn = np.matmul( un, b)   
        vn[n, (fl*i):(fl*(i+1))] = (d[ n, (fl*i):(fl*(i+1))] - np.matmul( Hn, y[i,:]))         # Eq. 12 from paper
      else:
        vn[n, (fl*i):(fl*(i+1))] = d[ n, (fl*i):(fl*(i+1))] 

  return vn

'''
@gen_y_seq
Reformatting the data (frames, channels, frame_data)
into (frame, (data_ch_1 concat. with data_ch_2 concat. with ... data_ch_X))
'''
def gen_y_seq(result_matrix):
  n_frames, channels, frame_len = result_matrix.shape
  y = np.zeros((n_frames, channels * frame_len))
  for i in range(n_frames):
    dump = []
    for j in range(channels):
      dump = np.append(dump, result_matrix[i,j,:])
    y[i,:]=dump
  return y

'''
@gen_corr
When eye blink occurs, the Rvv must be pulled from the data frame which preceeds it. 
Y is the data matrix in (frame, (data_ch_1 concat. data_ch_2 ... concat. data_ch_X))
Check is where the blink frames occur

In frames without eyeblinks, Rvv = Ryy = y * y^T
In frames with eyeblinks, Rvv = previous Ryy without a blink
Autocorrelation matrix calculated by (frame, (data_ch_1 concat. with ... data_ch_X)* (data_ch_1 concat. with ... data_ch_X)^T)
'''
def gen_corr(y, check):
  n_frames, l = y.shape  
  R = np.zeros((n_frames, l, l))
  for i in range(n_frames):
    if check[i] == 0:
      seq = np.outer(y[i,:], y[i,:].transpose())
      R[i, :, :] = seq
      prev = seq
    else:
      R[i, :, :] = prev
  return R

'''
@Umat
Generates the U matrix which is a concatenation of three matricies -- 0 I 0
'''
def Umat(n,N,L):
  #Un= O(Lx(n-1)L)  I(LxL)  O(Lx(N-n)L)
  o1=np.zeros((L,(n-1)*L))
  o2=np.zeros((L,(N-n)*L))
  i=np.identity(L)
  Un=np.concatenate((o1,i,o2), axis=1)
  return Un

'''
@test_frame_division
Testing to ensure the data division and reshaping into frames performed correctly.
'''
def test_frame_division(data, result, frame_len):
    check_1 = operator.eq(data[0,0],result[0,0,0])
    check_2 = operator.eq(data[0,frame_len],result[1,0,0])
    status = (check_1 and check_2)
    print("Frame division correct?", status)

'''
@test_find_blink_frames
Checks to make sure all blink frames are indicated.
Only fully satisfied when the significance is set to frame length in find_blink_frames.
Is good to check where the blinks are and are not reported.
Also helps with fine tuning frame length parameters.
'''
def test_find_blink_frames(test_b, bf, fl):
    # Note: blink frames may be missing intensionally
    # If the frame has very few blink data points in it, it makes more sense to keep it whole
    # See 'sensitivity' variable in find_blink_frames    
    # test_find_blink_frames(testing_blinks, bf, fl)   
    indicator = 0
    for i in test_b:
      should_be_blink_in_this_frame = math.floor(i/fl)
      if bf[should_be_blink_in_this_frame] != 1:
        print("Blink frame indicator is missing for frame", should_be_blink_in_this_frame)
        indicator = 1
    if indicator == 0:
      print("All blink frames found.")
    else:
      print("Error: check find_blink_frames")

'''
@flag_top_percent
To manually identify where the blinks occur in the test set based on amplitude
Identifies the top 5 percent for positive values
Identifies the bottom 5 percent for negative values
'''
def flag_top_percent(data, percent, padding):
      pos = []
      neg = []
      chs, l = data.shape
      flagged_array = np.zeros(l)
      for c in range(chs):
        for x in data[c,:]:
          if x < 0:
              neg.append(x)
          else:
              pos.append(x)
        thresh_pos = np.percentile(pos, 100 - percent)
        thresh_neg = np.percentile(neg, percent)
        f_pos = np.where(data[c] > thresh_pos, 1, 0)
        f_neg = np.where(data[c] < thresh_neg, 1, 0)
        flagged_array = flagged_array + f_pos + f_neg           # almost better to set the percent lower because it will sum anyway so it just has to catch the blink on "one"
      fa = np.zeros(l + padding)
      thr = chs/2       # if more than half the channels "VOTE" that a blink is occuring there, then blink = 1
      for i in range(l):
          if flagged_array[i] > thr:        # if more than half of the channels flagged it, then maintain blink
              fa[i - padding : i + padding] = np.ones( 2 * padding )
      return fa[:l]

if __name__ == "__main__":
    '''
    Experiment 1: Training data
    '''
    # Start timer
    start = timeit.default_timer()

    # Load training data
    data, blinks = load_data('Training_EEG.mat')
    

    testing_blinks = (blinks[0][0:])

    # Initialize parameters
    fl = 50                                # Choose frame length --- tested 200, 100, and 50 (best results with 50)
    chs = [ 0, 1, 5, 8]                    # Choose channels 
    # chs = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # chs = [3, 4, 5]         # works best with 1%
    # chs = [1, 11, 18]
    # chs = [1]
    data = data[ chs, :]                    

    # Dividing input data in a given frame length
    result_matrix = divide_into_frames(data, fl)

    # Checks if frame division occured without errors, status in terminal
    test_frame_division(data, result_matrix, fl)     

    # Defining parameters
    n_frames, channels, frame_len = result_matrix.shape
    l = n_frames*frame_len

    # Returns 1 if frame has blinks in it, else 0
    blink = np.zeros(l)
    for i in range(l):
      if i in blinks:
        blink[i] = 1

    print("blink matrix for training", blink.shape)
    bf = find_blink_frames(blink, fl) 
    #plot_blinks(bf)        

    # Reformatting the data (frames, channels, frame_data)
    # into (frame, (data_ch_1 concat. with data_ch_2 concat. with ... data_ch_X))
    y = gen_y_seq(result_matrix)                    #(192, 950)---->(frames,data)

    # In frames without eyeblinks, Rvv = Ryy
    # Autocorrelation matrix calculated by (frame, (data_ch_1 concat. with ... data_ch_X)* (data_ch_1 concat. with ... data_ch_X)^T)
    # Must now calculate Rvv when eye blink occurs, use Ryy of last non blink frame before it
    Rvv = gen_corr(y, bf)

    # Use same function setup but rather than selectively calculating Rvv, just calculate autocorr for every input
    # Passing in a zero matrix to ensure every input is used in the calculation
    # Calculating autocorr for signal + noise data 
    Ryy = gen_corr(y, np.zeros(n_frames))           #(192, 950, 950)

    # Creates transfer function and produces the clean eye blink signal
    clean = remove_blinks(Rvv, Ryy, data, y, fl, np.ones(len(bf)))

    print("Runtime for training set: ", timeit.default_timer()- start)

    plot_blinks(blink)
    plot_EEG(data, chs, "Original")
    plot_EEG(clean, chs, "Cleaned")

    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    axs[0].imshow(plt.imread("Original.png"))
    axs[0].title.set_text('Training EEG')
    axs[0].axis("off")
    axs[1].imshow(plt.imread("Cleaned.png"))
    axs[1].title.set_text('Filtered EEG')
    axs[1].axis("off")
    plt.show()



    '''
    Experiment 2: Testing data
    To detect the blink frames a 'detector' was built that identifies the blinks based on the amplitude.
    The peaked region of each negative and positve part of the blink is identified and the region is stretched to cover the region 
    distorted by the blink
    '''
    start = timeit.default_timer()
    mat = scipy.io.loadmat('Test_EEG.mat')
    test = mat["test_eeg"]
    # Limit to first 9600 samples for visualization purposes
    test = test[chs,:9600]

    # percent_to_flag selects the top fifth percentile of the data to identify the spikes
    percent_to_flag = 5
    # padding is the length of the blink region
    padding = 100
    flagged_array = flag_top_percent(test, percent_to_flag, padding)
    print("flagged array for testing", flagged_array.shape)
    test_bf= find_blink_frames(flagged_array, fl)
    #plot_blinks(test_bf)

    div_test=divide_into_frames(test,fl)
    y_test= gen_y_seq(div_test)
    test_frames,channels,frame_len=div_test.shape
    Ryy_test=gen_corr(y_test, np.zeros(test_frames))

    # the test blink frames are used here to select the region where to replace the test data
    clean=remove_blinks(Rvv,Ryy_test,test,y_test,fl,test_bf)

    print("Runtime: ", timeit.default_timer()- start)

    #plot_blinks(test_bf)
    plot_EEG(test, chs, "Original")
    plot_EEG(clean, chs, "Cleaned")

    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    axs[0].imshow(plt.imread("Original.png"))
    axs[0].title.set_text('Test EEG')
    axs[0].axis("off")
    axs[1].imshow(plt.imread("Cleaned.png"))
    axs[1].title.set_text('Filtered EEG')
    axs[1].axis("off")
    plt.show()