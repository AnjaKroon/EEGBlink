# Removing Eye Blink Noise Artifacts from Electroencephalograms (EEGs)
Wiener filter to estimate the eye blink components of the signal. These are then subtracted from the input signal to produce a clean version of the origional EEG signal.<br />
Eye blink detection is performed with two simple binary hypothesis tests.<br />

# Input Data
2 EEG recordings (training and test), 19 channels, 400 Hz F_sam<br />
Training set has EEG data and an eye blink indicator function for every data point. 
Test set has only EEG data. Eye blink indicator array may be developed via the detection algorithm.

# Output
Clean versions of 2 EEG recordings including plots. <br />

# Note
The code produced is based on the multi-channel wiener filtering first presented in the following published paper:
[https://www.sciencedirect.com/science/article/pii/S1746809418301149]
