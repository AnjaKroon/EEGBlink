# Objective: Removing blink artifacts from eye
Part a: filter to remove blink eye artifacts
Part b: filter eval

# General Idea:
y = x + v
y: singal + noise
x: blink component (noise)
v: clean brain signal (signal)
Step 1: estimate x (x_hat)
Step 2: y - x_hat = v_clean

# Input Data:
2 EEG recordings, 19 nodes, 400 Hz F_sam

# Output:
Clean versions of 2 EEG recordings
