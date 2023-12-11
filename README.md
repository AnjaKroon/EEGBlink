# Removing blink artifacts from eye
Part a: filter to remove blink eye artifacts <br />
Part b: filter eval<br />

# General Idea
y = x + v<br />
y: signal + noise<br />
x: blink component (noise)<br />
v: clean brain signal (signal)<br />
Step 1: estimate x (x_hat)<br />
Step 2: y - x_hat = v_clean<br />

# Input Data
2 EEG recordings, 19 nodes, 400 Hz F_sam<br />

# Output
Clean versions of 2 EEG recordings<br />
