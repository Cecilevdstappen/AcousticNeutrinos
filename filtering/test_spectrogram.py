import wave, struct
import numpy as np
from scipy.signal import freqz
import matplotlib.pyplot as plt
import sys
from optparse import OptionParser
import os


file_name = 'neutrino_12.0_1000_1_11_100000.wav'
f = wave.open(file_name,'r')
frames = f.readframes(-1)
u = np.frombuffer(frames, dtype="int16")
fs = f.getframerate()
print(fs)
# If Stereo
if f.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)

time = np.linspace(0,len(u)/fs, num = len(u))

plt.figure(0)
plt.plot(time,fs)
plt.xlabel("Time")
plt.title("Spectrogram")
plt.legend(['u(t)', 'y(t)'])
plt.grid()
plt.show()


