import wave, struct
from math import *
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from scipy import signal
import sys
import pylab
import os

import datetime
import string

from bandpass_filter import *

def graph_spectrogram(wav_file, begin, end, cutoff_low, cutoff_high, order):
    sound_info, Fs = get_wav_info(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)
    nfft = 64
    data = sound_info[int(begin*Fs): int(end*Fs)]
    data = butter_bandpass_filter(data, cutoff_low, cutoff_high, Fs, order)
    pylab.specgram(data, Fs=Fs, NFFT = nfft, \
                   noverlap =int(nfft/2), \
                   scale = 'dB', \
                   cmap='viridis_r', \
                   mode = 'magnitude')

    pylab.colorbar()
    pylab.ylim((0,50000)) # limits on the frequency shown
    datestring = begin 
    pylab.xlabel("T ({} seconds sinds start sample) [s]".format(datestring), \
                 size = 16, ha ='right', x=1.0)
    pylab.ylabel("PSD", size = 16, ha ='right', position=(0,1))
#    pylab.savefig('spectrogram.png')
    pylab.show()
    
def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = np.frombuffer(frames,dtype='int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


import string
def main(argv):
    begin = 0
    end = 100
    if len(argv) > 1:
        filename  = argv[1]
    if len(argv) > 2:
        begin = string.atof(argv[2]) # in seconds
        end = string.atof(argv[3]) # seconds
        assert end > begin , "end time should be after begin!"

    filename = argv[1]
    cutoff_low = 500
    cutoff_high = 7000
    order = 5 
    graph_spectrogram(filename, begin, end, cutoff_low, cutoff_high, order)

if __name__ == '__main__':
    main(sys.argv)
