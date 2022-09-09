#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:06:50 2022

@author: gebruiker
"""

from matplotlib import pyplot as mp
import numpy as np
from pycbc.filter import matched_filter, get_cutoff_indices #, make_frequency_series
from pycbc import types
from pycbc.types.array import complex_same_precision_as
from pycbc import psd
#from pycbc.strain.lines import complex_median
#from pycbc.events.coinc import mean_if_greater_than_zero
from pycbc import fft
import sys
sys.path.insert(0, '/home/gebruiker/SIPPY')
sys.path.insert(0, '/home/gebruiker/AcousticNeutrinos/filtering/design_FRF')
import os
os.path.join('/home/gebruiker/AcousticNeutrinos/filtering/Neutrino_data_files/neutrino_6_300_7000whitenoise.txt')
import wave
import scipy.signal as sg
from scipy.io.wavfile import read
from impulse import *
from mouse_button import *
from design_FRF import *
from SNR import resample_by_interpolation
from my_styles import *

set_paper_style()

plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')

# Make gaussian signal
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

mu = -1
sig = 1
x_values = np.linspace(-3, 3, 120)
dx = 120/6
Gauss = gaussian(x_values, mu, sig)
print(len(Gauss))

# Generate white noise

noise_file = 'output001.wav'
wav = wave.open(noise_file, 'r')
frames = wav.readframes(-1)
sound_info = np.frombuffer(frames,dtype='int16')
frame_rate = wav.getframerate()
wav.close()

time_trace= read(noise_file)
time_data_whale = np.arange(0, sound_info.size) / frame_rate
noise_whale = time_trace[1]
dt = 1/144000.
noise_td = types.TimeSeries(noise_whale, dt)
noise_fd = abs(noise_td.to_frequencyseries())

# Pycbc preparation

template = types.TimeSeries(Gauss, dx)
template.resize(len(noise_td))

seg_len = 2048
p_estimated = noise_td.psd(dt*seg_len, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
psd_inter = psd.interpolate(p_estimated, noise_td.delta_f)

snr_inter = matched_filter(template, noise_td, psd = psd_inter,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
