#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:07:35 2022

@author: gebruiker
"""
import matplotlib.pyplot as pp
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
import numpy as np
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
ax = pp.gca()
zpos = 10
rpos = 300

# Reading in data file

data_file = '../Neutrino_data_files/neutrino' + '_' + str(zpos) + '_' + str(rpos) + '_1_11_1' + '.txt' 
data = np.loadtxt(data_file,  usecols=(0), dtype='float', unpack=True)  
noise_signal_td = data #np.resize(data, 131072)
freq = 144000.
noise_signal_times = np.arange(0, len(noise_signal_td), 1)
noise_signal_times = noise_signal_times/144000.
noise_signal_td = noise_signal_td - np.mean(noise_signal_td)

flow = 20.; fhigh= 30000.

# Reading in signal file

template_file = '../Neutrino_data_files/neutrino_'+ str(zpos) +'_300_1_11.dat'
data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
time = data[0]
amplitude_og = data[1]

# Convolution with transferfunction

frf = design_FRF(Fs=144000)
#frf.highpass = True
# frf.pz2tf((375,12000,17000),(0.15,0.02,0.07),100000) # hydrophone with cap
frf.pz2tf((375,12000),(0.1,0.03),144000) # hydrophone without cap
# frf.pz2tf((12000,375),(0.02,0.1),100000) # hydrophone without cap
frf.mbutton.on()
amplitude_og = sg.lfilter(frf.num,frf.dnum,amplitude_og)
noise_signal_td = sg.lfilter(frf.num,frf.dnum,noise_signal_td)

pp.figure(1)
pp.plot(time, amplitude_og) #label ='z =' + str(zpos))
pp.grid('on',which='both',axis='x')
pp.grid('on',which='both',axis='y')
pp.title('E = 1e11 GeV, r = 300 m, z = 6 m')
pp.xlabel('Time (s)')
pp.ylabel('Amplitude (mPa)')
pp.legend()
pp.show()

pp.figure(2)
pp.clf()
pp.plot(noise_signal_times,noise_signal_td)#,label ='z =' + str(zpos))
pp.grid('on',which='both',axis='x')
pp.grid('on',which='both',axis='y')
pp.title('E = 1e11 GeV, r = 300 m, z = 6 m')
pp.xlabel('Time (s)')
pp.ylabel('Amplitude (mPa)')
pp.legend()
pp.show()
        