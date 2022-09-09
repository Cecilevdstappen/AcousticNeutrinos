#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:16:56 2022

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

mat_contents = sio.loadmat('./Datamat_Enkel_3FF_Atran_Sweep200Hz20kHz0degreev1.mat',squeeze_me=True,struct_as_record=False) 
mat_contents = sio.loadmat('./Datamat_Enkel_3FF_Atran_Sweep200Hz20kHznoCaps0degreev2.mat',squeeze_me=True,struct_as_record=False)

im=impulse()
#Set the sampling frequency to 100kHz
im.imp_Fs   = Fs
im.imp_NFFT = 131072
im.imp_PSDOn = 0
im.imp_Segment = 144
im.imp_Data = np.squeeze(amplitude_og)
im.imp_Data[:,1] = -im.imp_Data[:,1]
#Estimate the impulse response of the artificial DUT
ImpulseResponse(im,1,0) #member of class impulse
#Plot the result
plt.figure(2);plt.clf()
hgca = plt.gcf().number
f_fig, axs= plt.subplots(5, 1,sharex=False,sharey=False, num=hgca)
f_fig.tight_layout()
axs[0].stem(np.real(im.imp_Impulse[0:im.imp_NFFT]),label='Estimated Impulse response DUT. Fs = 100kHz')
axs[0].legend(loc='upper center', shadow=True, fontsize='x-small')
axs[0].minorticks_on()
axs[0].grid('on',which='both',axis='x')
axs[0].grid('on',which='major',axis='y')
axs[0].set_title(Experiment)
axs[0].set_xlabel('tau -> []')
axs[0].set_ylabel('h(tau)')