#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 15:26:41 2022

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



def fhighflow(snr, fl_min, fl_max, fh_min,_fh_max):
    
    for flow in np.arange(0,20000,10):
        for fhigh in np.arange(8000,30000,100):  
        
            pk, pidx = snr.abs_max_loc()
            peak_t = snr.sample_times[pidx]
            L_fhigh.append(fhigh)
            L_pk.append(pk)

    return L_pk,L_fhigh

def getmaxSNR(snr):
        pk, pidx = snr.abs_max_loc()
        peak_t = snr.sample_times[pidx]
        L_fhigh.append(fhigh)
        L_pk.append(pk)
        if (pk > max_snr):
            max_snr_time = peak_t
            max_snr = pk
            

def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)
    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

def plot_fhigh_pk(L_fhigh, L_pk, scale, flow):
    pp.plot(L_fhigh, L_pk, label = str(scale))
    pp.title("Value SNR peak as function of max frequency")
    pp.xlabel("max frequency")
    pp.ylabel("max SNR")
    pp.legend()
    pp.savefig("SNRpeak_fhigh_plot_whalenoise_7000"+str(flow)+".png")
    pp.show()
    pp.close()
    
    return