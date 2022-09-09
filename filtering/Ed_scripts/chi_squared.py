#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:18:50 2022

@author: gebruiker
""" 
import matplotlib.pyplot as pp
from pycbc.filter import matched_filter, get_cutoff_indices #, make_frequency_series
from pycbc import types
from pycbc.types.array import complex_same_precision_as
from pycbc import psd
from pycbc import fft
import sys
sys.path.insert(0, '/home/gebruiker/SIPPY')
sys.path.insert(0, '/home/gebruiker/AcousticNeutrinos/filtering/design_FRF')
import numpy as np
import wave
import scipy.signal as sg
from scipy.io.wavfile import read
from impulse import *
from mouse_button import *
from design_FRF import *
from SNR import resample_by_interpolation
from our_snr import *

from pycbc.vetoes import power_chisq

chisq = {}

# The number of bins to use. In principle, this choice is arbitrary. In practice,
# this is empirically tuned.
nbins = 2
chisq_inter = power_chisq(template, data_td, nbins, psd = psd_inter, low_frequency_cutoff=flow)
#chisq_inter = chisq_inter.crop(3, 2)

# Scale by the number of degrees of freedom
#dof = nbins * 2 - 2
#chisq_inter /= dof
# Show a couple of sizes
print('length chi-squared'+str(len(chisq_inter)))

fig = plt.figure(figsize=[10, 5])
ax = plt.gca()
ax.plot(chisq_inter.sample_times, chisq_inter)
ax.legend()
ax.set_title("chi-squared for psd_inter")
ax.grid()
ax.set_ylim(0, None)
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\chi^2_r$')
plt.show()
plt.close()