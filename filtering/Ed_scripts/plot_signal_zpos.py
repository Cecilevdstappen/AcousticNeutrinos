#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 15:47:11 2022

@author: gebruiker
"""

import numpy as np
import wave
import scipy.signal as sg
from scipy.io.wavfile import read
import matplotlib.pyplot as pp

#########################################################################
### data_td, noise plus signal.
# data file is just one column. Sample freq is 144kHz
L_max_snr = []
scale = 71
for z_data in np.arange(-4,-2,2):
    for scaling in np.arange(1,2,1): 
        #scaling = scaling*100000
        #zpos = float(z)
        ypos = 300
        
        #template_file = 'neutrino' + '_' + str(zpos) +'_150_1_11_144kHz.dat'
        template_file = 'neutrino_'+str(z_data) +'_300.dat'
        #template_file = 'neutrino_6_300.dat'
        data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
        time = data[0]
        amplitude_og = data[1]
        
        data_file = 'neutrino_' +'90'+'_300_1_11_'+str(scaling)+'.txt'
        data = np.loadtxt(data_file,  usecols=(0), dtype='float', unpack=True)  
        noise_signal_td = data
        noise_signal_resize = np.resize(noise_signal_td,131072)
           
        #pp.figure(1)
        #pp.clf()
        #pp.plot(time, amplitude_og, label ="transfer function signal")
        pp.plot(noise_signal_td, label=scaling, zorder=1/z_data)
#pp.plot(noise_signal_resize,"--",label="resized")
pp.grid('on',which='both',axis='x')
pp.grid('on',which='both',axis='y')
#pp.title(z_data)
pp.xlabel('time (s)')
pp.ylabel('Amplitude (mPa)')
pp.legend()
pp.show()