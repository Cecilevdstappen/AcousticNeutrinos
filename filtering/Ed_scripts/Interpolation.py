#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:22:02 2022

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
rpos=300

for zpos in (10,20,10):
    data_file = '../Neutrino_data_files/neutrino' + '_' + str(zpos) + '_' + str(rpos) + '_1_11_1' + '.txt' #'whitenoise.txt'
    #data_file = '../Neutrino_data_files/neutrino_' +'6'+'_300_1fixed_highpass.txt'
    #data_file = 'neutrino_10_300_1_11_100fixed_unscrambled.txt'
    data = np.loadtxt(data_file,  usecols=(0), dtype='float', unpack=True)  
    noise_signal_td = data #np.resize(data, 131072)
    print(len(noise_signal_td))
    freq = 144000.
    noise_signal_times = np.arange(0, len(noise_signal_td), 1)
    noise_signal_times = noise_signal_times/144000.
    noise_signal_td = noise_signal_td - np.mean(noise_signal_td)


#########################################################################

    psd_1 = np.ones(len(noise_signal_td))
    psd_1 = types.FrequencySeries(psd_1, 1.0986328125)
    
    #psd = np.linspace(1,0, len(noise_signal_td))
    noise_file = 'output001.wav'
    #noise_file = "white_noise_144kHz.wav"
    wav = wave.open(noise_file, 'r')
    frames = wav.readframes(-1)
    sound_info = np.frombuffer(frames,dtype='int16')
    frame_rate = wav.getframerate()
    wav.close()

    time_trace= read(noise_file)
    time_data_whale = np.arange(0, sound_info.size) / frame_rate
    noise_whale = time_trace[1]
    noise_whale = np.resize(noise_whale,len(noise_signal_td))
    dt = 1/144000.
    noise_td = types.TimeSeries(noise_whale, dt)
    noise_fd = abs(noise_td.to_frequencyseries())
    
    flow = 1000.; fhigh= 35000.
    

#########################################################################
### template is a theoretical prediction
## Q1 amplitude schalen?
## Q2 lengte tijdsas en offset?
    if zpos <= 8:
        template_file = '../Neutrino_data_files/neutrino_6_300.dat'
    else:
        template_file = '../Neutrino_data_files/neutrino_'+ str(zpos) +'_300_1_11.dat'
    data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
    time = data[0]
    amplitude_og = data[1]
    
#    if transfer_function == True:
#        frf = design_FRF(Fs=144000)
#        #frf.highpass = True
#        # frf.pz2tf((375,12000,17000),(0.15,0.02,0.07),100000) # hydrophone with cap
#        frf.pz2tf((375,12000),(0.1,0.03),144000) # hydrophone without cap
#        # frf.pz2tf((12000,375),(0.02,0.1),100000) # hydrophone without cap
#        frf.mbutton.on()
#        amplitude_og = sg.lfilter(frf.num,frf.dnum,amplitude_og)
#        noise_signal_td = sg.lfilter(frf.num,frf.dnum,noise_signal_td)
        
    
    amplitude = np.pad(amplitude_og, (len(noise_signal_td)-len(amplitude_og), 0) )
    amplitude = amplitude * 100
    dt = 1/144000.
    template = types.TimeSeries(amplitude, dt)
    template_fd = abs(template.to_frequencyseries())
    #noise_signal_td= np.resize(noise_signal_td,len(psd_scipy))
    data_td = types.TimeSeries(noise_signal_td, dt)
    
    
    L_psd_inter = []
    L_segment = 2048 #[131072,65536,16384,2048,256,128,8]
    #for i in L_segment:
    seg_len = 2048
    p_estimated = noise_td.psd(dt*L_segment, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
    p = psd.interpolate(p_estimated, noise_fd.delta_f)
    p = psd.inverse_spectrum_truncation(p, int(dt*seg_len*noise_td.sample_rate),low_frequency_cutoff=flow)
    psd_inter = p
    L_psd_inter.append(psd_inter)
   
    
    pp.plot(psd_inter.sample_frequencies/1000, psd_inter, label="PSD with interpolation")
    pp.scatter(p_estimated.sample_frequencies/1000, p_estimated,color='crimson',label = "PSD without interpolation")
    pp.plot([], [], ' ', label="Segment length psd = %.0f"%L_segment)
    pp.title("Estimated psd")
    pp.yscale('log')
    pp.xscale('log')
    pp.xlabel("frequency (kHz)")
    pp.ylabel("PSD (dB Re 1mPa$^2$/Hz)")
    pp.xlim(1,35)
    pp.legend(fontsize = 10)
    pp.show()
    pp.close()