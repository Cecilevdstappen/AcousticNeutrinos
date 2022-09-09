#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:14:24 2022

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
from pycbc.filter import sigma
import pylab

set_paper_style()

plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')
#########################################################################
### data_td, noise plus signal.
# data file is just one column. Sample freq is 144kHz
L_max_snr = []
L_scale = []

zpos = 6
rpos = 300
energy=11
scaling = 1

ax = pp.gca()
for zpos in ([6]):
    data_file = '../Neutrino_data_files/neutrino' + '_' + str(float(zpos))+'_'+str(rpos)+'_1_'+str(energy)+'_'+str(scaling)+'.txt'
    template_file = '../Neutrino_data_files/neutrino_'+str(float(zpos)) +'_'+str(rpos)+'_1_'+str(energy)+'.dat'
    #data_file = 'neutrino_10_300_1_11_100fixed_unscrambled.txt'
    data = np.loadtxt(data_file,  usecols=(0), dtype='float', unpack=True)  
    noise_signal_td = data #np.resize(data, 131072)
    print(len(noise_signal_td))
    freq = 144000.
    noise_signal_times = np.arange(0, len(noise_signal_td), 1)
    noise_signal_times = noise_signal_times/144000.
    noise_signal_td = noise_signal_td - np.mean(noise_signal_td)


#########################################################################

    
    flow = 1000.; fhigh= 35000.
    

#########################################################################

    
    data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
    time = data[0]
    amplitude_og = data[1]
    max_y = np.argmax(amplitude_og)  # Find the maximum y value
    max_t = time[max_y]
    
    amplitude = np.pad(amplitude_og, (len(noise_signal_td)-len(amplitude_og), 0) )
    amplitude = amplitude*1000
    dt = 1/144000.

    
    template = types.TimeSeries(amplitude, dt)
    template_fd = abs(template.to_frequencyseries())
    #noise_signal_td= np.resize(noise_signal_td,len(psd_scipy))
    data_td = types.TimeSeries(noise_signal_td, dt)
    L_segment = [2048]
    for i in L_segment:
        seg_len = i
        p_estimated = data_td.psd(dt*i, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
        p = psd.interpolate(p_estimated, data_td.delta_f)
        p = psd.inverse_spectrum_truncation(p, int(dt*seg_len*data_td.sample_rate),low_frequency_cutoff=flow)
        psd_inter = p
       
    max_snr = 0.
    max_snr_t = 0.
    L_fhigh = []
    L_pk = []
    
    
    max_snr = 0.
    #for fhigh in np.arange(8000,30000,100): 
    snr_inter = matched_filter(template, data_td, psd = psd_inter,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)

    pk, pidx = snr_inter.abs_max_loc()
    peak_t = snr_inter.sample_times[pidx]
    L_fhigh.append(fhigh)
    L_pk.append(pk)
    if (pk > max_snr):
        max_snr_time = peak_t
        max_snr = pk

    print("Maximum SNR {} = {:5.2f} at t = {:.4f} s".format(2048, max_snr, max_snr_time))
    L_max_snr.append(max_snr)
    print(max_snr_time)
    #L_scale.append(scale)
    
    
    max_snr_time = 0.462
    
    
    # The time, amplitude, and phase of the SNR peak tell us how to align
    # our proposed signal with the data.
    template = template.cyclic_time_shift(template.start_time)
    pylab.plot(template)
    pylab.show()
    # Shift the template to the peak time
    dt = max_snr_time - data_td.start_time
    print('data starttime ='+ str(data_td.start_time))
    aligned = template.cyclic_time_shift(max_snr_time+0.00733)
    
    # scale the template so that it would have SNR 1 in this data
    aligned /= sigma(aligned, psd=psd_inter, low_frequency_cutoff=1000.0)
    
    # Scale the template amplitude and phase to the peak value
    aligned = (aligned.to_frequencyseries() * max_snr).to_timeseries()
    aligned.start_time = data_td.start_time
    
    white_data = (data_td.to_frequencyseries() / psd_inter**0.5).to_timeseries()

#    # apply a smoothing of the turnon of the template to avoid a transient
#    # from the sharp turn on in the waveform.
#    tapered = aligned.highpass_fir(30, 512, remove_corrupted=False)
    white_template = (aligned.to_frequencyseries() / psd_inter**0.5).to_timeseries()
#    
#    white_data = white_data.highpass_fir(30., 512).lowpass_fir(300, 512)
#    white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)
    if zpos > 10:
        white_data = white_data.time_slice(max_snr_time-.02, max_snr_time+.01)
        white_template = white_template.time_slice(max_snr_time-.02, max_snr_time+.01)
    if zpos <= 10:
        white_data = white_data.time_slice(max_snr_time-.002, max_snr_time+.001)
        white_template = white_template.time_slice(max_snr_time-.002, max_snr_time+.001)
    
    pylab.figure(figsize=[15, 3])
    pylab.plot(white_data.sample_times, white_data/42.67, label="Data")
    pylab.plot(white_template.sample_times, white_template/42.67, label="Template")
    pp.title('E = 1e'+str(energy)+' GeV, r ='+str(rpos)+' m, z ='+ str(zpos)+' m')
    pp.xlabel('time (s)')
    pp.ylabel('Amplitude (mPa)')
    pylab.legend()
    pylab.show()


