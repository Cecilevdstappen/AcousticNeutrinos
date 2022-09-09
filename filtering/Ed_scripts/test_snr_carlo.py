#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:06:53 2022

@author: gebruiker
"""
import matplotlib.pyplot as pp
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
from SNR import resample_by_interpolation
from my_styles import *

set_paper_style()
def butter_highpass(cutoff, Fs, order=5):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    b, a = sg.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    

def butter_highpass_filter(data, cutoff, Fs, order=5):
    b, a = butter_highpass(cutoff, Fs, order=order)
    y = sg.lfilter(b, a, data)
    return y

pp.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')
data_file     = "output.wav"
time_series    = read(data_file)
sampling_rate = time_series[0]
data_array    = time_series[1]

random_nr = np.random.rand(1000)


count =0
reconstruction = 0
for i in np.arange(0,50,1):
    trace_length = pow(2,17)
    random_start = np.random.randint(0,100000,1)
    trace_start = random_start
    trace_end = trace_start + trace_length
    cut = 4
    # determine a noise realisation,
    # based on the FFT of the data that have been read
    # using different, random phase.
    # Used polar coordinates for complex numbers
    noise_scaling = 2e1
    y = data_array[int(trace_start) : int(trace_end)]
    Y = np.fft.fft(y)
    m = np.abs(Y)
    phase = 2*np.pi*random_nr[i]*(len(m)) - np.pi #55 is random
    Z = m*np.cos(phase) + 1j*m*np.sin(phase)
    z = np.fft.ifft(Z)
    z = z.real
    z*= noise_scaling
    
     
    signal_amplitude =700
    signal_period = 0.0002  #0.0002
    
    template_amplitude = 700#175
    template_period = 0.0002#0.001
    
    start = 0.1
    
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    n = 2**(11)
    dt =1/n
    x = np.linspace(0, 1, n)
    y_noise = z #noise_amplitude*np.random.normal(0,1,n)
    
    y_signal = np.zeros(len(x))
    y_template = np.zeros(len(x))
    
    signal_freq = 2/signal_period
    template_freq= 2/template_period
    
    for i, _x in enumerate(x):
        if ((_x > start) & (_x < start+signal_period)):
            y_signal[i] += signal_amplitude*np.sin(signal_freq*_x)
        if ((_x > start) & (_x < start+template_period)):  
            y_template[i] += template_amplitude*np.sin(template_freq*_x)# Sinus signal
        #y_signal[i] += signal_amplitude*gaussian(_x,start+signal_period/2, signal_period) #Gaussian signal
        #y_template[i]+= template_amplitude*gaussian(_x,start+template_period/2, template_period)
    
    #energy = 11
    #rpos = 300
    #z_template = 10
    #z_signal = 10
    #template_file = '../Neutrino_data_files/neutrino_'+str(float(z_template)) +'_'+str(rpos)+'_1_'+str(energy)+'.dat'
    #data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
    #time = data[0]
    #amplitude_og = data[1]
    #y_template =amplitude_og*1000
    #signal_file = '../Neutrino_data_files/neutrino_'+str(float(z_signal)) +'_'+str(rpos)+'_1_'+str(energy)+'.dat'
    #data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
    #time = data[0]
    ##dt = time[1]-time[0]
    #print('dt = '+str(dt))
    #y_signal = data[1]
    #y_signal = y_signal*10000
    #
    
    
    entry_point = int(len(z)/2)
    
    x = np.pad(y_signal, (entry_point, len(z)-len(y_signal)-entry_point),
               'constant', constant_values=(0., 0.))
    # add noise and signal and make sure that it of proper format
    
    data = (z + x) +np.mean(y)*np.ones(len(z))
    data = np.asarray(data, dtype = np.int16)
    data = butter_highpass_filter(data, 35, sampling_rate, order=5)
    data = types.TimeSeries(data, dt)
    template = types.TimeSeries(y_template[::-1], dt)
    template.resize(len(data))
    #template.resize(len(noise_td))
    data_fd = abs(data.to_frequencyseries())
    template_fd = abs(template.to_frequencyseries())
    cut = 50
    seg_len = 2048
    p_estimated = data.psd(dt*seg_len, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
    psd_inter = psd.interpolate(p_estimated, data.delta_f)
    snr = matched_filter(template, data, psd = psd_inter)
    snr = snr.crop(5, 5)
    
#    pp.plot(data.sample_times, data, label = 'data, A = '+str(signal_amplitude)+', period = '+str(signal_period))
#    pp.plot(template.sample_times+31.2, template,label = 'template, A = '+str(template_amplitude)+', period = '+str(template_period))
#    #pp.title('E = 1e'+str(energy)+' GeV, r ='+str(rpos)+' m)
#    pp.xlabel('Time (s)')
#    pp.ylabel('Amplitude (mPa)')
#    pp.legend()
#    pp.show()
#    pp.plot(snr.sample_times,abs(snr),label = 'A = '+str(template_amplitude)+', period = '+str(template_period))
#    pp.xlabel('Time (s)')
#    pp.ylabel('Signal-to-noise (dB)')
#    pp.legend()
#    pp.show()
    data = data.crop(5,5)
    snr_good = np.where(snr>cut)
    snr = types.TimeSeries(snr,dt)
    for i, j in enumerate(snr_good[0]):
        if abs(snr_good[0][i]-snr_good[0][i-1])>2:
            print(data.sample_times[entry_point])
            print((snr.sample_times[j]))
            delta_t = data.sample_times[entry_point] - (snr.sample_times[j])-0.1
            if abs(delta_t) <0.0005:
                count += 1
    
    if count >=1:   
        reconstruction += 1
         
print(reconstruction)
