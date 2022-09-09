#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:58:19 2022

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

#########################################################################
### data_td, noise plus signal.
# data file is just one column. Sample freq is 144kHz
L_max_snr = []
scaling = 1
rpos = 300
zpos = 6
energy = 11
transfer_function =False
for z_data in np.arange(0,21,1):
#    if zpos <= 8:
    data_file = '../Neutrino_data_files/neutrino_'+str(float(z_data))+'_'+str(rpos)+'_1_'+str(energy)+'_'+str(scaling)+'.txt'
    
    
    data = np.loadtxt(data_file,  usecols=(0), dtype='float', unpack=True)  
    noise_signal_td = data
    
    freq = 144000.
    noise_signal_times = np.arange(0, len(noise_signal_td), 1)
    noise_signal_times = noise_signal_times/144000.
    noise_signal_td = noise_signal_td - np.mean(noise_signal_td)
    
    
    #########################################################################
    
    flow = 20.; fhigh= 30000.
    
    #########################################################################
    ### template is a theoretical prediction
    ## Q1 amplitude schalen?
    ## Q2 lengte tijdsas en offset?
    template_file = '../Neutrino_data_files/neutrino_'+str(float(zpos))+'_'+str(rpos)+'_1_'+str(energy)+'.dat'
    #template_file = 'neutrino_6_300.dat'
    data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
    time = data[0]
    amplitude_og = data[1]
    
    if transfer_function == True:
        frf = design_FRF(Fs=144000)
        #frf.highpass = True
        # frf.pz2tf((375,12000,17000),(0.15,0.02,0.07),100000) # hydrophone with cap
        frf.pz2tf((375,12000),(0.1,0.03),144000) # hydrophone without cap
        # frf.pz2tf((12000,375),(0.02,0.1),100000) # hydrophone without cap
        #frf.mbutton.on()
        amplitude_og = sg.lfilter(frf.num,frf.dnum,amplitude_og)
        noise_signal_td = sg.lfilter(frf.num,frf.dnum,noise_signal_td)
        
    
    amplitude = np.pad(amplitude_og, (len(noise_signal_td)-len(amplitude_og), 0) )
    amplitude = amplitude * 100
    dt = 1/144000.
    
    
    template = types.TimeSeries(amplitude, dt)
    template_fd = abs(template.to_frequencyseries())
    #noise_signal_td= np.resize(noise_signal_td,len(psd_scipy))
    data_td = types.TimeSeries(noise_signal_td, dt)
    L_psd_inter = []
    L_psd_seg = []
    L_psd_scipy = []
    L_segment = [2048]#[131072,65536,16384,2048,256,128,8]
    for i in L_segment:
        seg_len = i
        p_estimated = data_td.psd(dt*i, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
        p = psd.interpolate(p_estimated, data_td.delta_f)
        p = psd.inverse_spectrum_truncation(p, int(dt*seg_len*data_td.sample_rate),low_frequency_cutoff=flow)
        psd_inter = p
        L_psd_inter.append(psd_inter)
       

        
        #pp.plot(psd_scipy.sample_frequencies, psd_scipy, label = "segment length %.0f"%i)
        pp.plot(psd_inter.sample_frequencies, psd_inter, label="segment length %.0f"%i)    
    pp.title("Estimated psd")
    pp.yscale('log')
    pp.xscale('log')
    pp.xlabel("frequency")
    pp.ylabel("psd")
    pp.legend()
    pp.show()
    pp.close()
     
    max_snr = 0.
    max_snr_t = 0.
    L_fhigh = []
    L_pk = []
    
    L_snr = ['psd_inter','psd_seg','psd_scipy','psd_1']
    ax = pp.gca()
    #for flow in np.arange(0,20000,10):
    #for i in L_snr:
    for j in np.arange(0,1,1):#(0,len(L_segment)-1,1):
        max_snr = 0.
        #for fhigh in np.arange(8000,30000,100): 
        snr_inter = matched_filter(template, data_td, psd = L_psd_inter[j],low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
        pk, pidx = snr_inter.abs_max_loc()
        peak_t = snr_inter.sample_times[pidx]
        L_fhigh.append(fhigh)
        L_pk.append(pk)
        if (pk > max_snr):
            max_snr_time = peak_t
            max_snr = pk
    
        print("Maximum SNR {} = {:5.2f} at t = {:.4f} s for z position {:.1f}".format(L_segment[j], max_snr, max_snr_time, z_data))
        L_max_snr.append(max_snr)
    
        
        ax.plot(snr_inter.sample_times, abs(snr_inter), label="%.0f"%L_segment[j])
    ax.legend()
    ax.set_title("matched filtering timeseries")
    ax.grid()
    #    pp.yscale('log')
    #    pp.xscale('log')
    #ax.set_ylim(0, None)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal-to-noise (mPa$^2$Hz$^{−1}$)')
    pp.show()
    pp.close()
    
    

    


