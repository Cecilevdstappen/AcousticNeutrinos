#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:28:33 2022

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
#########################################################################
### data_td, noise plus signal.
# data file is just one column. Sample freq is 144kHz
L_max_snr = []
L_scale = []

zpos = 6
ypos = 300

plot_fhigh = True
plot_flow = False
plot_SNR_fd = False
plot_Signal_fd = False
plot_SNR_scale = False
plot_spectrum_data = False
transfer_function = False
plot_signal_data=False
ax = pp.gca()
for zpos in (6,10):
    for scale in (1,10,100,1000):
        if zpos <= 8:
            data_file = '../Neutrino_data_files/neutrino_' +str(zpos)+'_300_'+str(scale)+'fixed.txt'
        else:
            data_file = '../Neutrino_data_files/neutrino_' +str(zpos)+'_300_1_11_'+str(scale)+'fixed.txt'
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
        
        flow = 20.; fhigh= 30000.
        
        
        if zpos <= 8:
            template_file = '../Neutrino_data_files/neutrino_'+str(zpos)+'_300.dat'
        else:
            template_file = '../Neutrino_data_files/neutrino_'+ str(zpos) +'_300_1_11.dat'
        data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
        time = data[0]
        amplitude_og = data[1]
        
        if transfer_function == True:
            frf = design_FRF(Fs=144000)
            #frf.highpass = True
            # frf.pz2tf((375,12000,17000),(0.15,0.02,0.07),100000) # hydrophone with cap
            frf.pz2tf((375,12000),(0.1,0.03),144000) # hydrophone without cap
            # frf.pz2tf((12000,375),(0.02,0.1),100000) # hydrophone without cap
            frf.mbutton.on()
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
        L_segment = [2048, 2048]#[131072,65536,16384,2048,256,128,8]
        for i in L_segment:
            seg_len = i
            p_estimated = noise_td.psd(dt*i, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
            p = psd.interpolate(p_estimated, noise_fd.delta_f)
            p = psd.inverse_spectrum_truncation(p, int(dt*seg_len*noise_td.sample_rate),low_frequency_cutoff=flow)
            psd_inter = p
            L_psd_inter.append(psd_inter)
        
         
        max_snr = 0.
        max_snr_t = 0.
        L_fhigh = []
        L_flow=[]
        L_pk = []
        
        L_snr = ['psd_inter','psd_seg','psd_scipy','psd_1']
           
        #    for flow in np.arange(0.,5000.,100):
        #        fhigh = 5000. 
        for fhigh in np.arange(8000,30000,100):
            flow = 0.
            max_snr = 0.
            #for fhigh in np.arange(8000,30000,100): 
            snr_inter = matched_filter(template, data_td, psd = psd_inter,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
           
            pk, pidx = snr_inter.abs_max_loc()
            peak_t = snr_inter.sample_times[pidx]
            L_fhigh.append(fhigh/1000)
            L_flow.append(flow/1000)
            L_pk.append(pk)
            if (pk > max_snr):
                max_snr_time = peak_t
                max_snr = pk
        
            #print("Maximum SNR {} = {:5.2f} at t = {:.4f} s".format(L_segment[j], max_snr, max_snr_time))
            L_max_snr.append(max_snr)
            #L_scale.append(scale)
    
        
#    ax.plot(snr_inter.sample_times, abs(snr_inter), label="%.0f"%L_segment[j])
#    ax.legend()
#    ax.set_title("matched filtering timeseries")
#    ax.grid()
##    pp.yscale('log')
##    pp.xscale('log')
#    #ax.set_ylim(0, None)
#    ax.set_xlabel('Time (s)')
#    ax.set_ylabel('Signal-to-noise (mPa$^2$Hz$^{−1}$)')
#    pp.show()
#    pp.close()
    
        if plot_signal_data==True:
            pp.figure(14)
            pp.plot(time, amplitude_og, label ='z =' + str(zpos)+',scale='+str(scale))
            pp.grid('on',which='both',axis='x')
            pp.grid('on',which='both',axis='y')
            pp.title('E = 1e11 GeV, r = 300 m')
            pp.xlabel('time (s)')
            pp.ylabel('Amplitude (mPa)')
            pp.legend()
            pp.show()
            
            pp.figure(13)
            pp.clf()
            pp.plot(noise_signal_times,noise_signal_td,label ='z =' + str(zpos)+',scale='+str(scale))
            pp.grid('on',which='both',axis='x')
            pp.grid('on',which='both',axis='y')
            pp.title('E = 1e11 GeV, r = 300 m')
            pp.xlabel('t -> [s]')
            pp.ylabel('Amplitude (mPa)')
            pp.legend()
            pp.show()
            
        if plot_Signal_fd == True:
            pp.plot(template_fd.sample_frequencies/1000, template_fd, label ='z =' + str(zpos))
            pp.title('E = 1e11 GeV, r = 300 m')
            pp.xlabel("frequency (kHz)")
            pp.ylabel('normalised power')
            pp.legend()
            pp.show()
            pp.close()
    
        if plot_fhigh == True:
            pp.plot(L_fhigh, L_pk, label ='z =' + str(zpos)+',scale='+str(scale))
            #pp.plot(L_psd_inter.sample_frequencies,L_psd_inter, label="signal with whale noise")
            pp.title("Value SNR peak as function of max frequency")
            pp.xlabel("max frequency (kHz)")
            pp.ylabel("max SNR")
            pp.legend()
            pp.savefig("SNRpeak_fhigh_plot_whalenoise_7000"+str(flow)+".png")
            pp.show()
            pp.close()
            
        if plot_flow == True:
            pp.plot(L_flow, L_pk, label = 'z =' + str(zpos)+',scale='+str(scale))
            pp.title("Value SNR peak as function of min frequency, f_max="+str(fhigh))
            pp.xlabel("min frequency (kHz)")
            pp.ylabel("max SNR")
            pp.legend()
            #pp.savefig("SNRpeak_flow_"+ str(flow)+".png")
            pp.show()
            pp.close()
        
        if plot_spectrum_data == True:
            pp.rcParams['agg.path.chunksize'] = 10000
            #pp.plot(data_fd.sample_frequencies,data_fd, label="signal with whale noise")
            pp.plot(template_fd.sample_frequencies,template_fd, label=str(zpos))
            pp.plot(psd_inter, label="signal with whale noise")
            #pp.plot(noise_fd.sample_frequencies,noise_fd,"r--", label = "whale noise")
            pp.title("Spectrum data")
            #pp.yscale("log")
            #pp.xscale("log")
            pp.legend()
            #pp.savefig("spectrum_sig_whalenoise.png")
            pp.show()
        
