#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:06:37 2022

@author: gebruiker
"""

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
os.path.join('/home/gebruiker/AcousticNeutrinos/filtering/Neutrino_data_files')
import numpy as np
import wave
import scipy.signal as sg
from scipy.io.wavfile import read
from impulse import *
from mouse_button import *
from design_FRF import *
from SNR import resample_by_interpolation
from pycbc.vetoes import power_chisq
from my_styles import *
set_paper_style()

plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')
#########################################################################
### data_td, noise plus signal.
# data file is just one column. Sample freq is 144kHz



L_max_snr = [] 

scaling = 1
rpos = 20
transfer_function=False
#for z_data in np.arange(10,110,10):
for energy in ([9]): 
    print("E = 1e"+str(energy))
    L_zpos = []
    L_max_snr_time = []
    for zpos in np.arange(1,21,1):
        L_zpos.append(zpos)
        max_snr = 0.
        L_amplitude_max = []
        
        data_file = '../Neutrino_data_files/neutrino_'+str(float(zpos))+'_'+str(rpos)+'_1_'+str(energy)+'_'+str(scaling)+'ss2.txt'
        template_file = '../Neutrino_data_files/neutrino_'+str(float(zpos))+'_'+str(rpos)+'_1_'+str(energy)+'.dat'
        data = np.loadtxt(data_file,  usecols=(0), dtype='float', unpack=True)  
        noise_signal_td = data
        #amplitude_max = max(noise_signal_td)
        
        #noise_signal_td = np.resize(noise_signal_td,131072)
        #print(len(noise_signal_td))
        freq = 144000.
        noise_signal_times = np.arange(0, len(noise_signal_td), 1)
        noise_signal_times = noise_signal_times/144000.
        noise_signal_td = noise_signal_td - np.mean(noise_signal_td)
        
       
        
        
        #########################################################################
        
        flow = 500.; fhigh= 30000.
        
        
        
        
        #    f, Pxx_den = sg.periodogram(noise_whale, fs=144000., nfft=512)
        #    Pxx_den = resample_by_interpolation(Pxx_den,257,65537)
        #    psd_period = types.FrequencySeries(Pxx_den, noise_fd.delta_f)
        
        #psd_scipy = psd.interpolate(p_scipy, noise_fd.delta_f)
        #psd_scipy = psd.inverse_spectrum_truncation(psd_scipy, int(dt*512*noise_td.sample_rate),low_frequency_cutoff=flow)
        
        
        
        #########################################################################
    
        
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
            
        amplitude_og =amplitude_og*1000
        amplitude_max = max(amplitude_og)
        
        L_amplitude_max.append(amplitude_max)
        amplitude = np.pad(amplitude_og, (len(noise_signal_td)-len(amplitude_og), 0) )
        dt = 1/144000.
        
        template = types.TimeSeries(amplitude, dt)
        template_fd = abs(template.to_frequencyseries())
        #noise_signal_td= np.resize(noise_signal_td,len(psd_scipy))
        data_td = types.TimeSeries(noise_signal_td, dt)
    
        L_psd_inter = []
        L_psd_seg = []
        L_psd_scipy = []
        L_segment = [2048] #length noisr_signal_td
        for i in L_segment:
            seg_len = i
            p_estimated = data_td.psd(dt*i, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
            p = psd.interpolate(p_estimated, data_td.delta_f)
            #p = psd.inverse_spectrum_truncation(p, int(dt*seg_len*data_td.sample_rate),low_frequency_cutoff=flow)
            psd_inter = p
            L_psd_inter.append(psd_inter)
 
         
        
        L_fhigh = []
        L_pk = []
        L_snr = ['psd_inter','psd_seg','psd_scipy','psd_1']
    #        print(template.delta_f)
    #        print(data_td.delta_f)
           
        #for flow in np.arange(0,20000,10):
        #for i in L_snr:
        j = 0
        #max_snr = 0.
        #for fhigh in np.arange(8000,30000,100): 
        snr_inter = matched_filter(template, data_td, psd = psd_inter,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
        snr_inter = snr_inter.crop(0.2, 0.2)
        pk, pidx = snr_inter.abs_max_loc()
        peak_t = snr_inter.sample_times[pidx]
        L_fhigh.append(fhigh)
        L_pk.append(pk)
        
        L_max_snr_time.append(peak_t)                        
        
    pp.plot(L_zpos, L_max_snr_time, '-',label="E = 1e" +str(energy))
pp.plot(L_zpos, np.ones((len(L_zpos,)))*0.462 ,'--', color = 'black',label='Simulated signal time')
pp.title('E = [1e9,1e13] GeV, r = '+ str(rpos)+' m, sea state 0, incl. tf')
pp.grid('on',which='both',axis='x')
pp.grid('on',which='both',axis='y')
pp.xlabel('Z position (m)')
pp.ylabel('Max SNR time (s)')
pp.ylim(0,1.2)
pp.legend(ncol=3)#(ncol = 3, frame = True)
pp.show()

            


    

    


