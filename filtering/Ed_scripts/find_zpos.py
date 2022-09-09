#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:48:09 2022

@author: gebruiker
"""

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
from my_styles import *
set_paper_style()

plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')
#########################################################################
### data_td, noise plus signal.
# data file is just one column. Sample freq is 144kHz

transfer_function=False
r_pos = 300
energy = 13
scaling = 1
#for energy in np.arange(12,14,1):
for energy in [9,10,11,12,13]:
    print('now at energy E = 1e'+str(energy))
    L_zpos_reco = []
    #for z_data in np.arange(10,110,10):
    L_zpos_simu = []
    for z_data in np.arange(1,21,1):
        max_snr = 0.
        L_max_snr = [] 
        L_zpos = []
        
        L_zpos_simu.append(z_data)
        for z_template in np.arange(1,21,1):
            L_zpos.append(z_template) 
            L_amplitude_max = []            
    
            data_file = '../Neutrino_data_files/neutrino' + '_' + str(float(z_data))+'_'+str(r_pos)+'_1_'+str(energy)+'_'+str(scaling)+'.txt'
            template_file = '../Neutrino_data_files/neutrino_'+str(float(z_template)) +'_'+str(r_pos)+'_1_'+str(energy)+'.dat'
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
            
        
            
            #########################################################################
        
            
            data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
            time = data[0]
            amplitude_og = data[1]
            amplitude_og =amplitude_og*1000 #convert to mPa
            amplitude_max = max(amplitude_og)
            
            L_amplitude_max.append(amplitude_max)
            
                    
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
            #amplitude = amplitude[::-1]
            dt = 1/144000.
            
            template = types.TimeSeries(amplitude, dt)
            template_fd = abs(template.to_frequencyseries())
            #noise_signal_td= np.resize(noise_signal_td,len(psd_scipy))
            data_td = types.TimeSeries(noise_signal_td, dt)
        
            L_psd_inter = []
            L_psd_seg = []
            L_psd_scipy = []
            L_segment = [32] #length noisr_signal_td
            for i in L_segment:
                seg_len = i
                p_estimated = data_td.psd(dt*i, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
                p = psd.interpolate(p_estimated, data_td.delta_f)
                #p = psd.inverse_spectrum_truncation(p, int(dt*seg_len*data_td.sample_rate),low_frequency_cutoff=flow)
                psd_inter = p
                L_psd_inter.append(psd_inter)
        
            max_snr_t = 0.
            L_fhigh = []
            L_pk = []
            L_snr = ['psd_inter','psd_seg','psd_scipy','psd_1']
    
            snr_inter = matched_filter(template, data_td, psd = psd_inter,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
            pk, pidx = snr_inter.abs_max_loc()
            peak_t = snr_inter.sample_times[pidx]
            L_fhigh.append(fhigh)
            L_pk.append(pk)
            #if (pk > max_snr):
            max_snr_time = peak_t
            max_snr = pk
    
            L_max_snr.append(max_snr)
            
            
#        pp.plot( np.arange(0,21,1),L_max_snr)
#        pp.show()
        max_value = max(L_max_snr)
        max_index = L_max_snr.index(max_value)
        z_reco = L_zpos[max_index]
        L_zpos_reco.append(z_reco)
                
        #print('The max SNR is {:.1f} at z position: {:.1f}, the actual z position was {:.1f}'.format(max_value, z_reco,z_data))
        #print(L_max_snr)
        
    
    pp.figure(2)
    #pp.plot(L_zpos,L_amplitude_max,label=(z_data, transferfunction))
    pp.plot(L_zpos_simu, L_zpos_reco,'-',label = "E = 1e" +str(energy))
    pp.scatter(L_zpos_simu, L_zpos_reco)
    pp.grid('on',which='both',axis='x')
    pp.grid('on',which='both',axis='y')
    pp.title('E = [1e9,1e13] GeV, r ='+str(r_pos)+' m')
    #pp.title('Amplitude vs z position, not scaled')
    #pp.xlabel('template zpos')
    pp.xlabel('Simulated z position (m)')
    pp.ylabel('Reconstructed z position (m)')
    #pp.yscale('log')
    #pp.xscale('log')
pp.plot(L_zpos_simu, L_zpos_simu, '-', label = "Correctly reconstructed z")
pp.legend(ncol=3)
pp.show()
pp.close()
                    


    

    


