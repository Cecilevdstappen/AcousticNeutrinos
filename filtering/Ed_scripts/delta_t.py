#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:15:04 2022

@author: gebruiker
"""


import matplotlib.pyplot as pp
    
import pycbc.filter
import pycbc.psd
from pycbc import types
from pycbc.types.array import complex_same_precision_as
from pycbc import fft
import sys
#sys.path.insert(0, '/home/gebruiker/AcousticNeutrinos/filtering/design_FRF')
#sys.path.insert(0, '/home/gebruiker/SIPPY')
#from numpy.fft import fft as npfft
import numpy as np
from numpy.linalg import norm as Norm
import wave
from scipy import signal
from scipy.io.wavfile import read
from scipy.signal import butter, lfilter
#from impulse import *
import math
import pickle

#from design_FRF import *

from my_styles import *

set_paper_style()

#plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')
Nperseg = int(2048)

def butter_highpass(cutoff, Fs, order=5):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, Fs, order=5):
    b, a = butter_highpass(cutoff, Fs, order=order)
    y = lfilter(b, a, data)
    return y

def Knudsen_curve(seastate = 0, frequency=1000):
    assert seastate < 7
    ss = [44.5, 50, 55., 61.5, 64.5, 66.5, 70]
    i = 1
    if frequency >0 :
        return ss[seastate] -17.* math.log10(frequency/1000.)
    else:
        return 1

def frequency_span(Fs, NFFT):
    return np.linspace(0, int(NFFT/2-1), int(NFFT/2))*Fs/NFFT

def spectrum(NFFT, Fs, data):
    length     = len(data)
    PSD        = np.zeros((int(NFFT/2), 1), dtype=float)
    freq       = frequency_span(Fs, NFFT)
    Segment    = int(length/NFFT)
    print("segment length =", NFFT)
    wnd        = np.hanning(NFFT)
    norm       = Norm(wnd)**2
    double2single   = 2.0
    for span in range(0, Segment):
        Bg          = span*NFFT
        end         = Bg+NFFT
        yw          = wnd*data[Bg:end]
        a           = np.fft.fft(yw, NFFT)
        ac          = np.conj(a)
        pxx         = np.abs(a*ac)
        PSD[:, 0]  +=  double2single*pxx[0:int(NFFT/2)]
        PSD[:, 0]  /= (float(Segment)*NFFT*norm) 
        #return 10*np.log10(PSD[:, 0])
        return types.FrequencySeries(10*np.log10(PSD[:, 0]), freq[1] - freq[0])

def resampled_signal(filename):
    time, bip = np.loadtxt(filename, usecols=(0,1), unpack=True)

    return time, bip

#    Fs = 1/(time[1] - time [0])
#    Fs_resampled = 144000.
#
#    number_resampled = int(round(len(bip)*Fs_resampled/Fs)) ## THIS MAY STRANGE RESULTS, do not rely on it..
#    bip_resampled = signal.resample(bip, number_resampled)
#    t_resampled   = np.linspace(0, (len(bip_resampled)-1)/Fs_resampled,
#                                len(bip_resampled))
#   
#    return t_resampled, bip_resampled
#    
def rename_me(filename, scaling):
    return filename.rsplit('.', 1)[0] + '_' + str(scaling)+ '.wav'
        
def write_out(bipfilename, scaling, bmax, bmin):
    print ("Generated file: ", bipfilename, scaling, bmax, bmin)
    
def datafile(filename):
    # read one column time trace (sample freq 144 kHz)
    data = np.loadtxt(filename,  usecols=(0,1), dtype='float', unpack=True) 
    dt = 1/144000.
    data_td = types.TimeSeries(data[1], dt)

    return data_td

def padme(array1, array2):
    # make length as the longest array
    if len(array1) > len(array2):
        array2 = np.pad(array2, (len(array1)-len(array2), 0) )
        return array1, array2
    elif len(array2) > len(array1):
        array1 = np.pad(array1, (len(array2)-len(array1), 0) )
        return array1, array2
    else:
        return array1, array2

def main(argv):
    # Need to get a SNR array using matched filtering
    # Matched filtering: template, data and a noise PSD

    random_nr = np.random.rand(1000)
    for energy in np.arange(9,13,1):
        print(energy)
        for zpos in np.arange(6,6.1,0.1):
            L_delta_t = []
            L_max_snr = []
            
            for i in range(0,1000):
                rpos = 300
                template_file = '../Neutrino_data_files/neutrino_'+str(round(float(zpos),1))+'_'+str(rpos)+'_1_'+str(energy)+'.dat' #argv[1]
                template = datafile(template_file)#to mPa
            
                data_file     = "output.wav"
                time_series    = read(data_file)
                sampling_rate = time_series[0]
                data_array    = time_series[1]
          
                trace_length = pow(2,17)
                random_start = np.random.randint(0,100000,1)
                trace_start = random_start
                trace_end = trace_start + trace_length
            
                # determine a noise realisation,
                # based on the FFT of the data that have been read
                # using different, random phase.
                # Used polar coordinates for complex numbers
                noise_scaling = 2e1
                y = data_array[int(trace_start) : int(trace_end)]
                Y = np.fft.fft(y)
                m = np.abs(Y)
                phase = 2*np.pi*random_nr[i-1]*(len(m)) - np.pi
                Z = m*np.cos(phase) + 1j*m*np.sin(phase)
                z = np.fft.ifft(Z)
                z = z.real
                z*= noise_scaling
            
                # read ascii file with the neutrino click
                bipfilename = '../Neutrino_data_files/neutrino_'+str(round(float(zpos),1))+'_'+str(rpos)+'_1_'+str(energy)+'.dat'
                time, bip  = resampled_signal(bipfilename)

                # padding to insert the neutrino click somewhere
                # (random position) in the data stream
                #entry_point = np.random.randint(0,len(data_array) - 20000, 1)[0]
                entry_point = int(len(z)/2)
            
                scaling = 1000 #to mPa
                bip *= scaling
                x = np.pad(bip, (entry_point, len(z)-len(bip)-entry_point),
                           'constant', constant_values=(0., 0.))
                # add noise and signal and make sure that it of proper format
            
                data = (z + x) +np.mean(y)*np.ones(len(z))
                data = np.asarray(data, dtype = np.int16)
                data = butter_highpass_filter(data, 35, sampling_rate, order=5)
                #data = butter_highpass_filter(data, 1000, sampling_rate, order=5)
            #    # Convolution with transferfunction
        #        frf = design_FRF(Fs=144000)
        #        frf.pz2tf((375,12000),(0.1,0.03),144000) # hydrophone without cap
        #        data = signal.lfilter(frf.num, frf.dnum, data)
        #        template_array = signal.lfilter(frf.num, frf.dnum, template_array)
            #
            #    
                # take care of file length
                #template_array, data_array = padme(template_array, data_array)
            
                # convert
                fs = sampling_rate
                dt = 1/fs
                data = types.TimeSeries(data, dt)
                #template = types.TimeSeries(template_array, dt)
                template.resize(len(data))
                
#                pp.plot(data.sample_times, data)
#                pp.show()
                # high pass filter for the data
                #filter_cut = 100
                #time_series_filtered = butter_highpass_filter(data, filter_cut, fs ,order = 5) 
            
            
            
               
            
                p_estimated = data.psd(Nperseg*1/fs, avg_method='mean',  window='hann') 
                p = pycbc.psd.interpolate(p_estimated, data.delta_f)
        
            
        
                flow = 1000
                fhigh = 35000
                snr = pycbc.filter.matched_filter(template, data,
                                                  psd = p,
                                                  low_frequency_cutoff=flow,
                                                      high_frequency_cutoff=fhigh)
                snr_crop = snr.crop(0.35, 0.35)
                
                data = data.crop(0.1,0.1)
                pk, pidx = snr_crop.abs_max_loc()
                pk = snr[51137]
                peak_t = snr.sample_times[pidx]
                
                delta_t = data.sample_times[pidx] - (peak_t)-0.1
            
                # 0.006 s is starting time of template
                L_delta_t.append(delta_t)
                #L_max_snr_time.append(peak_t)    
                L_max_snr.append(pk)          
            
#                pp.plot(data.sample_times,data, label = 'signal resampled')            
#                pp.xlabel('Time (s)')
#                pp.ylabel('Amplitude (mPa)')
#                pp.title('r = 300 m, z = 6 m, E = 1e'+str(energy))
#                pp.legend()
#                pp.show()
            aFile=open('../Neutrino_data_files/deltat_SNR_'+str(rpos)+'_'+str(round(float(zpos),1))+'_'+str(energy)+'_ss6_test.dat', 'wb')
            aFile.write(pickle.dumps({"delta_t":L_delta_t,"snr":L_max_snr}))
            aFile.close()
      
        
if __name__ == "__main__":
    sys.exit(main(sys.argv))
    
