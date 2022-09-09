#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:06:29 2022

@author: gebruiker
"""

import matplotlib.pyplot as pp
    
import pycbc.filter
import pycbc.psd
from pycbc import types
from pycbc.types.array import complex_same_precision_as
from pycbc import fft
import sys

#sys.path.insert(0, '/project/antares/cstappen/files/polys/scripts_with_carlo//SIPPY')
#sys.path.insert(0, '/home/gebruiker/AcousticNeutrinos/filtering/design_FRF')
#sys.path.insert(0, '/home/gebruiker/SIPPY')
#from numpy.fft import fft as npfft
import numpy as np
from numpy.linalg import norm as Norm
import wave
from scipy import signal
from scipy.io.wavfile import read
from scipy.signal import butter, lfilter
from impulse import *

#from design_FRF import *
#from impulse import *
import math
import pickle
#from design_FRF import *
import argparse
from my_styles import *

set_paper_style()
plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')
#pp.style.use('/project/antares/cstappen/files/polys/scripts_with_carlo/cecile.mplstyle')
Nperseg = int(2048)

def parse_args():
    parser = argparse.ArgumentParser(
    description='Submit single run',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--r',type=int,help="ID of the run to process; usually the run name.")
    parser.add_argument('--energy',type=int,help="ID of the run to process; usually the run name.")
    parser.add_argument('--zpos',type=float,help="Target final data type to produce.")
    parser.add_argument('--iterate',type=int,help="Target final data type to produce.")
    parser.add_argument('--dirpath',type=str,help="Target final data type to produce.")
    return parser.parse_args()

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


rpos = 300
zpos = 6
energy=11
random_nr = np.random.rand(1000)
L_mean_SNR = []
L_energy = []
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
noise_scaling = 1
y = data_array[int(trace_start) : int(trace_end)]
Y = np.fft.fft(y)
m = np.abs(Y)
phase = 2*np.pi*random_nr[1]*(len(m)) - np.pi
Z = m*np.cos(phase) + 1j*m*np.sin(phase)
z = np.fft.ifft(Z)
z = z.real
z*= noise_scaling
fs = sampling_rate
dt = 1/fs
#            frf = design_FRF(Fs=144000)
#            frf.pz2tf((375,12000),(0.1,0.03),144000) # hydrophone without cap
#            z = signal.lfilter(frf.num, frf.dnum, z)
#            template = signal.lfilter(frf.num, frf.dnum, template)
data= types.TimeSeries(z, dt)
#template = types.TimeSeries(template,dt)
template.resize(len(data))

flow = 1000.; fhigh= 35000.

i = 2048
p_estimated = data.psd(dt*i, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
psd_inter = pycbc.psd.interpolate(p_estimated, data.delta_f)

snr = pycbc.filter.matched_filter(template, data, psd = psd_inter,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
snr = snr.crop(0.1, 0.1)
snr_noise = abs(snr)
for energy in np.arange(9,12,1):  
    plot_hist_dt = True
    plot_hist_SNR = True
    plot_2Dhist = False
    scatter = False
    with open('/home/gebruiker/AcousticNeutrinos/filtering/Neutrino_data_files/deltat_SNR_'+str(rpos)+'_'+str(round(float(zpos),1))+'_'+str(energy)+'.dat', 'rb') as handle:
        b = pickle.load(handle)
    delta_t = b["delta_t"]
    delta_t = np.abs(delta_t)
    if energy >10:
        delta_t = np.abs(delta_t)
    SNR = b["snr"]
    SNR = np.array(SNR)

    print(delta_t[SNR>10])
    print (energy)
    L_energy.append(energy)
#

    #counts, bin_edges = np.histogram(np.abs(delta_t), bins=100)
#    print(counts)
#    print(bin_edges)

    #pp.hist2d(delta_t,SNR, bins = 50)
    #pp.scatter(delta_t,SNR)# range = (-0.8,-0.7))
    #pp.color(delta_t,SNR)
    fig, ax = pp.subplots()
#    if plot_hist_dt == True:
#    pp.figure(1, figsize=(6,5))
#    pp.hist(np.abs(delta_t), bins = 100)
#    pp.title('r = '+ str(rpos)+' m, z = '+str(zpos)+' m, sea state 0') #E = [1e9,1e13] GeV,
#    pp.grid('on',which='both',axis='x')
#    pp.grid('on',which='both',axis='y')
#    pp.xlabel(r'$\Delta$t (s)')
#    pp.ylabel('count')
#    #pp.yscale('log')
#    pp.text(0.8, 0.8, 'E = 1e'+str(energy),ha='right', va='top',transform=ax.transAxes,fontsize = 13)
#    pp.legend()
#    pp.savefig('/home/gebruiker/Pictures/Overleaf/hist_dt_E'+str(energy)+'_1e3.png')
#    pp.show()
#
#    if plot_hist_SNR == True:
    pp.figure(2, figsize=(6,5))
    pp.hist(SNR, bins = 100, linestyle = 'solid',histtype='stepfilled',label = 'E = 1e'+str(energy),zorder=1/energy)
    if energy == 9:
        pp.hist(snr_noise, bins = 100, linestyle = 'solid', histtype='stepfilled',label = 'Noise')
    #pp.axvline(x=4, color='black', linestyle='--')
    pp.title('r = '+ str(rpos)+' m, z = '+str(zpos)+' m, sea state 0')
    pp.grid('on',which='both',axis='x')
    pp.grid('on',which='both',axis='y')
    pp.xlabel('SNR (dB)')
    pp.ylabel('count')
    pp.yscale('log')
    pp.xscale('log')
    pp.ylim(0,15000)
    pp.xlim(0.04,230)
        #pp.text(0.8, 0.8, 'E = 1e'+str(energy),ha='right', va='top',transform=ax.transAxes,fontsize = 13)
pp.legend(ncol = 2,fontsize = 8)
pp.savefig('/home/gebruiker/Pictures/Overleaf/hist_SNR_E1012.png')#+str(energy)+'.png')
pp.show()
pp.close()


#    if scatter == True:
#        pp.figure(3, figsize=(6,5))
#        pp.scatter(delta_t,SNR, s=10)# range = (-0.8,-0.7))
#        pp.title('r = '+ str(rpos)+' m, z = '+str(zpos)+', sea state 0')
#        pp.grid('on',which='both',axis='x')
#        pp.grid('on',which='both',axis='y')
#        pp.xlabel(r'$\Delta$t (s)')
#        pp.ylabel('SNR max value (20log$_{10}$ (A$_{signal}$/A$_{noise}$))')
#        pp.yscale('log')
#        pp.text(0.8, 0.8, 'E = 1e'+str(energy),ha='right', va='top',transform=ax.transAxes,fontsize = 13)
#        pp.savefig('/home/gebruiker/Pictures/Overleaf/scatter_SNR_dt'+str(energy)+'.png')
#        pp.show()
#    
#    if plot_2Dhist == True:
#        pp.figure(4, figsize=(6,5))
#        pp.hist2d(delta_t,SNR, bins = 500, norm = matplotlib.colors.LogNorm())
#        pp.title('r = '+ str(rpos)+' m, z = '+str(zpos)+', sea state 0') #E = [1e9,1e13] GeV,
#        pp.grid('on',which='both',axis='x')
#        pp.grid('on',which='both',axis='y')
#        pp.xlabel(r'$\Delta$t (s)')
#        pp.text(0.8, 0.8, 'E = 1e'+str(energy),ha='right', va='top',transform=ax.transAxes,fontsize = 13)
#        pp.ylabel('SNR max value (20log$_{10}$ (A$_{signal}$/A$_{noise}$))')
#        #pp.yscale('log')
#        pp.savefig('/home/gebruiker/Pictures/Overleaf/2dhist'+str(energy)+'.png')
#        pp.show()
#    
