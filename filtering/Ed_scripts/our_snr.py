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
L_scale = []

zpos = 6
ypos = 300

plot_fhigh = False
plot_SNR_fd = False
plot_Signal_fd = False
plot_SNR_scale = False
plot_spectrum_data = False
transfer_function = False
plot_signal_data=False

for scale in np.arange(71,72,1):
    data_file = 'neutrino' + '_' + str(zpos) + '_' + str(ypos) + '_' + str(scale) + '.txt' #'whitenoise.txt'
    data = np.loadtxt(data_file,  usecols=(0), dtype='float', unpack=True)  
    noise_signal_td = data

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
    
    f, Pxx_den = sg.welch(noise_whale, fs=144000., nperseg=512, average='mean')
    Pxx_den = resample_by_interpolation(Pxx_den,257,65537)
    psd_scipy = types.FrequencySeries(Pxx_den, noise_fd.delta_f)
    
    #psd_scipy = psd.interpolate(p_scipy, noise_fd.delta_f)
    #psd_scipy = psd.inverse_spectrum_truncation(psd_scipy, int(dt*512*noise_td.sample_rate),low_frequency_cutoff=flow)

    

#########################################################################
### template is a theoretical prediction
## Q1 amplitude schalen?
## Q2 lengte tijdsas en offset?

    template_file = 'neutrino_6_300.dat'
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

    
    
    p = noise_td.psd(dt*512, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
    p = psd.interpolate(p, noise_fd.delta_f)
    p = psd.inverse_spectrum_truncation(p, int(dt*512*noise_td.sample_rate),low_frequency_cutoff=flow)
   
    # The noise is often shown in terms of the amplitude spectral density, which
    # is the square root of the power spectral density
    psd_inter = p

    
# psd is the noise frequency spectrum
# psd zou een float64 moeten zijn
    seg_len = 512
    seg_stride = int(seg_len/2)
    estimated_psd = psd.welch(noise_td,seg_len=seg_len,seg_stride=seg_stride, window='hann')
    psd_seg = psd.interpolate(estimated_psd, noise_fd.delta_f)

    pp.plot(psd_scipy.sample_frequencies, psd_scipy, label = "psd scipy")
    pp.plot(psd_seg.sample_frequencies, psd_seg, label="psd seg")
    pp.plot(psd_inter.sample_frequencies, psd_inter, label = "psd interpolate")
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

    #for flow in np.arange(0,20000,10):
    for fhigh in np.arange(8000,30000,100):
        snr_inter = matched_filter(template, data_td, psd = psd_inter,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
        snr_seg = matched_filter(template, data_td, psd = psd_seg,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
        snr_scipy = matched_filter(template, data_td, psd = psd_scipy,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
        snr_1 = matched_filter(template, data_td, psd = psd_1,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
        pk, pidx = snr_inter.abs_max_loc()
        peak_t = snr_inter.sample_times[pidx]
        L_fhigh.append(fhigh)
        L_pk.append(pk)
        if (pk > max_snr):
            max_snr_time = peak_t
            max_snr = pk

    print("Maximum SNR = {:5.2f} at t = {:.4f} s for scaling {:.4f}".format(max_snr, max_snr_time, scale))
    L_max_snr.append(max_snr)
    L_scale.append(scale)
    
    ax = pp.gca()
    #ax.plot(snr_inter.sample_times, abs(snr_inter), label='psd_inter')
    #ax.plot(snr_seg.sample_times, abs(snr_seg), label='psd_seg')
    ax.plot(snr_scipy.sample_times, abs(snr_scipy), label='psd_scipy')
    ax.plot(snr_1.sample_times, abs(snr_1), label='psd=1')    
    ax.legend()
    ax.set_title("matched filtering timeseries")
    ax.grid()
    ax.set_ylim(0, None)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal-to-noise (SNR)')
    pp.show()
    pp.close()
    
    if plot_signal_data==True:
        pp.figure(14)
        pp.clf()
        pp.plot(time, amplitude_og, label ="transfered signal")
        pp.grid('on',which='both',axis='x')
        pp.grid('on',which='both',axis='y')
        pp.title('Impulse response signal')
        pp.xlabel('t -> [s]')
        pp.legend()
        pp.show()
        
        pp.figure(13)
        pp.clf()
        pp.plot(noise_signal_times,noise_signal_td, label ="transfered data")
        pp.grid('on',which='both',axis='x')
        pp.grid('on',which='both',axis='y')
        pp.title('Impulse response data')
        pp.xlabel('t -> [s]')
        pp.legend()
        pp.show()

    if plot_fhigh == True:
        pp.plot(L_fhigh, L_pk, label = str(scale))
        pp.title("Value SNR peak as function of max frequency")
        pp.xlabel("max frequency")
        pp.ylabel("max SNR")
        pp.legend()
        pp.savefig("SNRpeak_fhigh_plot_whalenoise_7000"+str(flow)+".png")
        pp.show()
        pp.close()

if plot_SNR_fd == True:
    pp.plot(_snr.sample_frequencies,abs(_snr))
    pp.title("SNR frequency domain")
    pp.xlabel("frequency")
    pp.ylabel("SNR")
    pp.savefig("SNF_fd.png")
    pp.show()
    pp.close()


if plot_Signal_fd == True:
    pp.plot(template_fd.sample_frequencies, template_fd)
    pp.title("signal spectrum")
    pp.show()
    pp.close()

if plot_SNR_scale == True:
    pp.plot(L_scale,L_max_snr)
    pp.title("Max value SNR as function of signal scale")
    pp.title("f_low =" + str(flow) + "f_high=" + str(fhigh))
    pp.savefig("SNR_scale_plot.png")
    pp.show()

if plot_spectrum_data == True:
    pp.rcParams['agg.path.chunksize'] = 10000
    pp.plot(data_fd.sample_frequencies,data_fd, label="signal with whale noise")
    pp.plot(template_fd.sample_frequencies,template_fd, label="signal")
    pp.plot(noise_fd.sample_frequencies,noise_fd,"r--", label = "whale noise")
    pp.title("Spectrum data")
    pp.yscale("log")
    pp.xscale("log")
    pp.legend()
    pp.savefig("spectrum_sig_whalenoise.png")
    pp.show()
    
