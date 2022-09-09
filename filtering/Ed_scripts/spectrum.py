import matplotlib.pyplot as pp
import matplotlib
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
from numpy.linalg import norm as Norm
#matplotlib.use("pgf")

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
scaling= 1

plot_fhigh = False
plot_SNR_fd = False
plot_SNR_scale = False
plot_spectrum_data = False
transfer_function = False

ax = pp.gca()

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
    
def butter_highpass(cutoff, Fs, order=5):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    b, a = sg.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    

def butter_highpass_filter(data, cutoff, Fs, order=5):
    b, a = butter_highpass(cutoff, Fs, order=order)
    y = sg.lfilter(b, a, data)
    return y

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
    
for zpos in ([6,9,10,11,15]):
    L_rpos = []
    for rpos in ([300]):
        L_rpos.append(rpos)
    #    if zpos <= 8:
        data_file = '../Neutrino_data_files/neutrino' + '_' + str(float(zpos))+'_'+str(rpos)+'_1_'+str(energy)+'_'+str(scaling)+'.txt'
        #data_file = 'output.wav'
        template_file = '../Neutrino_data_files/neutrino_'+str(float(zpos)) +'_'+str(rpos)+'_1_'+str(energy)+'.dat'
        
        #data_file = '../Neutrino_data_files/neutrino_6_300_'+str(scaling)+'whitenoise.txt'
     #   else:
      #      data_file = '../Neutrino_data_files/neutrino_'+ str(zpos) +'_300_1_11_1.txt'
    
        #data_file = '../Neutrino_data_files/neutrino_' +'6'+'_300_1fixed_highpass.txt'
        #data_file = 'neutrino_10_300_1_11_100fixed_unscrambled.txt'
        data = np.loadtxt(data_file,  usecols=(0), dtype='float', unpack=True)  
        noise_signal_td = data #np.resize(data, 131072)
        print(len(noise_signal_td))
        freq = 144000.
        noise_signal_times = np.arange(0, len(noise_signal_td), 1)
        noise_signal_times = noise_signal_times/144000.
        noise_signal_td = noise_signal_td - np.mean(noise_signal_td)
        #noise_signal_td = butter_highpass_filter(noise_signal_td, 1000, noise_signal_times, order=5)
    
    
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
        
        
    
        
    #    f, Pxx_den = sg.periodogram(noise_whale, fs=144000., nfft=512)
    #    Pxx_den = resample_by_interpolation(Pxx_den,257,65537)
    #    psd_period = types.FrequencySeries(Pxx_den, noise_fd.delta_f)
        
        #psd_scipy = psd.interpolate(p_scipy, noise_fd.delta_f)
        #psd_scipy = psd.inverse_spectrum_truncation(psd_scipy, int(dt*512*noise_td.sample_rate),low_frequency_cutoff=flow)
    
        
    
    #########################################################################
    ### template is a theoretical prediction
    ## Q1 amplitude schalen?
    ## Q2 lengte tijdsas en offset?
    
       # else:
        #    template_file = '../Neutrino_data_files/neutrino_'+ str(zpos) +'_300_1_11.dat'
        data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
        time = data[0]
        amplitude_og = data[1]
        amplitude_og =amplitude_og*1000 #conversion to mPa
        #amplitude_og = amplitude_og[::-1] #time reversed template
        
        if transfer_function == True:
            frf = design_FRF(Fs=144000)
            #frf.highpass = True
            # frf.pz2tf((375,12000,17000),(0.15,0.02,0.07),100000) # hydrophone with cap
            frf.pz2tf((375,12000),(0.1,0.03),144000) # hydrophone without cap
            # frf.pz2tf((12000,375),(0.02,0.1),100000) # hydrophone without cap
            #frf.mbutton.on()
            amplitude_og = sg.lfilter(frf.num,frf.dnum,amplitude_og)
            noise_signal_td = sg.lfilter(frf.num,frf.dnum,noise_signal_td)
            
        
        #amplitude = np.pad(amplitude_og, (len(noise_signal_td)-len(amplitude_og), 0) )
        #amplitude, noise_signal_td = padme(amplitude_og, noise_signal_td)
        print('amplitude_og length = ' + str(len(amplitude_og)))
        dt = 1/144000.
    
        
        template = types.TimeSeries(amplitude_og, dt)
        template_fd = abs(template.to_frequencyseries())
        #noise_signal_td= np.resize(noise_signal_td,len(psd_scipy))
        data_td = types.TimeSeries(noise_signal_td, dt)
        data_fd = data_td.to_frequencyseries()
        template.resize(len(data_td))


        pp.figure(18,figsize=(6,5))
        pp.plot(template_fd.sample_frequencies/1000, template_fd, '-',label = 'z ='+str(zpos)+' m')
        pp.grid('on',which='both',axis='x')
        pp.grid('on',which='both',axis='y')
        pp.xlabel("Frequency (kHz)")
        pp.ylabel('Normalised power')
text =pp.text(2.4,2.1,'E = 1e'+str(energy)+' GeV\nr ='+str(rpos)+' m',horizontalalignment="right", verticalalignment="center", transform = ax.transAxes,fontsize = 13)
pp.xlim(0,35)
pp.legend()
pp.savefig("/home/gebruiker/Pictures/Overleaf/Spectrum_zpos.png",bbox_extra_artists=(text,),bbox_inches = 'tight')
pp.show()
pp.close()


    
