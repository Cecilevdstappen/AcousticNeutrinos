
import matplotlib.pyplot as pp
    
import pycbc.filter
import pycbc.psd
from pycbc import types
from pycbc.types.array import complex_same_precision_as
from pycbc import fft
import sys
sys.path.insert(0, '/home/gebruiker/AcousticNeutrinos/filtering/design_FRF')
sys.path.insert(0, '/home/gebruiker/SIPPY')
#from numpy.fft import fft as npfft
import numpy as np
from numpy.linalg import norm as Norm
import wave
from scipy import signal
from scipy.io.wavfile import read
from scipy.signal import butter, lfilter
#from impulse import *
import math
from design_FRF import *

from my_styles import *

set_paper_style()

plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')
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

#    return time, bip

    Fs = 1/(time[1] - time [0])
    Fs_resampled = 144000.

    number_resampled = int(round(len(bip)*Fs_resampled/Fs)) ## THIS MAY STRANGE RESULTS, do not rely on it..
    bip_resampled = signal.resample(bip, number_resampled)
    t_resampled   = np.linspace(0, (len(bip_resampled)-1)/Fs_resampled,
                                len(bip_resampled))
   
    return t_resampled, bip_resampled
    
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
    energy = 10
    rpos = 300
    zpos = 6
    rpos_template = 300
    template_file = '../Neutrino_data_files/neutrino_'+str(round(float(zpos),1))+'_'+str(rpos_template)+'_1_'+str(energy)+'.dat' #argv[1]
    template_array = datafile(template_file)*1000 #to mPa
    print(max(template_array))

    data_file     = argv[2]
    time_trace    = read(data_file)
    sampling_rate = time_trace[0]
    data_array    = time_trace[1]
    data_array = butter_highpass_filter(data_array, 1000, sampling_rate, order=5)

    # read ascii file with the neutrino click
    bipfilename = '../Neutrino_data_files/neutrino_'+str(round(float(zpos),1))+'_'+str(rpos)+'_1_'+str(energy)+'.dat'
    time, bip  = resampled_signal(bipfilename)

    # padding to insert the neutrino click somewhere
    # (random position) in the data stream
    #entry_point = np.random.randint(0,len(data_array) - 20000, 1)[0]
    entry_point = int(len(data_array)/5)

    scaling = 1000 #to mPa
    bip *= scaling
    x = np.pad(bip, (entry_point, len(data_array)-len(bip)-entry_point),
               'constant', constant_values=(0., 0.))
    # add noise and signal and make sure that it of proper format

    data = data_array + x 

##    # Convolution with transferfunction
#    frf = design_FRF(Fs=144000)
#    frf.pz2tf((375,12000),(0.1,0.03),144000) # hydrophone without cap
#    data = signal.lfilter(frf.num, frf.dnum, data)
#    template_array = signal.lfilter(frf.num, frf.dnum, template_array)
##
#    
    # take care of file length
    #template_array, data_array = padme(template_array, data_array)

    # convert
    fs = sampling_rate
    dt = 1/fs
    data = types.TimeSeries(data, dt)
    template = types.TimeSeries(template_array, dt)
    template.resize(len(data))

    
    print('neutrino at ', entry_point, entry_point*dt)

    # high pass filter for the data
    filter_cut = 100
    time_series_filtered = butter_highpass_filter(data, filter_cut, fs ,order = 5) 

    ### scalings factors
    scaling_factor_min = 1 #e3mili Pa or micro Pa
    scaling_factor_max = 2e1#e4
    psd3 = spectrum(Nperseg, fs, data*scaling_factor_min)
    psd  = spectrum(Nperseg, fs, data*scaling_factor_max)
    
    if 'plot' in argv:
        # plot time trace
        ax2 = pp.gca()
        ax2.plot(data.sample_times, time_series_filtered*scaling_factor_min,
                 label='data time trace')
        ax2.set_ylabel('pressure ($\mu$Pa)')
        ax2.set_xlabel('Time (s)')
        pp.show()
#        pp.close()


    # Knudsen curves: unit is db Re 1muPa^2/Hz
    Knudsen_noise_0 = np.array([Knudsen_curve(0,i) for i in psd.sample_frequencies])
    Knudsen_noise_6 = np.array([Knudsen_curve(6,i) for i in psd.sample_frequencies])

    #
    #if 'plot' in argv:  
    ax = pp.gca()      
    ax.semilogx(psd.sample_frequencies, psd, label='psd')
    ax.semilogx(psd3.sample_frequencies, psd3, label='psd')
    
    ax.semilogx(psd.sample_frequencies, Knudsen_noise_0, label='sea state 0')
    ax.semilogx(psd.sample_frequencies, Knudsen_noise_6, label='sea state 6')
    
    pp.xlim(1000,100000)
    ax.legend()
    ax.grid()
    ax.set_ylabel('PSD (dB Re 1$\mu$Pa$^2$/Hz)')
    ax.set_xlabel('Frequency (Hz)')



    p_estimated = data.psd(Nperseg*1/fs, avg_method='mean',  window='hann') 
    p = pycbc.psd.interpolate(p_estimated, data.delta_f)
    #dolphin = data.crop(8.518,11.470)
    data_fd = data.to_frequencyseries()
    #    p = psd.inverse_spectrum_truncation(p,
    #                                        int(dt*seg_len*noise_td.sample_rate),
    #                                        low_frequency_cutoff=flow)
    #
    #    psd_inter = p

#    fig, (ax0, ax1, ax2) = pp.subplots(3, 1)
    fig, (ax0, ax1) = pp.subplots(2, 1)
    ax = pp.gca()
    print(data.sample_times[2129590])
    dolphin = data[2129420:2129510]
    dolphin_fd = abs(dolphin.to_frequencyseries())
    #dolphin_fd = types.FrequencySeries(dolphin_fd, data_fd.delta_f) #
    if 'match' in argv:
        flow = 1000
        fhigh = 35000
        snr = pycbc.filter.matched_filter(template, data,
                                          psd = p,
                                          low_frequency_cutoff=flow,
                                          high_frequency_cutoff=fhigh)

        fig, axs = pp.subplots(2,1, figsize = (6,5),sharex=True)
        axs = axs.ravel()
        axs[0].plot(data.sample_times, data, label='Data, time signal t = 4.00 s')
        axs[1].plot(snr.sample_times, abs(snr), label='SNR')
        axs[0].set_ylabel('Amplitude (mPa)')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('SNR')
        axs[0].legend(loc='upper left')
        fig.suptitle('E = 1e'+str(energy)+' GeV, r = '+str(rpos)+' m, z = '+ str(zpos)+' m')

#        ax2.plot(template.sample_times, template, label='snr')
    pp.show()
#    pp.close()

    #pp.plot(template.sample_times, template)
    pp.figure(13, figsize=(6,5))
    pp.plot(dolphin.sample_times, dolphin )#label ='z =' + str(zpos))
    pp.grid('on',which='both',axis='x')
    pp.grid('on',which='both',axis='y')
    #pp.title('E = 1e'+str(energy)+' GeV, r ='+str(rpos)+' m, z ='+ str(zpos)+' m')
    pp.xlabel('Time (s)')
    pp.ylabel('Amplitude (mPa)')
    #pp.xlim(8.502,8.5026)
    pp.legend()
    #pp.savefig("/home/gebruiker/Pictures/Overleaf/Signal_z"+str(zpos)+'_r'+str(rpos)+'_E'+str(energy)+'scaling'+str(scaling))
    pp.show()
    
    pp.figure(13, figsize=(6,5))
    pp.plot(dolphin_fd.sample_frequencies/1000, dolphin_fd)#label ='z =' + str(zpos))
    pp.grid('on',which='both',axis='x')
    pp.grid('on',which='both',axis='y')
    #pp.title('E = 1e'+str(energy)+' GeV, r ='+str(rpos)+' m, z ='+ str(zpos)+' m')
    pp.xlabel('Frequency (kHz)')
    pp.ylabel('Normalised power')
    pp.xlim()
    pp.legend()
    #pp.savefig("/home/gebruiker/Pictures/Overleaf/Signal_z"+str(zpos)+'_r'+str(rpos)+'_E'+str(energy)+'scaling'+str(scaling))
    pp.show()
    
if __name__ == "__main__":
    sys.exit(main(sys.argv))

