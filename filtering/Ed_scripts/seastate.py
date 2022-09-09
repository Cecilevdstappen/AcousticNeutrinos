
import matplotlib.pyplot as pp
    
import pycbc.filter
import pycbc.psd
from pycbc import types
from pycbc.types.array import complex_same_precision_as
from pycbc import fft
from pycbc import psd
import sys
import math
#from numpy.fft import fft as npfft
import numpy as np
from numpy.linalg import norm as Norm
import wave
from scipy import signal
from scipy.io.wavfile import read
from scipy.signal import butter, lfilter
from my_styles import *

set_paper_style()

plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')
#from impulse import *
import math

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

    
def datafile(filename):
    # read one column time trace (sample freq 144 kHz)
    data = np.loadtxt(filename,  usecols=(0), dtype='float', unpack=True)  
    dt = 1/144000.
    data_td = types.TimeSeries(data, dt)

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
    
def psd_inter(seg_len,data):
    dt = 1/144000.
    p_estimated = data.psd(dt*seg_len, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
    psd_inter = psd.interpolate(p_estimated, data.delta_f)
    return psd_inter

def main(argv):
    # Need to get a SNR array using matched filtering
    # Matched filtering: template, data and a noise PSD
    zpos = 10
    rpos = 300
    energy = 11
    scaling = 1
    data_file = '../Neutrino_data_files/neutrino' + '_' + str(float(zpos))+'_'+str(rpos)+'_1_'+str(energy)+'_'+str(scaling)+'.txt'
    template_file = '../Neutrino_data_files/neutrino_'+str(float(zpos)) +'_'+str(rpos)+'_1_'+str(energy)+'.dat'   
    #template_array = datafile(template_file)
    data = np.loadtxt(data_file,  usecols=(0), dtype='float', unpack=True)  
    data_array = data
    template = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
    time = template[0]
    template_array = template[1]
    template_array =template_array*1000

    # take care of file length
    #    template_array, data_array = padme(template_array, data_array)

    # convert
    fs = 144000.
    dt = 1/fs
    data = types.TimeSeries(data_array, dt)
    template = types.TimeSeries(template_array, dt)
    template.resize(len(data))

    # high pass filter for the data
    filter_cut = 10000
    time_series_filtered = butter_highpass_filter(data, filter_cut, fs ,order = 5) 

    ### scalings factors
    scaling_factor_min = 1
    scaling_factor_max = 2e1
    #psd_initial = spectrum(Nperseg, fs, data*1000)
    #psd_initial = psd_inter(2048,data)
    psd2 = psd_inter(2048, data*scaling_factor_min*1000) #from mp to micro pascal
    psd2 = 10*np.log10(psd2)
    psd4  = psd_inter(2048, data*scaling_factor_max*1000)
    psd4 = 10*np.log10(psd4)
    psd3 = spectrum(Nperseg, fs, data*scaling_factor_min*1000) #from mp to micro pascal
    psd  = spectrum(Nperseg, fs, data*scaling_factor_max*1000)#from mp to micro pascal

    # plot time trace
    ax2 = pp.gca()
    ax2.plot(data.sample_times, time_series_filtered*scaling_factor_min, label='data time trace')
    ax2.set_ylabel('pressure (mPa)')
    ax2.set_xlabel('Time (s)')
    pp.show()
    pp.close()


    # Knudsen curves: unit is db Re 1muPa^2/Hz
    Knudsen_noise_0 = np.array([Knudsen_curve(0,i) for i in psd.sample_frequencies])
    Knudsen_noise_1 = np.array([Knudsen_curve(1,i) for i in psd.sample_frequencies])
    Knudsen_noise_2 = np.array([Knudsen_curve(2,i) for i in psd.sample_frequencies])
    Knudsen_noise_3 = np.array([Knudsen_curve(3,i) for i in psd.sample_frequencies])
    Knudsen_noise_4 = np.array([Knudsen_curve(4,i) for i in psd.sample_frequencies])
    Knudsen_noise_5 = np.array([Knudsen_curve(5,i) for i in psd.sample_frequencies])
    Knudsen_noise_6 = np.array([Knudsen_curve(6,i) for i in psd.sample_frequencies])

    pp.figure(2, figsize=(6,5))
    ax = pp.gca()
    
    
    #ax.semilogx(psd3.sample_frequencies, psd3,'-', label='Ernst-jan, psd scaling = '+str(scaling_factor_min))
    #ax.semilogx(psd.sample_frequencies, psd,'-', label='Ernst-jan, psd scaling = '+str(scaling_factor_max))
    #ax.semilogx(psd2.sample_frequencies/1000, psd2,'-')#,label = 'psd scaling = ' + str(scaling_factor_min))
    #ax.semilogx(psd2.sample_frequencies/1000, psd4,'-',label='psd scaling = ' + str(scaling_factor_max))

    ax.semilogx(psd.sample_frequencies/1000, Knudsen_noise_0,'-', label='ss 0')
    ax.semilogx(psd.sample_frequencies/1000, Knudsen_noise_1,'-', label='ss 1')
    ax.semilogx(psd.sample_frequencies/1000, Knudsen_noise_2,'-', label='ss 2')
    ax.semilogx(psd.sample_frequencies/1000, Knudsen_noise_3,'-', label='ss 3')
    ax.semilogx(psd.sample_frequencies/1000, Knudsen_noise_4,'-', label='ss 4')
    ax.semilogx(psd.sample_frequencies/1000, Knudsen_noise_5,'-', label='ss 5')
    ax.semilogx(psd.sample_frequencies/1000, Knudsen_noise_6,'-', label='ss 6')
    pp.ylim(0,100)
    #pp.xlim(1,35)
    #pp.text(0.95,0.9,'Segment length = 2048' ,horizontalalignment="right", verticalalignment="center", transform = ax.transAxes, fontsize = 11)
    ax.legend(ncol = 7,fontsize = 8)
    ax.grid()
    ax.set_ylabel('PSD (dB Re 1 $\mu$Pa$^2$/Hz)')
    ax.set_xlabel('Frequency (kHz)')

    if 'match' in argv:
        flow = 500
        fhigh = 30000
        snr = pycbc.filter.matched_filter(template, data,
                                          psd = psd3,
                                          low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)
        print(snr)

        ax.plot(snr.sample_times, abs(snr), label='snr')
    pp.show()
    pp.close()
    
if __name__ == "__main__":
    sys.exit(main(sys.argv))

