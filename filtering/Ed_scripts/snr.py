
import matplotlib.pyplot as pp
    
import pycbc.filter
import pycbc.psd
from pycbc import types
from pycbc.types.array import complex_same_precision_as
from pycbc import fft
import sys

import numpy as np
import wave
from scipy import signal
from scipy.io.wavfile import read
from impulse import *

def get_PSD_data(datafile):
    from scipy import signal
    data = np.loadtxt(datafile,  usecols=(0), dtype='float', unpack=True)  
    dt = 1/144000.
    data_td = types.TimeSeries(data, dt)

    f, Pxx_den = signal.welch(data, 144000, nperseg=131072)
    psd = types.FrequencySeries(Pxx_den, f[1]-f[0])
    
    return f,psd#data_td.to_frequencyseries()
    
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

def main():
    # Need to get a SNR array using matched filtering
    # Matched filtering: template, data and a noise PSD

    template_file = 'neutrino_6_300.dat'
    data_file = 'neutrino_6_300_71.txt'    
    template_array = datafile(template_file)
    data_array = datafile(data_file) 

    # take care of file length
#    template_array, data_array = padme(template_array, data_array)

    # convert
    dt = 1/144000.
    data = types.TimeSeries(data_array, dt)
    template = types.TimeSeries(template_array, dt)
    data_f = data.to_frequencyseries()

    template.resize(len(data))
   
    f_psd, psd = get_PSD_data(data_file)
    print(f_psd[:5])
    print(data_f.sample_frequencies[:5])
    print( data.delta_f, template.delta_f, psd.delta_f)
    psd.resize(len(data))
    
    flow = 100
    fhigh = 25000
    snr = pycbc.filter.matched_filter(template, data, psd = psd, low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)
    
    print(snr)

    ax = pp.gca()
    
    ax.loglog(psd.sample_frequencies, psd, label='psd')
#    ax.plot(snr.sample_times, abs(snr), label='snr')
#    ax.plot(template.sample_times, template, label = 'template')
#    ax.plot(data.sample_times, data, label = 'data')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal-to-noise (SNR)')
    pp.show()
    pp.close()

if __name__ == "__main__":
    sys.exit(main())

