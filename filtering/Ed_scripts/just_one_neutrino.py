from math import *
import numpy as np
import numpy.fft as fft
import sys
import matplotlib.pyplot as pp
from scipy import signal

#scaling = pow(10, 168./20)
sampling_rate = 144000.


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def write_out(bipfilename, scaling, bmax, bmin):
    print ("Generated file: ", bipfilename, scaling, bmax, bmin)

def rename_me(filename, scaling):
    return filename.rsplit('.', 1)[0] + \
        '_' + str(scaling)+ '_ss6.wav'


def resampled_signal(filename):
    time, bip = np.loadtxt(filename, usecols=(0,1), unpack=True)

    return time, bip

    Fs = 1/(time[1] - time [0])
    print(Fs)
    Fs_resampled = 144000.

    number_resampled = int(round(len(bip)*Fs_resampled/Fs)) ## THIS MAY STRANGE RESULTS, do not rely on it..
    bip_resampled = signal.resample(bip, number_resampled)
    t_resampled   = np.linspace(0, (len(bip_resampled)-1)/Fs_resampled,
                                len(bip_resampled))
   
    return t_resampled, bip_resampled

def waveclip(bipfilename, scaling):
    np.random.seed(scaling)
    from scipy.io.wavfile import read
    noisefile = "output.wav"
    time_trace = read(noisefile)
    sampling_rate = time_trace[0]
    time_series   = time_trace[1]

    # part of the read file of particular length
    trace_length = pow(2,17)
    trace_start = np.random.randint(0,len(time_series) - 1000, 1)[0]
    trace_end = trace_start + trace_length

    # determine a noise realisation,
    # based on the FFT of the data that have been read
    # using different, random phase.
    # Used polar coordinates for complex numbers
    noise_scaling = 2e1
    y = time_series[int(trace_start) : int(trace_end)]
    Y = np.fft.fft(y)
    m = np.abs(Y)
    phase = 2*pi*np.random.rand(len(m)) - pi
    Z = m*np.cos(phase) + 1j*m*np.sin(phase)
    z = np.fft.ifft(Z)
    z = z.real
    z*= noise_scaling
         
    # read ascii file with the neutrino click
    time, bip  = resampled_signal(bipfilename)

    # padding to insert the neutrino click somewhere
    # (random position) in the data stream
    #entry_point = np.random.randint(0,len(z) - 20000, 1)[0]
    entry_point = int(len(z)/2)
    print("entry point = {:.2f}".format(entry_point))
    bip *= (scaling*1000)
    x = np.pad(bip, (entry_point, len(z)-len(bip)-entry_point),
               'constant', constant_values=(0., 0.))
    # add noise and signal and make dure that it of proper format
    cutoff=35
    neutrino_noise_array = (x+z) + np.mean(y) * np.ones(len(z))
    neutrino_noise_array = np.asarray(neutrino_noise_array)#, dtype=np.int16)
    #neutrino_noise_array = butter_highpass_filter(neutrino_noise_array, cutoff, sampling_rate)

    np.savetxt(rename_me(bipfilename, scaling).rsplit('.', 1)[0] + '.txt', \
               neutrino_noise_array)
    file = write_out(bipfilename, scaling, bip.max(), bip.min())
    # write to wave file
    neutrino_noise_array = np.asarray(neutrino_noise_array, dtype=np.int16)
    from scipy.io.wavfile import write
    write(rename_me(bipfilename, scaling),
          sampling_rate,
          neutrino_noise_array)

import os
def main(argv):
    for energy in np.arange(9,13,1):
        for zpos in ([6,10,15]): 
            for rpos in ([5,100,300,500]):
                scaling = 1
                filename = '../Neutrino_data_files/neutrino_'+str(round(float(zpos),1))+'_'+str(rpos)+'_1_'+str(energy)+'.dat'
                #commandlinestring = 'octave -q one_pulse.m ' + str(float(zpos)) + \
                 #           ' ' + str(rpos) + ' neutrino'
                #os.system(commandlinestring)
                waveclip(filename, scaling)    

if __name__ == '__main__':
    main(sys.argv)
