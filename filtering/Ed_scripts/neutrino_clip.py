from math import *
import numpy as np
import numpy.fft as fft
import sys
from scipy import signal

#scaling = pow(10, 168./20)
sampling_rate = 144000.

def write_out(bipfilename, scaling, bmax, bmin):
    print "Generated file: ", bipfilename, scaling, bmax, bmin

def rename_me(filename, scaling):
    return filename.rsplit('.', 1)[0] + \
        '_' + str(scaling)+ '.wav'


def resampled_signal(filename):
    time, bip = np.loadtxt(filename, usecols=(0,1), unpack=True)

    Fs = 1/(time[1] - time [0])
    Fs_resampled = 144000.

    number_resampled = int(round(len(bip)*Fs_resampled/Fs))

    bip_resampled = signal.resample(bip, number_resampled)
    t_resampled   = np.linspace(0, len(bip_resampled)/Fs_resampled,
                                len(bip_resampled))
    return t_resampled, bip_resampled

def waveclip(bipfilename, scaling):
    np.random.seed(scaling)
    from scipy.io.wavfile import read
    noisefile = "1678020614.180210165811.wav"
    time_trace = read(noisefile)
    sampling_rate = time_trace[0]
    time_series   = time_trace[1]

    # part of the read file of particular length
    trace_length = pow(2,20)
    trace_start = np.random.randint(0,len(time_series) - 100000, 1)[0]
    trace_end = trace_start + trace_length

    # determine a noise realisation,
    # based on the FFT of the data that have been read
    # using different, random phase.
    # Used polar coordinates for complex numbers
    y = time_series[int(trace_start) : int(trace_end)]
    Y = np.fft.fft(y)
    m = np.abs(Y)
    phase = 2*pi*np.random.rand(len(m)) - pi
    Z = m*np.cos(phase) + 1j*m*np.sin(phase)
    z = np.fft.ifft(Z)
    z = z.real
        
    # read ascii file with the neutrino click
    time, bip  = resampled_signal(bipfilename)

    # padding to insert the neutrino click somewhere
    # (random position) in the data stream
    entry_point = np.random.randint(0,len(z) - 20000, 1)[0]
    bip *= scaling
    x = np.pad(bip, (entry_point, len(z)-len(bip)-entry_point),
               'constant', constant_values=(0., 0.))

    # add noise abd signal and make dure that it of proper format
    neutrino_noise_array = (x+z) + np.mean(y) * np.ones(len(z))
    neutrino_noise_array = np.asarray(neutrino_noise_array, dtype=np.int16)

#    np.savetxt(rename_me(bipfilename, scaling).rsplit('.', 1)[0] + '.txt', \
#               neutrino_noise_array)
    write_out(bipfilename, scaling, bip.max(), bip.min())
    # write to wave file
    from scipy.io.wavfile import write
    write(rename_me(bipfilename, scaling),
          sampling_rate,
          neutrino_noise_array)

import os
def main(argv):
    for zpos in range(-10, 10, 2):
        for ypos in range(300, 1501, 100):
            filename = 'neutrino_' + str(zpos) + '_' + str(ypos) + '.dat'
            commandlinestring = 'octave -q one_pulse.m ' + str(zpos) + \
                                ' ' + str(ypos) + ' neutrino'
            for scaling in range(100, 1000, 100):
                os.system(commandlinestring)
                waveclip(filename, scaling)    

if __name__ == '__main__':
    main(sys.argv)
