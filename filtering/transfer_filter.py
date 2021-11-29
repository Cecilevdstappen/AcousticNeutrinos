import wave, struct
import numpy as np
from scipy.signal import butter, lfilter
from scipy.fft import fft
from scipy.fftpack import fftfreq
import matplotlib.pyplot as plt
import sys
from optparse import OptionParser
import os

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    print (low, high)
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

#def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
#    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = lfilter(b, a, data)
#    return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    y = butter_lowpass_filter(data, highcut, fs, order)
    return butter_highpass_filter(y, lowcut, fs, order)

def writefile(fname, fileparams, data):
    wav_file = wave.open(fname, "w")
    wav_file.setparams(fileparams)
    for d in data:
        wav_file.writeframes(struct.pack('f',d))
        #wav_file.writeframes(struct.pack('h',d))
    wav_file.close()


def transfer_function_filter(data, a, b):
    return lfilter(b, a, data)

def transfer_function():
    import scipy.io
    # coefficients a and b are determined elsewhere
    mat = scipy.io.loadmat('coeff_armax_A.mat')
    a =  mat['G_den'][0]
    
    mat = scipy.io.loadmat('coeff_armax_B.mat')
    b = mat['G_num'][0]
    return a, b

def main(argv):

   # input parser
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.set_defaults(order=6)
    parser.set_defaults(low_cut=1e3)
    parser.set_defaults(high_cut=70e3)
    parser.set_defaults(Fs=144000)

    parser.add_option("-s", "--sampling", type="float", dest="Fs",
                      help="sampling frequency, default = 144000")
    parser.add_option("-o", "--order", type="int", dest="order",
                      help="Order of Butterwordth, default = 6")
    parser.add_option("-f", "--low", type="float", dest="low_cut",
                      help="Lower cut off  [Hz], default = 1e3")
    parser.add_option("-F", "--high", type="float", dest="high_cut",
                      help="High cut off, default = 50e3")

    (options, args) = parser.parse_args()


    # Filter requirements.
    Fs = options.Fs
    order = options.order
    cutoff_low = options.low_cut  # desired cutoff frequency of the filter, Hz
    cutoff_high = options.high_cut   # desired cutoff frequency of the filter, Hz
    
    #Read in dat file
    file_name = 'pulses.dat'
    waveFile = open(file_name,'r')
    lines = waveFile.readlines()
    time = [] #s
    y = []
    u = []

    for line in lines:
        if line.startswith("#") or len(line) <= 1: continue
        else: time.append(float(line.split(' ')[0])), y.append(float(line.split(' ')[1])),u.append(float(line.split(' ')[2]))
    
    length = len(y)
    #Read in wave file
    #filename = sys.argv[1]
    #waveFile = wave.open(filename, 'r')
    #length = waveFile.getnframes()
    #time_series_filtered = np.array([])

   
    a,b = transfer_function()
    time_series_filtered = transfer_function_filter(u,a,b)
    
#   
#    interval = 100
#   Nintervals  = int(length/interval)
#   a,b = transfer_function()
#    for i in range(0, Nintervals):
#        waveFile.setpos(int(i*interval))
#       if (i%10 == 0): print (i)   
#        time_series = []
#        for j in range(0, interval):
#            waveData = waveFile.readframes(1)
#            sample_point = struct.unpack("<h", waveData)
#            time_series.append(sample_point[0])
#        aid = transfer_function_filter(time_series, a, b)
#        aid = transfer_function_filter(u,a,b)
#        fourier_aid = fft(aid)
#            aid = butter_bandpass_filter(time_series, \
#                                     cutoff_low, \
#                                     cutoff_high, \
#                                     Fs, order)
#        time_series_filtered = np.concatenate((time_series_filtered, np.asarray(aid)), axis=None)
#        print len(time_series_filtered), len(aid)
#    print aid

    print (time_series_filtered[-1000:])
    # Filter the data, and plot both the original and filtered signals.

    path = sys.argv[-1]
    basename = os.path.splitext(os.path.basename(path))[0]
    #writefile(basename+'_filtered_coeff.wav', waveFile.getparams(), \
    #          time_series_filtered)

    plt.figure(1)
    plt.plot(time_series_filtered, label='Filtered signal')
    plt.plot(y,label="y")
    plt.xlabel('time (seconds)')
    #plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.savefig("plot_filtered_transferfunction_coeff_armax.png")
    plt.show()
    plt.close()

    plt.figure(2)
    plt.plot(fourier_aid, label="Transferfunction")
    plt.xlabel('frequency (Hz)')
    #plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    #plt.xlim(10**2, 10**5)
    #plt.axis('tight')
    plt.legend(loc='upper left')
    plt.xscale("log")
    plt.savefig("plot_transferfunction_coeff_armax.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main(sys.argv)

