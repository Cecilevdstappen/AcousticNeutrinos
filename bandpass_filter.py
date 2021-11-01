import wave, struct
import numpy as np
from scipy.signal import butter, lfilter
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
    #https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    #convert a and b to w and h
    w, h = freqz(b,a,worN=)
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
        wav_file.writeframes(struct.pack('f',d ))
        #wav_file.writeframes(struct.pack('h',d))
    wav_file.close()



def main(argv):

   # input parser
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.set_defaults(order=6)
    parser.set_defaults(low_cut=1e3)
    parser.set_defaults(high_cut=10e3)
    parser.set_defaults(Fs=144000)

    parser.add_option("-s", "--sampling", type="float", dest="Fs",
                      help="sampling frequency, default = 144000")
    parser.add_option("-o", "--order", type="int", dest="order",
                      help="Order of Butterwordth, default = 6")
    parser.add_option("-f", "--low", type="float", dest="low_cut",
                      help="Lower cut off  [Hz], default = 1e3")
    parser.add_option("-F", "--high", type="float", dest="high_cut",
                      help="High cut off, default = 10e3")

    (options, args) = parser.parse_args()


    # Filter requirements.
    Fs = options.Fs
    order = options.order
    cutoff_low = options.low_cut  # desired cutoff frequency of the filter, Hz
    cutoff_high = options.high_cut   # desired cutoff frequency of the filter, Hz
    print (cutoff_low)
    print (cutoff_high)
    
    filename = sys.argv[1]
    waveFile = wave.open(filename, 'r')
    length = waveFile.getnframes()
    time_series_filtered = np.array([])

    interval = 1000
    Nintervals  = int(length/interval)
    print (Nintervals)
    for i in range(0, Nintervals):
        waveFile.setpos(int(i*interval))
        if (i%10 == 0): print (i)
        time_series = []
        for j in range(0, interval):
            waveData = waveFile.readframes(1)
            sample_point = struct.unpack("<h", waveData)
            time_series.append(sample_point[0])
        aid = butter_bandpass_filter(time_series, \
                                     cutoff_low, \
                                     cutoff_high, \
                                     Fs, order)
        time_series_filtered = np.concatenate((time_series_filtered, np.asarray(aid)), axis=None)
        #print (len(time_series_filtered), len(aid))
    #print(aid)

    #print (time_series_filtered[-1000:])
    # Filter the data, and plot both the original and filtered signals.

    path = sys.argv[-1]
    basename = os.path.splitext(os.path.basename(path))[0]
    writefile(basename+'_filtered.wav', waveFile.getparams(), \
              time_series_filtered)
    
   
    plt.plot(time_series, aid, label='Filtered signal')
    plt.savefig("plot_in_bandpass.png")
    plt.xlabel('time (seconds)')
    #plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main(sys.argv)


