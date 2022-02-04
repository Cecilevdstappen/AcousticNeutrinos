import matplotlib.pyplot as pp
from pycbc.filter import matched_filter_core, get_cutoff_indices #, make_frequency_series
from pycbc import types
from pycbc.types.array import complex_same_precision_as
#from pycbc.strain.lines import complex_median
#from pycbc.events.coinc import mean_if_greater_than_zero
from pycbc import fft
import numpy as np
import wave
from scipy.io.wavfile import read


#########################################################################
### data_td, noise plus signal.
# data file is just one column. Sample freq is 144kHz
L_max_snr = []
L_scale = []

zpos = 6
ypos = 300

for scale in np.arange(7000,7001,1):
    data_file = 'neutrino' + '_' + str(zpos) + '_' + str(ypos) + '_' + str(scale) + 'whitenoise.txt'
    data = np.loadtxt(data_file,  usecols=(0), dtype='float', unpack=True)  
    noise_signal_td = data
    print(len(noise_signal_td))

    freq = 144000.
    noise_signal_times = np.arange(0, len(noise_signal_td), 1)
    noise_signal_times = noise_signal_times/144000.
    noise_signal_td = noise_signal_td - np.mean(noise_signal_td)

    pp.plot(noise_signal_times, noise_signal_td)
    pp.show()


#########################################################################
### psd is the noise frequency spectrum
### psd zou een float64 moeten zijn
#    seg_len = int(4 / delta_t)
#    seg_stride = int(seg_len / 2)
#    estimated_psd = pycbc.psd.welch(ts,
#                      seg_len=seg_len,
#                      seg_stride=seg_stride)
    psd = np.ones(len(noise_signal_td))
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
    

    #noise_whale = np.resize(noise_whale, len(noise_signal_td))
    dt = 1/144000.
    noise_td = types.TimeSeries(noise_whale, dt)
    noise_fd = abs(noise_td.to_frequencyseries())
    

#########################################################################
### template is a theoretical prediction
## Q1 amplitude schalen?
## Q2 lengte tijdsas en offset?

    template_file = 'neutrino_6_300.dat'
    data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
    time = data[0]
    amplitude = data[1]
    #pp.plot(time, amplitude)
    #pp.show()
    
    amplitude = np.pad(amplitude, (len(noise_signal_td)-len(amplitude), 0) )
    amplitude = amplitude * 100
    df = 144000./len(psd)
    dt = 1/144000.
    psd = types.FrequencySeries(psd, df)
    template = types.TimeSeries(amplitude, dt)
    template_fd = abs(template.to_frequencyseries())
    data_td = types.TimeSeries(noise_signal_td, dt)
    data_fd = abs(data_td.to_frequencyseries())
    flow = 0.; fhigh= 30000.
    max_snr = 0.
    max_snr_t = 0.
    L_fhigh = []
    L_pk = []

    #for flow in np.arange(0,20000,10):
    for fhigh in np.arange(8000,30000,100):
        snr,_snr, norm = matched_filter_core(template, data_td, low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
        #snr_fd = abs(abs(snr).to_frequencyseries())
        snr = snr*norm
        _snr = _snr*norm
    #    snr_fd = types.Array(np.zeros([len(snr)], dtype=complex_same_precision_as(snr)))
    #    df_snr = 144000./len(snr_fd)
    #    fft.fft(snr, snr_fd)
    #    snr_fd = types.FrequencySeries(snr_fd, df_snr)
        
    
        pk, pidx = snr.abs_max_loc()
        peak_t = snr.sample_times[pidx]
        L_fhigh.append(fhigh)
        L_pk.append(pk)
        if (pk > max_snr):
            max_snr_time = peak_t
            max_snr = pk
        #pp.plot(_snr.sample_frequencies,abs(_snr))
#        pp.title("f_low =" + str(flow) + "f_high=" + str(fhigh) + "scale =" + str(scale))
#        pp.xlabel("frequency")
#        pp.ylabel("SNR")
#pp.savefig("SNF_fd.png")
#    pp.show()
#    pp.close()
    print("Maximum SNR = {:5.2f} at t = {:.4f} s for scaling {:.4f}".format(max_snr, max_snr_time, scale))
    L_max_snr.append(max_snr)
    L_scale.append(scale)
    pp.plot(L_fhigh, L_pk, label = str(scale))
    pp.title("Value SNR peak as function of max frequency")
    pp.xlabel("max frequency")
    pp.ylabel("max SNR")
    pp.legend()
pp.savefig("SNRpeak_fhigh_plot_whalenoise_7000"+str(flow)+".png")
pp.show()
pp.close()
    
pp.plot(_snr.sample_frequencies,abs(_snr))
pp.title("SNR frequency domain")
pp.show()
pp.close()

pp.plot(template_fd.sample_frequencies, template_fd)
pp.title("signal spectrum")
pp.show()
pp.close()
#pp.plot(L_scale,L_max_snr)
#pp.title("Max value SNR as function of signal scale")
#pp.title("f_low =" + str(flow) + "f_high=" + str(fhigh))
#pp.savefig("SNR_scale_plot.png")
#pp.show()
#
pp.rcParams['agg.path.chunksize'] = 10000
pp.plot(data_fd.sample_frequencies,data_fd, label="signal with whale noise")
pp.plot(template_fd.sample_frequencies,template_fd, label="signal")
pp.plot(noise_fd.sample_frequencies,noise_fd,"r--", label = "whale noise")
pp.title("Spectrum data")
#pp.yscale("log")
#pp.xscale("log")
pp.legend()
pp.savefig("spectrum_sig_whalenoise.png")
pp.show()