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
from my_styles import *
from numpy.linalg import norm as Norm

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
plot_Signal_fd = True
plot_SNR_scale = False
plot_spectrum_data = False
transfer_function = True
plot_signal_data=True
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
    
for zpos in ([6]):
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
        print(len(amplitude_og))
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
        from scipy.io.wavfile import write    
        sampling_rate = 1e6/10 #1000 #144000 # adjust sample rate
        amplitude_ogs = amplitude_og / np.linalg.norm(amplitude_og) # adjust the amplitude            
        write("example_slow.wav", sampling_rate, amplitude_og.astype(np.int16))
        template_fd = abs(template.to_frequencyseries())
        #noise_signal_td= np.resize(noise_signal_td,len(psd_scipy))
        data_td = types.TimeSeries(noise_signal_td, dt)
        data_fd = data_td.to_frequencyseries()
        template.resize(len(data_td))
        L_psd_inter = []
        L_psd_seg = []
        L_psd_scipy = []
        L_segment = [2048]#[131072,65536,16384,2048,256,128,8]
        for i in L_segment:
            seg_len = i
            p_estimated = data_td.psd(dt*i, avg_method='mean',  window='hann') #data_td.sample_times[-1]/512
            #print(len(p_estimated))
            psd_inter = psd.interpolate(p_estimated, data_td.delta_f)
            #psd_inter = psd.inverse_spectrum_truncation(p, int(dt*seg_len*noise_td.sample_rate),low_frequency_cutoff=flow)
            #psd_inter = p
            L_psd_inter.append(psd_inter)
            print(len(L_psd_inter))
       
            seg_stride = int(seg_len/2)
            estimated_psd = psd.welch(data_td,seg_len=seg_len,seg_stride=seg_stride, window='hann')
            psd_seg = psd.interpolate(estimated_psd, data_td.delta_f)
            psd_seg = psd.inverse_spectrum_truncation(psd_seg, int(dt*seg_len*noise_td.sample_rate),low_frequency_cutoff=flow)
            L_psd_seg.append(psd_seg)
            
            nr_seg = i/2 +1
            f, Pxx_den= sg.welch(noise_whale, fs=144000., nperseg=i, average='mean')
            psd_scipy_estimated = types.FrequencySeries(Pxx_den, f[1]-f[0])
            Pxx_den = resample_by_interpolation(Pxx_den,nr_seg,65537)
            psd_scipy = types.FrequencySeries(Pxx_den, noise_fd.delta_f)
            L_psd_scipy.append(psd_scipy)
            
            #pp.plot(psd_scipy.sample_frequencies, psd_scipy, label = "segment length %.0f"%i)
            #pp.plot(psd_inter.sample_frequencies, psd_inter, label="segment length %.0f"%i)
            #pp.plot(psd_inter.sample_frequencies, psd_inter, label = "segment length %.0f"%i)
        #pp.plot(p_estimated.sample_frequencies, p_estimated,"o", markersize=0.5,label = "psd zonder interpolate")
        #pp.plot(psd_period.sample_frequencies, psd_period, label = "psd period")
        
    #    pp.title("Estimated psd")
    #    pp.yscale('log')
    #    pp.xscale('log')
    #    pp.xlabel("frequency (Hz)")
    #    pp.ylabel("PSD (dB Re 1mPa$^2$/Hz)")
    #    pp.legend()
    #    pp.show()
    #    pp.close()
        Nperseg = 2048
        psd3 = spectrum(Nperseg, freq, data_td)
        psd3 = psd.interpolate(psd3, data_td.delta_f)
        print (psd3.shape)#from mp to micro pascal
    #psd  = spectrum(Nperseg, fs, data*scaling_factor_max*1000)#from mp to micro pascal
        max_snr = 0.
        max_snr_t = 0.
        L_fhigh = []
        L_pk = []
        print(psd_inter.shape,noise_td.shape,template.shape)
        L_snr = ['psd_inter','psd_seg','psd_scipy','psd_1']
       
        #for flow in np.arange(0,20000,10):
        #for i in L_snr:
        for j in np.arange(0,len(L_segment),1):
            max_snr = 0.
            #for fhigh in np.arange(8000,30000,100): 
            snr_inter = matched_filter(template, data_td, psd = psd_inter,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
            snr_inter = snr_inter.crop(0.1, 0.1)
            data_td= data_td.crop(0.1, 0.1)
            #snr_seg = matched_filter(template, data_td, psd = L_psd_seg[j],low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
            #snr_scipy = matched_filter(template, data_td, psd = L_psd_scipy[j],low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
            #snr_1 = matched_filter(template, data_td, psd = psd_1,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
            #snr_period = matched_filter(template, data_td, psd = psd_period,low_frequency_cutoff=flow,high_frequency_cutoff=fhigh)
            pk, pidx = snr_inter.abs_max_loc()
            peak_t = snr_inter.sample_times[pidx]
            L_fhigh.append(fhigh)
            L_pk.append(pk)
            if (pk > max_snr):
                max_snr_time = peak_t
                max_snr = pk
                
            print("Maximum SNR {} = {:5.2f} at t = {:.4f} s".format(L_segment[j], max_snr, max_snr_time))#, scale))
            L_max_snr.append(max_snr)
            print(pk)
            #L_scale.append(scale)
##        
        fig, axs = pp.subplots(2,1, sharex=True)
        axs = axs.ravel()
        axs[0].plot(data_td.sample_times,data_td, label = 'data')
        axs[1].plot(snr_inter.sample_times+0.00705, abs(snr_inter), label="Maximum SNR at t=%.5f"%(max_snr_time+0.00705)) #
            #ax.plot(snr_seg.sample_times, abs(snr_seg), label="%.0f"%L_segment[j])
        #ax.plot(snr_scipy.sample_times, abs(snr_scipy), label='psd_scipy')
        #ax.plot(snr_1.sample_times, abs(snr_1), label='psd=1')
    #    ax.plot(snr_period.sample_times, abs(snr_period), label='psd=period')
        pp.legend()
        fig.suptitle('E = 1e'+str(energy)+' GeV, r = '+str(rpos)+' m, z = '+str(zpos)+' m, sea state 0')
        ax.grid()
    #    pp.yscale('log')
    #    pp.xscale('log')
        #ax.set_ylim(0, None)
        axs[1].set_xlabel('Time (s)')
        axs[0].set(ylabel='Amplitude (mPa)')
        axs[1].set(ylabel='Signal-to-noise (dB)')

        ax2 = ax.twinx()
        ax2.plot(snr_inter.sample_times,data_td, color='green')
        
        pp.show()
        #pp.savefig("/home/gebruiker/Pictures/Overleaf/SNR_z"+str(zpos)+'_r'+str(rpos)+'_E'+str(energy)+'scaling'+str(scaling))
        
#    if plot_spectrum_data == True:
#        pp.figure(18,figsize=(6,5))
#        pp.rcParams['agg.path.chunksize'] = 10000
#        pp.plot(data_fd.sample_frequencies/1000,abs(data_fd), label="Noise with signal")
#        pp.plot(template_fd.sample_frequencies/1000, abs(template_fd),'-', label ='z =' + str(zpos))
#        #pp.plot(template_fd.sample_frequencies,template_fd, label="signal")
#        #pp.plot(noise_fd.sample_frequencies,noise_fd,"r--", label = "whale noise")
#        pp.title('E = 1e'+str(energy)+' GeV, r = '+str(rpos)+' m')
#        pp.ylim(0,0.4)
#        pp.xlim(0,30)
#        pp.legend()
#        pp.xlabel("frequency (kHz)")
#        pp.ylabel('normalised power')
#        pp.savefig("/home/gebruiker/Pictures/Overleaf/Spectrum_noise_z"+str(zpos)+'_r'+str(rpos)+'_E'+str(energy)+".png")
#pp.show()
        if plot_signal_data==True:
            pp.figure(14, figsize=(6,5))
            pp.plot(template.sample_times, template)
            #pp.plot(time, amplitude_og)#label ='z =' + str(zpos))
            pp.grid('on',which='both',axis='x')
            pp.grid('on',which='both',axis='y')
            pp.xlabel('Time (s)')
            pp.ylabel('Amplitude (mPa)')
            pp.xlim(0.006,0.01)
            #pp.ylim(-30,40)
            #pp.text(2.4,2.1,'E = 1e'+str(energy)+' GeV\nr  = '+str(rpos)+' m\nz  = '+ str(zpos)+' m',horizontalalignment="right", verticalalignment="center", transform = ax.transAxes, fontsize = 13)
            pp.legend()
            #pp.savefig("/home/gebruiker/Pictures/Overleaf/Signal_z"+str(zpos)+'_r'+str(rpos)+'_E'+str(energy)+'.png')
            pp.show()
        
            pp.figure(13, figsize=(6,5))
            pp.clf()
            pp.plot(data_td.sample_times,data_td)
            pp.grid('on',which='both',axis='x')
            pp.grid('on',which='both',axis='y')
            pp.xlabel('Time (s)')
            pp.ylabel('Amplitude (mPa)')
            pp.title('Sea state 6, incl. transfer function')
            pp.text(2.0,2.7,'E = 1e'+str(energy)+' GeV\nr ='+str(rpos)+' m\nz ='+ str(zpos)+' m', transform = ax.transAxes,fontsize = 13)
            pp.ylim(-6000,6000)
            pp.legend()
            pp.savefig("/home/gebruiker/Pictures/Overleaf/Sigbkg_z"+str(zpos)+'_r'+str(rpos)+'_E'+str(energy)+'_ss6_tf.png')
            pp.show()
            
            pp.figure(14, figsize=(6,5))
            pp.clf()
            pp.plot(data_td.sample_times,data_td)
            pp.grid('on',which='both',axis='x')
            pp.grid('on',which='both',axis='y')
            pp.xlabel('Time (s)')
            pp.ylabel('Amplitude (mPa)')
            pp.title('Sea state 6, incl transfer function')
            pp.text(2.0,2.7,'E = 1e'+str(energy)+' GeV\nr ='+str(rpos)+' m\nz ='+ str(zpos)+' m', transform = ax.transAxes,fontsize = 13)
            pp.xlim(0.461,0.464)
            pp.legend()
            pp.savefig("/home/gebruiker/Pictures/Overleaf/Sigbkg_z"+str(zpos)+'_r'+str(rpos)+'_E'+str(energy)+'_ss6_zoom.png')
            pp.show()

        if plot_Signal_fd == True:
                pp.plot(template_fd.sample_frequencies/1000, template_fd, label = str(zpos))
                pp.grid('on',which='both',axis='x')
                pp.grid('on',which='both',axis='y')
                pp.xlabel("Frequency (kHz)")
                pp.ylabel('Normalised power')
                pp.text(0.6,0.6,'E = 1e'+str(energy)+' GeV\nr ='+str(rpos)+' m', transform = ax.transAxes,fontsize = 13)
                pp.legend()
                pp.xlim(0,30)
        pp.show()
        pp.close()

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


if plot_SNR_scale == True:
    pp.plot(L_scale,L_max_snr)
    pp.title("Max value SNR as function of signal scale")
    pp.title("f_low =" + str(flow) + "f_high=" + str(fhigh))
    pp.savefig("SNR_scale_plot.png")
    pp.show()


    
