# AcousticNeutrinos

Master thesis on acoustic neutrino detection. 

# Signal reconstruction

1. Use Corsika to simulate neutrino signal, as the energy distribution as function of longitudinal bin.

2. With the cs_mean_energy_100_showers.py the average can be taken from 100 showers generated by Corsika.

3. Use the run_thermoacousticmodel_cgatius.py to plot the (mean) Corsika output as pressure as function of time. The run_thermoacousticmodel_gatius.py can also be used to add background to the signal. Output is dat/wav file. 

4. This .wav file can be plotted using the audioplot script signal vs time. 

5. The Spectrogram or Spectroplot plot the signal + background dat file as amplitude vs frequency. 

# Filtering

1. With the SIPPY scripts Pulses_FIR_ARX_ARMAX.py and Neutrino_FIR_ARX_ARMAX.py the coefficients and filters can be determined using the output signal y and the input signal u.

2. The convolve_and_coeff.py convolves the measured transfer function Bodemat_Dubbel_1FF_Actran_Sweep200Hz20kHz0degree_AfterPressure_v2.dat and the neutrino signal (u) neutrino_12.0_1000_1_11.dat to get the output y.

3. The impulse_DUT.py makes use of impulse.py to determine the model. It uses the sample frequentie (Fs), de resonantie frequentie (f) en de relatieve demping (zeta) to detemine the coefficients and then detemines the model. 

4. Audioplot plots neutrino signal (+ background)

5. The scripts freq_bandpass_filter.py and transfer_filter.py use the scipy.signal.butter and scipy.signal.lfilter function to apply lowpass, highpass, bandpass filters and FIR filters. 

6. Use Audacity to get spectrogram from .wav file
