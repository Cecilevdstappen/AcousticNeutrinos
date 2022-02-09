# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 20:52:16 2022

@author: edopp
"""
import sys 
sys.path.insert(0, '/home/gebruiker/SIPPY')
sys.path.insert(0, '/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts')
sys.path.append('C:/gebruiker/AcousticNeutrinos/filtering/Ed_scripts')
from scipy import signal
from mouse_button import *
from design_FRF import *
import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
from our_snr import amplitude_og, time, noise_signal_td, noise_signal_times

frf = design_FRF(Fs=144000)
#frf.highpass = True

# frf.pz2tf((375,12000,17000),(0.15,0.02,0.07),100000) # hydrophone with cap
frf.pz2tf((375,12000),(0.1,0.03),144000) # hydrophone without cap
# frf.pz2tf((12000,375),(0.02,0.1),100000) # hydrophone without cap
frf.mbutton.on()

nPoints = 1024
#T = np.arange(0,nPoints,1)/frf.Fs
#x = np.zeros(nPoints)
#x[2] = 1
T = time
x = amplitude_og
y = sg.lfilter(frf.num,frf.dnum,x)
T_data = noise_signal_times
x_data = noise_signal_td 
y_data = sg.lfilter(frf.num,frf.dnum,x_data)
plt.figure(12);
plt.clf()
plt.plot(T,y,T,x)
plt.grid('on',which='both',axis='x')
plt.grid('on',which='both',axis='y')
plt.title('Impulse response signal')
plt.xlabel('t -> [s]')
plt.show()

plt.figure(13)
plt.clf()
plt.plot(T_data, x_data,T_data,y_data)
plt.grid('on',which='both',axis='x')
plt.grid('on',which='both',axis='y')
plt.title('Impulse response data')
plt.xlabel('t -> [s]')
plt.show()
