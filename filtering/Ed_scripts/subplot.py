#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:13:13 2022

@author: gebruiker
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.fft import fft, ifft, fftfreq, fftshift
#from my_styles import *
#
#set_paper_style()
#
#plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')

L_filename = []
rpos=300
energy=11
zpos = 4

fig, axs = plt.subplots(5,1, sharex=True)
axs = axs.ravel()

for _i, energy in enumerate([9,10,11,12,13]):
    template_file = '../Neutrino_data_files/neutrino_'+str(float(zpos)) +'_'+str(rpos)+'_1_'+str(energy)+'.dat'
    data = np.loadtxt(template_file,  usecols=(0,1), dtype='float', unpack=True)  
    time = data[0]
    amplitude_og = data[1]
    amplitude_og =amplitude_og*1000
    amplitude_og_ft = abs(fft(amplitude_og))
    fx = fftfreq(len(amplitude_og), time[1]-time[0])
    fx=fftshift(fx)
    print(time)
    print(amplitude_og)
    filename = 'neutrino_'+str(float(zpos)) +'_'+str(rpos)+'_1_'+str(energy)+'.png'
    #make the plots
    axs[_i].plot(fx,amplitude_og_ft, label ='z =' + str(zpos))
plt.suptitle('E = 1e11 GeV, r = 300 m')
plt.xlabel("Time (s)")
plt.ylabel('Amplitude (mPa)')
plt.legend()
plt.savefig(filename)

    
L_filename.append(filename)
fig.tight_layout()
plt.show()
"""
for i in range(0,len(L_filename)-1):
    img = mpimg.imread(str(L_filename[i]))
    axs[i,0].plot(img)
    axs[i,0].set_title(str(L_filename[i]))
    
""" 