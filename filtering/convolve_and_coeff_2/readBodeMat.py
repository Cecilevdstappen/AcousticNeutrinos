# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:17:53 2021

@author: edopp
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

#load Matlab mat file and extract 'Bodemat' structure
mat_contents = sio.loadmat('C:\projects2021\KM3Net\convolve_and_coeff_2\Datamat_Dubbel_1FF_Actran_Sweep200Hz20kHz0degree_AfterPressure_v2.mat',squeeze_me=True,struct_as_record=False)
Bodemat = mat_contents['Bodemat']

fr  = Bodemat.FR
hxy = Bodemat.hxy
coh = Bodemat.coh
Transfer = Bodemat.NameRsp+'/'+Bodemat.NameRef
 #Some simple plots to check the Bodemat
# plt.figure(1)
# plt.clf()
# plt.semilogx(fr,20.*np.log10(abs(hxy)))
# plt.figure(2)
# plt.clf()
# plt.plot(fr,coh)

plt.figure(1)
plt.clf()
hgcf = plt.gcf().number
nPlot = 3
b_ln1 = np.arange(0, nPlot, dtype=object)
plt.figure(num=hgcf, clear=True)
b_fig, b_axs = plt.subplots(nPlot, 1, num=hgcf)

Lrng = int(len(hxy))
rng = np.arange(Lrng, dtype=int)
# freqs = np.fft.fftfreq(self.Txy.shape[0], d=(1./self.Fs))
LowLim = 1
UpperLim = max(fr)

b_ln1[0] = b_axs[0].semilogx(
    fr[rng], 20.*np.log10(np.absolute(hxy[rng])),'r')
b_axs[0].minorticks_on()
b_axs[0].grid('on', which='both', axis='x')
b_axs[0].grid('on', which='major', axis='y')
b_axs[0].set_xlim(LowLim, UpperLim)
b_axs[0].set_ylabel('Txy -> dB re ['+'$\mu$Pa/'+'$\mu$Pa]')
b_axs[0].set_title(Bodemat.Experiment,loc='left')
b_axs[0].set_title(Transfer,loc='right')

# plt.subplot(312)
b_ln1[1] = b_axs[1].semilogx(
    fr[rng], 180.*((np.angle(hxy[rng])))/np.pi,'r')
b_axs[1].grid('on', which='both', axis='x')
b_axs[1].grid('on', which='major', axis='y')
b_axs[1].set_xlim(LowLim, UpperLim)
b_axs[1].set_ylabel('Phase ->  [$\circ$]')

# plt.subplot(313)
b_ln1[2] = b_axs[2].semilogx(fr[rng], coh[rng],'r')
b_axs[2].grid('on', which='both', axis='x')
b_axs[2].grid('on', which='major', axis='y')
b_axs[2].set_xlim(LowLim, UpperLim)
b_axs[2].set_ylabel('Coherence ->  []')
b_axs[2].set_xlabel('Frequency -> [Hz]')
