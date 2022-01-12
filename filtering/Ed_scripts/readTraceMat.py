  # -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:17:53 2021

@author: edopp
"""
# %%
from __future__ import division
from impulse import *
import scipy
import scipy.io as sio
from past.utils import old_div
from sippy import functionset as fset
from sippy import *
import numpy as np
import control.matlab as cnt
import matplotlib.pyplot as plt

#load Matlab mat file and extract 'Datamat' structure Enkel_3FF_Atran_Sweep200Hz20kHz0degreev1
#Datamat_Dubbel_1FF_Actran_Sweep200Hz20kHz0degree_AfterPressure_v2

mat_contents = sio.loadmat('./Datamat_Enkel_3FF_Atran_Sweep200Hz20kHz0degreev1.mat',squeeze_me=True,struct_as_record=False) 
mat_contents = sio.loadmat('./Datamat_Enkel_3FF_Atran_Sweep200Hz20kHznoCaps0degreev2.mat',squeeze_me=True,struct_as_record=False)
# mat_contents = sio.loadmat('C:\projects2021\KM3Net\convolve_and_coeff_2\Datamat_Dubbel_1FF_Atran_Sweep200Hz20kHz0degreev1.mat',squeeze_me=True,struct_as_record=False)
# mat_contents = sio.loadmat('C:\projects2021\KM3Net\convolve_and_coeff_2\Datamat_Dubbel_1FF_Actran_Sweep200Hz20kHz0degree_AfterPressure_v2.mat',squeeze_me=True,struct_as_record=False)

Datamat = mat_contents['Datamat']

Fs    = Datamat.Fs
tRaw  = Datamat.time
trace = Datamat.trace
rChnl = Datamat.NumChnl
Unit  = Datamat.UnitRef
Name  = Datamat.NameRef
Experiment = Datamat.Experiment
Ts    = 1./Fs

plt.figure(1)
plt.clf()
hgcf = plt.gcf().number
nPlot = 3
b_ln1 = np.arange(0, nPlot, dtype=object)
plt.figure(num=hgcf, clear=True)
b_fig, b_axs = plt.subplots(nPlot, 1, num=hgcf)
hgcr = plt.gcf().number
nPlot = (rChnl)
d_ln1 = np.arange(0, nPlot, dtype=object)
d_ln1 = np.arange(0, nPlot, dtype=object)
plt.figure(num=hgcr, clear=True)
d_fig, d_axs = plt.subplots(nPlot, 1, num=hgcr)
n_chl = 0
for Chnl in np.arange(rChnl):
    d_ln1[n_chl] = d_axs[n_chl].plot(
        tRaw, (trace[:, Chnl]),'r', label='Fs = 100 kHz')
    d_axs[n_chl].legend(loc='upper center', shadow=True, fontsize='medium')
    stdev = (np.std(trace[:, n_chl]))
    str = " %s   $\sigma$ = %6.4e %s    " % (Name[Chnl],stdev, Unit[Chnl]);
    d_axs[n_chl].set_title(str)
    str = "%s -> [%s] " % (Name[Chnl], Unit[Chnl]);
    d_axs[n_chl].set_ylabel(str)
    d_axs[n_chl].autoscale(tight=True)
    d_axs[n_chl].grid()
    d_axs[n_chl].grid('on', 'minor')
    n_chl += 1
d_axs[n_chl-1].set_xlabel('time ->[s]')
d_axs[1].set_title(Experiment,loc='left')
plt.show()
# %%
#Allocate an instance of class impulse
im=impulse()
#Set the sampling frequency to 100kHz
im.imp_Fs   = Fs
im.imp_NFFT = 4096
im.imp_PSDOn = 0
im.imp_Segment = 100
im.imp_Data = np.squeeze(trace)
im.imp_Data[:,1] = -im.imp_Data[:,1]
#Estimate the impulse response of the artificial DUT
ImpulseResponse(im,1,0) #member of class impulse
#Plot the result
plt.figure(2);plt.clf()
hgca = plt.gcf().number
f_fig, axs= plt.subplots(5, 1,sharex=False,sharey=False, num=hgca)
axs[0].stem(np.real(im.imp_Impulse[0:im.imp_NFFT]),label='Estimated Impulse response DUT. Fs = 100kHz')
axs[0].legend(loc='upper center', shadow=True, fontsize='medium')
axs[0].minorticks_on()
axs[0].grid('on',which='both',axis='x')
axs[0].grid('on',which='major',axis='y')
axs[0].set_title(Experiment)
axs[0].set_xlabel('tau -> []')
axs[0].set_ylabel('h(tau)')
# plt.legend(Experiment)

IFFT = int(im.imp_NFFT/2)
sx   = np.fft.fft((im.imp_Impulse[0:im.imp_NFFT]))
freq = np.arange(0,IFFT)*im.imp_Fs/im.imp_NFFT
axs[1].semilogx(freq,20.*np.log10(abs(sx[0:int(IFFT)])),label='Verification. |FRF| of the Impulse response')
axs[1].legend(loc='upper center', shadow=True, fontsize='medium')
axs[1].set_xlim(10,im.imp_Fs/2)
axs[1].set_ylim(-20,40)
axs[1].minorticks_on()
axs[1].grid('on',which='both',axis='x')
axs[1].grid('on',which='major',axis='y')
axs[1].set_title('FRF impulse response DUT')
axs[1].set_ylabel('FRF(f)')

axs[2].semilogx(freq,180.*np.unwrap(np.angle((sx[0:int(IFFT)])))/np.pi,label='Verification. phase(FRF) of the Impulse response')
axs[2].legend(loc='lower center', shadow=True, fontsize='medium')
axs[2].set_xlim(10,im.imp_Fs/2)
# axs[2].set_ylim(-20,40)
axs[2].minorticks_on()
axs[2].grid('on',which='both',axis='x')
axs[2].grid('on',which='major',axis='y')
axs[2].set_title('Phase impulse response DUT')
axs[2].set_ylabel('angel(f)')


freq2 = np.arange(0,im.imp_NFFT)*0.5*im.imp_Fs/im.imp_NFFT
axs[3].semilogx(freq2[0:int(im.imp_NFFT)],20.*np.log10(abs(im.imp_Txy[0:int(im.imp_NFFT)])),label='Estimated Txy of time traces')
axs[3].legend(loc='lower center', shadow=True, fontsize='medium')
axs[3].set_xlim(10,im.imp_Fs/2)
axs[3].set_ylim(-20,40)
axs[3].minorticks_on()
axs[3].grid('on',which='both',axis='x')
axs[3].grid('on',which='major',axis='y')
axs[3].set_title('Txy response DUT')
axs[3].set_ylabel('Txy(f)')

freq2 = np.arange(0,im.imp_NFFT)*0.5*im.imp_Fs/im.imp_NFFT
axs[4].semilogx(freq2[0:int(im.imp_NFFT)],(abs(im.imp_Cxy[0:int(im.imp_NFFT)])),label='Coherence function  of the estimated Txy of time traces')
axs[4].legend(loc='lower center', shadow=True, fontsize='medium')
axs[4].set_xlim(10,im.imp_Fs/2)
axs[4].minorticks_on()
axs[4].grid('on',which='both',axis='x')
axs[4].grid('on',which='major',axis='y')
axs[4].set_title('Coherence function DUT')
axs[4].set_xlabel('F -> [Hz]')
axs[4].set_ylabel('Cxy(f)')

# %%
# Define white noise as noise signal
imp=im.imp_Impulse[0:int(im.imp_NFFT/8)]
den=np.zeros(int(im.imp_NFFT/8))
den[0]=1.
gtf=cnt.tf(imp,den,1/im.imp_Fs)
plt.figure(3);plt.clf();cnt.bode(gtf,Hz=1)
plt.show()
# %%
filename_sig = 'neutrino_12.0_1000_1_11.dat'
f = open(filename_sig,'r')
lines = f.readlines()
time = [] #s
amplitude = []
dtypes = [('col0', float)]+[('col1', np.complex128)]
data = np.loadtxt(filename_sig,  usecols=(0,1), dtype=dtypes, unpack=True)#,  dtype='float,np.complex128)
time = data[0]
amplitude = data[1]
norm = np.linalg.norm(amplitude)
amplitude = amplitude/norm
end_time = time[-1]                                      # [s]
npts = len(time)-1
sampling_time = end_time/npts

# %%
impResponse = np.real(im.imp_Impulse[0:im.imp_NFFT])
convolved__t = scipy.signal.convolve(amplitude, impResponse, mode='full', method='auto')
plt.figure(4);plt.clf()
hgcc = plt.gcf().number
f_fig, axs= plt.subplots(2, 1,sharex=False,sharey=False, num=hgcc)
axs[0].plot(time,amplitude,label='Neutrino pulse. Fs = 200kHz')
axs[0].legend(loc='upper center', shadow=True, fontsize='medium')
axs[0].minorticks_on()
axs[0].grid('on',which='both',axis='x')
axs[0].grid('on',which='major',axis='y')
axs[0].set_title('neutrino_12.0_1000_1_11')
axs[0].set_xlabel('t -> [s]')
axs[0].set_ylabel('imp(t)')
# plt.legend(Experiment)

time2     = np.arange(0,len(convolved__t),1)*sampling_time
axs[1].plot(time2,convolved__t,label='Convolved Neutrino pulse. Fs = ???')
axs[1].legend(loc='upper center', shadow=True, fontsize='medium')
# axs[1].set_xlim(10,im.imp_Fs/2)
# axs[1].set_ylim(-20,40)
axs[1].minorticks_on()
axs[1].grid('on',which='both',axis='x')
axs[1].grid('on',which='major',axis='y')
axs[1].set_title('Impulse response DUT')
axs[1].set_xlabel('t -> [s]')
axs[1].set_ylabel('imp_conv(t)')

# %%
end_time = 0.3
npts = int(old_div(end_time, Ts)) + 1
white_noise_variance = [0.1]
e_t = fset.white_noise_var(npts, white_noise_variance)[0]
Time = np.linspace(0, end_time, npts)
Ytot, Time, Xsim = cnt.lsim(gtf, e_t, Time)
Usim = e_t

na_ord = [6]; nb_ord = [[6]]; nc_ord = [4]; theta = [[11]]

# ITERATIVE ARMAX
# Id_ARMAXi = system_identification(Ytot, Usim, 'ARMAX', ARMAX_orders = [na_ord, nb_ord, nc_ord, theta],Id
#                                   max_iterations = 300, ARMAX_mod = 'ILLS')

# # OPTIMIZATION-BASED ARMAX
# Id_ARMAXo = system_identification(Ytot, Usim, 'ARMAX', ARMAX_orders = [na_ord, nb_ord, nc_ord, theta],
#                                   max_iterations = 300, ARMAX_mod = 'OPT') 

# RECURSIVE ARMAX
Id_ARMAXr = system_identification(Ytot, Usim, 'ARMAX', ARMAX_orders = [na_ord, nb_ord, nc_ord, theta], 
                                  max_iterations=300, ARMAX_mod = 'RLLS')
# %%
# num4=Id_ARMAXr.NUMERATOR_H
# den4=Id_ARMAXr.DENOMINATOR_H
# gpz=cnt.tf(num4,den4,Ts)
plt.figure(5);plt.clf();tmp=cnt.bode(Id_ARMAXr.H,Hz=1)
