#!/usr/bin/env python
# coding: utf-8
"""
ARMAX Example

@author: Giuseppe Armenise, revised by RBdC
"""

from __future__ import division
from past.utils import old_div
from sippy import functionset as fset
from sippy.armax import Armax
from sippy.arx import ARX_id, select_order_ARX
from sippy import *
from scipy.io import savemat
import numpy as np
import control.matlab as cnt
import matplotlib.pyplot as plt
import wave, struct
import pylab
import sys
## TEST IDENTIFICATION METHODS for ARMAX model

#Data from pulse file
file_name = 'neutrino_12.0_1000_1_11_100000.wav'
f = wave.open(file_name,'r')
frames = f.readframes(-1)
u = np.frombuffer(frames, dtype="int16")
fs = f.getframerate()

# If Stereo
if f.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)

time = np.linspace(0,len(u)/fs, num = len(u))


# Define sampling time and Time vector
#sampling_time = 1.                                  # [s]
end_time = time[-1]                                      # [s]
npts = len(time)-1
sampling_time = end_time/npts
#Time = np.linspace(0, (end_time+2*sampling_time), (npts+2))
print(end_time)
print(time[:10])
#print(Time[:10])
# Define Generalize Binary Sequence as input signal 
switch_probability = 0.08  # [0..1]
#[Usim,_,_] = fset.GBN_seq(npts, switch_probability, Range = [-1, 1])
Usim= np.asarray(u)
# Define white noise as noise signal
white_noise_variance = [0.0001]
e_t = fset.white_noise_var(Usim.size, white_noise_variance)[0]

# ## Define the system (ARMAX model)

def transfer_function():
    import scipy.io
    # coefficients a and b are determined elsewhere
    mat = scipy.io.loadmat('coeff_armax_A.mat')
    a =  mat['G_den'][0]

    mat = scipy.io.loadmat('coeff_armax_B.mat')
    b = mat['G_num'][0]
    return a, b

DEN, NUM = transfer_function()
NUM_H = [1.,0.,0.,0.,0.]

# ### Define transfer functions

g_sample = cnt.tf(NUM, DEN, sampling_time)
h_sample = cnt.tf(NUM_H, DEN, sampling_time)

# ## Time responses

# ### Input reponse
Time = np.asarray(time)
Y1, Time, Xsim = cnt.lsim(g_sample, Usim, Time)
plt.figure(0)
plt.plot(Time,Usim)
plt.plot(Time, Y1)
plt.xlabel("Time")
plt.title("Time response y$_k$(u) = g$\cdot$u$_k$")
plt.legend(['u(t)', 'y(t)'])
plt.grid()
plt.show()

# ### Noise response

Y2, Time, Xsim = cnt.lsim(h_sample, e_t, Time)
plt.figure(1)
plt.plot(Time, e_t)
plt.plot(Time, Y2)
plt.xlabel("Time")
plt.title("Time response y$_k$(e) = h$\cdot$e$_k$")
plt.legend(['e(t)', 'y(t)'])
plt.grid()
plt.show()

# ## Total output
# $$Y_t = Y_1 + Y_2 = G.u + H.e$$

Ytot = Y1 + Y2
Utot = Usim + e_t
plt.figure(2)
plt.plot(Time, Utot)
plt.plot(Time, Ytot)
plt.xlabel("Time")
plt.title("Time response y$_k$ = g$\cdot$u$_k$ + h$\cdot$e$_k$")
plt.legend(['u(t) + e(t)', 'y_t(t)'])
plt.grid()


##### SYSTEM IDENTIFICATION from collected data

# choose identification mode

mode = 'FIXED'

if mode == 'IC':
    # use Information criterion
    
    Id_ARMAXi = syistem_identification(Ytot, Usim, 'ARMAX', IC='AIC', na_ord=[4, 4], nb_ord=[3, 3],
                              nc_ord=[2, 2], delays=[11, 11], max_iterations=300, ARMAX_mod = 'ILLS')
    
    
    Id_ARMAXo = system_identification(Ytot, Usim, 'ARMAX', IC='AICc', na_ord=[4, 4], nb_ord=[3, 3],
                                      nc_ord=[2, 2], delays=[11, 11], max_iterations=300, ARMAX_mod = 'OPT')
    
    Id_ARMAXr = system_identification(Ytot, Usim, 'ARMAX', IC='BIC', na_ord=[4, 4], nb_ord=[3, 3],
                              nc_ord=[2, 2], delays=[11, 11], max_iterations=300, ARMAX_mod = 'RLLS')
    


elif mode == 'FIXED':
    # use fixed model orders
    
    na_ord = [4]; nb_ord = [[3]]; nc_ord = [2]; theta = [[11]]
    
    # ITERATIVE ARMAX
    Id_ARMAXi = system_identification(Ytot, Usim, 'ARMAX', ARMAX_orders = [na_ord, nb_ord, nc_ord, theta],
                                      max_iterations = 300, ARMAX_mod = 'ILLS' )

    # OPTIMIZATION-BASED ARMAX
    Id_ARMAXo = system_identification(Ytot, Usim, 'ARMAX', ARMAX_orders = [na_ord, nb_ord, nc_ord, theta],
                                      max_iterations = 300, ARMAX_mod = 'OPT') 
    
    # RECURSIVE ARMAX
    Id_ARMAXr = system_identification(Ytot, Usim, 'ARMAX', ARMAX_orders = [na_ord, nb_ord, nc_ord, theta], 
                                      max_iterations=300, ARMAX_mod = 'RLLS')

# ARX
    Id_ARX = system_identification(Ytot, Usim, 'ARX', ARX_orders=[5, 5, 2])  #

# FIR
    Id_FIR = system_identification(Ytot, Usim, 'FIR', FIR_orders=[3, 0])

Y_armaxi = Id_ARMAXi.Yid.T
Y_armaxo = Id_ARMAXo.Yid.T
Y_armaxr = Id_ARMAXr.Yid.T   
    
Y_arx = Id_ARX.Yid.T
Y_fir = Id_FIR.Yid.T
# ## Check consistency of the identified system

plt.figure(3)
plt.plot(Time, Usim)
#plt.plot(time,u)
plt.ylabel("Input GBN")
plt.xlabel("Time")
plt.title("Input, identification data (Switch probability=0.08)")
plt.grid()
plt.show()

plt.figure(4)
plt.plot(time, u, label="u")
plt.plot(Time, Ytot,'--', label = "Y with coeff")
#plt.plot(Time, Y_armaxi)
#plt.plot(Time, Y_armaxo)
#plt.plot(Time, Y_armaxr)
plt.plot(Time, Y_fir, label ="FIR")
#plt.plot(Time,Y_arx,label = "ARX")
plt.grid()
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.title("Output, (identification data)")
#plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])
plt.legend()
plt.show()


##### VALIDATION of the identified system:
# ## Generate new time series for input and noise

switch_probability = 0.07  # [0..1]
input_range = [0.5, 1.5]
[U_valid,_,_] = fset.GBN_seq(npts, switch_probability, Range = input_range)
white_noise_variance = [0.01]
e_valid = fset.white_noise_var(U_valid.size, white_noise_variance)[0]
#
## Compute time responses for true system with new inputs

Yvalid1, Time, Xsim = cnt.lsim(g_sample, U_valid, Time)
Yvalid2, Time, Xsim = cnt.lsim(h_sample, e_valid, Time)
Ytotvalid = Yvalid1 + Yvalid2

# ## Compute time responses for identified systems with new inputs

# ARMAX - ILLS
Yv_armaxi = fset.validation(Id_ARMAXi,U_valid,Ytotvalid,Time)
#Best_estimate_i = Arm.find_best_estimate(Yv_armaxi,U_valid)
#savemat("coeff_armax.mat",{"Best_estimate_i":Best_estimate_i})

# ARMAX - OPT
Yv_armaxo = fset.validation(Id_ARMAXo,U_valid,Ytotvalid,Time)
#Best_estimate_o = Arm.find_best_estimate(y,u)

# ARMAX - RLLS 
Yv_armaxr = fset.validation(Id_ARMAXr,U_valid,Ytotvalid,Time)
#Best_estimate_r = Arm.find_best_estimate(y,u)

#print(Best_estimate_i)
#print(Best_estimate_o)
#print(Best_estimate_r)

# Plot
plt.figure(5)
plt.plot(Time, U_valid)
plt.ylabel("Input GBN")
plt.xlabel("Time")
plt.title("Input, validation data (Switch probability=0.07)")
plt.grid()
plt.show()

plt.figure(6)
plt.plot(Time, Ytotvalid)
plt.plot(Time, Yv_armaxi.T)
plt.plot(Time, Yv_armaxo.T)
plt.plot(Time, Yv_armaxr.T)
plt.xlabel("Time")
plt.ylabel("y_tot")
plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])
plt.grid()
plt.show()

#rmse = np.round(np.sqrt(np.mean((Ytotvalid - Yv_armaxi.T) ** 2)), 2)
EV = 100*(np.round((1.0 - np.mean((Ytotvalid - Yv_armaxi) ** 2)/np.std(Ytotvalid)), 2))
#plt.title("Validation: | RMSE ARMAX_i = {}".format(rmse))
plt.title("Validation: | Explained Variance ARMAX_i = {}%".format(EV))


## Bode Plots
w_v = np.logspace(-3,4,num=701)
plt.figure(7)
mag, fi, om = cnt.bode(g_sample,w_v)
mag1, fi1, om = cnt.bode(Id_ARMAXi.G,w_v)
mag2, fi2, om = cnt.bode(Id_ARMAXo.G,w_v)
mag3, fi3, om = cnt.bode(Id_ARMAXr.G,w_v)
plt.subplot(2,1,1), plt.loglog(om,mag), plt.grid(), 
plt.loglog(om,mag1), plt.loglog(om,mag2), plt.loglog(om,mag3)
plt.xlabel("w"),plt.ylabel("Amplitude Ratio"), plt.title("Bode Plot G(iw)")
plt.subplot(2,1,2), plt.semilogx(om,fi), plt.grid()
plt.semilogx(om,fi1), plt.semilogx(om,fi2), plt.semilogx(om,fi3),
plt.xlabel("w"),plt.ylabel("phase")
plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])

plt.figure(8)
mag, fi, om = cnt.bode(h_sample,w_v)
mag1, fi1, om = cnt.bode(Id_ARMAXi.H,w_v)
mag2, fi2, om = cnt.bode(Id_ARMAXo.H,w_v)
mag3, fi3, om = cnt.bode(Id_ARMAXr.H,w_v)
plt.subplot(2,1,1), plt.loglog(om,mag), plt.grid(), 
plt.loglog(om,mag1), plt.loglog(om,mag2), plt.loglog(om,mag3)
plt.xlabel("w"),plt.ylabel("Amplitude Ratio"), plt.title("Bode Plot H(iw)")
plt.subplot(2,1,2), plt.semilogx(om,fi), plt.grid()
plt.semilogx(om,fi1), plt.semilogx(om,fi2), plt.semilogx(om,fi3),
plt.xlabel("w"),plt.ylabel("phase")
plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])


## Step test
# G(z)
plt.figure(9)
yg1 = cnt.step(g_sample,Time)
yg2 = cnt.step(Id_ARMAXi.G,Time)
yg3 = cnt.step(Id_ARMAXo.G,Time)
yg4 = cnt.step(Id_ARMAXr.G,Time)
plt.plot(Time,yg1[0].T)
plt.plot(Time,yg2[0].T)
plt.plot(Time,yg3[0].T)
plt.plot(Time,yg4[0].T)
plt.title("Step Response G(z)")
plt.xlabel("time"),plt.ylabel("y(t)"), plt.grid(),
plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])
# H(z)
plt.figure(10)
yh1 = cnt.step(h_sample,Time)
yh2 = cnt.step(Id_ARMAXi.H,Time)
yh3 = cnt.step(Id_ARMAXo.H,Time)
yh4 = cnt.step(Id_ARMAXr.H,Time)
plt.plot(Time,yh1[0].T)
plt.plot(Time,yh2[0].T)
plt.plot(Time,yh3[0].T)
plt.plot(Time,yh4[0].T)
plt.title("Step Response H(z)")
plt.xlabel("time"),plt.ylabel("y(t)"), plt.grid(),
plt.legend(['System', 'ARMAX-I', 'ARMAX-0', 'ARMAX-R'])



