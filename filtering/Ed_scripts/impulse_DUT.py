# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:28:00 2021

@author: edopp
"""

from impulse import *
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

#Allocate an instance of class impulse
im=impulse()
#Set the sampling frequency to 100kHz
im.imp_Fs = 100000.

#Construct an artificial DUT and it's response
#The artificial DUT is a simple 2nd order system (a mechanical mass-spring system)
#The damping ration
zeta = 0.1
#The resonance frequency
f    = 15000.
Ts   = 1./im.imp_Fs
#Design the equivalent digital filter, representing the mass-spring system
b    = [1,0,0]
pole = discrete_2ndOrder_pole(im,zeta,f,Ts) #member of class impulse
a    = np.poly([pole,np.conjugate(pole)])
print(a)
#Input signal is a random white noise vector
u    = 0.5-np.random.rand(im.imp_length)
#Output is the response of the artificial DUT
y    = signal.lfilter(b,a,u)
#Copy input and output signals
# im.imp_Data = np.squeeze([u,y])
im.imp_Data = np.squeeze(trace)
#Estimate the impulse response of the artificial DUT
ImpulseResponse(im) #member of class impulse
#Plot the result
plt.figure(1);plt.clf();plt.stem(np.real(im.imp_Impulse))
plt.grid()
plt.title('Impulse response DUT')
plt.xlabel('tau -> []')
plt.ylabel('h(tau)')