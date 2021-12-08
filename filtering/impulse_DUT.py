# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:28:00 2021

@author: edopp
"""

from impulse import *
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import wave, struct


#Allocate an instance of class impulse
im=impulse()
#Set the sampling frequency to 100kHz
im.Fs = 100000.

file_name = 'neutrino_12.0_1000_1_11_100000.wav'
f = wave.open(file_name,'r')
frames = f.readframes(-1)
u = np.frombuffer(frames, dtype="int16")

#Construct an artificial DUT and it's response
#The artificial DUT is a simple 2nd order system (a mechanical mass-spring system)
#The damping ration
for i in np.arange(0.01,0.1,0.01):
#for j in np.arange(0,30000,3000): 
#for k in np.arange(20000,200000,20000):
    zeta = i
    #The resonance frequency
    f    = 15000.0
    Ts   = 1./im.Fs
    #Design the equivalent digital filter, representing the mass-spring system
    b    = [1,0,0]
    pole = discrete_2ndOrder_pole(im,zeta,f,Ts) #member of class impulse
    a    = np.poly([pole,np.conjugate(pole)])
    print ("for zeta is %f the a coefficient is %s" %(zeta, a))
    #Input signal is a random white noise vector
    #u    = 0.5-np.random.rand(im.length)
    #Output is the response of the artificial DUT
    y    = signal.lfilter(b,a,u)
    #Copy input and output signals
    im.Data = np.squeeze([u,y])
    #Estimate the impulse response of the artificial DUT
    ImpulseResponse(im) #member of class impulse
    #Plot the result
    plt.figure(1);plt.clf();plt.stem(np.real(im.Impulse))
    plt.grid()
    plt.title('Impulse response DUT, zeta = {:.2f}, f = {:.2f}, Ts = {:.2f}' .format(zeta,f,Ts))
    plt.xlabel('tau -> []')
    plt.xlim(0,60)
    plt.ylabel('h(tau)')
    plt.savefig("Ed_script_plots/Impluse_response"+'_' + str(zeta)+'_' + str(f)+'_'+str(Ts)+'.png')
    
