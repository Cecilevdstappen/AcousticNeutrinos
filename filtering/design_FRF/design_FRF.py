# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 17:32:13 2021

@author: edopp
"""
# %%
from __future__ import division
from sippy import *
import numpy as np
import control.matlab as cnt
import matplotlib.pyplot as plt
from numpy import  exp,  sqrt
from math import pi
from scipy.signal import butter, lfilter, freqz, detrend
from mouse_button import *

class design_FRF:
# %%
    def __init__(self, Fs=10000,Figno=10):
        self.Fs       = Fs
        self.Ts       = 1./self.Fs
        self.Figno    = Figno
        self.mbutton  = None
        self.highpass = False
        #Initial Gain system
        self.rn    = [0.97,0.97]
        self.rd    = [0.95]
        
        
    def pz2tf(self,fos,zetas,Fs):
        self.Fs    = Fs
        self.fos   = fos
        self.zetas = zetas
        self.Ts    = 1./Fs
        self.no    = size(fos)
        self.rd    = [0.95]
        self.num   = np.array([1.],dtype='float64')
        self.dnum  = [1.]
        self.Gain  = 1.
        
        
        #Create the complex conjugated-poles
        if self.no == 1:
           n,d,g     = self.discrete_2ndOrder_polynoom(self.zetas,self.fos,self.Ts)
           self.num  = np.convolve(self.num,n)
           self.dnum = np.convolve(self.dnum,d)
           self.Gain = sum(self.num)/sum(self.dnum)
           self.num /= self.Gain
           if self.highpass:
               bhigh,ahigh = self.butter_highpass_filter(300,self.Fs)
               self.num = np.convolve(self.num,bhigh)
               self.dnum = np.convolve(self.dnum,ahigh)
           # self.Gain *= g
        else:   
            for ii in range(self.no):
                n,d,g     = self.discrete_2ndOrder_polynoom(self.zetas[ii],self.fos[ii],self.Ts)
                if ii == 0:
                    self.num  = n
                    self.dnum = d
                    self.Gain = sum(self.num)/sum(self.dnum)
                    self.num /= self.Gain
                    if self.highpass:
                        bhigh,ahigh = self.butter_highpass_filter(100,self.Fs)
                        self.num = np.convolve(self.num,bhigh)
                        self.dnum = np.convolve(self.dnum,ahigh)
                else:
                    self.compose_parallel(n,d)
           
            self.Gain = sum(self.num)/sum(self.dnum)
            self.num /= self.Gain
            
        #Convolve Butterfilter and 4th-order system
        #High pass Butterworth filter
        if self.highpass:
            bhigh,ahigh = self.butter_highpass_filter(20,self.Fs)
            # self.num = np.convolve(self.num,bhigh)
            # self.dnum = np.convolve(self.dnum,ahigh)
 
        #Create Z-domain transfer function 
        self.tf = cnt.tf(self.num,self.dnum,self.Ts)
        
        #Plot bode response
        plt.figure(self.Figno);plt.clf();cnt.bode(self.tf,2.*pi*np.arange(100.,self.Fs/2.,0.1),Hz=1)
        #Plot the pole-zero map
        plt.figure((self.Figno+1));plt.clf();cnt.pzmap(self.tf,plot=1,grid=1)
        
        ax = plt.figure((self.Figno+1)).gca()
        line,=ax.plot(0,0,':')
        self.mbutton = mouse_button(line,ax,self.Fs)
        return
    
    def compose_parallel(self,n,d):
        self.num  = np.convolve(self.num,d) + np.convolve(n,self.dnum)
        self.dnum = np.convolve(self.dnum,d) 
        
    def discrete_2ndOrder_pole(self,zeta,f,Ts):
        #Compose S-domain pole (complex)
        wns  = 2.*pi*f*Ts
        a1   = -zeta*wns-1j*wns*sqrt(1.-zeta**2)
        #Convert S-domain pole to Z-domain (A simple transformation scheme)
        pole = exp(a1)
        return pole
    
    def discrete_2ndOrder_polynoom(self,zeta,f,Ts):
        #Compose S-domain pole (complex)
        wns  = 2.*pi*f*Ts
        a1   = -2.*exp(-zeta*wns)*cos(wns*sqrt(1.-zeta**2))
        a2   = exp(-2.*zeta*wns)
        b1   = exp(-zeta*wns)*sin(wns*sqrt(1.-zeta**2))
        b2   = 0
        #Convert S-domain pole to Z-domain (A simple transformation scheme)
        dnum = np.array([1,a1,a2])
        num  = np.array([1,b1,b2])
        Gain = sum(num)/sum(dnum)
        num  /= Gain
        return num,dnum,Gain
        
    
    def butter_highpass_filter(self,Fo,Fs):
         order = 4
    
         normal_cutoff = Fo / Fs
         b, a = butter(order, normal_cutoff, btype='highpass', analog=False,output='ba')
         return b,a
    
  