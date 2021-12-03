# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:28:52 2021

@author: edopp
"""
import math
from numpy.linalg import norm
from numpy.fft import fft
from numpy import cos
from numpy import pi
import numpy as np
from scipy import signal
import time
import matplotlib.pyplot as plt

#Define the Impulse class
class impulse(object):
    #Declare class members and initialise 
    def __init__(self,*args):
        self.Fs         = 1024
        self.length     = 102400 
        self.Nchnl      = 2 
        nargin          = len(args)
        self.refr       = 1 
        self.resp       = 0
        self.F0         = 63.
        self.norm       = 1.
        self.NFFT       = 4096
        self.NFFT2      = 2*self.NFFT
        self.ZEROPAD    = 4096
        self.Segment    = 100
        self.PSDOn      = 0
        self.WindowType = 'hanning'
        self.window     = np.hamming(self.NFFT)
        self.Data       = np.zeros((self.length,self.Nchnl))
        self.Impulse    = np.zeros((self.NFFT,self.Nchnl-1))
        self.freq       = []
        self.time       = []
        self.PSD        = np.ones((int(self.NFFT/2),self.Nchnl), dtype=float)
        self.Sx         = np.zeros((self.NFFT,self.Nchnl), dtype=complex)
        self.Sy         = np.zeros((self.NFFT,self.Nchnl), dtype=complex)
        self.Sxy        = np.zeros((self.NFFT,self.Nchnl-1), dtype=complex)
        self.Txy        = np.zeros((self.NFFT,self.Nchnl-1), dtype=complex)
        self.Sxx        = np.zeros((self.NFFT,self.Nchnl), dtype=float)
        self.Syy        = np.zeros((self.NFFT,self.Nchnl), dtype=float)
        self.Cxy        = np.zeros((self.NFFT,self.Nchnl-1), dtype=float)
        self.gg         = np.ones((self.Nchnl), dtype=float)
        self.Gain       = np.ones((self.Nchnl), dtype=float)
        self.Sens       = np.ones((self.Nchnl), dtype=float)
        self.Unit       = ['V']
        self.Name       = ['P']        
        self.GraphLabel = ['Exp']
        self.Title      = ['Graph']
        for nchl in range(1,self.Nchnl):
            self.Unit       += ['V']
            self.Name       += ['P']            
            self.GraphLabel += ['Exp']
            self.Title      += ['Graph']
 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #class members when the class instance is copied
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
    #Pretty print the class members
    def __repr__(self):
        Rspace = 30;
        str3  ="Fs:".rjust(Rspace)+" %d\n".ljust(0) %self.Fs
        str3 += "NFFT:".rjust(Rspace)+" %d\n".ljust(0) %self.NFFT
        str3 += "NFFT2:".rjust(Rspace)+" %d\n".ljust(0) %self.NFFT2
        str3 += "ZEROPAD:".rjust(Rspace)+" %d\n".ljust(0) %self.ZEROPAD
        str3 += "Segment:".rjust(Rspace)+" %d\n".ljust(0) %self.Segment
        str3 += "PSDOn:".rjust(Rspace)+" %d\n".ljust(0) %self.PSDOn
        str3 += "length:".rjust(Rspace)+" %d\n".ljust(0) %self.length
        str3 += "Nchnl:".rjust(Rspace)+" %d\n".ljust(0) %self.Nchnl
        str3 += "refr:".rjust(Rspace)+" %d\n".ljust(0) %self.refr
        str3 += "resp:".rjust(Rspace)+" %d\n".ljust(0) %self.resp
        str3 += "WindowType:".rjust(Rspace)+" '%s'\n".ljust(0) %self.WindowType
        str3 += "window:".rjust(Rspace)+" [%d]\n".ljust(0) % np.size(self.window)
        str3 += "Data:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Data))       
        str3 += "time:".rjust(Rspace)+" [%d]\n".ljust(0) % np.size(self.time)
        str3 += "freq:".rjust(Rspace)+" [%d]\n".ljust(0) % np.size(self.freq)
        str3 += "PSD:".rjust(Rspace)+" [%s]\n".ljust(0) %str(np.shape(self.PSD))
        str3 += "Sx:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sx))
        str3 += "Sy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sy))
        str3 += "Sxx:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sx))
        str3 += "Syy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sy)) 
        str3 += "Sxy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sxy))
        str3 += "Txy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Txy))
        str3 += "Cxy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Cxy))
        str3 += "Impulse:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Impulse))
        str3 += "GraphLabel:".rjust(Rspace)+" [%s]\n".ljust(0) % self.GraphLabel
        str3 += "Unit:".rjust(Rspace)+" [%s]\n".ljust(0) % self.Unit
        str3 += "Name:".rjust(Rspace)+" [%s]\n".ljust(0) % self.Name
        str3 += "Title:".rjust(Rspace)+" [%s]\n".ljust(0) % self.Title
        str3 += "Gain:".rjust(Rspace)+"[%s]\n".ljust(0) % str(self.Gain)
        str3 += "Sens:".rjust(Rspace)+"[%s]\n".ljust(0) % str(self.Sens)

        return str3;        
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
#Declare class methods
def ImpulseResponse(self, *args):
    nargin          = len(args)
    self.refr       = 0
    self.resp       = 1
    if nargin >= 1:
        self.refr = args[0]
    if nargin >= 2:
        self.resp = args[1]          
    if self.PSDOn == 1: 
        self.NFFT = self.Fs 
        self.ZEROPAD  = self.NFFT
    #Data should have a column orientated shape
    r,k = self.Data.shape
    if r < k: 
        self.Data = np.transpose(self.Data)
    #Check paramters and make a Hamming window    
    self.NFFT2    = self.NFFT + self.ZEROPAD
    self.length   = np.size(self.Data,0)     
    self.window   = np.hamming(self.NFFT2)
    #Some useful vectors
    self.freq     = np.linspace(0,int(self.NFFT2/2)-1,int(self.NFFT2/2))*float(self.Fs)/float(self.NFFT2)
    self.time     = np.arange(0,self.length,1)/self.Fs 
    #Point to the response and reference data
    Resp          = self.Data[:,self.resp]
    Ref           = self.Data[:,self.refr]      
    #Determine the number of data segments
    start_index   = 0
    stop_index    = self.NFFT+start_index
    self.Segment  = int(math.floor((len(Ref)-start_index)/self.NFFT))
    #Pre-allocated power spectra vectors
    self.Sxx      = np.zeros((self.NFFT2), dtype=float)
    self.Syy      = np.zeros((self.NFFT2), dtype=float)
    self.Sxy      = np.zeros((self.NFFT2), dtype=complex)
    #Estimate the power spectra's of the reference and response data
    for i in range(self.Segment):
       #For each data segment and average
       respData  = (signal.detrend(Resp[i*self.NFFT+start_index:i*self.NFFT+stop_index]))
       refData   = (signal.detrend(Ref[i*self.NFFT+start_index:i*self.NFFT+stop_index]))
       zeros     = np.zeros(np.shape(respData))
       respData  = self.window*(np.append(respData,zeros))
       refData   = self.window*(np.append(refData,zeros))
       #Estimate the the complex spectra of the reference and the response data  
       self.Sx   = (2./self.NFFT2)*np.fft.fft(refData)
       self.Sy   = (2./self.NFFT2)*np.fft.fft(respData)
       #Calculate the the power spectra of the reference and response data 
       self.Syy += np.absolute(self.Sy)**2
       self.Sxx += np.absolute(self.Sx)**2
       #Calculate the cross power spectrum (Summation. The response w.r.t. the reference)
       self.Sxy += self.Sy*np.conjugate(self.Sx)
       
    #Normalizing scale factor for window type 
    # and average the Data   
    self.norm   = norm(self.window)**2
    WinNorm     = self.Segment*self.norm
    #Scale Power spectra    
    self.Sxx    /= WinNorm
    self.Syy    /= WinNorm
    self.Sxy    /= WinNorm
    #Calculate the FRF (Frequency Response Function)
    self.Txy     = self.Sxy/(self.Sxx+1e-40)
    #Calculate the coherence frequency function
    self.Cxy     = np.absolute((self.Sxy)**2/(self.Sxx*self.Syy))
    # DC -component has no useful info
    self.Cxy[0]  = 0 
    #Calculate the non-circular impulse response 
    self.Impulse = np.fft.ifft(self.Txy)
    #Not used!!!. self.Impulse = np.fft.ifftshift(self.Impulse)
    return

def discrete_2ndOrder_pole(self,zeta,f,Ts):
    #Compose S-domain pole (complex)
    wns  = 2.*pi*f*Ts
    a1   = -zeta*wns-1j*wns*np.sqrt(1.-zeta**2)
    #Convert S-domain pole to Z-domain (A simple transformation scheme)
    pole = np.exp(a1)
    return pole

def test(self,zeta,f):
    #A quick simple and neat test
    Ts   = 1./self.Fs
    b    = [1,0,0]
    pole = discrete_2ndOrder_pole(self,zeta,f,Ts)
    a    = np.poly([pole,np.conjugate(pole)])
    
    vec       = 0.5-np.random.rand(self.length)
    vecf      = signal.lfilter(b,a,vec)
    self.Data = np.squeeze([vec,vecf])
    ImpulseResponse(self)
    plt.figure(10);plt.clf();plt.stem(np.real(self.Impulse))