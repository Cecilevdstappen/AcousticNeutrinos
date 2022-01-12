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
        self.imp_Fs         = 1024
        self.imp_length     = 102400 
        self.imp_Nchnl      = 2 
        nargin          = len(args)
        self.imp_refr       = 1 
        self.imp_resp       = 0
        self.imp_F0         = 63.
        self.imp_norm       = 1.
        self.imp_NFFT       = 4096
        self.imp_NFFT2      = 2*self.imp_NFFT
        self.imp_ZEROPAD    = 4096
        self.imp_Segment    = 100
        self.imp_PSDOn      = 0
        self.imp_WindowType = 'hanning'
        self.imp_window     = np.hamming(self.imp_NFFT)
        self.imp_Data       = np.zeros((self.imp_length,self.imp_Nchnl))
        self.imp_Impulse    = np.zeros((self.imp_NFFT,self.imp_Nchnl-1))
        self.imp_freq       = []
        self.imp_time       = []
        self.imp_PSD        = np.ones((int(self.imp_NFFT/2),self.imp_Nchnl), dtype=float)
        self.imp_Sx         = np.zeros((self.imp_NFFT,self.imp_Nchnl), dtype=complex)
        self.imp_Sy         = np.zeros((self.imp_NFFT,self.imp_Nchnl), dtype=complex)
        self.imp_Sxy        = np.zeros((self.imp_NFFT,self.imp_Nchnl-1), dtype=complex)
        self.imp_Txy        = np.zeros((self.imp_NFFT,self.imp_Nchnl-1), dtype=complex)
        self.imp_Sxx        = np.zeros((self.imp_NFFT,self.imp_Nchnl), dtype=float)
        self.imp_Syy        = np.zeros((self.imp_NFFT,self.imp_Nchnl), dtype=float)
        self.imp_Cxy        = np.zeros((self.imp_NFFT,self.imp_Nchnl-1), dtype=float)
        self.imp_gg         = np.ones((self.imp_Nchnl), dtype=float)
        self.imp_Gain       = np.ones((self.imp_Nchnl), dtype=float)
        self.imp_Sens       = np.ones((self.imp_Nchnl), dtype=float)
        self.imp_Unit       = ['V']
        self.imp_Name       = ['P']        
        self.imp_GraphLabel = ['Exp']
        self.imp_Title      = ['Graph']
        for nchl in range(1,self.imp_Nchnl):
            self.imp_Unit       += ['V']
            self.imp_Name       += ['P']            
            self.imp_GraphLabel += ['Exp']
            self.imp_Title      += ['Graph']
 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #class members when the class instance is copied
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
    #Pretty print the class members
    def __repr__(self):
        Rspace = 30;
        str3  ="Fs:".rjust(Rspace)+" %d\n".ljust(0) %self.imp_Fs
        str3 += "NFFT:".rjust(Rspace)+" %d\n".ljust(0) %self.imp_NFFT
        str3 += "NFFT2:".rjust(Rspace)+" %d\n".ljust(0) %self.imp_NFFT2
        str3 += "ZEROPAD:".rjust(Rspace)+" %d\n".ljust(0) %self.imp_ZEROPAD
        str3 += "Segment:".rjust(Rspace)+" %d\n".ljust(0) %self.imp_Segment
        str3 += "PSDOn:".rjust(Rspace)+" %d\n".ljust(0) %self.imp_PSDOn
        str3 += "length:".rjust(Rspace)+" %d\n".ljust(0) %self.imp_length
        str3 += "Nchnl:".rjust(Rspace)+" %d\n".ljust(0) %self.imp_Nchnl
        str3 += "refr:".rjust(Rspace)+" %d\n".ljust(0) %self.imp_refr
        str3 += "resp:".rjust(Rspace)+" %d\n".ljust(0) %self.imp_resp
        str3 += "WindowType:".rjust(Rspace)+" '%s'\n".ljust(0) %self.imp_WindowType
        str3 += "window:".rjust(Rspace)+" [%d]\n".ljust(0) % np.size(self.imp_window)
        str3 += "Data:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.imp_Data))       
        str3 += "time:".rjust(Rspace)+" [%d]\n".ljust(0) % np.size(self.imp_time)
        str3 += "freq:".rjust(Rspace)+" [%d]\n".ljust(0) % np.size(self.imp_freq)
        str3 += "PSD:".rjust(Rspace)+" [%s]\n".ljust(0) %str(np.shape(self.imp_PSD))
        str3 += "Sx:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.imp_Sx))
        str3 += "Sy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.imp_Sy))
        str3 += "Sxx:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.imp_Sx))
        str3 += "Syy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.imp_Sy)) 
        str3 += "Sxy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.imp_Sxy))
        str3 += "Txy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.imp_Txy))
        str3 += "Cxy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.imp_Cxy))
        str3 += "Impulse:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.imp_Impulse))
        str3 += "GraphLabel:".rjust(Rspace)+" [%s]\n".ljust(0) % self.imp_GraphLabel
        str3 += "Unit:".rjust(Rspace)+" [%s]\n".ljust(0) % self.imp_Unit
        str3 += "Name:".rjust(Rspace)+" [%s]\n".ljust(0) % self.imp_Name
        str3 += "Title:".rjust(Rspace)+" [%s]\n".ljust(0) % self.imp_Title
        str3 += "Gain:".rjust(Rspace)+"[%s]\n".ljust(0) % str(self.imp_Gain)
        str3 += "Sens:".rjust(Rspace)+"[%s]\n".ljust(0) % str(self.imp_Sens)

        return str3;        
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
#Declare class methods
def ImpulseResponse(self, *args):
    nargin          = len(args)
    self.imp_refr       = 0
    self.imp_resp       = 1
    if nargin >= 1:
        self.imp_refr = args[0]
    if nargin >= 2:
        self.imp_resp = args[1]          
    if self.imp_PSDOn == 1: 
        self.imp_NFFT = self.imp_Fs 
    self.imp_ZEROPAD  = self.imp_NFFT
    #Data should have a column orientated shape
    r,k = self.imp_Data.shape
    if r < k: 
        self.imp_Data = np.transpose(self.imp_Data)
    #Check paramters and make a Hamming window    
    self.imp_NFFT2    = self.imp_NFFT + self.imp_ZEROPAD
    self.imp_length   = np.size(self.imp_Data,0)     
    self.imp_window   = np.blackman(self.imp_NFFT2)
    #Some useful vectors
    self.imp_freq     = np.linspace(0,int(self.imp_NFFT2/2)-1,int(self.imp_NFFT2/2))*float(self.imp_Fs)/float(self.imp_NFFT2)
    self.imp_time     = np.arange(0,self.imp_length,1)/self.imp_Fs 
    #Point to the response and reference data
    Resp          = self.imp_Data[:,self.imp_resp]
    Ref           = self.imp_Data[:,self.imp_refr]      
    #Determine the number of data segments
    start_index   = 0
    stop_index    = self.imp_NFFT+start_index
    self.imp_Segment  = int(math.floor((len(Ref)-start_index)/self.imp_NFFT))
    #Pre-allocated power spectra vectors
    self.imp_Sxx      = np.zeros((self.imp_NFFT2), dtype=float)
    self.imp_Syy      = np.zeros((self.imp_NFFT2), dtype=float)
    self.imp_Sxy      = np.zeros((self.imp_NFFT2), dtype=complex)
    #Estimate the power spectra's of the reference and response data
    for i in range(self.imp_Segment):
       #For each data segment and average
       respData  = (signal.detrend(Resp[i*self.imp_NFFT+start_index:i*self.imp_NFFT+stop_index]))
       refData   = (signal.detrend(Ref[i*self.imp_NFFT+start_index:i*self.imp_NFFT+stop_index]))
       zeros     = np.zeros(np.shape(respData))
       respData  = self.imp_window*(np.append(respData,zeros))
       refData   = self.imp_window*(np.append(refData,zeros))
       #Estimate the the complex spectra of the reference and the response data  
       self.imp_Sx   = (2./self.imp_NFFT2)*np.fft.fft(refData)
       self.imp_Sy   = (2./self.imp_NFFT2)*np.fft.fft(respData)
       #Calculate the the power spectra of the reference and response data 
       self.imp_Syy += np.absolute(self.imp_Sy)**2
       self.imp_Sxx += np.absolute(self.imp_Sx)**2
       #Calculate the cross power spectrum (Summation. The response w.r.t. the reference)
       self.imp_Sxy += self.imp_Sy*np.conjugate(self.imp_Sx)
       
    #Normalizing scale factor for window type 
    # and average the Data   
    self.imp_norm   = norm(self.imp_window)**2
    WinNorm     = self.imp_Segment*self.imp_norm
    #Scale Power spectra    
    self.imp_Sxx    /= WinNorm
    self.imp_Syy    /= WinNorm
    self.imp_Sxy    /= WinNorm
    #Calculate the FRF (Frequency Response Function)
    self.imp_Txy     = self.imp_Sxy/(self.imp_Sxx+1e-40)
    #Calculate the coherence frequency function
    self.imp_Cxy     = np.absolute((self.imp_Sxy)**2/(self.imp_Sxx*self.imp_Syy))
    # DC -component has no useful info
    self.imp_Cxy[0]  = 0 
    #Calculate the non-circular impulse response 
    self.imp_Impulse = np.fft.ifft(self.imp_Txy)
    #Not used!!!. self.imp_Impulse = np.fft.ifftshift(self.imp_Impulse)
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
    Ts   = 1./self.imp_Fs
    b    = [1,0,0]
    pole = discrete_2ndOrder_pole(self,zeta,f,Ts)
    a    = np.poly([pole,np.conjugate(pole)])
    
    vec       = 0.5-np.random.rand(self.imp_length)
    vecf      = signal.lfilter(b,a,vec)
    self.imp_Data = np.squeeze([vec,vecf])
    ImpulseResponse(self)
    plt.figure(10);plt.clf();plt.stem(np.real(self.imp_Impulse))