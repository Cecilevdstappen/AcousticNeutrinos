from matplotlib import pyplot as plt
import numpy as np
from numpy import *

class mouse_button:
    def __init__(self, line,ax,Fs):
        self.line = line
        self.ax   = ax
        self.Fs   = Fs
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())

    def __call__(self, event):
        self.get_pzinfo(event)
        if event.inaxes!=self.line.axes: return   

    def on(self):
        self.cid = self.line.figure.canvas.mpl_connect('button_press_event', self)
        
    def off(self):
        if hasattr(self,'cid'):
            self.line.figure.canvas.mpl_disconnect(self.cid)
        try:
            if hasattr(self,'tt'):
                self.tt.remove()
        except:
            print('No label')
        
    def get_pzinfo(self,event):
        x = event.xdata
        y = event.ydata
        self.ts    = 1./self.Fs
        self.u     = x+1j*y;
        #Calculate properties of a 2-nd order AR-process
        self.zeta  = -np.cos(np.angle(np.log(self.u)));
        self.Freq  = abs(np.imag(np.log((self.u))))/self.ts/(2.*np.pi); # natural resonance self.Frequency
        self.Freq  = self.Freq*(np.sqrt(1-self.zeta**2)); # damped resonance frequency
        self.Tc    = 1./(2.*np.pi*self.Freq*self.zeta);
        #Calculate |H| with Freqz at frequency Freq
        self.AR    = np.poly([self.u,np.conjugate(self.u)]);
        self.rw    = exp(1j*2.*pi*self.Freq*self.ts);
        self.ex    = (0,1,2);
        self.rwvec = self.rw**self.ex;
        self.resp  = 1./abs(self.AR*self.rwvec);
        self.textstr = '\n'.join((
            r'pole=%.2f + j%.2f' % (x,y, ),
            r'|H|=%.2f' % (abs(self.u), ),
            r'Gain= %.4f' %(1./(1.-abs(self.u)), ),
            r'F0=%.2f Hz' % (self.Freq, ),
            r'Zeta=%.2f' %(self.zeta,)))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        
        # place a text box in upper left in axes coords
        #
        try:
            if hasattr(self,'tt'):
                self.tt.remove()
        except:
            print('No label')
                
        self.tt=self.ax.text(0.05,0.95, self.textstr, transform=self.ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props) #transform=self.ax.transAxes,
        plt.draw()
        # plt.show()

# Fs = 10000.
# # plt.figure(11)
# ax = plt.figure(11).gca()
# line,=ax.plot(0,0,':')
# mc = mouse_button(line,ax,Fs)
# plt.show()