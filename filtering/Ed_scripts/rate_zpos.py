#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:13:25 2022

@author: gebruiker
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:06:29 2022

@author: gebruiker
"""
import numpy as np
import matplotlib.pyplot as pp
import matplotlib
import pickle
from my_styles import *

set_paper_style()

plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')

plot_hist_dt = True
plot_hist_SNR = True
plot_2Dhist = True
scatter = True

rpos = 300
zpos = 6
for energy in np.arange(9,13,1):  
    L_zpos = []
    L_rate = []
    for zpos in np.arange(0,20,0.1):
        L_zpos.append(zpos)
        with open('/home/gebruiker/AcousticNeutrinos/filtering/Neutrino_data_files/deltat_SNR_'+str(rpos)+'_'+str(round(float(zpos),1))+'_'+str(energy)+'.dat', 'rb') as handle:
            b = pickle.load(handle)
        delta_t = np.abs(b["delta_t"])
        SNR = b["snr"]
        SNR = np.array(SNR)
        counts, bin_edges = np.histogram(abs(delta_t), bins=100)
        print(counts)
        
        count = 0
        for i in np.arange(0,len(delta_t),1):
            if np.abs(delta_t[i]) < 0.05:
                count +=1
        L_rate.append(count/10)
        


    #pp.hist2d(delta_t,SNR, bins = 50)
    #pp.scatter(delta_t,SNR)# range = (-0.8,-0.7))
    #pp.color(delta_t,SNR)
    fig, ax = pp.subplots()
    #if plot_hist_dt == True:
    pp.figure(1, figsize=(6,5))
    pp.plot(L_zpos, L_rate, '-', label = 'E = 1e'+str(energy))
    pp.title('r = '+ str(rpos)+' m, sea state 0') #E = [1e9,1e13] GeV,
    pp.grid('on',which='both',axis='x')
    pp.grid('on',which='both',axis='y')
    pp.xlabel('z (m)')
    pp.ylabel('percentage good reconstruction (%)')
    pp.ylim(0,101)
    #pp.yscale('log')
pp.legend()
pp.savefig('/home/gebruiker/Pictures/Overleaf/rate_zpos.png')
pp.show()
    

#if plot_2Dhist == True:
#    pp.figure(4, figsize=(6,5))
#    pp.hist2d(L_zpos, L_rate, bins = 100, norm = matplotlib.colors.LogNorm(),label = 'E = 1e'+str(energy))
#    pp.title('r = '+ str(rpos)+' m, sea state 0') #E = [1e9,1e13] GeV,
#    pp.grid('on',which='both',axis='x')
#    pp.grid('on',which='both',axis='y')
#    pp.xlabel('z (m)')
#    pp.ylabel('percentage good reconstruction (%)')
#    #pp.yscale('log')
#    pp.savefig('/home/gebruiker/Pictures/Overleaf/2dhist_rate_zpos.png')
#    pp.show()

