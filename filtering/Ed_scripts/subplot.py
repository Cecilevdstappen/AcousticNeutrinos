#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:13:13 2022

@author: gebruiker
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

L_filename = []
for i in np.arange(20,110,10):
    filename = 'zpos'+str(i)+'_signal.png'
    L_filename.append(filename)

for i in range(0,len(L_filename)-1):
    img = mpimg.imread(filename[i])
    fig, axs = plt.subplots(9, 0)
    axs[i, 0].plot(img)
    axs[i, 0].set_title(str(filename[i]))
    fig.tight_layout()
    