#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:57:49 2022

@author: gebruiker
"""
from scipy import signal

from scipy.fft import fft, fftshift
import numpy as np
import matplotlib.pyplot as plt
from my_styles import *

set_paper_style()
plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')
window = signal.windows.hann(51)
plt.figure(1, figsize=(6,5))
plt.plot(window)

plt.title("Hann window")

plt.ylabel("Amplitude")

plt.xlabel("Sample")

plt.show()

A = fft(window, 2048) / (len(window)/2.0)

freq = np.linspace(-0.5, 0.5, len(A))

response = np.abs(fftshift(A / abs(A).max()))

response = 20 * np.log10(np.maximum(response, 1e-10))

plt.plot(freq, response)

plt.axis([-0.5, 0.5, -120, 0])

plt.title("Frequency response of the Hann window")

plt.ylabel("Normalized magnitude [dB]")

plt.xlabel("Normalized frequency [cycles per sample]")
plt.show()
