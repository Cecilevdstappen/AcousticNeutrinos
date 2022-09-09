#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 01:07:04 2022

@author: gebruiker
"""
import pickle

with open('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/deltat_SNR_300_6_9.dat', 'rb') as handle:
    b = pickle.load(handle)
print(b["delta_t"])