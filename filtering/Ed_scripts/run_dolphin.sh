#!/bin/bash

#sox ../1678020614.180210165811.wav output.wav trim 0 5 : newfile : restart
sox 1678020614.180210165811.wav output.wav trim 0 20

python3 match_dolphin.py ../Neutrino_data_files/neutrino_6.0_300_1_11.dat output.wav match

