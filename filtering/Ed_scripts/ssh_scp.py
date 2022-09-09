#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:28:52 2022

@author: gebruiker
"""
import numpy as np
import os
remotehost = "ceciles@login.nikhef.nl"
#remotefile_dat = "/project/antares/cstappen/files/polys/scripts_cstappen_acousticpulse/neutrino_"+str(np.round(float(z_pos),1))+"_"+str(r_pos)+"_1_"+str(energy)+".dat"
#remotefile_txt = "/project/antares/cstappen/files/polys/scripts_cstappen_acousticpulse/neutrino_"+str(round(float(z_pos),1))+"_"+str(r_pos)+"_1_"+str(energy)+"_"+str(scaling)+"ss2.txt"
remote_file = "/project/antares/cstappen/files/polys/scripts_cstappen_acousticpulse/just_one_neutrino.py"
#os.system('scp %s:%s .' % (remotehost, remotefile_dat) )
os.system('scp %s:%s .' % (remotehost, remote_file) )