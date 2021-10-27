from math import *
import numpy as np
import numpy.fft as fft
import sys
from scipy import signal
import matplotlib.pyplot as plt

#scaling = pow(10, 168./20)                                                                                                                                                          
sampling_rate = 144000.

def write_out(bipfilename, scaling, bmax, bmin):
    print("Generated file: ", bipfilename, scaling, bmax, bmin)

def rename_me(filename, scaling):
    return filename.rsplit('.', 1)[0] + \
        '_' + str(scaling)+ '.wav'

def resampled_signal(filename):
    time, bip = np.loadtxt(filename, usecols=(0,1), unpack=True)

    Fs = 1/(time[1] - time [0])
    Fs_resampled = 144000.

    number_resampled = int(round(len(bip)*Fs_resampled/Fs))

    bip_resampled = signal.resample(bip, number_resampled)
    t_resampled   = np.linspace(0, len(bip_resampled)/Fs_resampled,
                                len(bip_resampled))
    return t_resampled, bip_resampled

def waveclip(bipfilename, scaling):
    np.random.seed(scaling)
    from scipy.io.wavfile import read
    noisefile = "spermwhale.wav"
    time_trace = read(noisefile)
    sampling_rate = time_trace[0]
    print (sampling_rate)
    time_series   = time_trace[1]

    # part of the read file of particular length                                                                                                                                     
    trace_length = pow(2,20)
    trace_start = np.random.randint(0,len(time_series) - 100000, 1)[0]
    trace_end = trace_start + trace_length

    # determine a noise realisation,                                                                                                                                                 
    # based on the FFT of the data that have been read                                                                                                                               
    # using different, random phase.                                                                                                                                                 
    # Used polar coordinates for complex numbers                                                                                                                                     
    y = time_series[int(trace_start) : int(trace_end)]
    Y = np.fft.fft(y)
    
    m = np.abs(Y)
    phase = 2*pi*np.random.rand(len(m)) - pi
    Z = m*np.cos(phase) + 1j*m*np.sin(phase)
    z = np.fft.ifft(Z)
    z = z.real

    # read ascii file with the neutrino click                                                                                                                                        
    time, bip  = resampled_signal(bipfilename)

    # padding to insert the neutrino click somewhere                                                                                                                                 
    # (random position) in the data stream                                                                                                                                           
    # entry_point = np.random.randint(0,len(z) - 20000, 1)[0]
    entry_point = int(len(z)/2)
    bip *= scaling
    x = np.pad(bip, (entry_point, len(z)-len(bip)-entry_point),
               'constant', constant_values=(0., 0.))

    # add noise abd signal and make dure that it of proper format                                                                                                                    
    neutrino_noise_array = (x+z) + np.mean(y) * np.ones(len(z))
    neutrino_noise_array = np.asarray(neutrino_noise_array, dtype=np.int16)

#    np.savetxt(rename_me(bipfilename, scaling).rsplit('.', 1)[0] + '.txt', \                                                                                                        
#               neutrino_noise_array)                                                                                                                                                
    write_out(bipfilename, scaling, bip.max(), bip.min())
    # write to wave file                                                                                                                                                             
    from scipy.io.wavfile import write
    write(rename_me(bipfilename, scaling),
          sampling_rate,
          neutrino_noise_array)

#-------------------------------                                                                                          
print(sys.argv, len(sys.argv))
import os
def main(argv):

    do_plot = True         #plot pressure vs time pulse                                                                                                                             
    do_pulse_file = True   #create file pressure vs time pulse                                                                                                                      
    wave_clip = True      #create audio clip pulse                                                                                                                                 

    #shower energies we want to explore                                                                                                                                             
    energies= np.arange(11,12,1) #exponent energy (in GeV)                                                                                                                         
    facts_energy = np.asarray([1]) #multiplicative factor energy                                                                                                                    

    #hydrophone positions we want to explore                                                                                                                                        
    hydrophone_rpos = [1000] #in meters, r=0 axis shower                                                                                                                            
    hydrophone_zpos = [12] #in meters, z=0 interaction point                                                                                                                         

    #shower energy loop                                                                                                                                                             
    for idx_e,energy in enumerate(energies):

        for fact_energy in facts_energy:
            print(fact_energy,'E',energy)

            #file containing the energy deposition per longitudinal and radial bin (corsika output)                                                                                 
            #input_file = 'dEdzdr_DAT{:02d}{:02d}_hadr.dat'.format(fact_energy,energy)
            input_file = 'cs_mean_dEdzdr_11106_150.dat' 
            for rpos in hydrophone_rpos:
                for zpos in hydrophone_zpos:

                        #create pressure vs time file                                                                                                                               
                        if do_pulse_file == True:

                            file_name = 'neutrino_' + str(round(float(zpos),2)) + '_' + str(rpos) + '_' + str(fact_energy) + '_' + str(energy)  + '.dat'
                            #file_name = 'neutrino_' + '_' + str(rpos) + '_' + str(fact_energy) + '_' + str(energy)  + '.dat'
                            print (file_name)
                            commandlinestring = 'octave -q one_pulse.m ' + str(round(float(zpos),2)) + \
                            ' ' + str(rpos) + ' ' + 'neutrino_' + ' ' + input_file + ' ' + str(energy) + ' ' + str(fact_energy)
                            os.system(commandlinestring)


                        #create pressure vs time plot                                                                                                                               
                        if do_plot == True:

                            f = open(file_name,'r')
                            lines = f.readlines()
                            amplitude = [] #Pa                                                                                                                                      
                            time = [] #s                                                                                                                                            

                            for line in lines:
                                if line.startswith("#") or len(line) <= 1: continue
                                else: amplitude.append(float(line.split(' ')[2])), time.append(float(line.split(' ')[1]))
                            f.close()

                            c_s = 1500 # speed of sound (m/s)                                                                                                                       
                            time_array = np.asarray(time)*(10**6)/5 #+rpos/c_s)*(10**6)/5
                            amplitude_array = np.asarray(amplitude)*(10**3) #from Pa to nP, mPa... depending on energy                                                              

                            plt.plot(time_array[1:len(time_array)-1],amplitude_array[1:len(time_array)-1])

                            plt.title("$z_{obs}=$"+str(round(zpos,1))+"$m, r_{obs}=$"+str(round(rpos,1))+"$m$",loc='right')
                            plt.ylabel("amplitude (mPa)")
                            plt.xlabel("arrival time (s)")
                            plt.grid(which='both', axis='y')
                            plt.grid(which='both', axis='x')
                            plt.savefig('acoustic_plot' + str(round(float(zpos),2)) + '_' + str(rpos) + '_' + str(fact_energy) + '_' + str(energy)  + '.png', dpi=200)
                            plt.show()
                            plt.close()


                        #create audio clip of the pulse                                                                                                                             
                        for scaling in range(100, 200, 100):
                            #generate wave clip                                                                                                                                     
                            if wave_clip == True:
                                pulse_file =  'neutrino_' +  str(round(float(zpos),0)) + '_' + str(rpos) + '_' + str(fact_energy) + '_' + str(energy) + '.dat'
                                waveclip(pulse_file, scaling*100000)


if __name__ == '__main__':
    main(sys.argv)




