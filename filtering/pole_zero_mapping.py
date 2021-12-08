from scipy.io import savemat
import numpy as np
import control.matlab as cnt
import matplotlib.pyplot as plt
import sympy
import sys
import scipy
import itertools
from sippy import functionset as fset


#read in transfer function file
filename = 'Bodemat_Dubbel_1FF_Actran_Sweep200Hz20kHz0degree_AfterPressure_v2.dat'
f = open(filename,'r')
lines = f.readlines()
freq = [] #s
y_im = []

#for line in lines:
#    if line.startswith("#") or len(line) <= 1: continue
#    else: freq.append(float(line.split(' ')[0])), y_im.append(np.imag(line.split(' ')[1]))

dtypes = [('col0', float)]+[('col1', np.complex128)]
data = np.loadtxt(filename,  usecols=(0,1), dtype=dtypes, unpack=True)#,  dtype='float,np.complex128)
tenthlines = itertools.islice(data[0], 0, None, 20)
y_tenthlines = itertools.islice(data[1],0,None,20)
for line in tenthlines:
    freq.append(line)
freq=np.asarray(freq)
for line in y_tenthlines:
    y_im.append(line)
y_im = np.asarray(y_im)
#freq=data[0]
#y_im=data[1]

print(freq.size)
print(y_im.size)
#read in signal file
filename_sig = 'neutrino_12.0_1000_1_11.dat'
f = open(filename_sig,'r')
lines = f.readlines()
time = [] #s
amplitude = []
dtypes = [('col0', float)]+[('col1', np.complex128)]
data = np.loadtxt(filename_sig,  usecols=(0,1), dtype=dtypes, unpack=True)#,  dtype='float,np.complex128)
time = data[0]
amplitude = data[1]
norm = np.linalg.norm(amplitude)
amplitude = amplitude/norm
#amplitude = np.roll(amplitude,-1000)


amplitude_f = np.fft.fft(amplitude)
norm = np.linalg.norm(amplitude_f)
amplitude_f = amplitude/norm
#amplitude_f.normalize()

white_noise_variance = [0.0001]
white_noise = fset.white_noise_var(y_im.size, white_noise_variance)[0]
white_noise_f = np.fft.fft(white_noise) 

transfer_function_t = np.fft.ifft(y_im, n=2*len(y_im))

y_im.resize(amplitude_f.size)
transfer_function_t.resize(amplitude.size)
#convolved_tf_t = scipy.signal.convolve(amplitude, transfer_function_t, mode='full', method='auto')
#convolved_tf_f = scipy.signal.convolve(amplitude_f, y_im, mode='full', method='auto')
convolved_tf_f = (amplitude_f * y_im)
convolved_tf_t = np.fft.ifft(convolved_tf_f)
norm = np.linalg.norm(convolved_tf_t)
convolved_tf_t = convolved_tf_t/norm

norm = np.linalg.norm(convolved_tf_f)
convolved_tf_f = convolved_tf_f/norm

#convolved_tf_f.normalize()
#Create transfer function from coeff
#transfer_freq = cnt.tf(num, den)
#transfer_freq = np.fft.ifft(transfer_freq, n=2*len(y_im))

plt.figure(1)
plt.plot(np.real(y_im), label='transferfunction')
#plt.plot(white_noise_f, label='white noise')
plt.plot(np.real(convolved_tf_f), label ="convolved")
plt.xlabel("frequency")
plt.ylabel(" Amplitude")
plt.title("Convolution transferfunction and white noise")
plt.legend()
plt.grid()
plt.show()


plt.figure(2)
plt.plot(np.real(transfer_function_t), label='transferfunction')
plt.plot(np.real(amplitude), label = ' signal' )
#plt.plot(white_noise, label='white noise')
plt.plot(convolved_tf_t, label ="convolved")
plt.xlabel("time")
plt.ylabel(" Amplitude")
#plt.xlim(0,1200)
plt.title("Convolution transferfunction and white noise time trace")
plt.legend()
plt.grid()
plt.show()


#Poles is the factorized denumerator array and zeros is the numerator array
poles,zeros = cnt.pzmap(transfer_function)

#play with poles and zeroes values and see how it changes the transferfunction
for i,j in np.arange(0,5,0.5):
    test_poles = i*poles
    test_zeros = i*zeros
    expand_zeroes = sympy.expand(test_zeros)
    expand_poles = sympy.expand(test_poles)
    test_transfer = cnt.tf(expand_zeroes, expand_poles)
    #test_trasferfunction.fft()


    plt.figure(2)
    plt.plot(test_transferfunction, label = "{0:.1f} * zeros and {1:.1f}" .format(i,j))
    plt.xlabel("frequency") 
    plt.ylabel(" Amplitude")
plt.title("Transferfunction a and b influence")
plt.legend()
plt.grid()
plt.show()


