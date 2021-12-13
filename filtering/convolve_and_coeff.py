from scipy.io import savemat
import numpy as np
import control.matlab as cnt
import matplotlib.pyplot as plt
import sympy
import sys
import scipy
import itertools
from sippy import functionset as fset
from sippy.armax import Armax
from sippy.arx import ARX_id, select_order_ARX


#read in transfer function file
filename = 'Bodemat_Dubbel_1FF_Actran_Sweep200Hz20kHz0degree_AfterPressure_v2.dat'
filename = 'Bodemat_Dubbel_1FF_Atran_Sweep200Hz20kHz90degreev3(1).dat'
f = open(filename,'r')
lines = f.readlines()
freq = [] #s
y_im = []

#for line in lines:
#    if line.startswith("#") or len(line) <= 1: continue
#    else: freq.append(float(line.split(' ')[0])), y_im.append(np.imag(line.split(' ')[1]))

dtypes = [('col0', float)]+[('col1', np.complex128)]
data = np.loadtxt(filename,  usecols=(0,1), dtype=dtypes, unpack=True)#,  dtype='float,np.complex128)
tenthlines = itertools.islice(data[0], 200, 20000, 7)
y_tenthlines = itertools.islice(data[1],200,20000,7)
for line in tenthlines:
    freq.append(line)
freq=np.asanyarray(freq)
print (freq)
for line in y_tenthlines:
    y_im.append(line)
y_im = np.asarray(y_im)
norm = np.linalg.norm(y_im)
y_im = y_im/norm

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
end_time = time[-1]                                      # [s]
npts = len(time)-1
sampling_time = end_time/npts

#amplitude = np.roll(amplitude,-1000)
window = np.hamming(2048)
amplitude = window*amplitude
amplitude_f = np.fft.fft(amplitude)
#freq = np.fft.fftfreq(time.shape[-1])
df = 1/(len(amplitude)*sampling_time)
N = len(amplitude)
freq = np.array([df*n if n<N/2 else df*(n-N) for n in range(N)])
#freq = np.fft.ifftshift(freq)
print (freq)
norm = np.linalg.norm(amplitude_f)
amplitude_f = amplitude_f/norm
#amplitude_f.normalize()

white_noise_variance = [0.0001]
white_noise = fset.white_noise_var(y_im.size, white_noise_variance)[0]
#white_noise = window*white_noise
white_noise_f = np.fft.fft(white_noise) 
norm = np.linalg.norm(white_noise_f)
white_noise_f = white_noise_f/norm

transfer_function_t = np.fft.ifft(y_im, n=2*len(y_im))

y_im.resize(amplitude_f.size)
transfer_function_t.resize(amplitude.size)
#convolved_tf_t = scipy.signal.convolve(amplitude, transfer_function_t, mode='full', method='auto')
#convolved_tf_f = scipy.signal.convolve(amplitude_f, y_im, mode='full', method='auto')
#convolved_tf_f = (white_noise_f * y_im)
convolved_tf_f = (amplitude_f * y_im)
convolved_tf_f = window * convolved_tf_f
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
#plt.plot(y_im, label='transferfunction')
#plt.plot(white_noise_f, label='white noise')
plt.plot(freq, amplitude_f, label = 'signal')
#plt.plot(convolved_tf_f,linestyle='dashed', label ="convolved")
plt.xlabel("frequency")
plt.ylabel(" Amplitude")
plt.title("Convolution transferfunction and white noise")
plt.legend()
plt.grid()
plt.show()


plt.figure(2)
plt.plot(transfer_function_t, label='transferfunction')
plt.plot(amplitude, label = ' signal' )
#plt.plot(white_noise, label='white noise')
plt.plot(convolved_tf_t,linestyle='dashed', label ="convolved")
plt.xlabel("time")
plt.ylabel(" Amplitude")
#plt.xlim(0,1200)
plt.title("Convolution transferfunction and white noise time trace")
plt.legend()
plt.grid()
plt.show()

#Determining a and b coefficient
u= amplitude_f
y_reshape = np.reshape(convolved_tf_t, (2048,))
# Determine a and b coefficients of ARMAX method
Arm = Armax(na_bounds=[4,4], nb_bounds=[3,3], nc_bounds=[2, 2], delay_bounds=[11, 11], dt=1/npts)
G_num_armax, G_den_armax, H_num_armax, H_den_armax, Vn_armax, y_id_armax, max_reached_armax = Arm._identify(y_reshape, u, na=4, nb=3, nc=2, delay=11,max_iterations=300)
print("The Armax coeff are: a=%s, b=%s, c=%s"%(H_den_armax, G_num_armax,H_num_armax)) 

#Determine a and b coefficient of ARX method
G_num_arx, G_den_arx, H_num_arx, Vn_arx, y_id_arx = ARX_id(y_reshape,u,na=4, nb=3, theta=0)
print("The ARX coeff are: a=%s, b=%s, c=%s"%(G_den_arx, G_num_arx,H_num_arx))


#Determine a and b coefficients of FIR method
G_num_fir, G_den_fir, H_num_fir, Vn_fir, y_id_fir = ARX_id(y_reshape,u,na=0, nb=3, theta=0)
print("The FIR coeff are: a=%s, b=%s, c=%s"%(G_den_fir, G_num_fir,H_num_fir))

#Poles is the factorized denumerator array and zeros is the numerator array
#poles,zeros = cnt.pzmap(transfer_function_f)

#play with poles and zeroes values and see how it changes the transferfunction
#for i,j in np.arange(0,5,0.5):
#    test_poles = i*poles
#    test_zeros = i*zeros
#    expand_zeroes = sympy.expand(test_zeros)
#    expand_poles = sympy.expand(test_poles)
#    test_transfer = cnt.tf(expand_zeroes, expand_poles)
    #test_trasferfunction.fft()


#    plt.figure(2)
#    plt.plot(test_transferfunction, label = "{0:.1f} * zeros and {1:.1f}" .format(i,j))
#    plt.xlabel("frequency") 
#    plt.ylabel(" Amplitude")
#plt.title("Transferfunction a and b influence")
#plt.legend()
#plt.grid()
#plt.show()



