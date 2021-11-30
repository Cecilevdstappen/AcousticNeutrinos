from scipy.io import savemat
import numpy as np
import control.matlab as cnt
import matplotlib.pyplot as plt

#read in transfer function file
file_name = ''
f = open(file_name,'r')
lines = f.readlines()
freq = [] #s
y_im = []


for line in lines:
    if line.startswith("#") or len(line) <= 1: continue
    else: freq.append(float(line.split(' ')[0])), y_im.append(float(line.split(' ')[1]))


#Create transfer function from coeff
#transfer_freq = cnt.tf(num, den)

transfer_function = np.fft.ifft(transfer_freq)

#Poles is the factorized denumerator array and zeros is the numerator array
poles,zeros = cnt.pzmap(transfer_function)

#play with poles and zeroes values and see how it changes the transferfunction
for i,j in np.arange(0,5,0.5):
    test_poles = i*poles
    test_zeros = i*zeros
    test_transferfunction = cnt.tf(test_zeros, test_poles)
    test_trasferfunction.fft()


    plt.figure(1)
    plt.plot(test_transferfunction, label = "{0:.1f} * zeros and {1:.1f}" .format(i,j))
    plt.xlabel("frequency") 
    plt.ylabel(" Amplitude")
plt.title("Transferfunction a and b influence")
plt.legend()
plt.grid()
plt.show()


