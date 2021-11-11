import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('GTK3Agg')
plt.get_backend()

def get_data(filename):
    data = np.loadtxt(filename,  usecols=(0,1), unpack=True) #Take the first and second column
    x = data[0]
    y = data[1]
    print(x)
    print(y)
    return x,y


def main():
    #file1 = 'spectrum_11_1000_6_.dat' 
    file1 = 'neutrino_12.0_1000_1_11.dat'
    x,y = get_data(file1)
    x_kHz = x*(10**-3)
    plt.plot(x_kHz, y, linewidth=1.5, label = 'proton shower')
    plt.grid()
    plt.grid('on', 'minor')

    #plt.xlim([0,200])
    plt.xlabel("frequency[kHz]", ha ='right', x=1.0)
    plt.ylabel("", ha ='right', position=(0,1))
    #plt.ylabel("Monte Carlo", ha ='right', position=(0,1))

    legend = plt.legend(loc='upper right')
    plt.title(str(file1)) 
    #plt.savefig("mc_points.png", dpi=200)
    #plt.savefig("energy_depo_05.png" , dpi = 200
    plt.savefig("spectrum_11_1000_6_spermwhale.png")
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
