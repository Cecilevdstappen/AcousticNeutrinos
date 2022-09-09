#----------------------------------------------------------------------------------------
# ENERGY DEPOSITION PROFILE HADRONIC AND ELECTROMAGNETIC SHOWERS
#----------------------------------------------------------------------------------------

# 1) Reads DAT....long and dEdr...dat files generated by CORSIKA-IW
#    plots and saves the averaged energy deposition wr shower depth and radial distance

# 2) Implements radial and longitudinal parametrization by ACoRNE Coll.

#----------------------------------------------------------------------------------------

import numpy as np
import math as m
import matplotlib.pyplot as plt
from my_styles import *

set_paper_style()

plt.style.use('/home/gebruiker/AcousticNeutrinos/filtering/Ed_scripts/cecile.mplstyle')
plt.rcParams.update({'font.size': 17})
plt.rcParams["figure.figsize"] = (8.5,9)
plt.rcParams["figure.dpi"] = 100
colors = ['orange','green','red','purple','brown','blue','grey','olive','cyan','gold','crimson']
linestyle = ['-','--','-.',':']

#----TO DO----
do_plot_dEdz = True                       #plot dEdz 
do_plot_dEdz_1_shower = False               #plot dEdz for single showers
do_plot_dEdr = True                       #plot dEdr
do_plot_dEdr_1_shower = False              #plot dEdr for single showers
create_file = False                       #create file mean dEdzdr
create_file_1_shower = False                #create file dEdzdr 1 shower (for e.m. showers)
do_plot_peakdEdz = False                   #plot peak dEdz as a function of energy
check_energy_deposit_particle_type = False
#-------------

#----INPUTS---
models = ['QGSJETII-03','QGSJETII-03 (no thin)','SIBYLL','QGSJETII-03 (no LPM)','QGSJETII-03 (factor 0.5 HE hadrCS)','QGSJETII-03 (factor 0.8 HE hadrCS)','EPOS 1.6','QGSJETII-03 (no LPM, no thin)','QGSJET 01c']
models_select = [0] #select specific models

num_showers_input = 100
num_showers_plot = 100
energies = np.arange(9,12) #exponent energy (in GeV)
facts_energy = np.asarray([1]) #multiplicative factor energy
depth = np.arange(10,20010,20) #g/cm2
electromagnetic = False #if shower is electromagnetic
#-------------


#FUNCTION LONGITUDINAL PARAMETRIZATION ACORNE
def dEdz_param(z,E):
    #units of GeV/gcm-2
    E_L = np.log10(E)
    P_1L_divE = 2.76*10**(-3) - 1.974*10**(-4)*E_L + 7.45*10**(-6)*E_L**2
    P_2L = -210.9 - 6.968*10**(-3)*E_L + 0.1551*E_L**2
    P_3L = -41.5 + 113.9*E_L - 4.103*E_L**2
    P_4L = 8.012 + 11.44*E_L - 0.5434*E_L**2
    P_5L = 0.7999*10**(-5) - 0.004843*E_L + 0.0002552*E_L**2
    P_6L = 4.563*10**(-5) - 3.504*10**(-6)*E_L + 1.315*10**(-7)*E_L**2
    L_zE = P_1L_divE*E*(((z-P_2L)/(P_3L-P_2L))**((P_3L-P_2L)/(P_4L+P_5L*z+P_6L*z**2)))*np.exp((P_3L-z)/(P_4L+P_5L*z+P_6L*z**2))
    return L_zE

#FUNCTION LONGITUDINAL PARAMETRIZATION ACORNE
def dEdr_param(z,r,E):
    #output: no units, z:list depths (gcm-2), r:list radius (gcm-2), E: energy (GeV)
    E_L = np.log10(E)
    C_1 = np.array([[0.1287E-01,-0.2573E+00, 0.9636E+00],[-0.4697E-04, 0.8072E-03, 0.5404E-03],[0.7344E-07,-0.1375E-05, 0.4488E-05]])
    C_2 = np.array([[-0.8905E-03,0.7727E-02, 0.1969E+01],[0.1173E-04,-0.1782E-03,-0.5093E-05],[-0.1058E-07, 0.1524E-06,-0.1069E-08]])
    ABC_1 = C_1*E_L**(np.array([2,1,0]))
    ABC_2 = C_2*E_L**(np.array([2,1,0]))
    ABC_1 = np.array(np.sum(ABC_1,1))
    ABC_2 = np.array(np.sum(ABC_2,1))
    P_1R = np.dot(ABC_1,np.array((np.ones(len(z)),z,z**2)))
    P_2R = np.dot(ABC_2,np.array((np.ones(len(z)),z,z**2)))

    gamma_prod = np.array([m.gamma(4.5-2*i)*m.gamma(i)/m.gamma(4.5-i) for i in P_2R]) #corrected from octave script
    integral = np.multiply(P_1R,gamma_prod)
    #integral[integral<=0] = np.mean(integral[integral>0])*1e6
    integral[integral<=0] = 0.77*1e6 #changed to be the same as octave script
    integral[-1] = 1e6
    frac_r_P1R = np.array([r/i for i in P_1R])

    term1 = np.power(np.transpose(frac_r_P1R),P_2R-1,dtype=np.complex)
    term2 = np.power(np.transpose(1+frac_r_P1R),P_2R-4.5,dtype=np.complex)
    term1 = np.transpose(term1)
    term2 = np.transpose(term2)

    dEdr_par = np.multiply(term1,term2).real
    dEdr_par[dEdr_par<0] = 0
    dEdr_norm_par = np.array([dEdr_par[idx,:]/i for idx,i in enumerate(integral)])
    return dEdr_norm_par



peak_meandEdz = []

for energy in energies:
    for fact_energy in facts_energy:

        print("Energy= ",fact_energy,'E',int(energy))
        z_max_models = np.zeros(len(models_select))

        #dEdrdz ACoRNE parametrization
        E = fact_energy*10**energy
        radial_steps =  np.arange(0.5,100,1)*1.025 #g/cm-2
        zstep = 20
        longi_steps = np.arange(10,2010,zstep) #g/cm-2
        dEdr_par = dEdr_param(longi_steps,radial_steps,E) #a.u.
        dEdz_par = dEdz_param(longi_steps,E) #GeV/gcm-2

        print(longi_steps[np.argmax(dEdz_par)]/(1.025*10**2))

        dEdzdr_par = np.zeros((len(longi_steps),len(radial_steps)))
        for idx_z in range(len(longi_steps)):
            dEdzdr_par[idx_z,:] = dEdz_par[idx_z]*dEdr_par[idx_z,:]

        print("check energy conservation ACoRNE param. for longi. distribution: ",np.sum(dEdz_par)*zstep/E)
        print("check energy conservation ACoRNE param. after radial distribution: ",np.sum(dEdzdr_par[:,:])*zstep/E)

        #dEdzdr from CORSIKA simulations
        for idx_model,model in enumerate(models_select):
            model_name = models[model]
            print("Model= ",model_name)

            #----from .long file--------
            if model == 0:
                name_file_dz = 'cg_{:s}_p_dEdr_lowestEcuts_DAT{:02d}{:02d}{:02d}.long'.format(str(num_showers_input),fact_energy,energy,model)
            if model == 2:
                name_file_dz = 'cg_20_p_dEdr_tocheck_DAT{:02d}{:02d}{:02d}.long'.format(fact_energy,energy,model)
            if model == 8:
                name_file_dz = 'cg_20_p_dEdr_tocheck_DAT{:02d}{:02d}{:02d}.long'.format(fact_energy,energy,model)
            name_file_dz = 'DAT011106.long' 
            f_dz = open(name_file_dz, 'r')
            lines_dz = f_dz.readlines()
            print (len(lines_dz))
            num_showers = int(len(lines_dz)/2012)
            dEdz_long = np.zeros((num_showers,1000))         
            mean_dEdz_long = np.zeros((1000))                
            total_energy_long = np.zeros((num_showers))
            dEdz_particle_type = np.zeros((num_showers,1000,4))
            mean_dEdz_particle_type = np.zeros((1000,4))
            #---------------------------
            
            #-----from dEdr...dat file---
            #name_file_dr = 'cg_dEdr_{:01d}{:02d}{:02d}_lowestEcuts.dat'.format(fact_energy,energy,model)
            #name_file_dr = 'cg_dEdr_{:01d}{:02d}00_lowestEcuts.dat'.format(fact_energy,energy)
            name_file_dr = '/home/gebruiker/AcousticNeutrinos/signal_reconstruction/output_cs_inputs_Sloan_QGSJET_1e'+str(energy)+'.dat' 
            f_dr = open(name_file_dr, 'r')
            lines_dr = f_dr.readlines()
            dEdzdr = np.zeros((num_showers_input,1000,20))    #array dEdzdr per shower (from dEdr...dat file)
            mean_dEdzdr = np.zeros((1000,20))
            """
            name_file_dr_2 = 'cg_dEdr_{:01d}{:02d}{:02d}_lowestEcuts.dat'.format(fact_energy,energy,model)
            #name_file_dr_2 = 'cg_dEdr_{:01d}{:02d}00_new_egsdat.dat'.format(fact_energy,energy)
            f_dr_2 = open(name_file_dr_2, 'r')
            lines_dr_2 = f_dr_2.readlines()
            dEdzdr_2 = np.zeros((num_showers,1000,20))  #array dEdzdr per shower (from dEdr...dat file)
            mean_dEdzdr_2 = np.zeros((1000,20))
            """
            #---------------------------
            
            mean_dEdzdr_old = np.zeros((1000,20))       #array dEdzdr with no lateral distribution
            diff_dep_ener = np.zeros((num_showers,1000))  #array to check correctness of lateral distr. modf


            for idx_shower in range(num_showers):

                #energy deposition at each depth (1001 vertical steps of 20gcm^-2)
                for idx_line,line in enumerate(lines_dz[1005+idx_shower*2012 : 2005+idx_shower*2012]):
                   
                    print (idx_line)
                    if idx_line == 100: break #if we consider 20m showers
                    string_line_dz = line.split()
                    string_line_dr = lines_dr[idx_line+1825*idx_shower].split()
                    #string_line_dr_2 = lines_dr_2[idx_line+1825*idx_shower].split()

                    dEdz_long[idx_shower,idx_line] = float(string_line_dz[9])-float(string_line_dz[8]) #we substract the neutrino's energy
                    dEdzdr[idx_shower,idx_line,:] = [float(i) for i in string_line_dr[1:21]]
                    #dEdzdr_2[idx_shower,idx_line,:] = [float(i) for i in string_line_dr_2[1:21]]

                    #check if:   sum lateral energy deposit / depth_i   =   logi energy deposit / depth_i   (to check correctness of lateral distr. modf)
                    diff_dep_ener[idx_shower][idx_line] = (dEdz_long[idx_shower][idx_line] - (np.sum(dEdzdr[idx_shower][idx_line])+float(string_line_dr[21])))/dEdz_long[idx_shower][idx_line]*100
                    #if diff_dep_ener[idx_shower][idx_line] > 1.0: #if difference bigger than 1% -> print
                        #print(idx_shower, idx_line, dEdz[idx_shower][idx_line], (np.sum(dEdzdr[idx_shower][idx_line])+float(string_line_dr[21])), diff_dep_ener[idx_shower][idx_line])

                    if check_energy_deposit_particle_type == True:
                        dEdz_particle_type[idx_shower,idx_line,0] = float(string_line_dz[1])
                        dEdz_particle_type[idx_shower,idx_line,1] = float(string_line_dz[2])+float(string_line_dz[3])
                        dEdz_particle_type[idx_shower,idx_line,2] = float(string_line_dz[4])+float(string_line_dz[5])
                        dEdz_particle_type[idx_shower,idx_line,3] = float(string_line_dz[6])+float(string_line_dz[7])

                
                if do_plot_dEdz_1_shower == True and idx_shower <= 3:
                    #plt.plot(depth/(1.025*10**2),np.sum(dEdzdr[idx_shower,:,:],1)*1.025*10**2/20)
                    #plt.legend(fontsize='small',loc='upper right')
                    plt.plot(depth/(1.025*10**2),np.sum(dEdz_particle_type[idx_shower,:,:]))

                if do_plot_dEdr_1_shower == True and idx_shower == 1:
                    #need to take into account different radial binnings to plot radial profile
                    arr110 = np.array([1,1/10])
                    diag_array = np.zeros((20,20),dtype=float)
                    a = np.kron(arr110,np.ones((1,10)))
                    np.fill_diagonal(diag_array,a)
                    dEdzdr_toplot = np.dot(dEdzdr[idx_shower,:,:],diag_array)
                    dEdzdr_toplot_2 = np.dot(dEdzdr_2[idx_shower,:,:],diag_array)

                    color = 0
                    for idx_depth in range(100,200,10):
                        radial_steps =  np.concatenate((np.arange(0.5,10.5,1.0),np.arange(10.5,105,10.0))) #center radial bins cm
                        plt.plot(radial_steps/10**2,dEdzdr_toplot[idx_depth,:]*(1.025*10**2)/20,c=colors[color],label='z={:.01f}m'.format(depth[idx_depth]/(1.025*10**2)))
                        color = color+1


                if create_file_1_shower == True:
                    name_file_1 = 'output_cs_inputs_Sloan_QGSJET_1e'+str(energy)+'.dat'
                    #name_file_1 = 'cg_dEdzdr_shower{:}_DAT{:02d}{:02d}{:02d}_e_lowestEcuts.dat'.format(idx_shower,fact_energy,energy,model)
                    np.savetxt(name_file_1,dEdzdr[idx_shower,:,:],fmt='%1.4e') 

                #total deposited energy to check for energy conservation
                total_energy_long[idx_shower] = np.sum(dEdzdr[idx_shower,:,:])
                fraction_energy = total_energy_long[idx_shower]/(fact_energy*10**energy)
                #print("fraction deposited energy 1 shower: ",fraction_energy)

            mean_dEdz_long = np.sum(dEdz_long,0)/num_showers
            print(num_showers,mean_dEdz_long)
            mean_dEdzdr = np.sum(dEdzdr,0)/num_showers
            print(depth[102],np.sum(mean_dEdzdr[0:77])/np.sum(mean_dEdzdr),np.sum(mean_dEdzdr[0:102])/(fact_energy*10**energy))
            print("mean fraction total deposited energy: ",np.sum(mean_dEdzdr[:,:])/(fact_energy*10**energy))
            #mean_dEdzdr_2 = np.sum(dEdzdr_2,0)/num_showers
            mean_dEdz_particle_type = np.sum(dEdz_particle_type,0)/num_showers

            #dEdz peak
            peak_meandEdz.append(np.amax(np.sum(mean_dEdzdr,1)))

            #compute Z position where there is the maximum energy deposit
            z_max = np.argmax(np.sum(mean_dEdzdr,1))*20/(1.025*10**2)+10/(1.025*10**2) #m (1:50 for 1e10GeV, 0:75 others)
            #z_max_2 = np.argmax(np.sum(mean_dEdzdr_2,1))*20/(1.025*10**2)+10/(1.025*10**2) #m (1:50 for 1e10GeV, 0:75 others)
            z_max_par = np.argmax(dEdz_par[1:100])*20/(1.025*10**2)+10/(1.025*10**2) #m (1:50 for 1e10GeV, 0:75 others)
            print("mean zmax 100 showers: ",z_max)                                      

            #create file with averaged dEdzdr
            if create_file == True:
                #name_file_1 = 'cg_mean_dEdzdr_{:s}_DAT{:02d}{:02d}{:02d}_lowestEcuts.dat'.format(str(num_showers_input),fact_energy,energy,model)
                name_file_1 = 'cs_mean_dEdzdr_'+str(energy)+'.dat'
                """
                mean_dEdzdr_modif = np.zeros((1000,20))
                radial_steps =  np.concatenate((np.arange(0.5,10.5,1.0),np.arange(10.5,105,10)))/(10**(2)) #cm to m, bin center
                longi_steps = np.arange(10,20010,20)/(10**(2)*1.025) #g/cm2 to m (longi steps in corsika are in g/cm2), bin center              
                for idx_z in range(1000):
                    for idx_r in range(0,20):

                        if idx_r == 0: weight_volume=(radial_steps[idx_r]**2)
                        else: weight_volume = (radial_steps[idx_r]**2-radial_steps[idx_r-1]**2)
                        mean_dEdzdr_modif[idx_z,idx_r] = mean_dEdzdr[idx_z,idx_r]*weight_volume
                print(mean_dEdzdr_modif[40,:])

                Aedepos = abs(mean_dEdzdr[:,8]-mean_dEdzdr[:,9])
                for i in range(100):
                    mean_dEdzdr_modif[:,10+i]=mean_dEdzdr_modif[:,10+i-1]-Aedepos
                    for j in range(1000):
                        if mean_dEdzdr_modif[j,10+i] < 0: mean_dEdzdr_modif[j,10+i]=0
                """

                np.savetxt(name_file_1,mean_dEdzdr,fmt='%1.4e') 

            #plot averaged dEdz
            if do_plot_dEdz == True:
                fig1 = plt.figure(num = 'Average E')
                plt.plot(depth/(1.025*10**2),mean_dEdz_long/(10**energy)*1.025*10**2/20,ls=linestyle[idx_model],label = 'E=1e'+str(energy))#label=model_name)
                #plt.plot(depth/(1.025*10**2),np.sum(mean_dEdzdr,1)*1.025*10**2/20,label=model_name)
                #if model == 0: plt.plot(depth/(1.025*10**2),mean_dEdz_long*1.025*10**2/20,ls=linestyle[idx_model],label = 'E=1e'+str(energy))#label=model_name)
                #else: plt.plot(depth/(1.025*10**2),mean_dEdz_long*1.025*10**2/20,c=colors[idx_model-1],ls=linestyle[idx_model],label = 'E=1e'+str(energy))#label=model_name)

            plt.title('hadronic shower: E ={:d}e{:d} GeV'.format(fact_energy,energy),loc='right')
            #plt.title('ν'+neutrinos[neutrino_type]+':  $E_{prim} = $'+"1e{:d} GeV".format(energy)+'$, N_{showers}$='+'{:d}'.format(num_showers_input),loc='right')
            #plt.ylabel("Energy deposition ($GeV \; / \; 20 \, gcm^{-2}$)")
            #plt.xlabel("Shower depth ($gcm^{-2}$)")
            plt.ylabel("longitudinal energy deposition (GeV/m)")
            plt.xlabel("z (m)")
            plt.xlim(0,20)       #plt.plot(depth/(1.025*10**2),np.sum(mean_dEdzdr_2,1)*1.025*10**2/20,':',label=model_name+', 3MeV cuts')
            plt.legend()
            plt.show()

            #check longi energy deposit by particle type
            if check_energy_deposit_particle_type == True:
                ptype_names = ['$\gamma$','$e^{\pm}$','$\mu$','hadrons']
                ptypes = [1,3,0,2]

                for ptype in ptypes:
                    plt.plot(depth[100:1000]/(1.025*10**2),dEdz_particle_type[idx_shower,100:1000,ptype]*1.025*10**2/20,label=ptype_names[ptype])
                    plt.plot(depth[:]/(1.025*10**2),mean_dEdz_particle_type[:,ptype]*1.025*10**2/20,ls=linestyle[ptype],c=colors[ptype+1],label=ptype_names[ptype])

                ptype=2

                #for idx_shower in range(0,15):
                #    plt.plot(depth[100:1000]/(1.025*10**2),dEdz_particle_type[idx_shower,100:1000,ptype]*1.025*10**2/20,lw=1,label=ptype_names[ptype]+', $n_{shower}$='+'{:}'.format(idx_shower))
                    #plt.plot(depth[100:1000]/(1.025*10**2),np.sum(dEdz_particle_type[idx_shower,100:1000,:],1)*1.025*10**2/20,lw=1,label='$n_{shower}$='+'{:}'.format(idx_shower))

                #plt.plot(depth[100:1000]/(1.025*10**2),mean_dEdz_particle_type[100:1000,ptype]*1.025*10**2/20,':',c='gold',lw=2.2,label=ptype_names[ptype]+', $<dEdz>_{100 showers}$')
                #plt.plot(depth[100:1000]/(1.025*10**2),np.sum(mean_dEdz_particle_type[100:1000,:],1)*1.025*10**2/20,':',c='gold',lw=2.2,label='$<dEdz>_{100 showers}$')
                    #plt.xlim(0,2.5*10**5)

        f_dz.close()
        f_dr.close()
        #f_dr_2.close()
        #print("Zmax = ", np.sum(z_max_models)/len(models_select), "g/cm2") #average Z between models at which we find energy deposition maximum -> have to take out model 3 (no LPM)
 


        #LONGITUDINAL PROFILE N SHOWERS
        if do_plot_dEdz == True or check_energy_deposit_particle_type == True: #plot longitudinal distribution N_showers-averaged
            #plt.plot(longi_steps/(1.025*10**2),dEdz_par*1.025*10**2,ls=linestyle[3],c=colors[4],label='ACoRNE \n parametrisation')
            plt.legend(loc='upper right')
            plt.title('hadronic shower: E ={:d}e{:d} GeV'.format(fact_energy,energy),loc='right')
            #plt.title('ν'+neutrinos[neutrino_type]+':  $E_{prim} = $'+"1e{:d} GeV".format(energy)+'$, N_{showers}$='+'{:d}'.format(num_showers_input),loc='right')
            #plt.ylabel("Energy deposition ($GeV \; / \; 20 \, gcm^{-2}$)")
            #plt.xlabel("Shower depth ($gcm^{-2}$)")
            plt.ylabel("longitudinal energy deposition (GeV/m)")
            plt.xlabel("z (m)")

            #limits plot
            if  electromagnetic == True:
                if energy < 7:   xmax=2000 
                elif energy == 7: xmax=3000
                elif energy == 8: xmax=4000 
                elif energy == 9: xmax=8000
                elif energy == 10: xmax=17500
            else: xmax=2000

            plt.xlim(0, 20)
            plt.ylim(0,2*10**10)
            #plt.tight_layout()
            #plt.show()
            #plt.savefig("si.png",dpi=200)
            plt.savefig('cg_{:s}FIG{:02d}{:02d}{:02d}_p_particletype.png'.format(str(num_showers_plot)+"_",fact_energy,energy,model),dpi=300)
            plt.show()
            #plt.close() 




        #LONGITUDINAL PROFILE 1 SHOWER
        if do_plot_dEdz_1_shower == True: #to plot longitudinal distribution 1 shower
            plt.title('electromagnetic shower'+':  $E_{prim} = $'+"{:d}e{:d} GeV".format(fact_energy,energy),loc='right') #for electron/proton shower
            plt.ylabel("longitudinal energy deposition ($GeV / m$)")
            plt.xlabel("shower depth ($m$)")

            #limits plot
            if  electromagnetic == True:
                if energy < 7:   xmax=2000 
                elif energy == 7: xmax=3000
                elif energy == 8: xmax=4000 
                elif energy == 9: xmax=8000
                elif energy == 10: xmax=17500
            else: xmax=2000

            plt.xlim(0, xmax/10**2)
            plt.tight_layout()
            plt.show()
            #plt.savefig('cg_1_FIG{:02d}{:02d}{:02d}_n1_e_dEdz.png'.format(fact_energy,energy,model+n),dpi=300)
            #plt.close()




        #RADIAL PROFILE N SHOWERS
        if do_plot_dEdr == True: #averaged radial energy deposited per 20gcm-2 vertical slice per unit radial distance
        
            #need to take into account different radial binnings to plot radial profile
            arr110 = np.array([1,1/10])
            diag_array = np.zeros((20,20),dtype=float)
            a = np.kron(arr110,np.ones((1,10)))
            np.fill_diagonal(diag_array,a)
            mean_dEdzdr_toplot = np.dot(mean_dEdzdr,diag_array)
            #mean_dEdzdr_toplot_2 = np.dot(mean_dEdzdr_2,diag_array)

            color = 0

            idx_shower = 11
            dEdzdr_toplot = np.dot(dEdzdr[idx_shower,:,:],diag_array)

            factors = [1,2,3,4,5,6]
            for idx,idx_depth in enumerate(range(10,70,10)):
            #for idx_depth in range(0,384,26):

                radial_steps =  np.concatenate((np.arange(0.5,10.5,1.0),np.arange(10.5,105,10.0))) #center radial bins cm (radial bins in corsika in cm)
                fig2 = plt.figure(2)
                #plt.plot(radial_steps/10**2,mean_dEdzdr_toplot[idx_depth,:]*(1.025*10**2)/20,c=colors[color],label='<dEdr>, z={:.0f}m'.format(depth[idx_depth]/(1.025*10**2)))
                #plt.plot(radial_steps/10**2,mean_dEdzdr_toplot[idx_depth,:]*(1.025*10**2)/20,c=colors[color],label='z={:.0f}m'.format(depth[idx_depth]/(1.025*10**2)))

               # plt.plot(radial_steps/10**2,dEdzdr_toplot[idx_depth,:]*(1.025*10**2)/20,':',c=colors[color],label='$n_{shower}=$'+'{:}'.format(idx_shower))
               # plt.plot(radial_steps/10**2,mean_dEdzdr_toplot[idx_depth,:],c=colors[color],label='z={:.0f}m'.format((depth[idx_depth])/(1.025*10**2)))
                if idx == 0 or idx==2 or idx == 4: plt.plot(radial_steps,factors[idx]*mean_dEdzdr_toplot[idx_depth,:]/np.sum(mean_dEdzdr_toplot[idx_depth,:]),c=colors[color],label='x {:}, z={:.0f}m'.format(factors[idx],depth[idx_depth]/(1.025*10**2)))
                #plt.plot(radial_steps,mean_dEdzdr_toplot[idx_depth,:],c=colors[color],label='$E_{cut}=1$ MeV, '+'z={:.0f}m'.format((depth[idx_depth])/(1.025*10**2)))
                #if idx == 0 or idx==2 or idx == 4: plt.plot(radial_steps,factors[idx]*mean_dEdzdr_toplot_2[idx_depth,:]/np.sum(mean_dEdzdr_toplot[idx_depth,:]),c=colors[color],ls=':')
#,label='$E_{cut}=100$ MeV'

                #plt.plot(radial_steps/10**2,mean_dEdzdr_toplot[idx_depth,:]*(1.025*10**2)/20,c=colors[color],label='egsdat_1, z={:.0f}m'.format(depth[idx_depth]/(1.025*10**2)))
                #plt.plot(radial_steps/10**2,mean_dEdzdr_toplot_2[idx_depth,:]*(1.025*10**2)/20,':',c=colors[color],label='egsdat_100')


                #plt.plot(radial_steps/10**2,mean_dEdzdr_toplot_2[idx_depth,:]*(1.025*10**2)/20,':',c=colors[color],label='3MeV cuts')

                z_max_param = longi_steps[np.argmax(dEdz_par)]
                print(z_max_param)
                diff_zmax_idxs = int((z_max_param-z_max*10**2*1.025)/20)
                print(diff_zmax_idxs)
                radial_steps =  np.arange(0.5,100,1)*1.025 #g/cm-2
                #plt.plot((radial_steps)/(1.025*10**2),dEdzdr_par[idx_depth-diff_zmax_idxs,:],':',c=colors[color],label='ACoRNE param., z={:.0f}m'.format((depth[idx_depth-diff_zmax_idxs])/(1.025*10**2)))
                #plt.plot((radial_steps)/(1.025*10**2),depth[idx_depth-diff_zmax_idxs]/((1.025*10**2))*dEdzdr_par[idx_depth-diff_zmax_idxs,:]/np.sum(dEdzdr_par[idx_depth-diff_zmax_idxs,:]),'--',c=colors[color],label='ACoRNE param., z={:.0f}m'.format((depth[idx_depth-diff_zmax_idxs])/(1.025*10**2)))
                #plt.plot((radial_steps)/(1.025),factors[idx]*dEdzdr_par[idx_depth,:]/np.sum(dEdzdr_par[idx_depth,:]),':',c=colors[color],label='ACoRNE param.'.format((depth[idx_depth])/(1.025*10**2)))
                color = color+1

                #plt.title('electromagnetic shower:  E = {:d}e{:d} GeV'.format(fact_energy,energy),loc='right')
                plt.title('hadronic shower:  E = {:d}e{:d} GeV'.format(fact_energy,energy),loc='right')

                
                plt.legend(fontsize='small',loc='upper center', bbox_to_anchor=(1.3, 1.0))
                #plt.legend(loc='lower left')
                plt.yscale("log")
                #plt.xscale("log")
                plt.xlim(0,1000)
                plt.ylim(10**(-3),2)

                #plt.ylim(2*10**(4),7*10**(9))
                #plt.ylabel("Energy deposition ($GeV \; / \; g^{2}cm^{-4}$)")
                #plt.xlabel("r ($gcm^{-2}$)")
                plt.ylabel("Radial energy deposition (GeV/m)")
                #plt.ylabel("normalised radial energy deposition")
                plt.xlabel("r (m)")
                #plt.xlabel("r (cm)")
                plt.tight_layout()
                plt.savefig('cg_{:s}FIG{:02d}{:02d}{:02d}_p_dEdr_radial.png'.format(str(num_showers_input)+"_",fact_energy,energy,model))
            plt.show()
            #plt.savefig('cg_{:s}FIG{:02d}{:02d}{:02d}_p_dEdzdr_{:02d}_0_100.png'.format(str(num_showers_input)+"_",fact_energy,energy,model+n,longi_steps[idx_depth]))
            #plt.savefig('cg_{:s}FIG{:02d}{:02d}{:02d}_p_dEdr_radial.png'.format(str(num_showers_input)+"_",fact_energy,energy,model+n))
            plt.close()   



#MAXIMUM ENERGY DEPOSITION DEPTH VS ENERGY
if do_plot_peakdEdz == True:
    energy_array = np.array([10**8,5*10**8,10**9,5*10**9,10**10,5*10**10,10**11,5*10**11,10**12])
    plt.scatter(energy_array,peak_meandEdz)
    print(peak_meandEdz)
    plt.xlim(5*10**7,5*10**12)
    plt.xlabel('primary energy (GeV)')
    plt.ylabel('peak dEdz')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('cg_peakdEdz_E_new.png')
    plt.close()
