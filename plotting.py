import matplotlib.pyplot as plt
import numpy as np
import FinalCode as fc
plt.style.use("ScientificPlots")

#############################################################################
# Numerical Time Plots
#############################################################################
if True:
    gauss = np.loadtxt("GaussTimes.csv", delimiter=",", dtype=str).astype(float)
    cpu = np.loadtxt("ThomasCPUTimes.csv", delimiter=",", dtype=str).astype(float)
    gpu = np.loadtxt("ThomasGPUTimes.csv", delimiter=",", dtype=str).astype(float)

    plt.scatter(2*np.arange(10, pow(10,3)),np.mean(gpu,axis=1)[0:pow(10,3)-10],label="GPU",c="#ff3333")
    plt.scatter(2*np.arange(10, pow(10,3)),np.mean(cpu,axis=1)[0:pow(10,3)-10],label="CPU",c="#3333ff")
    plt.scatter(2*np.arange(10, pow(10,3)),np.mean(gauss,axis=1),label="Gauss",c="#33cc33")
    plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15), ncols=3)
    plt.xlabel("Matrix (n)")
    plt.ylabel("Average Time (s)")
    plt.show()

    plt.scatter(2*np.arange(pow(10,4)+10, pow(10,5)),np.mean(gpu,axis=1)[pow(10,4):pow(10,5)],label="GPU",c="#ff3333")
    plt.scatter(2*np.arange(pow(10,4)+10, pow(10,5)),np.mean(cpu,axis=1)[pow(10,4):pow(10,5)],label="CPU",c="#3333ff")
    plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15), ncols=2)
    plt.xlabel("Matrix (n)")
    plt.ylabel("Average Time (s)")
    plt.show()


#############################################################################
# Numerical Approximation Plots 
#############################################################################
if False:
    Vx1,x1 = fc.thomas_cpu(5)
    Vx2,x2 = fc.thomas_cpu(10)
    Vx3,x3 = fc.thomas_cpu(20)

    def analytical(x):
        return 50*(1-pow(x,2))

    plt.plot(np.linspace(-1,1,200),analytical(np.linspace(-1,1,200)),"--",c="Gray",label = "Analytical")
    
    plt.scatter(x1,Vx1,label="0.2 $\Delta Z$",c="#ff3333")
    plt.scatter(x2,Vx2,label="0.1 $\Delta Z$",c="#3333ff")
    plt.scatter(x3,Vx3,label="0.05 $\Delta Z$",c="#33cc33")
    
    plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15), ncols=4)
    plt.xlabel("Position (mm)")
    plt.ylabel("$V_x \left(\\frac{\mathrm{mm}}{\mathrm{s}}\\right)$")
    plt.show()


#############################################################################
# Largest Errors 
#############################################################################
if False:
    half_n_list = np.arange(5,pow(10,4))
    max_errors = np.zeros(len(half_n_list))

    for ind,half_n in enumerate(half_n_list):
        Vx,x = fc.thomas_cpu(half_n)
        
        max_errors[ind] =  np.max(Vx-analytical(x),axis=0)
        
    plt.loglog(half_n_list*2,max_errors,c="#3333ff")
    
    plt.xlabel("$\log\left(n\\right)$")
    plt.ylabel("$\log\left(\mathrm{Maximum Error} \left(\\frac{\mathrm{mm}}{\mathrm{s}}\\right)\\right)$")
    plt.show()