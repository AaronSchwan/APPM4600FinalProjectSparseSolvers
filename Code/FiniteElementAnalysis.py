"""
Finite element analysis 

ONE-DIMENSIONAL EXAMPLE: FLOW IN INFINITESIMALLY EXTENDED CHANNELS
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#Local File Imports
import Matrice_Finder as mf
import ThomasAlg as ta

#Setting Default Plot Style
plt.style.use('seaborn-v0_8')

#Analytical solutions 
def Vx(z):
    return 50*(1-z**2)

#Numerical Solutions
M_DeltaZ_0_2,r_DeltaZ_0_2 = mf.generate_stiffness_matrix(10)
print(M_DeltaZ_0_2)
x_DeltaZ_0_2 = ta.thomas_alg_cpu(M_DeltaZ_0_2,r_DeltaZ_0_2)
print(x_DeltaZ_0_2)
"""
M_DeltaZ_0_1,r_DeltaZ_0_1 = mf.generate_stiffness_matrix(20)
x_DeltaZ_0_1 = ta.thomas_alg_decimal(M_DeltaZ_0_1,r_DeltaZ_0_1,50)


#Analytical Graph
z_space = np.linspace(-1,1,1000)
plt.plot(z_space,Vx(z_space),label="Analytical Solution")

#Numerical Solutions
plt.scatter(np.linspace(-1,1,len(x_DeltaZ_0_2)),x_DeltaZ_0_2,label="Numerical Delta Z = 0.2")
plt.scatter(np.linspace(-1,1,len(x_DeltaZ_0_1)),x_DeltaZ_0_1,label="Numerical Delta Z = 0.1")

#Formatting
plt.legend()
plt.xlabel("Z(mm)")
plt.ylabel("Vx(mm/s)")
plt.show()

#Creating an error plot 
center_analytical_val = Vx(0)

central_errors = []
n_vals = 1000
for i in range(10,n_vals,2):
    M,r = mf.generate_stiffness_matrix(i)
    x = ta.thomas_alg_decimal(M,r,100)
    
    #finding the error for even only
    #function approaches from above no absolutes needed
    central_errors.append(x[int(i/2)]-center_analytical_val)


plt.semilogy(range(10,n_vals,2),central_errors)
plt.xlabel("Divisions of Pipe")
plt.ylabel("Log(error at center)")
plt.show()
"""

def thomas_for_fea(n,eta = 1*pow(10,-5),dpdx = -0.001):
    delta_z = 2/n
    r0 = -dpdx/eta*pow(delta_z,2)
    
    x = np.zeros(n)

    x[n-1] = (n)/2

    def beta(i):
        return (i+1)/2
    def alpha(i):
        return (i+2)/(i+1)

    for i in reversed(range(int(n/2),n-1)):
        x[i] = beta(i)+x[i+1]/alpha(i)

    #mirroring
    for i in range(0,int(n/2)):
        x[i] = x[n-i-1]

    #no need to reverse due to symmetry
    #scale by r0 
    return x*r0


def thomas_fea_most_simplified(half_n,eta = 1*pow(10,-5),dpdx = -0.001):

    r0 = -dpdx/eta

    n = int(2*half_n)
    
    dx = 2/n
    min_x = -1+1/n


    x = np.arange(min_x,-min_x+1/n,dx)
    half_vx = np.arange(1,half_n+1)
    
    half_vx = (n-half_vx+1)*half_vx/2
    
    coeff = 4/pow(n,2)*r0

    half_vx = half_vx*coeff

    vx = np.zeros(n)
    vx[0:half_n]=half_vx
    vx[half_n:n]=np.flip(half_vx)




    
        

    return vx


    
print(thomas_fea_most_simplified(10000))

    
