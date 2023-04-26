"""
Finite element analysis 

ONE-DIMENSIONAL EXAMPLE: FLOW IN INFINITESIMALLY EXTENDED CHANNELS
"""

#Imports
import numpy as np
import matplotlib.pyplot as plt

#Local File Imports
import Matrice_Finder as mf
import ThomasAlg as ta

#Setting Default Plot Style
plt.style.use('seaborn-v0_8')

#Analytical solutions 
def Vx(z):
    return 50*(1-z**2)

#Numerical Solutions
M_DeltaZ_0_2 = np.array([[2,-1,0,0,0,0,0,0,0,0],
                        [-1,2,-1,0,0,0,0,0,0,0],
                        [0,-1,2,-1,0,0,0,0,0,0],
                        [0,0,-1,2,-1,0,0,0,0,0],
                        [0,0,0,-1,2,-1,0,0,0,0],
                        [0,0,0,0,-1,2,-1,0,0,0],
                        [0,0,0,0,0,-1,2,-1,0,0],
                        [0,0,0,0,0,0,-1,2,-1,0],
                        [0,0,0,0,0,0,0,-1,2,-1],
                        [0,0,0,0,0,0,0,0,-1,2]])

r_DeltaZ_0_2 = np.array([4,4,4,4,4,4,4,4,4,4])

x_DeltaZ_0_2 = ta.thomas_alg_decimal(M_DeltaZ_0_2.astype(float),r_DeltaZ_0_2.astype(float),100)

x = np.linalg.solve(M_DeltaZ_0_2,r_DeltaZ_0_2)
x_act = [20,36,48,56,60,60,56,48,36,20]


#Analytical Graph
z_space = np.linspace(-1,1,1000)
plt.plot(z_space,Vx(z_space),label="Analytical Solution")

#Numerical Solutions
plt.scatter(np.linspace(-1,1,len(x_DeltaZ_0_2)),x_DeltaZ_0_2,label="Numerical Delta Z = 0.2")

#Formatting
plt.legend()
plt.xlabel("Z(mm)")
plt.ylabel("Vx(mm/s)")
plt.show()