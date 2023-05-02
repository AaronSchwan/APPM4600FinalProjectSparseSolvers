"""
This is just a simple script to run the commands several times and save the times to later be averages

1) Scaling comparision of the memory required for solving a nxn with gauss and thomas (cpu)
2) Speed gauss vs thomas (cpu)
3) Thomas (cpu) vs Thomas (gpu) time scaling 
"""

#Standard Imports 
import numpy as np
import time

###########################################################################
#Gauss row elimination with pivoting
###########################################################################
def generate_stiffness_matrix(half_n:int,eta:float = 1*pow(10,-5),F:float = -0.001):
    """
    M,r,x = generate_stiffness_matrix(half_n:int,eta:float = 1*pow(10,-5),F:float = -0.001)

    # Explanation
    This is for creating matrices for the gauss with pivoting method to solve
    It is also applicable to some of the thomas algorithim implimentations just not 
    the ones in this file. It should be noted due to the no slip condition the edge values 
    are set to 0 and therefore the matrix does not include endpoints.

    # Inputs 
    * half_n = half the number of elements desired (half to gaurentee an even number)
    * eta = coefficient of viscosity (SI units)
    * F = force aplied (SI units)

    # Outputs
    * M = stiffness matrix for a FEM-Galerkin Method 
    * r = associated solution for this 
    * x = x-axis spacings


    # Conditions 
    * No Slip boundary Condition
    * Equation 32.42 32.47 32.37 CHAPTER 32 Finite Element Method
    * g^{(0)}-g^{(1)} = \frac{\dp}{\dx}*\frac{1}{2 \eta} (\Delta Z)^2
    * -g^{(N-1)}-g^{(N)} = \frac{\dp}{\dx}*\frac{1}{2 \eta} (\Delta Z)^2
    * -g^(j-1) + 2g^(j)-g^(j+1) = \frac{\dp}{\dx}*\frac{1}{2 \eta}(\Delta z)^2
    * M doesn't change becuase it's based on ratios of the g(j) no the addition 
    * For solution of -1 to 1
    """
    #Sizing
    n = 2*half_n
    #Calculating dz step size
    dz = 1/half_n
    
    #Allocating Memory to matrices creating r and x
    M = np.zeros([n,n])
    r = np.ones(n)*(-F/eta*pow(dz,2))
    x = np.arange(-1+dz/2,1,dz)


    #Asembling M by 2 on diagnoals and -1 on 1st off diagonals 
    for i in range(n):
        M[i,i]=2
        if i>0:
            M[i,i-1] = -1
        if i<n-1:
            M[i,i+1] = -1
   
   #Returns
    return M,r,x


###########################################################################
#Thomas Algorithim for CPU
###########################################################################

"""
Index solutions for alpha and beta so matrix creation is no longer necessary.
"""
def beta(i:int):
    return (i+1)/2
def alpha(i:int):
    return (i+2)/(i+1)


def thomas_cpu(half_n,eta = 1*pow(10,-5),F = -0.001):
    #Sizing
    n = 2*half_n
    #Calculating dz step size
    dz = 1/half_n
    #scaling factor
    r0 = (-F/eta*pow(dz,2))

    #Allocating Memory to matrices creating r and x
    vx = np.zeros(n)#velocity
    x = np.arange(-1+dz/2,1,dz)#position

    x[n-1] = (n)/2

    for i in reversed(range(int(n/2),n-1)):
        x[i] = beta(i)+x[i+1]/alpha(i)

    #mirroring
    vx = np.zeros(n)
    vx[0:half_n]=half_vx
    vx[half_n:n]=np.flip(half_vx)

    #no need to reverse due to symmetry
    #scale by r0 
    return x*r0




if __name__ == "__main__":
    half_n = 5
    M,r,x = generate_stiffness_matrix(half_n)
    print(M,r,x)