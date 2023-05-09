"""
This is just a simple script to run the commands several times and save the times to later be averages

1) Scaling comparision of the memory required for solving a nxn with gauss and thomas (cpu)
2) Speed gauss vs thomas (cpu)
3) Thomas (cpu) vs Thomas (gpu) time scaling 
"""

# Standard Imports
import numpy as np
import numba

import tracemalloc
import time
import sys


###########################################################################
# Gauss row elimination 
###########################################################################
def generate_stiffness_matrix(
    half_n: int, eta: float = 1 * pow(10, -5), F: float = -0.001
) -> np.ndarray | np.ndarray | np.ndarray:
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
    * M = stiffness matrix for a FEM-Galerkin Method (n by n numpy array)
    * r = associated solution for M (1 by n numpy array)
    * x = x-axis spacings (1 by n numpy array)

    # Conditions
    * No Slip boundary Condition
    * Equation 32.42 32.47 32.37 CHAPTER 32 Finite Element Method
    * g^{(0)}-g^{(1)} = \frac{\dp}{\dx}*\frac{1}{2 \eta} (\Delta Z)^2
    * -g^{(N-1)}-g^{(N)} = \frac{\dp}{\dx}*\frac{1}{2 \eta} (\Delta Z)^2
    * -g^(j-1) + 2g^(j)-g^(j+1) = \frac{\dp}{\dx}*\frac{1}{2 \eta}(\Delta z)^2
    * M doesn't change becuase it's based on ratios of the g(j) no the addition
    * For solution of -1 to 1
    """
    # Sizing
    n = 2 * half_n
    # Calculating dz step size
    dz = 1 / half_n

    # Allocating Memory to matrices creating r and x
    M = np.zeros([n, n])
    r = np.ones(n) * (-F / eta * pow(dz, 2))
    x = np.arange(-1 + dz / 2, 1, dz)

    # Asembling M by 2 on diagnoals and -1 on 1st off diagonals
    for i in range(n):
        M[i, i] = 2
        if i > 0:
            M[i, i - 1] = -1
        if i < n - 1:
            M[i, i + 1] = -1

    # Returns
    return M, r, x


def gauss_elimination(A, b):
    n = len(A)
    M = np.zeros([n, n + 1])
    x = np.zeros(n)
    # augment matrix
    for i in range(n):
        for j in range(n):
            M[i, j] = A[i, j]
        M[i, n] = b[i]

    # Fwd elim
    for i in range(n - 1):
        for j in range(i + 1, n):
            val = M[j, i] / M[i, i]
            M[j, :] = M[j, :] - M[i, :] * val

    # Bck elimination
    for i in reversed(range(n)):
        temp = 0
        for j in range(n):
            temp = temp + x[j] * M[i, j]
        x[i] = (M[i, n] - temp) / M[i, i]

    return x


###########################################################################
# Thomas Algorithim for CPU
###########################################################################


def thomas_cpu(
    half_n: int, eta: float = 1 * pow(10, -5), F: float = -0.001
) -> np.ndarray | np.ndarray:
    """
    vx,x = thomas_cpu(half_n:int,eta:float = 1*pow(10,-5),F:float = -0.001)

    # Explanation
    This solves for finite flow in an infintessimal channel using FEA-Gerlikin and
    the thomas algorithim. It should be noted due to the no slip condition the edge values
    are set to 0 and therefore the matrix does not include endpoints.

    # Inputs
    * half_n = half the number of elements desired (half to gaurentee an even number)
    * eta = coefficient of viscosity (SI units)
    * F = force aplied (SI units)

    # Outputs
    * vx = velocity at each x (1 by n numpy array)
    * x = x-axis spacings (1 by n numpy array)
    """

    # Index solutions for alpha and beta so matrix creation is no longer necessary.
    # Included as subfunctions so the GPU optimization remains unaffected
    def beta(i: int):
        return (i + 1) / 2

    def alpha(i: int):
        return (i + 2) / (i + 1)

    # Sizing
    n = 2 * half_n
    # Calculating dz step size
    dz = 1 / half_n
    # scaling factor
    r0 = -F / eta * pow(dz, 2)

    # Allocating Memory to matrices creating r and x
    vx = np.zeros(n)  # velocity
    x = np.arange(-1 + dz / 2, 1, dz)  # position

    # creating vx initial entry
    vx[n - 1] = (n) / 2

    # Back subsitution of vx
    # Due to symmetry only half is solved
    for i in reversed(range(int(n / 2), n - 1)):
        vx[i] = beta(i) + vx[i + 1] / alpha(i)

    # mirroring for full solution
    vx[0:half_n] = np.flip(vx[half_n:n])

    # Scaled after because this is more efficient
    # Returning vx and x
    return vx * r0, x


###########################################################################
# Thomas Algorithim for GPU
###########################################################################


@numba.njit
def thomas_gpu(
    half_n: int, eta: float = 1 * pow(10, -5), F: float = -0.001
) -> np.ndarray | np.ndarray:
    """
    vx,x = thomas_gpu(half_n:int,eta:float = 1*pow(10,-5),F:float = -0.001)

    # Explanation
    This solves for finite flow in an infintessimal channel using FEA-Gerlikin and
    the thomas algorithim. It should be noted due to the no slip condition the edge values
    are set to 0 and therefore the matrix does not include endpoints.

    This is not as memory efficent as the cpu version but is focused on speed

    # Inputs
    * half_n = half the number of elements desired (half to gaurentee an even number)
    * eta = coefficient of viscosity (SI units)
    * F = force aplied (SI units)

    # Outputs
    * vx = velocity at each x (1 by n numpy array)
    * x = x-axis spacings (1 by n numpy array)
    """

    def beta(i: int):
        return (i + 1) / 2

    def alpha(i: int):
        return (i + 2) / (i + 1)

    # Sizing
    n = 2 * half_n
    # Calculating dz step size
    dz = 1 / half_n
    # scaling factor
    r0 = -F / eta * pow(dz, 2)

    # Allocating Memory to matrices creating r and x
    vx = np.zeros(n)  # velocity
    x = np.arange(-1 + dz / 2, 1, dz)  # position

    # reversed list
    rev_n = np.flip(np.arange(half_n, n - 1))
    # alpha and betas calculations
    beta_vals_half = beta(rev_n)
    alpha_vals_half = alpha(rev_n)

    # creating vx initial entry
    vx[n - 1] = half_n

    # Back subsitution of vx split into parts for optimization
    for i in rev_n:
        vx[i] = beta_vals_half[n - i - 2] + vx[i + 1] / alpha_vals_half[n - i - 2]

    # mirroring for full solution
    vx[0:half_n] = np.flip(vx[half_n:n])

    # Scaled after because this is more efficient
    # Returning vx and x
    return vx * r0, x

def thomas_algorithm(M, r):
    """
    Standard Thomas algorithm that is valid for use on the CPU
    This is version 2 which takes in a full matrix and has the goal of making
    a memory-efficient version.
    """
    # Getting iteration constant
    n = len(r)

    # Allocating memory for solution vector
    x = np.zeros(n)

    # Priming solution
    r[0] = r[0] / M[0, 0]
    M[0, 1] = M[0, 1] / M[0, 0]

    # Forward substitution on matrix
    for i in range(1, n - 1):
        r[i] = (r[i] - M[i, i - 1] * r[i - 1]) / (M[i, i] - M[i, i - 1] * M[i - 1, i])
        M[i, i + 1] = M[i, i + 1] / (M[i, i] - M[i, i - 1] * M[i - 1, i])

    # final step in forward substitutions
    r[n - 1] = (r[n - 1] - M[n - 1, n - 2] * r[n - 2]) / (
        M[n - 1, n - 1] - M[n - 1, n - 2] * M[n - 2, n - 1]
    )

    # First step in backward substitution
    x[n - 1] = r[n - 1]

    # Backward substitution to solve
    for i in reversed(range(n - 1)):
        x[i] = r[i] - M[i, i + 1] * x[i + 1]

    return x
if __name__ == "__main__":
    half_n = 5

    ################################################################################
    # Gauss Row Elimination
    ################################################################################
    M, r, x = generate_stiffness_matrix(half_n)
    start = time.time()
    vx1 = gauss_elimination(M, r)
    print("Gauss", time.time() - start)

    vx4 = thomas_algorithm(M, r)
    print("Thomas",vx4)

    ################################################################################
    # CPU Thomas
    ################################################################################
    start = time.time()
    vx2, x = thomas_cpu(half_n)
    print("CPU", time.time() - start)

    ################################################################################
    # GPU Thomas
    ################################################################################
    # calling this will compile the function then the time should be representitive
    thomas_gpu(half_n)

    start = time.time()
    vx3, x = thomas_gpu(half_n)
    print("GPU", time.time() - start)

    # Verifying all methods are equivelent
    # Has to be is close becasue gauss elimination acrues more errors
    if np.allclose(vx2, vx3):
        print("All methods are equal")

    print(
        "Begining Tests##########################################################"
    )
    max_time = 10
    time_test_n = 20  # number of tests per n for an average
    half_n_range = np.arange(10, pow(10,3))

    ################################################################################
    # Gauss Running time and memory tests
    ################################################################################
    if False:
        print("Begining Gauss Row Elimination Time Tests")
        gauss_times = np.zeros([len(half_n_range), time_test_n])

        for ind_n, half_n in enumerate(half_n_range):
            for i in range(time_test_n):
                start = time.time()
                M, r, x = generate_stiffness_matrix(half_n)
                vx1 = gauss_elimination(M, r)
                end = time.time()
                gauss_times[ind_n, i] = end-start        
            sys.stdout.write("\r" + str(ind_n) + " of " + str(len(half_n_range) - 1) +" Last one took:"+ str(end-start))
        np.savetxt("GaussTimes.csv", gauss_times, delimiter=",")
    ################################################################################
    # Gauss Running time and memory tests
    ################################################################################
    if True:
        print("Begining Gauss Row Elimination Time Tests")
        Thomas = np.zeros([len(half_n_range), time_test_n])

        for ind_n, half_n in enumerate(half_n_range):
            for i in range(time_test_n):
                M, r, x = generate_stiffness_matrix(half_n)
                start = time.time()
                vx1 = thomas_algorithm(M, r)
                end = time.time()
                Thomas[ind_n, i] = end-start        
            sys.stdout.write("\r" + str(ind_n) + " of " + str(len(half_n_range) - 1) +" Last one took:"+ str(end-start))
        np.savetxt("Thomas.csv", Thomas, delimiter=",")

    ################################################################################
    # CPU Running time and memory tests
    ################################################################################
    if False:

        print("\nBegining Thomas CPU Time Tests")
        cpu_times = np.zeros([len(half_n_range), time_test_n])

        for ind_n, half_n in enumerate(half_n_range):
            start = time.time()

            for i in range(time_test_n):
                x, vx = thomas_cpu(half_n)
                
            
            end = time.time()
            sys.stdout.write("\r" + str(ind_n) + " of " + str(len(half_n_range) - 1) +" Last one took:"+ str(end-start))

            cpu_times[ind_n, i] = (end-start)/time_test_n
        np.savetxt("ThomasCPUTimes.csv", cpu_times, delimiter=",")

    ################################################################################
    # GPU Running time and memory tests
    ################################################################################
    if False:
        print("\nBegining Thomas GPU Time Tests")
        gpu_times = np.zeros([len(half_n_range), time_test_n])

        for ind_n, half_n in enumerate(half_n_range):
            for i in range(time_test_n):
                start = time.time()
                x, vx = thomas_gpu(half_n)
                end = time.time()
                gpu_times[ind_n, i] = end - start
                sys.stdout.write(
                    "\r" + str(ind_n) + " of " + str(len(half_n_range) - 1)
                )

        np.savetxt("ThomasGPUTimes.csv", gpu_times, delimiter=",")

    print("\nCompletted All Tests##############################################")
