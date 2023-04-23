import numpy as np
    


def thomas_alg_cpu(M,r):
    """
    Standard thomas algorithim that is valid for use on the cpu 
    This is version 1 which takes in a full matrix and creates 3 
    arrays to solve although not the most memory efficent this is the 
    easiest method to follow

    there have been no checks built into this
    """
    #finding the nxn size of matrix
    n = len(r)

    #allocating memory to matrices
    alpha = np.zeros(n)
    beta = np.zeros(n)
    x = np.zeros(n)

    #alpha beta matrice priming solutions
    alpha[0] = M[0,0]
    beta[0] = r[0]/M[0,0]

    #Forward subsitution to solve for alpha and beta
    for i in range(1,n):
        alpha[i] = M[i,i]-M[i-1,i]*M[i,i-1]/alpha[i-1]
        beta[i] = (r[i]-M[i-1,i]*beta[i-1])/alpha[i]
    
    #Priming x solutions
    x[n-1] = beta[n-1]

    #Backward subsitutions for x vector
    for i in reversed(range(n-1)):
        x[i-1] = beta[i-1]-M[i-1,i]*x[i]/alpha[i-1]

    #returning x vector
    return x

def thomas_alg_cpu_mem(M,r):
    """
    Standard thomas algorithim that is valid for use on the cpu 
    This is version 2 which takes in a full matrix has the goal of making
    a memory efficent version. 
    """
    #getting iteration constant
    n = len(r)

    #allocating memory for solution vector
    x = np.zeros(n)

    #Priming solution
    r[0] = r[0]/M[0,0]
    M[0,1] = M[0,1]/M[0,0]

    #Forward subsitution on matrix
    for i in range(1,n-1):
        r[i] = (r[i]-M[i,i-1]*r[i-1])/(M[i,i]-M[i,i-1]*M[i-1,i])
        M[i,i+1] = M[i,i+1]/(M[i,i] - M[i,i-1]*M[i-1,i])

    #final step in forward subsitutions 
    r[n-1] = (r[n-1]-M[n-1,n-2]*r[n-2])/(M[n-1,n-1]-M[n-1,n-2]*M[n-2,n-1])

    #first step in backward subsitution
    x[n-1] = r[n-1]

    #Backward subsitution to solve
    for i in reversed(range(n-1)):
        x[i] = r[i]-M[i,i+1]*x[i+1 ]

    return x
        





if __name__ == "__main__":
    M1 = np.array([[4,8,0,0],[8,18,2,0],[0,2,5,1.5],[0,0,1.5,1.75]])
    M2 = np.copy(M1)

    M2[3,0] = 1
    
    r = np.array([8,18,0.5,-1.75])

    x1 = thomas_alg_cpu(M1,r)
    x2 = thomas_alg_cpu(M2,r)

    if np.allclose(x1.dot(M1),r) and bool(~np.allclose(x2.dot(M2),r)):
        print("Passed Fully Tridiagonal Solution")
        
    else:
        print("Failed Fully Tridiagonal Test")


    x3 = thomas_alg_cpu_mem(M1,r)
    print(x1,x3)


