import numpy as np
    


def thomas_alg_cpu(M,r):
    n = len(r)

    alpha = np.zeros(n)
    beta = np.zeros(n)
    x = np.zeros(n)

    alpha[0] = M[0,0]
    beta[0] = r[0]/M[0,0]

    for i in range(1,n):
        alpha[i] = M[i,i]-M[i-1,i]*M[i,i-1]/alpha[i-1]
        beta[i] = (r[i]-M[i-1,i]*beta[i-1])/alpha[i]
    
    x[n-1] = beta[n-1]

    for i in range(n-1):
        x[n-2-i] = beta[n-2-i]-M[n-2-i,n-1-i]*x[n-1-i]/alpha[n-2-i]

    return x

        





if __name__ == "__main__":
    M = np.array([[4,8,0,0],[8,18,2,0],[0,2,5,1.5],[0,0,1.5,1.75]])
    r = np.array([8,18,0.5,-1.75])

    x = thomas_alg_cpu(M,r)
    print(x.dot(M)-r)


