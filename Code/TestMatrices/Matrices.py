import scipy.io as sio
import numpy as np

file_loc = r"C:\Users\schwa\Documents\GitHub\APPM4600FinalProjectSparseSolvers\Code\TestMatrices\Matrices\1138_bus.mat"
data = sio.loadmat(file_loc)
print(data)