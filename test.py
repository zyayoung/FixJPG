import numpy as np


enu = np.arange(100)
# enu = np.concatenate([2*enu, 2*enu+1])
enu = enu.reshape(2,-1).T.reshape(-1)
print(enu)
