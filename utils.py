import numpy as np

# Wrapper fucntion for np.max(), for readability
def ReLU(x : float) -> float:
    zeros = np.zeros((x.shape))
    return np.maximum(x, zeros)


def logistic(x : float) -> float:
    return 1/(1 + np.exp(-x))