import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_derivative(s):
    s = softmax(s).reshape(-1, 1)  # Softmax vector
    return np.diagflat(s) - np.dot(s, s.T)


print(softmax_derivative([11, 10, 15]))