import numpy as np

def RAI(fan_in, fan_out):
    V = np.random.randn(fan_out, fan_in + 1) * 0.6007 / fan_in ** 0.5
    for j in range(fan_out):
        k = np.random.randint(0, high=fan_in + 1)
        V[j, k] = np.random.beta(2, 1)
    W = V[:, :-1]
    b = np.reshape(V[:, -1], (fan_out, 1))
    return W.astype(np.float32), b.astype(np.float32)
    

# layer 1
W1 = np.random.normal(0, 2 / 784, size=(16, 784))
b1 = np.zeros((16, 1))
    
# layer 2
W2, b2 = RAI(16, 16)

# layer 3
W3, b3 = RAI(16, 10)
