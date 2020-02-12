import numpy as np
import matplotlib.pyplot as plt



def ndft(x,f,k):
    """non-equispaced discrete Fourier transform"""
    return np.dot(f, np.exp(2j * np.pi * k * x[:, np.newaxis]))
def psd(x):
    return abs(x**2)

N=1000
max_ = 80

x = -0.5 + np.random.rand(N+1)[1:]
f = 5*np.sin(24 * 2 * np.pi * x)+5*np.cos(12*2*np.pi*x)+np.random.rand(N)

k = np.arange(max_).astype(float)[1:]
f_k = ndft(x,f,k)
k_transform = 24*k**-1

plt.plot(k_transform,psd(f_k))
#plt.plot(k_transform,f_k.real)
#plt.plot(k_transform,f_k.imag)
plt.show()
