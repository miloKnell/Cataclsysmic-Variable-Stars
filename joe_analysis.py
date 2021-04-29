import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv('joe_mega.csv')

start = 5 #10 hour max #2.4
N = 100 #40
step = 0.001
k = np.arange(start,N,step)


def read_psd(filename):
    with open(filename,'r') as f:
        psd = f.read().strip().split('\n')
    psd = np.array(psd).astype(np.float32)
    return psd

psd = read_psd('psd.txt')
plt.plot(k**-1,psd)
plt.show()
