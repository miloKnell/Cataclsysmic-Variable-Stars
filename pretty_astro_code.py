import numpy as np
import matplotlib.pyplot as plt
import pickle

def ndft(x, f, k):
    '''non-equispaced discrete Fourier transform'''
    return np.dot(f, np.exp(2j * np.pi * k * x[:, np.newaxis]))

N = 40
step=0.001
k = np.arange(step,N,step)

with open('cbas.pkl','rb') as f:
    cbas = pickle.load(f)

def get_ndft(df):
    x=df.index.values
    f=df['Obj1'].values
    f_k = ndft(x,f,k)
    psd = np.abs(f_k)**2
    return psd

def plot_ndft(df,ax):
    ax.plot(k,get_ndft(df))

def plot_regular(df,ax):
    data = df['Obj1']
    ax.plot(data.index.to_list(),data.to_list())

def plot_on_one():
    '''Plot the ndft for every night on one graph'''
    for cba in cbas:
       plot_ndft(cba,plt)
    plt.show()

def plot_sum():
    '''Plot one function, the sum of every night's ndft'''
    ndfts = [get_ndft(cba) for cba in cbas]
    full = [sum([ndft[i] for ndft in ndfts]) for i in range(len(k))]
    plt.plot(k,full)
    plt.show()

def plot_with_subplots():
    '''Plot each night, show raw data and ndft'''
    f,axes=plt.subplots(2,9,sharey='row')

    for df,ax in zip(cbas,axes[0]):
        plot_ndft(df,ax)

    for df,ax in zip(cbas,axes[1]):
        plot_regular(df,ax)

    plt.show()
