import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import pandas as pd
import pickle

with open('cbas.pkl','rb') as f: #open data file, pickle is a format which is a binary file
    cbas = pickle.load(f)

def avg_dis(df):
    '''Find the average difference in sampling'''
    indices = df.index.tolist() #indices of the data file, which are the time samples
    return np.average([indices[i+1] - indices[i] for i in range(len(indices[:-1]))]) #get the difference between the current i and the next one

def plot_fft(df,ax):
    data = df['Obj1']
    b_fft = fft(data)
    T=avg_dis(df)
    N=len(b_fft)
    f = np.linspace(0,1/T, N) #create x axis
    ax.plot((f[1:]**-1)*24,np.abs(b_fft)**2[1:]) #do 1/x-axis * 24 to go from frequency to period, in hours

def plot_regular(df,ax):
    data = df['Obj1']
    ax.plot(data.index.to_list(),data.to_list())


def plot_with_subplots():
    '''Plot each night, show raw data and fft'''
    f,axes=plt.subplots(2,9,sharey='row',sharex='row')

    for df,ax in zip(cbas,axes[0]):
        plot_fft(df,ax)

    for df,ax in zip(cbas,axes[1]):
        plot_regular(df,ax)

    plt.show()


def plot_one_graph():
    '''Plot every night on a single graph'''
    for cba in cbas:
        plot_fft(cba,plt)
    plt.show()
