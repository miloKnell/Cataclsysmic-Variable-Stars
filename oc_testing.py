import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import itertools
from scipy.signal import argrelextrema
import math

from scipy.interpolate import interp1d

def ndft(x, f, k):
    '''non-equispaced discrete Fourier transform'''
    return np.dot(f, np.exp(-2j * np.pi * k * x[:, np.newaxis]))

start = 5 #10 hour max #2.4
N = 100 #40
step = 0.001
k = np.arange(start,N,step)

def get_ndft(df):
    x=df['Date'].values
    f=df['Obj1'].values
    f_k = ndft(x,f,k)
    psd = np.abs(f_k)**2
    return psd


def norm(df,n_sigmas=5,verbose=True):
    '''subtract mean light and sigma clip'''
    mean = np.sum(df['Obj1'])/len(df)
    std = np.std(df['Obj1'])
    to_drop=df[abs(df['Obj1']-mean)/std>n_sigmas]
    df=df.drop(to_drop.index) #drop outliers with sigma clipping
    df['Obj1']=df['Obj1']-mean
    if verbose == True:
        print(len(to_drop))
    return df.reset_index()


'''
period = 0.0119 #period of AM-CVN
df  = pd.read_csv('am894751.hz',sep='\s+',names=['Date','Obj1'])
#df = df[1171:2033]
x = df['Date']
y=df['Obj1']
plt.plot(x,y)
#plt.scatter(x%period,y)
#plt.show()

#ndft = get_ndft(df)
#plt.plot(k**-1,ndft)
#plt.show()
'''



#Use newer BH-lyn data
period = 0.15583333333333335
df  = pd.read_csv('bh986-00.ne',sep='\s+',names=['Date','Obj1'])
df=norm(df,n_sigmas=5)
x = df['Date']
y=df['Obj1']
f = interp1d(x,y)
y = pd.Series(f(x))

from sklearn.cluster import DBSCAN
cluster = DBSCAN(eps=0.3,min_samples=10).fit_predict(x.values.reshape(-1,1))
df['Cluster'] = cluster #seperate nights
'''
fig,axes = plt.subplots(11,1,sharey=True)
for (i,group),ax in zip(df.groupby(df['Cluster']),axes):
    x=group['Date']
    y=group['Obj1']
    #ax.plot(x%period/period,y,'bo',x%period/period+1,y,'bo') #periods
    ax.plot(k**-1,get_ndft(group)) #ndft
    #ax.axvline(period)
    #ax.plot(x,y) #on many

plt.show()




ndft = get_ndft(df)
plt.plot(k**-1,ndft)
plt.show()'''



superhump = 0.0844*24 #0.078*24
#superhump = 0.1423 * 24
x = df['Date']
y=df['Obj1']
x = (x-x[0])*24

#x = x[df['Cluster']==1].reset_index(drop=True)
#y = y[df['Cluster']==1].reset_index(drop=True)

def get_window(x,y,start,length,pad,return_index=False,return_windows = False):
    minima = [start]
    windows = []
    tmp_window = []

    for point in x.iteritems():
        i,n = point
        if i <= start: #skip early rows
            pass
        elif (n % pad < pad) and (n % (length+pad) > pad and n % (length+pad) < (length+pad)):
            tmp_window.append(point)
        else:
            if len(tmp_window) != 0:
                windows.append(tmp_window)
                tmp_window = []
            

    if len(tmp_window) !=0:
        windows.append(tmp_window)



    for window in windows:
        start_idx = window[0][0]
        local_minima_idx = argrelextrema(np.array([y[w[0]] for w in window]),np.less,order=1)[0]
        local_minima_idx += start_idx
        #local_minima_idx = argrelextrema(np.array([y[w[0]] for w in window]),np.less_equal,order=2)
        local_minima = [y[i] for i in local_minima_idx]
        i = np.argmin(local_minima)
        minima.append(local_minima_idx[i])

    y_minima = [y[i] for i in minima]

    if return_windows == True:
        return windows

    if return_index == True:
        return minima


    return y_minima

length = superhump*(0.9)
pad = superhump*(0.1)

x = x-x.iloc[0]

start = np.argmin(y[x < pad+length])

x = x-x[start]

minima = get_window(x,y,start,length,pad,return_index=True)
windows = get_window(x,y,start,length,pad,return_windows=True)
window = windows[1]


start_idx = window[0][0]

local_minima_idx = argrelextrema(np.array([y[w[0]] for w in window]),np.less,order=1)[0]

local_minima_idx+=start_idx

x_w = [x[w[0]] for w in window]
y_w = [y[w[0]] for w in window]

x_min = [x[i] for i in local_minima_idx]
y_min = [y[i] for i in local_minima_idx]



def plot_oc():
    o = np.array([x[i] for i in minima])
    c = np.arange(min(x),max(x),step=superhump)

    oc = []
    for obs in o:
        idx = (np.abs(c-obs).argmin())
        oc.append(obs-c[idx])
    plt.plot(oc)
    plt.show()


def plot_minima():
    plt.gca().invert_yaxis()
    plt.plot(x,y, 'bo', markersize=2)
    plt.plot([x[i] for i in minima],[y[i] for i in minima],'ko',markersize=5)
    plt.show()

def plot_ndft():
    ndft = get_ndft(df)
    plt.plot(k**-1,ndft)
    plt.show()
