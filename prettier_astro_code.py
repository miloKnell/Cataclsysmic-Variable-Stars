import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


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
    return df.reset_index())


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
period = 0.15583333333333335 #(0.078=superhump?)
df  = pd.read_csv('bh986-00.ne',sep='\s+',names=['Date','Obj1'])
df=norm(df,n_sigmas=5)
x = df['Date']
y=df['Obj1']

from sklearn.cluster import DBSCAN
cluster = DBSCAN(eps=0.3,min_samples=10).fit_predict(x.values.reshape(-1,1))
df['Cluster'] = cluster #seperate nights

fig,axes = plt.subplots(11,1,sharey=True)
for (i,group),ax in zip(df.groupby(df['Cluster']),axes):
    x=group['Date']
    y=group['Obj1']
    #ax.plot(x%period/period,y,'bo',x%period/period+1,y,'bo') #periods
    ax.plot(k**-1,get_ndft(group)) #ndft
    #ax.axvline(period)
    #ax.plot(x,y) #on many

#plt.scatter(x,y) #on one
plt.show()


#ndft = get_ndft(df)
#plt.plot(k**-1,ndft)
#plt.show()

