import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns  


def ndft(x, f, k):
    '''non-equispaced discrete Fourier transform'''
    return np.dot(f, np.exp(-2j * np.pi * k * x[:, np.newaxis]))

start= 2.4 #10 hour max
N = 40
step=0.001
k = np.arange(start,N,step)

with open('cba.pkl','rb') as f:
    cba = pickle.load(f)

with open('ulw.pkl','rb') as f:
    ulw = pickle.load(f)

with open('both.pkl','rb') as f:
    both = pickle.load(f)
    
with open('ndfts.pkl','rb') as f:
    ndfts = pickle.load(f)


def get_ndft(df):
    x=df['Date'].values
    f=df['Obj1'].values
    f_k = ndft(x,f,k)
    psd = np.abs(f_k)**2
    return psd

def get_plot_ndft(df,ax):
    ax.plot((k**-1)*24,get_ndft(df))

def plot_ndft(ndft,ax,label=None):
    ax.plot((k**-1)*24,ndft,label=label)

def plot_regular(df,ax):
    ax.plot(df['Date'].to_list(),df['Obj1'].to_list())


def plot_local_maxima(ndft,ax,return_only=False):
    max_indices = np.r_[True, ndft[1:] > ndft[:-1]] & np.r_[ndft[:-1] > ndft[1:], True] #funky np.r_
    maxima = ndft[max_indices]
    arg_max = (k[max_indices]**-1)*24
    if return_only == True:
        return arg_max,maxima
    ax.plot(arg_max,maxima,'ko',markersize=3)
    






def plot_on_one(ds,ax=plt,show=True):
    '''Plot the ndft for every night on one graph'''
    #ndfts = [get_ndft(s) for s in ds]
    with sns.husl_palette(len(ds),h=.5,l=.55):
        for i,ndft in enumerate(ndfts):
           plot_ndft(ndft,ax,label=i)
           plot_local_maxima(ndft,ax)
        plt.legend(loc='best')
        if show==True:
            plt.show()

def plot_sum(ds,ax=plt,show=True):
    '''Plot one function, the sum of every night's ndft'''
    #ndfts = [get_ndft(s) for s in ds]
    full = [sum([ndft[i] for ndft in ndfts]) for i in range(len(k))]
    ax.plot(k,full)
    if show==True:
        plt.show()

def plot_with_subplots(ds,ax=plt,show=Trueratio=True):
    '''Plot each night, show raw data and ndft'''
    if ratio == True:
        ratio = [s['Date'].iloc[-1]-s['Date'][0] for s in ds]
    f,axes=plt.subplots(2,len(ds),sharey='row',gridspec_kw={'width_ratios':ratio})
    #ndfts = [get_ndft(s) for s in ds]
    
    for i,(ndft,ax) in enumerate(zip(ndfts,axes[0])):
        plot_ndft(ndft,ax)
        plot_local_maxima(ndft,ax)
        ax.title.set_text(i)

    for i,(df,ax) in enumerate(zip(ds,axes[1])):
        plot_regular(df,ax)
        ax.title.set_text(i)

    plt.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)
    f.tight_layout()
    if show==True:
        plt.show()




def norm(ds):
    '''make end of each night the same as the start of the next'''
    dfs=[]
    for i,df in enumerate(ds):
        start=df['Obj1'].iloc[0]
        if i==0:
            end=start
        df['Obj1']=df['Obj1']-(start-end)
        end=df['Obj1'].iloc[-1]
        dfs.append(df)
    return dfs

def fold(ds,period,ax=plt,show=True):
    '''in progress folding'''
    period=period/24
    for df in ds:
        x=(df['Date']-df['Date'][0])/period
        ax.plot(x,df['Obj1'])
    if show==True:    
        plt.show()

both=norm(both)

#compare on one
'''plot_on_one(both,show=False)
max_=max([max(ndft) for ndft in ndfts])
plt.axvline(3.74,0,max_)
plt.axvline(4.75,0,max_)
plt.show()'''


#compare folded subplots
'''f,axes = plt.subplots(1,2)
fold(both,4.75,ax=axes[0],show=False)
axes[0].title.set_text('4.75')
fold(both,3.74,ax=axes[1],show=False)
axes[1].title.set_text('3.74')
plt.show()'''
