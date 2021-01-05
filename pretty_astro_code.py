import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def read_both():
    both = []
    files = os.listdir('data')
    for file in files:
        file = os.path.join('data',file)
        both.append(pd.read_csv(file))

    return both
        

def write_both(both):
    for i,df in enumerate(both):
        df.to_csv('data/{}-both.csv'.format(i))


def ndft(x, f, k):
    '''non-equispaced discrete Fourier transform'''
    return np.dot(f, np.exp(-2j * np.pi * k * x[:, np.newaxis]))

start= 2.4 #10 hour max
N = 40
step=0.001
k = np.arange(start,N,step)

both = read_both()


def read_joe(file):
    with open(file,'r') as f:
        data=f.readlines()
    cleaned = pd.DataFrame([[d for d in line.strip().split(' ') if d!=''] for line in data]).astype(float)
    cleaned.columns =['Date','Obj1']
    return cleaned

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

def plot_regular(df,ax=plt):
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
    ndfts = [get_ndft(s) for s in ds]
    with sns.husl_palette(len(ds),h=.5,l=.55):
        for i,ndft in enumerate(ndfts):
           plot_ndft(ndft,ax,label=i)
           plot_local_maxima(ndft,ax)
        if show==True:
            plt.legend(loc='best')
            plt.show()

def plot_sum(ds,ax=plt,show=True):
    '''Plot one line, the sum of every night's ndft'''
    ndfts = [get_ndft(s) for s in ds]
    full = [sum([ndft[i] for ndft in ndfts]) for i in range(len(k))] #at each point in k grid, sum
    ax.plot(24*(k**-1),full)
    if show==True:
        plt.show()

def plot_with_subplots(ds,ax=plt,show=True,ratio=True):
    '''Plot each night, show raw data and ndft'''
    if ratio == True:
        ratio = [s['Date'].iloc[-1]-s['Date'][0] for s in ds] #the length of the observation as a ratio of width
    f,axes=plt.subplots(2,len(ds),sharey='row',gridspec_kw={'width_ratios':ratio})
    ndfts = [get_ndft(s) for s in ds]
    
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



def norm(ds,n_sigmas=5):
    '''subtract mean light and sigma clip'''
    dfs=[]
    dropped=0
    for df in ds:
        mean = np.sum(df['Obj1'])/len(df)
        std = np.std(df['Obj1'])
        to_drop=df[abs(df['Obj1']-mean)/std>n_sigmas]
        df=df.drop(to_drop.index) #drop outliers with sigma clipping
        dropped+=len(to_drop)
        df['Obj1']=df['Obj1']-mean
        dfs.append(df.reset_index())
    print(dropped)
    return dfs

def fold(ds,period,ax=plt,show=True):
    '''Fold along a period'''
    period=period/24
    for df in ds:
        x=(df['Date']-df['Date'][0])/period
        ax.plot(x,df['Obj1'])
        try: #different arg styles for plt and plt.ax
            ax.xlim(0,2)
        except:
            ax.set_xlim(0,2)
    if show==True:    
        plt.show()


def compare_folded_periods(ds,periods):
    '''Compare folded periods as subplots'''
    f,axes = plt.subplots(1,len(periods),sharey='row')
    #max_=max([max(df['Obj1']) for df in ds])
    for period,ax in zip(periods,axes):
        fold(ds,period,ax=ax,show=False)
        ax.title.set_text('Period of {} hours'.format(period))
        ax.set_xlim(0,2)
        #ax.axvline(1,0,max_,color='k',lw=1.2)
    plt.show()


def compare_sigma_clip(ds,clips):
    '''Compare sigma clips visually'''
    f,axes = plt.subplots(1,len(clips),sharey='row',sharex='row')
    for clip,ax in zip(clips,axes):
        ds=norm(ds,n_sigmas=clip)
        plot_on_one(ds,ax=ax,show=False)
        ax.title.set_text('Clipping at {} sigmas'.format(clip))
        ax.axvline(3.74,color='k',lw=1.2)
    plt.show()
    
def plot_ndft_joe(df):
    '''Takes a single dataframe that has been preprocessed and displays the ndft'''
    ndfts = get_ndft(df)
    plt.plot((k**-1)*24,ndfts)
    plt.show()
    return ndfts

def norm_joe(df,n_sigmas):
    dropped=0
    mean = np.sum(df['Obj1'])/len(df)
    std = np.std(df['Obj1'])
    to_drop=df[abs(df['Obj1']-mean)/std>n_sigmas]
    df=df.drop(to_drop.index) #drop outliers with sigma clipping
    dropped+=len(to_drop)
    df['Obj1']=df['Obj1']-mean
    print(dropped)
    return df.reset_index()

def plot_joe(df):
    plt.plot(df['Date'],df['Obj1'])
    plt.show()

def night_sep_joe(df,n_sigmas=2):
    diffs=[]
    for i in range(len(joe)):
        diff=joe['Date'][i+1]-joe['Date'][i]
        diffs.append(diff)
        if i==len(joe)-2:
            break
    mean=np.mean(diffs)
    std=np.std(diffs)
    df=pd.Series(diffs)
    return df[abs(df-mean)/std>n_sigmas]

def split_joe(df,splits=None):
    if splits is None:
        splits=night_sep_joe(df)
    ds=[]
    splits=splits.index
    for i in range(len(splits)):
        start=splits[i]
        stop=splits[i+1]
        new_df=df[start+1:stop].reset_index()
        ds.append(new_df)
        if i==len(splits)-2:
            break
    return ds

joe=read_joe('bh986-00.ne')
joe=norm_joe(joe,n_sigmas=3)
plot_ndft_joe(joe)
#joe_split=split_joe(joe)
#plot_with_subplots(joe_split)

#both=norm(both,n_sigmas=5)
#joined=pd.concat(both)
#plot_ndft_joe(joined)
#compare_sigma_clip(both,[1,1.5,2,2.5,3,4])    
#plot_on_one(both)
#plot_sum(both)
#compare_folded_periods(both,[1.5,1.9,3.74,4.75,5.3])
