import time
import matplotlib.pyplot as plt
import numpy as np
#% matplotlib tk

def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

def plot_2d_data_with_chirp(dataset,v,pts,t_idx=(-2,10),cmap='seismic',c_wl=450):
    'Dataset: resultlike or raw data'
    'v: tuple with min/max value of colormap (-3,3)'
    't_idx: tuple with min/max timevalues'
    'cmap: String with matplotlib colormap'
    # Unpack XR dataarray to numpy arrays
    A = dataset.data
    t = A.time.to_numpy()
    wv = A.spectral.to_numpy()
    d = A.to_numpy()
    print(t_idx)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(15,8))

    # Fit picked points and make a plot
    T = pts[:,0]
    wl = (pts[:,1]-c_wl)/100
    z = np.polyfit(T,wl, 3)
    print(z)
    p = np.poly1d(z)
    xp = np.linspace(4,8)-c_wl/100
    xpp = xp*100+c_wl
    ax.plot(p(xp),xpp,lw=3,c='b')
    plt.show()
    
    c = ax.pcolormesh(t,wv,np.transpose(d), shading='nearest',cmap=cmap, vmin=v[0], vmax=v[1])
    ax.axis([t_idx[0], t_idx[1], wv.min(), wv.max()])
    fig.colorbar(c, ax=ax)

    
def plot_2d_data(dataset,v,t_idx=(-2,10),cmap='seismic'):
    'Dataset: resultlike or raw data'
    'v: tuple with min/max value of colormap (-3,3)'
    't_idx: tuple with min/max timevalues'
    'cmap: String with matplotlib colormap'

    # Unpack XR dataarray to numpy arrays
    A = dataset.data
    t = A.time.to_numpy()
    wv = A.spectral.to_numpy()
    d = A.to_numpy()
    print(t_idx)
    # Prepare figure
    fig, ax = plt.subplots(figsize=(15,8))
    c = ax.pcolormesh(t,wv,np.transpose(d), shading='nearest',cmap=cmap, vmin=v[0], vmax=v[1])
    ax.axis([t_idx[0], t_idx[1], wv.min(), wv.max()])
    fig.colorbar(c, ax=ax)
    plt.show()

def estimate_chirp(dataset,v,t_idx=(-2,10),cmap='seismic',c_wl=450, poly_order=3):
    'Dataset: resultlike or raw data'
    'v: tuple with min/max value of colormap (-3,3)'
    't_idx: tuple with min/max timevalues'
    'cmap: String with matplotlib colormap'
    'Ouputs: first poly coefficient, second poly coefficient, etc, center '
    # Unpack XR dataarray to numpy arrays
    A = dataset.data
    t = A.time.to_numpy()
    wv = A.spectral.to_numpy()
    d = A.to_numpy()

    ## first program loop pick points
    plot_2d_data(dataset,v,t_idx,cmap=cmap)
    tellme('You will give an estimation of the chirp, by clicking points')
    plt.waitforbuttonpress()

    while True:
        pts = []
        tellme('Select a maximum of 40 points to estimate the chirp, press middle mouse button to stop')
        pts = np.asarray(plt.ginput(40, timeout=-1))
        plt.close()
        break

    # Second program loop plot the fit figure
    fig, ax = plt.subplots(figsize=(15,10))

    # Fit picked points and make a plot
    T = pts[:,0]
    wl = (pts[:,1]-c_wl)/100
    z = np.polyfit(wl,T, poly_order)
    p = np.poly1d(z)
    xp = np.linspace(3,9)-c_wl/100
    xpp = xp*100+c_wl
    ax.plot(p(xp),xpp,lw=3,c='b')
    plt.show()
    
    c = ax.pcolormesh(t,wv,np.transpose(d), shading='nearest',cmap=cmap, vmin=v[0], vmax=v[1])
    ax.axis([t_idx[0], t_idx[1], wv.min(), wv.max()])
    fig.colorbar(c, ax=ax)
    print(z)
    return z, pts