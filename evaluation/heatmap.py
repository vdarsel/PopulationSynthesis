import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cm

def generate_color_map(x,y, bins=100, min_freq=1):
    hist = np.histogram2d(x,y,bins)[0].T[::-1]
    hist_flatten = hist.flatten()
    norm = clr.SymLogNorm(10, vmin=np.min(hist_flatten[hist_flatten>0]), vmax=np.max(hist_flatten[hist_flatten>0]))
    hist[hist<min_freq] = np.nan
    _,ax = plt.subplots()
    im = ax.imshow((hist), cmap=cm.jet, norm= norm)
    ax.figure.colorbar(im, ax=ax, )
    plt.axis('off')
    plt.show()

def generate_color_map_filter(x,y, bins=100, filter_bins=100, min_freq=1):
    hist = np.histogram2d(x,y,bins)[0].T[:filter_bins,:filter_bins][::-1]
    hist_flatten = hist.flatten()
    norm = clr.SymLogNorm(10, vmin=np.min(hist_flatten[hist_flatten>0]), vmax=np.max(hist_flatten[hist_flatten>0]))
    hist[hist<min_freq] = np.nan
    _,ax = plt.subplots()
    im = ax.imshow((hist), cmap=cm.plasma, norm= norm)
    ax.figure.colorbar(im, ax=ax, )
    plt.axis('off')
    plt.show()


def generate_color_map_save(x,y, path, name, bins=100, min_freq=1):
    hist = np.histogram2d(x,y,bins)[0].T[::-1]
    hist_flatten = hist.flatten()
    norm = clr.SymLogNorm(10, vmin=np.min(hist_flatten[hist_flatten>0]), vmax=np.max(hist_flatten[hist_flatten>0]))
    hist[hist<min_freq] = np.nan
    _,ax = plt.subplots()
    ax.plot([0,1],[0,1], color="red", alpha=0.5)
    im = ax.imshow((hist), cmap=cm.plasma, norm= norm, extent = [0,1,0,1])
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.figure.colorbar(im, ax=ax, )
    plt.savefig(f'{path}/{name}.png')
    plt.close()

def generate_color_map_filter_save(x,y, path, name, bins=100, filter_bins=100, min_freq=1):
    hist = np.histogram2d(x,y,bins)[0].T[:filter_bins,:filter_bins][::-1]
    hist_flatten = hist.flatten()
    norm = clr.SymLogNorm(10, vmin=np.min(hist_flatten[hist_flatten>0]), vmax=np.max(hist_flatten[hist_flatten>0]))
    hist[hist<min_freq] = np.nan
    _,ax = plt.subplots()
    ax.plot([0,1],[0,1], color="red", alpha=0.5)
    im = ax.imshow((hist), cmap=cm.plasma, norm= norm, extent = [0,filter_bins/bins,0,filter_bins/bins])
    ax.set_xlim([0,filter_bins/bins])
    ax.set_ylim([0,filter_bins/bins])
    ax.figure.colorbar(im, ax=ax, )
    plt.savefig(f'{path}/{name}.png')
    plt.close()