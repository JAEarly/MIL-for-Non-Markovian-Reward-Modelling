import numpy as np
import pandas as pd
from matplotlib import colors
import seaborn as sns
from .ax import single_axis


def smoothed_time_series(ax, radius, step, data=None, csv_name=None, mean_mode=False, q=(5,25), suppress_plot=False, colour="k", shade_alpha=0.5):
    """
    Plot smoothed time series from the columns of an array.
    Show median as a curve and percentiles specified in q as shading.
    """
    if data is None:
        data = pd.read_csv(f"{csv_name}.csv").values
    k_range = np.arange(0, data.shape[0]+step, step)
    if mean_mode:
        stats = []
        for k in k_range:
            window = data[max(k-radius,0):k+radius].mean(axis=0)
            stats.append([np.nanmean(window), np.nanmin(window), np.nanmax(window)])
            shade_num = 1
        stats = np.array(stats).T
    else:
        q = sorted(list(q))
        shade_num = len(q)
        if shade_num > 0: assert q[0] >= 0 and q[-1] < 50
        q_all = [50] + q + [100-qq for qq in reversed(q)]
        assert radius > step
        stats = np.array([np.nanpercentile(data[max(k-radius,0):k+radius], q=q_all) for k in k_range]).T
    if not suppress_plot:
        ax.plot(k_range, stats[0], c=colour) 
        if shade_num > 0:
            c = np.array(colors.to_rgba(colour))
            c[3] = shade_alpha / shade_num # Reduce alpha to overlay shading.
            for i in range(1,shade_num+1):
                ax.fill_between(k_range, stats[i], stats[-i], color=c, zorder=-1, lw=0)
    return stats

def coloured_2d_plot(ax, data=None, csv_name=None, dim_names=None, colour_by="idx", cmap="flare_r", cmap_lims=None, plot_type="line", alpha=1):
    """
    Plot scatter points or lines from a sequence of one- or two-column arrays, 
    coloured by either sequence index or within-array timestep.
    Then create a colour bar.
    """
    if plot_type == "line":      plot_func = ax.plot
    elif plot_type == "scatter": plot_func = ax.scatter
    else: raise ValueError
    if type(cmap) == str: cmap = sns_cmap(cmap)
    if data is None:
        df = pd.read_csv(f"{csv_name}.csv")
        split_points = np.array(list(df["time"] == 0)[1:] + [True]) 
        data = np.split(df[dim_names].values, np.where(split_points)[0]+1)

    if not isinstance(colour_by, str):
        assert isinstance(colour_by, np.ndarray) and all([len(x) == len(b) for x, b in zip(data, colour_by)])
    if cmap_lims is not None:
        low, high = cmap_lims
    else:
        if colour_by == "idx":    low, high = 0, max(1, len(data) - 1)
        elif colour_by == "time": low, high = 0, max(len(x) - 1 for x in data)
        else:                     low, high = colour_by.min(), colour_by.max()
    for i, x in enumerate(data):
        if len(x.shape) == 1: x = x.reshape(-1,1)
        n, d = x.shape; assert d in {1,2}, f"{i}: data must be one- or two-dimensional"
        if isinstance(colour_by, str) and colour_by == "idx": 
            if d == 1: plot_func(range(n), x[:,0], color=cmap(i / high)) # TODO: Is there a way to avoid duplication?
            else:      plot_func(x[:,0], x[:,1], color=cmap(i / high))
        else:
            if plot_type == "scatter": raise NotImplementedError("Colouring not working")
            if isinstance(colour_by, str) and colour_by == "time": c = [cmap(t / high) for t in range(len(x)-1)]
            else:                                                  c = [cmap((b - low) / (high - low)) for b in colour_by[i]]
            if d == 1:
                for t in range(len(x)-1):
                    plot_func(range(t,t+2), x[t:t+2,0], color=c[t], alpha=alpha)
            else:
                for t in range(len(x)-1):
                    plot_func(x[t:t+2,0], x[t:t+2,1], color=c[t], alpha=alpha)
    return cbar(ax, low, high, cmap=cmap)

def cbar(ax, low, high, label=None, t_inches=0.1, ori="v", cmap="flare_r", resolution=50):
    """
    Create a colour bar with the same width or height as the given axis.
    """
    if type(cmap) == str: cmap = sns_cmap(cmap)
    w_inches, h_inches = ax.figure.get_size_inches()
    data = np.linspace(low, high, num=resolution).reshape(-1,1)
    if ori == "v": 
        cbar_ax = single_axis(t_inches, h_inches, x_ticks=[], y_ticks=[low, high], right=True)
        cbar_ax.spines["bottom"].set_visible(False)
        cbar_ax.set_ylabel(label)
        extent=[0, 1, low, high]
    elif ori == "h": 
        data = data.T
        cbar_ax = single_axis(w_inches, t_inches, x_ticks=[low, high], y_ticks=[], top=True)
        cbar_ax.spines["left"].set_visible(False)
        cbar_ax.set_xlabel(label)
        extent=[low, high, 0, 1]
    else: raise ValueError()
    cbar_ax.imshow(data, origin="lower", aspect="auto", extent=extent,  
                   interpolation="none", cmap=cmap, vmin=low, vmax=high) # TODO: Make imshow function?
    return cbar_ax

def sns_cmap(cmap): return sns.color_palette(cmap, as_cmap=True)
