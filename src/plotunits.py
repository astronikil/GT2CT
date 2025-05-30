import numpy as np
import matplotlib.pyplot as plt

def plot_section(slice, cmap = plt.cm.Greys_r, 
                 fig_width = 8, fig_height = 8, 
                 vmin=None, vmax=None, ax = None, fig = None):

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(fig_width, fig_height)
    
    if vmin is not None:
        if ax is None:
            plt.imshow(slice, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(slice, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        if ax is None:
            plt.imshow(slice, cmap=cmap)
        else:
            ax.imshow(slice, cmap=cmap)
    ax.axis("off")
    return fig, ax
