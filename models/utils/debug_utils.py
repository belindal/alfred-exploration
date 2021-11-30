import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_mask(M, file_name, vmin=0, vmax=1):
    fig, ax = plt.subplots()
    cmap = mpl.cm.get_cmap()
    cmap.set_bad(color="white")
    im = ax.imshow(M, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im)
    fig.savefig(file_name + ".png" if ".png" not in file_name else file_name, dpi=150)
