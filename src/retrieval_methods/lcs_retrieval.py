import customtkinter as ctk
from tkinter import Toplevel, Label
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for _path in (_SRC, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import matplotlib.pyplot as plt
from retrieval_methods.Class_LCS import LCS
from utils.plots import Plots

plots = Plots()

def get_retrieved_images(Iref, Isamp, params):
    """
    Retrieves the retrieved images from the LCS class.

    Args:
        Iref (np.ndarray): Reference images.
        Isamp (np.ndarray): Sample images.
        params (dict): Parameters for the LCS class.
    """

    retrieved = LCS(I_refs=Iref, I_samps=Isamp, dict_params=params)

    n_subplots = (1,3)
    plot_size = (900, 200)
    plots_space = (0.05, 0.05)
    mark_bad = False
    scalebar_color="white"
    scalebar_pad=0.01
    scalebar_alpha= 0.5
    scalebar_fontsize= 10
    cbar_join = False
    fig, axes = plt.subplots(n_subplots[0], n_subplots[1], figsize=(plot_size[0]/72, plot_size[1]/72), gridspec_kw={'wspace': plots_space[0],'hspace': plots_space[1]})  

    plots.show_image(retrieved.T, None, mark_bad, axes[0], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Transmission", cbar_join)

    plots.show_image(retrieved.Dphi_x, None, mark_bad, axes[1], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Differential phase in X", cbar_join)

    plots.show_image(retrieved.Dphi_y, None, mark_bad, axes[2], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Differential phase in Y", cbar_join)

    fig.canvas.manager.set_window_title("Retrieved Images")
    plt.show()
        