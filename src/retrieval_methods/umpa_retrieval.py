import customtkinter as ctk
from tkinter import Toplevel, Label
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for _path in (_SRC, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import matplotlib.pyplot as plt
import sys
from retrieval_methods.Class_UMPA import UMPA
from utils.plots import Plots

plots = Plots()

def get_retrieved_images(Iref, Isamp, params):
    """
    Retrieves the retrieved images from the UMPA class.

    Args:
        Iref (np.ndarray): Reference images.
        Isamp (np.ndarray): Sample images.
        params (dict): Parameters for the UMPA class.
    """

    Nw = params["Nw"]
    max_shift = params["max_shift"]
    df_bool = params["df_bool"]

    dict_params = {"Nw": int(float(Nw)), "max_shift": int(float(max_shift)), "df_bool": df_bool}

    retrieved = UMPA(I_refs=Iref, I_samps=Isamp, dict_params=dict_params)
    n_subplots = (2,2)
    plot_size = (800, 600)
    plots_space = (0.05, 0.2)
    mark_bad = False
    scalebar_color="white"
    scalebar_pad=0.01
    scalebar_alpha= 0.5
    scalebar_fontsize= 10
    cbar_join = False
    fig, axes = plt.subplots(n_subplots[0], n_subplots[1], figsize=(plot_size[0]/72, plot_size[1]/72), gridspec_kw={'wspace': plots_space[0],'hspace': plots_space[1]})  

    if(n_subplots[0]*n_subplots[1] != 1):
        axes = axes.flatten()
    else:
        axes = [axes]

    plots.show_image(retrieved.T, None, mark_bad, axes[0], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Transmission", cbar_join)

    plots.show_image(retrieved.D, None, mark_bad, axes[1], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Dark Field", cbar_join, colormap=plt.cm.gist_yarg.copy())

    plots.show_image(retrieved.dx, None, mark_bad, axes[2], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Differential Phase in X", cbar_join)

    plots.show_image(retrieved.dy, None, mark_bad, axes[3], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Differential Phase in Y", cbar_join)

    fig.canvas.manager.set_window_title("Retrieved Images")
    plt.show()
