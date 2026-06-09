import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
for _path in (_SRC, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from simulation_scripts.poly_sim import Poly_simulationEI
from simulation_scripts.poly_sim import Poly_simulationInline
from simulation_scripts.poly_sim import Poly_simulationSBI
from simulation_scripts.poly_sim import Poly_simulationGBI
import numpy as np
import matplotlib.pyplot as plt
import threading
import csv
from tkinter import filedialog
from utils.plots import Plots

plots = Plots()

#--------------------- EI Simulations ------------------------------

def show_alignment_EI(dict_params):
    """
    Runs a single-step EI simulation for alignment and displays a histogram and reference image.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    dict_params_new = dict_params.copy()
    dict_params_new["Number of steps"] = "1"
    dict_params_new["Type of spectrum"] = "Mono"
    dict_params_new["Energy (keV)"] = "21.0"
    
    poly_sim = Poly_simulationEI(dict_params=dict_params_new, save_path=None)
    I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))

    for i, E in enumerate(poly_sim.Es):
        print(f"Obtaining reference and sample images for Energy = {E} keV")
        I_refs, I_samps = poly_sim.single_sim(E=E)
        I_refs_poly += poly_sim.S[i]*I_refs
        I_samps_poly += poly_sim.S[i]*I_samps
    
        for i, E in enumerate(poly_sim.Es):
            print(f"Obtaining reference and sample images for Energy = {E} keV")
            I_refs, I_samps = poly_sim.single_sim(E=E)
            I_refs_poly += poly_sim.S[i]*I_refs
            I_samps_poly += poly_sim.S[i]*I_samps

    image = I_refs_poly[0,:,:]
    odd_columns = image[:, 1::2]
    even_columns = image[:, 0::2]

    n_subplots = (1,2)
    plot_size = (900, 300)
    plots_space = (0.05, 0.05)
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

    plots.show_hist(np.ravel(odd_columns), ax=axes[0], fig=fig, title="Odd and Even Pixels Histogram", label="Odd pixels")
    plots.show_hist(np.ravel(even_columns), ax=axes[0], fig=fig, title="Odd and Even Pixels Histogram", label="Even pixels")
    axes[0].set_xlabel("Intensity (a.u.)")

    plots.show_image(image, dict_params["Sim. pixel (μm)"], mark_bad, axes[1], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Reference Image", cbar_join)

    fig.canvas.manager.set_window_title("Alignment Histogram")
    plt.show()

def run_EI_sim(dict_params, save_path):
    """
    Runs a full EI simulation and saves reference/sample images and configuration to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """

    print("Running simulation...")
    
    poly_sim = Poly_simulationEI(dict_params=dict_params, save_path=save_path)
    I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
    for i, E in enumerate(poly_sim.Es):
        print(f"Obtaining reference and sample images for Energy = {E} keV")
        I_refs, I_samps = poly_sim.single_sim(E=E)
        I_refs_poly += poly_sim.S[i]*I_refs
        I_samps_poly += poly_sim.S[i]*I_samps

    np.save(poly_sim.save_path + "/I_refs_poly.npy", I_refs_poly)
    np.save(poly_sim.save_path + "/I_samps_poly.npy", I_samps_poly)

    t_map_1 = poly_sim.samp.create_sample()[0]
    t_map_1_binned = poly_sim.sim.binning(t_map_1)

    np.save(poly_sim.save_path + "/t_map_1.npy", t_map_1_binned)
    np.save(poly_sim.save_path + "/Param_Card.npy", dict_params)

    print("Simulation finished.")

def run_CT_EI_sim(dict_params, save_path):
    """
    Runs a CT EI simulation over a range of angles and saves results to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    initial_angle = float(dict_params["Initial angle (deg)"])
    final_angle = float(dict_params["Final angle (deg)"])
    N_of_proj = int(float(dict_params["Number of projections"]))
    ct_angles = np.linspace(initial_angle, final_angle, N_of_proj, endpoint=False)
    print("Running simulation...")
    
    poly_sim = Poly_simulationEI(dict_params=dict_params, save_path=save_path)
    I_refs_poly_ct = np.zeros((N_of_proj, poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samps_poly_ct = np.zeros((N_of_proj, poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
    for i_ang, theta_y in enumerate(ct_angles):
        print(f"Obtaining reference and sample images for CT angle = {theta_y} deg")
        I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
        I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
        for i, E in enumerate(poly_sim.Es):
            print(f"Obtaining reference and sample images for Energy = {E} keV")
            I_refs, I_samps = poly_sim.single_sim(E=E, theta_y=theta_y)
            I_refs_poly += poly_sim.S[i]*I_refs
            I_samps_poly += poly_sim.S[i]*I_samps
        I_refs_poly_ct[i_ang,:,:] = I_refs_poly
        I_samps_poly_ct[i_ang,:,:] = I_samps_poly
    np.save(poly_sim.save_path + "/I_ref_poly_ct.npy", I_refs_poly_ct)
    np.save(poly_sim.save_path + "/I_samp_poly_ct.npy", I_samps_poly_ct)
    np.save(poly_sim.save_path + "/Param_Card.npy", dict_params)
    print("Simulation finished.")

#--------------------- Inline Simulations ------------------------------

def show_reference_Inline(dict_params):
    """
    Runs a single-step Inline simulation and displays reference and sample images.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """

    dict_params_new = dict_params.copy()
    dict_params_new["Type of spectrum"] = "Mono"
    dict_params_new["Energy (keV)"] = "21.0"
    
    poly_sim = Poly_simulationInline(dict_params=dict_params_new, save_path=None)
    I_ref_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim. binning_factor)))
    I_samp_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim. binning_factor)))
    
    for i, E in enumerate(poly_sim.Es):
        print(f"Obtaining reference and sample images for Energy = {E} keV")
        I_ref, I_samp = poly_sim.single_sim(E=E)
        I_ref_poly += poly_sim.S[i]*I_ref
        I_samp_poly += poly_sim.S[i]*I_samp
    
    image = I_ref_poly
    odd_columns = image[:, 1::2]
    even_columns = image[:, 0::2]

    n_subplots = (1,2)
    plot_size = (600, 250)
    plots_space = (0.05, 0.05)
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
    plots.show_image(image, dict_params["Sim. pixel (μm)"], mark_bad, axes[0], fig,  scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Reference Image", cbar_join)
    plots.show_image(I_samp_poly, dict_params["Sim. pixel (μm)"], mark_bad, axes[1], fig,  scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Sample Image", cbar_join)
    fig.canvas.manager.set_window_title("Reference and Sample Images")
    plt.show()

def run_Inline_sim(dict_params, save_path):
    """
    Runs a full Inline simulation and saves reference/sample images to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    print("Running simulation...")
    
    poly_sim = Poly_simulationInline(dict_params=dict_params, save_path=save_path)
    I_ref_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim. binning_factor)))
    I_samp_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim. binning_factor)))
    
    for i, E in enumerate(poly_sim.Es):
        print(f"Obtaining reference and sample images for Energy = {E} keV")
        I_ref, I_samp = poly_sim.single_sim(E=E)
        I_ref_poly += poly_sim.S[i]*I_ref
        I_samp_poly += poly_sim.S[i]*I_samp
    np.save(poly_sim.save_path + "/I_ref_poly.npy", I_ref_poly)
    np.save(poly_sim.save_path + "/I_samp_poly.npy", I_samp_poly)
    t_map_1 = poly_sim.samp.create_sample()[0]
    t_map_1_binned = poly_sim.sim.binning(t_map_1)
    np.save(poly_sim.save_path + "/t_map_1.npy", t_map_1_binned)
    print("Simulation finished.")

def run_Inline_sim_ct(dict_params, save_path):
    """
    Runs a CT Inline simulation over a range of angles and saves results to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    initial_angle = float(dict_params["Initial angle (deg)"])
    final_angle = float(dict_params["Final angle (deg)"])
    N_of_proj = int(float(dict_params["Number of projections"]))
    ct_angles = np.linspace(initial_angle, final_angle, N_of_proj, endpoint=False)
    print("Running simulation...")
    poly_sim = Poly_simulationInline(dict_params=dict_params, save_path=save_path)
    I_ref_poly_ct = np.zeros((N_of_proj, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samp_poly_ct = np.zeros((N_of_proj, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
    for i_ang, theta_y in enumerate(ct_angles):
        print(f"Obtaining reference and sample images for CT angle = {theta_y} deg")
        I_ref_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
        I_samp_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
        for i, E in enumerate(poly_sim.Es):
            print(f"Obtaining reference and sample images for Energy = {E} keV")
            I_ref, I_samp = poly_sim.single_sim(E=E, theta_y=theta_y)
            I_ref_poly += poly_sim.S[i]*I_ref
            I_samp_poly += poly_sim.S[i]*I_samp
        I_ref_poly_ct[i_ang,:,:] = I_ref_poly
        I_samp_poly_ct[i_ang,:,:] = I_samp_poly
    np.save(poly_sim.save_path + "/I_ref_poly_ct.npy", I_ref_poly_ct)
    np.save(poly_sim.save_path + "/I_samp_poly_ct.npy", I_samp_poly_ct)
    np.save(poly_sim.save_path + "/Param_Card.npy", dict_params)

    print("Simulation finished.")

#--------------------- SBI Simulations ------------------------------

def show_speckles(dict_params):
    """
    Runs a single-step SBI simulation and displays the reference speckle pattern.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    dict_params_new = dict_params.copy()
    dict_params_new["Number of steps"] = "1"
    dict_params_new["Type of spectrum"] = "Mono"
    dict_params_new["Energy (keV)"] = "21.0"
    poly_sim = Poly_simulationSBI(dict_params=dict_params_new, save_path=None)
    I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
    for i, E in enumerate(poly_sim.Es):
        print(f"Obtaining reference and sample images for Energy = {E} keV")
        I_refs, I_samps = poly_sim.single_sim(E=E)
        I_refs_poly += poly_sim.S[i]*I_refs
        I_samps_poly += poly_sim.S[i]*I_samps
    
    image = I_refs_poly[0,:,:]
    
    n_subplots = (1,1)
    plot_size = (300, 250)
    plots_space = (0.05, 0.05)
    mark_bad = False
    scalebar_color="white"
    scalebar_pad=0.01
    scalebar_alpha= 0.5
    scalebar_fontsize= 10
    cbar_join = False
    fig, axes = plt.subplots(n_subplots[0], n_subplots[1], figsize=(plot_size[0]/72, plot_size[1]/72), gridspec_kw={'wspace': plots_space[0],'hspace': plots_space[1]})  
    
    plots.show_image(image, dict_params["Sim. pixel (μm)"], mark_bad, axes, fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Reference Image", cbar_join)
    fig.canvas.manager.set_window_title("Reference Speckle Pattern")
    plt.show()

def run_SBI_sim(dict_params, save_path):
    """
    Runs a full SBI simulation and saves reference/sample images to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    print("Running simulation...")
    
    poly_sim = Poly_simulationSBI(dict_params=dict_params, save_path=save_path)
    I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
    for i, E in enumerate(poly_sim.Es):
        print(f"Obtaining reference and sample images for Energy = {E} keV")
        I_refs, I_samps = poly_sim.single_sim(E=E)
        I_refs_poly += poly_sim.S[i]*I_refs
        I_samps_poly += poly_sim.S[i]*I_samps
    np.save(poly_sim.save_path + "/I_refs_poly.npy", I_refs_poly)
    np.save(poly_sim.save_path + "/I_samps_poly.npy", I_samps_poly)
    t_map_1 = poly_sim.samp.create_sample()[0]
    t_map_1_binned = poly_sim.sim.binning(t_map_1)
    np.save(poly_sim.save_path + "/t_map_1.npy", t_map_1_binned)
    print("Simulation finished.")

def run_CT_SBI_sim(dict_params, save_path):
    """
    Runs a CT SBI simulation over a range of angles and saves results to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    initial_angle = float(dict_params["Initial angle (deg)"])
    final_angle = float(dict_params["Final angle (deg)"])
    N_of_proj = int(float(dict_params["Number of projections"]))
    ct_angles = np.linspace(initial_angle, final_angle, N_of_proj, endpoint=False)
    print("Running simulation...")
    poly_sim = Poly_simulationSBI(dict_params=dict_params, save_path=save_path)
    I_refs_poly_ct = np.zeros((N_of_proj, poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samps_poly_ct = np.zeros((N_of_proj, poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
    for i_ang, theta_y in enumerate(ct_angles):
        print(f"Obtaining reference and sample images for CT angle = {theta_y} deg")
        I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
        I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
        for i, E in enumerate(poly_sim.Es):
            print(f"Obtaining reference and sample images for Energy = {E} keV")
            I_refs, I_samps = poly_sim.single_sim(E=E, theta_y=theta_y)
            I_refs_poly += poly_sim.S[i]*I_refs
            I_samps_poly += poly_sim.S[i]*I_samps
        I_refs_poly_ct[i_ang,:,:] = I_refs_poly
        I_samps_poly_ct[i_ang,:,:] = I_samps_poly
    np.save(poly_sim.save_path + "/I_ref_poly_ct.npy", I_refs_poly_ct)
    np.save(poly_sim.save_path + "/I_samp_poly_ct.npy", I_samps_poly_ct)
    np.save(poly_sim.save_path + "/Param_Card.npy", dict_params)
    print("Simulation finished.")

#--------------------- GBI Simulations ------------------------------

def show_reference_GBI(dict_params):
    """
    Runs a single-step GBI simulation and displays the reference pattern.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    dict_params_new = dict_params.copy()
    dict_params_new["Num. of steps per dir."] = "1"
    dict_params_new["Type of spectrum"] = "Mono"
    dict_params_new["Energy (keV)"] = "21.0"
    poly_sim = Poly_simulationGBI(dict_params=dict_params_new, save_path=None)
    I_refs_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samps_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
    for i, E in enumerate(poly_sim.Es):
        print(f"Obtaining reference and sample images for Energy = {E} keV")
        I_refs, I_samps = poly_sim.single_sim(E=E)
        I_refs_poly += poly_sim.S[i]*I_refs
        I_samps_poly += poly_sim.S[i]*I_samps
    
    image = I_refs_poly[0,:,:]
    n_subplots = (1,1)
    plot_size = (300, 250)
    plots_space = (0.05, 0.05)
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
    plots.show_image(image, dict_params["Sim. pixel (μm)"], mark_bad, axes[0], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Reference Image", cbar_join)
    fig.canvas.manager.set_window_title("Reference Pattern")
    plt.show()

def run_GBI_sim(dict_params, save_path):
    """
    Runs a full GBI simulation and saves reference/sample images to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    print("Running simulation...")
    poly_sim = Poly_simulationGBI(dict_params=dict_params, save_path=save_path)
    I_refs_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samps_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    for i, E in enumerate(poly_sim.Es):
        print(f"Obtaining reference and sample images for Energy = {E} keV")
        I_refs, I_samps = poly_sim.single_sim(E=E)
        I_refs_poly += poly_sim.S[i]*I_refs
        I_samps_poly += poly_sim.S[i]*I_samps
    
    np.save(poly_sim.save_path + "/I_refs_poly.npy", I_refs_poly)
    np.save(poly_sim.save_path + "/I_samps_poly.npy", I_samps_poly)
    t_map_1 = poly_sim.samp.create_sample()[0]
    t_map_1_binned = poly_sim.sim.binning(t_map_1)
    np.save(poly_sim.save_path + "/t_map_1.npy", t_map_1_binned)
    print("Simulation finished.")



def run_CT_GBI_sim(dict_params, save_path):
    """
    Runs a CT GBI simulation over a range of angles and saves results to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    initial_angle = float(dict_params["Initial angle (deg)"])
    final_angle = float(dict_params["Final angle (deg)"])
    N_of_proj = int(float(dict_params["Number of projections"]))
    ct_angles = np.linspace(initial_angle, final_angle, N_of_proj, endpoint=False)
    print("Running simulation...")
    poly_sim = Poly_simulationGBI(dict_params=dict_params, save_path=save_path)
    I_refs_poly_ct = np.zeros((N_of_proj, poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samps_poly_ct = np.zeros((N_of_proj, poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    for i_ang, theta_y in enumerate(ct_angles):
        print(f"Obtaining reference and sample images for CT angle = {theta_y} deg")
        I_refs_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
        I_samps_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
        for i, E in enumerate(poly_sim.Es):
            print(f"Obtaining reference and sample images for Energy = {E} keV")
            I_refs, I_samps = poly_sim.single_sim(E=E, theta_y=theta_y)
            I_refs_poly += poly_sim.S[i]*I_refs
            I_samps_poly += poly_sim.S[i]*I_samps
        I_refs_poly_ct[i_ang,:,:] = I_refs_poly
        I_samps_poly_ct[i_ang,:,:] = I_samps_poly
    
    np.save(poly_sim.save_path + "/I_ref_poly_ct.npy", I_refs_poly_ct)
    np.save(poly_sim.save_path + "/I_samp_poly_ct.npy", I_samps_poly_ct)
    np.save(poly_sim.save_path + "/Param_Card.npy", dict_params)

    print("Simulation finished.")
