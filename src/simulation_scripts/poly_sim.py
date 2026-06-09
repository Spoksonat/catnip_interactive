import numpy as np

from simulation_scripts.wf_marker import GratingEI
from simulation_scripts.sample import SampleEI
from simulation_scripts.simulation import SimulationEI

from simulation_scripts.wf_marker import GratingGBI
from simulation_scripts.sample import SampleGBI
from simulation_scripts.simulation import SimulationGBI

from simulation_scripts.wf_marker import Sandpaper
from simulation_scripts.sample import SampleSBI
from simulation_scripts.simulation import SimulationSBI

from simulation_scripts.simulation import SimulationInline

class Poly_simulationEI:

    def __init__(self, dict_params, save_path) -> None:
        """
        Initializes the Poly_simulationEI class for polychromatic or monochromatic simulation.

        Args:
            dict_params (dict): Dictionary of simulation parameters.
            save_path (str): Path to save simulation results.
        """

        self.dict_params = dict_params
        self.save_path = save_path
        self.type_of_spectrum = dict_params["Type of spectrum"]
        self.N = int(float(dict_params["Number of steps"]))
        self.binning_factor = int(float(dict_params["Binning factor"]))

        a_str, b_str = dict_params["FOV (pix)"].strip("()").split(",")
        self.img_size = (int(float(a_str))*self.binning_factor, int(float(b_str))*self.binning_factor)

        if(self.type_of_spectrum == "Poly"):
            spectrum_data = np.genfromtxt(dict_params["Energy (keV)"])
            self.Es = spectrum_data[:,0]
            self.S = spectrum_data[:,1]/np.sum(spectrum_data[:,1])
        elif(self.type_of_spectrum == "Mono"):
            self.Es = np.array([float(dict_params["Energy (keV)"])])
            self.S = np.array([1.0])
        else:
            raise ValueError("Invalid option for type of source")
        
        #self.obtain_poly_Irefs_Isamps()

    def single_sim(self, E, theta_y=0) -> tuple:
        """
        Runs a single simulation for a given energy and rotation angle.

        Args:
            E (float): Energy value in keV.
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Reference and sample intensity stacks.
        """

        self.grat = GratingEI(dict_params=self.dict_params,
                            E = E)
        

        self.samp = SampleEI(dict_params=self.dict_params,
                           E = E)
        
        self.sim = SimulationEI(dict_params=self.dict_params,
                              grat=self.grat, 
                              samp=self.samp,
                              E = E,
                              theta_y=theta_y)
        
        I_refs, I_samps = self.sim.create_ref_samp_stacks()

        return I_refs, I_samps
    
class Poly_simulationGBI:

    def __init__(self, dict_params, save_path) -> None:
        """
        Initializes the Poly_simulationSGBI class for SGBI simulation.

        Args:
            dict_params (dict): Dictionary of simulation parameters.
            save_path (str): Path to save simulation results.
        """

        self.dict_params = dict_params
        self.save_path = save_path
        self.type_of_spectrum = dict_params["Type of spectrum"]
        self.N = int(float(dict_params["Num. of steps per dir."]))
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.sim_pixel_m = float(dict_params["Sim. pixel (μm)"])*1e-6/self.binning_factor

        a_str, b_str = dict_params["FOV (pix)"].strip("()").split(",")
        self.img_size = (int(float(a_str)), int(float(b_str)))

        self.px_um = float(dict_params["Period in X (μm)"])
        self.py_um = float(dict_params["Period in Y (μm)"])
        self.px_pix = int(self.px_um*1e-6/self.sim_pixel_m) # Period in pixels
        self.py_pix = int(self.py_um*1e-6/self.sim_pixel_m) # Period in pixels
        ratio_y, ratio_x = round(self.img_size[0]/self.py_pix), round(self.img_size[1]/self.px_pix)
        self.img_size = (ratio_y*self.py_pix*self.binning_factor, ratio_x*self.px_pix*self.binning_factor)
        
        if(self.type_of_spectrum == "Poly"):
            spectrum_data = np.genfromtxt(dict_params["Energy (keV)"])
            self.Es = spectrum_data[:,0]
            self.S = spectrum_data[:,1]/np.sum(spectrum_data[:,1])
        elif(self.type_of_spectrum == "Mono"):
            self.Es = np.array([float(dict_params["Energy (keV)"])])
            self.S = np.array([1.0])
        else:
            raise ValueError("Invalid option for type of source")
        
        #self.obtain_poly_Irefs_Isamps()

    def single_sim(self, E, theta_y=0) -> tuple:
        """
        Runs a single SGBI simulation for a given energy and rotation angle.

        Args:
            E (float): Energy value in keV.
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Reference and sample intensity stacks.
        """

        self.grat = GratingGBI(dict_params=self.dict_params,
                            E = E)
        

        self.samp = SampleGBI(dict_params=self.dict_params,
                           E = E)
        
        self.sim = SimulationGBI(dict_params=self.dict_params,
                              grat=self.grat, 
                              samp=self.samp,
                              E = E,
                              theta_y = theta_y)
        
        I_refs, I_samps = self.sim.create_ref_samp_stacks()

        return I_refs, I_samps 
    
class Poly_simulationSBI:

    def __init__(self, dict_params, save_path) -> None:
        """
        Initializes the Poly_simulationSBI class for SBI simulation.

        Args:
            dict_params (dict): Dictionary of simulation parameters.
            save_path (str): Path to save simulation results.
        """

        self.dict_params = dict_params
        self.save_path = save_path
        self.type_of_spectrum = dict_params["Type of spectrum"]
        self.N = int(float(dict_params["Number of steps"]))
        self.binning_factor = int(float(dict_params["Binning factor"]))

        a_str, b_str = dict_params["FOV (pix)"].strip("()").split(",")
        self.img_size = (int(float(a_str))*self.binning_factor, int(float(b_str))*self.binning_factor)

        if(self.type_of_spectrum == "Poly"):
            spectrum_data = np.genfromtxt(dict_params["Energy (keV)"])
            self.Es = spectrum_data[:,0]
            self.S = spectrum_data[:,1]/np.sum(spectrum_data[:,1])
        elif(self.type_of_spectrum == "Mono"):
            self.Es = np.array([float(dict_params["Energy (keV)"])])
            self.S = np.array([1.0])
        else:
            raise ValueError("Invalid option for type of source")
        
        #self.obtain_poly_Irefs_Isamps()

    def single_sim(self, E, theta_y=0) -> tuple:
        """
        Runs a single SBI simulation for a given energy and rotation angle.

        Args:
            E (float): Energy value in keV.
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Reference and sample intensity stacks.
        """

        self.grat = Sandpaper(dict_params=self.dict_params,
                            E = E)
        

        self.samp = SampleSBI(dict_params=self.dict_params,
                           E = E)
        
        self.sim = SimulationSBI(dict_params=self.dict_params,
                              grat=self.grat, 
                              samp=self.samp,
                              E = E,
                              theta_y=theta_y)
        
        I_refs, I_samps = self.sim.create_ref_samp_stacks()

        return I_refs, I_samps
    
class Poly_simulationInline:

    def __init__(self, dict_params, save_path) -> None:
        """
        Initializes the Poly_simulationInline class for inline simulation.

        Args:
            dict_params (dict): Dictionary of simulation parameters.
            save_path (str): Path to save simulation results.
        """

        self.dict_params = dict_params
        self.save_path = save_path
        self.type_of_spectrum = dict_params["Type of spectrum"]
        self.binning_factor = int(float(dict_params["Binning factor"]))

        a_str, b_str = dict_params["FOV (pix)"].strip("()").split(",")
        self.img_size = (int(float(a_str))*self.binning_factor, int(float(b_str))*self.binning_factor)

        if(self.type_of_spectrum == "Poly"):
            spectrum_data = np.genfromtxt(dict_params["Energy (keV)"])
            self.Es = spectrum_data[:,0]
            self.S = spectrum_data[:,1]/np.sum(spectrum_data[:,1])
        elif(self.type_of_spectrum == "Mono"):
            self.Es = np.array([float(dict_params["Energy (keV)"])])
            self.S = np.array([1.0])
        else:
            raise ValueError("Invalid option for type of source")
        
        #self.obtain_poly_Irefs_Isamps()

    def single_sim(self, E, theta_y=0) -> tuple:
        """
        Runs a single inline simulation for a given energy and rotation angle.

        Args:
            E (float): Energy value in keV.
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Reference and sample intensity images.
        """

        self.samp = SampleEI(dict_params=self.dict_params,
                              E = E)
        
        self.sim = SimulationInline(dict_params=self.dict_params, 
                                    samp=self.samp,
                                    E = E,
                                    theta_y = theta_y)
        
        I_ref, I_samp = self.sim.create_ref_samp()

        return I_ref, I_samp