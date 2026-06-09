import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse, ellipsoid, polygon
import scipy.fft
import xraylib as xl
from scipy.ndimage import gaussian_filter, rotate
from utils.geometry import *

class SampleEI:

    def __init__(self, dict_params, E: float) -> None:

        self.dict_params = dict_params
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.sim_pixel_m = float(dict_params["Sim. pixel (μm)"])*1e-6/self.binning_factor
        self.t_m = float(dict_params["Whole sample thickness (mm)"])*1E-3 
        self.t_pix = int(self.t_m/self.sim_pixel_m)
        self.mat_1 = dict_params["Sample material"]
        self.mat_2 = dict_params["Background material"]
        if ("Sample 2 material" in dict_params):
            self.mat_3 = dict_params["Sample 2 material"]
        self.mat_1_density = float(dict_params["Sample material density (g/cc)"])
        self.mat_2_density = float(dict_params["Background material density (g/cc)"])
        if ("Sample 2 material density (g/cc)" in dict_params):
            self.mat_3_density = float(dict_params["Sample 2 material density (g/cc)"])
        else:
            self.mat_3_density = 0
        
        self.E = E
        self.sample_geom = dict_params["Geometry"]

        a_str, b_str = dict_params["FOV (pix)"].strip("()").split(",")
        self.img_size = (int(float(a_str))*self.binning_factor, int(float(b_str))*self.binning_factor)

        
        n = xl.Refractive_Index(self.mat_1, E, self.mat_1_density)
        self.delta_1 = 1 - n.real
        self.mu_mass_1 = xl.CS_Total_CP(self.mat_1, E)

        n = xl.Refractive_Index(self.mat_2, E, self.mat_2_density)
        self.delta_2 = 1 - n.real
        self.mu_mass_2 = xl.CS_Total_CP(self.mat_2, E)

        if ("Sample 2 material" in dict_params):

            n = xl.Refractive_Index(self.mat_3, E, self.mat_3_density)
            self.delta_3 = 1 - n.real
            self.mu_mass_3 = xl.CS_Total_CP(self.mat_3, E)

        else:
            self.delta_3 = 0
            self.mu_mass_3 = 0
        
        #self.mu = 2*k*beta # Attenuation coefficient in 1/m
        self.mu_1 = self.mu_mass_1*self.mat_1_density*1e2
        self.mu_2 = self.mu_mass_2*self.mat_2_density*1e2
        self.mu_3 = self.mu_mass_3*self.mat_3_density*1e2
    
    def create_mammo_phantom(self, theta_y=0):
        """
        Generates thickness maps for a mammography phantom with microcalcifications arranged in a pentagon.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3) corresponding to microcalcifications, PMMA, and wax regions.
        """

        t_map_uCs = np.zeros(self.img_size)

        d_calc_in_um = float(self.dict_params["Diameter uC (μm)"])
        R_region_in_um = float(self.dict_params["Radius pentagon (μm)"])

        R_pix = int( d_calc_in_um*1e-6/(2*self.sim_pixel_m) )
        R_region_in_pix = int( R_region_in_um*1e-6/(self.sim_pixel_m) )
        t_PMMA_pix = int(self.t_m/self.sim_pixel_m)
        t_wax_pix = int(float(self.dict_params["Thickness wax (mm)"])*1e-3/self.sim_pixel_m)
    
        n = 5
        roots = R_region_in_pix*np.exp(2j * np.pi * np.arange(n) / n)
    
        # Map roots to pixel coordinates relative to the center
        for root in roots:
            x = int(round(self.img_size[1]//2 + root.real))
            y = int(round(self.img_size[0]//2 - root.imag))  # Invert y-axis for image coordinates
        
            # Ensure the coordinates are within the image bounds
            if 0 <= y < self.img_size[0] and 0 <= x < self.img_size[1]:
                t_map_uCs += create_ellipse_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, a=R_pix, b=R_pix, c=R_pix, cx=x, cy=y)
    
        t_map_PMMA = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_PMMA_pix)
        t_map_wax = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_wax_pix)
    
        t_map_1 = t_map_uCs*self.sim_pixel_m
        t_map_3 = (t_map_wax - t_map_uCs)*self.sim_pixel_m
        t_map_2 = (t_map_PMMA - t_map_wax)*self.sim_pixel_m

        return t_map_1, t_map_2, t_map_3
    
    def create_block(self, theta_y=0):
        """
        Generates thickness maps for a block sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """
 
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)
        t_block_in_pix = int(float(self.dict_params["Block thickness (mm)"])*1e-3/self.sim_pixel_m)
    
        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)

        t_map_block = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1]/2, Ly=self.img_size[0]/2, Lz=t_block_in_pix)

    
        t_map_1 = t_map_block*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_block)*self.sim_pixel_m
    
        return t_map_1, t_map_2, np.zeros(self.img_size)

    def create_angio_tube(self, theta_y=0):
        """
        Generates thickness maps for an angiographic tube sample with background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3) corresponding to lumen, background, and wall.
        """

        d_ext_in_mm = float(self.dict_params["External diameter (mm)"])
        d_int_in_mm = float(self.dict_params["Internal diameter (mm)"])

        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)
        D_tube_in_pix = int(d_ext_in_mm*1e-3/self.sim_pixel_m)
        D_int_in_pix = int(d_int_in_mm*1e-3/self.sim_pixel_m)
    
        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)
        t_map_cylinder = create_cylinder_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, D=D_tube_in_pix, h=self.img_size[0])

        t_map_cylinder_int = create_cylinder_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, D=D_int_in_pix, h=self.img_size[0])
    
        t_map_1 = t_map_cylinder_int*self.sim_pixel_m
        t_map_3 = (t_map_cylinder - t_map_cylinder_int)*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_cylinder)*self.sim_pixel_m
    
        return t_map_1, t_map_2, t_map_3
    
    def create_fibre(self, theta_y=0):
        """
        Generates thickness maps for a fibre sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """

        d_tube_in_mm = float(self.dict_params["Fibre diameter (mm)"])
        d_tube_in_pix = int(d_tube_in_mm*1e-3/self.sim_pixel_m)
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)

        t_map_tube = create_cylinder_proj(theta_y = theta_y, theta_z=45, img_size=self.img_size, D=d_tube_in_pix, h=0.9*np.sqrt(self.img_size[0]**2 + self.img_size[1]**2))

        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)
        
        t_map_1 = t_map_tube*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_tube)*self.sim_pixel_m

        return t_map_1, t_map_2, np.zeros(self.img_size)
    
    def create_wedge(self, theta_y=0):
        """
        Generates thickness maps for a wedge sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """

        t_block_in_mm = float(self.dict_params["Wedge thickness (mm)"])
        t_block_in_pix = int(t_block_in_mm*1e-3/self.sim_pixel_m)
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)

        t_map_wedge = create_wedge_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx = self.img_size[1]/2, Ly = self.img_size[0]/2, Lz=t_block_in_pix)

        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)

        t_map_1 = t_map_wedge*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_wedge)*self.sim_pixel_m

        return t_map_1, t_map_2, np.zeros(self.img_size)
    
    def create_sphere_bkg(self, theta_y=0):
        """
        Generates thickness maps for a sphere sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """

        d_sph_samp_in_mm = float(self.dict_params["Sphere diameter (mm)"])
        R_in_pix = int(d_sph_samp_in_mm*1e-3/(2*self.sim_pixel_m))               # Radius in pixels
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)
        
        t_map_sph = create_ellipse_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, a = R_in_pix, b=R_in_pix, c=R_in_pix)

        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)

        t_map_1 = t_map_sph*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_sph)*self.sim_pixel_m

        return t_map_1, t_map_2, np.zeros(self.img_size)
        
    def create_sample(self, theta_y=0):
        """
        Selects and generates the thickness map according to the sample geometry defined in the parameters.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Thickness maps generated by the corresponding function for the selected geometry.
        """
        if(self.sample_geom == "Mammo Phantom"):
            return self.create_mammo_phantom(theta_y=theta_y)
        elif(self.sample_geom == "block"):
            return self.create_block(theta_y=theta_y)
        elif(self.sample_geom == "angio tube"):
            return self.create_angio_tube(theta_y=theta_y)
        elif(self.sample_geom == "sphere"):
            return self.create_sphere_bkg(theta_y=theta_y)
        elif(self.sample_geom == "fibre"):
            return self.create_fibre(theta_y=theta_y)
        elif(self.sample_geom == "wedge"):
            return self.create_wedge(theta_y=theta_y)
        else:
            raise ValueError("Select a valid sample")
        
class SampleGBI:

    def __init__(self, dict_params, E: float) -> None:

        self.dict_params = dict_params
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.sim_pixel_m = float(dict_params["Sim. pixel (μm)"])*1e-6/self.binning_factor
        self.t_m = float(dict_params["Whole sample thickness (mm)"])*1E-3 
        self.t_pix = int(self.t_m/self.sim_pixel_m)
        self.mat_1 = dict_params["Sample material"]
        self.mat_2 = dict_params["Background material"]
        if ("Sample 2 material" in dict_params):
            self.mat_3 = dict_params["Sample 2 material"]
        self.mat_1_density = float(dict_params["Sample material density (g/cc)"])
        self.mat_2_density = float(dict_params["Background material density (g/cc)"])
        if ("Sample 2 material density (g/cc)" in dict_params):
            self.mat_3_density = float(dict_params["Sample 2 material density (g/cc)"])
        else:
            self.mat_3_density = 0
        
        self.E = E
        self.sample_geom = dict_params["Geometry"]

        a_str, b_str = dict_params["FOV (pix)"].strip("()").split(",")
        self.img_size = (int(float(a_str)), int(float(b_str)))

        self.px_um = float(dict_params["Period in X (μm)"])
        self.py_um = float(dict_params["Period in Y (μm)"])
        self.px_pix = int(self.px_um*1e-6/self.sim_pixel_m) # Period in pixels
        self.py_pix = int(self.py_um*1e-6/self.sim_pixel_m) # Period in pixels
        ratio_y, ratio_x = round(self.img_size[0]/self.py_pix), round(self.img_size[1]/self.px_pix)
        self.img_size = (ratio_y*self.py_pix*self.binning_factor, ratio_x*self.px_pix*self.binning_factor)


        
        n = xl.Refractive_Index(self.mat_1, E, self.mat_1_density)
        self.delta_1 = 1 - n.real
        self.mu_mass_1 = xl.CS_Total_CP(self.mat_1, E)

        n = xl.Refractive_Index(self.mat_2, E, self.mat_2_density)
        self.delta_2 = 1 - n.real
        self.mu_mass_2 = xl.CS_Total_CP(self.mat_2, E)

        if ("Sample 2 material" in dict_params):

            n = xl.Refractive_Index(self.mat_3, E, self.mat_3_density)
            self.delta_3 = 1 - n.real
            self.mu_mass_3 = xl.CS_Total_CP(self.mat_3, E)

        else:
            self.delta_3 = 0
            self.mu_mass_3 = 0
        
        #self.mu = 2*k*beta # Attenuation coefficient in 1/m
        self.mu_1 = self.mu_mass_1*self.mat_1_density*1e2
        self.mu_2 = self.mu_mass_2*self.mat_2_density*1e2
        self.mu_3 = self.mu_mass_3*self.mat_3_density*1e2
    
    def create_mammo_phantom(self, theta_y=0):
        """
        Generates thickness maps for a mammography phantom with microcalcifications arranged in a pentagon.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3) corresponding to microcalcifications, PMMA, and wax regions.
        """

        t_map_uCs = np.zeros(self.img_size)

        d_calc_in_um = float(self.dict_params["Diameter uC (μm)"])
        R_region_in_um = float(self.dict_params["Radius pentagon (μm)"])

        R_pix = int( d_calc_in_um*1e-6/(2*self.sim_pixel_m) )
        R_region_in_pix = int( R_region_in_um*1e-6/(self.sim_pixel_m) )
        t_PMMA_pix = int(self.t_m/self.sim_pixel_m)
        t_wax_pix = int(float(self.dict_params["Thickness wax (mm)"])*1e-3/self.sim_pixel_m)
    
        n = 5
        roots = R_region_in_pix*np.exp(2j * np.pi * np.arange(n) / n)
    
        # Map roots to pixel coordinates relative to the center
        for root in roots:
            x = int(round(self.img_size[1]//2 + root.real))
            y = int(round(self.img_size[0]//2 - root.imag))  # Invert y-axis for image coordinates
        
            # Ensure the coordinates are within the image bounds
            if 0 <= y < self.img_size[0] and 0 <= x < self.img_size[1]:
                f = self.dict_params["RMS scattering angle in X (μrad)"]
                R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
                R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
                if( (f == "" or float(f) == 0.0) or float(f)==1.0):
                    t_map_uCs += create_ellipse_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, a=R_pix, b=R_pix, c=R_pix, cx=x, cy=y)
                else:
                    t_map_uCs += create_ellipse_proj_DF(theta_y = theta_y, theta_z=0, img_size=self.img_size, binning_factor=self.binning_factor, a=R_pix, b=R_pix, c=R_pix, f=float(f), R_microst=R_microst_pix, cx=x, cy=y)
    
        t_map_PMMA = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_PMMA_pix)
        t_map_wax = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_wax_pix)
    
        t_map_1 = t_map_uCs*self.sim_pixel_m
        t_map_3 = (t_map_wax - t_map_uCs)*self.sim_pixel_m
        t_map_2 = (t_map_PMMA - t_map_wax)*self.sim_pixel_m

        return t_map_1, t_map_2, t_map_3
    
    def create_block(self, theta_y=0):
        """
        Generates thickness maps for a block sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """
 
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)
        t_block_in_pix = int(float(self.dict_params["Block thickness (mm)"])*1e-3/self.sim_pixel_m)
    
        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)
        f = self.dict_params["RMS scattering angle in X (μrad)"]
        R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
        R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
        if( (f == "" or float(f) == 0.0) or float(f)==1.0):
            t_map_block = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1]/2, Ly=self.img_size[0]/2, Lz=t_block_in_pix)
        else:
            t_map_block = create_box_proj_DF(theta_y = theta_y, theta_z=0, img_size=self.img_size, binning_factor=self.binning_factor, Lx=self.img_size[1]/2, Ly=self.img_size[0]/2, Lz=t_block_in_pix, f=float(f), R_microst=R_microst_pix)
    
        t_map_1 = t_map_block*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_block)*self.sim_pixel_m
    
        return t_map_1, t_map_2, np.zeros(self.img_size)

    def create_angio_tube(self, theta_y=0):
        """
        Generates thickness maps for an angiographic tube sample with background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3) corresponding to lumen, background, and wall.
        """

        d_ext_in_mm = float(self.dict_params["External diameter (mm)"])
        d_int_in_mm = float(self.dict_params["Internal diameter (mm)"])

        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)
        D_tube_in_pix = int(d_ext_in_mm*1e-3/self.sim_pixel_m)
        D_int_in_pix = int(d_int_in_mm*1e-3/self.sim_pixel_m)
    
        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)
        t_map_cylinder = create_cylinder_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, D=D_tube_in_pix, h=self.img_size[0])
        f = self.dict_params["RMS scattering angle in X (μrad)"]
        R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
        R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
        if( (f == "" or float(f) == 0.0) or float(f)==1.0):
            t_map_cylinder_int = create_cylinder_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, D=D_int_in_pix, h=self.img_size[0])
        else:
            t_map_cylinder_int = create_cylinder_proj_DF(theta_y = theta_y, theta_z=0, img_size=self.img_size, binning_factor=self.binning_factor, D=D_int_in_pix, h=self.img_size[0], f=float(f), R_microst=R_microst_pix)
    
        t_map_1 = t_map_cylinder_int*self.sim_pixel_m
        t_map_3 = (t_map_cylinder - t_map_cylinder_int)*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_cylinder)*self.sim_pixel_m
    
        return t_map_1, t_map_2, t_map_3
    
    def create_fibre(self, theta_y=0):
        """
        Generates thickness maps for a fibre sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """

        d_tube_in_mm = float(self.dict_params["Fibre diameter (mm)"])
        d_tube_in_pix = int(d_tube_in_mm*1e-3/self.sim_pixel_m)
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)

        f = self.dict_params["RMS scattering angle in X (μrad)"]
        R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
        R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
        if( (f == "" or float(f) == 0.0) or float(f)==1.0):
            t_map_tube = create_cylinder_proj(theta_y = theta_y, theta_z=45, img_size=self.img_size, D=d_tube_in_pix, h=0.9*np.sqrt(self.img_size[0]**2 + self.img_size[1]**2))
        else:
            t_map_tube = create_cylinder_proj_DF(theta_y = theta_y, theta_z=45, img_size=self.img_size, binning_factor=self.binning_factor, D=d_tube_in_pix, h=0.9*np.sqrt(self.img_size[0]**2 + self.img_size[1]**2), f=float(f), R_microst=R_microst_pix).T

        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)
        
        t_map_1 = t_map_tube*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_tube)*self.sim_pixel_m

        return t_map_1, t_map_2, np.zeros(self.img_size)
    
    def create_wedge(self, theta_y=0):
        """
        Generates thickness maps for a wedge sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """

        t_block_in_mm = float(self.dict_params["Wedge thickness (mm)"])
        t_block_in_pix = int(t_block_in_mm*1e-3/self.sim_pixel_m)
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)

        f = self.dict_params["RMS scattering angle in X (μrad)"]
        R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
        R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
        if( (f == "" or float(f) == 0.0) or float(f)==1.0):
            t_map_wedge = create_wedge_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx = self.img_size[1]/2, Ly = self.img_size[0]/2, Lz=t_block_in_pix)
        else:
            t_map_wedge = create_wedge_proj_DF(theta_y = theta_y, theta_z=0, img_size=self.img_size, binning_factor=self.binning_factor, Lx = (self.img_size[1]/2), Ly = (self.img_size[0]/2), Lz=(t_block_in_pix), f=float(f), R_microst=R_microst_pix)

        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)

        t_map_1 = t_map_wedge*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_wedge)*self.sim_pixel_m

        return t_map_1, t_map_2, np.zeros(self.img_size)
    
    def create_sphere_bkg(self, theta_y=0):
        """
        Generates thickness maps for a sphere sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """

        d_sph_samp_in_mm = float(self.dict_params["Sphere diameter (mm)"])
        R_in_pix = int(d_sph_samp_in_mm*1e-3/(2*self.sim_pixel_m))               # Radius in pixels
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)
        
        f = self.dict_params["RMS scattering angle in X (μrad)"]
        R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
        R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
        if( (f == "" or float(f) == 0.0) or float(f)==1.0):
            t_map_sph = create_ellipse_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, a = R_in_pix, b=R_in_pix, c=R_in_pix)
        else:
            t_map_sph = create_ellipse_proj_DF(theta_y = theta_y, theta_z=0, img_size=self.img_size, binning_factor=self.binning_factor, a = R_in_pix, b=R_in_pix, c=R_in_pix, f=float(f), R_microst=R_microst_pix)

        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)

        t_map_1 = t_map_sph*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_sph)*self.sim_pixel_m

        return t_map_1, t_map_2, np.zeros(self.img_size)
        
    def create_sample(self, theta_y=0):
        """
        Selects and generates the thickness map according to the sample geometry defined in the parameters.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Thickness maps generated by the corresponding function for the selected geometry.
        """
        if(self.sample_geom == "Mammo Phantom"):
            return self.create_mammo_phantom(theta_y=theta_y)
        elif(self.sample_geom == "block"):
            return self.create_block(theta_y=theta_y)
        elif(self.sample_geom == "angio tube"):
            return self.create_angio_tube(theta_y=theta_y)
        elif(self.sample_geom == "sphere"):
            return self.create_sphere_bkg(theta_y=theta_y)
        elif(self.sample_geom == "fibre"):
            return self.create_fibre(theta_y=theta_y)
        elif(self.sample_geom == "wedge"):
            return self.create_wedge(theta_y=theta_y)
        else:
            raise ValueError("Select a valid sample")
        
class SampleSBI:

    def __init__(self, dict_params, E: float) -> None:

        self.dict_params = dict_params
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.sim_pixel_m = float(dict_params["Sim. pixel (μm)"])*1e-6/self.binning_factor
        self.t_m = float(dict_params["Whole sample thickness (mm)"])*1E-3 
        self.t_pix = int(self.t_m/self.sim_pixel_m)
        self.mat_1 = dict_params["Sample material"]
        self.mat_2 = dict_params["Background material"]
        if ("Sample 2 material" in dict_params):
            self.mat_3 = dict_params["Sample 2 material"]
        self.mat_1_density = float(dict_params["Sample material density (g/cc)"])
        self.mat_2_density = float(dict_params["Background material density (g/cc)"])
        if ("Sample 2 material density (g/cc)" in dict_params):
            self.mat_3_density = float(dict_params["Sample 2 material density (g/cc)"])
        else:
            self.mat_3_density = 0
        
        self.E = E
        self.sample_geom = dict_params["Geometry"]

        a_str, b_str = dict_params["FOV (pix)"].strip("()").split(",")
        self.img_size = (int(float(a_str))*self.binning_factor, int(float(b_str))*self.binning_factor)

        
        n = xl.Refractive_Index(self.mat_1, E, self.mat_1_density)
        self.delta_1 = 1 - n.real
        self.mu_mass_1 = xl.CS_Total_CP(self.mat_1, E)

        n = xl.Refractive_Index(self.mat_2, E, self.mat_2_density)
        self.delta_2 = 1 - n.real
        self.mu_mass_2 = xl.CS_Total_CP(self.mat_2, E)

        if ("Sample 2 material" in dict_params):

            n = xl.Refractive_Index(self.mat_3, E, self.mat_3_density)
            self.delta_3 = 1 - n.real
            self.mu_mass_3 = xl.CS_Total_CP(self.mat_3, E)

        else:
            self.delta_3 = 0
            self.mu_mass_3 = 0
        
        #self.mu = 2*k*beta # Attenuation coefficient in 1/m
        self.mu_1 = self.mu_mass_1*self.mat_1_density*1e2
        self.mu_2 = self.mu_mass_2*self.mat_2_density*1e2
        self.mu_3 = self.mu_mass_3*self.mat_3_density*1e2
    
    def create_mammo_phantom(self, theta_y=0):
        """
        Generates thickness maps for a mammography phantom with microcalcifications arranged in a pentagon.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3) corresponding to microcalcifications, PMMA, and wax regions.
        """

        t_map_uCs = np.zeros(self.img_size)

        d_calc_in_um = float(self.dict_params["Diameter uC (μm)"])
        R_region_in_um = float(self.dict_params["Radius pentagon (μm)"])

        R_pix = int( d_calc_in_um*1e-6/(2*self.sim_pixel_m) )
        R_region_in_pix = int( R_region_in_um*1e-6/(self.sim_pixel_m) )
        t_PMMA_pix = int(self.t_m/self.sim_pixel_m)
        t_wax_pix = int(float(self.dict_params["Thickness wax (mm)"])*1e-3/self.sim_pixel_m)
    
        n = 5
        roots = R_region_in_pix*np.exp(2j * np.pi * np.arange(n) / n)
    
        # Map roots to pixel coordinates relative to the center
        for root in roots:
            x = int(round(self.img_size[1]//2 + root.real))
            y = int(round(self.img_size[0]//2 - root.imag))  # Invert y-axis for image coordinates
        
            # Ensure the coordinates are within the image bounds
            if 0 <= y < self.img_size[0] and 0 <= x < self.img_size[1]:
                f = self.dict_params["RMS scattering angle in X (μrad)"]
                R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
                R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
                if( (f == "" or float(f) == 0.0) or float(f)==1.0):
                    t_map_uCs += create_ellipse_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, a=R_pix, b=R_pix, c=R_pix, cx=x, cy=y)
                else:
                    t_map_uCs += create_ellipse_proj_DF(theta_y = theta_y, theta_z=0, img_size=self.img_size, binning_factor=self.binning_factor, a=R_pix, b=R_pix, c=R_pix, f=float(f), R_microst=R_microst_pix, cx=x, cy=y)
    
        t_map_PMMA = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_PMMA_pix)
        t_map_wax = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_wax_pix)
    
        t_map_1 = t_map_uCs*self.sim_pixel_m
        t_map_3 = (t_map_wax - t_map_uCs)*self.sim_pixel_m
        t_map_2 = (t_map_PMMA - t_map_wax)*self.sim_pixel_m

        return t_map_1, t_map_2, t_map_3
    
    def create_block(self, theta_y=0):
        """
        Generates thickness maps for a block sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """
 
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)
        t_block_in_pix = int(float(self.dict_params["Block thickness (mm)"])*1e-3/self.sim_pixel_m)
    
        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)
        f = self.dict_params["RMS scattering angle in X (μrad)"]
        R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
        R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
        if( (f == "" or float(f) == 0.0) or float(f)==1.0):
            t_map_block = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1]/2, Ly=self.img_size[0]/2, Lz=t_block_in_pix)
        else:
            t_map_block = create_box_proj_DF(theta_y = theta_y, theta_z=0, img_size=self.img_size, binning_factor=self.binning_factor, Lx=(self.img_size[1]/2), Ly=(self.img_size[0]/2), Lz=(t_block_in_pix), f=float(f), R_microst=R_microst_pix)
    
        t_map_1 = t_map_block*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_block)*self.sim_pixel_m
    
        return t_map_1, t_map_2, np.zeros(self.img_size)

    def create_angio_tube(self, theta_y=0):
        """
        Generates thickness maps for an angiographic tube sample with background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3) corresponding to lumen, background, and wall.
        """

        d_ext_in_mm = float(self.dict_params["External diameter (mm)"])
        d_int_in_mm = float(self.dict_params["Internal diameter (mm)"])

        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)
        D_tube_in_pix = int(d_ext_in_mm*1e-3/self.sim_pixel_m)
        D_int_in_pix = int(d_int_in_mm*1e-3/self.sim_pixel_m)
    
        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)
        t_map_cylinder = create_cylinder_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, D=D_tube_in_pix, h=self.img_size[0])
        f = self.dict_params["RMS scattering angle in X (μrad)"]
        R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
        R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
        if( (f == "" or float(f) == 0.0) or float(f)==1.0):
            t_map_cylinder_int = create_cylinder_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, D=D_int_in_pix, h=self.img_size[0])
        else:
            t_map_cylinder_int = create_cylinder_proj_DF(theta_y = theta_y, theta_z=0, img_size=self.img_size, binning_factor=self.binning_factor, D=D_int_in_pix, h=self.img_size[0], f=float(f), R_microst=R_microst_pix)
    
        t_map_1 = t_map_cylinder_int*self.sim_pixel_m
        t_map_3 = (t_map_cylinder - t_map_cylinder_int)*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_cylinder)*self.sim_pixel_m
    
        return t_map_1, t_map_2, t_map_3
    
    def create_fibre(self, theta_y=0):
        """
        Generates thickness maps for a fibre sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """

        d_tube_in_mm = float(self.dict_params["Fibre diameter (mm)"])
        d_tube_in_pix = int(d_tube_in_mm*1e-3/self.sim_pixel_m)
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)

        f = self.dict_params["RMS scattering angle in X (μrad)"]
        R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
        R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
        if( (f == "" or float(f) == 0.0) or float(f)==1.0):
            t_map_tube = create_cylinder_proj(theta_y = theta_y, theta_z=45, img_size=self.img_size, D=d_tube_in_pix, h=0.9*np.sqrt(self.img_size[0]**2 + self.img_size[1]**2))
            print("Normal")
        else:
            t_map_tube = create_cylinder_proj_DF(theta_y = theta_y, theta_z=45, img_size=self.img_size, binning_factor=self.binning_factor, D=d_tube_in_pix, h=0.9*np.sqrt(self.img_size[0]**2 + self.img_size[1]**2), f=float(f), R_microst=R_microst_pix)
            

        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)
        
        t_map_1 = t_map_tube*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_tube)*self.sim_pixel_m

        return t_map_1, t_map_2, np.zeros(self.img_size)
    
    def create_wedge(self, theta_y=0):
        """
        Generates thickness maps for a wedge sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """

        t_block_in_mm = float(self.dict_params["Wedge thickness (mm)"])
        t_block_in_pix = int(t_block_in_mm*1e-3/self.sim_pixel_m)
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)

        f = self.dict_params["RMS scattering angle in X (μrad)"]
        R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
        R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
        if( (f == "" or float(f) == 0.0) or float(f)==1.0):
            t_map_wedge = create_wedge_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx = self.img_size[1]/2, Ly = self.img_size[0]/2, Lz=t_block_in_pix)
        else:
            t_map_wedge = create_wedge_proj_DF(theta_y = theta_y, theta_z=0, img_size=self.img_size, binning_factor=self.binning_factor, Lx = (self.img_size[1]/2), Ly = (self.img_size[0]/2), Lz=(t_block_in_pix), f=float(f), R_microst=R_microst_pix)

        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)

        t_map_1 = t_map_wedge*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_wedge)*self.sim_pixel_m

        return t_map_1, t_map_2, np.zeros(self.img_size)
    
    def create_sphere_bkg(self, theta_y=0):
        """
        Generates thickness maps for a sphere sample on a background.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Three thickness maps (t_map_1, t_map_2, t_map_3), where the third is a zero array.
        """

        d_sph_samp_in_mm = float(self.dict_params["Sphere diameter (mm)"])
        R_in_pix = int(d_sph_samp_in_mm*1e-3/(2*self.sim_pixel_m))               # Radius in pixels
        t_bkg_in_pix = int(self.t_m/self.sim_pixel_m)
        
        f = self.dict_params["RMS scattering angle in X (μrad)"]
        R_microst_um = float(self.dict_params["RMS scattering angle in Y (μrad)"])
        R_microst_pix = R_microst_um*1e-6/(self.sim_pixel_m)
        if( (f == "" or float(f) == 0.0) or float(f)==1.0):
            t_map_sph = create_ellipse_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, a = R_in_pix, b=R_in_pix, c=R_in_pix)
        else:
            t_map_sph = create_ellipse_proj_DF(theta_y = theta_y, theta_z=0, img_size=self.img_size, binning_factor=self.binning_factor, a = R_in_pix, b=R_in_pix, c=R_in_pix, f=float(f), R_microst=R_microst_pix)

        t_map_bkg = create_box_proj(theta_y = theta_y, theta_z=0, img_size=self.img_size, Lx=self.img_size[1], Ly=self.img_size[0], Lz=t_bkg_in_pix)

        t_map_1 = t_map_sph*self.sim_pixel_m
        t_map_2 = (t_map_bkg - t_map_sph)*self.sim_pixel_m

        return t_map_1, t_map_2, np.zeros(self.img_size)
        
    def create_sample(self, theta_y=0):
        """
        Selects and generates the thickness map according to the sample geometry defined in the parameters.

        Args:
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.

        Returns:
            tuple: Thickness maps generated by the corresponding function for the selected geometry.
        """
        if(self.sample_geom == "Mammo Phantom"):
            return self.create_mammo_phantom(theta_y=theta_y)
        elif(self.sample_geom == "block"):
            return self.create_block(theta_y=theta_y)
        elif(self.sample_geom == "angio tube"):
            return self.create_angio_tube(theta_y=theta_y)
        elif(self.sample_geom == "sphere"):
            return self.create_sphere_bkg(theta_y=theta_y)
        elif(self.sample_geom == "fibre"):
            return self.create_fibre(theta_y=theta_y)
        elif(self.sample_geom == "wedge"):
            return self.create_wedge(theta_y=theta_y)
        else:
            raise ValueError("Select a valid sample")
