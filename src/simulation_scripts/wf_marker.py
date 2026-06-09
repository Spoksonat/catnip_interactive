import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.draw import polygon, ellipsoid
import scipy.fft
import scipy.ndimage
import scipy.constants
import xraylib as xl
from scipy.ndimage import shift, convolve
from scipy.signal import fftconvolve
import random

class GratingEI:

    def __init__(self, dict_params, E:float) -> None:

        self.dict_params = dict_params
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.sim_pixel_m = float(dict_params["Sim. pixel (μm)"])*1e-6/self.binning_factor
        self.px_um = float(dict_params["Period (μm)"]) # Period in um
        self.px_pix = int(self.px_um*1e-6/self.sim_pixel_m) # Period in pixels
        self.fringe_width_um = float(dict_params["Fringe width (μm)"])
        self.shift_x_um = float(dict_params["Shift grating lateral axis (μm)"])
        self.t_m = float(dict_params["Grating thickness (μm)"])*1e-6 # Thickness
        self.mat = dict_params["Material"] # Material name
        self.mat_density = float(dict_params["Material density (g/cc)"]) # Material density in g/cm3
        self.bin_grat = None # Grating binary pattern
        self.N = int(float(dict_params["Number of steps"]))
        self.E = E

        a_str, b_str = dict_params["FOV (pix)"].strip("()").split(",")
        self.img_size = (int(float(a_str))*self.binning_factor, int(float(b_str))*self.binning_factor)

        n = xl.Refractive_Index(self.mat, E, self.mat_density)
        self.delta = 1 - n.real
        beta = n.imag
        self.mu_mass = xl.CS_Total_CP(self.mat, E)
        
        #self.mu = 2*k*beta # Attenuation coefficient in 1/m
        self.mu = self.mu_mass*self.mat_density*1e2
    
    def create_fringe(self) -> np.ndarray:
        """
        Creates a binary mask representing a single fringe of the grating.

        Returns:
            np.ndarray: Binary mask of the fringe.
        """
        self.fringe_width_pix = int(self.fringe_width_um * 1e-6/self.sim_pixel_m)
        mask = np.ones((self.img_size[0], self.fringe_width_pix))
        return mask
    
    def create_bin_grat(self) -> np.ndarray:
        """
        Generates the binary grating pattern using the fringe mask and stores it in self.bin_grat.

        Returns:
            np.ndarray: Binary grating pattern.
        """
        grating = np.zeros(self.img_size)
        fringe = self.create_fringe()
        grating[int(self.img_size[0]/2), int(self.shift_x_um*1e-6/self.sim_pixel_m)::self.px_pix] = 1
        grating = scipy.fft.ifftn(scipy.fft.fftn(grating, workers = -1)*scipy.fft.fftn(fringe, grating.shape, workers=-1),workers = -1)
        grating = np.abs(grating)
        grating = (grating > 1e-2).astype(int)  # Thresholding to create binary pattern
        self.bin_grat = grating

    def create_bin_grat_2_mask(self):
        """
        Generates a binary grating pattern with two shifted masks and stores it in self.bin_grat.
        """
        grating_1 = np.zeros(self.img_size)
        grating_2 = np.zeros(self.img_size)

        fringe_width_pix = int(self.px_pix/2)
        fringe = np.ones((self.img_size[0], fringe_width_pix))

        effective_fringe_width_pix = int(self.fringe_width_um * 1e-6/self.sim_pixel_m)

        grating_1[int(self.img_size[0]/2), int(self.shift_x_um*1e-6/self.sim_pixel_m)::self.px_pix] = 1
        grating_1 = scipy.fft.ifftn(scipy.fft.fftn(grating_1, workers = -1)*scipy.fft.fftn(fringe, grating_1.shape, workers=-1),workers = -1)
        grating_1 = np.abs(grating_1)
        grating_1 = (grating_1 > 1e-2).astype(int)  # Thresholding to create binary pattern

        grating_2[int(self.img_size[0]/2), int(self.shift_x_um*1e-6/self.sim_pixel_m) + (effective_fringe_width_pix - fringe_width_pix)::self.px_pix] = 1
        grating_2 = scipy.fft.ifftn(scipy.fft.fftn(grating_2, workers = -1)*scipy.fft.fftn(fringe, grating_2.shape, workers=-1),workers = -1)
        grating_2 = np.abs(grating_2)
        grating_2 = (grating_2 > 1e-2).astype(int)  # Thresholding to create binary pattern

        self.bin_grat = grating_1  + grating_2 
    
    def grat_pos_lin(self) -> np.ndarray:
        """
        Calculates linear positions for grating steps.

        Returns:
            np.ndarray: Array of positions for each step.
        """
        step_X = int(self.px_pix/self.N)#int(self.px_pix - self.fringe_width_pix)#int(self.px_pix/self.N)
        x = step_X * np.arange(self.N)
        positions = []

        for i in range(self.N):
            point = [x[i], 0]
            positions.append(point)
    
        positions = np.array(positions)
    
        return positions
        
    def obtain_grat_array(self) -> np.ndarray:
        """
        Generates an array of shifted grating patterns according to calculated positions.

        Returns:
            np.ndarray: Array of shifted grating patterns.
        """
        self.create_bin_grat()

        grat_array = []

        poss = self.grat_pos_lin()
        
        for pos in poss:
            pos_x, pos_y = pos[0], pos[1]
            grat_shifted = scipy.ndimage.shift(self.bin_grat, (-pos_y, pos_x), order=3, mode='grid-wrap')
            #np.roll(self.bin_grat, shift=(-pos_y, pos_x), axis=(0, 1))
            grat_array.append(grat_shifted)

        grat_array = np.array(grat_array)

        return grat_array
    
class GratingGBI:

    def __init__(self, dict_params, E: float) -> None:

        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.sim_pixel_m = float(dict_params["Sim. pixel (μm)"])*1e-6/self.binning_factor
        self.px_um = float(dict_params["Period in X (μm)"])
        self.py_um = float(dict_params["Period in Y (μm)"])
        self.px_pix = int(self.px_um*1e-6/self.sim_pixel_m) # Period in pixels
        self.py_pix = int(self.py_um*1e-6/self.sim_pixel_m) # Period in pixels
        self.t_m = float(dict_params["Thickness (μm)"])*1e-6 # Thickness
        self.mat = dict_params["Material"] # Material name
        self.mat_density = float(dict_params["Material density (g/cc)"]) # Material density in g/cm3
        self.bin_grat = None # Grating binary pattern
        self.E = E
        self.k =  2. * np.pi * (E * 1e3 * 1.6022e-19) / (scipy.constants.h * scipy.constants.c)
        self.N = int(float(dict_params["Num. of steps per dir."]))
    
        a_str, b_str = dict_params["FOV (pix)"].strip("()").split(",")
        self.img_size = (int(float(a_str)), int(float(b_str)))

        ratio_y, ratio_x = round(self.img_size[0]/self.py_pix), round(self.img_size[1]/self.px_pix)
        self.img_size = (ratio_y*self.py_pix*self.binning_factor, ratio_x*self.px_pix*self.binning_factor)


        n = xl.Refractive_Index(self.mat, E, self.mat_density)
        self.delta = 1 - n.real
        beta = n.imag
        self.mu_mass = xl.CS_Total_CP(self.mat, E)
        
        #self.mu = 2*k*beta # Attenuation coefficient in 1/m
        self.mu = self.mu_mass*self.mat_density*1e2

        if(dict_params["Phase shift"] == "Auto"):
            self.ph_shift = self.delta*self.k*self.t_m
        else:
            self.ph_shift = float(dict_params["Phase shift"])*np.pi # Phase shift

        self.radius_um = float(dict_params["Hole radius (μm)"])

    def create_hole(self) -> np.ndarray:
        """
        Creates a binary mask for a hole with a specified number of vertices.

        Returns:
            np.ndarray: Binary mask of the hole.
        """
        self.radius_pix = int(self.radius_um * 1e-6 / self.sim_pixel_m)
        cx, cy = self.radius_pix, self.radius_pix
        mask = np.zeros((2*self.radius_pix+1, 2*self.radius_pix+1))
        theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False) # Circle as a 1000-point polygon
        x = self.radius_pix * np.cos(theta)
        y = self.radius_pix * np.sin(theta)

        vertices = np.array([cx + x, cy + y]).T
        rr, cc = polygon(vertices[:, 1], vertices[:, 0], self.img_size)
        mask[rr, cc] = 1
        return mask
    
    
    def create_bin_grat(self) -> np.ndarray:
        """
        Generates the binary grating pattern using the hole mask and stores it in self.bin_grat.

        Returns:
            np.ndarray: Binary grating pattern.
        """
        grating = np.zeros(self.img_size)

        hole = self.create_hole()
        
        grating[::self.py_pix, ::self.px_pix] = 1
        grating[int(self.py_pix/2)::self.py_pix, int(self.px_pix/2)::self.px_pix] = 1
        #grating = convolve(grating, hole, mode='wrap').astype(int)
        grating = scipy.fft.ifftn(scipy.fft.fftn(grating, workers = -1)*scipy.fft.fftn(hole, grating.shape, workers=-1),workers = -1)
        grating = np.abs(grating)
        grating = (grating > 1e-2).astype(int)  # Thresholding to create binary pattern
        self.bin_grat = 1 - grating
    
    def grat_pos_serpent_shear(self) -> np.ndarray:
        """
        Calculates serpent-like positions with shear for grating steps.

        Returns:
            np.ndarray: Array of positions for each step in serpent-like shear order.
        """
        step_X, step_Y = int(self.px_pix/self.N), int((self.py_pix/2)/self.N)
        x = step_X * np.arange(self.N)
        y = step_Y * np.arange(self.N)
    
        XX, YY = np.meshgrid(x,y)
        XX[1::2] = XX[1::2, ::-1]
        rows, cols = XX.shape[0], XX.shape[1]
        positions = []

        #m = 1/np.tan(np.radians(60))
        m = 1/(self.py_pix/self.px_pix)
    
        for i in range(rows):
            for j in range(cols):
                XX_new = XX[i, j] + m*YY[i, j]
                point = [XX_new, YY[i, j]]
                positions.append(point)
    
        positions = np.array(positions)
    
        return positions
        
    def obtain_grat_array(self) -> np.ndarray:
        """
        Generates an array of shifted grating patterns according to serpent-like shear positions.

        Returns:
            np.ndarray: Array of shifted grating patterns.
        """
        self.create_bin_grat()

        grat_array = []
        poss = self.grat_pos_serpent_shear()
        
        for pos in poss:
            pos_x, pos_y = pos[0], pos[1]
            grat_shifted = scipy.ndimage.shift(self.bin_grat, (-pos_y, pos_x), order=3, mode='grid-wrap')
            #np.roll(self.bin_grat, shift=(-pos_y, pos_x), axis=(0, 1))
            grat_array.append(grat_shifted)

        grat_array = np.array(grat_array)

        return grat_array
    
class Sandpaper:
    def __init__(self, dict_params, E: float) -> None:
        
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.sim_pixel_m = float(dict_params["Sim. pixel (μm)"])*1e-6/self.binning_factor
        self.r_um = float(dict_params["Grain size (μm)"])/2.0
        self.r_m = self.r_um*1e-6
        self.r_pix = int(self.r_m/self.sim_pixel_m)
        self.t_m = float(dict_params["Paper thickness (μm)"])*1e-6
        self.mat_samp = dict_params["Grain material"]
        self.samp_density = float(dict_params["Grain material density (g/cc)"])
        self.mat_bkg = dict_params["Paper material"]
        self.bkg_density = float(dict_params["Paper material density (g/cc)"])
        #self.separation_factor = 1.0
        self.n_sand = int(float(dict_params["Number of sandpapers"]))
        self.N = int(float(dict_params["Number of steps"]))
        self.step_size_um = float(dict_params["Step size (μm)"])

        a_str, b_str = dict_params["FOV (pix)"].strip("()").split(",")
        self.img_size = (int(float(a_str))*self.binning_factor, int(float(b_str))*self.binning_factor)

        self.n_specks = int(float(dict_params["Number of grains"])) #int(self.separation_factor*np.prod(self.img_size)*(self.sim_pixel_m**2)/(np.pi*(self.r_m**2)))
        self.E = E
        

        n = xl.Refractive_Index(self.mat_samp, E, self.samp_density)
        self.delta_samp = 1 - n.real
        beta = n.imag
        self.mu_mass_samp = xl.CS_Total_CP(self.mat_samp, E)

        n = xl.Refractive_Index(self.mat_bkg, E, self.bkg_density)
        self.delta_bkg = 1 - n.real
        beta = n.imag
        self.mu_mass_bkg = xl.CS_Total_CP(self.mat_bkg, E)
        
        #self.mu = 2*k*beta # Attenuation coefficient in 1/m
        self.mu_samp = self.mu_mass_samp*self.samp_density*1e2
        self.mu_bkg = self.mu_mass_bkg*self.bkg_density*1e2

    def create_sph(self, R):
        """
        Creates a spherical mask and projects it along one axis.

        Args:
            R (int): Radius of the sphere in pixels.

        Returns:
            np.ndarray: 2D projection of the sphere.
        """
        sph = ellipsoid(R,R,R).astype(int)
        proj_sph = np.sum(sph, axis=-1)
        return proj_sph
    
    def random_no_overlap(self, scale_factor):
        """
        Generates random positions for grains without overlap.

        Args:
            scale_factor (float): Scaling factor for random seed.

        Returns:
            tuple: Arrays of x and y coordinates for grain centers.
        """
        np.random.seed(int(100/scale_factor))

        x_range = np.arange(2*self.r_pix + 1, self.img_size[1] - (2*self.r_pix + 1))  # x-values: 0 to 199
        y_range = np.arange(2*self.r_pix + 1, self.img_size[0] - (2*self.r_pix + 1))  # y-values: 0 to 199
    
        # Use meshgrid to create grid of points
        x_grid, y_grid = np.meshgrid(x_range, y_range)
    
        # Combine the x and y grids into an array of points
        points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    
        # Define the distance threshold
        d = 2*self.r_pix + 1
    
        # Generate unique pairs dynamically
        unique_pairs = set()
        flag = True
        while len(unique_pairs) < self.n_specks:
            pair_ind = tuple(random.sample(range(len(points)),1))[0]
            pair = (points[pair_ind][0], points[pair_ind][1])   # Retrieve the points by index
            if(flag):
                unique_pairs.add(pair)
                flag = False
            else: 
                possX, possY = np.array(list(unique_pairs))[:,0], np.array(list(unique_pairs))[:,1]
                if np.all(np.sqrt( (pair[0] - possX )**2 + (pair[1] - possY)**2) > d):
                    unique_pairs.add(pair)
    
        unique_pairs = list(unique_pairs)
        unique_pairs = np.array(unique_pairs)
    
        cX, cY = unique_pairs[:,0], unique_pairs[:,1]
    
        return cX, cY
    
    def create_grat(self, ind_sandpaper) -> np.ndarray:
        """
        Generates a sandpaper pattern for a given index.

        Args:
            ind_sandpaper (int): Index of the sandpaper.

        Returns:
            np.ndarray: Pattern for the sandpaper.
        """
        sandpaper = np.zeros(self.img_size, dtype=np.complex128)

        np.random.seed(ind_sandpaper)
        ##posX, posY = self.random_no_overlap(scale_factor)
        posX = np.random.randint(0, self.img_size[1], self.n_specks)
        posY = np.random.randint(0, self.img_size[0], self.n_specks)
        sph = self.create_sph(self.r_pix)
        sandpaper[posY, posX] = 1
        #sandpaper = convolve(sandpaper, sph, mode='wrap')
        sandpaper = scipy.fft.ifftn(scipy.fft.fftn(sandpaper, workers = -1)*scipy.fft.fftn(sph, sandpaper.shape, workers=-1),workers = -1)
        sandpaper = np.abs(sandpaper)
        #sandpaper = ndi.gaussian_filter(np.random.normal(size=self.img_size), 2*self.r_pix)
        
        return sandpaper*self.sim_pixel_m

    def create_multiple_grats(self):

        grating = np.zeros(self.img_size)

        for ind_sandpaper in range(self.n_sand):
            grating += self.create_grat(ind_sandpaper+1)

        self.grat = grating
    
    def grat_pos_lab(self) -> np.ndarray:
        """
        Calculates positions for sandpaper steps in a stair-wise manner.

        Returns:
            np.ndarray: Array of positions for each step.
        """
        step_X, step_Y = (self.step_size_um*1e-6/self.sim_pixel_m), (self.step_size_um*1e-6/self.sim_pixel_m)
        positions = [np.array([0,0])]

        for i in range(1,self.N):
            if(i % 2 != 0): 
                point = positions[-1] + np.array([step_X, 0])
            else:
                point = positions[-1] + np.array([0, step_Y])
        
            positions.append(point)
    
        positions = np.array(positions)
    
        return positions
    
    def grat_pos_random(self) -> np.ndarray:
        """
        Calculates positions for sandpaper steps in a random-walk.

        Returns:
            np.ndarray: Array of positions for each step.
        """
        step = (self.step_size_um*1e-6/self.sim_pixel_m)
        positions = [np.array([0,0])]

        for i in range(1,self.N):
            angle = random.uniform(-np.pi, np.pi)
            point = positions[-1] + np.array([step*np.cos(angle), step*np.sin(angle)])

            positions.append(point)

        positions = np.array(positions)
    
        return positions
    
    def obtain_grat_array(self) -> np.ndarray:

        self.create_multiple_grats()

        grat_array = []

        poss = self.grat_pos_random()
        
        for pos in poss:
            pos_x, pos_y = pos[0], pos[1]
            grat_shifted = scipy.ndimage.shift(self.grat, (-pos_y, pos_x), order=3, mode='grid-wrap')
            #np.roll(self.grat, shift=(-pos_y, pos_x), axis=(0, 1))
            grat_array.append(grat_shifted)

        grat_array = np.array(grat_array)

        return grat_array