import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import scipy.constants
from tqdm import tqdm
import scipy.constants
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage.draw import disk

class SimulationSBI:

    def __init__(self, dict_params, grat, samp, E, theta_y=0) -> None:
        """
        Initializes the SimulationSBI class for Sandpaper-based simulation.

        Args:
            dict_params (dict): Dictionary of simulation parameters.
            grat: Grating object.
            samp: Sample object.
            E (float): Energy value in keV.
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.
        """

        self.grat = grat
        self.samp = samp
        self.dict_params = dict_params
        self.type_of_source = dict_params["Source geometry"]
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.d_source_det = float(dict_params["Source-Detector distance (m)"])
        self.d_source_grat = float(dict_params["Source-Grating distance (m)"])
        self.d_source_samp = self.d_source_grat + float(dict_params["Grating-Sample distance (m)"])
        self.d_grat_det = self.d_source_det - self.d_source_grat
        self.d_samp_det = self.d_source_det - self.d_source_samp
        self.d_grat_samp = np.abs(self.d_source_samp - self.d_source_grat)
        self.d_prop = self.d_samp_det - (float(dict_params["Whole sample thickness (mm)"])*1e-3)/2
        self.wf_init = np.ones(self.grat.img_size)
        self.k =  2. * np.pi * (E * 1e3 * 1.6022e-19) / (scipy.constants.h * scipy.constants.c)   # 1/m
        self.fwhm = float(dict_params["FWHM PSF (pix)"])
        self.psf_size = (2*self.fwhm+1, 2*self.fwhm+1)
        self.f_um = float(dict_params["Focal spot size (μm)"])
        self.n_ph = float(dict_params["Num. events per pixel"])
        self.theta_y = theta_y
    
    def inter_sandpaper_det(self, wf: np.ndarray, grating_t: np.ndarray) -> np.ndarray:
        """
        Applies the sandpaper interaction to the wavefront.

        Args:
            wf (np.ndarray): Input wavefront.
            grating_t (np.ndarray): Sandpaper thickness map.

        Returns:
            np.ndarray: Modified wavefront after sandpaper interaction.
        """
        return wf*np.exp(-1j*self.k*(self.grat.delta_samp*grating_t + self.grat.delta_bkg*self.grat.t_m*np.ones(self.grat.img_size)))*np.exp(-(self.grat.mu_samp*grating_t + self.grat.mu_bkg*self.grat.t_m*np.ones(self.grat.img_size))/2)
    
    def inter_samp_det(self, wf: np.ndarray, t_map_1: np.ndarray, t_map_2: np.ndarray, t_map_3: np.ndarray) -> np.ndarray:
        """
        Applies the sample interaction to the wavefront.

        Args:
            wf (np.ndarray): Input wavefront.
            t_map_1, t_map_2, t_map_3 (np.ndarray): Thickness maps for sample regions.

        Returns:
            np.ndarray: Modified wavefront after sample interaction.
        """
        return wf*np.exp(-1j* self.k * (self.samp.delta_1*t_map_1 + self.samp.delta_2*t_map_2 + self.samp.delta_3*t_map_3)) * np.exp(-(self.samp.mu_1*t_map_1 + self.samp.mu_2*t_map_2 + self.samp.mu_3*t_map_3)/2)

    def fres_ker_fourier(self, z):
        """
        Computes the Fresnel kernel in Fourier space for propagation.

        Args:
            z (float): Propagation distance.

        Returns:
            np.ndarray: Fresnel kernel in Fourier space.
        """
        u = 2 * np.pi * scipy.fft.fftfreq(self.grat.img_size[1]) / self.grat.sim_pixel_m
        v = 2 * np.pi * scipy.fft.fftfreq(self.grat.img_size[0]) / self.grat.sim_pixel_m
        UU, VV = np.meshgrid(u, v)
        #fres_ker_fou = np.exp(1j*self.k*z)* np.exp(-(1/2)*1j * (z/ self.k) * (UU ** 2 + VV**2))
        fres_ker_fou = np.exp(1j * z * np.sqrt(self.k**2 - UU ** 2 - VV**2))
        return fres_ker_fou

    def propagation(self, wf: np.ndarray, z: float) -> np.ndarray:
        """
        Propagates the wavefront by a distance z using the Fresnel kernel.

        Args:
            wf (np.ndarray): Input wavefront.
            z (float): Propagation distance.

        Returns:
            np.ndarray: Propagated wavefront.
        """
        
        fres_ker_fou = self.fres_ker_fourier(z=z)
        wf_prop = scipy.fft.ifftn(scipy.fft.fftn(wf, fres_ker_fou.shape, workers = -1)*fres_ker_fou, workers = -1)

        return wf_prop
    
    def binning(self, image: np.ndarray) -> np.ndarray:
        """
        Applies binning to the image according to the binning factor.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Binned image.
        """
        new_shape = (image.shape[0] // self.binning_factor, image.shape[1] // self.binning_factor)
        binned_image = image.reshape(new_shape[0], self.binning_factor, new_shape[1], self.binning_factor).mean(axis=(1, 3))

        return binned_image
    
    def convolve_PSF(self, image: np.ndarray) -> np.ndarray:
        """
        Convolves the image with a point spread function (PSF).

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image after PSF convolution.
        """
        x, y = np.arange(self.psf_size[1]), np.arange(self.psf_size[0])
        XX, YY = np.meshgrid(x,y)
        psf = np.exp(-4*np.log(2)*(XX**2 + YY**2)/self.fwhm**2)
        image_conv = scipy.fft.ifftn(scipy.fft.fftn(image, workers=-1)*scipy.fft.fftn(psf,image.shape, workers=-1), workers=-1)
        return np.abs(image_conv)
    
    def convolve_PSF_total(self, image: np.ndarray) -> np.ndarray:
        """
        Applies system PSF convolution, considering source geometry.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image after system PSF convolution.
        """
        if(self.type_of_source == "Cone"):
            M_samp = (self.d_source_samp + self.d_samp_det)/self.d_source_samp
            f_pix = (self.f_um*1e-6/self.grat.sim_pixel_m)
            self.fwhm_sys = int(np.sqrt((self.fwhm/M_samp)**2 + (f_pix**2)*((M_samp-1)/M_samp)**2))
        else:
            self.fwhm_sys = int(self.fwhm)
        convolved_image = gaussian_filter(image, sigma=self.fwhm_sys/2.355)
        return convolved_image
    
    def add_dark_field(self, image, t_map_1):
        """
        Adds dark-field blur to the image based on sample thickness.

        Args:
            image (np.ndarray): Input image.
            t_map_1 (np.ndarray): Thickness map for region 1.
            patch_size (int, optional): Size of the patch for local blurring.

        Returns:
            np.ndarray: Image with dark-field effect.
        """
        ## sigma = z_prop*theta/pixel_size, theta = scattering angle
        blurred = np.zeros_like(image)
        h, w = image.shape
        t_map_1 = self.binning(t_map_1)
        theta_max_x = float(self.dict_params["RMS scattering angle in X (μrad)"])*1e-6
        theta_max_y = float(self.dict_params["RMS scattering angle in Y (μrad)"])*1e-6
        theta_map_x = theta_max_x*(t_map_1 - t_map_1.min())/(t_map_1.max() - t_map_1.min())
        theta_map_y = theta_max_y*(t_map_1 - t_map_1.min())/(t_map_1.max() - t_map_1.min())
        N_sw = 11
        patch_size = 2*N_sw + 1 
        for i in range(N_sw, h-N_sw):
            for j in range(N_sw, w-N_sw):
                patch = image[i-N_sw:i+N_sw+1, j-N_sw:j+N_sw+1]
                sigma_x = (self.d_prop * theta_map_x[i,j]) / (self.grat.sim_pixel_m)
                sigma_y = (self.d_prop * theta_map_y[i,j]) / (self.grat.sim_pixel_m)
                #blurred_patch = gaussian_filter(patch, sigma=sigma)
                x, y = np.arange(patch_size), np.arange(patch_size)
                XX, YY = np.meshgrid(x,y)
                if((sigma_x == 0) or (sigma_y==0)):
                    kernel = np.zeros(patch.shape)
                    kernel[N_sw, N_sw] = 1
                else: 
                    kernel = np.exp(-((XX - N_sw)**2)/(2*sigma_x**2))*np.exp(-((YY-N_sw)**2)/(2*sigma_y**2))
                    kernel = kernel/np.sum(kernel)
                blurred[i,j] = np.sum(patch * kernel)
        blurred[:N_sw,:] = image[:N_sw,:]
        blurred[-N_sw:,:] = image[-N_sw:,:]
        blurred[:,:N_sw] = image[:,:N_sw]
        blurred[:,-N_sw:] = image[:,-N_sw:]
        return blurred
    
    def create_ref_samp(self, bin_grat: np.ndarray, t_map_1: np.ndarray, t_map_2: np.ndarray, t_map_3: np.ndarray) -> np.ndarray:
        """
        Simulates reference and sample images for a single grating position.

        Args:
            bin_grat (np.ndarray): Binary grating pattern.
            t_map_1, t_map_2, t_map_3 (np.ndarray): Thickness maps for sample regions.

        Returns:
            tuple: Reference and sample images.
        """
        
        if(self.type_of_source == "Cone"):
            M_1 = (self.d_source_grat + self.d_grat_det)/self.d_source_grat
            M_2 = (self.d_source_grat + self.d_grat_samp)/self.d_source_grat
            M_3 = (self.d_source_samp + self.d_samp_det)/self.d_source_samp
        else:
            M_1, M_2, M_3 = 1, 1, 1 


        wf_bg = self.propagation(wf=self.wf_init, z=self.d_source_grat)
        wf_grat = self.inter_sandpaper_det(wf=wf_bg, grating_t=bin_grat)
        #wf_grat = ndi.gaussian_filter(np.random.normal(size=self.grat.img_size), 2*self.grat.r_pix) + 1j*ndi.gaussian_filter(np.random.normal(size=self.grat.img_size), 2*self.grat.r_pix)
        
        wf_ref = wf_grat
        wf_ref = self.propagation(wf=wf_ref, z=self.d_grat_det/M_1)#self.d_grat_samp+self.samp.t_m
        wf_ref = self.scale(np.real(wf_ref), M=M_1) +1j*self.scale(np.imag(wf_ref), M=M_1)
        I_ref = np.abs(wf_ref)**2
        I_ref = self.binning(I_ref)
        I_ref = self.convolve_PSF_total(image=I_ref)
        I_ref = np.random.poisson(lam=self.n_ph*I_ref)
        

        wf_samp = wf_grat
        wf_samp = self.propagation(wf=wf_samp, z=(self.d_grat_samp - self.samp.t_m/2)/M_2)
        wf_samp = self.scale(np.real(wf_samp), M=M_2) +1j*self.scale(np.imag(wf_samp), M=M_2)
        wf_samp = self.inter_samp_det(wf=wf_samp, t_map_1=t_map_1, t_map_2=t_map_2, t_map_3=t_map_3)
        wf_samp = self.propagation(wf=wf_samp, z=(self.d_prop + self.samp.t_m/2)*(M_2/M_3))
        wf_samp = self.scale(np.real(wf_samp), M=M_3) +1j*self.scale(np.imag(wf_samp), M=M_3)
        I_samp = np.abs(wf_samp)**2
        I_samp = self.binning(I_samp)
        I_samp = self.convolve_PSF_total(image=I_samp)
        #I_samp = self.add_dark_field(image=I_samp, t_map_1=t_map_1)
        I_samp = np.random.poisson(lam=self.n_ph*I_samp)
        
        return I_ref, I_samp
    
    def create_ref_samp_stacks(self) -> np.ndarray:
        """
        Simulates stacks of reference and sample images for all grating positions.

        Returns:
            tuple: Arrays of reference and sample images for all positions.
        """

        grat_array = self.grat.obtain_grat_array()
        t_map_1, t_map_2, t_map_3 = self.samp.create_sample(self.theta_y)
        I_refs = []
        I_samps = []

        for bin_grat in tqdm(grat_array):
            I_ref, I_samp = self.create_ref_samp(bin_grat=bin_grat, t_map_1=t_map_1, t_map_2=t_map_2, t_map_3=t_map_3)
            I_refs.append(I_ref)
            I_samps.append(I_samp)

        I_refs = np.array(I_refs)
        I_samps = np.array(I_samps)

        return I_refs, I_samps
    
    def scale(self, img, M):
        """
        Scales the image according to the magnification factor.

        Args:
            img (np.ndarray): Input image.
            M (float): Magnification factor.

        Returns:
            np.ndarray: Scaled image.
        """
        if(self.type_of_source == "Cone"):
            #h, w = img.shape[:2]
            #zoom_factor = M
            ## For multichannel images we don't want to apply the zoom factor to the RGB
            ## dimension, so instead we create a tuple of zoom factors, one per array
            ## dimension, with 1's for any trailing dimensions after the width and height.
            #zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
            ## Bounding box of the zoomed-in region within the input array
            #zh = int(np.round(h / zoom_factor))
            #zw = int(np.round(w / zoom_factor))
            #top = (h - zh) // 2
            #left = (w - zw) // 2
    #
            #out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, order=0)
    #
            ## `out` might still be slightly larger than `img` due to rounding, so
            ## trim off any extra pixels at the edges
            #trim_top = ((out.shape[0] - h) // 2)
            #trim_left = ((out.shape[1] - w) // 2)
            #out = out[trim_top:trim_top+h, trim_left:trim_left+w]

            zoom_factor = M
            M, N = img.shape
            M_new = int(np.round(M * zoom_factor))
            N_new = int(np.round(N * zoom_factor))
        
            # Forward FFT and center
            f = np.fft.fftshift(np.fft.fft2(img))
        
            # Create zero-padded array
            F_zoomed = np.zeros((M_new, N_new), dtype=complex)
        
            # Determine cropping/padding indices
            m0, n0 = M // 2, N // 2
            m1, n1 = M_new // 2, N_new // 2
        
            # Copy central spectrum (or pad if zoom out)
            r_min = min(m0, m1)
            c_min = min(n0, n1)
            F_zoomed[m1 - r_min:m1 + r_min, n1 - c_min:n1 + c_min] = \
                f[m0 - r_min:m0 + r_min, n0 - c_min:n0 + c_min]
        
            # Inverse FFT
            zoomed = np.abs(np.fft.ifft2(np.fft.ifftshift(F_zoomed)))
        
            # Normalize
            zoomed *= zoom_factor**2

            # Center-crop to original size
            crop_m = (M_new - M) // 2
            crop_n = (N_new - N) // 2
            result = zoomed[crop_m:crop_m + M, crop_n:crop_n + N]
            out = result
        
        else:
            out = img

        return out
    
class SimulationGBI:

    def __init__(self, dict_params, grat, samp, E, theta_y=0) -> None:
        """
        Initializes the SimulationSGBI class for SGBI simulation.

        Args:
            dict_params (dict): Dictionary of simulation parameters.
            grat: Grating object.
            samp: Sample object.
            E (float): Energy value in keV.
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.
        """

        self.grat = grat
        self.samp = samp
        self.dict_params = dict_params
        self.type_of_source = dict_params["Source geometry"]
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.d_source_det = float(dict_params["Source-Detector distance (m)"])
        self.d_source_grat = float(dict_params["Source-Grating distance (m)"])
        self.d_source_samp = self.d_source_grat + float(dict_params["Grating-Sample distance (m)"])
        self.d_grat_det = self.d_source_det - self.d_source_grat
        self.d_samp_det = self.d_source_det - self.d_source_samp
        self.d_grat_samp = np.abs(self.d_source_samp - self.d_source_grat)
        self.d_prop = self.d_samp_det - (float(dict_params["Whole sample thickness (mm)"])*1e-3)/2
        self.wf_init = np.ones(self.grat.img_size)
        self.k =  2. * np.pi * (E * 1e3 * 1.6022e-19) / (scipy.constants.h * scipy.constants.c)   # 1/m
        self.fwhm = float(dict_params["FWHM PSF (pix)"])
        self.psf_size = (2*self.fwhm+1, 2*self.fwhm+1)
        self.f_um = float(dict_params["Focal spot size (μm)"])
        self.n_ph = float(dict_params["Num. events per pixel"])
        self.theta_y = theta_y
    
    def inter_grat_det(self, wf: np.ndarray, bin_grat: np.ndarray) -> np.ndarray:
        """
        Applies the grating interaction to the wavefront.

        Args:
            wf (np.ndarray): Input wavefront.
            bin_grat (np.ndarray): Binary grating pattern.

        Returns:
            np.ndarray: Modified wavefront after grating interaction.
        """
        if(self.dict_params["Phase shift"] != "Auto"):
            return wf*np.exp(-1j* self.grat.ph_shift*bin_grat)*np.exp(-(self.grat.mu/2)*self.grat.t_m*bin_grat)
        else:
            return wf*np.exp(-1j* self.k*self.grat.delta*self.grat.t_m*bin_grat)*np.exp(-(self.grat.mu/2)*self.grat.t_m*bin_grat)

    
    def inter_samp_det(self, wf: np.ndarray, t_map_1: np.ndarray, t_map_2: np.ndarray, t_map_3: np.ndarray) -> np.ndarray:
        """
        Applies the sample interaction to the wavefront.

        Args:
            wf (np.ndarray): Input wavefront.
            t_map_1, t_map_2, t_map_3 (np.ndarray): Thickness maps for sample regions.

        Returns:
            np.ndarray: Modified wavefront after sample interaction.
        """
        return wf*np.exp(-1j* self.k * (self.samp.delta_1*t_map_1 + self.samp.delta_2*t_map_2 + self.samp.delta_3*t_map_3)) * np.exp(-(self.samp.mu_1*t_map_1 + self.samp.mu_2*t_map_2 + self.samp.mu_3*t_map_3)/2)

    def fres_ker_fourier(self, z):
        """
        Computes the Fresnel kernel in Fourier space for propagation.

        Args:
            z (float): Propagation distance.

        Returns:
            np.ndarray: Fresnel kernel in Fourier space.
        """
        u = 2 * np.pi * scipy.fft.fftfreq(self.grat.img_size[1]) / self.grat.sim_pixel_m
        v = 2 * np.pi * scipy.fft.fftfreq(self.grat.img_size[0]) / self.grat.sim_pixel_m
        UU, VV = np.meshgrid(u, v)
        #fres_ker_fou = np.exp(1j*self.k*z)* np.exp(-(1/2)*1j * (z/ self.k) * (UU ** 2 + VV**2))
        fres_ker_fou = np.exp(1j * z * np.sqrt(self.k**2 - UU ** 2 - VV**2))
        return fres_ker_fou

    def propagation(self, wf: np.ndarray, z: float) -> np.ndarray:
        """
        Propagates the wavefront by a distance z using the Fresnel kernel.

        Args:
            wf (np.ndarray): Input wavefront.
            z (float): Propagation distance.

        Returns:
            np.ndarray: Propagated wavefront.
        """
        
        fres_ker_fou = self.fres_ker_fourier(z=z)
        wf_prop = scipy.fft.ifftn(scipy.fft.fftn(wf, fres_ker_fou.shape, workers = -1)*fres_ker_fou, workers = -1)

        return wf_prop
    
    def binning(self, image: np.ndarray) -> np.ndarray:
        """
        Applies binning to the image according to the binning factor.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Binned image.
        """
        new_shape = (image.shape[0] // self.binning_factor, image.shape[1] // self.binning_factor)
        binned_image = image.reshape(new_shape[0], self.binning_factor, new_shape[1], self.binning_factor).mean(axis=(1, 3))

        return binned_image
    
    def convolve_PSF(self, image: np.ndarray) -> np.ndarray:
        """
        Convolves the image with a point spread function (PSF).

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image after PSF convolution.
        """
        x, y = np.arange(self.psf_size[1]), np.arange(self.psf_size[0])
        XX, YY = np.meshgrid(x,y)
        psf = np.exp(-4*np.log(2)*(XX**2 + YY**2)/self.fwhm**2)
        image_conv = scipy.fft.ifftn(scipy.fft.fftn(image, workers=-1)*scipy.fft.fftn(psf,image.shape, workers=-1), workers=-1)
        return np.abs(image_conv)
    
    def convolve_PSF_total(self, image: np.ndarray) -> np.ndarray:
        """
        Applies system PSF convolution, considering source geometry.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image after system PSF convolution.
        """
        if(self.type_of_source == "Cone"):
            M_samp = (self.d_source_samp + self.d_samp_det)/self.d_source_samp
            f_pix = (self.f_um*1e-6/self.grat.sim_pixel_m)
            self.fwhm_sys = int(np.sqrt((self.fwhm/M_samp)**2 + (f_pix**2)*((M_samp-1)/M_samp)**2))
        else:
            self.fwhm_sys = int(self.fwhm)
        convolved_image = gaussian_filter(image, sigma=self.fwhm_sys/2.355)
        return convolved_image
    
    def add_dark_field(self, image, t_map_1):
        """
        Adds dark-field blur to the image based on sample thickness.

        Args:
            image (np.ndarray): Input image.
            t_map_1 (np.ndarray): Thickness map for region 1.
            patch_size (int, optional): Size of the patch for local blurring.

        Returns:
            np.ndarray: Image with dark-field effect.
        """
        ## sigma = z_prop*theta/pixel_size, theta = scattering angle
        blurred = np.zeros_like(image)
        h, w = image.shape
        t_map_1 = self.binning(t_map_1)
        theta_max_x = float(self.dict_params["RMS scattering angle in X (μrad)"])*1e-6
        theta_max_y = float(self.dict_params["RMS scattering angle in Y (μrad)"])*1e-6
        theta_map_x = theta_max_x*(t_map_1 - t_map_1.min())/(t_map_1.max() - t_map_1.min())
        theta_map_y = theta_max_y*(t_map_1 - t_map_1.min())/(t_map_1.max() - t_map_1.min())
        N_sw = 11
        patch_size = 2*N_sw + 1 
        for i in range(N_sw, h-N_sw):
            for j in range(N_sw, w-N_sw):
                patch = image[i-N_sw:i+N_sw+1, j-N_sw:j+N_sw+1]
                sigma_x = (self.d_prop * theta_map_x[i,j]) / (self.grat.sim_pixel_m)
                sigma_y = (self.d_prop * theta_map_y[i,j]) / (self.grat.sim_pixel_m)
                #blurred_patch = gaussian_filter(patch, sigma=sigma)
                x, y = np.arange(patch_size), np.arange(patch_size)
                XX, YY = np.meshgrid(x,y)
                if((sigma_x == 0) or (sigma_y==0)):
                    kernel = np.zeros(patch.shape)
                    kernel[N_sw, N_sw] = 1
                else: 
                    kernel = np.exp(-((XX - N_sw)**2)/(2*sigma_x**2))*np.exp(-((YY-N_sw)**2)/(2*sigma_y**2))
                    kernel = kernel/np.sum(kernel)
                blurred[i,j] = np.sum(patch * kernel)
        blurred[:N_sw,:] = image[:N_sw,:]
        blurred[-N_sw:,:] = image[-N_sw:,:]
        blurred[:,:N_sw] = image[:,:N_sw]
        blurred[:,-N_sw:] = image[:,-N_sw:]
        return blurred
    
    def create_ref_samp(self, bin_grat: np.ndarray, t_map_1: np.ndarray, t_map_2: np.ndarray, t_map_3: np.ndarray) -> np.ndarray:
        """
        Simulates reference and sample images for a single grating position.

        Args:
            bin_grat (np.ndarray): Binary grating pattern.
            t_map_1, t_map_2, t_map_3 (np.ndarray): Thickness maps for sample regions.

        Returns:
            tuple: Reference and sample images.
        """
        
        if(self.type_of_source == "Cone"):
            M_1 = (self.d_source_grat + self.d_grat_det)/self.d_source_grat
            M_2 = (self.d_source_grat + self.d_grat_samp)/self.d_source_grat
            M_3 = (self.d_source_samp + self.d_samp_det)/self.d_source_samp
        else:
            M_1, M_2, M_3 = 1, 1, 1 


        wf_bg = self.propagation(wf=self.wf_init, z=self.d_source_grat)
        wf_grat = self.inter_grat_det(wf=wf_bg, bin_grat=bin_grat)
        
        wf_ref = wf_grat
        wf_ref = self.propagation(wf=wf_ref, z=self.d_grat_det/M_1)#self.d_grat_samp+self.samp.t_m
        wf_ref = self.scale(np.real(wf_ref), M=M_1) +1j*self.scale(np.imag(wf_ref), M=M_1)
        I_ref = np.abs(wf_ref)**2
        I_ref = self.binning(I_ref)
        I_ref = self.convolve_PSF_total(image=I_ref)
        I_ref = np.random.poisson(lam=self.n_ph*I_ref)
        

        wf_samp = wf_grat
        wf_samp = self.propagation(wf=wf_samp, z=(self.d_grat_samp - self.samp.t_m/2)/M_2)
        wf_samp = self.scale(np.real(wf_samp), M=M_2) +1j*self.scale(np.imag(wf_samp), M=M_2)
        wf_samp = self.inter_samp_det(wf=wf_samp, t_map_1=t_map_1, t_map_2=t_map_2, t_map_3=t_map_3)
        wf_samp = self.propagation(wf=wf_samp, z=(self.d_prop + self.samp.t_m/2)*(M_2/M_3))
        wf_samp = self.scale(np.real(wf_samp), M=M_3) +1j*self.scale(np.imag(wf_samp), M=M_3)
        I_samp = np.abs(wf_samp)**2
        I_samp = self.binning(I_samp)
        I_samp = self.convolve_PSF_total(image=I_samp)
        #I_samp = self.add_dark_field(image=I_samp, t_map_1=t_map_1)
        I_samp = np.random.poisson(lam=self.n_ph*I_samp)
        
        return I_ref, I_samp
    
    def create_ref_samp_stacks(self) -> np.ndarray:
        """
        Simulates stacks of reference and sample images for all grating positions.

        Returns:
            tuple: Arrays of reference and sample images for all positions.
        """

        grat_array = self.grat.obtain_grat_array()
        t_map_1, t_map_2, t_map_3 = self.samp.create_sample(self.theta_y)
        I_refs = []
        I_samps = []

        for bin_grat in tqdm(grat_array):
            I_ref, I_samp = self.create_ref_samp(bin_grat=bin_grat, t_map_1=t_map_1, t_map_2=t_map_2, t_map_3=t_map_3)
            I_refs.append(I_ref)
            I_samps.append(I_samp)

        I_refs = np.array(I_refs)
        I_samps = np.array(I_samps)

        return I_refs, I_samps
    
    def scale(self, img, M):
        """
        Scales the image according to the magnification factor.

        Args:
            img (np.ndarray): Input image.
            M (float): Magnification factor.

        Returns:
            np.ndarray: Scaled image.
        """
        if(self.type_of_source == "Cone"):
            #h, w = img.shape[:2]
            #zoom_factor = M
            ## For multichannel images we don't want to apply the zoom factor to the RGB
            ## dimension, so instead we create a tuple of zoom factors, one per array
            ## dimension, with 1's for any trailing dimensions after the width and height.
            #zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
            ## Bounding box of the zoomed-in region within the input array
            #zh = int(np.round(h / zoom_factor))
            #zw = int(np.round(w / zoom_factor))
            #top = (h - zh) // 2
            #left = (w - zw) // 2
    #
            #out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, order=0)
    #
            ## `out` might still be slightly larger than `img` due to rounding, so
            ## trim off any extra pixels at the edges
            #trim_top = ((out.shape[0] - h) // 2)
            #trim_left = ((out.shape[1] - w) // 2)
            #out = out[trim_top:trim_top+h, trim_left:trim_left+w]

            zoom_factor = M
            M, N = img.shape
            M_new = int(np.round(M * zoom_factor))
            N_new = int(np.round(N * zoom_factor))
        
            # Forward FFT and center
            f = np.fft.fftshift(np.fft.fft2(img))
        
            # Create zero-padded array
            F_zoomed = np.zeros((M_new, N_new), dtype=complex)
        
            # Determine cropping/padding indices
            m0, n0 = M // 2, N // 2
            m1, n1 = M_new // 2, N_new // 2
        
            # Copy central spectrum (or pad if zoom out)
            r_min = min(m0, m1)
            c_min = min(n0, n1)
            F_zoomed[m1 - r_min:m1 + r_min, n1 - c_min:n1 + c_min] = \
                f[m0 - r_min:m0 + r_min, n0 - c_min:n0 + c_min]
        
            # Inverse FFT
            zoomed = np.abs(np.fft.ifft2(np.fft.ifftshift(F_zoomed)))
        
            # Normalize
            zoomed *= zoom_factor**2

            # Center-crop to original size
            crop_m = (M_new - M) // 2
            crop_n = (N_new - N) // 2
            result = zoomed[crop_m:crop_m + M, crop_n:crop_n + N]
            out = result
        
        else:
            out = img

        return out
    
class SimulationEI:

    def __init__(self, dict_params, grat, samp, E, theta_y=0) -> None:
        """
        Initializes the SimulationEI class for Edge Illumination simulation.

        Args:
            dict_params (dict): Dictionary of simulation parameters.
            grat: Grating object.
            samp: Sample object.
            E (float): Energy value in keV.
            theta_y (float, optional): Rotation angle around the Y axis. Default is 0.
        """

        self.grat = grat
        self.samp = samp
        self.dict_params = dict_params
        self.type_of_source = dict_params["Source geometry"]
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.M_mask = 2*self.grat.sim_pixel_m*self.binning_factor/(self.grat.px_um*1e-6) 
        self.shift_z_mask_cm = float(dict_params["Shift grating in prop. axis (cm)"])
        self.d_source_det = float(dict_params["Source-Detector distance (m)"])
        self.d_source_grat = self.d_source_det/self.M_mask + (self.shift_z_mask_cm*1e-2)
        self.d_source_samp = self.d_source_grat + float(dict_params["Grating-Sample distance (m)"])
        self.d_grat_det = self.d_source_det - self.d_source_grat
        self.d_samp_det = self.d_source_det - self.d_source_samp
        self.d_grat_samp = np.abs(self.d_source_samp - self.d_source_grat)
        self.d_prop = self.d_samp_det - (float(dict_params["Whole sample thickness (mm)"])*1e-3)/2
        self.wf_init = np.ones(self.grat.img_size)#np.random.rand(img_size[0], img_size[1])*np.exp(1j*np.random.uniform(-np.pi, np.pi, size=img_size))##
        self.k =  2. * np.pi * (E * 1e3 * 1.6022e-19) / (scipy.constants.h * scipy.constants.c)   # 1/m
        self.fwhm = float(dict_params["FWHM PSF (pix)"])
        self.psf_size = (2*self.fwhm+1, 2*self.fwhm+1)
        self.f_um = float(dict_params["Focal spot size (μm)"])
        self.n_ph = float(dict_params["Num. events per pixel"])
        self.theta_y = theta_y

    def inter_grat_det(self, wf: np.ndarray, bin_grat: np.ndarray) -> np.ndarray:
        """
        Applies the grating interaction to the wavefront.

        Args:
            wf (np.ndarray): Input wavefront.
            bin_grat (np.ndarray): Binary grating pattern.

        Returns:
            np.ndarray: Modified wavefront after grating interaction.
        """
        return wf*np.exp(-1j* self.grat.delta*self.k*self.grat.t_m*bin_grat)*np.exp(-(self.grat.mu/2)*self.grat.t_m*bin_grat)
    
    def inter_samp_det(self, wf: np.ndarray, t_map_1: np.ndarray, t_map_2: np.ndarray, t_map_3: np.ndarray) -> np.ndarray:
        """
        Applies the sample interaction to the wavefront.

        Args:
            wf (np.ndarray): Input wavefront.
            t_map_1, t_map_2, t_map_3 (np.ndarray): Thickness maps for sample regions.

        Returns:
            np.ndarray: Modified wavefront after sample interaction.
        """
        return wf*np.exp(-1j* self.k * (self.samp.delta_1*t_map_1 + self.samp.delta_2*t_map_2 + self.samp.delta_3*t_map_3)) * np.exp(-(self.samp.mu_1*t_map_1 + self.samp.mu_2*t_map_2 + self.samp.mu_3*t_map_3)/2)

    def fres_ker_fourier(self, z):
        """
        Computes the Fresnel kernel in Fourier space for propagation.

        Args:
            z (float): Propagation distance.

        Returns:
            np.ndarray: Fresnel kernel in Fourier space.
        """
        u = 2 * np.pi * scipy.fft.fftfreq(self.grat.img_size[1]) / self.grat.sim_pixel_m
        v = 2 * np.pi * scipy.fft.fftfreq(self.grat.img_size[0]) / self.grat.sim_pixel_m
        UU, VV = np.meshgrid(u, v)
        #fres_ker_fou = np.exp(1j*self.k*z)* np.exp(-(1/2)*1j * (z/ self.k) * (UU ** 2 + VV**2))
        fres_ker_fou = np.exp(1j * z * np.sqrt(self.k**2 - UU ** 2 - VV**2))
        return fres_ker_fou

    def propagation(self, wf: np.ndarray, z: float) -> np.ndarray:
        """
        Propagates the wavefront by a distance z using the Fresnel kernel.

        Args:
            wf (np.ndarray): Input wavefront.
            z (float): Propagation distance.

        Returns:
            np.ndarray: Propagated wavefront.
        """
        
        fres_ker_fou = self.fres_ker_fourier(z=z)
        wf_prop = scipy.fft.ifftn(scipy.fft.fftn(wf, fres_ker_fou.shape, workers = -1)*fres_ker_fou, workers = -1)

        return wf_prop
    
    def binning(self, image: np.ndarray) -> np.ndarray:
        """
        Applies binning to the image according to the binning factor.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Binned image.
        """
        new_shape = (image.shape[0] // self.binning_factor, image.shape[1] // self.binning_factor)
        binned_image = image.reshape(new_shape[0], self.binning_factor, new_shape[1], self.binning_factor).mean(axis=(1, 3))

        return binned_image
    
    def convolve_PSF(self, image: np.ndarray) -> np.ndarray:
        """
        Convolves the image with a point spread function (PSF).

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image after PSF convolution.
        """
        x, y = np.arange(self.psf_size[1]), np.arange(self.psf_size[0])
        XX, YY = np.meshgrid(x,y)
        psf = np.exp(-4*np.log(2)*(XX**2 + YY**2)/self.fwhm**2)
        image_conv = scipy.fft.ifftn(scipy.fft.fftn(image, workers=-1)*scipy.fft.fftn(psf,image.shape, workers=-1), workers=-1)
        return np.abs(image_conv)
    
    def convolve_PSF_total(self, image: np.ndarray) -> np.ndarray:
        """
        Applies system PSF convolution, considering source geometry.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image after system PSF convolution.
        """
        if(self.type_of_source == "Cone"):
            M_samp = (self.d_source_samp + self.d_samp_det)/self.d_source_samp
            f_pix = (self.f_um*1e-6/self.grat.sim_pixel_m)
            self.fwhm_sys = int(np.sqrt((self.fwhm/M_samp)**2 + (f_pix**2)*((M_samp-1)/M_samp)**2))
        else:
            self.fwhm_sys = int(self.fwhm)
        #x, y = np.arange(2*self.fwhm_sys+1), np.arange(2*self.fwhm_sys+1)
        #XX, YY = np.meshgrid(x,y)
        #psf = np.exp(-4*np.log(2)*(XX**2 + YY**2)/self.fwhm_sys**2)
        #image_conv = scipy.fft.ifftn(scipy.fft.fftn(image, workers=-1)*scipy.fft.fftn(psf,image.shape, workers=-1), workers=-1)
        #convolved_image = np.abs(image_conv)
        convolved_image = gaussian_filter(image, sigma=self.fwhm_sys/2.355)
        return convolved_image
    
    def add_dark_field(self, image, t_map_1, patch_size=32):
        """
        Adds dark-field blur to the image based on sample thickness.

        Args:
            image (np.ndarray): Input image.
            t_map_1 (np.ndarray): Thickness map for region 1.
            patch_size (int, optional): Size of the patch for local blurring.

        Returns:
            np.ndarray: Image with dark-field effect.
        """
        ## sigma = z_prop*theta/pixel_size, theta = scattering angle
        blurred = np.zeros_like(image)
        h, w = image.shape
        t_map_1 = self.binning(t_map_1)
        theta_max_x = float(self.dict_params["RMS scattering angle in X (μrad)"])*1e-6
        theta_max_y = float(self.dict_params["RMS scattering angle in Y (μrad)"])*1e-6
        theta_map_x = theta_max_x*(t_map_1 - t_map_1.min())/(t_map_1.max() - t_map_1.min())
        theta_map_y = theta_max_y*(t_map_1 - t_map_1.min())/(t_map_1.max() - t_map_1.min())
        N_sw = 11
        patch_size = 2*N_sw + 1 
        for i in range(N_sw, h-N_sw):
            for j in range(N_sw, w-N_sw):
                patch = image[i-N_sw:i+N_sw+1, j-N_sw:j+N_sw+1]
                sigma_x = (self.d_prop * theta_map_x[i,j]) / (self.grat.sim_pixel_m)
                sigma_y = (self.d_prop * theta_map_y[i,j]) / (self.grat.sim_pixel_m)
                #blurred_patch = gaussian_filter(patch, sigma=sigma)
                x, y = np.arange(patch_size), np.arange(patch_size)
                XX, YY = np.meshgrid(x,y)
                if((sigma_x == 0) or (sigma_y==0)):
                    kernel = np.zeros(patch.shape)
                    kernel[N_sw, N_sw] = 1
                else: 
                    kernel = np.exp(-((XX - N_sw)**2)/(2*sigma_x**2))*np.exp(-((YY-N_sw)**2)/(2*sigma_y**2))
                    kernel = kernel/np.sum(kernel)
                blurred[i,j] = np.sum(patch * kernel)
        blurred[:N_sw,:] = image[:N_sw,:]
        blurred[-N_sw:,:] = image[-N_sw:,:]
        blurred[:,:N_sw] = image[:,:N_sw]
        blurred[:,-N_sw:] = image[:,-N_sw:]
        return blurred
    
    def create_ref_samp(self, bin_grat: np.ndarray, t_map_1: np.ndarray, t_map_2: np.ndarray, t_map_3: np.ndarray) -> np.ndarray:
        """
        Simulates reference and sample images for a single grating position.

        Args:
            bin_grat (np.ndarray): Binary grating pattern.
            t_map_1, t_map_2, t_map_3 (np.ndarray): Thickness maps for sample regions.

        Returns:
            tuple: Reference and sample images.
        """
        
        if(self.type_of_source == "Cone"):
            M_1 = (self.d_source_grat + self.d_grat_det)/self.d_source_grat
            M_2 = (self.d_source_grat + self.d_grat_samp)/self.d_source_grat
            M_3 = (self.d_source_samp + self.d_samp_det)/self.d_source_samp
        else:
            M_1, M_2, M_3 = 1, 1, 1 


        wf_bg = self.propagation(wf=self.wf_init, z=self.d_source_grat)
        wf_grat = self.inter_grat_det(wf=wf_bg, bin_grat=bin_grat)
        
        wf_ref = wf_grat
        wf_ref = self.propagation(wf=wf_ref, z=self.d_grat_det/M_1)#self.d_grat_samp+self.samp.t_m
        wf_ref = self.scale(np.real(wf_ref), M=M_1) +1j*self.scale(np.imag(wf_ref), M=M_1)
        I_ref = np.abs(wf_ref)**2
        I_ref = self.binning(I_ref)
        I_ref = self.convolve_PSF_total(image=I_ref)
        I_ref = np.random.poisson(lam=self.n_ph*I_ref)
        

        wf_samp = wf_grat
        wf_samp = self.propagation(wf=wf_samp, z=(self.d_grat_samp - self.samp.t_m/2)/M_2)
        wf_samp = self.scale(np.real(wf_samp), M=M_2) +1j*self.scale(np.imag(wf_samp), M=M_2)
        wf_samp = self.inter_samp_det(wf=wf_samp, t_map_1=t_map_1, t_map_2=t_map_2, t_map_3=t_map_3)
        wf_samp = self.propagation(wf=wf_samp, z=(self.d_prop + self.samp.t_m/2)*(M_2/M_3))
        wf_samp = self.scale(np.real(wf_samp), M=M_3) +1j*self.scale(np.imag(wf_samp), M=M_3)
        I_samp = np.abs(wf_samp)**2
        I_samp = self.binning(I_samp)
        I_samp = self.convolve_PSF_total(image=I_samp)
        #I_samp = self.add_dark_field(image=I_samp, t_map_1=t_map_1)
        I_samp = np.random.poisson(lam=self.n_ph*I_samp)
        
        return I_ref, I_samp
    
    def create_ref_samp_stacks(self) -> np.ndarray:
        """
        Simulates stacks of reference and sample images for all grating positions.

        Returns:
            tuple: Arrays of reference and sample images for all positions.
        """

        grat_array = self.grat.obtain_grat_array()
        t_map_1, t_map_2, t_map_3 = self.samp.create_sample(self.theta_y)
        I_refs = []
        I_samps = []

        for bin_grat in tqdm(grat_array):
            I_ref, I_samp = self.create_ref_samp(bin_grat=bin_grat, t_map_1=t_map_1, t_map_2=t_map_2, t_map_3=t_map_3)
            I_refs.append(I_ref)
            I_samps.append(I_samp)

        I_refs = np.array(I_refs)
        I_samps = np.array(I_samps)

        return I_refs, I_samps
    
    def scale(self, img, M):
        """
        Scales the image according to the magnification factor.

        Args:
            img (np.ndarray): Input image.
            M (float): Magnification factor.

        Returns:
            np.ndarray: Scaled image.
        """
        if(self.type_of_source == "Cone"):
            #h, w = img.shape[:2]
            #zoom_factor = M
            ## For multichannel images we don't want to apply the zoom factor to the RGB
            ## dimension, so instead we create a tuple of zoom factors, one per array
            ## dimension, with 1's for any trailing dimensions after the width and height.
            #zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
            ## Bounding box of the zoomed-in region within the input array
            #zh = int(np.round(h / zoom_factor))
            #zw = int(np.round(w / zoom_factor))
            #top = (h - zh) // 2
            #left = (w - zw) // 2
    #
            #out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, order=0)
    #
            ## `out` might still be slightly larger than `img` due to rounding, so
            ## trim off any extra pixels at the edges
            #trim_top = ((out.shape[0] - h) // 2)
            #trim_left = ((out.shape[1] - w) // 2)
            #out = out[trim_top:trim_top+h, trim_left:trim_left+w]

            zoom_factor = M
            M, N = img.shape
            M_new = int(np.round(M * zoom_factor))
            N_new = int(np.round(N * zoom_factor))
        
            # Forward FFT and center
            f = np.fft.fftshift(np.fft.fft2(img))
        
            # Create zero-padded array
            F_zoomed = np.zeros((M_new, N_new), dtype=complex)
        
            # Determine cropping/padding indices
            m0, n0 = M // 2, N // 2
            m1, n1 = M_new // 2, N_new // 2
        
            # Copy central spectrum (or pad if zoom out)
            r_min = min(m0, m1)
            c_min = min(n0, n1)
            F_zoomed[m1 - r_min:m1 + r_min, n1 - c_min:n1 + c_min] = \
                f[m0 - r_min:m0 + r_min, n0 - c_min:n0 + c_min]
        
            # Inverse FFT
            zoomed = np.abs(np.fft.ifft2(np.fft.ifftshift(F_zoomed)))
        
            # Normalize
            zoomed *= zoom_factor**2

            # Center-crop to original size
            crop_m = (M_new - M) // 2
            crop_n = (N_new - N) // 2
            result = zoomed[crop_m:crop_m + M, crop_n:crop_n + N]
            out = result
        
        else:
            out = img

        return out
    
class SimulationInline:

    def __init__(self, dict_params, samp, E, theta_y) -> None:
        """
        Initializes the SimulationInline class for inline simulation.

        Args:
            dict_params (dict): Dictionary of simulation parameters.
            samp: Sample object.
            E (float): Energy value in keV.
            theta_y (float): Rotation angle around the Y axis.
        """

        self.samp = samp
        self.type_of_source = dict_params["Source geometry"]
        self.binning_factor = int(float(dict_params["Binning factor"]))
        self.d_source_det = float(dict_params["Source-Detector distance (m)"])
        self.d_source_samp = float(dict_params["Source-Sample distance (m)"])
        self.d_samp_det = self.d_source_det - self.d_source_samp
        self.d_prop = self.d_samp_det - (float(dict_params["Whole sample thickness (mm)"])*1e-3)/2
        self.wf_init = np.ones(self.samp.img_size)#np.random.rand(img_size[0], img_size[1])*np.exp(1j*np.random.uniform(-np.pi, np.pi, size=img_size))##
        self.k =  2. * np.pi * (E * 1e3 * 1.6022e-19) / (scipy.constants.h * scipy.constants.c)   # 1/m
        self.fwhm = float(dict_params["FWHM PSF (pix)"])
        self.psf_size = (2*self.fwhm+1, 2*self.fwhm+1)
        self.f_um = float(dict_params["Focal spot size (μm)"])
        self.n_ph = float(dict_params["Num. events per pixel"])
        self.theta_y = theta_y
    
    def inter_samp_det(self, wf: np.ndarray, t_map_1: np.ndarray, t_map_2: np.ndarray, t_map_3: np.ndarray) -> np.ndarray:
        """
        Applies the sample interaction to the wavefront.

        Args:
            wf (np.ndarray): Input wavefront.
            t_map_1, t_map_2, t_map_3 (np.ndarray): Thickness maps for sample regions.

        Returns:
            np.ndarray: Modified wavefront after sample interaction.
        """
        return wf*np.exp(-1j* self.k * (self.samp.delta_1*t_map_1 + self.samp.delta_2*t_map_2 + self.samp.delta_3*t_map_3)) * np.exp(-(self.samp.mu_1*t_map_1 + self.samp.mu_2*t_map_2 + self.samp.mu_3*t_map_3)/2)

    def fres_ker_fourier(self, z):
        """
        Computes the Fresnel kernel in Fourier space for propagation.

        Args:
            z (float): Propagation distance.

        Returns:
            np.ndarray: Fresnel kernel in Fourier space.
        """
        u = 2 * np.pi * scipy.fft.fftfreq(self.samp.img_size[1]) / self.samp.sim_pixel_m
        v = 2 * np.pi * scipy.fft.fftfreq(self.samp.img_size[0]) / self.samp.sim_pixel_m
        UU, VV = np.meshgrid(u, v)
        #fres_ker_fou = np.exp(1j*self.k*z)* np.exp(-(1/2)*1j * (z/ self.k) * (UU ** 2 + VV**2))
        fres_ker_fou = np.exp(1j * z * np.sqrt(self.k**2 - UU ** 2 - VV**2))
        return fres_ker_fou

    def propagation(self, wf: np.ndarray, z: float) -> np.ndarray:
        """
        Propagates the wavefront by a distance z using the Fresnel kernel.

        Args:
            wf (np.ndarray): Input wavefront.
            z (float): Propagation distance.

        Returns:
            np.ndarray: Propagated wavefront.
        """
        
        fres_ker_fou = self.fres_ker_fourier(z=z)
        wf_prop = scipy.fft.ifftn(scipy.fft.fftn(wf, fres_ker_fou.shape, workers = -1)*fres_ker_fou, workers = -1)

        return wf_prop
    
    def binning(self, image: np.ndarray) -> np.ndarray:
        """
        Applies binning to the image according to the binning factor.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Binned image.
        """
        new_shape = (image.shape[0] // self.binning_factor, image.shape[1] // self.binning_factor)
        binned_image = image.reshape(new_shape[0], self.binning_factor, new_shape[1], self.binning_factor).mean(axis=(1, 3))

        return binned_image
    
    def convolve_PSF(self, image: np.ndarray) -> np.ndarray:
        """
        Convolves the image with a point spread function (PSF).

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image after PSF convolution.
        """
        x, y = np.arange(self.psf_size[1]), np.arange(self.psf_size[0])
        XX, YY = np.meshgrid(x,y)
        psf = np.exp(-4*np.log(2)*(XX**2 + YY**2)/self.fwhm**2)
        image_conv = scipy.fft.ifftn(scipy.fft.fftn(image, workers=-1)*scipy.fft.fftn(psf,image.shape, workers=-1), workers=-1)
        return np.abs(image_conv)
    
    def convolve_PSF_total(self, image: np.ndarray) -> np.ndarray:
        """
        Applies system PSF convolution, considering source geometry.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Image after system PSF convolution.
        """
        if(self.type_of_source == "Cone"):
            M_samp = (self.d_source_samp + self.d_samp_det)/self.d_source_samp
            f_pix = (self.f_um*1e-6/self.samp.sim_pixel_m)
            self.fwhm_sys = int(np.sqrt((self.fwhm/M_samp)**2 + (f_pix**2)*((M_samp-1)/M_samp)**2))
        else:
            self.fwhm_sys = int(self.fwhm)
        #x, y = np.arange(2*self.fwhm_sys+1), np.arange(2*self.fwhm_sys+1)
        #XX, YY = np.meshgrid(x,y)
        #psf = np.exp(-4*np.log(2)*(XX**2 + YY**2)/self.fwhm_sys**2)
        #image_conv = scipy.fft.ifftn(scipy.fft.fftn(image, workers=-1)*scipy.fft.fftn(psf,image.shape, workers=-1), workers=-1)
        #convolved_image = np.abs(image_conv)
        convolved_image = gaussian_filter(image, sigma=self.fwhm_sys/2.355)
        return convolved_image
    
    def create_ref_samp(self) -> np.ndarray:
        """
        Simulates reference and sample images for the inline setup.

        Returns:
            tuple: Reference and sample images.
        """

        t_map_1, t_map_2, t_map_3 = self.samp.create_sample(self.theta_y)
        
        if(self.type_of_source == "Cone"):
            M_1 = (self.d_source_det)/self.d_source_samp
        else:
            M_1 = 1


        wf_bg = self.propagation(wf=self.wf_init, z=self.d_source_samp - self.samp.t_m/2)
        
        wf_ref = wf_bg
        wf_ref = self.propagation(wf=wf_ref, z=self.d_prop/M_1)#self.d_grat_samp+self.samp.t_m
        wf_ref = self.scale(np.real(wf_ref), M=M_1) +1j*self.scale(np.imag(wf_ref), M=M_1)
        I_ref = np.abs(wf_ref)**2
        I_ref = self.binning(I_ref)
        I_ref = self.convolve_PSF_total(image=I_ref)
        I_ref = np.random.poisson(lam=self.n_ph*I_ref)
        

        wf_samp = wf_bg
        wf_samp = self.inter_samp_det(wf=wf_samp, t_map_1=t_map_1, t_map_2=t_map_2, t_map_3=t_map_3)
        wf_samp = self.propagation(wf=wf_samp, z=(self.d_prop + self.samp.t_m/2)/M_1)
        wf_samp = self.scale(np.real(wf_samp), M=M_1) +1j*self.scale(np.imag(wf_samp), M=M_1)
        I_samp = np.abs(wf_samp)**2
        I_samp = self.binning(I_samp)
        I_samp = self.convolve_PSF_total(image=I_samp)
        I_samp = np.random.poisson(lam=self.n_ph*I_samp)
        
        return I_ref, I_samp
      
    def scale(self, img, M):
        """
        Scales the image according to the magnification factor.

        Args:
            img (np.ndarray): Input image.
            M (float): Magnification factor.

        Returns:
            np.ndarray: Scaled image.
        """
        if(self.type_of_source == "Cone"):
            """
            h, w = img.shape[:2]
            zoom_factor = M
            # For multichannel images we don't want to apply the zoom factor to the RGB
            # dimension, so instead we create a tuple of zoom factors, one per array
            # dimension, with 1's for any trailing dimensions after the width and height.
            zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
    
            out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, order=0)
    
            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]
            """

            zoom_factor = M
            M, N = img.shape
            M_new = int(np.round(M * zoom_factor))
            N_new = int(np.round(N * zoom_factor))
        
            # Forward FFT and center
            f = np.fft.fftshift(np.fft.fft2(img))
        
            # Create zero-padded array
            F_zoomed = np.zeros((M_new, N_new), dtype=complex)
        
            # Determine cropping/padding indices
            m0, n0 = M // 2, N // 2
            m1, n1 = M_new // 2, N_new // 2
        
            # Copy central spectrum (or pad if zoom out)
            r_min = min(m0, m1)
            c_min = min(n0, n1)
            F_zoomed[m1 - r_min:m1 + r_min, n1 - c_min:n1 + c_min] = \
                f[m0 - r_min:m0 + r_min, n0 - c_min:n0 + c_min]
        
            # Inverse FFT
            zoomed = np.abs(np.fft.ifft2(np.fft.ifftshift(F_zoomed)))
        
            # Normalize
            zoomed *= zoom_factor**2

            # Center-crop to original size
            crop_m = (M_new - M) // 2
            crop_n = (N_new - N) // 2
            result = zoomed[crop_m:crop_m + M, crop_n:crop_n + N]
            out = result
        
        else:
            out = img

        return out
    