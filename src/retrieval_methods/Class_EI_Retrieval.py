import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import rotate
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter


class Retrieval:
    def __init__(self, I_refs: np.ndarray, I_samps: np.ndarray, dict_params) -> None:
        """
        Initializes the Retrieval class for EI phase retrieval.

        Args:
            I_refs (np.ndarray): Reference intensity images.
            I_samps (np.ndarray): Sample intensity images.
            dict_params (dict): Dictionary of retrieval parameters.
        """
        self.I_refs = I_refs
        self.I_samps = I_samps
        self.binning_retrieval = dict_params["binning retrieval"]
        self.bool_correct = dict_params["correct damaged"]
        self.N = I_refs.shape[0]
        self.make_binning()
        self.RAWX = self.I_samps
        self.phase_retrieval()
    
    def binning(self, img, binning_factor):
        """
        Applies binning to an image using the specified binning factor.

        Args:
            img (np.ndarray): Input image.
            binning_factor (int): Factor by which to bin the image.

        Returns:
            np.ndarray: Binned image.
        """
        if(binning_factor == 1):
            binned_image = img
        elif(binning_factor%2 != 0):
            pad_x, pad_y = img.shape[1]%binning_factor, img.shape[0]%binning_factor 
            img = img[:-pad_y,:-pad_x]
        new_shape = (img.shape[0]//binning_factor, img.shape[1]//binning_factor)
        binned_image = img.reshape(new_shape[0], binning_factor, new_shape[1], binning_factor).mean(axis=(1,3))
        return binned_image
    
    def correct_raws(RAW):
        """
        Corrects damaged pixels in a raw image by replacing NaNs and outliers.

        Args:
            RAW (np.ndarray): Raw image data.

        Returns:
            np.ndarray: Corrected image.
        """
        mean = np.nanmean(RAW[RAW != 0.0])
        RAW = np.nan_to_num(RAW, nan=mean)
        RAW[RAW > np.percentile(RAW, 99)] = 0.0
        RAW[RAW < np.percentile(RAW, 1)] = 0.0
        RAW[RAW == 0] = mean
        return RAW
    
    def make_binning(self):
        """
        Applies binning to all reference and sample images and updates class attributes.
        """
        I_refs_binned = []
        I_samps_binned = []
        for i in range(len(self.I_refs)):
            I_refs_binned.append(self.binning(img=self.I_refs[i,:,:], binning_factor=self.binning_retrieval))
            I_samps_binned.append(self.binning(img=self.I_samps[i,:,:], binning_factor=self.binning_retrieval))

        self.I_refs = np.array(I_refs_binned)
        self.I_samps = np.array(I_samps_binned)

    def reconstruct_final(self, odd, even, n_imgs, inverse=False):
        """
        Reconstructs final odd and even images from separated odd/even stacks.

        Args:
            odd (list): List of odd images.
            even (list): List of even images.
            n_imgs (int): Number of images.
            inverse (bool, optional): If True, reverse image order.

        Returns:
            tuple: Arrays of reconstructed odd and even images.
        """
        final_odd=np.transpose(np.zeros((self.RAWX.shape[1],n_imgs*(self.RAWX.shape[1]//2))))
        final_even=np.transpose(np.zeros((self.RAWX.shape[1],n_imgs*(self.RAWX.shape[1]//2))))
        #print(final.shape)
        for i in range(n_imgs):
            fact=1+i
            odd_t=np.transpose(odd[i])
            even_t=np.transpose(even[i])
            for j in range((self.RAWX.shape[1]//2)):
                #print(odd_t[j])
                if inverse==True:
                    i_index=n_imgs-i-1
                else:
                    i_index=i
                final_odd[i_index+n_imgs*j]=odd_t[j]
                final_even[i_index+n_imgs*j]=even_t[j]
        return np.transpose(final_odd),np.transpose(final_even)

    def phase_retrieval(self):
        """
        Performs phase retrieval using the odd/even image separation and updates phase and attenuation attributes.
        """
        P=[]
        I=[]
        for i in range(self.N):
            P.append(self.RAWX[i][:,::2])
            I.append(self.RAWX[i][:,1::2])

        test_odd_i,test_even_i=np.asarray(self.reconstruct_final(P,I,self.N,True))
        tot_inverse=test_odd_i+test_even_i
        test_odd,test_even=np.asarray(self.reconstruct_final(P,I,self.N,False))
        tot_normal=test_odd+test_even

        t_odd= (test_odd + test_odd_i)*255/np.max(test_odd + test_odd_i)
        t_even=(test_even + test_even_i)*255/np.max(test_even + test_even_i)
        
        PhaseD=((test_odd-test_even)/(test_odd+test_even))
        Att=(t_odd+t_even)/np.max(t_odd+t_even)

        self.diff_phase = PhaseD
        self.T = Att


