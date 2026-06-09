import numpy as np
from scipy import signal as sig

class Inline:
    def __init__(self, I_ref, I_samp, dict_params):
        """
        Initializes the Inline retrieval class.

        Args:
            I_ref (np.ndarray): Reference intensity image.
            I_samp (np.ndarray): Sample intensity image.
            dict_params (dict): Dictionary of retrieval parameters.
        """
        
        self.dict_params = dict_params
        self.I_ref = I_ref
        self.I_samp = I_samp
        self.FFcorr()

    def FFcorr(self):
        """
        Performs flat-field correction by dividing the sample image by the reference image.
        """
    
        self.T = self.I_samp/self.I_ref