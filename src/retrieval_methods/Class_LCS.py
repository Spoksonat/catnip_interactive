import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.optimize import lsq_linear
from tqdm import tqdm

class LCS:
    def __init__(self, I_refs: np.ndarray, I_samps: np.ndarray, dict_params) -> None:
        """
        Initializes the LCS class for Local Contrast Separation retrieval.

        Args:
            I_refs (np.ndarray): Reference intensity images stack.
            I_samps (np.ndarray): Sample intensity images stack.
            dict_params (dict): Dictionary of retrieval parameters.
        """
        self.I_refs = I_refs/np.max(I_refs)
        self.I_samps = I_samps/np.max(I_refs)
        self.dict_params = dict_params

        self.grad_I_refs_x = np.zeros((I_refs.shape))
        self.grad_I_refs_y = np.zeros((I_refs.shape))
        for i in range(I_refs.shape[0]):
            grad_y, grad_x = np.gradient(I_refs[i,:,:])  # Compute gradients along y and x
            self.grad_I_refs_x[i,:,:] = grad_x
            self.grad_I_refs_y[i,:,:] = grad_y

        self.retrieve_LCS_Popcorn()

    def LCS_ij(self, method: str, i: int, j: int) -> tuple:
        """
        Solves the local system for a single pixel using analytical or LSQ-restricted method.

        Args:
            method (str): Solution method ("Analytical" or "LSQ-Restricted").
            i (int): Row index.
            j (int): Column index.

        Returns:
            tuple: Solution vector x, system matrix A, and right-hand side b.
        """

        b = self.I_refs[:,i,j]
        A = np.zeros((self.I_refs.shape[0],3))
        A[:,0] = self.I_samps[:,i,j]
        A[:,1] = self.grad_I_refs_x[:,i,j]
        A[:,2] = self.grad_I_refs_y[:,i,j]
        alpha = 0#np.std(A) / 10000
        
        if(method=="Analytical"):
            x = (np.linalg.inv(A.T @ A + alpha*np.eye(3)) @ A.T) @ b
        elif(method=="LSQ-Restricted"):
            A_p = np.vstack((A, alpha*np.eye(3)))
            b_p = np.vstack((b.reshape(-1,1),np.zeros(3).reshape(-1,1)))
            lower_bounds = [0, -np.inf, -np.inf] 
            upper_bounds = [np.inf, np.inf, np.inf]
            bounds = (lower_bounds, upper_bounds)
            result = lsq_linear(A_p, b_p.flatten(), bounds=bounds)
            x = result.x
        else:
            raise ValueError("Select between Analytical or LSQ-Restricted")

        return x, A, b
    
    def retrieve_LCS(self, method: str) -> tuple:
        """
        Retrieves transmission and phase gradients for all pixels using the specified method.

        Args:
            method (str): Solution method ("Analytical" or "LSQ-Restricted").

        Returns:
            tuple: Transmission (T), phase gradient in x (Dphi_x), and phase gradient in y (Dphi_y).
        """

        inv_T = np.zeros((self.I_refs.shape[-2], self.I_refs.shape[-1]))
        Dphi_x = np.zeros((self.I_refs.shape[-2], self.I_refs.shape[-1]))
        Dphi_y = np.zeros((self.I_refs.shape[-2], self.I_refs.shape[-1]))

        for i in tqdm(range(self.I_refs.shape[-2])):
            for j in range(self.I_refs.shape[-1]):
                inv_T[i,j], Dphi_x[i,j], Dphi_y[i,j] = self.LCS_ij(method,i,j)[0]

        self.T = 1/inv_T
        self.Dphi_x = Dphi_x
        self.Dphi_y = Dphi_y
    
    def retrieve_LCS_Popcorn(self):
        """
        Retrieves transmission and phase gradients for all pixels using QR decomposition for improved stability.
        """

        Nz, Ny, Nx= self.I_refs.shape
        LHS=np.ones(((Nz, Ny, Nx)))
        RHS=np.ones((((Nz,3, Ny, Nx))))
        solution=np.ones(((3, Ny, Nx)))
    
        #Prepare system matrices
        for i in range(Nz):
            #Right handSide
            gY_IrIr,gX_IrIr=np.gradient(self.I_refs[i])
            RHS[i]=[self.I_samps[i],gY_IrIr, gX_IrIr]
            LHS[i]=self.I_refs[i]
    
        #Solving system for each pixel 
        for i in tqdm(range(Ny)):
            for j in range(Nx):
                a=RHS[:,:,i,j]
                b=LHS[:,i,j]
                Q,R = np.linalg.qr(a) # qr decomposition of A
                Qb = np.dot(Q.T,b) # computing Q^T*b (project b onto the range of A)
                
                if R[2,2]==0 or R[1,1]==0 or R[0,0]==0:
                    temp=[1,0,0]
                else:
                    temp = np.linalg.solve(R,Qb) # solving R*x = Q^T*b
                solution[:,i,j]=temp
            
        self.T = 1/solution[0]
        self.Dphi_y = solution[1]
        self.Dphi_x = solution[2]