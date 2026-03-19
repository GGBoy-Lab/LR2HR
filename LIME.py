import numpy as np
from scipy.fft import *
from skimage import exposure
import cv2
from tqdm import trange


class LIME:
    """
    LIME class for image enhancement, improving visual quality by decomposing
    the image into illumination and reflectance components.
    """

    # initiate parameters
    def __init__(self, iterations, alpha, rho, gamma, strategy, exact):
        """
        Initialize LIME class parameters.

        Parameters:
        iterations: Number of iterations.
        alpha: Controls the smoothness of the reflectance component.
        rho: Ratio for updating the miu parameter.
        gamma: Used to adjust the brightness of the final enhanced image.
        strategy: Selected strategy affecting the calculation method of weight matrix W.
        exact: Whether to use the exact algorithm.
        """
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy
        self.exact = exact

    # load pictures and normalize
    def load(self, imgPath):
        """
        Load image and normalize it.

        Parameters:
        imgPath: Path to the image file.
        """
        self.loadimage(cv2.imread(imgPath) / 255)

    # initiate Dx,Dy,DTD
    def loadimage(self, L):
        """
        Initialize image data and related matrices.

        Parameters:
        L: Loaded image data.
        """
        self.L = L
        self.row = self.L.shape[0]
        self.col = self.L.shape[1]

        self.T_esti = np.max(self.L, axis=2)
        self.Dv = -np.eye(self.row) + np.eye(self.row, k=1)
        self.Dh = -np.eye(self.col) + np.eye(self.col, k=-1)

        dx = np.zeros((self.row, self.col))
        dy = np.zeros((self.row, self.col))
        dx[1, 0] = 1
        dx[1, 1] = -1
        dy[0, 1] = 1
        dy[1, 1] = -1
        dxf = fft2(dx)
        dyf = fft2(dy)
        self.DTD = np.conj(dxf) * dxf + np.conj(dyf) * dyf

        self.W = self.Strategy()

    # strategy 2
    def Strategy(self):
        """
        Calculate weight matrix W according to selected strategy.

        Returns:
        Weight matrix W.
        """
        if self.strategy == 2:
            self.Wv = 1 / (np.abs(self.Dv @ self.T_esti) + 1)
            self.Wh = 1 / (np.abs(self.T_esti @ self.Dh) + 1)
            return np.vstack((self.Wv, self.Wh))
        else:
            return np.ones((self.row * 2, self.col))

    # T subproblem
    def T_sub(self, G, Z, miu):
        """
        Solve T subproblem.

        Parameters:
        G: Current value of G variable.
        Z: Current value of Z variable.
        miu: Current value of miu parameter.

        Returns:
        Updated T value.
        """
        X = G - Z / miu
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]

        numerator = fft2(2 * self.T_esti + miu * (self.Dv @ Xv + Xh @ self.Dh))
        denominator = self.DTD * miu + 2
        T = np.real(ifft2(numerator / denominator))

        return exposure.rescale_intensity(T, (0, 1), (0.001, 1))

    # G subproblem
    def G_sub(self, T, Z, miu, W):
        """
        Solve G subproblem.

        Parameters:
        T: Current value of T variable.
        Z: Current value of Z variable.
        miu: Current value of miu parameter.
        W: Weight matrix W.

        Returns:
        Updated G value.
        """
        epsilon = self.alpha * W / miu
        temp = np.vstack((self.Dv @ T, T @ self.Dh)) + Z / miu
        return np.sign(temp) * np.maximum(np.abs(temp) - epsilon, 0)

    # Z subproblem
    def Z_sub(self, T, G, Z, miu):
        """
        Solve Z subproblem.

        Parameters:
        T: Current value of T variable.
        G: Current value of G variable.
        Z: Current value of Z variable.
        miu: Current value of miu parameter.

        Returns:
        Updated Z value.
        """
        return Z + miu * (np.vstack((self.Dv @ T, T @ self.Dh)) - G)

    # miu subproblem
    def miu_sub(self, miu):
        """
        Update miu parameter.

        Parameters:
        miu: Current value of miu.

        Returns:
        Updated miu value.
        """
        return miu * self.rho

    def run(self):
        """
        Run LIME algorithm for image enhancement according to selected strategy.

        Returns:
        Enhanced image.
        """
        # accurate algorithm
        if self.exact:
            T = np.zeros((self.row, self.col))
            G = np.zeros((self.row * 2, self.col))
            Z = np.zeros((self.row * 2, self.col))
            miu = 1

            for i in trange(0, self.iterations):
                T = self.T_sub(G, Z, miu)
                G = self.G_sub(T, Z, miu, self.W)
                Z = self.Z_sub(T, G, Z, miu)
                miu = self.miu_sub(miu)

            self.T = T ** self.gamma
            self.R = self.L / np.repeat(self.T[..., None], 3, axis=-1)
            return exposure.rescale_intensity(self.R, (0, 1)) * 255
        # TODO: rapid algorithm
        else:
            pass
