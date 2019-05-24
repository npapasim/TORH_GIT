# Written by Vassili Savinov on 06/11/2018
# class to handle repreentations via toroidal harmonics

# it is initialized from N*M*E array
# the first two indices are for theta and phi
# and must give a good grid
# the last index is for eta, and can have just few steps

# I will also allow to initialize it by providing call-back functions
# to functions in Carthesian and toroidal coordinates

import numpy as np
import numpy.fft as npfft
from DTORH import DTORH

class TorHarmRep:
    # torCoordRep=N*M*E array
    # eta-range vector for the eta-values
    # thetaCounts = number of nodes along theta-dimension. Assuming thetaRange=[-pi, -pi+step, ..., pi-step]
    # phiCounts = number of nodes along phi-dimension. Assuming thetaRange=[0, -step, ..., 2*pi-step]
    # torhMode=is the gamma function used to normalize the toroidal functions
    def __init__(self, thetaCounts, phiCounts, etaRange, torCoordRep, torhMode=0):
        ########## check sanity of the inputs
        if not (isinstance(thetaCounts, int) and thetaCounts >= 3):
            raise ValueError('TorHarmRep: thetaCounts should be integer>=3')

        if not (isinstance(phiCounts, int) and phiCounts >= 3):
            raise ValueError('TorHarmRep: phiCounts should be integer>=3')

        if not (isinstance(etaRange, np.ndarray) and np.isreal(etaRange).all() and len(etaRange) >= 2):
            raise ValueError('TorHarmRep: etaRange should be real-valued numpy array with 2 or more entries')

        if not (isinstance(torCoordRep, np.ndarray) and len(torCoordRep.shape) == 3):
            raise ValueError('TorHarmRep: torCoordRep should be a 3d numpy array')

        if ( torCoordRep.shape[0] != thetaCounts   )    or \
           ( torCoordRep.shape[1] != phiCounts     )    or \
           ( torCoordRep.shape[2] != len(etaRange) ):

            raise ValueError('TorHarmRep: torCoordRep should be sized for theta, phi, eta')

        if not ( isinstance(torhMode,int) and (torhMode==0 or torhMode==1 or torhMode==2) ):
            raise ValueError('TorHarmRep: Need torhMode=0 or torhMode=1 or torhMode=2')

        self.torhMode = torhMode #store
        # done

        ########## condition the function and then take 2d DFT to get the nm values
        thetaRange = np.arange(thetaCounts)*(2.0*np.pi/thetaCounts)-np.pi
        phiRange = np.arange(phiCounts)*(2.0*np.pi/phiCounts)

        TH, PH, ETA = np.meshgrid(thetaRange, phiRange, etaRange, indexing='ij')# 'ij': do not swap the first two inputs

        # condition for further processing
        condFunc = torCoordRep / np.sqrt( np.cosh(ETA) - np.cos(TH) )

        ########## get the P,Q coeffs using dft
        # the problem with that is that one ends-up with indices in the wrong order
        nmFunc = np.zeros(condFunc.shape, dtype=np.complex)

        # create the range of indices in the same order as they would come out of DFT
        nRange = np.arange(thetaCounts)
        mRange = np.arange(phiCounts)
        self.N, self.M = np.meshgrid(nRange, mRange, indexing='ij')

        # compute the dft matrix
        for iEta in range(len(etaRange)):
            nmFunc[:, :, iEta] = ((1-2*np.mod(self.N, 2))/(thetaCounts*phiCounts))*npfft.fft2(condFunc[:, :, iEta])

        ######### get the a^P and a^Q coefficients for each NM
        # get the necessary P, Q, functions
        with DTORH() as dtorh: (self.pCube, self.qCube) = \
            dtorh.GetCubeNMZ(thetaCounts, phiCounts, np.cosh(etaRange), self.torhMode)

        # now compute the NM-matricies to find the coefficients
        d1Mat = np.sum( self.pCube * self.pCube, axis=2)
        d2Mat = np.sum( self.qCube * self.qCube, axis=2)
        hMat = np.sum( self.pCube * self.qCube, axis=2)

        w1Mat = np.sum( self.pCube * nmFunc, axis=2)
        w2Mat = np.sum( self.qCube * nmFunc, axis=2)

        # coefficient for P-functions
        self.aPMat = (d2Mat * w1Mat - hMat * w2Mat) / (d1Mat * d2Mat - np.abs(hMat)**2)
        # same for Q
        self.aQMat = (d1Mat * w2Mat - hMat * w1Mat) / (d1Mat * d2Mat - np.abs(hMat)**2)


