# Written by Vassili Savinov on 06/11/2018
# unit test for the the TorHarmRep class

import unittest
from TorHarmRep import TorHarmRep
import numpy as np
from DTORH import DTORH
import numpy.random as npr

class TestTorHarmRep(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)# initialize parent class
        # prepare testing variables
        self.thetaCounts = 14
        self.phiCounts = 25
        self.etaRange = np.array((1.5, 2.0, 2.5), dtype=np.float)
        #from
        self.torCoordRep=np.zeros( (self.thetaCounts, self.phiCounts, len(self.etaRange) ), dtype=np.complex)

        # numerical precision for tests
        self.tiny = 1.0e-5

    # test inputs
    def test_InputSanity(self):
        # normal operation
        TorHarmRep(self.thetaCounts, self.phiCounts, self.etaRange, self.torCoordRep)

        # exception generation with bad inputs
        with self.assertRaises(ValueError):
            TorHarmRep(0, self.phiCounts, self.etaRange, self.torCoordRep)
            ###
        with self.assertRaises(ValueError):
            TorHarmRep(self.thetaCounts, 0, self.etaRange, self.torCoordRep)
        ###
        with self.assertRaises(ValueError):
            TorHarmRep(self.thetaCounts, self.phiCounts, np.array( (1, 2+1j) ), self.torCoordRep)
        ###
        with self.assertRaises(TypeError):
            TorHarmRep(self.thetaCounts, self.phiCounts, np.array( (1)), self.torCoordRep)
        ###
        with self.assertRaises(ValueError):
                TorHarmRep(self.thetaCounts, self.phiCounts, (1, 2 ), self.torCoordRep)
        ###
        with self.assertRaises(ValueError):
                TorHarmRep(self.thetaCounts, self.phiCounts, self.etaRange, np.zeros((3, 4)))

        ###
        with self.assertRaises(ValueError):
                TorHarmRep(self.thetaCounts, self.phiCounts, self.etaRange, np.zeros((self.thetaCounts, self.phiCounts, len(self.etaRange)+1)))

        ###
        with self.assertRaises(ValueError):
                TorHarmRep(self.thetaCounts, self.phiCounts, self.etaRange, self.torCoordRep, 3)

    # generate a candidate for decomposition and show that decomposition works
    def test_Decomposition(self):
        n = 4
        m = 5


        #generate inputs
        with DTORH() as dtorh: (plVec, qlVec) = dtorh.FixedM(np.cosh(self.etaRange), m, n)

        aP = np.array(10.8, dtype=np.complex)
        aQ = np.array(4.8, dtype=np.complex)

        # prep NM
        thetaRange = np.arange(self.thetaCounts) * (2.0 * np.pi / self.thetaCounts) - np.pi
        phiRange = np.arange(self.phiCounts) * (2.0 * np.pi / self.phiCounts)

        TH, PHI, ETA = np.meshgrid(thetaRange, phiRange, self.etaRange, indexing='ij')  # 'ij': do not swap the first two inputs

        # create cubes out of vectors (repeat eta-vectors along theta and phi directions)
        pCube = np.repeat( np.repeat(plVec[None, :], self.phiCounts, axis=0)[None, :, :], self.thetaCounts, axis=0)
        qCube = np.repeat( np.repeat(qlVec[None, :], self.phiCounts, axis=0)[None, :, :], self.thetaCounts, axis=0)

        # prep good self.torCoordRep
        self.torCoordRepGood = np.sqrt( np.cosh(ETA) - np.cos(TH) )*np.exp(1j*(n*TH+m*PHI))*(aP*pCube+aQ*qCube)

        # get decomposition
        thr=TorHarmRep(self.thetaCounts, self.phiCounts, self.etaRange, self.torCoordRepGood)

        # now thr.aPMat and thr.aQMat should contain non-zero entries with aP and aQ



        #######  test for aP
        pRatioMat = thr.aPMat/aP
        pMax = pRatioMat.flatten()[np.abs(pRatioMat).argmax()]
        pContrastVal = (np.sum(np.abs(pRatioMat)) - np.abs(pMax))/np.abs(pMax) # should be very small if there is a single large cell
        #
        self.assertTrue( np.abs(1.0-pMax)<self.tiny and pContrastVal<self.tiny )

        #######  test for aQ
        qRatioMat = thr.aQMat / aQ
        qMax = qRatioMat.flatten()[np.abs(qRatioMat).argmax()]
        qContrastVal = (np.sum(np.abs(qRatioMat)) - np.abs(qMax)) / np.abs(qMax)  # should be very small if there is a single large cell
        #
        self.assertTrue(np.abs(1.0 - qMax) < self.tiny and qContrastVal < self.tiny)



