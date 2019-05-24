# Written by Vassili Savinov on 17/09/2018
# a class wrapper to load and drop the DLL with the DTORTH functions

import ctypes as ct
import numpy as np
import scipy.special as sps

class DTORH:
    class NewNTooSmall(Exception): pass# new N is too small in dtorh1
    class NewMTooSmall(Exception): pass  # new M is too small in dtorh1
    class Internal(Exception): pass  # uknown error

    dllTorHarm = None

    # keep it simple for now, the DLL must be in the same folder
    #gets called by __enter__ amongst other things
    def __init__(self):
        self.dllTorHarm = ct.CDLL('wrapDTORH64.dll')

    def __enter__(self):# with statement
        return self

    def __exit__(self, exc_type, exc_value, traceback):# with statement
        dllHandle=self.dllTorHarm._handle
        del self.dllTorHarm#release object
        ct.windll.kernel32.FreeLibrary(dllHandle)#release lib

    # handle the errors raised by the DTORH functions
    def __HandleErrorCode(self, errCode):
        if errCode==0: return# no error

        if errCode   == 1 : raise DTORH.Internal('IPRE MUST BE 1 OR 2')
        elif errCode == 2 : raise DTORH.Internal('YOU MUST CHOOSE MODE=2')
        elif errCode == 3 : raise DTORH.Internal('M IS TOO LARGE FOR MODE=0. BETTER TRY MODE=1')
        elif errCode == 4 : raise DTORH.Internal('M IS TOO LARGE FOR MODE=0. BETTER TRY MODE=1')
        elif errCode == 5 : raise DTORH.Internal('IMPROPER ARGUMENT. Z MUST BE GREATER THAN 1')
        elif errCode == 6 : raise DTORH.Internal('M IS TOO LARGE FOR MODE=1,2')
        elif errCode == 10: raise DTORH.Internal('Invalid mode (need 0,1,2)!')
        elif errCode == 11: raise DTORH.Internal('Continued fraction convergence failed!')
        else:               raise DTORH.Internal('Uknown error')

    # mode 0 means evalute as equired
    # mode 1 normalize results by gamma(m+1/2)
    # mode 2, similar to mode 1, but without restriction on large z
    def FixedM(self, zVec, mVal, nVal, mode=0):
        plVec=np.zeros(len(zVec), dtype=np.double)
        qlVec = np.zeros(len(zVec), dtype=np.double)

        # allow to go one higher and to start at zero
        # +100 just for extra memory, it sometimes failes without it
        qlMem = np.zeros(nVal + 2+100, dtype=np.double)  # scaled by 1/gamma(m+1/2)
        plMem = np.zeros(nVal + 2+100, dtype=np.double)

        # diagonostics
        c_errCode=ct.c_int(0)
        c_newN=ct.c_int(0)

        iZ=0

        for zVal in zVec:
            self.dllTorHarm.wrapDTORH1((ct.c_double)(zVal),
                                  (ct.c_int)(mVal),
                                  (ct.c_int)(nVal),
                                  plMem.ctypes.data_as(ct.POINTER(ct.c_double)),
                                  qlMem.ctypes.data_as(ct.POINTER(ct.c_double)),
                                  ct.pointer(c_newN),
                                  ct.pointer(c_errCode),
                                  ct.c_int(mode))

            if c_newN.value<nVal:
                raise DTORH.NewNTooSmall('FixedM: New N is smaller than the target!')

            self.__HandleErrorCode(c_errCode.value)

            plVec[iZ] = plMem[nVal]
            qlVec[iZ] = qlMem[nVal]

            iZ=iZ+1

        return (plVec, qlVec)

    # get a cube of values with first dimension for n indices
    # second for m indices
    # third for zRange values
    # nCount means I will consider indices n=0...(nCount-1)
    # Folding means that the values for n and m larger than nCount/2 and mCount/2
    # will be folded back onto them-selves like in DFT
    # scale_fold include the aditional scaling for negative m PQs
    def GetCubeNMZ(self, nCountFull, mCountFull, zRange, mode=0, folding=False, scale_fold=True):
        # if we are folding there is no need to compute all values
        if folding:
            nCount=np.floor(nCountFull/2).astype(int)+1
            mCount = np.floor(mCountFull / 2).astype(int)+1
        else:
            nCount=nCountFull
            mCount=mCountFull

        #########  basic compute

        # allocate memory
        PCube = np.zeros( (nCount, mCount, len(zRange)), dtype=np.double )
        QCube = np.zeros( (nCount, mCount, len(zRange)), dtype=np.double )

        # allocate memory for each run

        mMax = mCount - 1 # fortran goes 0...max inclusive
        nMax = nCount - 1

        c_newM = ct.c_int(0)
        c_newN = ct.c_int(0)
        c_errCode = ct.c_int(0)

        qVec = -1.2*np.ones((nCount+1) * (mCount+1), dtype=np.double)# I get seg-faults if I don't provide one extra element of
        # memory
        pVec = -1.2*np.ones((nCount+1) * (mCount+1), dtype=np.double)

        # compute for each z-value
        for iZ in range(len(zRange)):
            self.dllTorHarm.wrapDTORH3(      (ct.c_double)(zRange[iZ]),
                                              (ct.c_int)(mCount),
                                              (ct.c_int)(nCount),
                                              (ct.c_int)(mMax),
                                              (ct.c_int)(nMax),
                                              pVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                                              qVec.ctypes.data_as(ct.POINTER(ct.c_double)),
                                              ct.pointer(c_newM),
                                              ct.pointer(c_newN),
                                              ct.pointer(c_errCode), ct.c_int(mode))

            if c_newN.value<nMax:
                raise DTORH.NewNTooSmall('FixedM: New N is smaller than the target!')

            if c_newM.value<mMax:
                raise DTORH.NewMTooSmall('FixedM: New M is smaller than the target!')

            self.__HandleErrorCode(c_errCode.value)# handle errors

            # tested this by computing data in matrix and one-by-one: this is the correct way to get out data.
            PCube[:, :, iZ] = pVec.reshape((nCount+1, mCount+1))[0:-1, 0:-1]  # now it is [N,M] array
            QCube[:, :, iZ] = qVec.reshape((nCount+1, mCount+1))[0:-1, 0:-1]  # now it is [N,M] array

        #### sort out folding
        if not folding:
            return PCube, QCube
        else:
            PCubeFull = np.zeros( (nCountFull, mCountFull, len(zRange)), dtype=np.double )
            QCubeFull = np.zeros( (nCountFull, mCountFull, len(zRange)), dtype=np.double )

            # n,m normal
            PCubeFull[0:nCount, 0:mCount, :] = PCube
            QCubeFull[0:nCount, 0:mCount, :] = QCube

            if scale_fold:
                # I need scaling it is easier to do it for all terms
                m_range = np.arange(0, mCount)
                n_range = np.arange(0, nCount)
                n_cube, m_cube, _ = np.meshgrid(n_range, m_range, zRange, indexing='ij')
                fold_prefactor_cube = ((-1) ** (mCountFull - m_cube)) * \
                                      sps.gamma(n_cube - ((mCountFull - m_cube)) + 1 / 2) / \
                                      sps.gamma(n_cube + ((mCountFull - m_cube)) + 1 / 2)

            else:
                fold_prefactor_cube = np.ones(PCube.shape, dtype=np.double)

            prefac_PCube = fold_prefactor_cube * PCube
            prefac_QCube = fold_prefactor_cube * QCube

            # fold on m
            if np.mod(mCountFull, 2) == 0:  # even n
                PCubeFull[0:nCount, mCount:, :] = prefac_PCube[0:nCount, np.arange(mCount - 2, 0, -1), :]
                QCubeFull[0:nCount, mCount:, :] = prefac_QCube[0:nCount, np.arange(mCount - 2, 0, -1), :]
            else:
                PCubeFull[0:nCount, mCount:, :] = prefac_PCube[0:nCount, np.arange(mCount - 1, 0, -1), :]
                QCubeFull[0:nCount, mCount:, :] = prefac_QCube[0:nCount, np.arange(mCount - 1, 0, -1), :]

            # mirror in n
            if np.mod(nCountFull,2)==0:# even n
                PCubeFull[nCount:, :, :] = PCubeFull[np.arange(nCount-2, 0, -1), :, :]
                QCubeFull[nCount:, :, :] = QCubeFull[np.arange(nCount - 2, 0, -1), :, :]
            else:
                PCubeFull[nCount:, :, :] = PCubeFull[np.arange(nCount - 1, 0, -1), :, :]
                QCubeFull[nCount:, :, :] = QCubeFull[np.arange(nCount - 1, 0, -1), :, :]

            return PCubeFull, QCubeFull



