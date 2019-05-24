# Written by Vassili Savinov on 11/03/19
# decompose the field of the flying doughtnut in toroidal harmonics
# The relevant mathematical operations have been obtained by the Mathematica field

import unittest
from ToroHarmVecRep import ToroHarmVecRep
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.colors as mpc

# load the datat computed by mathematica
col_names = ['X', 'Y', 'Z', 'q1', 'q2',
             'r_dot_E', 'L_dot_E_im', 'r_dot_B', 'L_dot_B_im']
fdData = pd.read_csv('FD_numData_10q1.csv', names=col_names)

# prep data
X = np.array(fdData['X'])
Y = np.array(fdData['Y'])
Z = np.array(fdData['Z'])
#
r_dot_E = np.array(fdData['r_dot_E'], dtype=np.complex)
L_dot_E = 1j*np.array(fdData['L_dot_E_im'], dtype=np.complex)
#
r_dot_B = np.array(fdData['r_dot_B'], dtype=np.complex)
L_dot_B = 1j*np.array(fdData['L_dot_B_im'], dtype=np.complex)

##### do the decomposition
fdToroHarmVecRep = ToroHarmVecRep(X, Y, Z,
                                    raw_divFTens=0*X,
                                    raw_rDotFTens=r_dot_B,
                                    raw_LDotFTens=L_dot_B,
                                    nCount=20, mCount=35)

#
cCoeff_max_val_first=10*np.log10(np.max( np.abs(fdToroHarmVecRep.cCoeff_Tens[0,:,:]) ))
cCoeff_max_val_second=10*np.log10(np.max( np.abs(fdToroHarmVecRep.cCoeff_Tens[1,:,:]) ))
cCoeff_col_abs_norm_first = mpc.Normalize(vmin=cCoeff_max_val_first-60, vmax=cCoeff_max_val_first)
cCoeff_col_abs_norm_second = mpc.Normalize(vmin=cCoeff_max_val_second-60, vmax=cCoeff_max_val_second)
#
#
col_pha_norm = mpc.Normalize(vmin=-180, vmax=180)

pl.style.use('dark_background')

tickStep = 5#dB

pl.figure(1)
###
pl.subplot(221)
pl.imshow(10*np.log10( np.abs(np.squeeze( fdToroHarmVecRep.cCoeff_Tens[0,:,:] )) ), norm=cCoeff_col_abs_norm_first, cmap=pl.cm.hot)
pl.colorbar().set_label(' (dB)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|b^{(1)}\\right|$')
###
pl.subplot(222)
pl.imshow(10*np.log10( np.abs(np.squeeze( fdToroHarmVecRep.cCoeff_Tens[1,:,:] )) ), norm=cCoeff_col_abs_norm_second, cmap=pl.cm.hot)
pl.colorbar().set_label(' (dB)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|b^{(2)}\\right|$')


########## phase

pl.subplot(223)
pl.imshow(np.angle(np.squeeze(fdToroHarmVecRep.cCoeff_Tens[0,:,:]), deg=True), norm=col_pha_norm, cmap=pl.cm.jet)
pl.colorbar().set_label('(deg)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|c^{(2)}\\right|$')

pl.subplot(224)
pl.imshow(np.angle(np.squeeze(fdToroHarmVecRep.cCoeff_Tens[1,:,:]), deg=True), norm=col_pha_norm, cmap=pl.cm.jet)
pl.colorbar().set_label('(deg)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|c^{(2)}\\right|$')

## now do it for normalized values
# value of the harmonic (mean) and the coefficient
first_norm = np.abs(np.squeeze(fdToroHarmVecRep.cCoeff_Tens[0,:,:]*np.median(fdToroHarmVecRep.psiTens[0,:,:,:], axis=2)))
# value of the harmonic (mean) and the coefficient
second_norm = np.abs(np.squeeze(fdToroHarmVecRep.cCoeff_Tens[1,:,:]*np.median(fdToroHarmVecRep.psiTens[1,:,:,:], axis=2)))

norm_max_val=10*np.log10(np.max( [np.max( first_norm ), np.max( second_norm )]))
norm_norm = mpc.Normalize(vmin=norm_max_val-30, vmax=norm_max_val)

pl.figure(2)
###
pl.subplot(211)
pl.imshow(10*np.log10( first_norm ), norm=norm_norm, cmap=pl.cm.hot)
pl.colorbar().set_label(' (dB)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|b^{(1)}\\right|$')
###
pl.subplot(212)
pl.imshow(10 * np.log10(second_norm), norm=norm_norm, cmap=pl.cm.hot)
pl.colorbar().set_label(' (dB)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|b^{(2)}\\right|$')


### what is the difference between the first and second kind harmonics for the specified range
ratioTens = np.abs(np.squeeze(fdToroHarmVecRep.psiTens[0,:,:,:]/fdToroHarmVecRep.psiTens[1,:,:,:]))
harmRatioMean = np.mean(ratioTens, axis=2)*np.squeeze(np.abs(fdToroHarmVecRep.cCoeff_Tens[0,:,:]/fdToroHarmVecRep.cCoeff_Tens[1,:,:]))
harmRatioMedian = np.median(ratioTens, axis=2)

# want to check that the first kind (coeff*mean value) always dominates over the second kind
pl.figure(3)
#ratioTens_max_val=10*np.log10(np.max(harmRatioMean))
ratioTens_norm = mpc.Normalize(vmin=-60, vmax=20)
###
pl.imshow(10*np.log10(harmRatioMean), norm=ratioTens_norm, cmap=pl.cm.hot)
pl.colorbar().set_label(' (dB)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, fdToroHarmVecRep.cCoeff_Tens.shape[2], tickStep))
pl.title('$P/Q$')

pl.show()