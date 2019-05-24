# Written by Vassili Savinov on 13/03/2019
# apply the toroidal vector harmonic decomposition
# to a localized solution to Laplace equation

import pylab as pl
import numpy as np
import scipy.special as sps
from ToroHarmVecRep import ToroHarmVecRep
import matplotlib.colors as mpc

# The field I will use is L(r^-(l+1)Y_lm), where L is the
# angular momenutm operator, r is radius (spherical) and Y_lm is the sphericla
# harmonic

# prepare the lattice of points
xVec = np.linspace(-5, 5, 10)
yVec = np.linspace(-5, 5, 10)
zVec = np.linspace(-5, 5, 10)

xTens, yTens, zTens = np.meshgrid(xVec, yVec, zVec, indexing='ij')
rTens = np.sqrt(xTens**2 + yTens**2 + zTens**2)

# work out the angles
thetaTens = np.arctan2( np.sqrt(xTens**2 + yTens**2), zTens )
phiTens = np.arctan2(yTens, xTens)

# only the L.F term is nonzero
#
l_num = 5# ang mom magnitude
m_num = 3 # ang mom projection number
#
L_dot_F_tens = l_num*(l_num + 1)*(rTens**(-(l_num+1)))*sps.sph_harm(m_num, l_num, thetaTens, phiTens)
#L_dot_F_tens = l_num*(l_num + 1)*(rTens**l_num)*sps.sph_harm(m_num, l_num, thetaTens, phiTens)

# sort by radius
i_rad_desc = np.argsort(rTens.flatten())
# drop the lowest 3%
i_cut_off = 1#np.ceil(np.prod(rTens.shape)*0.03).astype(int)
#
xCleanVec = xTens.flatten()[i_rad_desc[i_cut_off:]]
yCleanVec = yTens.flatten()[i_rad_desc[i_cut_off:]]
zCleanVec = zTens.flatten()[i_rad_desc[i_cut_off:]]
L_dot_F_CleanVec = L_dot_F_tens.flatten()[i_rad_desc[i_cut_off:]]


toroHarmVec = ToroHarmVecRep(xCleanVec, yCleanVec, zCleanVec,
                             raw_divFTens = 0*xCleanVec, raw_rDotFTens=0*xCleanVec,
                             raw_LDotFTens=L_dot_F_CleanVec, nCount=20, mCount=35)

### what is the difference between the first and second kind harmonics for the specified range
ratioTens = np.abs(np.squeeze(toroHarmVec.psiTens[0,:,:,:]/toroHarmVec.psiTens[1,:,:,:]))
harmRatioMean = np.mean(ratioTens, axis=2)*np.squeeze(np.abs(toroHarmVec.bCoeff_Tens[0,:,:]/toroHarmVec.bCoeff_Tens[1,:,:]))
harmRatioMedian = np.median(ratioTens, axis=2)

############# now plot

#
bCoeff_max_val_first=10*np.log10(np.max( np.abs(toroHarmVec.bCoeff_Tens[0,:,:]) ))
bCoeff_max_val_second=10*np.log10(np.max( np.abs(toroHarmVec.bCoeff_Tens[1,:,:]) ))
bCoeff_col_abs_norm_first = mpc.Normalize(vmin=bCoeff_max_val_first-60, vmax=bCoeff_max_val_first)
bCoeff_col_abs_norm_second = mpc.Normalize(vmin=bCoeff_max_val_second-60, vmax=bCoeff_max_val_second)
#
#
col_pha_norm = mpc.Normalize(vmin=-180, vmax=180)

pl.style.use('dark_background')

tickStep = 5#dB

pl.figure(1)
###
pl.subplot(221)
pl.imshow(10*np.log10( np.abs(np.squeeze( toroHarmVec.bCoeff_Tens[0,:,:] )) ), norm=bCoeff_col_abs_norm_first, cmap=pl.cm.hot)
pl.colorbar().set_label(' (dB)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, toroHarmVec.bCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, toroHarmVec.bCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|b^{(1)}\\right|$')
###
pl.subplot(222)
pl.imshow(10*np.log10( np.abs(np.squeeze( toroHarmVec.bCoeff_Tens[1,:,:] )) ), norm=bCoeff_col_abs_norm_second, cmap=pl.cm.hot)
pl.colorbar().set_label(' (dB)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, toroHarmVec.bCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, toroHarmVec.bCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|b^{(2)}\\right|$')


########## phase

pl.subplot(223)
pl.imshow(np.angle(np.squeeze(toroHarmVec.bCoeff_Tens[0,:,:]), deg=True), norm=col_pha_norm, cmap=pl.cm.jet)
pl.colorbar().set_label('(deg)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, toroHarmVec.bCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, toroHarmVec.bCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|c^{(2)}\\right|$')

pl.subplot(224)
pl.imshow(np.angle(np.squeeze(toroHarmVec.bCoeff_Tens[1,:,:]), deg=True), norm=col_pha_norm, cmap=pl.cm.jet)
pl.colorbar().set_label('(deg)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, toroHarmVec.bCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, toroHarmVec.bCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|c^{(2)}\\right|$')

## now do it for normalized values
# value of the harmonic (mean) and the coefficient
first_norm = np.abs(np.squeeze(toroHarmVec.bCoeff_Tens[0,:,:]*np.median(toroHarmVec.psiTens[0,:,:,:], axis=2)))
# value of the harmonic (mean) and the coefficient
second_norm = np.abs(np.squeeze(toroHarmVec.bCoeff_Tens[1,:,:]*np.median(toroHarmVec.psiTens[1,:,:,:], axis=2)))

norm_max_val=10*np.log10(np.max( [np.max( first_norm ), np.max( second_norm )]))
norm_norm = mpc.Normalize(vmin=norm_max_val-60, vmax=norm_max_val)

pl.figure(2)
###
pl.subplot(211)
# pl.imshow(10*np.log10( first_norm ), norm=norm_norm, cmap=pl.cm.hot)
pl.imshow( first_norm, cmap=pl.cm.hot)
pl.colorbar().set_label('lin')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, toroHarmVec.cCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, toroHarmVec.cCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|b^{(1)}\\right|$')
###
pl.subplot(212)
# pl.imshow(10 * np.log10(second_norm), norm=norm_norm, cmap=pl.cm.hot)
pl.imshow( second_norm, cmap=pl.cm.hot)
pl.colorbar().set_label('lin')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, toroHarmVec.cCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, toroHarmVec.cCoeff_Tens.shape[2], tickStep))
#pl.title('$\\left|b^{(2)}\\right|$')


pl.figure(3)
#ratioTens_max_val=10*np.log10(np.max(harmRatioMean))
ratioTens_norm = mpc.Normalize(vmin=-60, vmax=20)
###
pl.imshow(10*np.log10(harmRatioMean), norm=ratioTens_norm, cmap=pl.cm.hot)
pl.colorbar().set_label(' (dB)')
pl.xlabel('m - order')
pl.ylabel('n - order')
pl.yticks(np.arange(0, toroHarmVec.bCoeff_Tens.shape[1], tickStep))
pl.xticks(np.arange(0, toroHarmVec.bCoeff_Tens.shape[2], tickStep))
pl.title('$P/Q$')

pl.show()