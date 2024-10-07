import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import xmitgcm
import sys
sys.path.append('/home3/sreich/MITgcm_c68w/MITgcm/utils/python/MITgcmutils')
from MITgcmutils import rdmds

sys.path.append('/home3/sreich/jupyter/jupyter_notebooks')
from read_write import *
from patchface3D import *

nx=90
ny=nx*13
nz=50

factor = 3 # lowres * factor = hires

nxh=nx*factor
nyh=ny*factor

iter = '0001'

dirroot='/nobackup/sreich/multigrid_test_ou/c68w/'

dirrun_lr = dirroot + 'llc90/run.v4_rls4.077d3.iter0.diva/'
dirrun_hr = dirroot + 'llc270/run.fwd_270.076c.iter0/'
#dirrun_hr = '/nobackup/sreich/llc270_c68w_runs/run_pk0000841536_1200s/'

dirrun_pup = dirroot + 'llc90/run.v4_rls4.077d3.saltmon.pack_unpack_iter1/'
#dirrun_pup = dirroot + 'llc90/run.v4_rls4.077d3_fwd_90.pack_unpack/'

xx_salt = rdmds(dirrun_pup + 'xx_salt.000000' + iter)

print(np.min(xx_salt), np.max(xx_salt))


# read in high-res and low-res grids

xc_lr = rdmds(dirrun_lr + 'XC')
yc_lr = rdmds(dirrun_lr + 'YC')

xc_hr = rdmds(dirrun_hr + 'XC')
yc_hr = rdmds(dirrun_hr + 'YC')
#yc_hr = grid_270.YC.values.reshape(nyh, nxh)

# hfacc_hr = rdmds(dirrun_hr + 'hFacC')
maskc_hr = rdmds(dirrun_hr + 'maskCtrlC')



# Replace NaNs with the mean of their neighbors
def inpaint_nans(arr):
    isnan = np.isnan(arr)
    arr[isnan] = generic_filter(arr, np.nanmean, size=3)[isnan] # average of 3x3 square of points
    return arr

def fill_border_nans(arr):
    # Fill NaNs in rows
    for i in range(arr.shape[0]):
        row_non_nans = np.where(~np.isnan(arr[i, :]))[0]
        if len(row_non_nans) > 0:
            if np.isnan(arr[i, 0]):
                arr[i, 0] = arr[i, row_non_nans[0]]
            if np.isnan(arr[i, -1]):
                arr[i, -1] = arr[i, row_non_nans[-1]]

    # Fill NaNs in columns
    for j in range(arr.shape[1]):
        col_non_nans = np.where(~np.isnan(arr[:, j]))[0]
        if len(col_non_nans) > 0:
            if np.isnan(arr[0, j]):
                arr[0, j] = arr[col_non_nans[0], j]
            if np.isnan(arr[-1, j]):
                arr[-1, j] = arr[col_non_nans[-1], j]

    return arr




def linear_interp(xx_lr):

    if len(xx_lr.shape) < 3:

        # linear interpolation
        xx_hr = griddata((xc_lr.ravel(), yc_lr.ravel()), xx_lr.ravel(), (xc_hr.ravel(), yc_hr.ravel()), method='linear')
        xx_hr = xx_hr.reshape(xc_hr.shape[0], xc_hr.shape[1])


        # set floor and ceiling to that of low-res
        lmin = np.min(xx_lr)
        lmax = np.max(xx_lr)
        xx_hr = np.clip(xx_hr, lmin, lmax)

    else:

        xx_hr = np.zeros((xx_lr.shape[0], nyh, nxh))
        for i in range(xx_lr.shape[0]):

            # linear interpolation
            tmp = griddata((xc_lr.ravel(), yc_lr.ravel()), xx_lr[i,:,:].ravel(), (xc_hr.ravel(), yc_hr.ravel()), method='linear')
            tmp = tmp.reshape(xc_hr.shape[0], xc_hr.shape[1])

            # handle NaNs
            # SKIP FOR NOW (TAKES TOO LONG)
            #tmp = inpaint_nans(tmp)

            # set floor and ceiling to that of low-res
            lmin = np.min(xx_lr[i,:,:])
            lmax = np.max(xx_lr[i,:,:])
            tmp = np.clip(tmp, lmin, lmax)

            # SKIP FOR NOW (TAKES TOO LONG)
            #tmp = fill_border_nans(tmp)

            xx_hr[i,:,:] = tmp

    return xx_hr


xx_salt_hr = linear_interp(xx_salt)#*maskc_hr
xx_salt_hr[np.isnan(xx_salt_hr)] = 0

write_float32(dirrun_pup + 'xx_hires/xx_salt.000000' + iter + '.data', xx_salt_hr)
