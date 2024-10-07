import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import sys
sys.path.append('/home3/sreich/MITgcm_c68w/MITgcm/utils/python/MITgcmutils')
from MITgcmutils import rdmds

sys.path.append('/home3/sreich/jupyter/jupyter_notebooks')
from read_write import *
from patchface3D import *

run_dir = '/nobackup/sreich/multigrid_test_ou/c68w/llc270/run.fwd_270.076c.saltmon.iter0/'

nz = 50
nx = 270
ny = 270*13

m_saltmon = rdmds(run_dir + 'm_saltmon.0000000000')
# m_thetamon = rdmds(run_dir + 'm_thetamon.0000000001')

#theta_data = read_float32(run_dir + 'T_monthly_woa09_highlatnearsurface_masked_llc270').reshape(12, nz, ny, nx)
salt_data = read_float32(run_dir + 'S_monthly_woa09_highlatnearsurface_masked_llc270').reshape(12, nz, ny, nx)


#theta_data[theta_data <= -1.8] = np.nan
#theta_data[theta_data >= 40] = np.nan


#misfit_theta = m_thetamon - theta_data[0,:,:,:]

#err_theta = read_float32(run_dir + 'Theta_sigma_smoothed_method_02_masked_merged_capped_llc270_areascaled.bin').reshape(nz, ny, nx)

#err_theta[err_theta == 0] = np.nan
#cost_theta = (misfit_theta / err_theta)**2


salt_data[salt_data <= 25] = np.nan
salt_data[salt_data >= 40] = np.nan

misfit_salt = (m_saltmon - salt_data[0,:,:,:])

err_salt = read_float32(run_dir + 'Salt_sigma_smoothed_method_02_masked_merged_capped_llc270_areascaled.bin').reshape(nz, ny, nx)

err_salt[err_salt == 0] = np.nan
cost_salt = (misfit_salt / err_salt)**2

print(np.nansum(cost_salt), np.count_nonzero(~np.isnan(cost_salt)))




# coarsen cost
nxlr = 90
nylr = nxlr*13

reshaped_cost_salt = cost_salt.reshape(nz, nylr, 3, nxlr, 3)

lr_cost_salt = reshaped_cost_salt.mean(axis=(2, 4))


print(np.nansum(lr_cost_salt), np.count_nonzero(~np.isnan(lr_cost_salt)))
