import numpy as np
import xarray as xr
import pandas as pd

import xmitgcm
import ecco_v4_py as ecco
import swot_ssh_utils as swot

import sys 
from scipy.spatial import KDTree


### Load in LLC90 Grid ###

data_dir = '/scratch/08382/shoshi/MITgcm_c68p/eccov4r4_with_tides_stampede2/run_stampede2'
xcfile=data_dir+'/XC.data'
fb = open(xcfile, "rb")
nx=90
ny=90*13
xc=np.fromfile(fb,dtype='>f')


ycfile=data_dir+'/YC.data'
fb = open(ycfile, "rb")
nx=90
ny=90*13
yc=np.fromfile(fb,dtype='>f') # xc, yc in shape 90x1170 = 105,300 x 1



### Read in SWOT Data from Input Args ###

#swot_data_dir = './data/SWOT_SIMULATED_L2_KARIN_SSH_GLORYS_SCIENCE_V1/'
swot_data_dir = './data/beta_pre_validated/SWOT_L2_LR_SSH_1.1/Expert/'

num_input_files = len(sys.argv) - 1 # total number of arguments is one less than len(argv)

filename1 = sys.argv[1] # sys.argv[0] is name of python file
filename2 = sys.argv[2]

ds_swot1 = xr.open_dataset(swot_data_dir + filename1)

ds_swot2 = xr.open_dataset(swot_data_dir + filename2)


# apply corrections
ssha = ds_swot1.ssha_karin_2
flag = ds_swot1.ancillary_surface_classification_flag
ssha = np.where(flag == 0, ssha, np.nan)

distance = ds_swot1.cross_track_distance.values

ssha_1 = swot.fit_bias(
        ssha, distance,
        check_bad_point_threshold=0.1,
        remove_along_track_polynomial=True
    )

ds_swot1.ssha_karin_2.values = ssha_1


ssha = ds_swot2.ssha_karin_2
flag = ds_swot2.ancillary_surface_classification_flag
ssha = np.where(flag == 0, ssha, np.nan)

distance = ds_swot2.cross_track_distance.values

ssha_2 = swot.fit_bias(
        ssha, distance,
        check_bad_point_threshold=0.1,
        remove_along_track_polynomial=True
    )

ds_swot2.ssha_karin_2.values = ssha_2

# get timestamp from filename
starttime = filename2.split('_')[7] # second input file starttime

year = starttime[:4]
month = starttime[4:6]
day = starttime[6:8]
hour = starttime[9:11]
minute = starttime[11:13]
sec = starttime[13:]


hour_start = np.datetime64(year + '-' + month + '-' + day + 'T' + hour)
hour_end = hour_start + np.timedelta64(1,'h')


# convert to pandas dataframe
df_swot1 = ds_swot1.to_dataframe()
df_swot2 = ds_swot2.to_dataframe()


# reset index so can easily combine multiple dfs 
df_swot1.reset_index(inplace = True)
df_swot2.reset_index(inplace = True)

# select data in current hour
df_swot1 = df_swot1[np.logical_and(hour_start <= df_swot1.time, df_swot1.time <= hour_end)]
df_swot2 = df_swot2[np.logical_and(hour_start <= df_swot2.time, df_swot2.time <= hour_end)]


# combine data from both files to fill the hour
df_swot = pd.concat([df_swot1, df_swot2])

# if there are three input files
if num_input_files == 3:
    filename3 = sys.argv[3]
    ds_swot3 = xr.open_dataset(swot_data_dir + filename3)
    
    ssha = ds_swot3.ssha_karin_2
    flag = ds_swot3.ancillary_surface_classification_flag
    ssha = np.where(flag == 0, ssha, np.nan)

    distance = ds_swot3.cross_track_distance.values

    ssha_3 = swot.fit_bias(
            ssha, distance,
            check_bad_point_threshold=0.1,
            remove_along_track_polynomial=True
        )

    ds_swot3.ssha_karin_2.values = ssha_3
    
    df_swot3 = ds_swot3.to_dataframe()
    df_swot3.reset_index(inplace = True)
    df_swot3 = df_swot3[np.logical_and(hour_start <= df_swot3.time, df_swot3.time <= hour_end)]
    df_swot = pd.concat([df_swot, df_swot3])


# convert longitude from [0,360] to [-180, 180] to match convention in LLC90
# https://stackoverflow.com/questions/53345442/about-changing-longitude-array-from-0-360-to-180-to-180-with-python-xarray
df_swot['longitude'] = (df_swot['longitude'] + 180) % 360 - 180


### Regridding with KD Tree ###

# For each swot point, find nearest llc point
llc_lats = yc
llc_lons = xc

llc_coords = np.c_[llc_lats, llc_lons]
swot_coords = np.c_[df_swot['latitude'], df_swot['longitude']]

kd_tree = KDTree(llc_coords)
distance, nearest_swot_index_in_llc = kd_tree.query(swot_coords, k=1)


# add flag to swot df
df_swot.insert(4, "llc_coords_index", nearest_swot_index_in_llc.astype('int'), True)


# groupby llc (lat, lon) index to get average (and stddev) swot ssh
gb = df_swot.groupby(['llc_coords_index'])
counts = gb.size().to_frame(name='counts')
gb_stats = (counts
.join(gb.agg({'ssha_karin_2': 'mean'}).rename(columns={'ssha_karin_2': 'ssha_karin_2_mean'}))
.join(gb.agg({'ssha_karin_2': 'std'}).rename(columns={'ssha_karin_2': 'ssha_karin_2_std'}))
.reset_index()
)

# convert lats_flag, lons_flag from indices to actual lat and lon values
indices = gb_stats.llc_coords_index.values

write_llc_lats = llc_lats[indices]
write_llc_lons = llc_lons[indices]

# insert lats and lons into stats df
gb_stats.insert(2, "latitude", write_llc_lats, True)
gb_stats.insert(3, "longitude", write_llc_lons, True)



### Merge with Entire Domain ###

# create global llc90 df
global_df = pd.DataFrame({'latitude': yc.astype('f8'), 'longitude': xc.astype('f8')})

# merge with df over swot swath
gb_stats['latitude'] = gb_stats['latitude'].astype('f8')
gb_stats['longitude'] = gb_stats['longitude'].astype('f8')
combined_df = global_df.merge(gb_stats, how = 'left', on = ['latitude', 'longitude'])




### Finally, Write to Output ###

# change nans to -9999
combined_df.loc[combined_df['counts'].isna(), 'counts'] = -9999
combined_df.loc[combined_df['ssha_karin_2_mean'].isna(), 'ssha_karin_2_mean'] = -9999
combined_df.loc[combined_df['ssha_karin_2_std'].isna(), 'ssha_karin_2_std'] = -9999


# insert timestamp
combined_df.insert(0, "DateTime", pd.Timestamp(year=int(year), month=int(month), day=int(day), 
                                               hour=int(hour)), True)

# write in compact binary format
out_dir = './regridded_data_1day_repeat/'

fname = 'swot' + year + month + day + hour + '.bin'
xmitgcm.utils.write_to_binary(combined_df['ssha_karin_2_mean'].values, out_dir + fname)





