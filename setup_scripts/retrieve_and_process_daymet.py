# generate basins
import os
import time

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import geopandas as gpd
import multiprocessing as mp
import xarray as xr
import rioxarray as rxr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_folder = os.path.join(BASE_DIR, 'processed_data/processed_dem/')
region_files = os.listdir(DEM_folder)
region_codes = sorted(list(set([e.split('_')[0] for e in region_files])))

#########################
# input file paths
#########################
daymet_tile_dir = os.path.join(BASE_DIR, 'input_data/DAYMET/')
daymet_output_dir = os.path.join(BASE_DIR, 'processed_data/DAYMET/')
daymet_temp_folder = os.path.join(daymet_output_dir, f'temp')

# i'm using a temp folder on an external SSD because the dataset is huge
daymet_tile_dir = '/media/danbot2/2023_1TB_T7'
daymet_proj = '+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'

daymet_tile_index_path = os.path.join(BASE_DIR, 'input_data/DAYMET/Daymet_v4_Tiles.geojson')

tiles_wkt = gpd.read_file(daymet_tile_index_path).crs.to_wkt()

# masks used to clip the geospatial layers
mask_path = os.path.join(BASE_DIR, 'input_data/region_bounds/BC_study_region_polygon_4326.geojson')
reproj_bounds_path_4326 = os.path.join(BASE_DIR, 'input_data/region_bounds/convex_hull_4326.shp')
reproj_bounds_path_4269 = os.path.join(BASE_DIR, 'input_data/region_bounds/convex_hull_4269.shp')
reproj_bounds_daymet = os.path.join(BASE_DIR, 'input_data/region_bounds/region_polygon_daymet_crs.shp')

if not os.path.exists(reproj_bounds_path_4269):
    mask = gpd.read_file(mask_path)
    mask = mask.to_crs('EPSG:4269')
    mask.geometry = mask.convex_hull  
    mask.to_file(reproj_bounds_path_4269)
if not os.path.exists(reproj_bounds_path_4326):
    mask = gpd.read_file(mask_path)
    mask = mask.to_crs('EPSG:4326') 
    mask.geometry = mask.convex_hull
    mask.to_file(reproj_bounds_path_4326)
# reproject the region bounds to the daymet projection
if not os.path.exists(reproj_bounds_daymet):
    daymet_mask = gpd.read_file(mask_path).to_crs(daymet_proj)
    daymet_mask_reproj = daymet_mask.to_crs(daymet_proj)
    daymet_mask_reproj.geometry = daymet_mask_reproj.convex_hull
    daymet_mask_reproj.to_file(reproj_bounds_daymet)


## Daymet data retrieval and processing
def get_covering_daymet_tile_ids(region_polygon):
    dm_tiles = gpd.read_file(daymet_tile_index_path)
    # import the region polygon and reproject it to the daymet projection
    region_polygon = region_polygon.to_crs(dm_tiles.crs)

    # get the intersection with the region polygon
    tiles_df = dm_tiles.sjoin(region_polygon)
    tiles_df = tiles_df.sort_values(by=['Latitude (Min)', 'Longitude (Min)'])
    tile_ids = sorted(tiles_df['TileID'].values)
    print(f'   ...There are {len(tile_ids)} tiles covering the region.')
    return tile_ids


def download_daymet_tiles(param, years):    
    print(f'Downloading DAYMET {param} data.')
    daymet_url_base = 'https://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/2129/tiles/'
    base_command = f'wget -q --show-progress --progress=bar:force --limit-rate=3m {daymet_url_base}'
    batch_commands = []
    for yr in years:
        for tile in tile_ids:
            file = os.path.join(daymet_tile_dir, f'{param}/{tile}_{yr}_{param}.nc')
            
            if (not os.path.exists(file)) & (not os.path.exists(file.replace('.nc', '_monthly.nc'))):                
                cmd = base_command + f'{yr}/{tile}_{yr}/{param}.nc -O {file}'
                batch_commands.append(cmd)

    # download the files in parallel
    print(f'   ...{len(batch_commands)} daymet {param} files remaining to download.')
    with mp.Pool() as pl:
        pl.map(os.system, batch_commands)
        
    
def _open_dataset(f, grp=None):
    """Open a dataset using ``xarray``
    """
    with xr.open_dataset(f, engine="netcdf4", chunks={},
                         mask_and_scale=True, cache=False) as ds:
        return ds.load()


def resample_annual(param, tid, output_fpath):
    """
    Adapted from pydaymet get_bygeom()
    """
    param_folder = os.path.join(daymet_tile_dir, param)
    clm_files = sorted([os.path.join(param_folder, e) for e in os.listdir(param_folder) if (e.startswith(tid) & ~e.endswith('.xml'))])   
    if len(clm_files) == 0:
        print(f'No files found for {param} tile id: {tid}')
        return False
    
    ds = xr.concat((_open_dataset(f) for f in clm_files), dim='time')[param]    
    #  write crs BEFORE AND AFTER resampling!
    ds.rio.write_nodata(np.nan, inplace=True)
    ds = ds.rio.write_crs(daymet_proj)
    
    if param in ['prcp']:
        # the mean daily precip will be computed below
        pass        
    elif param == 'swe':
        # HYSETS uses average annual maximum
        ds = ds.resample(time='1y').max(keep_attrs=True)
    else:
        ds = ds.resample(time='1y').mean(keep_attrs=True)

    param_mean = ds.mean('time', keep_attrs=True)
    param_mean.rio.write_crs(daymet_proj)
    param_mean.rio.write_nodata(np.nan, inplace=True)
    param_mean.rio.to_raster(output_fpath)
    del ds
    return True


def retrieve_tiles_by_id(param, tid):
    t0 = time.time()
    param_folder = os.path.join(daymet_tile_dir, param)
    clm_files = sorted([os.path.join(param_folder, e) for e in os.listdir(param_folder) if (e.startswith(tid) & ~e.endswith('.xml'))])   
    if len(clm_files) == 0:
        print(f'No files found for {param} tile id: {tid}')
        return False
    ds = xr.concat((_open_dataset(f) for f in clm_files), dim='time')[param] 
    ds.rio.write_nodata(np.nan, inplace=True)
    ds = ds.rio.write_crs(daymet_proj)
    t1 = time.time()
    print(f'loaded tile {tid} set in {t1-t0:.1f}s')
    return ds
    

def create_tile_mosaic(param, output_fpath, file_pattern):
    daymet_temp_dir = os.path.join(daymet_output_dir, 'temp')
    print(f'   ...processing {param} tile mosaic for final output file.')
    vrt_fname = f'{param}_mosaic.vrt'
    vrt_fpath = os.path.join(daymet_temp_dir, vrt_fname)

    # warp and save the file path
    # assemble the mosaic
    cmd = f'gdalbuildvrt {vrt_fpath} {daymet_temp_dir}/{file_pattern}'

    warp_cmd = f'gdalwarp -multi -cutline {reproj_bounds_daymet} -crop_to_cutline -wo CUTLINE_ALL_TOUCHED=TRUE -t_srs EPSG:3005 -co TILED=YES -co BIGTIFF=YES -co COMPRESS=DEFLATE -co NUM_THREADS=ALL_CPUS {vrt_fpath} {output_fpath}'
    if not os.path.exists(vrt_fpath):
        os.system(cmd)
        os.system(warp_cmd)
        os.remove(vrt_fpath)


region_polygon = gpd.read_file(mask_path)
tile_ids = get_covering_daymet_tile_ids(region_polygon)

# completed 
# ['srad', 'swe', 'tmax', 'tmin', 'vp']
daymet_params = ['prcp']

years = list(range(1980, 2023))

all_data = []
for param in daymet_params:
    mosaic_output_fpath = os.path.join(daymet_output_dir, f'{param}_mosaic_3005.tiff')
    if os.path.exists(mosaic_output_fpath):
        print(f'{param} mosaic file already processed.')
        continue
    
    daymet_param_folder = os.path.join(daymet_tile_dir, param)
    for f in [daymet_param_folder, daymet_temp_folder]:
        if not os.path.exists(f):
            os.makedirs(f)
    
    # download_daymet_tiles(param, years)   
    
    for tid in tile_ids:
        output_fpath = os.path.join(daymet_temp_folder, f'{tid}_{param}_mean_annual.tiff')        
        if not os.path.exists(output_fpath):
            data = retrieve_tiles_by_id(param, tid)
            try:
                # compute mean annual values for each tile and save to a tiff
                resample_annual(param, tid, output_fpath)
            except Exception as ex:
                print(f'Resampling failed on {param} tile id: {tid}')
                print(ex)
                print('')
                # continue
        
    
    file_pattern = f'*_{param}_mean_annual.tiff'
    create_tile_mosaic(param, mosaic_output_fpath, file_pattern)
    
    # delete temp files
    # for f in os.listdir(daymet_temp_folder):
    #     os.remove(os.path.join(daymet_temp_folder, f))

#####
#
#  To extend the DAYMET dataset to add computed parameters, 
#  follow the examples below
#
#####
def compute_low_precip_frequency(ds, output_fpath, threshold=1.0):
    """
    Frequency of low precipitation days:
        -where precipitation < 1mm/day, or
    """
    #  write crs BEFORE AND AFTER resampling!
    non_nan_mask = ds.notnull()
    # count the number of dry days in each year
    # a dry day is one where precip = 0
    ds = (ds < threshold).where(non_nan_mask).resample(time='1Y', keep_attrs=True).sum('time', skipna=False) / 365.0
    ds.rio.write_nodata(np.nan, inplace=True)
    ds = ds.rio.write_crs(daymet_proj)
    del non_nan_mask
    mean = ds.mean('time', keep_attrs=True, skipna=False)
    mean.rio.write_crs(daymet_proj)
    mean.rio.write_nodata(np.nan, inplace=True)
    mean.rio.to_raster(output_fpath)
    return True


def consecutive_run_lengths(ds):
    # Find the change points
    # If any NaN values are present in the input array, return NaN
    if np.isnan(ds).all():
        return np.nan
    
    change = np.concatenate(([0], np.where(ds[:-1] != ds[1:])[0] + 1, [len(ds)]))
    # Calculate lengths and filter by True values
    lengths = np.diff(change)
    true_lengths = lengths[ds[change[:-1]]]
    # max_run = np.max(true_lengths) if true_lengths.size > 0 else 0
    mean_run = np.mean(true_lengths) if true_lengths.size > 0 else 0
    
    del ds
    del lengths

    return mean_run


def compute_low_prcp_duration(ds, output_fpath, threshold=1.0):
    # 1. Convert to a boolean array where True indicates values below the threshold
    nan_locations = ds.isnull().all(dim='time')
    below_threshold = (ds < threshold)
    below_threshold.rio.write_nodata(np.nan, inplace=True)
    below_threshold = below_threshold.rio.write_crs(daymet_proj)
    longest_runs = xr.apply_ufunc(consecutive_run_lengths, below_threshold.groupby('time.year'), 
                                  input_core_dims=[['time']], 
                                  vectorize=True, 
                                  dask='parallelized', 
                                  output_dtypes=[int])
    print('    finished computing longest runs')
    
    longest_runs = longest_runs.rio.write_crs(daymet_proj)
    # 3. Calculate the longest duration for each year
    
    # 4. Calculate the mean duration over all the years
    longest_runs = longest_runs.where(~nan_locations)
    
    mean_longest_run = longest_runs.mean('year', skipna=False, keep_attrs=True).round(1)
    
    mean_longest_run.rio.write_crs(daymet_proj)
    mean_longest_run.rio.write_nodata(np.nan, inplace=True)
    mean_longest_run.rio.to_raster(output_fpath)
    return True


def compute_high_precip_frequency(ds, output_fpath, threshold=5.0):
    # load the mean annual precip raster
    non_nan_mask = ds.notnull()
    # count the number of dry days in each year
    # a dry day is one where precip = 0
    mean = ds.mean('time', keep_attrs=True, skipna=False)
    n_days = ds.time.size

    # find the frequency of days where the precip is greater than 5 x mean annual precip
    # by pixel-wise comparison, and divide by the length of the time dimension
    ds = (ds >= threshold * mean).where(non_nan_mask).sum('time', skipna=False) / n_days
    ds.rio.write_nodata(np.nan, inplace=True)
    ds = ds.rio.write_crs(daymet_proj)
    ds.rio.to_raster(output_fpath)
    del non_nan_mask
    del mean
    del ds
    return True
    

def compute_high_precip_duration(ds, output_fpath, threshold=5.0):
    # 1. Convert to a boolean array where True indicates values below the threshold
    # non_nan_mask = ds.notnull()
    mean = ds.mean('time', keep_attrs=True, skipna=False)
    
    above_threshold = (ds >= threshold * mean)
    nan_locations = ds.isnull().all(dim='time')
    del ds
    above_threshold.rio.write_nodata(np.nan, inplace=True)
    above_threshold = above_threshold.rio.write_crs(daymet_proj)
    print('    computing longest runs...')
    # 3. Calculate the duration of consecutive days >= 5x threshold
    longest_runs = xr.apply_ufunc(consecutive_run_lengths, above_threshold.groupby('time.year'), 
                                  input_core_dims=[['time']], 
                                  vectorize=True, 
                                  dask='parallelized', 
                                  output_dtypes=[int])
    print('    finished computing longest runs')
    del above_threshold

    longest_runs = longest_runs.rio.write_crs(daymet_proj)
    
    longest_runs = longest_runs.where(~nan_locations)    
    mean_longest_run = longest_runs.mean('year', skipna=False, keep_attrs=True).round(1)
    mean_longest_run.rio.write_crs(daymet_proj)
    mean_longest_run.rio.write_nodata(np.nan, inplace=True)
    mean_longest_run.rio.to_raster(output_fpath)
    return True


def set_computation_by_param(tid, param, output_fpath):
        # retrieve the precipitation data to compute statistics
        data = retrieve_tiles_by_id('prcp', tid)
        if param == 'low_prcp_freq':
            print(f'   Computing P(p<1mm) on {tid}')
            completed = compute_low_precip_frequency(data, output_fpath)
        elif param == 'low_prcp_duration':
            print(f'   Computing mean low precip duration')
            completed = compute_low_prcp_duration(data, output_fpath)
        elif param == 'high_prcp_freq':
            print(f'   Computing P(p >= 5 x mean annual)')
            completed = compute_high_precip_frequency(data, output_fpath)
        elif param == 'high_prcp_duration':
            print(f'   Computing mean high precip duration')
            completed = compute_high_precip_duration(data, output_fpath)
        else:
            print(f'No function set for processing parameter {param}')
            pass
        del data
        return completed


# compute low and high precip event frequency and duration
for param in ['high_prcp_freq','low_prcp_freq', 'high_prcp_duration', 'low_prcp_duration']:
    output_fpath = os.path.join(daymet_output_dir, f'{param}_mosaic_3005.tiff')
    if os.path.exists(output_fpath):
        print(f'{param} mosaic file already processed.')
        continue
    for tid in tile_ids:
        print(f'Processing tile id {tid}.')
        mean_annual_fpath = os.path.join(daymet_temp_folder, f'{tid}_{param}_mean_annual.tiff')
        if os.path.exists(mean_annual_fpath):
            print(f'   ...tile id {tid} already processed.')
        else:
            tiles_processed = set_computation_by_param(tid, param, mean_annual_fpath)
            print(f'   ...finished processing mean annual {param} for tile id {tid}.')
    
    file_pattern = f'*_{param}_mean_annual.tiff'
    create_tile_mosaic(param, output_fpath, file_pattern)
