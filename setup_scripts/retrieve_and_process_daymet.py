# generate basins
import os
import time

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import geopandas as gpd
import multiprocessing as mp
import xarray as xr


import basin_processing_functions as bpf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_folder = os.path.join(BASE_DIR, 'processed_data/processed_dem/')
region_files = os.listdir(DEM_folder)
region_codes = sorted(list(set([e.split('_')[0] for e in region_files])))

#########################
# input file paths
#########################
daymet_tile_dir = os.path.join(BASE_DIR, 'input_data/DAYMET/')
daymet_output_dir = os.path.join(BASE_DIR, 'processed_data/DAYMET/')

# i'm using a temp folder on an external SSD because the dataset is huge
daymet_tile_dir = '/media/danbot/Samsung_T51/large_sample_hydrology/common_data/DAYMET'
daymet_tile_dir = '/media/danbot/2023_1TB_T7'
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
        ds = ds.resample(time='1y').sum(keep_attrs=True, skipna=False)
    elif param == 'swe':
        # HYSETS uses average annual maximum
        ds = ds.resample(time='1y').max(keep_attrs=True)
    else:
        ds = ds.resample(time='1y').mean(keep_attrs=True)

    annual_mean = ds.mean('time', keep_attrs=True)
    annual_mean.rio.write_crs(daymet_proj)
    annual_mean.rio.write_nodata(np.nan, inplace=True)
    annual_mean.rio.to_raster(output_fpath)
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
    annual_mean = ds.mean('time', keep_attrs=True, skipna=False)
    annual_mean.rio.write_crs(daymet_proj)
    annual_mean.rio.write_nodata(np.nan, inplace=True)
    annual_mean.rio.to_raster(output_fpath)
    return True
    

def consecutive_run_lengths(arr):
    # Find the change points
    # If any NaN values are present in the input array, return NaN
    if np.isnan(arr).all():
        return np.nan
    
    change = np.concatenate(([0], np.where(arr[:-1] != arr[1:])[0] + 1, [len(arr)]))
    # Calculate lengths and filter by True values
    lengths = np.diff(change)
    true_lengths = lengths[arr[change[:-1]]]
    
    max_run = np.max(true_lengths) if true_lengths.size > 0 else 0
    
    del arr
    del lengths

    return max_run


def compute_dry_duration(ds, output_fpath, threshold=1.0): 
    
    # 1. Convert to a boolean array where True indicates values below the threshold
    nan_locations = ds.isnull().all(dim='time')
    below_threshold = (ds < threshold)
    below_threshold.rio.write_nodata(np.nan, inplace=True)
    below_threshold = below_threshold.rio.write_crs(daymet_proj)
    print('    computing longest runs...')
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
    
    mean_longest_run = longest_runs.mean('year', skipna=False, keep_attrs=True).round(0)
    
    mean_longest_run.rio.write_crs(daymet_proj)
    mean_longest_run.rio.write_nodata(np.nan, inplace=True)
    mean_longest_run.rio.to_raster(output_fpath)
    return True
    

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
    print(f'processing {param}')
    
    daymet_param_folder = os.path.join(daymet_tile_dir, param)
    daymet_temp_folder = os.path.join(daymet_output_dir, f'temp')
    for f in [daymet_param_folder, daymet_temp_folder]:
        if not os.path.exists(f):
            os.makedirs(f)
    
    
    download_daymet_tiles(param, years)
    
        
    for tid in tile_ids:
        output_fpath = os.path.join(daymet_temp_folder, f'{tid}_{param}_mean_annual.tiff')
        dry_days_output_fpath = os.path.join(daymet_temp_folder, f'{tid}_{param}_dry_days_mean_annual.tiff')
        dry_duration_output_fpath = os.path.join(daymet_temp_folder, f'{tid}_{param}_dry_duration_mean_annual.tiff')
        
        if os.path.exists(dry_days_output_fpath) & os.path.exists(dry_duration_output_fpath):
            print(f'tile id {tid} already processed.')
            continue
        
        # os.path.exists(output_fpath) & 
        
        data = retrieve_tiles_by_id(param, tid)
        
        # if not os.path.exists(output_fpath):
        #     try:
        #         resample_annual(data.copy(), tid, output_fpath)
        #     except Exception as ex:
        #         print(f'Resampling failed on {param} tile id: {tid}')
        #         print(ex)
        #         print('')
        #         # continue            

        if param == 'prcp':
            if not os.path.exists(dry_days_output_fpath):
                # compute the P(dry_days)
                print(f'   Computing P(dry days) on {tid}')
                compute_low_precip_frequency(data, dry_days_output_fpath)
                # pass
            if not os.path.exists(dry_duration_output_fpath):
                print(f'   Computing max dry duration on {tid}')
                # compute the longest dry period of each year and compute the mean annual
                compute_dry_duration(data, dry_duration_output_fpath)
        del data
            
    dry_days_mosaic_fpath = os.path.join(daymet_output_dir, f'{param}_dry_days_mosaic_3005.tiff')
    dry_duration_mosaic_fpath = os.path.join(daymet_output_dir, f'{param}_dry_duration_mosaic_3005.tiff')
    for f in [dry_duration_mosaic_fpath, dry_days_mosaic_fpath]:
        if not os.path.exists(f):
            base_string = f.split('/')[-1]
            if 'dry_days' in base_string:
                mosaic_param = f'{param}_dry_days'
                file_pattern = f'*_{param}_dry_days_mean_annual.tiff'
            elif 'dry_duration' in base_string:
                mosaic_param = f'{param}_dry_duration'
                file_pattern = f'*_{param}_dry_duration_mean_annual.tiff'
            else:
                mosaic_param = param
                file_pattern = f'*_{param}_mean_annual.tiff'
            
            create_tile_mosaic(mosaic_param, f, file_pattern)

    if os.path.exists(output_fpath):
        # delete temp files
        pass
        # for f in os.listdir(daymet_temp_folder):
        #     os.remove(os.path.join(daymet_temp_folder, f))
    print(asfasd)
