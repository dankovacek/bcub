# derive flow accumulation network from DEM
import os

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import rioxarray as rxr

from whitebox.whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()
# change to True for verbose output
wbt.verbose = False

DEM_source = 'USGS_3DEP'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'input_data/')

DEM_DIR = os.path.join(BASE_DIR, 'processed_data/processed_dem/')

processed_files = os.listdir(DEM_DIR)

region_files = os.listdir(DEM_DIR)
region_codes = sorted([e.split('_')[0] for e in region_files])

def retrieve_raster(fpath):
    rds = rxr.open_rasterio(fpath, masked=True, mask_and_scale=True)
    crs = rds.rio.crs.to_epsg()
    affine = rds.rio.transform(recalc=False)
    return rds, crs, affine

test_dem, crs, affine = retrieve_raster(os.path.join(DEM_DIR, processed_files[0]))


for region in sorted(list(set(region_codes))):
    print(f'    processing flow direction and accumulation for {region}')
    dem_file = f'{region}_{DEM_source}_{crs}.tif'
    dem_path = os.path.join(DEM_DIR, dem_file)
    
    filled_dem_file = f'{region}_{DEM_source}_{crs}_dem_filled.tif'
    out_d8_file = f'{region}_{DEM_source}_{crs}_fdir.tif'
    out_accum_file = f'{region}_{DEM_source}_{crs}_accum.tif'
    out_stream_file = f'{region}_{DEM_source}_{crs}_stream.tif'
    stream_link_file = f'{region}_{DEM_source}_{crs}_link.tif'

    filled_dem_path = os.path.join(DEM_DIR, filled_dem_file)
    d8_path = os.path.join(DEM_DIR, out_d8_file)
    accum_path = os.path.join(DEM_DIR, out_accum_file)
    stream_path = os.path.join(DEM_DIR, out_stream_file)
    stream_link_path = os.path.join(DEM_DIR, stream_link_file)
    
    if not os.path.exists(filled_dem_path):
        wbt.fill_depressions(
            dem_path,
            filled_dem_path,
        )

    if not os.path.exists(d8_path):
        wbt.d8_pointer(
            filled_dem_path,
            d8_path, 
        )
    if not os.path.exists(accum_path):
        wbt.d8_flow_accumulation(
            filled_dem_path, 
            accum_path,
            pntr=False,
        )

    # If you want to check the maximum flow accumulation value
    # to test if there is precision overflow, look to see if 
    # a) the maximum accumulation value is less than expected and 
    # b) the maximum accumulation value is near a power of 2 (i.e. 2^15)

    acc, _, _ = retrieve_raster(os.path.join(DEM_DIR, out_accum_file))
    # print(acc)
    # print(np.nanmax(acc))


    # UGS 3DEP is 1 arcsecond resolution
    # so the projected resolution varies with latitude.
    resolution = acc.rio.resolution()
    dx, dy = abs(resolution[0]), abs(resolution[1])
    
    # determine the threshold number of cells 
    # corresponding to the minimum drainage area
    minimum_basin_size = 5 # km^2
    threshold = int(minimum_basin_size * 1E6 / (dx * dy))
    
    if not os.path.exists(stream_path):
        wbt.extract_streams(
            accum_path, 
            stream_path, 
            threshold, 
            zero_background=False, 
        )
        
    if not os.path.exists(stream_link_path):
        wbt.stream_link_identifier(
            d8_path, 
            stream_path, 
            stream_link_path, 
            esri_pntr=False, 
            zero_background=False, 
        )