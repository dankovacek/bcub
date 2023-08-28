import os, glob

from multiprocessing import Pool

import numpy as np
import pandas as pd
import geopandas as gpd 
import rioxarray as rxr

from shapely.geometry import Polygon, Point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_DIR = os.path.join(BASE_DIR, 'input_data/DEM')
DEM_DIR = '/media/danbot/Samsung_T51/geospatial_data/DEM_data/USGS_3DEP/'

# ensure the folders exist
for p in [DEM_DIR]:
    if not os.path.exists(p):
        os.mkdir(p)

input_data_dir = os.path.join(BASE_DIR, 'input_data/')

tile_links = pd.read_csv(os.path.join(input_data_dir, 'file_lists/USGS_3DEP_tile_links.txt'), header=None)
url_list = tile_links.values.flatten().tolist()

# targets = ['n50w121', 'n50w122']#, 'n48w121', 'n49w121']

# url_list = [u for u in url_list if u.split('/')[-1].split('_')[2] in targets]

existing_files = os.listdir(DEM_DIR)
url_list = [u for u in url_list if u.split('/')[-1] not in existing_files]

# download_target = '/media/danbot/Samsung_T51/geospatial_data/DEM_data/USGS_3DEP/'

def download_file(url):    
    filename = url.split('/')[-1]
    command = f'wget {url} -P {DEM_DIR}'
    save_path = f'{DEM_DIR}/{filename}'
    save_path = os.path.join(DEM_DIR, filename)

    if not os.path.exists(save_path):
        os.system(command)
    
with Pool() as p:
    p.map(download_file, url_list)

dem_files = os.listdir(DEM_DIR)
test_file = [f for f in dem_files if f.endswith('.tif')]
if not test_file:
    raise Exception; f'No .tif files found at {DEM_DIR}.'
test_fpath = os.path.join(DEM_DIR, test_file[0])
raster = rxr.open_rasterio(test_fpath)
# get the crs of the first file
crs = raster.rio.crs
print(f'Creating raster vrt in crs: {crs}')

# first, get the nodata value
no_val = f'gdalinfo {test_fpath} | grep No'
os.system(no_val)



# vrt_output_path = os.path.join(input_data_dir, 'vrt_files')
# this command builds the dem mosaic "virtual raster"
# setting the -vrtnodata 0 flag to ensure that ocean pixels are set to 0
mosaic_file = f'USGS_3DEP_DEM_mosaic_{crs.to_epsg()}.vrt'
vrt_command = f"gdalbuildvrt -resolution highest -a_srs {crs} -vrtnodata 0 {input_data_dir}/{mosaic_file} {DEM_DIR}/*.tif"
os.system(vrt_command)
print(f'Created {mosaic_file} in {input_data_dir}')
