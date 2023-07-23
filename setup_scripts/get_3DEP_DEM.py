import os, glob

from multiprocessing import Pool

import numpy as np
import pandas as pd
import geopandas as gpd 
import rioxarray as rxr

from shapely.geometry import Polygon, Point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEM_DIR = os.path.join(BASE_DIR, 'input_data/DEM')

# ensure the folders exist
for p in [DEM_DIR]:
    if not os.path.exists(p):
        os.mkdir(p)

input_data_dir = os.path.join(BASE_DIR, 'input_data/')

tile_links = pd.read_csv(os.path.join(input_data_dir, 'file_lists/USGS_3DEP_tile_links.txt'), header=None)
url_list = tile_links.values


def download_file(url):
    
    filename = url[0].split('/')[-1]
    command = f'wget {url[0]} -P {DEM_DIR}'
    save_path = f'{DEM_DIR}/{filename}'

    if not os.path.exists(save_path):
        os.system(command)
    
        # folder_name = filename.split('.')[0]
        # os.system(f'tar -xf {DEM_DIR}/{filename} -C {DEM_DIR}')
    print('')

with Pool() as p:
    p.map(download_file, url_list[:3])

dem_files = os.listdir(DEM_DIR)
test_file = [f for f in dem_files if f.endswith('.tif')][0]
raster = rxr.open_rasterio(os.path.join(DEM_DIR, test_file))
# get the crs of the first file
crs = raster.rio.crs

# vrt_output_path = os.path.join(input_data_dir, 'vrt_files')
# this command builds the dem mosaic "virtual raster"
mosaic_file = 'USGS_3DEP_DEM_mosaic.vrt'
vrt_command = f"gdalbuildvrt -resolution highest -a_srs {crs} {input_data_dir}/{mosaic_file} {DEM_DIR}/*.tif"
os.system(vrt_command)

# remove the downloaded tar files
# for f in glob.glob(f'{EENV_DIR}/*.tar.gz'):
#     os.remove(f)
