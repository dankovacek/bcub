import os, glob

from multiprocessing import Pool

import numpy as np
import pandas as pd
import geopandas as gpd 

from shapely.geometry import Polygon, Point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEM_DIR = os.path.join(BASE_DIR, 'source_data/dem_data')
# earth_env_folder = ''
EENV_DIR = os.path.join(DEM_DIR, 'EarthEnv_DEM90')

# ensure the folders exist
for p in [DEM_DIR, EENV_DIR]:
    if not os.path.exists(p):
        os.mkdir(p)

HYSETS_DIR = os.path.join(BASE_DIR, 'source_data/HYSETS_data/')

hysets_df = pd.read_csv(os.path.join(HYSETS_DIR, 'HYSETS_watershed_properties.txt'), sep=';')
hysets_locs = [Point(x, y) for x, y in zip(hysets_df['Centroid_Lon_deg_E'].values, hysets_df['Centroid_Lat_deg_N'])]
hysets_df = gpd.GeoDataFrame(hysets_df, geometry=hysets_locs, crs='EPSG:4326')
(minx, miny, maxx, maxy) = hysets_df.geometry.total_bounds

vrt_out_path = os.path.join(BASE_DIR, 'processed_data/processed_dem')

# specify the total bounds of North America and Mexico
minx, maxx = -180, -50
miny, maxy = 10, 85

# create an array to define each decimal degree within the total bounds
tiles = []
for x in range(minx, maxx, 5):
    for y in range(miny, maxy, 5):
        corners = [
            (x, y), 
            (x + 5, y),
            (x + 5, y + 5), 
            (x, y + 5)]
        tiles += [Polygon(corners)]

# convert to geodataframe
tiles_to_check = gpd.GeoDataFrame(geometry=tiles, crs='EPSG:4326')
# for each decimal degree pair, find out if it falls within 
# the land mass of NA + MEX

# The EarthEnv DEM90 tiles are provided in an interactive web application form  Instead of clicking on all the relevant map tiles, we can instead load the HYSETS station list and find the set of coordinate pairs (in degrees) representing the land mass of North America and Mexico.
# Shapefile of North America and Mexico land mass
# https://maps.princeton.edu/catalog/stanford-ns372xw1938
na_bound_file = os.path.join(BASE_DIR, 'source_data/misc/stanford-ns372xw1938-geojson.json')
   

polygon_df = gpd.read_file(na_bound_file)

bounds_df = polygon_df[polygon_df['country'].isin(['CAN', 'MEX', 'USA'])]
state_excludes = ['US-HI']
bounds_df = bounds_df[~bounds_df['stateabb'].isin(state_excludes)]

# Exclude Puerto Rico because it is not in HYSETS
name_excludes = ['Puerto Rico', 'Navassa Island', 'United States Virgin Islands']
bounds_df = bounds_df[~bounds_df['name'].isin(name_excludes)]

# get the extents of the polygon describing Alaska
ak_bounds = bounds_df[bounds_df['stateabb'] == 'US-AK'].copy()
for i, row in ak_bounds.iterrows():
    _, _, max_y, _ = row['geometry'].bounds
    if max_y > 0:
        ak_bounds.loc[i, 'max_x'] = True
    else:
        ak_bounds.loc[i, 'max_x'] = False
    

bounds_df = bounds_df.dissolve()

bounds_fpath = os.path.join(BASE_DIR, 'source_data/misc/CANUSAMEX_bounds.geojson')

bounds_df.to_file(bounds_fpath)


def custom_round(x, base=5):
    "round to the nearest base"
    return base * round(x/base)

overlapping_tiles_df = gpd.sjoin(tiles_to_check, bounds_df, predicate='intersects')

overlapping_tiles_df = overlapping_tiles_df[['geometry', 'name', 'country', 'stateabb']]


coord_pairs = []
for _, g in overlapping_tiles_df.iterrows():
    pts = list(g.geometry.exterior.coords)
    pts = [(int(e[0]), int(e[1])) for e in pts]
    coord_pairs += pts

coord_pairs = list(set(coord_pairs))

# match formatting order of EarthEnv DEM file naming convention, i.e.:
# http://mirrors.iplantcollaborative.org/earthenv_dem_data/EarthEnv-DEM90/EarthEnv-DEM90_N55W125.tar.gz

# the download url format is the following:
# http://mirrors.iplantcollaborative.org/earthenv_dem_data/EarthEnv-DEM90/EarthEnv-DEM90_N55W110.tar.gz

formatted_coord_strings = [(f'N{p[1]:02d}W{abs(p[0]):03d}') for p in coord_pairs]

# add the part of AK that goes past the Int'l Date Line
east_coords = [(170, 50), (175, 50), (170, 55), (175, 55)]
formatted_coord_strings += [(f'N{p[1]:02d}E{p[0]:03d}') for p in east_coords]

# format filenames to compile the list of urls
# remove any that already exist
file_list = [f'EarthEnv-DEM90_{s}.tar.gz' for s in formatted_coord_strings]

existing_file_coord_strings = [p.split('_')[-1].split('.')[0] for p in os.listdir(EENV_DIR)]

file_list = [f for f in file_list if f.split('_')[-1].split('.')[0] not in existing_file_coord_strings]

def download_file(filename):
    url = f'http://mirrors.iplantcollaborative.org/earthenv_dem_data/EarthEnv-DEM90/{filename}'

    command = f'wget {url} -P {EENV_DIR}'
    save_path = f'{EENV_DIR}/{filename}'

    if not os.path.exists(save_path):
        os.system(command)
    
        folder_name = filename.split('.')[0]
        os.system(f'tar -xf {EENV_DIR}/{filename} -C {EENV_DIR}')

# with Pool() as p:
#     p.map(download_file, file_list)
    
# this command builds the dem mosaic "virtual raster"
mosaic_file = 'EENV_DEM_mosaic.vrt'
vrt_command = f"gdalbuildvrt -resolution highest -a_srs epsg:4326 {vrt_out_path}/{mosaic_file} {EENV_DIR}/EarthEnv-DEM90_*.bil"
os.system(vrt_command)

# remove the downloaded tar files
for f in glob.glob(f'{EENV_DIR}/*.tar.gz'):
    os.remove(f)
