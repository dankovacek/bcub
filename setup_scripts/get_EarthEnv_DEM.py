import os, glob

from multiprocessing import Pool

import numpy as np
import pandas as pd
import geopandas as gpd


from shapely.geometry import Polygon, Point

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEM_source = 'EENV_DEM'

DATA_DIR = os.path.join(BASE_DIR, 'data/')

# this is a custom path because I want my files saved to an external disk
DATA_DIR = '/media/danbot/Samsung_T5/geospatial_data/basin_generator/data/'

EENV_DEM_DIR = os.path.join(DATA_DIR, f'{DEM_source}/')

if not os.path.exists(EENV_DEM_DIR):
    os.mkdir(EENV_DEM_DIR)

HYSETS_DIR = os.path.join(DATA_DIR, 'HYSETS_data/')

hysets_df = pd.read_csv(os.path.join(HYSETS_DIR, 'HYSETS_watershed_properties.txt'), sep=';')
hysets_locs = [Point(x, y) for x, y in zip(hysets_df['Centroid_Lon_deg_E'].values, hysets_df['Centroid_Lat_deg_N'])]
hysets_df = gpd.GeoDataFrame(hysets_df, geometry=hysets_locs, crs='EPSG:4326')
(minx, miny, maxx, maxy) = hysets_df.geometry.total_bounds

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
# na_bound_file = os.path.join(BASE_DIR, 'source_data/misc/stanford-ns372xw1938-geojson.json')
   
# bounds_fpath = os.path.join(DATA_DIR, 'misc/CANUSAMEX_bounds.geojson')
bounds_fpath = os.path.join(DATA_DIR, 'misc/region_cvx_hull.geojson')

bounds_df = gpd.read_file(bounds_fpath)

def custom_round(x, base=5):
    "round to the nearest base"
    return base * round(x/base)

# find the tiles that intersect with the North America / Mexico bounds
# created in the previous step
overlapping_tiles_df = gpd.sjoin(tiles_to_check, bounds_df, predicate='intersects')

overlapping_tiles_df = overlapping_tiles_df[['geometry']]

# get the coordinate pairs of the bounding boxes of all tiles.
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

# format filenames to compile the list of urls
# remove any that already exist
file_list = [f'EarthEnv-DEM90_{s}.tar.gz' for s in formatted_coord_strings]

existing_file_coord_strings = [p.split('_')[-1].split('.')[0] for p in os.listdir(EENV_DEM_DIR)]

file_list = [f for f in file_list if f.split('_')[-1].split('.')[0] not in existing_file_coord_strings]

def download_file(filename):
    url = f'http://mirrors.iplantcollaborative.org/earthenv_dem_data/EarthEnv-DEM90/{filename}'

    command = f'wget {url} -P {EENV_DEM_DIR}'
    save_path = f'{EENV_DEM_DIR}/{filename}'

    if not os.path.exists(save_path):
        os.system(command)
        os.system(f'tar -xf {EENV_DEM_DIR}/{filename} -C {EENV_DEM_DIR}')

with Pool() as p:
    p.map(download_file, file_list)


# remove the downloaded tar files
for f in glob.glob(f'{EENV_DEM_DIR}/*.tar.gz'):
    os.remove(f)
