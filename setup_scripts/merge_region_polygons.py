import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd

from shapely import Polygon


def close_holes(poly: Polygon) -> Polygon:
        """
        Close polygon holes by limitation to the exterior ring.
        Args:
            poly: Input shapely Polygon
        Example:
            df.geometry.apply(lambda p: close_holes(p))
            
            See solution: https://stackoverflow.com/a/61466689
        """
        if poly.interiors:
            return Polygon(list(poly.exterior.coords))
        else:
            return poly


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'input_data/HYDRO_basins/')
id_list_fpath = os.path.join(DATA_DIR, 'HYDRO_basin_ids.csv')
output_folder = os.path.join(BASE_DIR, 'input_data/region_polygons')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

id_list = pd.read_csv(id_list_fpath)

region_codes = sorted(list(set(id_list['region_code'].values)))

def open_file_and_retrieve_ids(file, ids):
    gdf = gpd.read_file(os.path.join(DATA_DIR, file))
    return gdf[gdf['HYBAS_ID'].isin(ids)]

for rc in region_codes:
    region_info = id_list[id_list['region_code'] == rc].copy()
    region_files = list(set(region_info['file'].values))
    region_polygons = []
    print(f'Processing {rc}')
    for rf in region_files:
        foo = region_info[region_info['file'] == rf].copy()
        ids = foo['HYBAS_ID'].values
        polygons = open_file_and_retrieve_ids(rf, ids)
        region_polygons.append(polygons)
    region_gdf = gpd.GeoDataFrame(pd.concat(region_polygons))
    region_gdf = region_gdf.dissolve()
    region_gdf = region_gdf.explode(index_parts=False)
    region_gdf['geometry'] = region_gdf.geometry.apply(lambda p: close_holes(p))
    region_gdf = region_gdf.dissolve()
    
    
    # save the output
    region_fname = f'{rc}_{region_gdf.crs.to_epsg()}.geojson'
    region_fpath = os.path.join(output_folder, region_fname)
    region_gdf.to_file(region_fpath)
    
