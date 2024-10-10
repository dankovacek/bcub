
# The HydroSheds polygons do not align with 
# major basin boundaries derived from the USGS 3DEP dataset.

# To adjust the region polygons to the unique input dem, 
# we will run the following algorithm:
# For each region:
# 1. Load the region polygon and compute the area
# 2. Query the BCUB dataset for the largest delineated sub-basin
# 3. Query the second largest delineated sub-basin

import psycopg2
# import psycopg2.extras as extras

import os
# from time import time
import pandas as pd
from shapely.validation import make_valid
from shapely.geometry import Point, Polygon, MultiPolygon

import numpy as np
# import multiprocessing as mp
import rioxarray as rxr
# import xarray as xr
# from osgeo import gdal, ogr, osr

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

from whitebox.whitebox_tools import WhiteboxTools
wbt = WhiteboxTools()
wbt.verbose = True

import warnings
warnings.filterwarnings('ignore')

conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
}
schema_name = 'basins_schema'
DEM_source = 'USGS_3DEP'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'input_data/')
DEM_DIR = os.path.join(DATA_DIR, 'DEM/')
PROCESSED_DEM_DIR = os.path.join(BASE_DIR, 'processed_data/processed_dem/')
output_folder = os.path.join(DATA_DIR, 'adjusted_region_polygons')
# dem tile mosaic "virtual raster"
mosaic_path = os.path.join(BASE_DIR, f'processed_data/{DEM_source}_DEM_mosaic_4269.vrt')


region_bounds = gpd.read_file(os.path.join(DATA_DIR, 'BCUB_regions_HydroBASINS.geojson'))
region_bounds = region_bounds.to_crs(3005)
region_bounds['area'] = region_bounds.geometry.area / 1e6
region_bounds = region_bounds[['region_code', 'area', 'geometry']]

# check that all region_bounds geometries are valie
region_bounds['valid_geom'] = region_bounds.geometry.is_valid
if not region_bounds.valid_geom.all():
    print('Some geometries are invalid. Attempting to repair.')
    region_bounds['geometry'] = region_bounds.geometry.apply(lambda g: make_valid(g))
    region_bounds['valid_geom'] = region_bounds.geometry.is_valid
if not region_bounds.valid_geom.all():
    raise Exception('Some geometries are still invalid.')

def retrieve_raster(filename):
    fpath = os.path.join(DEM_DIR, filename)
    raster = rxr.open_rasterio(fpath, mask_and_scale=True)
    crs = raster.rio.crs
    affine = raster.rio.transform(recalc=False)
    return raster, crs, affine, fpath


def get_crs_and_resolution(fname):
    raster = rxr.open_rasterio(fname)
    crs = raster.rio.crs.to_epsg()
    res = raster.rio.resolution()  
    return crs, res


dem_crs, (w_res, h_res) = get_crs_and_resolution(mosaic_path)


def load_region_polygon(rc, which_mask='base'):
    folder = os.path.join(BASE_DIR, 'input_data', 'region_polygons')
    if which_mask == 'base':
        fname = f'{rc}_4269_final.geojson'
    else:
        fname = f'{rc}_4269_dem_clip_final.geojson'
    
    path = os.path.join(folder, fname)
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs('EPSG:3005')
    return gdf


def query_largest_basin(rc):
    conn = psycopg2.connect(**conn_params)
    query = f"""
    SELECT id
    FROM basins_schema.basin_attributes
    WHERE region_code = '{rc}'
    ORDER BY drainage_area_km2 DESC
    LIMIT 1;
    """
    basin_id_df = pd.read_sql_query(query, conn)
    basin_id = basin_id_df.id.values[0]

    row_query = f"""
    SELECT *
    FROM basins_schema.basin_attributes
    WHERE id = {basin_id};
    """
    gdf = gpd.read_postgis(row_query, conn, geom_col='basin')
    return gdf


def query_geoms_by_id(ids, which_geom='basin'):

    ids_placeholder = ', '.join(['%s'] * len(ids))  # Create placeholders for each ID
  
    conn = psycopg2.connect(**conn_params)
    query = f"""
    SELECT *
    FROM basins_schema.basin_attributes
    WHERE id IN ({ids_placeholder});
    """
    gdf = gpd.read_postgis(query, conn, params=ids, geom_col=which_geom)
    return gdf


def get_polygon_difference(basis, overlay):
    # find the mismatch between two polygons
    if type(basis) == dict:
        difference = basis['geometry'].difference(overlay.geometry)
    else:
        difference = basis.geometry.difference(overlay.geometry)
    df = gpd.GeoDataFrame(geometry=difference.geometry.values, crs='EPSG:3005')
    df = df.explode(index_parts=True, ignore_index=True)
    df['area'] = df.geometry.area / 1e6
    return df.sort_values(by='area', ascending=False)


def get_closest_point_to_edge(pts, polygon):
    # get the closest point to the edge of the polygon
    # for each point in the list of points
    # return the point that is closest to the edge
    pts['distance_to_boundary'] = pts.geometry.apply(lambda point: point.distance(polygon.geometry.boundary.iloc[0]))
    return pts.loc[[pts['distance_to_boundary'].idxmin()]]


def get_max_raster_acc(raster, region_mask, n_try, existing_pts=pd.DataFrame()):
    # find the max accumulation value/coords in a raster
    prev_max = []
    if existing_pts.empty:
        print(' getting max acc value.')
        max_acc = raster.max().squeeze()
    else:
        print('   getting max of unseen region')
        points_to_exclude = [(pt.x, pt.y) for pt in existing_pts.geometry]
        
        # mask coordinates of points we've alread found
        # mask = xr.DataArray(np.zeros_like(raster, dtype=bool), coords=raster.coords)
        # Mark the mask as True for coordinates to be excluded
        for x, y in points_to_exclude:
            # if {'x': x, 'y': y} in raster_mod.coords:  # Check if the coordinates exist in the raster
            if (x in raster.coords['x'].values) and (y in raster.coords['y'].values):
                raster.loc[dict(x=x, y=y)] = np.nan

        # find the max of new points
        # Compute the maximum of the raster ignoring NaN values
        max_acc = raster.max().squeeze()
        print(f'        max_acc: {max_acc:.0f}')
    
    print(f'   Round {n_try}: Max accumulation value: {max_acc:.0f}')
    # get the pour point of the largest accumulation value
    idx_max = raster.where(raster==max_acc, drop=True)

    # Stack the data to collapse all dimensions, filtering out NaN values
    stacked = idx_max.stack(z=('x', 'y')).dropna(dim='z')

    # Extract the coordinates
    coords = np.array(stacked.z)
    pts = [Point(x, y) for x, y in coords]
    pt_df = gpd.GeoDataFrame(geometry=pts, crs=acc_crs)
    pt_df['accumulation'] = max_acc.item()
    pts_fname = f'{region_code}_pour_points_max_acc_{n_try}.geojson'
    pts_path = os.path.join(BASE_DIR, 'input_data', pts_fname)
    new_pt_df = gpd.GeoDataFrame(pd.concat([existing_pts, pt_df]), crs=acc_crs)
    new_pt_df.to_file(pts_path, driver='GeoJSON')

    filtered_pts = pt_df[~pt_df['accumulation'].isin(prev_max)].copy()
    closest_pt = get_closest_point_to_edge(filtered_pts, region_mask)
    pt_fname = f'{region_code}_closest_acc_pt_{n_try}.geojson'
    pt_path = os.path.join(BASE_DIR, 'input_data', pt_fname)
    closest_pt.to_file(pt_path, driver='GeoJSON')
    return new_pt_df, closest_pt


def query_largest_basin_in_polygon(geom, existing_ppt_ids):
    """ Query the database for the pour point within the polygon
    corresponding to the largest basin."""
    polygon_wkt = geom.wkt
    params = [polygon_wkt]
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    query = """
    SELECT id
    FROM basins_schema.basin_attributes
    WHERE ST_Within(pour_pt, ST_GeomFromText(%s, 3005))
    """
    if len(existing_ppt_ids) > 0:
        placeholders = ','.join(['%s'] * len(existing_ppt_ids))  # Create a string of placeholders
        query += f"AND id NOT IN ({placeholders}) "
        params += [str(e) for e in existing_ppt_ids]
    # Add ordering and limit
    query += """
    ORDER BY drainage_area_km2 DESC LIMIT 1;
    """   
     
    cur.execute(query, params)

    # Fetch the results
    result = cur.fetchone()
    # close the cursor
    cur.close()
    if result is not None:
        basin = query_geoms_by_id(list(result), 'basin')
        ppt = query_geoms_by_id(list(result), 'pour_pt')
        return ppt, basin
    else:
        print('  No points found.')
        return pd.DataFrame(), pd.DataFrame()


region_codes = ['08A', '08B', '08C', '08D', 
                '08E', '08F', '08G', '10E',
                'HGW', 'VCI', 'WWA', 'HAY',
                'FRA', 'PCR', 'CLR', 'YKR', 
                'LRD', 'ERK']
region_codes = ['CLR']
# don't reprocess completed adjusted region polygon files (ending with R0)
remaining_regions = [e for e in region_codes if not os.path.exists(os.path.join(output_folder, f'{e}_covering_basins_R0.geojson'))]

completed = []
for rc in remaining_regions:
    raster_crs = 3005
    polygon_output_fpath = os.path.join(BASE_DIR, 'input_data', 'adjusted_region_polygons', f'{rc}_covering_basins.geojson')
    
    if os.path.exists(polygon_output_fpath):
        print(f'   {rc} covering polygon set already processed. Skipping.')
        completed.append(rc)
        completed.append(polygon_output_fpath)
        continue
    else:
        print(f'   Processing {rc} initial covering polygon set from HydroBASINS polygons.')

    fdir_fname = f'{rc}_{DEM_source}_3005_fdir.tif'
    fdir_path = os.path.join(PROCESSED_DEM_DIR, fdir_fname)
    buffered_basins_fname = f'{rc}_buffered_basins.tif'
    basins_output = os.path.join(PROCESSED_DEM_DIR, buffered_basins_fname)
    if not os.path.exists(basins_output):
        print(f'   Processing basins for {rc}.')
        wbt.basins(
            fdir_path,
            basins_output,
            esri_pntr=False,
            callback=None,
        )

    basin_polygon_fname = f'{rc}_buffered_basins.shp'
    polygon_path = os.path.join(output_folder, 'temp', basin_polygon_fname)
    if not os.path.exists(polygon_path):
        wbt.raster_to_vector_polygons(
            basins_output,
            polygon_path,
        )

    gdf = gpd.read_file(polygon_path, crs='EPSG:3005')
    gdf['area'] = gdf.geometry.area / 1e6
    gdf = gdf[gdf['area'] > 1.0]
    covering_basins_fname = f'{rc}_covering_basins.geojson'
    covering_basins_path = os.path.join(output_folder, covering_basins_fname)

    gdf.to_file(covering_basins_path, driver='GeoJSON')
    if os.path.exists(covering_basins_path):
        temp_files = os.listdir(os.path.join(output_folder, 'temp'))
        # for f in temp_files:
        #     os.remove(os.path.join(output_folder, 'temp', f))
    
    print(f'Adjusting {rc} covering set -- removing edge basins contained in neighbouring region polygons.')
    output_fname = f'{rc}_covering_basins_R0.geojson'
    adjusted_basins_path = os.path.join(output_folder, output_fname)
    if os.path.exists(adjusted_basins_path):
        print(f'   {rc} already processed. Skipping.')
        continue
    
    covering_basins_fname = f'{rc}_covering_basins.geojson'
    covering_basins_path = os.path.join(output_folder, covering_basins_fname)
    df = gpd.read_file(covering_basins_path)

    _ = df.sindex
    assert df.crs == region_bounds.crs

    # simplify the geometry by a small amount
    df['geometry'] = df.geometry.simplify(10)
    # add a zero buffer to fix invalid geometries
    df['geometry'] = df.geometry.buffer(0)

    # check if all geometries in df area valid
    if not df.geometry.is_valid.all():
        print('Some geometries are invalid. Attempting to repair.')
        # df['geometry'] = df.geometry.apply(lambda g: make_valid(g))
        
        df['valid_geom'] = df.geometry.is_valid
        if not df.valid_geom.all():
            raise Exception('Some geometries are still invalid.')
        # df = df.explode(index_parts=False)
        # Assuming df has been exploded and now contains only simple geometries
        df = df[df['geometry'].apply(lambda x: isinstance(x, (Polygon, MultiPolygon)))]
        print('All geometries are valid.')
    # find all neighbouring regions intersecting with the current target region
    nbrs = gpd.sjoin(region_bounds, df, how='left', predicate='overlaps')
    nbrs = nbrs[(nbrs['region_code'] != rc) & (~np.isnan(nbrs['area_right']))]
    
    nbr_regions = list(set(nbrs['region_code']))
    ids_to_remove = []
    for nbr_rc in nbr_regions:
        print(f'   Processing {rc}-{nbr_rc} overlapping regions.')
        nbr_clip_fname = f'{nbr_rc}_4269_dem_clip_final.geojson'
        nbr_fpath = os.path.join(DATA_DIR, 'adjusted_region_polygons', 'adjusted_clipping_masks', nbr_clip_fname)
        nbr_polygon = gpd.read_file(nbr_fpath)
        nbr_polygon = nbr_polygon.to_crs(df.crs)
        nbr_polygon = nbr_polygon.dissolve()

        # get all the polygons in the covering set that intersect with the neighbouring region
        intersection = df[df.intersects(nbr_polygon.geometry.values[0])].copy()
        
        # get the area of each polygon that intersects
        intersection['original_area'] = intersection.geometry.area / 1e6
        
        # get the intersection of each polygon in the covering set with the neighbouring region
        intersection = gpd.overlay(intersection, nbr_polygon, how='intersection')
        intersection['intersecting_area'] = intersection.geometry.area / 1e6

        # we are only interested in polygons that have nonzero intersecting area
        intersection = intersection[intersection['intersecting_area'] > 0]

        # get the ratio of the intersection to the total area of the polygon
        intersection['ratio'] = intersection['intersecting_area'] / intersection['original_area']
        
        # we want to exclude the polygons that are covered by the neighboring region polygon        
        intersection = intersection.sort_values(by='ratio', ascending=False)      
        covered_ids = intersection[intersection['ratio'] > 0.9]['FID'].values
        ids_to_remove += list(covered_ids)

    print('')
    n_polygons = len(df)
    df = df[~df['FID'].isin(ids_to_remove)]
    n_end = len(df)
    print(f'   {n_polygons - n_end} polygons removed from the covering set.')
    
    df.to_file(adjusted_basins_path, driver='GeoJSON')

assembled_region_polygon_fname = f'BCUB_regions_R0.geojson'
if not os.path.exists(os.path.join(DATA_DIR, assembled_region_polygon_fname)):
    print('Assembling the adjusted region polygons into a single file.')
    region_polygon_files = [e for e in os.listdir(output_folder) if e.endswith('R0.geojson')]
    if len(region_polygon_files) < 18:
        raise Exception(f'Missing adjusted region polygon files ({len(region_polygon_files)}/18 found). Skipping assembly.')
    regions = []
    for f in region_polygon_files:
        gdf = gpd.read_file(os.path.join(output_folder, f))
        regions.append(gdf)
    assembled_gdf = gpd.GeoDataFrame(pd.concat(regions), crs=3005)
    assembled_gdf.to_file(os.path.join(DATA_DIR, assembled_region_polygon_fname), driver='GeoJSON')

    




    