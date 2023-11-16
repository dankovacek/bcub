# generate basins
import os
import time
import psutil
import random

import warnings
warnings.filterwarnings('ignore')
import psycopg2

import numpy as np
import geopandas as gpd
import pandas as pd
import rioxarray as rxr

import multiprocessing as mp

from shapely.geometry import Point
from shapely.validation import make_valid
from shapely import wkt
import pyarrow.parquet as pq

from whitebox.whitebox_tools import WhiteboxTools

import basin_processing_functions as bpf

wbt = WhiteboxTools()
wbt.verbose = False

DEM_source = 'USGS_3DEP'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_folder = os.path.join(BASE_DIR, 'processed_data/processed_dem/')
region_files = os.listdir(DEM_folder)
region_codes = sorted(list(set([e.split('_')[0] for e in region_files])))

DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')

schema = 'basins_schema'
attribute_table = 'basin_attributes'

conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
}

def retrieve_raster(region):
    filename = f'{region}_USGS_3DEP_3005.tif'
    fpath = os.path.join(DEM_folder, filename)
    raster = rxr.open_rasterio(fpath, mask_and_scale=True)
    crs = raster.rio.crs
    affine = raster.rio.transform(recalc=False)
    return raster, crs, affine, fpath


def get_region_area(region):
    polygon_path = os.path.join(DATA_DIR, 'merged_basin_groups/region_polygons/')
    poly_files = os.listdir(polygon_path)
    file = [e for e in poly_files if e.startswith(region)]
    if len(file) != 1:
        raise Exception; 'Region shape file not found.'
    fpath = os.path.join(polygon_path, file[0])
    gdf = gpd.read_file(fpath)
    gdf = gdf.to_crs(3005)
    return gdf['geometry'].area.values[0] / 1E6


def retrieve_polygon(fname):
    shp_num = fname.split('_')[-1].split('.')[0]
    return (shp_num, gpd.read_file(fname))


def check_polygon_df(region):
    poly_path = os.path.join(DATA_DIR, f'derived_basins/{region}_derived_basin_sample.geojson')
    if os.path.exists(poly_path):
        return True
    else:
        return False


def create_pour_point_gdf(stream, acc, pts, crs, n_chunks=2):
    """Break apart the list of stream pixels to avoid memory 
    allocation issue when indexing large rasters.

    Args:
        stream (_type_): _description_
        confluences (_type_): _description_
        n_chunks (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    
    n_chunks = int(10 * np.log(len(pts)))

    # print(f'{len(pts)} points, breaking into {n_chunks} chunks.')

    conf_chunks = np.array_split(np.asarray(pts), indices_or_sections=n_chunks)

    point_array = []
    acc_array = []

    for chunk in conf_chunks:

        xis = [int(c[0]) for c in chunk]
        yis = [int(c[1]) for c in chunk]
        acc_array += [acc.data[0][c[0], c[1]] for c in chunk]

        ppts = stream[0, xis, yis]
        coords = tuple(map(tuple, zip(ppts.coords['x'].values, ppts.coords['y'].values)))
        point_array += [Point(p) for p in coords]

    df = pd.DataFrame()
    df['num_acc_cells'] = acc_array
    df['pt_id'] = list(range(len(df)))

    gdf = gpd.GeoDataFrame(df, geometry=point_array, crs=crs)
    # print(f'Created dataframe with {len(gdf)} pour points')
    del point_array, df, conf_chunks
    return gdf


def check_for_ppt_batches(batch_folder):    
    if not os.path.exists(batch_folder):
        os.mkdir(batch_folder)
        return False
    existing_batches = os.listdir(batch_folder)
    return len(existing_batches) > 0


def create_batches(df, filesize, temp_ppt_filepath):    

    # divide the dataframe into chunks for batch processing
    # save the pour point dataframe to temporary filRes
    # and limit temporary raster files to ?GB / batch
    
    # batch_limit = 2.5E3
    batch_limit = 2E6
    n_batches = int(filesize * len(df) / batch_limit) + 1
    print(f'        ...running {n_batches} batch(es) on {filesize:.1f}MB raster.')
    batch_paths = []
    n = 0
    if len(df) * filesize < batch_limit:
        temp_fpath = temp_ppt_filepath.replace('.shp', f'_{n:04}.shp')
        df.to_file(temp_fpath)
        batch_paths.append(temp_fpath)
        # idx_batches.append(df.index.values)
    else:
        # randomly shuffle the indices 
        # to split into batches
        indices = df.index.values
        random.shuffle(indices)
        batches = np.array_split(np.array(indices), indices_or_sections=n_batches)
        for batch in batches:
            n += 1
            batch_gdf = df.iloc[batch].copy()
            # idx_batches.append(batch)
            temp_fpath = temp_ppt_filepath.replace('.shp', f'_{n:04}.shp')
            batch_gdf.to_file(temp_fpath)
            # keep just the filename
            batch_paths.append(temp_fpath)

    # return list(zip(batch_paths, idx_batches))
    return batch_paths


def batch_basin_delineation(fdir_path, ppt_batch_path, temp_raster_path):

    wbt.unnest_basins(
        fdir_path, 
        ppt_batch_path, 
        temp_raster_path,
        esri_pntr=False, 
        # callback=default_callback
    )



def dump_poly(inputs):
    """Take the polygon batches and create raster clips with each polygon.
    Save the individual polygons as geojson files for later use.

    Args:
        inputs (array): raster file, vector file, temporary file folder, and the raster crs.

    Returns:
        string: list of filepaths for the clipped rasters.
    """
    region, layer, i, row, crs, raster_fpath, temp_folder = inputs

    bdf = gpd.GeoDataFrame(geometry=[row['basin_geometry']], crs=crs)

    basin_fname = f'buffered_basin_temp_{i:05d}.geojson'
    basin_fpath = os.path.join(temp_folder, basin_fname)
    raster_fname = f'{region}_basin_{layer}_temp_{int(i):05}.tif'
    fpath_out = os.path.join(temp_folder, raster_fname)

    if (not os.path.exists(basin_fpath)):
        bdf.geometry = bdf.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)    
        bdf.to_file(basin_fpath, driver='GeoJSON')

    # New filename. Assumes input raster file has '.tif' extension
    # Might need to change how you build the output filename         
    if not os.path.exists(fpath_out):
        # Do the actual clipping
        command = f'gdalwarp -s_srs {crs} -cutline {basin_fpath} -crop_to_cutline -multi -of gtiff {raster_fpath} {fpath_out} -wo NUM_THREADS=ALL_CPUS'
        print('doing gdalwarp')
        print(command)
        print('   ...')
        os.system(command)

    g = None
    
    return fpath_out


def process_dem_by_basin(region, batch_gdf, raster_crs, region_raster_fpath, temp_folder, n_procs):
    
    ct0 = time.time()
    polygon_inputs = [(region, 'dem', i, row, raster_crs, region_raster_fpath, temp_folder) for i, row in batch_gdf.iterrows()]
    with mp.Pool() as pl:
        temp_raster_paths = pl.map(dump_poly, polygon_inputs)
    
    ct1 = time.time()
    print(f'    {(ct1-ct0)/60:.1f}min to create {len(polygon_inputs)} clipped rasters.')
    pl.close()
    with mp.Pool(n_procs) as pl:
        terrain_data = pl.map(bpf.process_terrain_attributes, temp_raster_paths)
        ct2 = time.time()
        print(f'    {(ct2-ct1)/60:.1f}min to process terrain attributes for {len(polygon_inputs)} basins.')
    
    terrain_gdf = pd.DataFrame(terrain_data)
    
    return terrain_gdf


def clean_up_temp_files(temp_folder, batch_rasters):    
    temp_files = [f for f in os.listdir(temp_folder) if 'temp' in f]
    raster_clips = [e for e in os.listdir(temp_folder) if DEM_source in e]
    all_files = batch_rasters + raster_clips + temp_files
    for f in list(set(all_files)):
        os.remove(os.path.join(temp_folder, f))


def raster_to_vector_basins_batch(input):

    raster_fname, raster_crs, resolution, min_area, temp_folder = input
    raster_path = os.path.join(temp_folder, raster_fname)
    raster_no = int(raster_fname.split('_')[-1].split('.')[0])
    polygon_path = os.path.join(temp_folder, f'temp_polygons_{raster_no:05}.shp')
    
    # this function creates rasters of ordered 
    # sets of non-overlapping basins
    wbt.raster_to_vector_polygons(
        raster_path,
        polygon_path,
    )

    gdf = gpd.read_file(polygon_path, crs=raster_crs)

    # simplify the polygon geometry to avoid self-intersecting polygons
    simplify_dim = 0.5 * np.sqrt(resolution[0]**2 + resolution[1]**2)
    simplify_dim = abs(resolution[0])
    buffer_dim = 10
    gdf.geometry = gdf.geometry.buffer(buffer_dim)
    gdf.geometry = gdf.geometry.simplify(simplify_dim)
    gdf.geometry = gdf.geometry.buffer(-1.0 * buffer_dim)
    
    gdf = bpf.filter_and_explode_geoms(gdf, min_area)
    if not (gdf.geometry.geom_type == 'Polygon').all():
        gdf = bpf.filter_and_explode_geoms(gdf, min_area)

    assert (gdf.geometry.geom_type == 'Polygon').all()
    
    return gdf


def convert_to_parquet(merged_basins, output_fpath): 
    # we want to end up with multiple geometries: 
    # i) pour point, 2) basin polygon, 3) basin centroid 
    # these column names must match the names mapped to geometry 
    # columns in the populate_postgis file 
    merged_basins['basin_geometry'] = merged_basins['geometry'] 
    merged_basins['centroid_geometry'] = merged_basins['geometry'].centroid
    merged_basins['geometry'] = [Point(x, y) for x, y in zip(merged_basins['ppt_lon_m_3005'], merged_basins['ppt_lat_m_3005'])]
    
    # if the parquet file exists, append the data if it isn't duplicated
    if os.path.exists(output_fpath):
        schema = pq.read_schema(output_fpath)
        existing_data = gpd.read_parquet(output_fpath, schema=schema)
        output_data = pd.concat([existing_data, merged_basins])
        output_data.drop_duplicates(subset=['ppt_lon_m_3005', 'ppt_lat_m_3005'], inplace=True)
        output_data = gpd.GeoDataFrame(output_data, geometry='geometry', crs=3005)
    else:
        output_data = merged_basins

    # convert to parquet format 
    # keep the index as this will be used to 
    # reference polygons and attributes later 
    output_data.to_parquet(output_fpath, index=True) 
    

def get_processed_ppts(region, schema, attribute_table):
    """Query the database for rows with a valid basin geometry.
    Return the pour point coordinates as a tuple"""
    q = f"""
    SELECT ST_AsText(pour_pt)
    FROM {schema}.{attribute_table} 
    WHERE region_code = '{region}' AND basin IS NOT NULL;
    """
    with warnings.catch_warnings():
        # ignore warning for non-SQLAlchemy Connecton
        # see github.com/pandas-dev/pandas/issues/45660
        warnings.simplefilter('ignore', UserWarning)
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(q)
                results = cur.fetchall() 
                pts = [e[0] for e in results]
                df = pd.DataFrame(pts, columns=['pt'])
                df['geometry'] = df.apply(wkt.loads)
                return gpd.GeoDataFrame(df, geometry='geometry')
                

def filter_processed_pts(region, ppt_gdf):
    processed_pts = get_processed_ppts(region, schema, attribute_table)
    joined = gpd.sjoin(ppt_gdf, processed_pts, how='left', op='intersects')
    filtered_gdf = joined[joined['index_right'].isna()]
    n_processed = len(processed_pts)
    n_ppts = len(ppt_gdf)
    print(f'    {len(filtered_gdf)} pour points have not been processed, dropping {n_ppts - n_processed} duplicates.')
    return filtered_gdf



out_crs = 3005
min_basin_area = 1.0 # km^2
rn = 0
n_processed_basins = 0

t0 = time.time()
rn += 1

def query_unprocessed_params(param, schema, attribute_table):
    q = f"SELECT id, region_code, ppt_x, ppt_y, basin FROM {schema}.{attribute_table} WHERE {param} IS NULL;"
    # set up query connection 
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            results = cur.fetchall()
            return results


def update_db(param, value, bid, schema, attribute_table):
    q = f"UPDATE {schema}.{attribute_table} SET {param} = {value} WHERE id = {bid};"
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            conn.commit()
            return True
        

def process_dem_attrs(param, bid, region, ppt_x, ppt_y, basin):
    parquet_fpath = os.path.join(DATA_DIR, f'derived_basins/{region}/{region}_basins.parquet')
    # open the parquet file
    basin_gdf = gpd.read_parquet(parquet_fpath)

    # filter to the basin of interest by matching ppt_x and ppt_y
    basin_info = basin_gdf[(basin_gdf['ppt_lon_m_3005'] == ppt_x) & (basin_gdf['ppt_lat_m_3005'] == ppt_y)].copy()
    # get the attribute
    return basin_info



params = ['slope_deg']
schema = 'basins_schema'
attribute_table = 'basin_attributes'

for param in params:
    results = query_unprocessed_params(param, schema, attribute_table)
    
    print(f'Updating {len(results)} {param} values.')
    for res in results:
        bid, region, ppt_x, ppt_y, basin = res
        basin_info = process_dem_attrs(param, bid, region, ppt_x, ppt_y, basin)
        
        p = param
        if param == 'slope_deg':
            p = 'Slope_deg'

        
        updated_value = basin_info[p].values[0]
        print(f'    {param} for basin {bid} is {updated_value}.')
        if np.isnan(updated_value):
            print(f'    {param} for basin {bid} is {updated_value}, trying to re-process param.')
            region_raster_fpath = os.path.join(DEM_folder, f'{region}_USGS_3DEP_3005.tif')
            temp_folder = os.path.join(DATA_DIR, f'basin_attributes/temp/')
            terrain_results = process_dem_by_basin(region, basin_info, 'EPSG:3005', 
                                                region_raster_fpath, temp_folder, 1)
            print(terrain_results)
            print(asdfasdf)

            terrain_gdf = pd.DataFrame(terrain_results)
            # sort by ID column
            terrain_gdf.sort_values(by='ID', inplace=True)
            terrain_gdf.set_index('ID', inplace=True)
        else:
            print(f'Updating {param} for basin {bid} to {updated_value}.')
        
            value_updated = update_db(param, updated_value, bid, schema, attribute_table)
            print(f'Updated {param} for basin {bid} to {updated_value} ({value_updated}).')
        
            
   
            
 
    
