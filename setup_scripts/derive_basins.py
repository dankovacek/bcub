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


def process_dem_by_basin(region, batch_gdf, raster_crs, region_raster_fpath, temp_folder, n_procs):
    
    ct0 = time.time()
    polygon_inputs = [(region, 'dem', i, row, raster_crs, region_raster_fpath, temp_folder) for i, row in batch_gdf.iterrows()]
    with mp.Pool() as pl:
        temp_raster_paths = pl.map(bpf.dump_poly, polygon_inputs)
    
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

region_codes = [
    # 'HGW',
    # 'VCI',
    # 'WWA', 
    # '08C', 
    # '08F', 
    # '08G', 
    # '08A', '08B', 
    # '08E',
    # '08C', '10E', 
    # 'ERK', 'HAY', 'YKR', 
    # 'CLR', 'PCR', 'FRA',
    'LRD', 
    ]

idx_dict = {}
tracker_dict = {}
    
def main():
    out_crs = 3005
    min_basin_area = 1.0 # km^2
    rn = 0
    n_processed_basins = 0
    for region in region_codes:
        t0 = time.time()
        rn += 1
        idx_dict[region] = {}
        tracker_dict[region] = []

        fdir_file = f'{region}_{DEM_source}_3005_fdir.tif'
        fdir_path = os.path.join(DEM_folder, fdir_file)

        # get the file size in MB
        filesize = (os.path.getsize(fdir_path) >> 20 )

        temp_folder = os.path.join(DATA_DIR, f'derived_basins/{region}/temp/') 
        ppt_folder = os.path.join(DATA_DIR, f'pour_points/{region}/')
        output_polygon_folder = os.path.join(DATA_DIR, f'derived_basins/{region}')

        for f in [temp_folder, output_polygon_folder]:
            if not os.path.exists(f):
                os.makedirs(f)

        print(f'Processing region {region} ({rn}/{len(region_codes)})')

        region_raster, region_raster_crs, _, region_raster_fpath = retrieve_raster(region)
        raster_resolution = tuple(abs(e) for e in region_raster.rio.resolution())
        
        print(f'    {region} raster crs = {region_raster_crs}.')
          
        # track the pour point indices:
        #     these indices correspond to the region polygon rasters
        #     but recall they were masked with the region polygons
        #     using a buffer length equal to the DEM resolution

        # check for an output file to allow for partial updating
        # by tracking (unique) ppt indices
        output_folder = os.path.join(DATA_DIR, f'derived_basins/{region}/')        
        output_fpath = os.path.join(output_folder, f'{region}_basins.parquet')
        # attributes_fpath = os.path.join(output_folder, f'{region}_basin_attributes.geojson')
        # if os.path.exists(output_fpath):
        #     continue
        
        # check if the ppt batches have already been created
        batch_folder = os.path.join(temp_folder, 'ppt_batches/')
        batches_exist = check_for_ppt_batches(batch_folder)

        # break the pour point dataframe into batches to limit the RAM used in 
        # processing basins for each pour point.
        temp_ppt_filepath = os.path.join(batch_folder, f'temp_ppts.shp')

        memory_info = psutil.virtual_memory()
        available_memory = memory_info.available / 1E9

        if not batches_exist:
            print(f'    Generating new batch paths.')
            # Break the pour points into unique sets
            # to avoid duplicate delineations
            ppt_file = os.path.join(ppt_folder, f'{region}_pour_pts_filtered.geojson')
            
            ppt_gdf = gpd.read_file(ppt_file)
            # filter out any ppts that are alread in the database
            # first query all the ppt_x and ppt_y values for this region
            ppt_gdf = filter_processed_pts(region, ppt_gdf)
            if ppt_gdf.empty:
                print(f'    All pour points for {region} have already been processed.')
                continue

            batch_ppt_paths = create_batches(ppt_gdf, filesize, temp_ppt_filepath)
        else:
            print(f'    Retrieving existing batch paths.')
            batch_ppt_files = list(sorted([f for f in os.listdir(batch_folder) if f.endswith('.shp')]))
            batch_ppt_paths = [os.path.join(batch_folder, f) for f in batch_ppt_files]

        n_batch_files = len(batch_ppt_paths)
        batch_output_files = []
        for ppt_batch_path in batch_ppt_paths:
            t_batch_start = time.time()
            batch_no = int(ppt_batch_path.split('_')[-1].split('.')[0])
            print(f'  Starting processing {region} batch {batch_no+1}/{n_batch_files}')
            temp_fname = f'{region}_temp_raster.tif'
            temp_basin_raster_path = os.path.join(temp_folder, temp_fname)            
            batch_output_fpath = output_fpath.replace('.parquet', f'_{batch_no:04d}.geojson')
            if os.path.exists(batch_output_fpath):
                print(f'    {batch_output_fpath} already exists. Skipping.')
                batch_output_files.append(batch_output_fpath)
                continue
            # creates the minimum set of rasters with non-overlapping polygons
            batch_rasters = sorted([e for e in os.listdir(temp_folder) if e.startswith(temp_fname.split('.')[0])])
            if len(batch_rasters) <= 2:
                batch_basin_delineation(fdir_path, ppt_batch_path, temp_basin_raster_path)
                batch_rasters = sorted([e for e in sorted(os.listdir(temp_folder)) if e.endswith('.tif')])
                
            tb1 = time.time()
            print(f'    Basins delineated for {region} ppt batch {batch_no} in {(tb1-t_batch_start)/60:.1f}min. \n      {len(batch_rasters)} raster sub-batches to process.')

            batch_size_GB = len(batch_rasters) * filesize / 1E3
            
            n_procs = max(1, int(np.floor((available_memory - 22) / batch_size_GB)) )
            if region in ['PCR', 'FRA', 'LRD']:
                # these regions have large rasters and require more memory
                n_procs = 1
            
            print(f'    Allocating {n_procs:.0f} processes for a batch size of {batch_size_GB:.1f}GB')
            
            # process the raster batches in parallel
            path_inputs = [(br, f'EPSG:{region_raster_crs}', raster_resolution, min_basin_area, temp_folder) for br in batch_rasters]
                        
            # use fewer processes here because step is already parallelized?
            p = mp.Pool(n_procs)
            trv0 = time.time()
            all_polygons = p.map(raster_to_vector_basins_batch, path_inputs)
            # concatenate polygons
            batch_gdf = gpd.GeoDataFrame(pd.concat(all_polygons), crs=region_raster_crs)
            p.close()
            
            trv1 = time.time()
            print(f'    {trv1-trv0:.2f}s to convert {len(batch_rasters)} rasters to vector basins.')            
            
            # sort by VALUE to maintain ordering from ppt batching / raster basin unnesting operation
            batch_gdf.sort_values(by='VALUE', inplace=True)
            batch_gdf.reset_index(inplace=True, drop=True)
            
            # compute drainage area for each polygon
            batch_gdf['drainage_area_km2'] = batch_gdf['geometry'].area / 1E6
            
            # if ordering is correct, we don't have to do this step
            # we can instead just iterate over polygons, derive attributes,
            # assign values to ppt dataframe.
            # add the pour point location info to each polygon
            batch_gdf = bpf.match_ppt_to_polygons_by_order(ppt_batch_path, batch_gdf, raster_resolution)

            # extract terrain attributes
            assert batch_gdf.crs.to_epsg() == region_raster_crs; 'Batch gdf crs does not match region raster crs.'
            
            terrain_results = process_dem_by_basin(region, batch_gdf, region_raster_crs, 
                                                     region_raster_fpath, temp_folder, n_procs)
            
            terrain_gdf = pd.DataFrame(terrain_results)
            # sort by ID column
            terrain_gdf.sort_values(by='ID', inplace=True)
            terrain_gdf.set_index('ID', inplace=True)
            
            # check that the batch_gdf and terrain_gdf have the same index
            assert (batch_gdf.index == terrain_gdf.index).all()
            
            # concatenate the terrain columns to the batch_gdf
            merged_gdf = gpd.GeoDataFrame(pd.concat([batch_gdf, terrain_gdf], axis=1), crs=region_raster_crs)
            merged_gdf['region_code'] = region
            merged_gdf.to_file(batch_output_fpath)
            batch_output_files.append(batch_output_fpath)
            
            # clean up the temporary files
            clean_up_temp_files(temp_folder, batch_rasters)
                    
        merged_basins = bpf.merge_geojson_files(batch_output_files, output_fpath, output_polygon_folder)
        
        # save the results to a parquet file containing geometry
        convert_to_parquet(merged_basins, output_fpath)
        
        t_n = time.time()
        n_processed_basins = max(1, n_processed_basins)
        ut = (t_n - t0) / n_processed_basins
        print(f'Total processing time for {region}: {t_n-t0:.1f}s ({ut:.2f}/basin).')
        
        for f in batch_output_files:
            os.remove(f)
        for f in os.listdir(batch_folder):
            os.remove(os.path.join(batch_folder, f))
        os.rmdir(batch_folder)
        print('')
        print('      ------------------------------------')
        print('')
            

if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print('')
    print('###################################################')
    print('')
    print(f'Script completed in {t1-t0:.2f}s.')
    print('__________________________________________________')
