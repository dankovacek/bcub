import geopandas as gpd
import dask_geopandas as dgp
import pyarrow.parquet as pq
import psycopg2
import psycopg2.extras as extras
import os
from time import time
import pandas as pd

# import rasterio as rio

# from concurrent.futures import ThreadPoolExecutor

import numpy as np
import multiprocessing as mp
# import dask
# from dask.diagnostics import ResourceProfiler, ProgressBar
# from dask.distributed import Client, LocalCluster, progress
# from dask import dataframe as dd
# from dask.delayed import delayed

# from shapely.geometry import Point, Polygon


db_host = 'localhost'
db_name = 'basins'
db_user = 'postgres'
db_password = 'pgpass'
schema_name = 'basins_schema'

common_data = '/home/danbot2/code_5820/large_sample_hydrology/common_data'

nalcms_2020_folder = os.path.join(common_data, 'NALCMS_NA')

dtype_dict = {
    'double': 'FLOAT',
    'int64': 'INT',
    'float64': 'FLOAT',
    'string': 'VARCHAR(255)',
    'object': 'VARCHAR(255)',
    'bool': 'SMALLINT',
}

geom_dict = {
    'geometry': 'geom',
    'centroid_geometry': 'centroid',
    'basin_geometry': 'basin',
}   

# the integer groups are the land cover classes in the NALCMS data
# as classified in HYSETS (Arsenault, 2020)
groups = {
    f'land_use_forest_frac': [1, 2, 3, 4, 5, 6], 
    f'land_use_shrubs_frac': [7, 8, 11],
    f'land_use_grass_frac': [9, 10, 12, 13, 16],
    f'land_use_wetland_frac': [14],
    f'land_use_crops_frac': [15],
    f'land_use_urban_frac': [17],
    f'land_use_water_frac': [18],
    f'land_use_snow_ice_frac': [19]
}


def basic_table_change(db_host, db_name, db_user, db_password, query):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    conn.close()
    

def basic_query(db_host, db_name, db_user, db_password, query):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    conn.close()
    if len(result) > 0:
        return [e[0] for e in result]
    else:
        return None

def add_table_columns(schema_name, db_host, db_name, db_user, db_password, new_cols):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()

    table_name = f'basin_attributes'
    print(f'    adding new cols {new_cols}.')
    # add the new columns to the table
    
    for col in new_cols:
        dtype = 'FLOAT'
        if 'land_use' in col:
            dtype = 'INT'

        cur.execute(f'ALTER TABLE {schema_name}.{table_name} ADD COLUMN IF NOT EXISTS {col} {dtype};')

    conn.commit()
    conn.close()
    


def alter_column_names(schema_name, db_host, db_name, db_user, db_password, old_cols, new_cols):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()

    table_name = f'basin_attributes'
    for i in range(len(old_cols)):
        print(f'    renaming {old_cols[i]} to {new_cols[i]}')
        cur.execute(f'ALTER TABLE {schema_name}.{table_name} RENAME COLUMN {old_cols[i]} TO {new_cols[i]};')

    conn.commit()
    conn.close()


def clip_nalcms_values_by_polygon(raster_table, attribute_table, year, id_list):
    raster_values = ','.join([str(e) for e in list(range(1, 20))])
    ids = ','.join([str(e) for e in id_list])
    t0 = time()
    # q = f"""
    #     SELECT id, (ST_PixelAsCentroids({raster_table}.nalcms_{year}_raster)).val AS pixel_value, COUNT(*) as count
    #         FROM {raster_table}, {attribute_table}
    #         WHERE {attribute_table}.id IN ({ids})
    #         AND ST_Intersects(
    #             {raster_table}.nalcms_{year}_raster, 
    #             {attribute_table}.basin
    #             )
    #         GROUP BY id, pixel_value;
    # """
    # f"SELECT /*+ PARALLEL(your_table, 8) */ * FROM ({}) AS your_table"
    q = f"""
    WITH subquery AS (
    SELECT id, (ST_PixelAsCentroids({raster_table}.nalcms_{year}_raster)).val AS pixel_value, COUNT(*) AS count
    FROM {raster_table}, {attribute_table}
    WHERE {attribute_table}.id IN ({ids})
    AND ST_Intersects({raster_table}.nalcms_{year}_raster, {attribute_table}.basin)
    GROUP BY id, pixel_value
    )
    SELECT id, jsonb_agg(jsonb_build_array(value_list.value, ROUND(COALESCE(subquery.count, 0)::float / total_count * 100)::int))
    FROM (
        SELECT DISTINCT value FROM unnest(ARRAY[{raster_values}]) AS value -- Modify the array as per your specified integer values
    ) AS value_list
    LEFT JOIN (
        SELECT id, pixel_value, count, SUM(count) OVER (PARTITION BY id) AS total_count
        FROM subquery
    ) AS subquery ON value_list.value = subquery.pixel_value
    WHERE subquery.id IS NOT NULL
    GROUP BY id;
    """
    # q = f"""
    # SELECT ST_AsGDALRaster(ST_Union(ST_Clip(r.nalcms_{year}_raster, a.basin)), 'GTiff', ARRAY['COMPRESS=LWZ']) AS clipped_raster_binary
    # FROM {raster_table} r, {attribute_table} a
    # WHERE a.id = {polygon_id}
    # AND ST_Intersects(r.nalcms_{year}_raster, a.basin);
    # """

    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
        
    # Execute the query in parallel
    cur.execute(q)

    # Fetch the results
    results = cur.fetchall()

    # print(f'  {len(results)} results returned.')
    df = reformat_result(results)
    
    # Close the cursor and connection
    cur.close()
    conn.close()
    return df

def reformat_result(result):
    """
    Reformat the query result into a dataframe with integer values representing
    the percentage of each land use class in the basin.
    Each row represents a basin, each column represents a land use class.
    """
    data = [{'id': entry[0], **{value[0]: value[1] for value in entry[1]}} for entry in result]
    df = pd.DataFrame.from_dict(data)
    df = df.fillna(0).astype(int)
    df.set_index('id', inplace=True)

    # Group the columns and sum the values
    grouped_df = pd.DataFrame()
    for label, group in groups.items():
        # Check if all numbers in the group exist as columns in the DataFrame
        existing_columns = [col for col in group if col in df.columns]
        if existing_columns:
            grouped_df[label] = df[existing_columns].sum(axis=1, skipna=True)
        else:
            grouped_df[label] = 0

    return grouped_df


def retrieve_unprocessed_ids(n_ids, lu_cols):
    id_query = f"""SELECT id FROM basins_schema.basin_attributes """
    n = 0
    for c in lu_cols:
        if n == 0:
            id_query += f"WHERE ({c} IS NULL OR {c} != {c}) "
            n = 1
        else:
            id_query += f"AND ({c} IS NULL OR {c} != {c}) "
    id_query += f"LIMIT {n_ids};"
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    cur.execute(id_query)
    res = cur.fetchall()
    conn.close()
    return [e[0] for e in res]


def count_processed_ids(lu_cols):
    id_query = f"""SELECT COUNT(*) FROM basins_schema.basin_attributes """
    n = 0
    for c in lu_cols:
        if n == 0:
            id_query += f"WHERE ({c} IS NOT NULL AND {c} = {c}) "
            n = 1
        else:
            id_query += f"AND ({c} IS NOT NULL AND {c} = {c}) "
        
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    cur.execute(id_query)
    res = cur.fetchone()
    print(f'Number of processed IDs: {res[0]}')
    conn.close()


def process_NALCMS(input_data):
    year, id_list, lu_cols = input_data
        
    # existing = get_existing_nalcms_vals('basins_schema.nalcms_2010', 'basins_schema.basin_attributes', id_list)
    # df = pd.DataFrame.from_dict(existing, orient='index', columns=['existing'])
    t1 = time()
    times = []
    # for y in [2010, 2015, 2020]:
    raster_table = f'{schema_name}.nalcms_{year}'
    attribute_table = 'basins_schema.basin_attributes'
    df = clip_nalcms_values_by_polygon(raster_table, attribute_table, year, id_list)

    t2 = time()
    times.append(t2-t1)
    avg_t = sum(times) / len(id_list)
    # sort by id
    df.sort_index(inplace=True)
    df = df.reset_index()
    # print(f'    Average time to process {len(id_list)} basins for {year} land cover data: {avg_t:.2f}s')
    return df


def create_raster_table(db_host, db_name, db_user, db_password, schema_name, raster_fpath, table_name):
    # use command to add a 
    
    # raster2pgsql -I -C -M -F -s 3005 -t auto raster_file.tif basins_schema.nalcms_2020_table | psql -d basins

    # drop all columns that contain the substring "land_use_2015"
    query = "ALTER TABLE basins_schema.nalcms_2015 DROP COLUMN IF EXISTS land_use_2015*;"


def extend_postgis_db(db_host, db_name, db_user, db_password, schema_name, total_bounds):
    # connect to the PostGIS database
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    print('    Database connected...')

    # query the database to get the existing columns
    existing_col_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'basin_attributes';"
    cur.execute(existing_col_query)
    existing_cols = [e[0] for e in cur.fetchall()]

    # get one geometry from the database to use for the 
    # raster extraction all geometry columns are EPSG:3005
    # geom_query = f"SELECT geom FROM {schema_name}.basin_attributes LIMIT 1;"
    # test = gpd.read_postgis(geom_query, conn, geom_col='geom')
    # print(test.crs)
    
    all_land_use_cols = sorted(list(set([e for e in existing_cols if ('land_use_2010' in e)])))
    land_use_categories = [e.split('land_use_')[1][5:] for e in all_land_use_cols]
    
    # rename the existing land use cols to contain 2010 to indicate which year
    # modified_cols = [f"land_use_2010_{e.split('land_use_')[1]}" for e in all_land_use_cols]
    # alter_column_names(schema_name, db_host, db_name, db_user, db_password, all_land_use_cols, modified_cols)
    
    # create new table column
    # print('    Starting column insert...')
    # create new columns corresponding to the 2020 land use data
    new_land_use_columns = [f"land_use_2015_{e.split('land_use_2010_')[1]}" for e in all_land_use_cols]
    # add_table_columns(schema_name, db_host, db_name, db_user, db_password, new_land_use_columns)

    # polygon geometry column is in EPSG: 3005,
    # clipped NALCMS raster is also in EPSG: 3005
    # so we can use the ST_Intersects function to find the land use
    # for each polygon

    # create a query to use the ST_Intersects function to mask the land use raster with all the polygons
    # and then aggregate the land use values for each polygon
    extract_land_use_values(schema_name, db_host, db_name, db_user, db_password, land_use_fpath)
    print(asdf)
        
    # extras.execute_values(cur, query, tuples)
    # conn.commit()
    t6 = time()
    # print(f'    Inserted {len(tuples)} rows in {t6-t4:.2f} seconds.')
    # commit the changes
    print(f'    Changes committed.  Finished processing')
    print('-------------------------------------------------')
    print('')
    conn.close()


def update_lulc_data(new_data, year):
    # establish a connection to the database
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    # update the database with the new land use data

    ids = tuple([int(e) for e in new_data['id'].values])
    # print(ids)
    # print(asdfasd)

    # data_tuples = [tuple(*[str(a) for a in x[1:]], x[0].astype(str)) for x in new_data.to_numpy()]
    data_tuples = list(new_data.itertuples(index=False, name=None))
    # for row in data_tuples:
    #     print(row)
    #     print([type(e) for e in row])
    
    # print(asdfas)
    # cols_str = ','.join(new_data.columns)
    cols = new_data.columns.tolist()

    query = f"""
    UPDATE {schema_name}.basin_attributes AS basin_tab
        SET {cols[1]}_{year} = data.v1,
            {cols[2]}_{year} = data.v2,
            {cols[3]}_{year} = data.v3,
            {cols[4]}_{year} = data.v4,
            {cols[5]}_{year} = data.v5,
            {cols[6]}_{year} = data.v6,
            {cols[7]}_{year} = data.v7,
            {cols[8]}_{year} = data.v8
        FROM (VALUES %s) AS data(id, v1, v2, v3, v4, v5, v6, v7, v8)
    WHERE basin_tab.id = data.id;
    """

    t0 = time()

    extras.execute_values(cur, query, data_tuples)
    
    # commit the changes
    conn.commit()
    t1 = time()
    # test the update
    # select_query = f"SELECT id, land_use_forest_frac_{year}, land_use_shrubs_frac_{year}, land_use_water_frac_{year} FROM {schema_name}.basin_attributes WHERE id IN %s;"
    
    # cur.execute(select_query, (ids,))

    # updated_rows = cur.fetchall()
    # print('testing result:')
    # for row in updated_rows:
    #     print(row)
    
    
    print('')
    dt = t1-t0
    ut = len(data_tuples) / dt
    print(f' {dt:.1f}s for {len(data_tuples)} polygons ({ut:.1f}/second)')
    conn.close()


def get_id_list(n_ids, lu_cols, use_test_ids=False):

    test_ids = [
        (770158, 2.04), (1089918, 8.49), (770161, 1.53), (770162, 1.20), (770165, 2.99),
        (1119937, 4.75), (334767, 2.09), (1089940, 53.39), (1131591, 1.82), (770169, 1.05)
        ]
    if use_test_ids:
        id_list = test_ids
    else:
        ta = time()
        id_list = retrieve_unprocessed_ids(n_ids, lu_cols)

        tb = time()
        print(f'    Unprocessed ID query time = {tb-ta:.2f}s for {len(id_list)} rows')
    return id_list

if __name__ == '__main__':

    # parquet_dir = 'processed_data/parquet_basins'
    taa = time()
    # bc_bounds_file = os.path.join(common_data, 'BC_border/BC_study_region_polygon_R0.geojson')
    # t0 = time()
    # bc_bounds = gpd.read_file(bc_bounds_file)
    # t1 = time()
    # print(f'Loaded BC bounds in {t1 - t0:.2f} seconds.  crs={bc_bounds.crs}')

    # query the database for all columns
    col_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'basin_attributes';"
    existing_cols = basic_query(db_host, db_name, db_user, db_password, col_query)
    
    # get just the land use columns
    year = 2020
    all_land_use_cols = sorted(list(set([e for e in existing_cols if (f'land_use_' in e)])))
    land_use_cols = [e for e in all_land_use_cols if e.endswith(f'_{year}')]
    # print(land_use_cols)
    # for c in all_land_use_cols:
        # print(land_use_cols)
        # new_col = 'land_use_' + c.split(f'_gdalwarp_')[1] + f'_{year}'
        # add_col_query = f"ALTER TABLE basins_schema.basin_attributes ADD COLUMN {new_col} INT;"
        # print(add_col_query)
        # basic_table_change(db_host, db_name, db_user, db_password, add_col_query)

    n_iterations = 50000
    n_ids_per_iteration = 20

    id_list = get_id_list(n_iterations * n_ids_per_iteration, land_use_cols)
    
    # count_processed_ids(land_use_cols)

    batches = np.array_split(id_list, n_iterations)
    input_data = [(year, b, land_use_cols) for b in batches]

    p = mp.Pool()
    results = p.map(process_NALCMS, input_data)
    df = pd.concat(results, axis=0, ignore_index=True)
        
    print(f'   ...updating {len(df)} database rows')
    update_lulc_data(df, year)

    tbb = time()
    ut = len(id_list) / (tbb - taa)
    print(f'Finished postgis database modification in {tbb - taa:.2f} seconds ({ut:.2f}basins/s).')
    