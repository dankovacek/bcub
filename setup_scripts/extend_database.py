
import psycopg2
import psycopg2.extras as extras
import os
import warnings
import re
from time import time
import pandas as pd
import random
from shapely.validation import make_valid

import numpy as np
import multiprocessing as mp

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

# control the order of operations
# both of these steps require multiple days to process
process_nalcms = False
process_glhymps = False
process_daymet = True

conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
}
schema_name = 'basins_schema'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#########################
# update these file paths
#########################
DATA_DIR = os.path.join(BASE_DIR, 'input_data/')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')

nalcms_dir = os.path.join(BASE_DIR, 'input_data/NALCMS/')
glhymps_dir = os.path.join(BASE_DIR, 'input_data/GLHYMPS/')

daymet_dir = os.path.join(PROCESSED_DATA_DIR, 'daymet/')

glhymps_fpath = os.path.join(glhymps_dir, 'GLHYMPS_clipped_3005.geojson')
nalcms_fpath = os.path.join(glhymps_dir, 'NA_NALCMS_landcover_2010_3005_clipped.tif')

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
land_use_groups = {
    f'land_use_forest_frac': [1, 2, 3, 4, 5, 6], 
    f'land_use_shrubs_frac': [7, 8, 11],
    f'land_use_grass_frac': [9, 10, 12, 13, 16],
    f'land_use_wetland_frac': [14],
    f'land_use_crops_frac': [15],
    f'land_use_urban_frac': [17],
    f'land_use_water_frac': [18],
    f'land_use_snow_ice_frac': [19]
}


# names of daymet climate parameter indices

# PostgreSQL data type mapping
postgres_types = {
    'int64': 'INTEGER',
    'float64': 'DOUBLE PRECISION',
    'bool': 'BOOLEAN',
    'datetime64[ns]': 'TIMESTAMP',
    'object': 'TEXT',
    # Add more mappings as needed
}

if process_glhymps:
    # load the glhymps data
    t0 = time()
    glhymps_gdf = gpd.read_file(glhymps_fpath)
    t1 = time()
    print(f'    GLHYMPS data loaded in {t1-t0:.1f}s')


def basic_table_change(query):
    cur.execute(query)
    conn.commit()
    

def basic_query(query):
    cur.execute(query)
    result = cur.fetchall()

    if len(result) > 0:
        return [e[0] for e in result]
    else:
        return None


def add_table_columns(schema_name, new_cols):
    table_name = f'basin_attributes'
    # print(f'    adding new cols {new_cols}.')
    # add the new columns to the table
    
    for col in new_cols:
        if 'land_use' in col:
            dtype = 'INT'
        elif 'FLAG' in col:
            dtype = 'INT'
        elif 'daymet' in col:
            dtype = 'INT'
        else:
            dtype = 'FLOAT'

        cur.execute(f'ALTER TABLE {schema_name}.{table_name} ADD COLUMN IF NOT EXISTS {col} {dtype};')

    conn.commit()


def alter_column_names(schema_name, old_cols, new_cols):

    table_name = f'basin_attributes'
    for i in range(len(old_cols)):
        print(f'    renaming {old_cols[i]} to {new_cols[i]}')
        cur.execute(f'ALTER TABLE {schema_name}.{table_name} RENAME COLUMN {old_cols[i]} TO {new_cols[i]};')

    conn.commit()
    

def reformat_nalcms_result(result):
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
    for label, group in land_use_groups.items():
        # Check if all numbers in the group exist as columns in the DataFrame
        existing_columns = [col for col in group if col in df.columns]
        if existing_columns:
            grouped_df[label] = df[existing_columns].sum(axis=1, skipna=True)
        else:
            grouped_df[label] = 0

    return grouped_df


def retrieve_unprocessed_ids(n_ids, check_cols):
    id_query = f"""SELECT id FROM basins_schema.basin_attributes """
    n = 0
    for c in check_cols:
        if n == 0:
            id_query += f"WHERE ({c} IS NULL OR {c} != {c}) "
            n = 1
        else:
            id_query += f"OR ({c} IS NULL OR {c} != {c}) "
    id_query += f"LIMIT {n_ids};"

    cur.execute(id_query)
    res = cur.fetchall()
    return [e[0] for e in res]


def check_for_basin_geometry(id_list):
    # check if the basin geometry exists for each id
    q = f"""SELECT id FROM basins_schema.basin_attributes WHERE id IN ({','.join([str(e) for e in id_list])}) AND basin IS NULL;"""
    cur.execute(q)
    res = cur.fetchall()
    
    if len(res) > 0:
        print(f'    {len(res)}/{id_list} basins do not have a valid geometry.')
        print(f'    Try first running derive_basins.py to create basin geometries.')
        
    return res


def get_unprocessed_attribute_rows(columns, which_set, region=None):
    
    id_query = f"""
    SELECT id, region_code FROM basins_schema.basin_attributes 
    WHERE 
        ({columns[0]} IS NULL OR {columns[0]} != {columns[0]}) """
    for c in columns[1:]:
        id_query += f"OR ({c} IS NULL OR {c} != {c}) "
    
    if region is not None:
        id_query += f"AND (region_code = '{region}') "
    
    id_query += f"ORDER by id ASC;"

    cur.execute(id_query)
    results = cur.fetchall()
    df = pd.DataFrame(results, columns=['id', 'region_code'])
    # groups = df.groupby('region_code').count()
    print(f'Number of unprocessed {which_set} rows: {len(df)}')
    # print(groups)
    return df['id'].values


def create_raster_table(schema_name, raster_fpath, table_name):
    raster_table_check = f"""
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = '{schema_name}'
        AND table_name = '{table_name}'
    );"""
    
    exists = basic_query(raster_table_check)
    
    print(f'Raster table {table_name} exists: {exists[0]}')
    
    db_password = conn_params['password']
    db_name = conn_params['dbname']
    
    # use the following command to add a raster table to the database:
    # https://postgis.net/docs/using_raster_dataman.html
    # in the example below, modify the raster_file.tif to match the full path
    # raster2pgsql -I -C -M -F -s 3005 -t auto raster_file.tif basins_schema.nalcms_2020_table | psql -d basins
    if not exists[0]:
        print(f'Adding {raster_fpath.split("/")[-1]} raster2pgsql command')
        command = f"raster2pgsql -d -e -I -C -Y 1000 -M -s 3005 -t 100x100 -P {raster_fpath} {schema_name}.{table_name} | PGPASSWORD={db_password} psql -U postgres -d {db_name} -h localhost -p 5432"    
        print(command)
        os.system(command)
    
    # if/when something messes up in this step, 
    # drop all columns that contain the substring "land_use_<year>"
    # query = f"ALTER TABLE basins_schema.nalcms_2015 DROP COLUMN IF EXISTS land_use_{year}*;"


def add_vector_layer_to_db(schema_name, vector_fpath, table_name):
    """
    Adds a vector file as a table to the database after checking if the table already exists.  
    
    See the following discussions:
    https://gis.stackexchange.com/questions/195172/import-a-shapefile-to-postgis-with-ogr2ogr-gives-unable-to-open-datasource
    """
    table_check = f"""
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = '{schema_name}'
        AND table_name = '{table_name}'
    );"""
    
    exists = basic_query(table_check)
    
    print(f'Table {table_name} exists: {exists[0]}')
    db_user = conn_params['user']
    db_password = conn_params['password']
    db_name = conn_params['dbname']
    if not exists[0]:
        print(f'Adding {vector_fpath.split("/")[-1]} using ogr2ogr command')
        command = f'ogr2ogr -f "PostgreSQL" PG:"host=localhost user={db_user} password={db_password} dbname={db_name}" {vector_fpath} -nln basins_schema.{table_name} -lco GEOMETRY_NAME=geometry'
        print(command)
        os.system(command)

    # check glhymps geometry and make valid
    valid_geoms = f"""
    UPDATE basins_schema.glhymps
    SET geometry = ST_MakeValid(geometry)
    WHERE NOT ST_IsValid(geometry);
    """
    basic_table_change(valid_geoms)


def update_database(new_data, schema_name, table_name, column_suffix=""):
    
    # update the database with the new land use data
    # ids = tuple([int(e) for e in new_data['id'].values])
    data_tuples = list(new_data.itertuples(index=False, name=None))
    
    cols = new_data.columns.tolist()
    set_string = ', '.join([f"{e}{column_suffix} = data.v{j}" for e,j in zip(cols[1:], range(1,len(cols[1:])+1))])
    v_string = ', '.join([f"v{e}" for e in range(1,len(cols[1:])+1)])
    
    query = f"""
    UPDATE {schema_name}.{table_name} AS basin_tab
        SET {set_string}
        FROM (VALUES %s) AS data(id, {v_string})
    WHERE basin_tab.id = data.id;
    """
    t0 = time()    
    with warnings.catch_warnings():
        # ignore warning for non-SQLAlchemy Connecton
        # see github.com/pandas-dev/pandas/issues/45660
        warnings.simplefilter('ignore', UserWarning)
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                extras.execute_values(cur, query, data_tuples)
                # commit the changes
                conn.commit()
    t1 = time()
    ut = len(data_tuples) / (t1-t0)
    # print(f'    {t1-t0:.1f}s to update {len(data_tuples)} polygons ({ut:.1f}/second)')


def clip_dem_values_by_polygon(inputs):

    t0 = time()
    
    id_list, schema_name, raster_table, basin_geom_table = inputs
    
    if len(id_list) == 0:
        return gpd.GeoDataFrame()
    
    ids = ','.join([str(int(e)) for e in id_list])

    q = f"""
    WITH pixel_matrix AS (
        SELECT
            {schema_name}.{basin_geom_table}.id AS basin_id,
            (ST_PixelAsCentroids({raster_table}.rast)).val AS pixel_value
        FROM
            {schema_name}.{raster_table},
            {schema_name}.{basin_geom_table}
        WHERE
            {schema_name}.{basin_geom_table}.id IN ({ids})
            AND ST_Intersects({raster_table}.rast, {schema_name}.{basin_geom_table}.basin)
    )
    SELECT
        basin_id,
        ARRAY_AGG(pixel_value) AS pixel_matrix
    FROM
        pixel_matrix
    GROUP BY
        basin_id;
    """
    # Execute the query in parallel
    # create a new connection for each process
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            # Fetch the results
            results = cur.fetchall()

    # print(f'  {len(results)} results returned.')
    data = [{'id': entry[0], **{value[0]: value[1] for value in entry[1]}} for entry in results]
    df = pd.DataFrame.from_dict(data)
    df = df.fillna(0).astype(int)
    df.set_index('id', inplace=True)
            
    return df


def clip_raster_values_by_polygon(raster_table, basin_geom_table, id_list):
    raster_values = ','.join([str(e) for e in list(range(1, 20))])
    ids = ','.join([str(int(e)) for e in id_list])
    t0 = time()
    
    if len(id_list) == 0:
        print('No ids found in the list.')
        return gpd.GeoDataFrame()
    
    q = f"""
    WITH selected_basins AS (
        SELECT id, basin 
        FROM {basin_geom_table}
        WHERE id IN ({ids})
    ), clipped_rasters AS (
        SELECT 
            b.id,
            ST_Clip(r.rast, 1, b.basin, TRUE) AS rast_clipped
        FROM 
            {raster_table} r
        JOIN selected_basins b ON ST_Intersects(r.rast, b.basin)
    ), raster_values AS (
        SELECT 
           clipped_rasters.id, vals.val
        FROM 
            clipped_rasters,
            LATERAL ST_PixelAsPoints(clipped_rasters.rast_clipped) AS vals
    ), value_counts AS (
        SELECT
            id,
            raster_values.val as pixel_value,
            COUNT(*) as count
        FROM
            raster_values
        GROUP BY
            id, raster_values.val
    ), aggregated_data AS (
        SELECT
            id,
            value_list.value,
            COALESCE(value_counts.count, 0) as count,
            SUM(value_counts.count) OVER (PARTITION BY id) AS total_count
        FROM (
            SELECT DISTINCT value 
            FROM unnest(ARRAY[{raster_values}]) AS value 
    ) AS value_list
        LEFT JOIN value_counts ON value_list.value = value_counts.pixel_value
    )
    SELECT
        id,
        jsonb_agg(jsonb_build_array(value, ROUND(count::float / total_count * 100)::int))
    FROM 
        aggregated_data
    GROUP BY
        id;
    """
    # create a new connection for each process
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            # Fetch the results
            results = cur.fetchall()
            # print(f'  {len(results)} results returned.')
    
    df = reformat_nalcms_result(results)
    return df


def reformat_daymet_result(results, param):
    # the result is a lis of tuples, where the first element is the basin id
    # and the second is the array of values associated with the bid
    # format a dataframe with the parameter name as the column name
    # and bid as the id column, where the mean of bid values is in the param column
    bids, mean_vals, n_samples = [], [], []
    for bid, values in results:
        bids.append(bid)
        if len(values) == 0:
            print(f'    No values found for basin {bid}')
            mean_vals.append(0)
            n_samples.append(0)
        else:
            mean_vals.append(round(np.mean(values), 0))
            n_samples.append(len(values))

    df = pd.DataFrame()
    df['id'] = bids
    df[param] = mean_vals
    df['daymet_sample_size'] = n_samples
    if 'dry_days' in param:
        df[param] = (df[param] * 100.0).round(0)
    df[param] = df[param].astype(int)
    return df


def clip_daymet_values_by_polygon(raster_table, basin_geom_table, id_list, table_name):
    
    ids = ','.join([str(int(e)) for e in id_list])
    daymet_param = '_'.join(table_name.split('_')[1:])
    if len(id_list) == 0:
        print('No ids found in the list.')
        return pd.DataFrame()
    
    q = f"""
    WITH pixel_matrix AS (
        SELECT
            {basin_geom_table}.id AS basin_id,
            (ST_PixelAsCentroids({raster_table}.rast)).val AS pixel_value
        FROM
            {raster_table},
            {basin_geom_table}
        WHERE
            {basin_geom_table}.id IN ({ids})
            AND ST_Intersects({raster_table}.rast, {basin_geom_table}.basin)
    )
    SELECT
        basin_id,
        ARRAY_AGG(pixel_value) AS pixel_matrix
    FROM
        pixel_matrix
    GROUP BY
        basin_id;
    """
    # create a new connection for each process
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            # Fetch the results
            results = cur.fetchall()
    
    df = reformat_daymet_result(results, daymet_param)
    return df


def process_raster(inputs):
    raster_set, id_list, schema_name, table_name, column_suffix = inputs
    
    t1 = time()
    times = []
    
    raster_table = f'{schema_name}.{table_name}'
    basin_geom_table = f'{schema_name}.basin_attributes'
    if table_name.startswith('nalcms'):
        df = clip_raster_values_by_polygon(raster_table, basin_geom_table, id_list)
    elif table_name.startswith('daymet'):
        df = clip_daymet_values_by_polygon(raster_table, basin_geom_table, id_list, table_name)
    t2 = time()
    times.append(t2-t1)
    ut =  len(id_list) / sum(times)
    # sort by id
    df.sort_index(inplace=True)
    df = df.reset_index(drop=True)
    update_database(df, schema_name, 'basin_attributes', column_suffix=column_suffix)
    print(f'    Average time to process {len(id_list)} basins for {raster_set} daymet data: {ut:.2f}basins/s')
    # return df
    return len(id_list)


def averaging_functions(group):
    # try:
    #     unique_ids = list(set(group['id'].values))
    # except Exception as ex:
    #     for c in group.columns:
    #         print(group[[c]].head())
    #     raise ex

    # if len(unique_ids) > 1:
    #     raise Exception('More than one ID found in the group', unique_ids)
    # else:
    #     obj_id = unique_ids[0]

    permeability_col = 'logk_ice_x'
    porosity_col = 'porosity_x'
    original_basin_area = list(set(group['area'].values))
        
    cols = ['id', 'porosity_x100', 'logk_ice_x100', 'k_stdev_x100', 'permafrost_FLAG', 'soil_area_sum',  'area', 'soil_FLAG']
    dtypes = ['int', 'int', 'int', 'int', 'int', 'float', 'float', 'int']
    type_dict = dict(zip(cols, dtypes))

    total_area = round(group['soil_polygon_areas'].sum(), 2)

    group = group[group['soil_polygon_areas'] > 0]
    if group.empty:
        data = [[obj_id, None, None, None, 1, None, original_basin_area[0]]]
        df = pd.DataFrame(data, columns=cols)
        return df.astype(type_dict)
    
    group['frac_areas'] = group['soil_polygon_areas'] / group['soil_polygon_areas'].sum()
    avg_porosity = np.average(group[porosity_col], weights=group['frac_areas'])
    # drop any rows with 0 permeability
    # group = group[group[permeability_col] > 0]
    # don't take the log because the values are already log transformed
    geom_avg_permeability = np.average(group[permeability_col], weights=group['frac_areas'])
    perm_stdev = np.average(group['k_stdev_x1'], weights=group['frac_areas'])
    permafrost_flag = group['prmfrst'].values.any()

    if len(original_basin_area) > 1:
        raise Exception('More than one basin area found in the group', original_basin_area)

    data = [[obj_id, avg_porosity, geom_avg_permeability, perm_stdev, permafrost_flag, total_area, original_basin_area[0]]]
    df = pd.DataFrame(data, columns=cols)

    return df.astype(type_dict)
    

def get_basin_geometries(basin_ids):
    
    basin_id_str = ','.join([str(int(e)) for e in basin_ids])
    q = f"""
    SELECT id, basin 
    FROM basins_schema.basin_attributes
    WHERE id IN ({basin_id_str});
    """
    t0 = time()
    with warnings.catch_warnings():
        # ignore warning for non-SQLAlchemy Connecton
        # see github.com/pandas-dev/pandas/issues/45660
        warnings.simplefilter('ignore', UserWarning)
        basin_df = gpd.read_postgis(q, conn, geom_col='basin')
    
    t1 = time()
    # print(f'    Basin geometry query time = {t1-t0:.1f}s for {len(basin_ids)} rows')

    return basin_df


def process_glhymps_by_basin(input_data):
    # print('Processing GLHYMPS data by basin...')
    bid, original_basin_area, clipped_soil_df = input_data
    t0 = time()

    # full column is logK_Ice_x100_INT
    permeability_col = 'logK_Ice_x'
    # full column is K_stdevx100
    stdev_permeability_col = 'K_stdev_x1'
    # full column is poroxity_x100_INT
    porosity_col = 'Porosity_x'
    # permafrost
    permafrost_col = 'Prmfrst'

    clipped_soil_df['area'] = clipped_soil_df['geometry'].area / 1E6
    tot_area = clipped_soil_df['area'].sum()

    # for c in clipped_soil_df.columns:
    #     print(clipped_soil_df[[c]])
    
    # print(f' original basin geometry area: {original_basin_area:.2f} km2 vs. {tot_area:.2f} km2')
    
    area_diff = 100 * abs(original_basin_area - tot_area) / original_basin_area
    geom_flag = 0
    if area_diff > 10.0:
        geom_flag = 1
    
    cols = ['id', 'porosity_x100', 'logk_ice_x100', 'k_stdev_x100', 'permafrost_FLAG', 'soil_FLAG']
    dtypes = ['int', 'int', 'int', 'int', 'int', 'int']
    type_dict = dict(zip(cols, dtypes))

    # if the total area is zero, return a row with null values
    if tot_area == 0:
        data = [[bid, 0, 0, 0, 0, 1]]
        df = pd.DataFrame(data, columns=cols)
        return df.astype(type_dict)

    clipped_soil_df['frac_areas'] = clipped_soil_df['area'] / clipped_soil_df['area'].sum()
    avg_porosity = np.average(clipped_soil_df[porosity_col], weights=clipped_soil_df['frac_areas'])
    # drop any rows with 0 permeability
    # group = group[group[permeability_col] > 0]
    # don't take the log because the values are already log transformed
    geom_avg_permeability = np.average(clipped_soil_df[permeability_col], weights=clipped_soil_df['frac_areas'])
    perm_stdev = np.average(clipped_soil_df[stdev_permeability_col], weights=clipped_soil_df['frac_areas'])
    permafrost_flag = clipped_soil_df[permafrost_col].values.any()
    
    data = [[bid, avg_porosity, geom_avg_permeability, perm_stdev, permafrost_flag, geom_flag]]
    df = pd.DataFrame(data, columns=cols)

    return df.astype(type_dict)


def fix_geoms(p):
    if p.is_valid:
        return p
    fixed_p = make_valid(p.buffer(0))
    if not fixed_p.is_valid:
        print(f'   Couldnt fix geometry')
    return fixed_p


def clip_soil_parallel(row):

    bid = row['id']
    b_area = row['basin'].area / 1E6
    try:
        soil_clip = gpd.clip(glhymps_gdf, mask=row['basin'])
    except Exception as ex:
        print(f'     Error processing basin {bid} ({b_area:.1f} km2).')
        print('      ...try to fix glhymps geometry...')
        # simplify the basin geometry to speed up overlay operation
        if b_area < 5:
            threshold =500
            buffer = 50
        else:
            threshold = 1000
            buffer = 1000

        simplified_polygon = make_valid(row['basin'].simplify(threshold).buffer(buffer))
        simple_df = gpd.GeoDataFrame({'geometry': [simplified_polygon]}, crs='EPSG:3005')
        intersected_vectors = gpd.sjoin(glhymps_gdf, simple_df, how='left', predicate='intersects')
        intersected_vectors['geometry'] = intersected_vectors['geometry'].apply(fix_geoms)
        soil_clip = gpd.clip(intersected_vectors, mask=row['basin'])
        
    return (bid, b_area, soil_clip)


def process_glhymps_by_basin_batch(ids):
    basin_batch = get_basin_geometries(ids)
    basin_batch['basin'] = basin_batch['basin'].apply(fix_geoms) 
    # crs = basin_batch.crs
    # batch_results = []
    with mp.Pool() as pool:
        batch_results = basin_batch.apply(clip_soil_parallel, axis=1).tolist()

    batch_inputs = [e for e in batch_results if e != None]

    with mp.Pool() as pool:
        results = pool.map(process_glhymps_by_basin, batch_inputs)

    output = pd.concat(results)
    
    return output


def intersect_glhymps_by_basin_polygon(soil_table, basin_geom_table, id_list):
            
    with warnings.catch_warnings():
        # ignore warning for non-SQLAlchemy Connecton
        # see github.com/pandas-dev/pandas/issues/45660
        warnings.simplefilter('ignore', UserWarning)
        with psycopg2.connect(**conn_params) as conn:
            
            with conn.cursor() as cur: 
                basin_ids = ','.join([str(int(e)) for e in id_list])

                # q = f"""
                # SELECT 
                #     t.id, t.area, t.basin,
                #     s.*,
                #     ST_Intersection(ST_Simplify(t.basin, 100), s.geometry) AS intersected_soil_geoms
                # FROM  
                #     {basin_geom_table} AS t
                # JOIN
                #     {soil_table} AS s ON ST_Intersects(t.basin, s.geometry)
                # WHERE
                #     t.id IN ({basin_ids})
                # GROUP BY
                #     t.id, 
                #     t.area,
                #     t.basin,
                #     s.geometry,
                #     s.logk_ice_x,
                #     s.porosity_x,
                #     s.k_stdev_x1, 
                #     s.prmfrst;
                # """

                print(q)
                # print(dfsa)
                gdf = gpd.read_postgis(q, conn, geom_col='intersected_soil_geoms')
                print(gdf)
                print(len(gdf))
                print(asdf)
                gdf['soil_polygon_areas'] = round(gdf['intersected_soil_geoms'].area / 1E6, 2)
                result = gdf.groupby('id').apply(averaging_functions)
                
                if not result.empty:
                    # add a flag if sum of intersected soil polygon areas 
                    # is less than 10% different from the original basin areas
                    result['soil_FLAG'] = (abs(result['area'] - result['soil_area_sum']) / result['area'] > 0.1).astype(int)
                    # only keep soil property related columns
                result.drop(['soil_area_sum', 'area'], axis=1, inplace=True)
        
    return result
        

def create_hysets_table(df, table_name):
    """Create a table and populate with data from the HYSETS dataset.
    The 'geometry' column represents the basin centroid.
    
    ########
    # This is an awful function and yes, I am ashamed.
    ########

    Args:
        df (pandas dataframe): dataframe containing the HYSETS data
    """
    # rename the Watershed_ID column to id
    print('Creating HYSETS table...')    
    # convert centroid geometry to 3005 
    df = df.to_crs(3005)
        
    # get columns and dtypes
    hysets_cols = ['Watershed_ID', 'Source', 'Name', 'Official_ID', 
                   'Centroid_Lat_deg_N', 'Centroid_Lon_deg_E', 'Drainage_Area_km2', 
                   'Drainage_Area_GSIM_km2', 'Flag_GSIM_boundaries', 
                   'Flag_Artificial_Boundaries', 'Elevation_m', 'Slope_deg', 
                   'Gravelius', 'Perimeter', 'Flag_Shape_Extraction', 'Aspect_deg', 
                   'Flag_Terrain_Extraction', 'Land_Use_Forest_frac', 'Land_Use_Grass_frac', 'Land_Use_Wetland_frac', 
                   'Land_Use_Water_frac', 'Land_Use_Urban_frac', 'Land_Use_Shrubs_frac', 'Land_Use_Crops_frac', 
                   'Land_Use_Snow_Ice_frac', 'Flag_Land_Use_Extraction', 
                   'Permeability_logk_m2', 'Porosity_frac', 'Flag_Subsoil_Extraction', 
                   ]
    # convert flag columns to boolean
    df = df[hysets_cols + ['geometry']]
    flag_cols = [e for e in sorted(df.columns) if e.lower().startswith('flag')]
    df[flag_cols] = df[flag_cols].astype(bool)
    df['Watershed_ID'] = df['Watershed_ID'].astype(int)
    
    # add 2010 suffix to land use columns to match bcub dataset
    land_use_cols = [e for e in hysets_cols if e.startswith('Land_Use')]
    
    # print(asdfsd)
    soil_cols = ['Permeability_logk_m2', 'Porosity_frac']    
    
    # drop rows with null values
    # in the future, we should probably fill these values by 
    # deriving the basin polygon and extracting the values
    # this could be part of a validation procedure
    df = df[~df[soil_cols + land_use_cols].isna().any(axis=1)]

    # remap soil columns to match the bcub dataset
    df.rename(columns={'Permeability_logk_m2': 'logk_ice_x100', 
                       'Porosity_frac': 'porosity_x100',}, inplace=True)
    
    
    
    # remap land use columns to match the bcub dataset
    df.rename(columns={e: f'{e}_2010' for e in land_use_cols}, inplace=True)
    
    land_use_cols = [e for e in df.columns if e.startswith('Land_Use')]
    df[land_use_cols] = 100 * df[land_use_cols]
    
    soil_cols = ['logk_ice_x100', 'porosity_x100']
    # multiply the soil values by 100 to match the format of the GLHYMPS data 
    # in the BCUB
    df[soil_cols] = 100 * df[soil_cols].round(1)
    

    # convert the centroid geometry to WKB
    df['centroid'] = df['geometry'].to_wkb(hex=True)
    
    cols = [e for e in df.columns if e not in ['geometry', 'centroid']]
    
    df = df[['centroid'] + cols]

    # get column dtypes 
    dtypes = [postgres_types[str(df[c].dtype)] for c in cols]
    
    # create the table query
    q = f'''CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
        centroid geometry(POINT, 3005),
        '''
    for c, d in zip(cols, dtypes):
        q += f'{c} {d},'
    q = q[:-1] + ');'
    with warnings.catch_warnings():
        # ignore warning for non-SQLAlchemy Connecton
        # see github.com/pandas-dev/pandas/issues/45660
        warnings.simplefilter('ignore', UserWarning)
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(q)                
                # convert the centroid geometry to WKB
                cur.execute(f"UPDATE {schema_name}.{table_name} SET centroid = ST_PointFromWKB(decode(centroid, 'hex'));")    

      
                tuples = list(df[['centroid'] + cols].itertuples(index=False, name=None))
                cols_str = ', '.join(['centroid'] + cols)
                query = f"""
                INSERT INTO {schema_name}.{table_name} ({cols_str})
                VALUES %s;
                """
                extras.execute_values(cur, query, tuples)
    
    return df
    

def get_id_list(n_ids, check_cols, use_test_ids=False):

    test_ids = [
        (770158, 2.04), (1089918, 8.49), (770161, 1.53), (770162, 1.20), (770165, 2.99),
        (1119937, 4.75), (334767, 2.09), (1089940, 53.39), (1131591, 1.82), (770169, 1.05)
        ]
    if use_test_ids:
        id_list = test_ids
    else:
        ta = time()
        id_list = retrieve_unprocessed_ids(n_ids, check_cols)

        tb = time()
        print(f'    Unprocessed ID query time = {tb-ta:.2f}s for {len(id_list)} rows')
    return id_list


def check_if_table_exists(table_name):
    """
    """
    query = f"""
    SELECT EXISTS (SELECT 1 FROM information_schema.tables 
    WHERE table_name = '{table_name}') AS table_exists;
    """
    cur.execute(query)
    return cur.fetchone()[0]


def check_if_column_exists(schema_name, table_name, column_name):
    """Check if a column exists in a table.

    Args:
        table_name (str): name of the table to check

    Returns:
        bool: true or false if the index exists
    """
    query = f"""
    SELECT EXISTS (
        SELECT 1
        FROM   information_schema.columns
        WHERE
            table_schema = '{schema_name}'
            AND table_name = '{table_name}'
            AND column_name = '{column_name}'
    ) as column_exists;
    """
    cur.execute(query)
    return cur.fetchone()[0]


def create_spatial_index(schema_name, table_name, geom, geom_idx_name):
    print(f'Creating spatial index for {schema_name}.{table_name}') 
    query = f'CREATE INDEX {geom_idx_name} ON {schema_name}.{table_name} USING GIST ({geom});'
    cur.execute(query)
    conn.commit()


def create_spatial_index_on_raster(schema, table_name, geom):
    """
    Add a PostGIS GIST index on the raster tiles in a table.
    https://gis.stackexchange.com/a/416332
    """
    
    idx_name = f'{table_name}_raster_idx'
    geom_idx_exists = check_spatial_index(table_name, idx_name, schema)
    
    if not geom_idx_exists:
        print(f'Creating spatial index {table_name}_geom_idx')
        query = f'CREATE INDEX {idx_name} ON {schema}.{table_name} USING GIST(ST_Envelope(rast));'
        cur.execute(query)
    else:
        print(f'    Spatial index {table_name}_geom_idx already exists')
        
    tile_idx_name = f'{table_name}_tile_idx'
    tile_extent_idx_exists = check_spatial_index(table_name, tile_idx_name, schema)
    tile_col_exists = check_if_column_exists(schema, table_name, 'tile_extent')

    if (not tile_col_exists):
        print(f'Creating spatial index for {schema}.{table_name}')
        query = f"""
        SELECT AddGeometryColumn ('{schema}','{table_name}','tile_extent', 3005, 'POLYGON', 2);
        UPDATE {schema}.{table_name}
        SET tile_extent = ST_Envelope(rast);
        """
        cur.execute(query)
        conn.commit()
        
    if not tile_extent_idx_exists:
        query = f"""
        CREATE INDEX {tile_idx_name} ON {schema}.{table_name} USING GIST(ST_Envelope(tile_extent));
        """
        cur.execute(query)
        conn.commit()
    else:
        print(f'    Spatial index {table_name}_tile_extent_idx already exists')
                
    print('Spatial index queries completed')


def check_spatial_index(table_name, idx_name, schema_name):
    """Check if a spatial index exists for the given table.  If not, create it.

    Args:
        table_name (str): name of the table to check

    Returns:
        bool: true or false if the index exists
    """
    query = f"""
    SELECT EXISTS (
        SELECT 1
        FROM   pg_indexes
        WHERE
            schemaname = '{schema_name}'
            AND tablename = '{table_name}'
            AND indexname = '{idx_name}'
    ) as index_exists;
    """ 
    cur.execute(query)
    return cur.fetchone()[0]

    
def table_setup(schema_name, daymet_params):
    """Check if DEM, NALCMS, GLHYMPS, and Daymet tables exist in the database.
    If not, add them and create spatial indices.

    Args:
        schema_name (string): name of the schema to add the tables to
        nalcms_year: year of the NALCMS data to add to the database.  
        There must be a corresponding file in the input data directory.
    """
    # add the DEM raster as a database table
    # dem_fpath = os.path.join(common_data, 'DEM/DEM_3005.tif')
    raster_fname = 'USGS_3DEP_DEM_mosaic_clipped_3005.vrt'
    raster_fpath = os.path.join(PROCESSED_DATA_DIR, raster_fname)
    table_name = f'usgs_3dep'
    dem_table_exists = check_if_table_exists(table_name)
    if not dem_table_exists:
        create_raster_table(schema_name, raster_fpath, table_name)
        create_spatial_index_on_raster(schema_name, table_name.lower(), 'rast')
    
    # add the land use raster to the database
    # use the clipped version.
    for yr in [2010, 2015, 2020]:
        table_name = f'nalcms_{yr}'
        nalcms_table_exists = check_if_table_exists(table_name)
        if not nalcms_table_exists:
            print(f'    processing NALCMS {yr}')
            nalcms_fpath = os.path.join(nalcms_dir, f'NA_NALCMS_landcover_{yr}_30m_3005.tif')            
            create_raster_table(schema_name, nalcms_fpath, table_name)
            create_spatial_index_on_raster(schema_name, table_name, 'rast')
        else:
            print(f'    NALCMS {yr} table already exists.')

    # add the climate use rasters to the database
    for param in daymet_params:
        table_name = f'daymet_{param}'
        daymet_param_table_exists = check_if_table_exists(table_name)
        if not daymet_param_table_exists:
            print(f'    processing daymet {param}')
            daymet_fpath = os.path.join(daymet_dir, f'{param}_mosaic_3005.tiff')
            create_raster_table(schema_name, daymet_fpath, table_name)
            create_spatial_index_on_raster(schema_name, table_name, 'rast')
    
    # add the GLHYMPS soil geometry to the database
    # use the clipped version.
    # table_name = f'glhymps'
    # glhymps_table_exists = check_if_table_exists(table_name)
    # if not glhymps_table_exists:
    #     glhymps_geom_col = 'geometry'
    #     add_vector_layer_to_db(schema_name, glhymps_fpath, table_name)
        
    # check for spatial index on the glhymps geometries
    # careful here, the ogr2ogr command will prepend wkb_ onto the geometry column name
    # geom_idx_name = f'{table_name}_geom_idx'
    # geom_index_exists = check_spatial_index(table_name, geom_idx_name, schema_name)
    # print(f'   Spatial index {geom_idx_name} exists: {geom_index_exists}')
    # if not geom_index_exists:
    #     create_spatial_index(schema_name, 'glhymps', glhymps_geom_col, geom_idx_name)

    # check if the basin attributes table has a spatial index
    basin_geom_idx_name = 'basin_geom_idx'
    basin_index_exists = check_spatial_index('basin_attributes', basin_geom_idx_name, schema_name)
    if not basin_index_exists:
        create_spatial_index(schema_name, 'basin_attributes', 'basin', basin_geom_idx_name)
            

def main():

    taa = time()
    # query the columns in the basin_attributes table
    terrain_attributes = ['slope_deg', 'aspect_deg', 'elevation_m', 'drainage_area_km2']
    land_cover_attributes = [f'{e}_{y}' for e in list(land_use_groups.keys()) for y in [2010, 2015, 2020]]
    soil_attributes = ['logk_ice_x100','k_stdev_x100', 'porosity_x100', 'permafrost_FLAG', 'soil_FLAG']
    # climate_attributes = ['prcp', 'tmax', 'tmin', 'vp', 'swe', 'srad']
    daymet_params = ['prcp', 'tmin', 'tmax', 'vp', 'swe', 'srad', 'prcp_dry_days', 'prcp_dry_duration']

    # check if all raster tables have been created
    table_setup(schema_name, daymet_params)

    new_cols = land_cover_attributes + soil_attributes + terrain_attributes + daymet_params + ['daymet_sample_size']
    add_table_columns(schema_name, new_cols)

    ## Add land cover data to the database
    region = None
    if process_nalcms:
        for raster_set in [2010, 2015, 2020]:
            lc_attrs = [e for e in land_cover_attributes if e.endswith(str(raster_set))]
            nalcms_ids = get_unprocessed_attribute_rows(lc_attrs, 'land cover', region)
            nalcms_ids = []
            table_name = f'nalcms_{raster_set}'
            if len(nalcms_ids) > 0:
                # basin_geometry_check = check_for_basin_geometry(nalcms_ids)
                print(f'    Processing {len(nalcms_ids)} land cover rows.')
                rows_per_batch = 20
                column_suffix = f'_{raster_set}'
                if len(nalcms_ids) > rows_per_batch:
                    n_iterations = int(len(nalcms_ids) / rows_per_batch) 
                    batches = np.array_split(nalcms_ids, n_iterations)
                    input_data = [(raster_set, id_list, schema_name, table_name, column_suffix) for id_list in batches]
                else:
                    n_iterations = 1
                    input_data = [(raster_set, nalcms_ids, schema_name, table_name, column_suffix)]
                        
                t0 = time()
                n_processed = 0
                with mp.Pool() as p:
                    n_processed = p.map(process_raster, input_data)
                t1 = time()
                t_hr = (t1-t0) / 3600
                print(f'    Processed {int(rows_per_batch)} batches in {t_hr:.2f}h ({sum(n_processed)} rows)')
        
    ## Add soil information to the database
    t0 = time()
    unprocessed_basin_ids = get_unprocessed_attribute_rows(soil_attributes, 'soil')
    # randomly shuffle the ids so we split up ordered clusters of large basins
    random.shuffle(unprocessed_basin_ids)
    t1 = time()
    print(f'    Unprocessed soil ID query time = {t1-t0:.1f}s for {len(unprocessed_basin_ids)} rows')

    if (len(unprocessed_basin_ids) > 0) & process_glhymps:
        # basin_geometry_check = check_for_basin_geometry(soil_ids)
        # soil_table = f'{schema_name}.glhymps'
        # basin_geom_table = f'{schema_name}.basin_attributes'
        # reduce the batch size if RAM is limited (1000 is too big for 128GB RAM)
        # 10 rows appears to use about (50GB of RAM -- verify??)
        soil_batch_size = 100
        if len(unprocessed_basin_ids) > soil_batch_size:
            
            n_batches = int(len(unprocessed_basin_ids) / soil_batch_size)
            batches = np.array_split(unprocessed_basin_ids, n_batches)

            # get a dataframe of basins by id
            # input_data = [(e['id'], e['basin']) for e in basin_batch.iterrows()]
            input_data = batches
            # input_data = [(soil_table, basin_geom_table, id_batch) for id_batch in batches]

        else:
            n_batches = 1
            input_data = [unprocessed_basin_ids]
        
        n = 0
        print(f'    Processing {n_batches} soil batches, {soil_batch_size} basins/batch')
        uts = []
        for id_batch in input_data:
            n += 1
            t1 = time()
            
            df = process_glhymps_by_basin_batch(id_batch)
            update_database(df, schema_name, 'basin_attributes')
            t2 = time()
            ut = len(id_batch) / (t2-t1)
            uts.append(ut)
            mean_ut = np.mean(uts)
            print(f'  Time to process batch {n}/{n_batches}: {t2-t1:.1f}s ({mean_ut:.1f} basins/s)')

    
    ## Add climate data to the database
    daymet_table = f'daymet'

    for param in daymet_params:
        daymet_ids = get_unprocessed_attribute_rows(daymet_params, param)
        
        if len(daymet_ids) > 0:
            basin_geometry_check = check_for_basin_geometry(daymet_ids)
            print(f'    Processing {len(daymet_ids)} climate index rows.')
            rows_per_batch = 50
            table_name = f'daymet_{param}'
            column_suffix = ''
            if len(daymet_ids) > rows_per_batch:
                n_iterations = int(len(daymet_ids) / rows_per_batch) 
                batches = np.array_split(daymet_ids, n_iterations)
                input_data = [(param, id_list, schema_name, table_name, column_suffix) for id_list in batches]
            else:
                n_iterations = 1
                input_data = [(param, daymet_ids, schema_name, table_name, column_suffix)]
                    
            t0 = time()
            n_processed = 0
            with mp.Pool() as p:
                n_processed = p.map(process_raster, input_data)
            t1 = time()
            t_hr = (t1-t0) / 3600
            print(f'    Processed {int(rows_per_batch)} daymet batches in {t_hr:.2f}h ({sum(n_processed)} rows)')

    
    # add HYSETS station data to the database
    hysets_table_name = 'hysets_basins'
    hysets_table_exists = check_if_table_exists(hysets_table_name)
    hysets_fpath = os.path.join(DATA_DIR, 'HYSETS_data/HYSETS_watershed_properties_BCUB.geojson')
    hysets_df = gpd.read_file(hysets_fpath)
    if not hysets_table_exists:        
        # remove punctuation marks from column data
        hysets_df['Name'] = hysets_df['Name'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        hysets_df = create_hysets_table(hysets_df, hysets_table_name)
    
    hysets_spatial_index = 'hysets_centroid_idx'
    # check if the index exists
    hysets_index_exists = check_spatial_index(hysets_table_name, hysets_spatial_index, schema_name)
    if not hysets_index_exists:
        # add a spatial index to the basin centroid points
        create_spatial_index(schema_name, hysets_table_name, 'centroid', hysets_spatial_index)

    tbb = time()
    print(f'Finished postgis database extension in {tbb - taa:.2f} seconds.')
    

with psycopg2.connect(**conn_params) as conn:
    cur = conn.cursor() 
    main()

cur.close()
conn.close()
