
import psycopg2
import psycopg2.extras as extras
import os
from time import time
import pandas as pd

import numpy as np
import multiprocessing as mp

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd


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
nalcms_dir = os.path.join(BASE_DIR, 'input_data/NALCMS/')
glhymps_dir = os.path.join(BASE_DIR, 'input_data/GLHYMPS/')

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
            id_query += f"AND ({c} IS NULL OR {c} != {c}) "
    id_query += f"LIMIT {n_ids};"
    
    cur.execute(id_query)
    res = cur.fetchall()
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
        
    cur.execute(id_query)
    res = cur.fetchone()
    print(f'Number of processed IDs: {res[0]}')


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


def update_database(new_data, schema_name, column_suffix=""):
    
    # update the database with the new land use data
    ids = tuple([int(e) for e in new_data['id'].values])
    data_tuples = list(new_data.itertuples(index=False, name=None))
    
    cols = new_data.columns.tolist()
    set_string = ', '.join([f"{e}{column_suffix} = data.v{j}" for e,j in zip(cols[1:], range(1,len(cols[1:])+1))])
    v_string = ', '.join([f"v{e}" for e in range(1,len(cols[1:])+1)])
    
    query = f"""
    UPDATE {schema_name}.basin_attributes AS basin_tab
        SET {set_string}
        FROM (VALUES %s) AS data(id, {v_string})
    WHERE basin_tab.id = data.id;
    """

    t0 = time()
    
    # with psycopg2.connect(**conn_params) as conn:
    cur = conn.cursor()

    extras.execute_values(cur, query, data_tuples)
    # commit the changes
    conn.commit()
    t1 = time()
    ut = len(data_tuples) / (t1-t0)
    print(f' {t1-t0:.1f}s for {len(data_tuples)} polygons ({ut:.1f}/second)')


def process_NALCMS(input_data):
    # year, id_list, lu_cols, schema_name = input_data
    year, id_list, schema_name = input_data
        
    # existing = get_existing_nalcms_vals('basins_schema.nalcms_2010', 'basins_schema.basin_attributes', id_list)
    # df = pd.DataFrame.from_dict(existing, orient='index', columns=['existing'])
    t1 = time()
    times = []
    
    raster_table = f'{schema_name}.nalcms_{year}'
    basin_geom_table = f'{schema_name}.basin_attributes'
    df = clip_nalcms_values_by_polygon(raster_table, basin_geom_table, id_list)
    
    t2 = time()
    times.append(t2-t1)
    avg_t = sum(times) / len(id_list)
    # sort by id
    df.sort_index(inplace=True)
    df = df.reset_index()
    print(f'    Average time to process {len(id_list)} basins for {year} land cover data: {avg_t:.2f}s')
    return df


def clip_nalcms_values_by_polygon(raster_table, basin_geom_table, id_list):
    raster_values = ','.join([str(e) for e in list(range(1, 20))])
    ids = ','.join([str(int(e)) for e in id_list])
    t0 = time()

    q = f"""
    WITH subquery AS (
        SELECT id, (ST_PixelAsCentroids({raster_table}.rast)).val AS pixel_value, COUNT(*) AS count
        FROM {raster_table}, {basin_geom_table}
        WHERE {basin_geom_table}.id IN ({ids})
        AND ST_Intersects({raster_table}.rast, {basin_geom_table}.basin)
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
    
    # Execute the query in parallel
    # create a new connection for each process
    with psycopg2.connect(**conn_params) as conn:
        cur = conn.cursor()
        cur.execute(q)

        # Fetch the results
        results = cur.fetchall()
    cur.close()
    conn.close()
    # print(f'  {len(results)} results returned.')
    df = reformat_result(results)
        
    return df


def averaging_functions(group):
    permeability_col = 'logk_ice_x'
    porosity_col = 'porosity_x'
        
    id = group['id'].values[0]
    group['frac_areas'] = group['soil_polygon_areas'] / group['soil_polygon_areas'].sum()
    avg_porosity = np.average(group[porosity_col], weights=group['frac_areas'])
    # drop any rows with 0 permeability
    # group = group[group['permeability_no_permafrost'] > 0]
    # don't take the log because the values are already log transformed
    geom_avg_permeability = np.average(group[permeability_col], weights=group['frac_areas'])
    perm_stdev = np.average(group['k_stdev_x1'], weights=group['frac_areas'])
    permafrost_flag = group['prmfrst'].values.any()
    
    original_basin_area = list(set(group['area'].values))
    if len(original_basin_area) > 1:
        raise Exception('More than one basin area found in the group', original_basin_area)
    soil_area_sum = round(group['soil_polygon_areas'].sum(), 2)
    cols = ['id', 'porosity_x100', 'logk_ice_x100', 'k_stdev_x100', 'permafrost_FLAG', 'soil_area_sum',  'area']
    dtypes = ['int', 'int', 'int', 'int', 'int', 'float', 'float']
    type_dict = dict(zip(cols, dtypes))
    data = [[id, avg_porosity, geom_avg_permeability, perm_stdev, permafrost_flag, soil_area_sum, original_basin_area[0]]]
    df = pd.DataFrame(data, columns=cols)
    return df.astype(type_dict)
    

def intersect_glhymps_by_basin_polygon(inputs):
    
    soil_table, basin_geom_table, id_list = inputs
        
    with psycopg2.connect(**conn_params) as conn:
        cur = conn.cursor() 
    
        basin_ids = ','.join([str(int(e)) for e in id_list])
            
        q = f"""
        SELECT 
            t.id, t.area, t.basin,
            s.geometry, s.logk_ice_x, s.porosity_x, s.k_stdev_x1, s.prmfrst,
            ST_Intersection(t.basin, s.geometry) AS intersected_soil_geoms
        FROM  
            {basin_geom_table} AS t
        JOIN
            {soil_table} AS s ON ST_Intersects(t.basin, s.geometry)
        WHERE
            t.id IN ({basin_ids})
        GROUP BY
            t.id, 
            t.area,
            t.basin,
            s.geometry,
            s.logk_ice_x,
            s.porosity_x,
            s.k_stdev_x1, 
            s.prmfrst;
        """
                
        # Execute the query in parallel
        gdf = gpd.read_postgis(q, conn, geom_col='intersected_soil_geoms')
        gdf['soil_polygon_areas'] = round(gdf['intersected_soil_geoms'].area / 1E6, 2)

        gdf = gdf.groupby('id').apply(averaging_functions)
        # add a flag if sum of intersected soil polygon areas 
        # is less than 1% different from the original basin areas
        gdf['soil_FLAG'] =  ((gdf['area'] - gdf['soil_area_sum']) / gdf['area'] > 0.01).astype(int)
        # drop 'soil_area_sum', 'area' columns
        gdf.drop(['soil_area_sum', 'area'], axis=1, inplace=True)
    cur.close()
    conn.close()
    return gdf
        

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
    print(query)
    print('')
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


def main():

    # parquet_dir = 'processed_data/parquet_basins'
    taa = time()
    # bc_bounds_file = os.path.join(common_data, 'BC_border/BC_study_region_polygon_R0.geojson')
    # t0 = time()
    # bc_bounds = gpd.read_file(bc_bounds_file)
    # t1 = time()
    # print(f'Loaded BC bounds in {t1 - t0:.2f} seconds.  crs={bc_bounds.crs}')
    nalcms_year = 2010
    
    # add the land use raster to the database
    # use the clipped version.
    nalcms_fpath = os.path.join(nalcms_dir, f'NA_NALCMS_landcover_{nalcms_year}_3005_clipped.tif')
    table_name = f'nalcms_{nalcms_year}'
    create_raster_table(schema_name, nalcms_fpath, table_name)
    create_spatial_index_on_raster(schema_name, 'nalcms_2010', 'rast')
    
    # add the GLHYMPS soil geometry to the database
    # use the clipped version.
    table_name = f'glhymps'
    geom_col = 'geometry'
    add_vector_layer_to_db(schema_name, glhymps_fpath, table_name)
    
    # check for spatial index on the glhymps geometries
    # careful here, the ogr2ogr command will prepend wkb_ onto the geometry column name
    geom_idx_name = f'{table_name}_geom_idx'
    geom_index_exists = check_spatial_index(table_name, geom_idx_name, schema_name)

    if not geom_index_exists:
        create_spatial_index(schema_name, 'glhymps', geom_col, geom_idx_name)
        
    # check if the basin attributes table has a spatial index
    basin_geom_idx_name = 'basin_geom_idx'
    basin_index_exists = check_spatial_index('basin_attributes', basin_geom_idx_name, schema_name)
    if not basin_index_exists:
        create_spatial_index(schema_name, 'basin_attributes', 'basin', basin_geom_idx_name)
        
    # query the columns in the basin_attributes table
    tables = ['basin_attributes', f'nalcms_{nalcms_year}', 'glhymps']
    for t in tables:
        col_query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{t}';"
        db_cols = basic_query(col_query)

    land_use_attributes = [f'{e}_{nalcms_year}' for e in list(land_use_groups.keys())]
    soil_attributes = ['logk_ice_x100','k_stdev_x100', 'porosity_x100', 'permafrost_FLAG', 'soil_FLAG']
    terrain_attributes = ['slope_deg', 'aspect_deg', 'elevation_m', 'drainage_area_km2']

    new_cols = land_use_attributes + soil_attributes + terrain_attributes
    add_table_columns(schema_name, new_cols)
    
    all_columns = land_use_attributes + terrain_attributes + soil_attributes
    
    n_iterations = 5
    n_ids_per_iteration = 100
    use_test_ids = False
    id_list = get_id_list(n_iterations * n_ids_per_iteration, all_columns, use_test_ids=use_test_ids)

    if use_test_ids:
        id_list = [e[0] for e in id_list]
        
    # count_processed_ids(land_use_cols)

    # batches = np.array_split(id_list, n_iterations)
    batches = np.array_split(id_list, n_iterations)
    input_data = [(nalcms_year, id_list, schema_name) for id_list in batches]
    
    p = mp.Pool()
    results = p.map(process_NALCMS, input_data)
    df = pd.concat(results, axis=0, ignore_index=True)
    p.close()
            
    print(f'   ...updating {len(df)} land cover rows')
    update_database(df, schema_name, column_suffix=f'_{nalcms_year}')
    
    t0 = time()
    
    p = mp.Pool()
    soil_table = f'{schema_name}.glhymps'
    basin_geom_table = f'{schema_name}.basin_attributes'        
    input_data = [(soil_table, basin_geom_table, id_batch) for id_batch in batches]
    
    # results = []
    # for ip in input_data:
    #     result = intersect_glhymps_by_basin_polygon(ip)
    #     results.append(result)

    results = p.map(intersect_glhymps_by_basin_polygon, input_data)
    
    df = pd.concat(results, axis=0, ignore_index=True)
    
    t1 = time()
    batch_time = (t1 - t0) / n_iterations
    ut = len(id_list) / (t1 - t0)
    print(f'    Time to process {n_iterations} batches: {t1-t0:.1f} ({batch_time:.0f} s/batch, {ut:.3f} basins/s)')
    p.close()
    
    print(f'   ...updating {len(df)} soil rows')
    # update the database
    update_database(df, schema_name)

    tbb = time()
    ut = len(id_list) / (tbb - taa)
    print(f'Finished postgis database modification in {tbb - taa:.2f} seconds ({ut:.2f}basins/s).')
    

with psycopg2.connect(**conn_params) as conn:
    cur = conn.cursor() 
    main()
cur.close()
conn.close()
