
import psycopg2
import psycopg2.extras as extras
import os
from time import time
import pandas as pd

import numpy as np
import multiprocessing as mp

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

glhymps_fpath = os.path.join(glhymps_dir, 'GLHYMPS_clipped_3005.gpkg')
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
        dtype = 'FLOAT'
        if 'land_use' in col:
            dtype = 'INT'

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


def add_geopackage_layer_to_db(schema_name, gpkg_fpath, table_name):
    """
    Adds a geopackage as a table to the database after checking if the table already exists.  
    
    See the following discussions:
    https://gis.stackexchange.com/questions/290582/uploading-geopackage-contents-to-postgresql
    
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
        print(f'Adding {gpkg_fpath.split("/")[-1]} using ogr2ogr command')
        command = f'ogr2ogr -f "PostgreSQL" PG:"host=localhost user={db_user} password={db_password} dbname={db_name}" -nlt PROMOTE_TO_MULTI {gpkg_fpath} -nln basins_schema.{table_name}'
        print(command)
        os.system(command)


def extend_postgis_db(schema_name, total_bounds, year):
    """_summary_

    Args:
        db_host (_type_): _description_
        db_name (_type_): _description_
        db_user (_type_): _description_
        db_password (_type_): _description_
        schema_name (_type_): _description_
        total_bounds (_type_): _description_
        year (_type_): _description_
    """
    # connect to the PostGIS database
    # query the database to get the existing columns
    existing_col_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'basin_attributes';"
    cur.execute(existing_col_query)
    existing_cols = [e[0] for e in cur.fetchall()]
    
    # get one geometry from the database to use for the 
    # raster extraction. all geometry columns are EPSG:3005
    geom_query = f"SELECT basin FROM {schema_name}.basin_attributes LIMIT 1;"
    test = gpd.read_postgis(geom_query, conn, geom_col='geom')
    print(test.crs)
    print(test)
    print(existing_cols)
    print(asdfsdf)

    all_land_use_cols = sorted(list(set([e for e in existing_cols if ('land_use_2010' in e)])))
    land_use_categories = [e.split('land_use_')[1][5:] for e in all_land_use_cols]

    # rename the existing land use cols to indicate which year
    # modified_cols = [f"land_use_2010_{e.split('land_use_')[1]}" for e in all_land_use_cols]
    # alter_column_names(schema_name, db_host, db_name, db_user, db_password, all_land_use_cols, modified_cols)
    
    # create new table column
    # print('    Starting column insert...')
    # create new columns corresponding to the land use data for a given year
    new_land_use_columns = [f"land_use_{year}_{e.split(f'land_use_{year}_')[1]}" for e in all_land_use_cols]
    print(new_land_use_columns)

    # polygon geometry column is in EPSG: 3005,
    # clipped NALCMS raster is also in EPSG: 3005
    # so we can use the ST_Intersects function to find the land use
    # for each polygon

    # create a query to use the ST_Intersects function to mask the land use raster with all the polygons
    # and then aggregate the land use values for each polygon
    # extract_land_use_values(schema_name, db_host, db_name, db_user, db_password, land_use_fpath)
    # print(asdf)
        
    # extras.execute_values(cur, query, tuples)
    # conn.commit()
    t6 = time()
    # print(f'    Inserted {len(tuples)} rows in {t6-t4:.2f} seconds.')
    # commit the changes
    print(f'    Changes committed.  Finished processing')
    print('-------------------------------------------------')
    print('')


def update_lulc_data(new_data, year, schema_name):
    
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



def process_NALCMS(input_data):
    year, id_list, lu_cols, schema_name = input_data
        
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
    # print(f'    Average time to process {len(id_list)} basins for {year} land cover data: {avg_t:.2f}s')
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
    cur.execute(q)

    # Fetch the results
    results = cur.fetchall()

    # print(f'  {len(results)} results returned.')
    df = reformat_result(results)
        
    return df


def averaging_functions(group):
    group['frac_areas'] = group['soil_polygon_areas'] / group['soil_polygon_areas'].sum()
    avg_porosity = round(np.average(group['porosity'], weights=group['frac_areas']), 3)
    # drop any rows with 0 permeability
    # group = group[group['permeability_no_permafrost'] > 0]
    # don't take the log because the values are already log transformed
    geom_avg_permeability = np.average(group['permeability_no_permafrost'], weights=group['frac_areas']).round(3)
    original_basin_areas = list(set(group['original_basin_area'].values))
    if len(original_basin_areas) > 1:
        raise Exception('More than one basin area found in the group', original_basin_areas)
    soil_area_sum = round(group['soil_polygon_areas'].sum(), 2)
    id = group['id'].values[0]
    return pd.Series([id, avg_porosity, geom_avg_permeability, soil_area_sum, original_basin_areas[0]], index=['id', 'porosity', 'permeability_no_permafrost', 'soil_area_sum', 'original_basin_area'])
    

def intersect_glhymps_by_basin_polygon(inputs):
    
    soil_table, basin_geom_table, id_list = inputs
    
    with psycopg2.connect(**conn_params) as conn:
        cur = conn.cursor() 
    
        basin_ids = ','.join([str(int(e)) for e in id_list])
            
        q = f"""
        WITH TransformedBasins AS (
            SELECT 
                id,
                ST_Area(b.basin) AS original_basin_area,
                ST_Transform(b.basin, 4326) AS transformed_basin            
            FROM 
                {basin_geom_table} AS b
            WHERE
                b.id IN ({basin_ids})
        )
        SELECT 
            t.id, -- Assuming your soil_table has an 'id' column to identify each record.
            t.original_basin_area,
            s.porosity,
            s.permeability_no_permafrost,
            s.permeability_permafrost,
            s.permeability_standard_deviation,
            ST_Transform(ST_Intersection(t.transformed_basin, s.geom), 3005) AS intersected_soil_geoms
        FROM  
            TransformedBasins AS t,
            {soil_table} AS s
        WHERE 
            ST_Intersects(t.transformed_basin, s.geom)
            AND NOT ST_IsEmpty(ST_Intersection(t.transformed_basin, s.geom))
        GROUP BY
            t.id, 
            t.transformed_basin,
            t.original_basin_area,
            s.geom, 
            s.porosity, 
            s.permeability_no_permafrost,
            s.permeability_permafrost,
            s.permeability_standard_deviation
        ORDER BY
            t.id ASC;
        """
        
        # Execute the query in parallel
        gdf = gpd.read_postgis(q, conn, geom_col='intersected_soil_geoms')
        gdf['soil_polygon_areas'] = round(gdf['intersected_soil_geoms'].area / 1E6, 2)
        gdf['original_basin_area'] = round(gdf['original_basin_area'] / 1E6, 2)

        gdf = gdf.groupby('id').apply(averaging_functions)
        # assert all basin areas are less than 1% different from the original basin areas
        assert all((gdf['original_basin_area'] - gdf['soil_area_sum']) / gdf['original_basin_area'] < 0.01)
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


def create_spatial_index(schema_name, table_name, geom):
    print(f'Creating spatial index for {schema_name}.{table_name}') 
    query = f'CREATE INDEX {table_name}_geom_idx ON {schema_name}.{table_name} USING GIST ({geom});'
    cur.execute(query)
    conn.commit()


def create_spatial_index_on_raster(schema, table_name, geom):
    """
    Add a PostGIS GIST index on the raster tiles in a table.
    https://gis.stackexchange.com/a/416332
    """
   
    geom_idx_exists = check_spatial_index(table_name, f'{table_name}_geom_idx', schema)
    
    if not geom_idx_exists:
        print(f'Creating spatial index {table_name}_geom_idx')
        query = f'CREATE INDEX {table_name}_geom_idx ON {schema}.{table_name} USING GIST(ST_Envelope(rast));'
        cur.execute(query)
    else:
        print(f'Spatial index {table_name}_geom_idx already exists')
        
    tile_extent_idx_exists = check_spatial_index(table_name, '{table_name}_tile_extent_idx', schema)
    
    tile_extent_exists = check_if_column_exists(schema, table_name, 'tile_extent')

    if (not tile_extent_idx_exists) & (not tile_extent_exists):
        print(f'Creating spatial index for {schema}.{table_name}')
        query = f"""
        SELECT AddGeometryColumn ('{schema}','{table_name}','tile_extent', 3005, 'POLYGON', 2);
        UPDATE {schema}.{table_name}
        SET tile_extent = ST_Envelope(rast);
        """
        cur.execute(query)
        conn.commit()
        query = f"""
        CREATE INDEX {table_name}_tile_extent_idx ON {schema}.{table_name} USING GIST(ST_Envelope(tile_extent));
        """
        cur.execute(query)
        conn.commit()
    else:
        print(f'Spatial index {table_name}_tile_extent_idx already exists')
                
    print('Spatial index queries completed')


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
    add_geopackage_layer_to_db(schema_name, glhymps_fpath, table_name)
    
    # check for spatial index on the glhymps geometries
    spatial_index_exists = check_spatial_index(table_name, 'glhymps', schema_name)
    geom_index_exists = check_spatial_index(table_name, 'glhymps_geom_idx', schema_name)

    if (not spatial_index_exists) & (not geom_index_exists):
        create_spatial_index(schema_name, 'glhymps', 'geom')
        
    # query the columns in the basin_attributes table
    tables = ['basin_attributes', f'nalcms_{nalcms_year}', 'glhymps']
    for t in tables:
        col_query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{t}';"
        db_cols = basic_query(col_query)

    land_use_attributes = [f'{e}_{nalcms_year}' for e in list(land_use_groups.keys())]
    # soil_columns = ['porosity', 'permeability_no_permafrost', 'permeability_permafrost']
    soil_attributes = ['porosity_frac', 'permeability_logk_m2']
    terrain_attributes = ['slope_deg', 'aspect_deg', 'elevation_m', 'drainage_area_km2']

    new_cols = land_use_attributes + soil_attributes + terrain_attributes
    add_table_columns(schema_name, new_cols)
    
    all_columns = land_use_attributes + terrain_attributes + soil_attributes
    
    n_iterations = 50
    n_ids_per_iteration = 10
    use_test_ids = False
    id_list = get_id_list(n_iterations * n_ids_per_iteration, all_columns, use_test_ids=use_test_ids)
    
    if use_test_ids:
        id_list = [e[0] for e in id_list]
        
    # count_processed_ids(land_use_cols)

    # batches = np.array_split(id_list, n_iterations)
    batches = np.array_split(id_list, n_iterations)
    
    # p = mp.Pool()
    # results = p.map(process_NALCMS, input_data)
    # df = pd.concat(results, axis=0, ignore_index=True)
    # p.close()
            
    # print(f'   ...updating {len(df)} land cover rows')
    # update_lulc_data(df, nalcms_year)
    
    
    p = mp.Pool()
    soil_table = f'{schema_name}.glhymps'
    basin_geom_table = f'{schema_name}.basin_attributes'
    t0 = time()
    # for id_batch in batches:
    #     df = intersect_glhymps_by_basin_polygon(soil_table, basin_geom_table, id_batch)
    #     print(df)
    #     t1 = time()
    #     ut = len(id_batch) / (t1 - t0)
    #     print(f'    Time to process batch: {t1-t0:.1f} ({ut:.3f} basins/s)')
        
    input_data = [(soil_table, basin_geom_table, id_batch) for id_batch in batches]
    
    # for ip in input_data:
    #     result = intersect_glhymps_by_basin_polygon(ip)
    # print(asdf)
    results = p.map(intersect_glhymps_by_basin_polygon, input_data)
    
    df = pd.concat(results, axis=0, ignore_index=True)
    t1 = time()
    batch_time = (t1 - t0) / n_iterations
    ut = len(id_list) / (t1 - t0)
    print(f'    Time to process {n_iterations} batches: {t1-t0:.1f} ({batch_time:.0f} s/batch, {ut:.3f} basins/s)')
    p.close()
    
    print(f'   ...updating {len(df)} soil rows')
    # update_soil_data(df, nalcms_year)
    

    tbb = time()
    ut = len(id_list) / (tbb - taa)
    print(f'Finished postgis database modification in {tbb - taa:.2f} seconds ({ut:.2f}basins/s).')
    

with psycopg2.connect(**conn_params) as conn:
    cur = conn.cursor() 
    main()
cur.close()
conn.close()
