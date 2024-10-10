
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


daymet_dir = os.path.join(PROCESSED_DATA_DIR, 'DAYMET/')

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


def add_table_columns(schema_name, table_name, new_cols):
    
    # print(f'    adding new cols {new_cols}.')
    # add the new columns to the table
    
    for col in new_cols:
        if 'land_use' in col:
            dtype = 'INT'
        elif 'FLAG' in col:
            dtype = 'INT'
        elif ('prcp' in col) & ~('duration' in col):
            dtype = 'INT'
        elif 'sample_size' in col:
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


def get_unprocessed_attribute_rows(column, table, region=None):

    id_query = f"""
    SELECT watershed_id, official_id FROM basins_schema.{table} 
    WHERE ({column} IS NULL OR {column} != {column})"""
    
    if region is not None:
        id_query += f"AND (region_code = '{region}') "
    
    id_query += f"ORDER by watershed_id ASC;"
    cur.execute(id_query)
    results = cur.fetchall()
    df = pd.DataFrame(results, columns=['watershed_id', 'official_id'])
    return list(df['watershed_id'].values)


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
        FROM (VALUES %s) AS data(watershed_id, {v_string})
    WHERE basin_tab.watershed_id = data.watershed_id;
    """

    t0 = time()
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            extras.execute_values(cur, query, data_tuples)
            # commit the changes
            conn.commit()
            t1 = time()
            ut = len(data_tuples) / (t1-t0)
            print(f'    {t1-t0:.1f}s to update {len(data_tuples)} polygons ({ut:.1f}/second)')



def check_all_null_or_nan(values):
    """Check if all values in a list are null or nan."""
    return all(x is None or (isinstance(x, float) and np.isnan(x)) for x in values)


def nearest_pixel_query(bid, basin_geom_table, raster_table):
    q = f"""
    SELECT
        b.watershed_id,
        b.Drainage_Area_km2,
        ST_Distance(b.centroid, ST_Centroid(vals.geom)) AS dist_to_nearest_pixel,
        vals.val AS nearest_pixel_value
    FROM
        {basin_geom_table} b,
        LATERAL (
            SELECT (ST_PixelAsPoints(r.rast)).*
            FROM {raster_table} r
            ORDER BY b.centroid <-> (ST_PixelAsPoints(r.rast)).geom LIMIT 1
        ) AS vals
    WHERE 
        b.watershed_id = {bid};
    """
    # create a new connection for each process
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            # Fetch the results
            results = cur.fetchall()
            return results[0]


def reformat_daymet_result(results, param, basin_geom_table, raster_table):
    # the result is a lis of tuples, where the first element is the basin id
    # and the second is the array of values associated with the bid
    # format a dataframe with the parameter name as the column name
    # and bid as the id column, where the mean of bid values is in the param column
    bids, mean_vals, areas, n_samples = [], [], [], []
    for bid, area, values in results:        
        # check if there are no values for the basin or if all values are null
        if (len(values) == 0) | check_all_null_or_nan(values):
            print(f'    No values found for basin {bid}: {area}km^2')
            # query the neraest pixel value for basins too small or irregularly
            # shaped to intersect a raster pixel
            bid, _, distance, nearest_px_val = nearest_pixel_query(bid, basin_geom_table, raster_table)
            if distance > 1000:
                print(f'    Nearest pixel is greater than raster resolution: {distance:.1f}m.')
                continue
            mean_vals.append(nearest_px_val)
            areas.append(area)
            n_samples.append(1)
            bids.append(bid)
        else:
            mean_vals.append(round(np.mean(values), 3))
            n_samples.append(len(values))
            areas.append(area)
            bids.append(bid)
    
    df = pd.DataFrame()
    df['watershed_id'] = bids
    df[param] = mean_vals
    # df['samples_size'] = n_samples
    # df['area'] = areas
    if ('prcp_freq' in param):
        # for frequency, convert to percentage and store as integer
        df[param] = (df[param] * 100.0).round(0)
    if param == 'prcp':
        df[param] = df[param].round(0).astype(int)
    df.set_index('watershed_id', inplace=True)
    return df


def clip_daymet_values_by_polygon(raster_table, basin_geom_table, id_list, param):
    
    ids = ','.join([str(int(e)) for e in id_list])
    if len(id_list) == 0:
        print('No ids found in the list.')
        return pd.DataFrame()
    
    # note in the ST_Clip function, the last argument is TRUE, which means
    # that the output gets cropped to the extent of the input polygon
    # the second argument is the band number, which is 1 for daymet rasters

    q = f"""
    WITH selected_basins AS (
        SELECT watershed_id, basin_geometry, Drainage_Area_km2
        FROM {basin_geom_table}
        WHERE watershed_id IN ({ids})
    ), clipped_rasters AS (
        SELECT 
            b.watershed_id,
            b.Drainage_Area_km2,
            ST_Clip(r.rast, 1, b.basin_geometry, TRUE) AS rast_clipped
        FROM 
            {raster_table} r
        JOIN selected_basins b ON ST_Intersects(r.rast, b.basin_geometry)
    ), raster_values AS (
        SELECT 
            c.watershed_id, c.Drainage_Area_km2, vals.val
        FROM 
            clipped_rasters c
            CROSS JOIN LATERAL ST_PixelAsPoints(c.rast_clipped) AS vals
    ) SELECT 
        sb.watershed_id, sb.Drainage_Area_km2, COALESCE(ARRAY_AGG(rv.val), ARRAY[]::int[]) AS pixel_values
    FROM 
        selected_basins sb
    LEFT JOIN raster_values rv ON sb.watershed_id = rv.watershed_id
    GROUP BY
        sb.watershed_id, sb.Drainage_Area_km2;
    """
    
    # create a new connection for each process
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            # Fetch the results
            results = cur.fetchall()
    
    df = reformat_daymet_result(results, param, basin_geom_table, raster_table)
    return df


def process_raster(inputs):
    raster_set, id_list, schema_name, table_name, polygon_table, column_suffix, param = inputs
    
    t1 = time()
    times = []
    
    raster_table = f'{schema_name}.{table_name}'
    basin_geom_table = f'{schema_name}.{polygon_table}'

    if table_name.startswith('daymet'):
        df = clip_daymet_values_by_polygon(raster_table, basin_geom_table, id_list, param)
    else:
        print(f'    Unknown raster table name: {table_name}')
    t2 = time()
    times.append(t2-t1)
    ut =  len(id_list) / sum(times)
    # sort by id
    df.sort_index(inplace=True)
    df = df.reset_index(drop=False) 
    return df


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


def fix_geoms(p):
    if p.is_valid:
        return p
    fixed_p = make_valid(p.buffer(0))
    if not fixed_p.is_valid:
        print(f'   Couldnt fix geometry')
    return fixed_p

      
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
    daymet_params = ['prcp', 'tmin', 'tmax', 'vp', 'swe', 'srad', 
                     'high_prcp_freq','low_prcp_freq', 'high_prcp_duration', 'low_prcp_duration']

    # check if all raster tables have been created
    # table_setup(schema_name, daymet_params)
    new_cols = daymet_params 
    add_table_columns(schema_name, 'hysets_basins', new_cols)

    t0 = time()
    ## Add climate data to the database
    for param in daymet_params:
        daymet_ids = get_unprocessed_attribute_rows(param, 'hysets_basins')
        print(f' Processing {len(daymet_ids)} Daymet {param} rows.')
        
        if len(daymet_ids) > 0:
            # basin_geometry_check = check_for_basin_geometry(daymet_ids)
            print(f'    Processing {len(daymet_ids)} climate index rows.')
            rows_per_batch = 5000
            table_name = f'daymet_{param}'
            column_suffix = ''
            if len(daymet_ids) > rows_per_batch:
                n_iterations = int(len(daymet_ids) / rows_per_batch) 
                batches = np.array_split(daymet_ids, n_iterations)
                input_data = [(param, id_list, schema_name, table_name, 'hysets_basins', column_suffix, param) for id_list in batches]
            else:
                n_iterations = 1
                input_data = [(param, daymet_ids, schema_name, table_name, 'hysets_basins', column_suffix, param)]
                    
            t0 = time()
            n_processed = 0
            with mp.Pool() as p:
                daymet_results = p.map(process_raster, input_data)
            
            daymet_df = pd.concat(daymet_results)

            update_database(daymet_df, schema_name, 'hysets_basins', column_suffix=column_suffix)
            t1 = time()
            t_hr = (t1-t0) / 3600
            print(f'    Processed {len(input_data)} daymet batches in {t_hr:.2f}h ({len(daymet_df)} rows)')


    tbb = time()
    print(f'Finished postgis database extension in {tbb - taa:.2f} seconds.')
    

with psycopg2.connect(**conn_params) as conn:
    cur = conn.cursor() 
    main()

cur.close()
conn.close()
