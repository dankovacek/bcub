
import psycopg2
import psycopg2.extras as extras
import os
import warnings

from time import time
import pandas as pd
import random
from shapely.validation import make_valid

import numpy as np
import multiprocessing as mp

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

# suppresss sqlalchemy warning when querying using geopandas
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# control the order of operations
# Each of these steps requires multiple days to process
process_nalcms = False
process_glhymps = False
process_daymet =True
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
daymet_dir = os.path.join(PROCESSED_DATA_DIR, 'DAYMET/')
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
land_cover_groups = {
    f'land_use_forest_frac': [1, 2, 3, 4, 5, 6], 
    f'land_use_shrubs_frac': [7, 8, 11],
    f'land_use_grass_frac': [9, 10, 12, 13, 16],
    f'land_use_wetland_frac': [14],
    f'land_use_crops_frac': [15],
    f'land_use_urban_frac': [17],
    f'land_use_water_frac': [18],
    f'land_use_snow_ice_frac': [19],
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
    glhymps_sindex = glhymps_gdf.sindex


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
    

def reformat_nalcms_result(result, year):
    """
    Reformat the query result into a dataframe with integer values representing
    the percentage of each land use class in the basin.
    Each row represents a basin, each column represents a land use class.
    """
    data = [{'id': entry[0], 
             'drainage_area_km2': entry[1], 
             'total_pixel_count': entry[2],
             **{e['value']:  e['count'] for e in entry[3]}} for entry in result]
    
    
    df = pd.DataFrame.from_dict(data)
    value_cells = [c for c in df.columns if c not in ['id', 'drainage_area_km2', 'total_pixel_count']]
    
    df[value_cells] = df[value_cells].fillna(0).astype(int)
    df.set_index('id', inplace=True)
    # ensure that the total clipped raster pixel count is equal to the sum of the land use class counts
    df['lulc_cell_count'] = df[value_cells].sum(axis=1)
    assert all(df['lulc_cell_count'] == df['total_pixel_count'])
    # compute an approximate clipped raster area based on the resolution of the NALCMS raster
    df['lulc_area'] = df['lulc_cell_count'] * 30.402068966776415 * 30.402068966776415 / 1E6
    df['land_use_FLAG'] = ((df['lulc_area'] - df['drainage_area_km2']).abs() / df['drainage_area_km2'] > 0.05).astype(int)

    # Group the columns and sum the values
    grouped_df = pd.DataFrame()
    for label, group in land_cover_groups.items():
        # Check if all numbers in the group exist as columns in the DataFrame
        existing_columns = [col for col in group if col in df.columns]
        if existing_columns:
            grouped_df[label] = df[existing_columns].sum(axis=1, skipna=True)
        else:
            grouped_df[label] = 0

    # convert to proportions
    grouped_df = (grouped_df.div(grouped_df.sum(axis=1), axis=0) * 100).round(0).astype(int)

    # assert that the indices are in exactly the same order
    assert all(grouped_df.index == df.index)
    #  add the land cover area check flag if desired
    # grouped_df[f'land_use_FLAG{year}'] = df['land_use_FLAG']
    
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


def get_zero_val_attribute_rows(column, region=None, limit=None):
    
    id_query = f"""
    SELECT id, region_code FROM basins_schema.basin_attributes 
    WHERE ({column} = 0 OR {column} != {column})"""
    # for c in columns[1:]:
    #     id_query += f"OR ({c} IS NULL OR {c} != {c}) "
    
    if region is not None:
        id_query += f"AND (region_code = '{region}') "
    
    if limit is not None:
        id_query += f"ORDER by id ASC "
        id_query += f"LIMIT {limit};"
    else:
        id_query += f"ORDER by id ASC;"
    
    cur.execute(id_query)
    results = cur.fetchall()
    df = pd.DataFrame(results, columns=['id', 'region_code'])
    return list(df['id'].values)

def get_unprocessed_attribute_rows(column, region=None, limit=None):
    
    id_query = f"""
    SELECT id, region_code FROM basins_schema.basin_attributes 
    WHERE ({column} IS NULL OR {column} != {column})"""
    # for c in columns[1:]:
    #     id_query += f"OR ({c} IS NULL OR {c} != {c}) "
    
    if region is not None:
        id_query += f"AND (region_code = '{region}') "
    
    if limit is not None:
        id_query += f"ORDER by id ASC "
        id_query += f"LIMIT {limit};"
    else:
        id_query += f"ORDER by id ASC;"
    
    cur.execute(id_query)
    results = cur.fetchall()
    df = pd.DataFrame(results, columns=['id', 'region_code'])
    return list(df['id'].values)


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
    # t0 = time()
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            try:
                extras.execute_values(cur, query, data_tuples)
                # commit the changes
                conn.commit()
            except Exception as e:
                print(f'Error updating database: {e}')
                print(data_tuples)
                print(query)
                raise Exception(e)           
            
            # t1 = time()
            # ut = len(data_tuples) / (t1-t0)
            # print(f'    {t1-t0:.1f}s to update {len(data_tuples)} polygons ({ut:.1f}/second)')


def clip_nalcms_raster_values_by_polygon(raster_table, basin_geom_table, id_list):
    # raster_values = ','.join([str(e) for e in list(range(1, 20))])
    ids = ','.join([str(int(e)) for e in id_list])
    # t0 = time()

    if len(id_list) == 0:
        print('No ids found in the list.')
        return gpd.GeoDataFrame()
    
    q = f"""
    WITH selected_basins AS (
        SELECT id, basin, drainage_area_km2
        FROM {basin_geom_table}
        WHERE id IN ({ids})
    ), clipped_rasters AS (
        SELECT 
            b.id, 
            b.drainage_area_km2,
            ST_Clip(r.rast, 1, b.basin, TRUE) AS rast_clipped
        FROM 
            {raster_table} r
        JOIN selected_basins b 
        ON ST_Intersects(ST_ConvexHull(r.rast), b.basin)
    ), value_counts AS (
        SELECT
            id,
            drainage_area_km2,
            vc.value,
            vc.count
        FROM
            clipped_rasters cr,
            LATERAL ST_ValueCount(cr.rast_clipped) AS vc(value, count)
    ), aggregated_counts AS (
        SELECT
            id,
            drainage_area_km2,
            value,
            sum(count) as total_count
        FROM
            value_counts
        GROUP BY
            id, drainage_area_km2, value
    )
    SELECT
        id,
        drainage_area_km2,
        SUM(total_count) AS total_pixel_count,
        jsonb_agg(jsonb_build_object('value', value, 'count', total_count)) AS value_counts
    FROM
        aggregated_counts
    GROUP BY
        id, drainage_area_km2;
    """
    
    # create a new connection for each process
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            # print(q)
            cur.execute(q)
            # Fetch the results
            results = cur.fetchall()
            # print(f'  {len(results)} results returned.')
    
    return results


def check_all_null_or_nan(values):
    """Check if all values in a list are null or nan."""
    return all(x is None or (isinstance(x, float) and np.isnan(x)) for x in values)


def nearest_pixel_query(bids, basin_geom_table, raster_table):
    bid_list_str = ', '.join(map(str, bids))

    q = f"""
    WITH centroids AS (
        SELECT id, centroid
        FROM {basin_geom_table}
        WHERE id IN ({bid_list_str})
    )
    SELECT
        c.id,
        ST_Value(r.rast, ST_ClosestPoint(r.rast::geometry, c.centroid)) AS nearest_pixel_value
    FROM
        centroids c
    JOIN LATERAL (
        SELECT r.rast
        FROM {raster_table} r
        WHERE ST_Value(r.rast, ST_ClosestPoint(r.rast::geometry, c.centroid)) IS NOT NULL
        AND ST_Value(r.rast, ST_ClosestPoint(r.rast::geometry, c.centroid)) <> 'NaN'::float
        ORDER BY c.centroid <-> r.rast::geometry
        LIMIT 1
    ) AS r ON true;
    """
    # create a new connection for each process
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            # Fetch the results
            results = cur.fetchall()
            return results


def reformat_daymet_result(results, param, basin_geom_table, raster_table):
    # the result is a lis of tuples, where the first element is the basin id
    # and the second is the array of values associated with the bid
    # format a dataframe with the parameter name as the column name
    # and bid as the id column, where the mean of bid values is in the param column
    bids, mean_vals, areas, n_samples = [], [], [], []
    missing_bids = []
    for bid, area, values in results:
        # check if there are no values for the basin or if all values are null
        if (len(values) == 0) | check_all_null_or_nan(values):
            # query the neraest pixel value for basins too small or irregularly
            # shaped to intersect a raster pixel
            missing_bids.append(bid)         
        else:
            mean_vals.append(round(np.mean(values), 3))
            # n_samples.append(len(values))
            bids.append(bid)
    
    if len(missing_bids) > 0:
        print(f'    No values returned for {len(missing_bids)} basins.  Reverting to nearest pixel value.')
        print(f'    This is generally caused by very small basins (~1 km2) \n    not covering an entire raster pixel (1km grid).')
        # find the nearest pixel value for basins that do not cover a raster pixel
        results = nearest_pixel_query(missing_bids, basin_geom_table, raster_table)
        checked_ids = [e[0] for e in results]
        closest_vals = [e[1] for e in results]

        bids += checked_ids
        mean_vals += closest_vals
    
    df = pd.DataFrame()
    df['id'] = bids
    df[param] = mean_vals

    # df['daymet_sample_size'] = n_samples
    # df['area'] = areas
    if ('prcp_freq' in param):
        # for frequency, convert to percentage and store as an integer
        df[param] = (df[param] * 100.0).round(0)
    if param == 'prcp':
        df[param] = df[param].round(0).astype(int)
    df.set_index('id', inplace=True)
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
        SELECT id, basin, drainage_area_km2 as area
        FROM {basin_geom_table}
        WHERE id IN ({ids})
    ), clipped_rasters AS (
        SELECT 
            b.id,
            b.area,
            ST_Clip(r.rast, 1, b.basin, TRUE) AS rast_clipped
        FROM
            {raster_table} r
        JOIN selected_basins b ON ST_Intersects(r.rast, b.basin)
    ), raster_values AS (
        SELECT 
            c.id, c.area, vals.val
        FROM 
            clipped_rasters c
            CROSS JOIN LATERAL ST_PixelAsPoints(c.rast_clipped, 1) AS vals
    ) SELECT 
        sb.id, sb.area, COALESCE(ARRAY_AGG(rv.val), ARRAY[]::double precision[]) AS pixel_values
    FROM 
        selected_basins sb
    LEFT JOIN raster_values rv ON sb.id = rv.id
    GROUP BY
        sb.id, sb.area;
    """
    # create a new connection for each process
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            # Fetch the results
            results = cur.fetchall()
    
    return results


def process_raster(inputs, which_set='nalcms'):
    raster_set, id_list, schema_name, table_name, column_suffix, param = inputs
    
    t1 = time()
    times = []
    
    raster_table = f'{schema_name}.{table_name}'
    basin_geom_table = f'{schema_name}.basin_attributes'
    if which_set == 'nalcms':
        t0 = time()
        nalcms_results = clip_nalcms_raster_values_by_polygon(raster_table, basin_geom_table, id_list)
        t1 = time()
        # print(f'    NALCMS query time = {t1-t0:.2f}s for {len(id_list)} rows')   
        df = reformat_nalcms_result(nalcms_results, column_suffix) 
        
    elif which_set == 'daymet':
        daymet_results = clip_daymet_values_by_polygon(raster_table, basin_geom_table, id_list, param)
        df = reformat_daymet_result(daymet_results, param, basin_geom_table, raster_table)
    else:
        print(f'    Unknown raster table name: {table_name}')
    
    df = df.reset_index(drop=False)
    return df


# def averaging_functions(group):
#     # try:
#     #     unique_ids = list(set(group['id'].values))
#     # except Exception as ex:
#     #     for c in group.columns:
#     #         print(group[[c]].head())
#     #     raise ex

#     # if len(unique_ids) > 1:
#     #     raise Exception('More than one ID found in the group', unique_ids)
#     # else:
#     #     obj_id = unique_ids[0]

#     permeability_col = 'logk_ice_x'
#     porosity_col = 'porosity_x'
#     original_basin_area = list(set(group['area'].values))
        
#     cols = ['id', 'porosity_x100', 'logk_ice_x100', 'k_stdev_x100', 'permafrost_FLAG', 'soil_area_sum',  'area', 'soil_FLAG']
#     dtypes = ['int', 'int', 'int', 'int', 'int', 'float', 'float', 'int']
#     type_dict = dict(zip(cols, dtypes))

#     total_area = round(group['soil_polygon_areas'].sum(), 2)

#     group = group[group['soil_polygon_areas'] > 0]
#     if group.empty:
#         data = [[obj_id, None, None, None, 1, None, original_basin_area[0]]]
#         df = pd.DataFrame(data, columns=cols)
#         return df.astype(type_dict)
    
#     group['frac_areas'] = group['soil_polygon_areas'] / group['soil_polygon_areas'].sum()
#     avg_porosity = np.average(group[porosity_col], weights=group['frac_areas'])
#     # drop any rows with 0 permeability
#     # group = group[group[permeability_col] > 0]
#     # don't take the log because the values are already log transformed
#     geom_avg_permeability = np.average(group[permeability_col], weights=group['frac_areas'])
#     perm_stdev = np.average(group['k_stdev_x1'], weights=group['frac_areas'])
#     permafrost_flag = group['prmfrst'].values.any()

#     if len(original_basin_area) > 1:
#         raise Exception('More than one basin area found in the group', original_basin_area)

#     data = [[obj_id, avg_porosity, geom_avg_permeability, perm_stdev, permafrost_flag, total_area, original_basin_area[0]]]
#     df = pd.DataFrame(data, columns=cols)

#     return df.astype(type_dict)
    

def get_basin_geometries(basin_ids):
    
    basin_id_str = ','.join([str(int(e)) for e in basin_ids])
    q = f"""
    SELECT id, drainage_area_km2, basin 
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


def process_glhymps_by_basin_batch(ids):

    id_string = ', '.join([str(e) for e in ids])
    
    q = f"""
    WITH filtered_basins AS (
        SELECT id, drainage_area_km2, basin
        FROM basins_schema.basin_attributes
        WHERE id IN ({id_string})
    )
    SELECT
        b.id AS id,
        b.drainage_area_km2 as drainage_area_km2,
        g.porosity_x as porosity_x100, -- note that the column name gets truncated
        g.logk_ice_x as logk_ice_x100, -- note that the column name gets truncated 
        g.k_stdev_x1 as k_stdev_x100,   -- note that the column name gets truncated
        g.prmfrst as permafrost_flag,
        ST_Intersection(b.basin, g.geometry) AS clipped_geom
    FROM
        filtered_basins AS b
    INNER JOIN
        basins_schema.glhymps AS g
    ON
        ST_Intersects(ST_ConvexHull(b.basin), g.geometry);
    """
    with psycopg2.connect(**conn_params) as conn:
        df = gpd.read_postgis(q, conn, geom_col='clipped_geom')
        df['area_km2'] = df['clipped_geom'].area / 1e6
        # remove any empty geometries
        df = df[df['area_km2'] > 0][[c for c in df.columns if c != 'clipped_geom']]

    # Function to compute weighted mean
    def weighted_mean(values, weights):
        return np.average(values, weights=weights)
    
    # batch_inputs = [e for e in batch_results if e != None]
    ####
    ## NOTE: since the permeability values are log transformed, the weighted mean is the geometric mean
    ####
    results = df.groupby('id').agg({
        # 'clipped_geom': 'first',  # This will not affect geometry in dissolve
        'porosity_x100': lambda x: weighted_mean(x, df.loc[x.index, 'area_km2']),
        'logk_ice_x100': lambda x: weighted_mean(x, df.loc[x.index, 'area_km2']),
        'k_stdev_x100': lambda x: weighted_mean(x, df.loc[x.index, 'area_km2']),
        'permafrost_flag': lambda x: np.any(x), # True if any are true
        'area_km2': lambda x: np.sum(x),
        'drainage_area_km2': 'first'
    })

    # create a soil flag if the area_km2 (clipped vector area) is > 5% different from the drainage_area_km2
    results['soil_flag'] = (abs(results['area_km2'] - results['drainage_area_km2']) / results['drainage_area_km2'] > 0.05).astype(int)
    results[['porosity_x100', 'logk_ice_x100', 'k_stdev_x100', 'permafrost_flag']] = results[['porosity_x100', 'logk_ice_x100', 'k_stdev_x100', 'permafrost_flag']].astype(int)

    output_cols = ['porosity_x100', 'logk_ice_x100', 'k_stdev_x100', 'permafrost_flag', 'soil_flag']
    results = results[output_cols]
    results.reset_index(drop=False, inplace=True)
    
    return results

       
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
    extension_query = """CREATE EXTENSION IF NOT EXISTS postgis_raster;"""
    cur.execute(extension_query)
    conn.commit()
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
    query = f'CREATE INDEX {geom_idx_name} ON {schema_name}.{table_name} USING GIST({geom});'
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
    table_name = f'usgs_3dep'
    dem_table_exists = check_if_table_exists(table_name)
    if not dem_table_exists:
        raster_fname = 'USGS_3DEP_DEM_mosaic_clipped_3005.vrt'
        raster_fpath = os.path.join(PROCESSED_DATA_DIR, raster_fname)
        create_raster_table(schema_name, raster_fpath, table_name)
        create_spatial_index_on_raster(schema_name, table_name.lower(), 'rast')
    
    # add the land use raster to the database
    # use the clipped version.
    for yr in [2010, 2015, 2020]:
        table_name = f'nalcms_{yr}'
        nalcms_table_exists = check_if_table_exists(table_name)
        if not nalcms_table_exists:
            print(f'    processing NALCMS {yr}')
            nalcms_fpath = os.path.join(nalcms_dir, f'NA_NALCMS_landcover_{yr}_3005_clipped.tif')            
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
    
    
        # check if the basin attributes table has a spatial index
    basin_geom_idx_name = 'basin_geom_idx'
    basin_index_exists = check_spatial_index('basin_attributes', basin_geom_idx_name, schema_name)
    if not basin_index_exists:
        create_spatial_index(schema_name, 'basin_attributes', 'basin', basin_geom_idx_name)

    
    # add the GLHYMPS soil geometry to the database
    # use the clipped version.
    table_name = f'glhymps'
    glhymps_table_exists = check_if_table_exists(table_name)
    if not glhymps_table_exists:
        add_vector_layer_to_db(schema_name, glhymps_fpath, table_name)
        
    # check for spatial index on the glhymps geometries
    # careful here, the ogr2ogr command will prepend wkb_ onto the geometry column name
    geom_idx_name = f'{table_name}_geom_idx'
    geom_index_exists = check_spatial_index(table_name, geom_idx_name, schema_name)
    print(f'   Spatial index {geom_idx_name} exists: {geom_index_exists}')
    if not geom_index_exists:
        glhymps_geom_col = 'geometry'
        create_spatial_index(schema_name, 'glhymps', glhymps_geom_col, geom_idx_name)


def process_nalcms_batch(input_batch):
    lc_results = process_raster(input_batch, 'nalcms')
    year = input_batch[4]
    update_database(lc_results, schema_name, 'basin_attributes', column_suffix=year)
    return True


def clip_soil_parallel(row):
    # bid, basin, glhymps_gdf = data
    bid = row['id']
    b_area = row['drainage_area_km2']
    first_clip = gpd.clip(row['clipped_glhymps'], mask=row['basin'].bounds)
    # filter out very small polygons  (0.5% of smallest area)
    first_clip = first_clip[first_clip.geometry.area > 1e6 * 0.005]
    basin = row['basin']
    if b_area > 10000:
        # simplify by a small margin if the basin is very large
        basin = make_valid(basin.simplify(100))
    try:
        soil_clip = gpd.clip(first_clip, mask=basin)
        soil_clip = soil_clip[soil_clip.geometry.area > 1e6 * 0.005]
        
    except Exception as ex:        
        first_clip.geometry = first_clip.geometry.apply(lambda x: fix_geoms(x))
        soil_clip = gpd.clip(first_clip, mask=basin)
        
    
    result = process_glhymps_by_basin((bid, b_area, soil_clip))
    # return (bid, b_area, soil_clip)
    return result


def process_glhymps_by_basin_clipping(df):

    t0 = time()
    clips = []
    for row_data in df.itertuples():
        row = list(row_data)
        bid = row[1]
        da = row[2]
        basin = row[3]
        # clips = [dfglhymps_gdf)]
        g = gpd.GeoDataFrame({'geometry': [basin]}, crs=df.crs)
        # bbox = g.total_bounds
        clip = gpd.clip(glhymps_gdf.copy(), g)
        # conver the clip to a geodataframe
        clip['bid'] = bid
        # clip['clip_area'] = clip.area.sum() / 1e6
        clip['basin_drainage_area_km2'] = da  
        clips.append(clip)
    
    all_clips = gpd.GeoDataFrame(pd.concat(clips, ignore_index=True), crs=df.crs)
    all_clips = all_clips.explode()
    all_clips['clipped_area_km2'] = all_clips.geometry.area / 1e6

    # shapefile columns are truncated to 10 characters, need to rename
    name_mapper = {'Porosity_x': 'porosity_x100', 'logK_Ice_x': 'logk_ice_x100', 'K_stdev_x1': 'k_stdev_x100'}
    all_clips.rename(columns=name_mapper, inplace=True)
    
    t1 = time()
    print(f'    Clipping time = {t1-t0:.1f}s for {len(df)} rows')

    # Function to compute weighted mean
    def weighted_mean(values, weights):
        return np.average(values, weights=weights)

    results = df.groupby('id').agg({
        # 'clipped_geom': 'first',  # This will not affect geometry in dissolve
        'porosity_x100': lambda x: weighted_mean(x, df.loc[x.index, 'clipped_area_km2']),
        'logk_ice_x100': lambda x: weighted_mean(x, df.loc[x.index, 'clipped_area_km2']),
        'k_stdev_x100': lambda x: weighted_mean(x, df.loc[x.index, 'clipped_area_km2']),
        'permafrost_flag': lambda x: np.any(x), # True if any are true
        'area_km2': lambda x: np.sum(x),
        'drainage_area_km2': 'first'
    })

    t2 = time()

    # create a soil flag if the area_km2 (clipped vector area) is > 5% different from the drainage_area_km2
    results['soil_flag'] = (abs(results['area_km2'] - results['drainage_area_km2']) / results['drainage_area_km2'] > 0.05).astype(int)
    results[['porosity_x100', 'logk_ice_x100', 'k_stdev_x100', 'permafrost_flag']] = results[['porosity_x100', 'logk_ice_x100', 'k_stdev_x100', 'permafrost_flag']].astype(int)

    output_cols = ['porosity_x100', 'logk_ice_x100', 'k_stdev_x100', 'permafrost_flag', 'soil_flag']
    results = results[output_cols]
    results.reset_index(drop=False, inplace=True)
    t3 = time()
    
    print(f'   Aggregation time = {t3-t2:.1f}s for {len(results)} rows')
    return results


def process_glhymps_by_basin(input_data):
    # print('Processing GLHYMPS data by basin...')
    bid, original_basin_area, clipped_soil_df = input_data
    clipped_soil_df['area'] = clipped_soil_df.geometry.area / 1e6
    total_clipped_area = clipped_soil_df['area'].sum() 
    
    # full column is logK_Ice_x100_INT
    permeability_col = 'logK_Ice_x'
    # full column is K_stdevx100
    stdev_permeability_col = 'K_stdev_x1'
    # full column is poroxity_x100_INT
    porosity_col = 'Porosity_x'
    # permafrost
    permafrost_col = 'Prmfrst'

    area_diff = abs(original_basin_area - total_clipped_area) / original_basin_area
    geom_flag = 0
    if area_diff > 0.05:
        geom_flag = 1
    
    cols = ['id', 'Porosity_x100', 'logK_Ice_x100', 'k_stdev_x100', 'permafrost_FLAG', 'soil_FLAG']
    dtypes = ['int', 'int', 'int', 'int', 'int', 'int']
    type_dict = dict(zip(cols, dtypes))

    # if the total area is zero, return a row with null values
    if total_clipped_area == 0:
        data = [[bid, 0, 0, 0, 0, 1]]
        df = pd.DataFrame(data, columns=cols)
        return df.astype(type_dict)

    clipped_soil_df['frac_areas'] = clipped_soil_df['area'] / total_clipped_area
    avg_porosity = np.average(clipped_soil_df[porosity_col], weights=clipped_soil_df['frac_areas'])

    # don't take the log because the values are already log transformed
    # this is equivalent to the geometric mean
    geom_avg_permeability = np.average(clipped_soil_df[permeability_col], weights=clipped_soil_df['frac_areas'])
    perm_stdev = np.average(clipped_soil_df[stdev_permeability_col], weights=clipped_soil_df['frac_areas'])
    permafrost_flag = clipped_soil_df[permafrost_col].values.any()
    
    data = [[bid, avg_porosity, geom_avg_permeability, perm_stdev, permafrost_flag, geom_flag]]
    df = pd.DataFrame(data, columns=cols)

    return df.astype(type_dict)

def main():

    taa = time()
    # query the columns in the basin_attributes table
    terrain_attributes = ['slope_deg', 'aspect_deg', 'elevation_m', 'drainage_area_km2']
    # add a land cover flag if desired
    land_cover_cols = list(land_cover_groups.keys()) #+ ['land_cover_FLAG']
    land_cover_attributes = [f'{e}_{y}' for e in land_cover_cols for y in [2010, 2015, 2020]]

    soil_attributes = ['logk_ice_x100','k_stdev_x100', 'porosity_x100', 'permafrost_FLAG', 'soil_FLAG']
    daymet_params = ['prcp', 'tmin', 'tmax', 'vp', 'swe', 'srad', 
                     'high_prcp_freq','low_prcp_freq', 'high_prcp_duration', 'low_prcp_duration']

    # check if all raster tables have been created
    table_setup(schema_name, daymet_params)

    print('    ....Table setup completed')
    new_cols = land_cover_attributes + soil_attributes + terrain_attributes + daymet_params 
    add_table_columns(schema_name, new_cols)

    ## Add soil, land cover, and climate attributes to the catchments
    # Completed Daymet, terrain, and GLHYMPS
    regions = ['CLR', 'YKR', 'ERK', 'HAY', 'PCR', 'WWA', 'FRA', 'LRD', 'LRD']
    
    region = None
    if process_nalcms:
        for raster_set in [2010, 2015, 2020]:
            lc_attrs = sorted(list(set([e for e in land_cover_attributes if e.endswith(str(raster_set))])))
            unprocessed_basin_ids = []
            for p in lc_attrs[:1]:
                unprocessed_basin_ids += get_unprocessed_attribute_rows(p, region)
            # randomly shuffle the ids so we split up ordered clusters of large basins
            # random.shuffle(unprocessed_basin_ids)
            nalcms_ids = list(set(unprocessed_basin_ids))

            table_name = f'nalcms_{raster_set}'
            if len(nalcms_ids) > 0:
                # basin_geometry_check = check_for_basin_geometry(nalcms_ids)
                print(f'    Processing {len(nalcms_ids)} {raster_set} land cover rows.')
                rows_per_batch = 2500
                column_suffix = f'_{raster_set}'
                if len(nalcms_ids) > rows_per_batch:
                    n_batches = int(len(nalcms_ids) / rows_per_batch)
                    print(f'   Processing {n_batches} batches')
                    batches = np.array_split(nalcms_ids, n_batches)
                    input_data = [(raster_set, id_list, schema_name, table_name, column_suffix, None) for id_list in batches]
                else:
                    n_batches = 1
                    input_data = [(raster_set, nalcms_ids, schema_name, table_name, column_suffix, None)]
                        
                t0 = time()

                ta = time()
                with mp.Pool() as pool:
                    result = pool.map(process_nalcms_batch, input_data)
                tb = time()
                print(f'    ...processed {len(nalcms_ids)} rows in {tb-ta:.2f}s MULTIPROCESSING')

                # for batch in input_data:
                #     lc_results = process_raster(batch, 'nalcms')
                #     t1 = time()
                #     t_hr = (t1-t0) / 3600
                    # print(f'    Processed {len(lc_results)} rows in {t1-t0:.2f}h (s) ({n_batches} batches)')
                    # update_database(lc_results, schema_name, 'basin_attributes', column_suffix=column_suffix)
                # tc = time()
                # print(f'    Processed {len(nalcms_ids)} rows in {tc-tb:.2f}s (s) SEPARATE BATCHES')
            else:
                print(f'    No {raster_set} land cover rows to process.')
                
    ## Add soil information to the database
    t0 = time()
    unprocessed = []
    if process_glhymps:
        # this is pretty ugly, but the PostGIS method gets hung up on
        # geometries that become invalid after intersection

        print(f'Processing GLHYMPS data... ({region})')
        for p in soil_attributes:
            unprocessed += get_unprocessed_attribute_rows(p, region)
    
        unprocessed_basin_ids = list(set(unprocessed))
        
        # randomly shuffle the ids so we split up ordered clusters of large basins
        random.shuffle(unprocessed_basin_ids)
        t1 = time()
        print(f'    Unprocessed soil ID query time = {t1-t0:.1f}s for {len(unprocessed_basin_ids)} rows')
        
        if (len(unprocessed_basin_ids) > 0):
            soil_batch_size = 50
            if len(unprocessed_basin_ids) > soil_batch_size:
                
                n_batches = int(len(unprocessed_basin_ids) / soil_batch_size)
                batches = np.array_split(unprocessed_basin_ids, n_batches)
                input_data = batches
            else:
                n_batches = 1
                input_data = [unprocessed_basin_ids]
            
            n = 0
            print(f'    Processing {n_batches} soil batches, {soil_batch_size} basins/batch')
            uts = []
            for id_batch in input_data:
                n += 1
                t1 = time()       
                # df = process_glhymps_by_basin_batch(id_batch)
                basin_batch = get_basin_geometries(id_batch)
                basin_batch['basin'] = basin_batch['basin'].apply(fix_geoms)
                tb = time()
                
                # Use spatial index to filter intersecting geometries
                def use_sindex_to_prep(basin):
                    possible_matches_indices = list(glhymps_sindex.intersection(basin.bounds))
                    return glhymps_gdf.iloc[possible_matches_indices, :].copy()
                
                basin_batch['clipped_glhymps'] = basin_batch.apply(lambda x: use_sindex_to_prep(x['basin']), axis=1)
                
                # use multiprocessing to clip
                clip_inputs = [e for _, e in basin_batch.iterrows()]
                with mp.Pool() as pool:
                    batch_results = pool.map(clip_soil_parallel, clip_inputs)

                t2a = time()
                # print(f'   multiprocessing clip time = {t2a-tb:.1f}s for {len(basin_batch)} basins')
                # batch_inputs = [e for e in batch_results if e != None]
                # results = [e for e in batch_results if e != None]

                output = pd.concat(batch_results)
                output.columns = [e.lower() for e in output.columns]

                update_database(output, schema_name, 'basin_attributes')
                
                t2 = time()
                
                # print(f'    ...batch update time {t2-t2a:.2f}s')

                ut = len(id_batch) / (t2-t1)
                uts.append(ut)
                mean_ut = np.mean(uts)
                print(f'  Time to process batch {n}/{n_batches}: {t2-t1:.1f}s ({mean_ut:.1f} basins/s)')
                
    ## Add climate data to the database
    if process_daymet:
        # region = None
        for param in ['prcp']:#daymet_params:
            daymet_ids = get_unprocessed_attribute_rows(param, region)
            # daymet_ids = get_zero_val_attribute_rows(param, region)
            if len(daymet_ids) > 0:
                # basin_geometry_check = check_for_basin_geometry(daymet_ids)
                print(f'Processing {param} for {len(daymet_ids)} rows.')
                rows_per_batch = 10000
                table_name = f'daymet_{param}'
                column_suffix = ''
                if len(daymet_ids) > rows_per_batch:
                    n_iterations = int(np.ceil(len(daymet_ids) / rows_per_batch))
                    batches = np.array_split(daymet_ids, n_iterations)
                    input_data = [(param, id_list, schema_name, table_name, column_suffix, param) for id_list in batches]
                else:
                    input_data = [(param, daymet_ids, schema_name, table_name, column_suffix, param)]
                        
                t0 = time()
                n_complete = 0
                for batch in input_data:
                    n_complete += 1
                    daymet_results = process_raster(batch, 'daymet')
                    # daymet_df = pd.concat(daymet_results)
                    update_database(daymet_results, schema_name, 'basin_attributes', column_suffix=column_suffix)
                    t1 = time()
                    t_hr = (t1-t0) / 3600
                    print(f'   Processed daymet ({param}) batch {n_complete}/{len(input_data)} in {t_hr:.2f}h')
            else:
                print(f'    ...No {param} rows to process.')
    tbb = time()
    print(f'Finished postgis database extension in {tbb - taa:.2f} seconds.')
    

with psycopg2.connect(**conn_params) as conn:
    cur = conn.cursor() 
    main()

cur.close()
conn.close()
