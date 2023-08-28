
import psycopg2
import psycopg2.extras as extras
import os
from time import time
import pandas as pd

import numpy as np
import multiprocessing as mp

from itertools import permutations

from shapely.geometry import Point

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

import warnings

warnings.filterwarnings('ignore')

# The KL divergence of P from Q is the expected excess surprise
# from using Q as a model when the actual distribution is P


conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
}

schema_name = 'basins_schema'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'input_data/')
HYSETS_DIR = os.path.join(BASE_DIR, 'input_data/HYSETS_data')

# test_region = 'VCI'

# def filter_test_region_hysets_stations(region_code):    
#     # filter the hsyets stations to only those in the test region to reduce load time
#     # import the test region polygon 
#     hs_df = gpd.read_file(os.path.join(HYSETS_DIR, 'HYSETS_watershed_properties.geojson'))
#     region_polygon = gpd.read_file(os.path.join(DATA_DIR, f'region_polygons/{region_code}_4326.geojson'))    
#     region_stns = hs_df.sjoin(region_polygon)
#     region_stns.to_file(os.path.join(HYSETS_DIR, f'HYSETS_{region_code}_properties.geojson'), driver='GeoJSON')
#     return region_stns, region_polygon
    
    # import the hysets stations

# region_code = 'VCI'
# region_stn_file = os.path.join(HYSETS_DIR, f'HYSETS_{region_code}_properties.geojson')
# if not os.path.exists(region_stn_file):
#     print('Region stations not yet filtered, creating now...')
#     region_stns, region_polygon = filter_test_region_hysets_stations(region_code)
# else:
#     print('Retrieving filtered region stations...')
#     region_stns = gpd.read_file(region_stn_file)
#     region_polygon = gpd.read_file(os.path.join(DATA_DIR, f'region_polygons/{region_code}_4326.geojson'))  


def get_cml_locations(region_polygon):
    # given a polygon query the database for all cmls within that polygon
    # return a geodataframe of the cmls and their properties
    
    # convert the polygon to wkt
    region_polygon = region_polygon.to_crs(3005)
    polygon_wkt = region_polygon.geometry.iloc[0].wkt

    query = f"""
    SELECT * 
    FROM basins_schema.basin_attributes
    WHERE ST_Within(pour_pt, ST_GeomFromText(%s, 3005))
    LIMIT 10;
    """
    
    with psycopg2.connect(**conn_params) as conn:
        # cur = conn.cursor() 
        # cur.execute(query, (polygon_wkt,))
        
        # get the results
        # results = cur.fetchall()
        cml_df = gpd.read_postgis(query, conn, geom_col='pour_pt', params=(polygon_wkt,))
        for c in cml_df.columns:
            print(cml_df[[c]].head())
            print('')
        print(cml_df)
    
    # cur.close()
    # conn.close()
    return cml_df


def get_hysets_locations(region_polygon, search_buffer=1000):
    # convert the polygon to wkt
    region_polygon = region_polygon.to_crs(3005)
    polygon_wkt = region_polygon.geometry.iloc[0].wkt
    
    query = f"""
    SELECT * 
    FROM basins_schema.hysets_basins
    WHERE ST_Within(centroid, ST_GeomFromText(%s, 3005))
    LIMIT 10;
    """
    with psycopg2.connect(**conn_params) as conn:
        # cur = conn.cursor() 
        # cur.execute(query, (polygon_wkt,))
        
        # get the results
        # results = cur.fetchall()
        hs_df = gpd.read_postgis(query, conn, geom_col='centroid', params=(polygon_wkt,))
    return hs_df


def create_normalized_views(schema_name, table_name, cols):
    norm_string_array = [f'({col} - (SELECT MIN({col}) FROM {schema_name}.{table_name})) / (SELECT MAX({col}) - MIN({col}) FROM {schema_name}.{table_name}) AS {col}_normalized' for col in cols]
    normalized_strings = ',\n'.join(norm_string_array)
    query = f"""
    CREATE VIEW normalized_{table_name} AS
    SELECT
        id,
        {normalized_strings}
    FROM {schema_name}.{table_name};
    """
    print(query)
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query) 
            print('Query executed successfully')   


def get_extreme_val(schema_name, table_name, col, which_extreme='MIN'):
    # get the min and max values for each column in the table
    # col_string = ','.join([f"{which_extreme}({c})" for c in cols])
    # get the extreme values for each column in the table
    query = f"SELECT {which_extreme}({col}) AS max_{col} FROM {schema_name}.{table_name} WHERE {col} = {col}"
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query) 
            result = cur.fetchone()[0]
            return result


def get_minmax_dict(schema_name, static_attributes):
    minmax_dict = {}
    for table in ['basin_attributes', 'hysets_basins']:
        minmax_dict[table] = {}
        for c in static_attributes:
            max_vals = get_extreme_val(schema_name, table, c, 'MAX')
            min_vals = get_extreme_val(schema_name, table, c, 'MIN')
            minmax_dict[table][c] = {'min': min_vals, 'max': max_vals}
    return minmax_dict

    
def get_table_cols(schema_name, table_name):
    query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = '{schema_name}'
    AND table_name   = '{table_name}'
    ORDER BY table_name, column_name;
    """
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query) 
            results = cur.fetchall()
            return [e[0] for e in results]
    
    
def check_if_columns_exist(schema_name, table_name, columns):
    col_exists = []
    for c in columns:
        query = f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = '{schema_name}'
                AND table_name   = '{table_name}'
                AND column_name = '{c}'
            ) AS column_exists;
            """
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                result = cur.fetchone()[0]
                col_exists.append(result)
    return col_exists
       

def get_attributes(schema_name, table_name, static_attributes):
    col_string = ','.join(static_attributes + ['centroid']) 
    query= f'SELECT {col_string} FROM {schema_name}.{table_name};'
    gdf = gpd.read_postgis(query, psycopg2.connect(**conn_params), geom_col='centroid')
    return gdf
    

def circular_diff(a, b, n=""):
    # print(f'{n} circ diff: ', a, b)
    return np.minimum(np.abs(a - b), 360 - np.abs(a - b))

    
def log_diff(a, b, n=""):
    # set zero values to 1E-6 to avoid divide by zero errors
    a = np.where(a == 0, 1E-9, a)
    b = np.where(b == 0, 1E-9, b)    
    return np.abs(np.log(a) - np.log(b))


def abs_diff(a, b, n=""):
    return np.abs(a - b)


def spatial_dist(a, b, n=""):
    return a.distance(b) / 1E6

    
def determine_nearest_proxy(inputs):
    cml, obs, functions = inputs
    # attributes = list(functions.keys())
    dist_cols = [f'diff_{k}' for k in functions.keys()]
    # calculate the difference between the observed and the row values on each attribute
    # and return the index of the smallest sum of differences
    for _, cml_row in cml.iterrows():
        # get the minimum value and the index of the minimum value
        db_id = cml_row['id']
        distances = pd.DataFrame()
        for k, _ in functions.items():
            try:
                distances[f'diff_{k}'] = functions[k](cml_row[k], obs[k])
            except Exception as ex:
                
                print(k, ex)
                raise Exception; 'damn!!!'
        
        distances['L1_norm'] = distances.abs().sum(axis=1)
        
        min_idx = distances['L1_norm'].idxmin()
        val = distances['L1_norm'].iloc[min_idx]
        cml.loc[cml['id'] == db_id, 'baseline_station_idx'] = min_idx
        cml.loc[cml['id'] == db_id, 'baseline_station_dist'] = val
    
    return cml
    

def find_unprocessed_distance_idxs(schema_name, table_name, cols):
    
    # join the null query checks on all columns with an OR statement
    col_string = ' OR '.join([f'{col} is NULL' for col in cols])
    query = f"""
    SELECT id
    FROM {schema_name}.{table_name}
    -- where any of the columns are null
    WHERE {col_string};
    """
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
            return [e[0] for e in results]


def run_query(query, tuples=None):
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                if tuples is None:
                    cur.execute(query)
                else:
                    extras.execute_values(cur, query, tuples, page_size=1000)                
                # conn.commit()
                print('Query executed successfully')
    except Exception as e:
        print(e)
        print('Query failed.')
    finally:
        conn.close()


def add_distance_col(schema_name, table_name, column):
    if column.endswith('dist'):
        dt = 'FLOAT'
    elif column.endswith('idx'):
        dt = 'INTEGER'
    else:
        raise Exception; 'Column name must end with "dist" or "idx" or define another data type.'
    query = f"""
    ALTER TABLE {schema_name}.{table_name}
    ADD COLUMN IF NOT EXISTS {column} {dt};
    """
    run_query(query)
         
         
def update_database_with_distances(new_results, schema_name, table_name):
    data = new_results[['id', 'baseline_station_idx', 'baseline_station_dist']].copy()
    n_pts = len(data)
    print(f'  Updating database with new {n_pts} baseline distances...')
    # filter out nan rows for any column
    data.dropna(inplace=True, axis=0, how='any', subset=['id', 'baseline_station_idx', 'baseline_station_dist'])
    n_good_pts = len(data)
    print(f'  Dropped {n_pts - n_good_pts} na values...')
    tuples = list(data.itertuples(index=False, name=None))
    
    # Step 1: Create a temporary table
    try: 
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TEMP TABLE IF NOT EXISTS temp_update_table
                    (id BIGINT primary key, baseline_station_idx BIGINT, baseline_station_dist FLOAT);
                """)
                # Step 2: Insert your data into the temporary table
                extras.execute_values(
                    cur,
                    """
                    INSERT INTO temp_update_table (id, baseline_station_idx, baseline_station_dist) VALUES %s;
                    """,
                    tuples
                )
                conn.commit()
                # Step 3: Update the main table
                cur.execute(f"""
                    UPDATE {schema_name}.{table_name} main
                    SET 
                        baseline_station_idx = temp.baseline_station_idx,
                        baseline_station_dist = temp.baseline_station_dist
                    FROM temp_update_table temp
                    WHERE main.id = temp.id;
                """)
                conn.commit()
    except Exception as ex:
        print(ex)
    finally:
        conn.close()
    
    
def check_if_table_exists(schema_name, table_name):
    query = f"""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = '{schema_name}'
            AND table_name   = '{table_name}'
        ) AS table_exists;
        """
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()[0]
            return result


cml_cols = get_table_cols(schema_name, 'basin_attributes')
hysets_cols = get_table_cols(schema_name, 'hysets_basins')

static_attributes = [c for c in cml_cols if (c in hysets_cols) & (c != 'centroid')]

minmax_dict = get_minmax_dict(schema_name, static_attributes)

diff_funcs = {
    # 'centroid_lat_deg_n': abs_diff, 'centroid_lon_deg_e': abs_diff, 
    'centroid': spatial_dist,
    'drainage_area_km2': log_diff,
    'elevation_m': abs_diff, 'slope_deg': abs_diff, 'aspect_deg': circular_diff,
    'land_use_forest_frac_2010': abs_diff, 'land_use_grass_frac_2010': abs_diff, 'land_use_wetland_frac_2010': abs_diff, 
    'land_use_water_frac_2010': abs_diff, 'land_use_urban_frac_2010': abs_diff, 'land_use_shrubs_frac_2010': abs_diff,  
    'land_use_crops_frac_2010': abs_diff, 'land_use_snow_ice_frac_2010': abs_diff,
    'logk_ice_x100': abs_diff, 'porosity_x100': abs_diff
}

# cml_df = get_attributes(schema_name, 'basin_attributes', ['id'] + static_attributes)
cml_df = gpd.read_postgis(f'SELECT * FROM basins_schema.basin_attributes', psycopg2.connect(**conn_params), geom_col='centroid')

hs_df = get_attributes(schema_name, 'hysets_basins', ['watershed_id'] + static_attributes)
assert cml_df.crs == hs_df.crs


RESULTS_DIR = os.path.join(BASE_DIR, 'results')
results_fpath = os.path.join(RESULTS_DIR, 'nearest_stations.geojson')
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

baseline_cols = ['baseline_station_idx', 'baseline_station_dist']
dist_cols_exist = check_if_columns_exist(schema_name, 'basin_attributes', baseline_cols)

for v, c in zip(dist_cols_exist, baseline_cols):
    if not v:
        print(f'   ...adding {c} columns to BCUB dataset.')
        add_distance_col(schema_name, 'basin_attributes', c)


def generate_batches(idxs, df, hs_df, diff_funcs, batch_size=500):
    """generator to avoid creating copies of the dataframe in 
    memory outside of each process."""
    n_batches = int(np.ceil(len(idxs) / batch_size))
    batch_idxs = np.array_split(idxs, n_batches)
    for b in batch_idxs:
        yield (df.loc[df['id'].isin(b), :].copy(), hs_df.copy(), diff_funcs)

unprocessed_distance_idxs = find_unprocessed_distance_idxs(schema_name, 'basin_attributes', baseline_cols)

if len(unprocessed_distance_idxs) > 0:    
    print(f'Found unprocessed baseline distances.  Processing {len(unprocessed_distance_idxs)} locations...')
    # these ids should match the "id" column in basin_attributes table
    unprocessed_ids = cml_df.loc[cml_df['id'].isin(sorted(unprocessed_distance_idxs)), 'id'].copy().values
    with mp.Pool(8) as pool:
        # results = pool.map(determine_nearest_proxy, batches)
        results = pool.map(determine_nearest_proxy, generate_batches(unprocessed_ids, cml_df, hs_df, diff_funcs))
        # print(f'Batch time: {t_batch-t0:.2f}s ({ut:.3f}s per row, N={len(b)} batch size)')

    cml_df = gpd.GeoDataFrame(pd.concat(results, axis=0), crs=cml_df.crs, geometry='centroid')

    baseline_cols = [c for c in cml_df.columns if 'baseline' in c]
    # all_results.to_file(results_fpath, driver='GeoJSON')
    update_database_with_distances(cml_df, 'basins_schema', 'basin_attributes')

       

def calculate_cml_distance(inputs):
    cml_id, df, diff_funcs = inputs
    target_loc = df.loc[df['id'] == cml_id, :].copy()
    
    diff_cols = []
    for k, _ in diff_funcs.items():
        if k == 'centroid':
            target_data = target_loc['centroid']
        else:
            target_data = [target_loc[k].values[0]] * len(df)
        df[f'diff_{k}'] = diff_funcs[k](target_data, df[k])
        diff_cols.append(f'diff_{k}')
    # calculate the sum of all rows starting with 'diff_k'
    df['L1_norm'] = df[diff_cols].abs().sum(axis=1)
    df = df[df['L1_norm'] > 0]
    df['distance_change'] = df['baseline_station_dist'] - df['L1_norm']
    # filter out any rows where the distance change is negative, 
    # menaing the new distance is greater than the baseline distance
    improved_locs = df[df['distance_change'] > 0].copy()
    
    tot_reduction = improved_locs['distance_change'].sum()
    n_improved_locs = len(improved_locs)
    mean_improved_dist = improved_locs['distance_change'].mean()
    #
    #
    # add a distance to track the furthest spatial distance from the target location
    #
    #
    data = {'id': target_loc['id'], 
            'tot_reduction': tot_reduction, 
            'n_improved_locs': n_improved_locs, 
            'mean_improved_dist': mean_improved_dist, 
            'geometry': Point(target_loc['ppt_lon_m_3005'], 
                              target_loc['ppt_lat_m_3005'])
            }
    return pd.DataFrame(data, index=[cml_id])


def generate_inputs(df, diff_funcs):
    """generator to avoid creating copies of the dataframe in 
    memory outside of each process."""
    for i in df['id'].values.tolist():
        yield (i, df.copy(), diff_funcs)

cml_dist_col_exists = check_if_columns_exist(schema_name, 'basin_attributes', ['cml_distance'])[0]

if not cml_dist_col_exists:
    print('Adding expected CML distances to BCUB dataset.')
    t0 = time()
    # you may need to decrease the number of processes or introduce batches to avoid memory issues
    with mp.Pool() as pool:
        best_sites = pool.map(calculate_cml_distance, generate_inputs(cml_df, diff_funcs))

    t1 = time()
    ut = (t1-t0) / len(cml_df)
    print(f'Total time: {t1-t0:.2f}s ({ut:.2f}s/row, N={len(cml_df)})')
    
result_df = gpd.GeoDataFrame(pd.concat([e for e in best_sites if e is not None]), crs='EPSG:3005')

result_fpath = os.path.join(RESULTS_DIR, 'best_sites_20230822.geojson')
result_df.to_file(result_fpath)

    