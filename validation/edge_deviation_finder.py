import os

import pandas as pd
import geopandas as gpd
import numpy as np

import psycopg2
import psycopg2.extras as extras
import os
import warnings
from sqlalchemy import create_engine  

import multiprocessing as mp
from time import time

conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
}
schema_name = 'basins_schema'
basin_table = 'basin_attributes'
new_table_name = 'edge_deviation'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'validation/data/')

deviation_files = [e for e in os.listdir(DATA_DIR) if e.endswith('_bounds_deviations_test.geojson')]
region_codes = sorted([e.split('_')[0] for e in deviation_files])

def postgis_query(query, type='fetch'):
    # Example function to execute a PostgreSQL query
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    
    try:
        cur.execute(query)
        rows = cur.fetchall()
        if type == 'write':
            conn.commit()
        return rows
    except Exception as e:
        conn.rollback()
        print(f"Error executing query: {e}")
    finally:
        cur.close()
        conn.close()
    


# for rc in region_codes:
#     # get the deviations by region
#     deviation_df = gpd.read_file(os.path.join(DATA_DIR, f'{rc}_bounds_deviations_test.geojson'))
    
#     nbr_codes = [e for e in deviation_df['region_code_1'].values if e is not None]
#     # get the hydrobasins deviations
#     hb = deviation_df[deviation_df['region_code_1'].isin(nbr_codes)].copy().dissolve()
#     # hb.to_file(f'Hydrobasins_deviations_{rc}.geojson')
#     # get the BCUB deviations
#     bdf = deviation_df[~deviation_df['region_code_1'].isin(nbr_codes)].copy()
#     bdf['region_code'] = rc
#     bdf = bdf[['region_code', 'geometry']]
#     bdf = bdf.explode()
#     bdf.geometry = bdf.geometry.buffer(0)
#     bdf.reset_index(inplace=True, drop=True)
#     print(bdf)
#     # check that all geometries are valid
#     assert bdf.geometry.is_valid.all()
#     temp_path = os.path.join(DATA_DIR, f'{rc}_BCUB_deviations.geojson')
#     bdf.to_file(temp_path)

#     cmd = f'ogr2ogr -f "PostgreSQL" PG:"dbname={conn_params["dbname"]} user={conn_params["user"]} password={conn_params["password"]}" {temp_path} -nln {new_table_name} -nlt PROMOTE_TO_MULTI -lco GEOMETRY_NAME=geom -lco FID=id'
#     print(f'Adding {rc} deviations to {new_table_name}. ')
#     os.system(cmd)


# q = f'CREATE INDEX IF NOT EXISTS idx_{new_table_name}_geom ON {new_table_name} USING GIST(geom);'
# postgis_query(q, 'fetch')
# q = f"""SELECT ba.basin
# FROM basins_schema.basin_attributes AS ba
# JOIN edge_deviation AS ed ON ba.region_code = ed.region_code
# WHERE ST_Intersects(ba.basin, ed.geom)
# LIMIT 10;
# """


def check_deviation_proportion(row, crs, deviation_parts):

    idx = row['id']
    this_basin = gpd.GeoDataFrame(geometry=[row['geometry']], crs=crs)
    catchment_area = this_basin.area.values[0]

    tb = time()

    # use a spatial query to narrow down the list of deviations to check
    deviation_sindex = deviation_parts.sindex
    possible_intersections_index = list(deviation_sindex.query(row['geometry'], predicate='intersects'))
    possible_touching_index = list(deviation_sindex.query(row['geometry'], predicate='touches'))
    intersecting_deviations = deviation_parts.iloc[possible_intersections_index]
    touching_deviations = deviation_parts.iloc[possible_touching_index]

    # intersecting_deviations = gpd.sjoin(deviation_df, this_basin, how='inner', predicate='intersects')
    # touching_deviations = gpd.sjoin(deviation_parts, this_basin, how='inner', predicate='touches')
    tc = time()
     # it's about 0.1s per basin
    # print(f'{idx}: time to compute intersecting and touching: {tc-tb:.3f}s   {len(intersecting_deviations)} {len(touching_deviations)}')
    touching_area = 0
    if not touching_deviations.empty:
        # print('  Bound-touching deviations found!  --simply add to outside area')
        touching_area = touching_deviations.geometry.area.sum()
    
    inside = intersecting_deviations.clip(this_basin)
    outside = intersecting_deviations.overlay(this_basin, how='difference', keep_geom_type=True)
    inside_area = inside.geometry.area.sum()
    outside_area = outside.geometry.area.sum() + touching_area
    inside_pct_area = inside_area / catchment_area
    outside_pct_area = outside_area / catchment_area
    tot_pct_area = inside_pct_area + outside_pct_area

    td = time()
    # print(f'{idx}: time to compute inside and outside pct areas: {td-tc:.3f}s') # it's about 0.1s per basin
    
    flag = False
    if tot_pct_area >= 0.05:
        flag = True
        # print(f'    FLAG: Uncertainty at region bounds amounts to {tot_pct_area*100:.0f}% of the (unsimplified) area ({catchment_area/1e6:.1f} km^2).')
    
    return (idx, inside_pct_area, outside_pct_area, flag)


def process_data_wrapper(args):
    return check_deviation_proportion(**args)

def get_edge_basins(region_code, output_path):

    # query the rows from basins_schema.basin_attributes
    # that intersect with edge_deviation for a given region_code
    q = f"""
    WITH filtered_basins AS (
        SELECT
            b.id,
            b.region_code,
            b.drainage_area_km2,
            b.basin
        FROM
            basins_schema.basin_attributes AS b
        WHERE
            b.region_code = '{region_code}'
    ), 
    union_deviations AS (
        SELECT
            ST_Union(geom) AS deviations
        FROM
            edge_deviation
        WHERE
            region_code = '{region_code}'
    ),
    intersecting_basins AS (
        SELECT
            fb.id,
            fb.region_code,
            fb.drainage_area_km2,
            fb.basin 
        FROM
            filtered_basins AS fb,
            union_deviations as ud
        WHERE
            ST_Intersects(fb.basin, ud.deviations)
    )
    SELECT
        id,
        region_code,
        drainage_area_km2,
        basin geometry
    FROM
        intersecting_basins;
    """

    db_connection_url = f"postgresql://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"
    con = create_engine(db_connection_url)  
    t0 = time()
    df = gpd.read_postgis(q, con, geom_col='geometry')
    # df.to_file(output_path, driver='GeoJSON')
    t1 = time()
    print(f'  {len(df)} results returned in {t1-t0:.1f}s')
    return df

for rc in region_codes:
    continue
    boundary_deviation_file = os.path.join(DATA_DIR, f'{rc}_BCUB_deviations.geojson')
    if not os.path.exists(boundary_deviation_file):
        print(f'  No boundary deviations for {rc}.')
        continue
    
    temp_path = os.path.join(DATA_DIR, f'{rc}_intersecting_basins.geojson')
    # if not os.path.exists(temp_path):
    print(f'  Querying {rc} basins from database.')
    df = get_edge_basins(rc, temp_path)
    # else:
    #     print(f'  Reading {rc} basins from file.')
    #     df = gpd.read_file(temp_path)

    crs = df.crs.to_epsg()
    
    bcub_deviations = gpd.read_file(boundary_deviation_file)
    deviation_sindex = bcub_deviations.sindex

    # keep only basins where the total deviation area is greater than 5% of the basin area
    # this is a rough heuristic to avoid unnecessary computation on larger geometries
    total_deviation_area = bcub_deviations.geometry.area.sum() / 1e6
    df = df[total_deviation_area >= 0.05 * df['drainage_area_km2']]

    assert bcub_deviations.crs == df.crs

    to_check = [{'row': row, 'crs':crs, 'deviation_parts': bcub_deviations.copy()} for _, row in df.iterrows()]

    rows_per_batch = 200
    n_batches = int(np.ceil(len(df) / rows_per_batch))
    hdf_batches = np.array_split(to_check, n_batches)
    all_results = []
    batch_no = 0
    total_flags = 0
    for batch in hdf_batches:
        batch_no += 1
        t0 = time()
        with mp.Pool(1) as pl:
            results = pl.map(process_data_wrapper, batch)
            all_results += results
        t1 = time()
        print(f'    {t1-t0:.0f}s to process batch {batch_no} of {n_batches}')
        result_df = pd.DataFrame(all_results, columns=['id', 'inside_pct_area', 'outside_pct_area', 'FLAG_boundary'])
        result_df = result_df[result_df['FLAG_boundary']].copy()
        # convert from fraction to integer percentage 
        result_df[['inside_pct_area', 'outside_pct_area']] = (100 * result_df[['inside_pct_area', 'outside_pct_area']]).round(0).astype(int)
        data_tuples = list(result_df.itertuples(index=False, name=None))
        # Construct the set_string and v_string
        set_string = "geometry_flag = TRUE, inside_pct_area_FLAG = data.inside_pct_area, outside_pct_area_FLAG = data.outside_pct_area"
        v_string = "inside_pct_area, outside_pct_area"

        if not result_df.empty:
            flag_id_str = ', '.join(result_df['id'].astype(str).values)
            # write a query to set all rows with a FLAG_boundary == True to True
            query = f"""
            UPDATE {schema_name}.{basin_table} AS basin_table
            SET {set_string}
            FROM (VALUES %s) AS data(id, {v_string})
            WHERE basin_table.id = data.id;
            """
            # Connect to the database and execute the query
            conn = psycopg2.connect(**conn_params)
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, query, data_tuples, template=None, page_size=500
                )
                conn.commit()

            # Close the connection
            conn.close()

        total_flags += len(result_df)

    print(f'  {len(result_df)} {rc} basins flagged for boundary uncertainty.')