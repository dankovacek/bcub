import geopandas as gpd
import pyarrow.parquet as pq
import psycopg2
import psycopg2.extras as extras
import os
from time import time
import pandas as pd

# from dask import dataframe as dd
# import multiprocessing as mp

import numpy as np

dtype_dict = {
    'double': 'FLOAT',
    'int64': 'INT',
    'float64': 'FLOAT',
    'string': 'VARCHAR(255)',
    'object': 'VARCHAR(255)',
    'bool': 'SMALLINT',
}

geom_dict = {
    'basin_geometry': 'basin',
    'centroid_geometry': 'centroid',
    'geometry': 'pour_pt',
}

def create_table_initialization_query(parquet_schema, schema_name, db_host, db_name, db_user, db_password):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS postgis;')
    conn.commit()
    cur.execute(f'CREATE SCHEMA IF NOT EXISTS {schema_name};')
    conn.commit()
    print('    Schema created.')
    # metadata = parquet_schema.metadata
        # Create a list of column names and types
    geom_columns = [c for c in parquet_schema.names if 'geometry' in c]
    
    # despite no geometry types of multipolygon found, 
    # multipolygon type error for polygon specified column persists
    # to fix, try switching to generic Geometry type:
    # ALTER TABLE my_table ALTER COLUMN geom TYPE geometry(Geometry,3005);
    
    # i think postgres needs a column named geom

    q = f'''CREATE TABLE IF NOT EXISTS {schema_name}.basin_attributes (
        id BIGSERIAL PRIMARY KEY,
        centroid geometry(POINT, 3005),
        pour_pt geometry(POINT, 3005),
        basin geometry(POLYGON, 3005),
        '''
        
    column_names = []
    column_types = []
    for field in parquet_schema:
        ###
        ### check if dtypes are correct!
        ####
        if field.name not in geom_columns:
            column_names.append(field.name)
            column_types.append(dtype_dict[field.type])

    for col, col_type in zip(column_names, column_types):
        if col in ['__index_level_0__', 'ID', 'FID']:
            continue            
        if col not in geom_columns:
            q += f' {col} {col_type},'
            print(f'    Added column {col} of type {col_type} to table.')
            
    q = q[:-1] + ');'
    cur.execute(q)
    conn.commit()
    conn.close()
    print('    Table created: "basin_attributes"')
    

def populate_table_columns(df, schema_name, geometry_cols, non_geo_cols, db_host, db_name, db_user, db_password):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    
    crs = df.crs.to_epsg()
    for gc in geometry_cols:
        table_name = f'basin_attributes'
        # add the geometry column to the table
        geom_type = 'POINT'
        col_name = geom_dict[gc]
        if gc == 'basin_geometry':
            geom_type = 'POLYGON'
                    
        print(f'     {gc} will be converted to geometry({geom_type}, {crs}).')
        cur.execute(f'ALTER TABLE {schema_name}.{table_name} ADD COLUMN IF NOT EXISTS {col_name} geometry({geom_type},{crs});')
        # add the data to the table
        if gc == 'basin_geometry':
            cur.execute(f"UPDATE {schema_name}.{table_name} SET {col_name} = ST_GeomFromEWKB(decode({col_name}, 'hex'));")
        else:
            cur.execute(f"UPDATE {schema_name}.{table_name} SET {col_name} = ST_PointFromWKB(decode({col_name}, 'hex'));")
    
    # add non-geometry columns to the table
    print('     Adding non-geometry columns to table...')
    nc = 0
    for c in non_geo_cols:
        nc += 1
        table_name = 'basin_attributes'
        dtype_name = df[c].dtype.name
        dtype = dtype_dict[dtype_name]
        
        if c in ['ID', 'flag_acc_match']: 
            c = 'object_id'
            dtype = 'INT'
                        
        # print(f'   {c} is of type {dtype_name} and will be converted to {dtype}.')
        cur.execute(f'ALTER TABLE {schema_name}.{table_name} ADD COLUMN IF NOT EXISTS {c} {dtype};')
    print('     Added ALL columns to table.')
    conn.commit()
    conn.close()


def test_query(schema_name, db_host, db_name, db_user, db_password):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    q = f"select id, basin from {schema_name}.basin_attributes order by id desc limit 4"
    
    try:
        test_df = gpd.read_postgis(q, conn, geom_col='basin')
    except Exception as e:
        print(e)
        return False, pd.DataFrame()

    return True, test_df


def count_basins_in_db(region):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    q = f"select count(*) from {schema_name}.basin_attributes where region_code = '{region}'"
    cur = conn.cursor()
    cur.execute(q)
    n_basins = cur.fetchall()[0][0]
    conn.close()
    return n_basins


def get_ppt_count(region):
    # open the pour point file and count the number of unique pour points
    ppt_file = os.path.join(data_dir, 'pour_points/', region, f'{region}_pour_pts_filtered.geojson')
    if not os.path.exists(ppt_file):
        raise Exception(f'Pour point file not found: {ppt_file}.  You must first run find_pour_points.py then lakes_filter.py to generate this file.')
    else:
        ppt_df = gpd.read_file(ppt_file)
        return len(ppt_df)


# @delayed
def load_file_dgp(filename, schema):
    t0 = time()
    # df = dgp.read_parquet(filename, 
    #                       calculate_divisions=True,
    #                       schema=schema 
    #                       )
    df = gpd.read_parquet(filename, schema=schema)
    # df = df.repartition(npartitions=nparts)
    # replace nan with None
    df = df.replace(to_replace=np.nan, value=None)
    
    # drop the id, fid, and object_id columns
    drop_cols = [e for e in ['ID', 'FID', 'object_id'] if e in df.columns]
    df = df.drop(labels=drop_cols, axis=1)
    
    return df


def reformat_geom_cols(df, geometry_cols, nparts):
    # encode geometries to well known binary
    for c in geometry_cols:
        df[c+'_wkb'] = df[c].to_wkb(hex=True)

    # drop original geometry columns    
    drop_cols = ['geometry', 'centroid_geometry', 'basin_geometry']    
    df = df.drop(labels=drop_cols, axis=1)
    return df


def check_polygon_flags(df):
    """add a flag if the difference between the derived polygon area 
    and the area defined by the flow accumulation at the pour point is
    greater than 5% of the 'area' (also defined by the polygon area 
    but by a different function).

    Args:
        df (dataframe): basin attributes dataframe
    """
    df['FLAG_acc_match'] = ((df['ppt_acc'] - df['acc_polygon']).abs() / df['acc_polygon'] > 0.05).astype(int)
    return df


def update_constraint(schema_name, db_host, db_name, db_user, db_password):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    # print('    Adding unique constraint to cell_idx column...')
    # unique_constraint_query = f"ALTER TABLE {schema_name}.basin_attributes ADD CONSTRAINT IF NOT EXISTS raster_cell_idx UNIQUE (cell_idx);"
    cur.execute(f'CREATE SCHEMA IF NOT EXISTS {schema_name};')
    # cur.execute(unique_constraint_query)
    conn.commit()


def convert_parquet_to_postgis_db(parquet_dir, db_host, db_name, db_user, db_password, 
                                     schema_name,  total_bounds):
    # connect to the PostGIS database    
    db_initialized, _ = test_query(schema_name, db_host, db_name, db_user, db_password)
    print(f'     ...the database is initialized: {db_initialized}')
    test_dir = os.path.join(parquet_dir, 'VCI')
    if not os.path.exists(test_dir):
        raise Exception(f'No parquet files found in {parquet_dir}.  derive_basins.py must be run first.')
    test_file = [e for e in os.listdir(test_dir) if e.endswith('.parquet')]
    parquet_schema = pq.read_schema(os.path.join(test_dir, test_file[0]))
        
    if not db_initialized:

        if len(test_file) == 0:
            raise Exception(f'No parquet files found in {parquet_dir}.')
        print(f'    Postgis database table is empty, creating table and populating columns...')
        create_table_initialization_query(parquet_schema, schema_name, db_host, db_name, db_user, db_password)

    update_constraint(schema_name, db_host, db_name, db_user, db_password)
    
    minx, miny, maxx, maxy = total_bounds
    divisor = 1000
    minx = int(np.floor(minx / divisor) * divisor)
    miny = int(np.floor(miny / divisor) * divisor)
    maxx = int(np.ceil(maxx / divisor) * divisor)
    maxy = int(np.ceil(maxy / divisor) * divisor)

    # completed = ['08A', 'HGW', 'VCI', '08C', '08D', 
    # '08F', 'FRA', 'WWA', 'HAY', '08G', '10E',
    #  'CLR', 'PCR','YKR', 'ERK', '08B', '08E', 'LRD]
    
    region_codes = ['10E']#sorted(os.listdir(parquet_dir))
    
    for rc in region_codes:
        
        print(f'Processing {rc} region -----------------------')

        n_basins_in_db = count_basins_in_db(rc)
        n_ppt = get_ppt_count(rc)
        
        if n_basins_in_db > n_ppt:
            raise Exception(f'    {rc} there should not be more db rows than pour points...')
        if n_basins_in_db == n_ppt:
            print(f'    {rc} region already processed ({n_basins_in_db} basins/ppts), skipping...')
            continue
            
        fpath = os.path.join(parquet_dir, rc)
        nparts = len(os.listdir(fpath))
        print(f'    Loading file {fpath.split("/")[-1]}...')
        
        df = load_file_dgp(fpath, parquet_schema)
        
        geometry_cols = [c for c in df.columns if 'geometry' in c]
        non_geo_cols = [c for c in df.columns if 'geometry' not in c]
        
        df = check_polygon_flags(df)
          
        if len(non_geo_cols) > len(set(non_geo_cols)):
            raise Exception('Duplicated column names in non-geometry columns.')
        
        if not db_initialized:
            print('populate columns---------------------')
            populate_table_columns(df, schema_name, geometry_cols, non_geo_cols, db_host, db_name, db_user, db_password)
        
        t3 = time()
        df = reformat_geom_cols(df, geometry_cols, nparts)
        t4 = time()
                
        print(f'   Dask geopandas converted columns to wkb in {t4-t3:.2f} seconds.')
        # populate non-geometry values
        print('    Starting db insert...')
        
        conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
        cur = conn.cursor()

        print('    Database connected...')        
        all_cols = [geom_dict[e] for e in geometry_cols] + non_geo_cols
                
        # res = df.apply(insert_vals_to_db, cur=cur, geom_cols=geometry_cols, non_geo_cols=non_geo_cols, axis=1)
        tuples = list(df[[e+'_wkb' for e in geometry_cols] + non_geo_cols].itertuples(index=False, name=None))   

        cols_str = ', '.join(all_cols)
        query = f'''
        INSERT INTO {schema_name}.basin_attributes ({cols_str}) 
        VALUES %s;'''
        
        for col in ['basin', 'centroid', 'pour_pt']:
            # create GIST spatial index on the geometry columns
            print(f'    Creating spatial index on {col} column...')
            cur.execute(f'CREATE INDEX IF NOT EXISTS {col}_idx ON {schema_name}.basin_attributes USING GIST({col});')

        extras.execute_values(cur, query, tuples)
        
        conn.commit()
        t6 = time()
        print(f'    Inserted {len(tuples)} rows in {t6-t4:.2f} seconds.')
        # commit the changes
        print(f'    Changes committed.  Finished processing {rc}')
        print('-------------------------------------------------')
        print('')
        conn.close()
        del df


if __name__ == '__main__':

    db_host = 'localhost'
    db_name = 'basins'
    db_user = 'postgres'
    db_password = 'pgpass'
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_data_dir = os.path.join(base_dir, 'input_data')
    data_dir = os.path.join(base_dir, 'processed_data')
    parquet_dir = os.path.join(data_dir, 'derived_basins/')
    
    bc_bounds_file = os.path.join(input_data_dir, 'region_bounds/BC_study_region_polygon_4326.geojson')
    t0 = time()
    bc_bounds = gpd.read_file(bc_bounds_file)
    t1 = time()
    print(f'Loaded BC bounds in {t1 - t0:.2f} seconds.  crs={bc_bounds.crs}')

    # get the total bounds of the study area
    #  minx, miny, maxx, maxy
    total_bounds = bc_bounds.total_bounds
    t2 = time()
    print(f'Retrieved total bounds in {t2 - t1:.2f} seconds.')
    schema_name = 'basins_schema'

    taa = time()
    convert_parquet_to_postgis_db(
        parquet_dir,
        db_host, db_name, db_user, db_password, 
        schema_name, total_bounds)
    
    tbb = time()
    print(f'Finished populating postgis database in {tbb - taa:.2f} seconds.')
    