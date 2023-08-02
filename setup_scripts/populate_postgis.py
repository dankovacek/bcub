import geopandas as gpd
import dask_geopandas as dgp
import pyarrow.parquet as pq
import psycopg2
import psycopg2.extras as extras
import os
from time import time
import pandas as pd

from dask import dataframe as dd

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
    'geometry': 'geom',
    'centroid_geometry': 'centroid',
    'basin_geometry': 'basin',
}   

def create_table_initialization_query(parquet_schema, schema_name, db_host, db_name, db_user, db_password):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    cur.execute(f'CREATE SCHEMA IF NOT EXISTS {schema_name};')
    print('    Schema created.')
    # metadata = parquet_schema.metadata
        # Create a list of column names and types
    geom_columns = [c for c in parquet_schema.names if 'geometry' in c]
    
    # print(geom_columns)
    # print(asdfsd)

    # despite no geometry types of multipolygon found, 
    # multipolygon type error for polygon specified column persists
    # to fix, try switching to generic Geometry type:
    # ALTER TABLE my_table ALTER COLUMN geom TYPE geometry(Geometry,3005);

    q = f'''CREATE TABLE IF NOT EXISTS {schema_name}.basin_attributes (
        id SERIAL PRIMARY KEY,
        geom geometry(POINT, 3005),
        centroid geometry(POINT, 3005),
        basin geometry(POLYGON, 3005),
        '''
        
    column_names = []
    column_types = []
    for field in parquet_schema:
        ###

        ### this needs to be changed to set dtypes manually

        ####
        if field.name not in geom_columns:
            column_names.append(field.name)
            column_types.append(dtype_dict[field.type])

    for col, col_type in zip(column_names, column_types):
        if col == 'ID':
            col = 'object_id'
        if col not in geom_columns:
            q += f' {col} {col_type},'
    q = q[:-1] + ');'
    cur.execute(q)
    conn.commit()
    conn.close()
    print('    Table created: basin_attributes.')
    

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
    for c in non_geo_cols:
        table_name = 'basin_attributes'
        dtype_name = df[c].dtype.name
        dtype = dtype_dict[dtype_name]
        if c in ['lulc_check', 'flag_acc_match']:
            dtype = 'BIT'
        
        if c == 'ID': 
            c = 'object_id'
            dtype = 'INT'
        
        # print(f'   {c} is of type {dtype_name} and will be converted to {dtype}.')
        cur.execute(f'ALTER TABLE {schema_name}.{table_name} ADD COLUMN IF NOT EXISTS {c} {dtype};')
    print('     Added ALL columns to table.')

    conn.commit()
    conn.close()


def test_query(schema_name, db_host, db_name, db_user, db_password):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    q = f"select id, geom from {schema_name}.basin_attributes order by id desc limit 4"
    try:
        test_df = gpd.read_postgis(q, conn, geom_col='geom')
    except Exception as e:
        print(e)
        return False, pd.DataFrame()

    return True, test_df


# @dask.delayed
def load_file_dask(filename, nparts, cols):
    ddf = dd.read_parquet(filename, split_row_groups=True, columns=cols)
    ddf = ddf.repartition(npartitions=nparts)
    return ddf

def load_file_gpd(filename):
    return gpd.read_parquet(filename)

# @delayed
def load_file_dgp(filename, schema):
    df = dgp.read_parquet(filename, 
                          calculate_divisions=True,
                          schema=schema 
                          )
    # df = df.repartition(npartitions=nparts)
    return df

# @delayed
def load_gpkg(filename, nparts, cols):
    df = dgp.read_file(filename, 
                    #    calculate_divisions=True, 
                    #    split_row_groups=True, 
                    #    columns=cols
                       )
    df = df.repartition(npartitions=nparts)
    return df


def spatial_repartition(df, schema, fc, nparts):
    hilbert = df.spatial_shuffle(by="hilbert", npartitions=nparts)
    hilbert.to_parquet(
        "processed_data/partitioned_parquet", 
        compression="snappy",
        name_function=lambda i: f"{fc}_hilbert-{i:05d}.parquet",
        )

def reformat_geom_cols(df, geometry_cols, nparts):
    # encode geometries to wkb
    for c in geometry_cols:
        df[c+'_wkb'] = df[c].to_wkb(hex=True)

    # drop original geometry columns    
    drop_cols = ['geometry', 'basin_geometry', 'centroid_geometry']    
    df = df.drop(labels=drop_cols, axis=1)
    # return dgp.from_geopandas(df.compute(), npartitions=nparts)
    return df.compute()


def update_constraint(schema_name, db_host, db_name, db_user, db_password):
    conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
    cur = conn.cursor()
    query = f"ALTER TABLE {schema_name}.basin_attributes ADD CONSTRAINT basin_attributes_pkey PRIMARY KEY (id);"
    cur.execute(f'CREATE SCHEMA IF NOT EXISTS {schema_name};')


def convert_parquet_to_postgis_db(parquet_dir, db_host, db_name, db_user, db_password, 
                                     schema_name,  total_bounds):
    # connect to the PostGIS database    
    db_initialized, _ = test_query(schema_name, db_host, db_name, db_user, db_password)
    
    if not db_initialized:
        test_dir = os.path.join(parquet_dir, '08P')
        test_file = [e for e in os.listdir(test_dir) if e.endswith('.parquet')]
        if len(test_file) == 0:
            raise Exception(f'No parquet files found in {parquet_dir}.')

        parquet_schema = pq.read_schema(os.path.join(test_dir, test_file[0]))
        
        print(parquet_schema)
        print(asdfsd)

        print(f'    Postgis database table is empty, creating table and populating columns...')
        create_table_initialization_query(parquet_schema, schema_name, db_host, db_name, db_user, db_password)
        populate_table_columns(df, schema_name, geometry_cols, non_geo_cols, db_host, db_name, db_user, db_password)
    
    update_constraint(schema_name, db_host, db_name, db_user, db_password)
    
    minx, miny, maxx, maxy = total_bounds
    divisor = 1000
    minx = int(np.floor(minx / divisor) * divisor)
    miny = int(np.floor(miny / divisor) * divisor)
    maxx = int(np.ceil(maxx / divisor) * divisor)
    maxy = int(np.ceil(maxy / divisor) * divisor)


    # print(folders)
    # print(asdf)
    # completed = ['08A', '07U', '08F', 'ERockies', '08H', 
    # '08B', '09A', '07O', '08O', '08P', 
    # '08G', '07G', '08H', '08D', '08N', 
    # '08C', 'Peace']

    for rc in ['08P']:
            
        fpath = os.path.join(parquet_dir, rc)
        nparts = len(os.listdir(fpath))
        print(f'    Loading file {fpath}...')
        df = load_file_dgp(fpath, parquet_schema)
        

        t3 = time()

        # find all multipolygons
        # multi = df[df['geometry'].geom_type == 'MultiPolygon'].compute().copy()
        # if len(multi) > 0:
        #     print(len(multi))
        #     print(multi)
        #     print(asdfsadf)

        geometry_cols = [c for c in df.columns if 'geometry' in c]
        non_geo_cols = [c for c in df.columns if 'geometry' not in c]
        non_geo_cols = [c for c in non_geo_cols if c != 'ID']
        non_geo_cols = [c for c in non_geo_cols if c != 'index']

        if len(non_geo_cols) > len(set(non_geo_cols)):
            raise Exception('Duplicated column names in non-geometry columns.')

        df = reformat_geom_cols(df, geometry_cols, nparts)
        t4 = time()
        
        print(f'   Dask geopandas converted columns to wkb in {t4-t3:.2f} seconds.')
        
        # populate non-geometry values
        print('    Starting db insert...')
        
        conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
        cur = conn.cursor()

        print('    Database connected...')
        
        all_cols = [geom_dict[e] for e in geometry_cols] + non_geo_cols
        # print(all_cols)
        # res = df.apply(insert_vals_to_db, cur=cur, geom_cols=geometry_cols, non_geo_cols=non_geo_cols, axis=1)
        tuples = list(df[[e+'_wkb' for e in geometry_cols] + non_geo_cols].itertuples(index=False, name=None))
        for i in range(len(all_cols)):
            if all_cols[i] not in ['centroid', 'basin', 'geometry', 'geom', 'basin_geometry', 'centroid_geometry']: 
                print(all_cols[i], tuples[0][i])
        
        # insert the array of tuples into the db but 
        # handle the case where the tuple already exists
        string_vec = ', '.join(['%s' for e in all_cols])
        cols_str = ', '.join(all_cols)
        # we need to avoid duplicate geometries
        update_cols = [e for e in non_geo_cols if e != 'object_id']
        unique_cols = [geom_dict[e] for e in geometry_cols] + ['object_id']
        unique_cols_str = ', '.join(unique_cols)
        update_cols_line = ', '.join(update_cols)
        excluded_line = ', '.join([f'EXCLUDED.{e}' for e in update_cols])
        
        query = f'''INSERT INTO {schema_name}.basin_attributes ({cols_str}) VALUES %s;'''
        extras.execute_values(cur, query, tuples)
        
        # extras.execute_values(cur, query, tuples)
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
    
    bc_bounds_file = os.path.join(input_data_dir, 'BC_study_region_polygon_R0_3005.geojson')
    t0 = time()
    bc_bounds = gpd.read_file(bc_bounds_file)
    t1 = time()
    print(f'Loaded BC bounds in {t1 - t0:.2f} seconds.  crs={bc_bounds.crs}')

    # get the total bounds of the study area
    #  minx, miny, maxx, maxy
    total_bounds = bc_bounds.total_bounds
    t2 = time()
    print(f'Got total bounds in {t2 - t1:.2f} seconds.')
    schema_name = 'basins_schema'

    taa = time()
    convert_parquet_to_postgis_db(
        parquet_dir,
        db_host, db_name, db_user, db_password, 
        schema_name, total_bounds)
    
    tbb = time()
    print(f'Finished populating postgis database in {tbb - taa:.2f} seconds.')
    