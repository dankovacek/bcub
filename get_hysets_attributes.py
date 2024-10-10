import pandas as pd
import numpy as np
import geopandas as gpd
import psycopg2

columns = [
    'official_id', 'watershed_id', 'name', 'centroid_lat_deg_n', 'centroid_lon_deg_e', 
    'drainage_area_km2', 'drainage_area_gsim_km2', 'flag_gsim_boundaries', 'flag_artificial_boundaries', 
    'elevation_m', 'slope_deg', 'gravelius', 'perimeter', 'flag_shape_extraction', 'aspect_deg', 'flag_terrain_extraction', 
    'land_use_forest_frac_2010', 'land_use_grass_frac_2010', 'land_use_wetland_frac_2010', 
    'land_use_water_frac_2010', 'land_use_urban_frac_2010', 'land_use_shrubs_frac_2010', 
    'land_use_crops_frac_2010', 'land_use_snow_ice_frac_2010', 'flag_land_use_extraction', 
    'logk_ice_x100', 'porosity_x100', 'flag_subsoil_extraction', 'year_from', 'year_to', 
    'record_length', 'agency', 'status', 'updated_official_basin', 'in_bcub', 
    'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp', 
    'high_prcp_freq', 'high_prcp_duration', 'low_prcp_freq', 'low_prcp_duration'
]

def get_hysets_attributes(columns, conn):
    """
    Get hysets attributes by hysets_id
    :param hysets_id: int
    :param conn: psycopg2 connection
    :return: pd.DataFrame
    """
    # Get hysets attributes
    if columns is None:
        query = f"""
        SELECT * FROM basins_schema.hysets_basins
        WHERE in_bcub = True
        LIMIT 10;
        """
        hysets_attributes = gpd.read_postgis(query, conn, geom_col='centroid')
    else:
        column_string = ', '.join(columns)
        query = f"""
        SELECT {column_string} FROM basins_schema.hysets_basins
        WHERE in_bcub = True
        ORDER BY official_id;
        """
        hysets_attributes = pd.read_sql_query(query, conn)
    return hysets_attributes

conn_info = {   
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432'
}
conn = psycopg2.connect(**conn_info)

df = get_hysets_attributes(columns, conn)
df.to_csv('BCUB_HYSETS_properties_with_climate.csv', index=False)