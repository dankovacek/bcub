# Here we merge the HydroBASINS polygons for each region into a single polygon.
# we generate an ocean mask from the NALCMS data and use it to adjust the coastline boundary of the region polygons.
# we create two versions of the region polygons: 
#     one with the coastline adjusted to the ocean mask, and 
#     one with the coastline buffered by 5km to prevent basin delineation restriction.
# this script also handles the confounding issue of the buffer causing double coverage 
# of small islands near region boundaries.  
# The final output includes the following sets of geometries:
#    i) one file with the covering set of polygons for display / organization
#    ii) one set of region polygons with a buffer of 5 km added to prevent 
#        basin delineation restriction from lower resolution HydroBASINS polygons
#        these polygons should be used to clip DEM for raster processing,
#        but the pour point search should use the unbuffered polygons.
#    iii) a set of tiled ocean polygons used for masking (avoids stream vestiges along coastlines
#    iv) a set of ice mass polygons 
#    v) a set of lake polygons for filtering pour points -- this is considered more detailed and
#       accurate than the HydroBASINS lake polygons since it is based on 30m NAMCLS data.

from time import time
import os
# os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import rioxarray as rxr 
# import dask_geopandas as dgpd
import numpy as np
# from multiprocessing import Pool

from osgeo import gdal, ogr, osr

from shapely.geometry import Point
from shapely.validation import make_valid
from basin_processing_functions import get_crs_and_resolution, fill_holes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'input_data/BasinATLAS/HydroBASINS/')
NALCMS_DIR = os.path.join(BASE_DIR, 'input_data/', 'NALCMS/')
HYSETS_DIR = os.path.join(BASE_DIR, 'input_data/', 'HYSETS_data/')
id_list_fpath = os.path.join(DATA_DIR, 'HYDRO_basin_ids.csv')
output_folder = os.path.join(BASE_DIR, 'input_data/region_polygons')

DEM_source = 'USGS_3DEP'
mosaic_path = os.path.join(BASE_DIR, f'processed_data/{DEM_source}_DEM_mosaic_4269.vrt')
dem_crs, (w_res, h_res) = get_crs_and_resolution(mosaic_path)

id_list = pd.read_csv(id_list_fpath)

region_codes = sorted(list(set(id_list['region_code'].values)))


def open_file_and_retrieve_ids(file, ids):
    gdf = gpd.read_file(os.path.join(DATA_DIR, file))
    return gdf[gdf['HYBAS_ID'].isin(ids)]


def process_region_polygon(rc, ocean_mask):
    """
    Care needs to be taken here to set CRS to a consistent value.
    The final mask should be in the same CRS as the DEM (EPSG:4269).
    Many steps require the geometries to be in a projected crs, 
    so we'll use EPSG:3005 for the intermediate steps.
    """
    proj_crs = f'EPSG:3005'
    t0 = time()
    region_info = id_list[id_list['region_code'] == rc].copy()
    # the region files are the HydroBASINS polygons we
    # assemble to use as a starting point for the DEM clipping mask
    region_files = list(set(region_info['file'].values))
    region_polygons = []
    
    print(f'Processing {rc}')
    for rf in region_files:
        region = region_info[region_info['file'] == rf].copy()
        ids = region['HYBAS_ID'].values
        polygons = open_file_and_retrieve_ids(rf, ids)
        region_polygons.append(polygons)
    region_gdf = gpd.GeoDataFrame(pd.concat(region_polygons))
    region_gdf = region_gdf.to_crs(proj_crs)
    t1 = time()
    print(f'   ...{rc} loaded in {t1-t0:.2f} seconds.')

    region_gdf = region_gdf.explode(index_parts=False)
    region_gdf['geometry'] = fill_holes(region_gdf.geometry)
    region_gdf = region_gdf.dissolve() 
    region_gdf.sindex
    t2 = time()

    dem_clipping_mask = region_gdf.copy()
    # set a buffer distance of 5km -- this is to prevent the 
    # HYDROBasins polygons from restricting the basin delineation
    buff = 5000
    if rc in ['HGW', 'VCI']:
        buff = 250
    dem_clipping_mask = gpd.GeoDataFrame(geometry=dem_clipping_mask.buffer(buff).values, crs=proj_crs)
    
    # explode the multipolygon and fill holes
    dem_clipping_mask = dem_clipping_mask.explode(index_parts=True)
    dem_clipping_mask['geometry'] = fill_holes(dem_clipping_mask.geometry)

    if ocean_mask.crs != dem_clipping_mask.crs:
        ocean_mask = ocean_mask.to_crs(proj_crs)
    # adjust the coastlines to match the ocean mask
    # first, check if the region intersects the ocean mask    
    ocean_intersection = ocean_mask.intersects(dem_clipping_mask.unary_union)
    t3 = time()
    print(f'   ...checking if the region intersects the ocean mask: {ocean_intersection.any()} ({t3-t2:.2f}s.)')
    buffered_gdf = dem_clipping_mask.copy()
    if ocean_intersection.any():
        # dissolve the intersecting ocean mask
        print(f'   ...{rc} buffered and dissolved the ocean mask.  Adjusting coastline boundary.')
        coastline_tiles = gpd.overlay(dem_clipping_mask, ocean_mask, how='intersection')
        # simplify and buffer the ocean mask to prevent capture of ocean near shoreline
        # otherwise vestigial rivers appear along the shoreline.
        coastal_buffer = 5 * 30
        coastline_tiles.geometry = coastline_tiles.buffer(coastal_buffer).simplify(30)
        coastline_tiles = coastline_tiles.dissolve()
        coastline_tiles = coastline_tiles.explode(index_parts=False)
        coastline_tiles.geometry = fill_holes(coastline_tiles.geometry)
        t3b = time()
        print(f'   ...intersected region polygon with ocean mask in {t3b-t3:.2f} seconds.')
        print('    ...subtracting the ocean mask from the combined polygon.')
        assert region_gdf.crs == coastline_tiles.crs == buffered_gdf.crs
        region_gdf = gpd.overlay(region_gdf, coastline_tiles, how='difference', keep_geom_type=True)
        print(f'   ...{rc} subtracting buffered intersection.')
        buffered_gdf = gpd.overlay(buffered_gdf, coastline_tiles, how='difference', keep_geom_type=True)

        # Now extend the region polygon to the ocean mask 
        # by taking advantage that the largest area of the 
        # buffered polygon is along the interior
        print(f'   ...{rc} extending the region polygon to the ocean mask.')
        diff_gdf = gpd.overlay(region_gdf, buffered_gdf, how='symmetric_difference', keep_geom_type=True)
        # check and make geometries valid
        diff_gdf = diff_gdf.explode(index_parts=False)
        diff_gdf['geometry'] = diff_gdf.geometry.apply(lambda g: make_valid(g))
        diff_gdf = diff_gdf[diff_gdf.geometry.is_valid]
        diff_gdf['area'] = diff_gdf.geometry.area
        # remove the largest area component (the land buffer left after clipping the coastline)
        diff_gdf = diff_gdf[diff_gdf['area'] != diff_gdf['area'].max()]
        # recombine the differenced set with the region_polygon
        region_gdf = gpd.overlay(region_gdf, diff_gdf, how='union', keep_geom_type=True)
        # fill holes
        region_gdf = region_gdf.dissolve()
        region_gdf = region_gdf.explode(index_parts=False)
        region_gdf['geometry'] = fill_holes(region_gdf.geometry)

        # region_gdf = region_gdf.to_crs(f'EPSG:3005')
        # buffered_gdf = buffered_gdf.to_crs(f'EPSG:3005')
        region_gdf = region_gdf.explode(index_parts=False)
        buffered_gdf = buffered_gdf.explode(index_parts=False)
        region_gdf['area'] = region_gdf.geometry.area / 1e6
        buffered_gdf['area'] = buffered_gdf.geometry.area / 1e6
        region_gdf = region_gdf[region_gdf['area'] > 1]
        buffered_gdf = buffered_gdf[buffered_gdf['area'] > 1]
    region_gdf = region_gdf.to_crs(f'EPSG:{dem_crs}')
    buffered_gdf = buffered_gdf.to_crs(f'EPSG:{dem_crs}')
    region_gdf = region_gdf.dissolve()
    buffered_gdf = buffered_gdf.dissolve()
    
    return region_gdf, buffered_gdf
    

# the clipping masks provided have been manually adjusted, 
# so don't run this unless you want to redo this process!!
# unbuffered_files = []
# for rc in ['HAY']:#['CLR']:# region_codes:
#     region_fname = f'{rc}_{dem_crs}.geojson'
#     region_fpath = os.path.join(output_folder, region_fname)
#     unbuffered_files.append(region_fpath)
#     if not os.path.exists(region_fpath):
#         df, buffered_df = process_region_polygon(rc, ocean_mask)
#         df.to_file(region_fpath)
#         buffered_region_fname = f'{rc}_{dem_crs}_dem_clipping_mask.geojson'
#         buffered_region_fpath = os.path.join(output_folder, buffered_region_fname)
#         buffered_df.to_file(buffered_region_fpath)

regions_fpath = os.path.join(BASE_DIR, 'input_data', 'BCUB_regions_merged_R0.geojson')
if not os.path.exists(regions_fpath):
    region_dfs = []
    updated_region_polygon_folder = os.path.join(BASE_DIR, 'input_data', 'region_polygons')
    updated_region_polygon_files = os.listdir(updated_region_polygon_folder)
    updated_region_polygon_files = [e for e in updated_region_polygon_files if e.endswith('_R0.geojson')]

    print('Region polygon files found:')
    for f in updated_region_polygon_files:        
        print(f'    {f}')
    for file in updated_region_polygon_files:
        region_df = gpd.read_file(os.path.join(updated_region_polygon_folder, file))
        region_df = region_df.explode(index_parts=False)
        region_df.geometry = region_df.buffer(0)
        region_df.geometry = region_df.geometry.apply(lambda g: make_valid(g))
        merged = gpd.GeoDataFrame(geometry=[region_df.unary_union], crs=region_df.crs)
        merged = merged.explode(index_parts=False)
        merged.geometry = fill_holes(merged.geometry)
        merged.geometry = merged.buffer(0)

        merged['region_code'] = file.split('_')[0]
        crs = region_df.crs
        region_dfs.append(merged)

    all_regions = gpd.GeoDataFrame(pd.concat(region_dfs), crs=crs)[['region_code', 'geometry']]
    all_regions.to_file(regions_fpath, driver='GeoJSON')
else:
    all_regions = gpd.read_file(regions_fpath)


# open the hysets properties file
hysets_properties_fpath = os.path.join(HYSETS_DIR, 'HYSETS_watershed_properties.txt')
hysets_df = pd.read_csv(hysets_properties_fpath, sep=';')

hysets_centroids = hysets_df.apply(lambda row: Point(row['Centroid_Lon_deg_E'], row['Centroid_Lat_deg_N']), axis=1)
hysets_gdf = gpd.GeoDataFrame(hysets_df, geometry=hysets_centroids, crs='EPSG:4326')
hysets_gdf = hysets_gdf.to_crs(all_regions.crs)

hysets_gdf = hysets_gdf.to_crs(3005)

all_contained_pts = []
for rc in all_regions['region_code'].unique():
    print(f'Processing {rc}')
    region_gdf = all_regions[all_regions['region_code'] == rc].copy()
    
    region_gdf = region_gdf.to_crs(3005)
    # simplify the geometry and add a 300m buffer to capture stations that are located just outside the region polygon
    region_gdf['geometry'] = region_gdf.simplify(100)
    region_gdf['geometry'] = region_gdf.buffer(500)
    # do a spatial join to get the hysets points contained within any of the region polygons
    hysets_in_regions = gpd.sjoin(hysets_gdf, region_gdf, predicate='intersects', how='inner')    

    # drop the index_right column
    hysets_in_regions = hysets_in_regions.drop(columns=['index_right'])
    # add 09AG003 to the YKR region
    if rc == 'YKR':
        added_stn = hysets_gdf[hysets_gdf['Official_ID'] == '09AG003'].copy()
        # add the stn to hysets_in_regions
        hysets_in_regions = gpd.GeoDataFrame(pd.concat([hysets_in_regions, added_stn]), crs=hysets_in_regions.crs)
    elif rc == 'LRD':
        added_stn = hysets_gdf[hysets_gdf['Official_ID'] == '10ED002'].copy()
        hysets_in_regions = gpd.GeoDataFrame(pd.concat([hysets_in_regions, added_stn]), crs=hysets_in_regions.crs)
        
    hysets_in_regions['region_code'] = rc
    all_contained_pts.append(hysets_in_regions)

output = gpd.GeoDataFrame(pd.concat(all_contained_pts), crs=hysets_in_regions.crs)
output.to_file(os.path.join(HYSETS_DIR, 'HYSETS_BCUB_stations_R0.geojson'), driver='GeoJSON')
print(f'{len(output)} stations found in the region polygons.')