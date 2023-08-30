# generate basins
import os
# import time
import shutil

import warnings
warnings.filterwarnings('ignore')

# import numpy as np
import geopandas as gpd
# import pandas as pd
import rioxarray as rxr

import basin_processing_functions as bpf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_folder = os.path.join(BASE_DIR, 'processed_data/processed_dem/')
region_files = os.listdir(DEM_folder)
region_codes = sorted(list(set([e.split('_')[0] for e in region_files])))

#########################
# input file paths
#########################
nalcms_dir = os.path.join(BASE_DIR, 'input_data/NALCMS/')
nalcms_fpath = os.path.join(nalcms_dir, 'NA_NALCMS_2010_v2_land_cover_30m.tif')
reproj_nalcms_path = os.path.join(BASE_DIR, 'input_data/NALCMS/NA_NALCMS_landcover_2010_3005_clipped.tif')

BasinATLAS_dir = os.path.join(BASE_DIR, 'input_data/BasinATLAS')

# masks used to clip the geospatial layers
mask_path = os.path.join(BASE_DIR, 'input_data/region_bounds/BC_study_region_polygon_4326.geojson')
reproj_bounds_path_4326 = os.path.join(BASE_DIR, 'input_data/region_bounds/convex_hull_4326.shp')
reproj_bounds_path_4269 = os.path.join(BASE_DIR, 'input_data/region_bounds/convex_hull_4269.shp')

if not os.path.exists(reproj_bounds_path_4269):
    mask = gpd.read_file(mask_path)
    mask.geometry = mask.convex_hull
    mask = mask.to_crs('EPSG:4269')        
    mask.to_file(reproj_bounds_path_4269)
if not os.path.exists(reproj_bounds_path_4326):
    mask = gpd.read_file(mask_path)
    mask.geometry = mask.convex_hull
    mask = mask.to_crs('EPSG:4326')        
    mask.to_file(reproj_bounds_path_4326)
       
# use the mask geometry to clip the GLHYMPS vector file
glhymps_dir = os.path.join(BASE_DIR, 'input_data/GLHYMPS/')
glhymps_fpath = os.path.join(glhymps_dir, 'GLHYMPS.shp')
hydroLakes_fpath = os.path.join(BASE_DIR, 'input_data/BasinATLAS/HydroLAKES_polys_v10.gdb/HydroLAKES_polys_v10.gdb')
hydroLakes_clipped_fpath = os.path.join(BASE_DIR, 'input_data/BasinATLAS/HydroLAKES_clipped.gpkg')
glhymps_clipped_fpath = os.path.join(glhymps_dir, 'GLHYMPS_clipped.gpkg')
reproj_glhymps_fpath = os.path.join(glhymps_dir, 'GLHYMPS_clipped_3005.geojson')
reproj_mask_glhymps_crs = os.path.join(BASE_DIR, 'input_data/region_bounds/convex_hull_glhymps_crs.shp')
reproj_mask_nalcms_crs = os.path.join(BASE_DIR, 'input_data/region_bounds/convex_hull_nalcms_crs.shp')

# clip and reproject the GLHYMPS vector data. 
if not os.path.exists(reproj_glhymps_fpath):
    # if not os.path.exists(glhymps_fpath):
    #     raise Exception('Download and unzip the GLHYMPS data, see the README for details.')
    
    layer_name = 'Final_GLHYMPS_Polygon'
    
    glhymps_original_wkt = """
    PROJCRS["World_Cylindrical_Equal_Area",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["World_Cylindrical_Equal_Area",
        METHOD["Lambert Cylindrical Equal Area",
            ID["EPSG",9835]],
        PARAMETER["Latitude of 1st standard parallel",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8823]],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1]],
    USAGE[
        SCOPE["Not known."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["ESRI",54034]]
    """
    
    # get the mask geometry and reproject it using the original GLHYMPS projection
    if not os.path.exists(reproj_mask_glhymps_crs):
        mask = gpd.read_file(reproj_bounds_path_4326).to_crs(glhymps_original_wkt)
        mask.to_file(reproj_mask_glhymps_crs)
    
    mask = gpd.read_file(reproj_mask_glhymps_crs)
    bounds_string = ' '.join([str(int(e)) for e in mask.bounds.values[0]])
    print('Clipping GLHYMPS vector file to region bounds.')
    # clip the GLHYMPS vector file to the region bounds using ogr2ogr
    clip_command = f"ogr2ogr -nlt PROMOTE_TO_MULTI -f GPKG -clipsrc {bounds_string} {glhymps_clipped_fpath} {glhymps_fpath}"
    if not os.path.exists(glhymps_clipped_fpath):
        print(clip_command)
        print('')
        os.system(clip_command)
    else:
        print('   ...clipped GLHYMPS file already exists.')
    
    # reproject the clipped GLHYMPS vector file to EPSG 3005
    print('Reprojecting clipped GLHYMPS vector file.')
    warp_command = f"ogr2ogr -t_srs EPSG:3005 -f GeoJSON {reproj_glhymps_fpath} {glhymps_clipped_fpath}"
    if not os.path.exists(reproj_glhymps_fpath):
        os.system(warp_command)
    
    # clean up temporary files
    if os.path.exists(reproj_glhymps_fpath):
        for f in [glhymps_clipped_fpath, glhymps_fpath, glhymps_clipped_fpath]:
            if os.path.exists(f):
                # if the file is a directory, remove it recursively
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
                

# clip and reproject the NALCMS raster data
if not os.path.exists(reproj_nalcms_path):
    
    if not os.path.exists(nalcms_fpath):
        raise Exception('Download and unzip the NALCMS data, see the README for details.')
    
    nalcms_data = rxr.open_rasterio(nalcms_fpath)
    nalcms_wkt = nalcms_data.rio.crs.wkt
    
    # get the mask geometry and reproject it using the original NALCMS projection
    if not os.path.exists(reproj_mask_nalcms_crs):
        mask = gpd.read_file(reproj_bounds_path_4326).to_crs(nalcms_wkt)
        mask.to_file(reproj_mask_nalcms_crs)
    
    # first clip the raster, 
    print('Clipping NALCMS raster to region bounds.')
    clipped_nalcms_path = os.path.join(BASE_DIR, 'input_data/NALCMS/NA_NALCMS_landcover_2010_clipped.tif')
    clip_command = f"gdalwarp -s_srs '{nalcms_wkt}' -cutline {reproj_mask_nalcms_crs} -crop_to_cutline -multi -of gtiff {nalcms_fpath} {clipped_nalcms_path} -wo NUM_THREADS=ALL_CPUS"
    if not os.path.exists(clipped_nalcms_path):
        os.system(clip_command)
    
    print('\nReprojecting clipped NALCMS raster.')
    # insert  "-co COMPRESS=LZW" in the command below to use compression (slower but much smaller file size)
    warp_command = f"gdalwarp -q -s_srs '{nalcms_wkt}' -t_srs EPSG:3005 -of gtiff {clipped_nalcms_path} {reproj_nalcms_path} -r bilinear -wo NUM_THREADS=ALL_CPUS"
    if not os.path.exists(reproj_nalcms_path):
        os.system(warp_command) 
    
    # remove the intermediate step
    if os.path.exists(clipped_nalcms_path):
        os.remove(clipped_nalcms_path)
        
# clip and reproject the HydroLAKES layer
if not os.path.exists(hydroLakes_clipped_fpath):
    mask = gpd.read_file(mask_path)
    bounds_string = ' '.join([str(int(e)) for e in mask.bounds.values[0]])
    print('Clipping GLHYMPS vector file to region bounds.')
    command = f'ogr2ogr -nlt PROMOTE_TO_MULTI -f GPKG -clipsrc {bounds_string} {hydroLakes_clipped_fpath} {hydroLakes_fpath}'
    os.system(command)
    
        