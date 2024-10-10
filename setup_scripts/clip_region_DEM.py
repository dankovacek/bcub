import os
import zipfile
import time
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
import rioxarray as rxr

# from shapely.geometry import Polygon
from shapely.validation import make_valid

from basin_processing_functions import get_crs_and_resolution, fill_holes

# specify the DEM source
DEM_source = 'USGS_3DEP'
output_dem_crs = 3005

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'input_data/')
DEM_DIR = os.path.join(DATA_DIR, 'DEM/')
PROCESSED_DEM_DIR = os.path.join(BASE_DIR, 'processed_data/processed_dem/')

# region_polygon_folder = os.path.join(DATA_DIR, 'adjusted_region_polygons/')
region_polygon_folder = os.path.join(DATA_DIR, 'adjusted_region_polygons/')
if not os.path.exists(PROCESSED_DEM_DIR):
    os.mkdir(PROCESSED_DEM_DIR)

# dem tile mosaic "virtual raster"
mosaic_path = os.path.join(BASE_DIR, f'processed_data/{DEM_source}_DEM_mosaic_4269.vrt')

dem_crs, (w_res, h_res) = get_crs_and_resolution(mosaic_path)

def check_mask_validity(mask_path, dem_crs, region_code):
    """Checks the validity of a mask and corrects it if necessary.

    Args:
        mask_path (str): location of polygon mask

    Raises:
        Exception: if the mask is a GeometryCollection as opposed to Polygon or MultiPolygon
    """

    mask = gpd.read_file(mask_path)

    mask = mask.to_crs(dem_crs)
    temp_mask_fname = mask_path.replace('.geojson', '_temp_mask.shp')   
    # check if any geometry types are GeometryCollection
    if any(mask.geometry.geom_type == 'GeometryCollection'): 
        raise Exception; 'geometry should not be GeometryCollection'
    
    if mask.geometry.is_valid.values.all():
        print(f'   ...mask is valid.')
        mask.to_file(temp_mask_fname, driver='ESRI Shapefile', layer=region_code)
    else:
        file = mask_path.split('/')[-1]
        print(f'   ...{file} mask is invalid.  Attempting to repair.')
        mask = mask.to_crs(3005)
        # mask = mask.explode(index_parts=False).reset_index(drop=True)
        mask['geometry'] = mask.geometry.apply(lambda m: make_valid(m))
        mask = mask.explode(index_parts=False).reset_index(drop=True)
        mask['area'] = mask.geometry.area  / 1e6
        mask = mask[mask['area'] > 1]
        mask = mask.to_crs(dem_crs)
        
        # drop invalid geometries
        mask = mask[mask.geometry.is_valid]
        
        # reproject to correspond with DEM tile mosaic
        mask = mask[mask.geometry.geom_type == 'Polygon']
        
        if all(mask.geometry.is_valid):
            mask = mask.dissolve()
            mask.to_file(mask_path, driver='GeoJSON')
            mask.to_file(temp_mask_fname, driver='ESRI Shapefile', layer=region_code)            
            print(f'   ...invalid mask corrected.')
        else:
            print(f'   ...invalid mask could not be corrected')
    return temp_mask_fname


region_codes = [
    '08A', '08B', '08C', '08D',
    '08E', '08F', '08G', '10E',
    'HGW', 'VCI', 'WWA', 'HAY',
    'CLR', 'PCR', 'FRA',
    'ERK',  'YKR', 'LRD',
    ]
region_codes = ['CLR']
# or set a custom list of masks to process
i = 0
for region_code in region_codes:
    t0 = time.time()

    cutline_fname = f'{region_code}_4269_dem_clip_final_RA.geojson'    
    named_layer = region_code

    cutline_fpath = os.path.join(region_polygon_folder, 'adjusted_clipping_masks', cutline_fname)
    # open the file to get the named_layer
    mask_df = gpd.read_file(cutline_fpath)
    # named_layer = region_code
    named_layer = f'{region_code}_4269_dem_clip_final_RA'
    print(f'Starting DEM clipping on {region_code}.')
    
    temp_mask_fpath = check_mask_validity(cutline_fpath, dem_crs, region_code)
    # named_layer = temp_mask_fpath.split('/')[-1].split('.')[0]

    # set the output initial path and reprojected path
    out_path = os.path.join(PROCESSED_DEM_DIR, f'{region_code}_{DEM_source}_{dem_crs}.tif')
    out_path_reprojected = os.path.join(PROCESSED_DEM_DIR, f'{region_code}_{DEM_source}_3005.tif')

    # if you want to modify the resulting DEM resolution,
    # you'll need to replace 1 with some other factor here
    rfactor = 1
    trw = abs(w_res*rfactor)
    trh = abs(h_res*rfactor)

    if not os.path.exists(out_path_reprojected):
        if rfactor == 1:
            command = f'gdalwarp -s_srs epsg:{dem_crs} -cutline {cutline_fpath} -cl {named_layer} -crop_to_cutline -multi -co "TILED=YES" -of gtiff -co "COMPRESS=LZW" -wo NUM_THREADS=ALL_CPUS {mosaic_path} {out_path}'
        else:
            command = f'gdalwarp -s_srs epsg:{dem_crs} -cutline {temp_mask_fpath} -cl {named_layer} -ot Float32 -crop_to_cutline -tr {trw} {trh} -multi -of gtiff {mosaic_path} {out_path} -wo CUTLINE_ALL_TOUCHED=TRUE -wo NUM_THREADS=ALL_CPUS'
        print('')
        print('__________________')
        print(command)
        print('')
        try:
            os.system(command)
        except Exception as e:
            raise Exception; e
    else:
        fname = out_path_reprojected.split('/')[-1]
        print(f'   ...{fname} exists, skipping dem cutline operation..')

    if not os.path.exists(out_path_reprojected):
        # reproject to epsg 3005
        lr = rxr.open_rasterio(out_path, masked=True, default='dem')
        lr = lr.rio.reproject(3005)
        lr.rio.to_raster(out_path_reprojected)
        # test to make sure the reprojected file is valid
        if os.path.exists(out_path_reprojected):
            valid_raster = rxr.open_rasterio(out_path_reprojected)
            lr_shape = valid_raster.rio.shape
            n_pix = lr_shape[0] * lr_shape[0]
            print(f'   ...img has {n_pix:.2e} pixels')
            # remove the un-projected file
            os.remove(out_path)
    else:
        fname = out_path_reprojected.split('/')[-1]
        print(f'   ...{fname} exists, skipping dem reprojection..')
    
    t1 = time.time()
    print(f'      {i+1}/{len(region_codes)} Completed tile merge: {out_path_reprojected} created in {t1-t0:.1f}s.')
    print('')
    i += 1
    if os.path.exists(temp_mask_fpath):
        os.remove(temp_mask_fpath)
    

# update the vrt with the resulting dem tif files 
# as the .bil files have been deleted.
output_mosaic_path = mosaic_path.replace('_4269.vrt', f'_clipped_{output_dem_crs}.vrt')

sys_command = f'gdalbuildvrt -resolution highest -a_srs EPSG:{output_dem_crs} {output_mosaic_path} {PROCESSED_DEM_DIR}/*_{DEM_source}_3005.tif'

os.system(sys_command)