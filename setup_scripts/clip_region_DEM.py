import os

import zipfile

import time

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import geopandas as gpd
import rioxarray as rxr

from shapely.geometry import Polygon

from shapely.validation import make_valid

# specify the DEM source
DEM_source = 'USGS_3DEP'
output_dem_crs = 3005

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'input_data/')
DEM_DIR = os.path.join(DATA_DIR, 'DEM/')
mask_dir = os.path.join(DATA_DIR, 'region_polygons/')

# dem tile mosaic "virtual raster"
mosaic_path = os.path.join(DATA_DIR, f'{DEM_source}_DEM_mosaic_4269.vrt')


def get_crs_and_resolution(fname):
    raster = rxr.open_rasterio(fname)
    crs = raster.rio.crs.to_epsg()
    res = raster.rio.resolution()   
    return crs, res


dem_crs, (w_res, h_res) = get_crs_and_resolution(mosaic_path)


def check_mask_validity(mask_path):
    """Checks the validity of a mask and corrects it if necessary.

    Args:
        mask_path (str): location of polygon mask

    Raises:
        Exception: if the mask is a GeometryCollection as opposed to Polygon or MultiPolygon
    """

    mask = gpd.read_file(mask_path)
    gtype = mask.geometry.values[0].geom_type
    if gtype == 'GeometryCollection':
        raise Exception; 'geometry should not be GeometryCollection'

    if mask.geometry.is_valid.values[0]:
        print(f'   ...mask is valid.')
        # mask.to_file(mask_path, driver='GeoJSON')
    else:
        file = mask_path.split('/')[-1]
        print(f'   ...{file} mask is invalid:')
        mask['geometry'] = mask.geometry.apply(lambda m: make_valid(m))
        mask = mask.dissolve()
        # drop invalid geometries
        mask = mask[mask.geometry.is_valid]
        mask['area'] = mask.geometry.area
        # reproject to 4326 to correspond with DEM tile mosaic
        mask = mask.to_crs(4326)
          
        if all(mask.geometry.is_valid):
            mask.to_file(mask_path, driver='GeoJSON')
            print(f'   ...invalid mask corrected.')
        else:
            print(f'   ...invalid mask could not be corrected')


if not os.path.exists(DATA_DIR + 'region_polygons'):
    with zipfile.ZipFile(mask_dir + 'region_polygons.zip', 'r') as zip_ref:
        zip_ref.extractall(mask_dir + 'region_polygons')

all_masks = [e for e in os.listdir(mask_dir) if e.endswith('.geojson')]

# get the region code shorthand
region_codes = [e.split('_')[0] for e in all_masks]

# or set a custom list of masks to process
region_codes = ['08P']

i = 0
for code in region_codes:
    t0 = time.time()
    file = [e for e in all_masks if code in e][0]
    
    fpath = os.path.join(mask_dir, file)

    if '_' not in file:
        splitter = '.'
    else:
        splitter = '_'
    grp_code = file.split(splitter)[0]
    print(f'Starting polygon merge on {grp_code}.')
    
    named_layer = file.split('.')[0]

    mask_check = check_mask_validity(fpath)

    # set the output initial path and reprojected path
    out_path = f'{DATA_DIR}/processed_dem/{grp_code}_{DEM_source}_{dem_crs}.tif'
    out_path_reprojected = f'{DATA_DIR}/processed_dem/{grp_code}_{DEM_source}_3005.tif'

    # if you want to modify the resulting DEM resolution,
    # you'll need to replace 1 with some other factor here
    rfactor = 1
    trw = abs(w_res*rfactor)
    trh = abs(h_res*rfactor)

    if not os.path.exists(out_path_reprojected):

        if rfactor == 1:
            command = f'gdalwarp -s_srs epsg:{dem_crs} -cutline {fpath} -cl {named_layer} -crop_to_cutline -multi -of gtiff {mosaic_path} {out_path} -wo NUM_THREADS=ALL_CPUS'
        else:
            command = f'gdalwarp -s_srs epsg:{dem_crs} -cutline {fpath} -cl {named_layer} -ot Float32 -crop_to_cutline -tr {trw} {trh} -multi -of gtiff {mosaic_path} {out_path} -wo CUTLINE_ALL_TOUCHED=TRUE -wo NUM_THREADS=ALL_CPUS'
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
    print(f'      {i+1}/{len(all_masks)} Completed tile merge: {out_path_reprojected} created in {t1-t0:.1f}s.')
    print('')
    print('')
    i += 1
    

# update the vrt with the resulting dem tif files 
# as the .bil files have been deleted.
output_mosaic_path = mosaic_path.replace('.vrt', f'_clipped_{output_dem_crs}.vrt')
sys_command = f'gdalbuildvrt -resolution highest -a_srs EPSG:{output_dem_crs} {output_mosaic_path} {DATA_DIR}processed_dem/*_{DEM_source}_3005.tif'

os.system(sys_command)