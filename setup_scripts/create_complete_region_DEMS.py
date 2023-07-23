import os
import sys

import time

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import geopandas as gpd
import rioxarray as rxr

from shapely.geometry import Polygon

from shapely.validation import make_valid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')

mask_dir = DATA_DIR + f'merged_basin_groups/final_polygons/'

processed_dem_dir = os.path.join(BASE_DIR, 'processed_data/processed_dem')

t0 = time.time()

# specify the DEM source
# either 'EarthEnv_DEM' or 'USGS_3DEP'
DEM_source = 'EENV_DEM'
# DEM_source = 'USGS_3DEP'

epsg_code = 4326
vrt_file = f'{DEM_source}_mosaic_4326.vrt'
if DEM_source == 'USGS_3DEP':
    epsg_code = 4269
    vrt_file = f'{DEM_source}_mosaic_{epsg_code}.vrt'
    # dem_dir = '/media/danbot/Samsung_T5/geospatial_data/DEM_data/'

dem_mosaic_file = os.path.join(processed_dem_dir, vrt_file)

bc_region_final_polygons_folder = DATA_DIR + 'merged_basin_groups/final_polygons/'

save_path = DATA_DIR + f'processed_dem/{DEM_source}/'
# save_path = f'/media/danbot/Samsung_T5/geospatial_data/DEM_data/processed_dem/{DEM_source}/'


if not os.path.exists(save_path):
    os.makedirs(save_path)


def get_crs_and_resolution(fname):
    raster = rxr.open_rasterio(fname)
    crs = raster.rio.crs.to_epsg()
    res = raster.rio.resolution()   
    return crs, res


dem_crs, (w_res, h_res) = get_crs_and_resolution(dem_mosaic_file)


def check_mask_validity(mask_path):
    # m = '08NM071_original.geojson'
    # mask_path = f'/home/danbot/Documents/code/hysets_validation/setup_scripts/DEM_treatment_test/{m}'
    # mask_path = '/media/danbot/Samsung_T5/geospatial_data/WSC_data/2021-12-basins/all/08NM071/basin/08NM071_DrainageBasin_BassinDeDrainage.shp'
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
        
        fixed = all(mask.geometry.is_valid)
        
        if fixed:
            mask.to_file(mask_path, driver='GeoJSON')
            print(f'   ...invalid mask corrected: {fixed}')
            # mask.to_file('DEM_treatment_test/08NM071_original.geojson')
            # print(asdfsad)
            # mask = mask.to_crs(3005)
            # mask = mask.buffer(1000, resolution=16, single_sided=True)
            # m = '08NM071_buffered.geojson'
            # mask.to_file(f'DEM_treatment_test/{m}')
        else:
            print(f'   ...invalid mask could not be corrected')


all_masks = [e for e in os.listdir(bc_region_final_polygons_folder) if e.endswith('.geojson')]

all_codes = [e.split('_')[0] for e in all_masks]
# all_codes = [e for e in all_codes if e in ['07U', 'ERockies', '08N']]

i = 0
for code in all_codes:

    file = [e for e in all_masks if code in e][0]
    
    fpath = bc_region_final_polygons_folder + file

    if '_' not in file:
        splitter = '.'
    else:
        splitter = '_'
    grp_code = file.split(splitter)[0]
    print(f'Starting polygon merge on {grp_code}.')

    
    named_layer = file.split('.')[0]

    mask_check = check_mask_validity(fpath)
    
    for res in ['res1']:
        #['res8', 'res4', 'res2', 'res1']
        # set the output initial path and reprojected path
        out_path = f'{save_path}{grp_code}_{DEM_source}_{epsg_code}_{res}.tif'
        out_path_reprojected = f'{save_path}{grp_code}_{DEM_source}_3005_{res}.tif'

        if res == 'res8':
            rfactor = 8
        elif res == 'res4':
            rfactor = 4
        elif res == 'res2':
            rfactor = 2
        else:
            rfactor = 1

        trw = abs(w_res*rfactor)
        trh = abs(h_res*rfactor)

        if not os.path.exists(out_path_reprojected):

            if rfactor == 1:
                command = f'gdalwarp -s_srs epsg:{epsg_code} -cutline {fpath} -cl {named_layer} -crop_to_cutline -multi -of gtiff {dem_mosaic_file} {out_path} -wo NUM_THREADS=ALL_CPUS'
            else:
                command = f'gdalwarp -s_srs epsg:4326 -cutline {fpath} -cl {named_layer} -crop_to_cutline -tr {trw} {trh} -multi -of gtiff {dem_mosaic_file} {out_path} -wo CUTLINE_ALL_TOUCHED=TRUE -wo NUM_THREADS=ALL_CPUS'
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

        
        # check # pixels low res        
        if not os.path.exists(out_path_reprojected):
            # reproject to epsg 3005
            lr = rxr.open_rasterio(out_path, masked=True, default='dem')
            lr = lr.rio.reproject(3005)
            lr.rio.to_raster(out_path_reprojected)
            lr_shape = lr.rio.shape
            n_pix = lr_shape[0] * lr_shape[0]
            print(f'   ...{res} img has {n_pix:.2e} pixels')
            os.remove(out_path)
        else:
            fname = out_path_reprojected.split('/')[-1]
            print(f'   ...{fname} exists, skipping dem reprojection..')
    
    t1 = time.time()
    print(f'      {i}/{len(all_masks)} Completed tile merge: {out_path_reprojected} created in {t1-t0:.1f}s.')
    print('')
    print('')
    i += 1
    