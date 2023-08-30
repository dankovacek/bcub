# generate basins
import os

import time
from turtle import clear

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr

from shapely.geometry import Point

from numba import jit

# from multiprocessing import Pool

# ADD
# ADD  Import lake polygons:
# ADD       -filter out confluences in lakes
# ADD       -find all stream network intersections with lake polygons
# ADD       -set all as confluences except max value (outlet)
# ADD       - or rather any whose connecting cells are increasing in acc
#               -this will get messy, guaranteed

DEM_source = 'USGS_3DEP'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')
DEM_DIR = os.path.join(BASE_DIR, 'processed_data/processed_dem/')

output_dir = os.path.join(DATA_DIR, f'pour_points/')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# this should correspond with the threshold accumulation
# set in "derive_flow_accumulation.py"
min_basin_area = 2 # km^2
# min number of cells comprising a basin
# basin_threshold = int(min_basin_area * 1E6 / (90 * 90)) 

region_files = os.listdir(DEM_DIR)

region_codes = sorted(list(set([e.split('_')[0] for e in region_files])))
# region_codes = ['08P']


def retrieve_raster(region, raster_type, crs=3005):
    filename = f'{region}_{DEM_source}_{crs}_{raster_type}.tif'
    raster_path = os.path.join(DEM_DIR, f'{filename}')
    raster = rxr.open_rasterio(raster_path, mask_and_scale=True)
    crs = raster.rio.crs
    affine = raster.rio.transform(recalc=False)
    clipped = clip_raster_with_polygon(region, raster, crs, raster.rio.resolution())
    return clipped, crs, affine


def clip_raster_with_polygon(region, raster, crs, resolution):
    region_polygon = get_region_polygon(region)
    region_polygon = region_polygon.to_crs(crs)
    assert region_polygon.crs == crs
    # add a buffer to the region mask to simplify windowing
    # operations for raster processing (avoid cutting edges)
    buffer_len = round(abs(resolution[0]), 0)
    clipped = raster.rio.clip(region_polygon.buffer(buffer_len).geometry)
    return clipped


def get_region_polygon(region):
    polygon_path = os.path.join(BASE_DIR, 'input_data/region_polygons/')
    poly_files = os.listdir(polygon_path)
    file = [e for e in poly_files if e.startswith(region)]
    if len(file) == 0:
        raise Exception; 'Region shape file not found.'
    fpath = os.path.join(polygon_path, file[0])
    gdf = gpd.read_file(fpath)
    return gdf
  

def get_region_area(region):
    gdf = get_region_polygon(region)
    gdf = gdf.to_crs(3005)
    return gdf['geometry'].area.values[0] / 1E6


def mask_flow_direction(S_w, F_w):  

    inflow_dirs = np.array([
        [4, 8, 16],
        [2, 0, 32],
        [1, 128, 64]
    ])
    # mask the flow direction by element-wise multiplication
    # using the boolean stream matrix
    F_sm = np.multiply(F_w, S_w)

    F_m = np.where(F_sm == inflow_dirs, True, False)

    return F_m
   
   
def check_CONF(i, j, ppts, S_w, F_m):
    # if more than one outer cell flow direction 
    # (plus centre = True) points to centre 
    # then it is a confluence.
    F_mc = F_m.copy()
    F_mc[1, 1] = 1

    # get inflow cell indices
    inflow_cells = np.argwhere(F_mc)

    fp_idx = f'{i},{j}'

    if len(inflow_cells) <= 2:
        # don't overwrite an already checked cell
        if 'CONF' not in ppts[fp_idx]:
            ppts[fp_idx]['CONF'] = False        
    else:
        ppts[fp_idx]['CONF'] = True        

        for (ci, cj) in inflow_cells:
            ix = ci + i - 1
            jx = cj + j - 1

            pt_idx = f'{ix},{jx}'

            # cells flowing into the focus 
            # cell are confluence cells
            if pt_idx not in ppts:
                ppts[pt_idx] = {'CONF': True}
            else:
                ppts[pt_idx]['CONF'] = True

    return ppts


def create_pour_point_gdf(stream, ppt_df, crs):
    """Break apart the list of stream pixels to avoid memory 
    allocation issue when indexing large rasters.

    Args:
        stream (array: binary matrix of stream (1) and hillslope (0) cells
        ppd_df (dataframe): dataframe of pour point indices
        crs (str or int): coordiante reference system (i.e. EPSG code, i.e. 4326) 

    Returns:
        geodataframe: geodataframe of pour points, geometry is Point
    """
    
    n_chunks = int(10 * np.log(len(ppt_df)))

    # split the dataframe into chunks 
    # because calling coordinates of the raster
    # from a large array seems to be memory intensive.
    conf_chunks = np.array_split(ppt_df, n_chunks)
    processed_chunks = []
    for chunk in conf_chunks:
        ppts = stream[0, chunk['ix'].values, chunk['jx'].values]
        coords = tuple(map(tuple, zip(ppts.coords['x'].values, ppts.coords['y'].values)))
        chunk['geometry'] = [Point(p) for p in coords]
        processed_chunks.append(chunk)

    return gpd.GeoDataFrame(pd.concat(processed_chunks), crs=crs)
    

def process_ppts(S, F, A):
    
    # is_confluence = find_stream_confluence
    stream_px = np.argwhere(S == 1)
    
    ppts = {}
    
    for (i, j) in stream_px:
        
        c_idx = f'{i},{j}'
        if c_idx not in ppts:
            ppts[c_idx] = {}
        ppt = ppts[c_idx]

        # Add river outlets, as these are by definition
        # confluences and especially prevalent in coastal regions
        focus_cell_acc = A[i, j]
        focus_cell_dir = F[i, j]

        ppt['acc'] = focus_cell_acc

        if focus_cell_dir == 0:
            # the focus cell is already defined as a stream cell
            # and if its direction value is nan or 0, 
            # there is no flow direction and it's an outlet cell.
            ppt['OUTLET'] = True
            # by definition an outlet cell is also a confluence
            ppt['CONF'] = True
        else:
            ppt['OUTLET'] = False

        # create a 3x3 boolean matrix of stream cells centred
        # on the focus cell
        S_w = S[max(0, i-1):i+2, max(0, j-1):j+2].copy()
        F_w = F[max(0, i-1):i+2, max(0, j-1):j+2].copy()
        
        # create a boolean matrix for cells that flow into the focal cell
        F_m = mask_flow_direction(S_w, F_w)
        
        # check if cell is a stream confluence
        # set the target cell to false by default
        ppts = check_CONF(i, j, ppts, S_w, F_m)
        
    ppt_df = pd.DataFrame.from_dict(ppts, orient='index')
    ppt_df.index.name = 'cell_idx'
    ppt_df.reset_index(inplace=True) 
    
    # split the cell indices into columns and convert str-->int
    ppt_df['ix'] = [int(e.split(',')[0]) for e in ppt_df['cell_idx']]
    ppt_df['jx'] = [int(e.split(',')[1]) for e in ppt_df['cell_idx']]
    
    # filter for stream points that are an outlet or a confluence
    ppt_df = ppt_df[(ppt_df['OUTLET'] == True) | (ppt_df['CONF'] == True)]
    return ppt_df, len(stream_px)


cell_tracking_info = {}

regions_to_process = sorted(list(set(region_codes)))
# regions_to_process = ['HGW']
processed_regions = list(set([e for e in os.listdir(output_dir) if len(os.listdir(os.path.join(output_dir, e))) > 0]))
n_points_total = 0
for region in [r for r in regions_to_process if r not in processed_regions]:
    print('')
    print(f'Processing candidate pour points for {region}.')

    region_area_km2 = get_region_area(region)

    rt0 = time.time()
    
    stream, crs, affine = retrieve_raster(region, 'stream')

    fdir, crs, affine = retrieve_raster(region, 'fdir')

    acc, crs, affine = retrieve_raster(region, 'accum')

    rt1 = time.time()
    print(f'   ...time to load resources: {rt1-rt0:.1f}s.')

    t0 = time.time()

    # get raster data in matrix form
    S = stream.data[0]
    F = fdir.data[0]
    A = acc.data[0]

    ppt_df, total_pts = process_ppts(S, F, A)
    
    na_pts = ppt_df[ppt_df[ppt_df.columns].isnull().any(axis=1)].copy()
    
    t_end = time.time()
    n_pts_conf = len(ppt_df[ppt_df['CONF']])
    n_pts_outlet = len(ppt_df[ppt_df['OUTLET']])

    print('')
    print(f'Completed {region} in  {t_end-t0:.1f}s')
    print(f'Of {total_pts} total stream cells:')
    print(f'    {n_pts_conf} ({100*n_pts_conf/total_pts:.0f}%) are stream confluences,')
    print(f'    {n_pts_outlet} ({100*n_pts_outlet/total_pts:.0f}%) are stream outlets.')
    # print(f'    {n_pts_grab} ({100*n_pts_grab/n_pts_tot:.0f}%) are accumulation change of >= {min_basin_area} km^2,')
    # print(f'    {n_pts_grap} ({100*n_pts_grap/n_pts_tot:.0f}%) are accumulation changes of >= 50 % of the target cell,')
    if len(na_pts) > 0:
        print('')
        print('**Warning**')
        print('Null rows are being cleared from the dataframe:')
        print(na_pts.head())
        print('')
    else:
        print(f'    No null rows encountered ({len(na_pts)}).')
        print('')

    output_folder = os.path.join(DATA_DIR, f'pour_points/{region}/')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    ppt_gdf = create_pour_point_gdf(stream, ppt_df, crs)
    
    del stream
    del acc
    del fdir

    # save output to geojson
    output_fname = f'{region}_pour_pts.geojson'
    output_path = os.path.join(output_folder, output_fname)
    if not ppt_gdf.empty:
        ppt_gdf.to_file(output_path, driver='GeoJSON')       
    n_points_total += len(ppt_gdf)
    t1 = time.time()
    print(f'   ...{len(ppt_gdf)} points created.  Time to write ppt file: {t1-t0:.1f}\n')
    del ppt_gdf
    
print(f'Found {n_points_total} pour points in total.')