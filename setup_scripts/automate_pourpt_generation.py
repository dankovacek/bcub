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
DEM_DIR = os.path.join(BASE_DIR, 'input_data/processed_dem/')

output_dir = os.path.join(DATA_DIR, f'pour_points/')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# this should correspond with the threshold accumulation
# set in "derive_flow_accumulation.py"
min_basin_area = 2 # km^2
# min number of cells comprising a basin
basin_threshold = int(min_basin_area * 1E6 / (90 * 90)) 

region_files = os.listdir(DEM_DIR)

region_codes = sorted(list(set([e.split('_')[0] for e in region_files])))
region_codes = ['08P']


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


def mask_lakes_by_region(region, ldf):
    ta = time.time()
    print(f'   ...importing {region} region polygon')
    region_polygon = get_region_polygon(region)
    # reproject to match nhn crs
    # region_polygon = region_polygon.to_crs(4617)
    tb = time.time()
    print(f'   ...region polygon opened in {tb-ta:.2f}s')
    ta = time.time()
    region_polygon = region_polygon.to_crs(4617)    
    tc = time.time()
    region_lakes = gpd.sjoin(ldf, region_polygon, how='inner', predicate='intersects')
    tb = time.time()
    print(f'   ...lakes sjoin processed in {tb-tc:.2f}s')
    return region_lakes
    

def filter_ppts_by_lakes_geom(region, ppt_gdf):
    """Filter out "confluence" pour points that lie in permanent water bodies.
    
    Permanency code:
    -1 unknown
    0 no value available
    1 permanent
    2 intermittent

    Args:
        region (_type_): _description_
        lakes_df (_type_): _description_
        
    water_definition Label Code Definition
    ----------------------------- ---- ----------
    None 0 No Waterbody Type value available.
    Canal 1 An artificial watercourse serving as a navigable waterway or to
    channel water.
    Conduit 2 An artificial system, such as an Aqueduct, Penstock, Flume, or
    Sluice, designed to carry water for purposes other than
    drainage.
    Ditch 3 Small, open manmade channel constructed through earth or
    rock for the purpose of conveying water.
    *Lake 4 An inland body of water of considerable area.
    *Reservoir 5 A wholly or partially manmade feature for storing and/or
    regulating and controlling water.
    Watercourse 6 A channel on or below the earth's surface through which water
    may flow.
    Tidal River 7 A river in which flow and water surface elevation are affected by
    the tides.
    *Liquid Waste 8 Liquid waste from an industrial complex.

    Returns:
        _type_: _description_
    """
    lakes_fpath = os.path.join(DATA_DIR, 'study_region_waterbodies.geojson')
    region_lakes_fpath = os.path.join(DATA_DIR, f'region_waterbodies/{region}_waterbodies.geojson')
    if not os.path.exists(region_lakes_fpath):
        t1 = time.time()
        ldf = gpd.read_file(lakes_fpath)
        t2 = time.time()
        print(f'   Lakes layer opened in {t2-t1:.0f}s')
        print(f'    Creating lakes geometry file for {region}')
        region_lakes = mask_lakes_by_region(region, ldf)
        region_lakes = region_lakes.to_crs(3005)
        region_lakes.to_file(region_lakes_fpath)
        n_lakes = len(region_lakes)
        print(f'    File saved.  There are {n_lakes} water body objects in {region}.')
    else:
        region_lakes = gpd.read_file(region_lakes_fpath)

    assert ppt_gdf.crs == lakes_df.crs
    lakes_df['area_km2'] = lakes_df.geometry.area / 1E6
    lakes_df = lakes_df[['acquisition_technique', 'area_km2', 'water_definition', 'planimetric_accuracy', 'permanency', 'geometry']]
    lakes_df.geometry = lakes_df.geometry.explode(index_parts=False)
    
    # lake_ppts = region_ppts[region_ppts.within(lakes_df.geometry)].copy()
    lake_ppts = gpd.sjoin(ppt_gdf, lakes_df, how='left', predicate='within')
    
    # filter for water_definition code 
    # filter out all lakes, reservoirs, liquid waste, tidal river (>3) but include "watercourses" (6)
    ppt_filter = ((lake_ppts['water_definition'] < 3) | (lake_ppts['water_definition'] == 6)) | (lake_ppts['index_right'].isna())
    filtered_ppts = lake_ppts[ppt_filter] 
    
    # lake_ppts = p.disjoint((region_ppts.geometry, lakes_df.geometry)
    print(f'    {len(filtered_ppts)}/{len(ppt_gdf)} ppts are not in lakes.')
    return filtered_ppts


def get_region_area(region):
    gdf = get_region_polygon(region)
    gdf = gdf.to_crs(3005)
    return gdf['geometry'].area.values[0] / 1E6


def random_stream_point_selection(stream_px, A):
    """
    Here we simply return all of the stream pixels.
    The random selection will occur later, but it is 
    faster to read the file once and do iterative 
    random selections instead of creating a temporary 
    gdf in the random selection step for every simulation.
    """
    ppts = []
    c = 0
    for i, j in stream_px:
        cell_acc = A[i, j]
        ppts.append((i, j, cell_acc, c))
        c += 1

    pct_tracked = len(ppts) / len(stream_px) * 100
    print(f'Tracked {c} randomly selected points.')
    print(f'{len(ppts):.3e}/{len(stream_px):.3e} stream pixels are points of interest ({pct_tracked:.0f}%).')
        
    return ppts, pct_tracked, c


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
   
   
def check_acc_gradient(i, j, A, F, F_m, threshold, method):
    """_summary_

    Args:
        region (_type_): _description_
        idxs (_type_): _description_
        threshold (_type_): _description_

    Returns:
        _type_: _description_
    """

    # get the accumulation value of the target cell
    focus_cell_acc = A[i, j]
            
    # retrieve flow accumulation from a 3x3 window 
    # around the target pixel
    A_w = A[i-1:i+2, j-1:j+2].copy()
    
    # calculate the acc gradient w.r.t. the focal cell
    # and filter out non-stream cells
    A_g = np.abs(focus_cell_acc - A_w)

    # flip the bit on the cell the focal cell points to
    # so its counted in the gradient calculation. 
    
    # flow direction matrix
    d8_matrix = np.array([
        [64, 128, 1],
        [32, 0, 2],
        [16, 8, 4]
    ])
    focal_cell_direction = np.argwhere(F[i, j]==d8_matrix)
    dir_idx = focal_cell_direction[0]

    F_mc = F_m.copy()
    
    F_mc[dir_idx[0],dir_idx[1]] = True

    # mask the flow direction by element-wise multiplication
    # using the boolean flow direction matrix.
    # this yields all stream cells flowing into focal cell
    A_gf = np.multiply(A_g, F_mc)

    # GRAB - AB stands for absolute -- find all cells where
    # the accumulation difference is greater than a constant
    # number of cells (minimum area change of interest).
    threshold_crit = (A_gf > threshold).any()

    # proportion criteria not needed for the GRAB method
    proportion_crit = True
    if method == 'GRAP':
        # AP stands for area proportional -- find cells where
        # the difference is greater than 10% of the accumulation
        # of the focal cell
        proportion_threshold = 0.5 * focus_cell_acc
        proportion_crit = (A_gf > proportion_threshold).any()

    return threshold_crit & proportion_crit  


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


def create_pour_point_gdf(region, stream, ppt_df, crs, n_chunks=2):
    """Break apart the list of stream pixels to avoid memory 
    allocation issue when indexing large rasters.

    Args:
        stream (_type_): _description_
        confluences (_type_): _description_
        n_chunks (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
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

    gdf = gpd.GeoDataFrame(pd.concat(processed_chunks), crs=crs)
    print(f'    {len(gdf)} pour points created.') 

    output_path = os.path.join(DATA_DIR, f'pour_points/{region}/{region}_all_ppts.geojson')

    t0 = time.time()
    # output to a single gdf geojson
    gdf.to_file(output_path, driver='GeoJSON')
    t1 = time.time()
    print(f'   ...time to write ppt file: {t1-t0:.1f}\n')
    return gdf


separate_output_files = False

cell_tracking_info = {}

regions_to_process = sorted(list(set(region_codes)))
processed_regions = list(set([e.split('_')[0] for e in os.listdir(output_dir)]))

for region in [r for r in regions_to_process if r not in processed_regions]:
    print('')
    print(f'Processing candidate pour points for {region}.')

    region_area_km2 = get_region_area(region)

    # alternatively, set a sample size and use random selection 
    # ppt_sample_size = int(region_area_km2 / 100)
    # # ppt_sample_size = 10
    # # print(f'Generating {ppt_sample_size} pour points for a region of {region_area_km2:.1f} km^2 to yield 1 station per 100 km^2.')

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

    # is_confluence = find_stream_confluence
    stream_px = np.argwhere(S == 1)

    ppts = {}
    nn = 0
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
        
        # ppt['GRAB'] = check_acc_gradient(i, j, A, F, F_m, basin_threshold, 'GRAB')

        # ppt['GRAP'] = check_acc_gradient(i, j, A, F, F_m, basin_threshold, 'GRAP')
        
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
    
    na_pts = ppt_df[ppt_df[ppt_df.columns].isnull().any(axis=1)].copy()
    
    t_end = time.time()

    n_pts_tot = len(stream_px)
    n_pts_conf = len(ppt_df[ppt_df['CONF']])
    # n_pts_grab = len(ppt_df[ppt_df['GRAB']])
    # n_pts_grap = len(ppt_df[ppt_df['GRAP']])
    n_pts_outlet = len(ppt_df[ppt_df['OUTLET']])

    print('')
    print(f'Completed {region} in  {t_end-t0:.1f}s')
    print(f'Of {n_pts_tot} total stream cells:')
    print(f'    {n_pts_conf} ({100*n_pts_conf/n_pts_tot:.0f}%) are stream confluences,')
    # print(f'    {n_pts_grab} ({100*n_pts_grab/n_pts_tot:.0f}%) are accumulation change of >= {min_basin_area} km^2,')
    # print(f'    {n_pts_grap} ({100*n_pts_grap/n_pts_tot:.0f}%) are accumulation changes of >= 50 % of the target cell,')
    print(f'    {n_pts_outlet} ({100*n_pts_outlet/n_pts_tot:.0f}%) are stream outlets.')
    if len(na_pts) > 0:
        print('')
        print('**Warning**')
        print('Null rows are being cleared from the dataframe:')
        print(na_pts.head())
        print('')
    else:
        print(f'    No null rows encountered ({len(na_pts)}).')
        print('')

    print(f'    {len(ppt_df)} ({100*len(ppt_df)/n_pts_tot:.0f}%) are unique points for basin delineation.')
    print('')

    output_folder = os.path.join(DATA_DIR, f'pour_points/{region}/')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    ppt_gdf = create_pour_point_gdf(region, stream, ppt_df, crs)

    # use the script below to save points of interest
    # to separate files by method

    output_fname = f'{region}_pour_pts.geojson'
    output_path = os.path.join(output_folder, output_fname)
    if not ppt_gdf.empty:
        ppt_gdf.to_file(output_path, driver='GeoJSON')
