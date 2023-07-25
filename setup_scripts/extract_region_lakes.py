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
import xarray as xr
import rasterio as rio

import multiprocessing as mp

from shapely.geometry import Point, LineString

from numba import jit
import fiona


DEM_source = 'USGS_3DEP'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')
DEM_DIR = os.path.join(BASE_DIR, 'input_data/processed_dem')

output_path = os.path.join(DATA_DIR, f'pour_points/')
if not os.path.exists(output_path):
    os.mkdir(output_path)

# this should correspond with the threshold accumulation
# set in "derive_flow_accumulation.py"
min_basin_area = 2 # km^2
# min number of cells comprising a basin

lakes_fpath = os.path.join(DATA_DIR, 'study_region_waterbodies.geojson')

# region_files = os.listdir(processed_dem_dir)

# region_codes = sorted(list(set([e.split('_')[0] for e in region_files])))


def retrieve_raster(region, region_polygon, raster_type):
    filename = f'{region}_{DEM_source}_3005_{raster_type}.tif'
    raster_path = os.path.join(processed_dem_dir, f'{filename}')
    raster = rxr.open_rasterio(raster_path, mask_and_scale=True)
    crs = raster.rio.crs
    affine = raster.rio.transform(recalc=False)
    clipped = clip_raster_with_polygon(region, raster, crs, raster.rio.resolution())
    return clipped, crs, affine


def get_region_polygon(region):
    polygon_path = os.path.join(DATA_DIR, 'merged_basin_groups/region_polygons/')
    poly_files = os.listdir(polygon_path)
    file = [e for e in poly_files if e.startswith(region)]
    if len(file) == 0:
        raise Exception; f'{region} Region shape file not found.'
    fpath = os.path.join(polygon_path, file[0])
    gdf = gpd.read_file(fpath)
    return gdf


def clip_raster_with_polygon(region, raster, crs, resolution):
    region_polygon = get_region_polygon(region)
    region_polygon = region_polygon.to_crs(crs)
    assert region_polygon.crs == crs
    # add a buffer to the region mask to simplify windowing
    # operations for raster processing (avoid cutting edges)
    buffer_len = round(abs(resolution[0]), 0)
    clipped = raster.rio.clip(region_polygon.buffer(buffer_len).geometry)
    return clipped


def mask_lakes_by_region(region, region_polygon, ldf):
    ta = time.time()
    print(f'   ...importing {region} region polygon')
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


def filter_ppts_by_lakes_geom(region, region_lakes, region_ppts):
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

    # streams = retrieve_raster(region, 'stream')
    # region_ppts = region_ppts.to_crs(lakes_df.crs)
    assert region_ppts.crs == region_lakes.crs
    region_lakes['area'] = region_lakes.geometry.area
    # filter lakes smaller than 0.01 km^2
    ldf = region_lakes[region_lakes['area'] > 10000].copy()
    ldf = ldf[['acquisition_technique', 'area', 'water_definition', 'planimetric_accuracy', 'permanency', 'geometry']]
    ldf = ldf.explode(index_parts=False)
    
    # filter by water_definition code 
    # get lakes and reservoirs (5 & 6)
    lake_filter = (ldf['water_definition'] > 3) & (ldf['water_definition'] < 6)
    filtered_lakes = ldf[lake_filter].copy()
    
    # merge contiguous polygons 
    filtered_lakes = gpd.GeoDataFrame(geometry=[filtered_lakes.geometry.unary_union], crs='EPSG:3005')
    filtered_lakes = filtered_lakes.explode().reset_index(drop=True)
    # lake_ppts = region_ppts[region_ppts.within(lakes_df.geometry)].copy()
    ppt_lakes = gpd.sjoin(filtered_lakes, region_ppts, how='inner', predicate='intersects')   
    ppt_lakes['lake_idx'] = ppt_lakes.index
    
    n_lakes = len(set(ppt_lakes['lake_idx']))
    print(f'   {n_lakes}/{len(region_lakes)} lakes contain ppts.')
    
    return ppt_lakes
    
    
def redistribute_vertices(geom, distance):
    if geom.geom_type in ['LineString', 'LinearRing']:
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        ls = gpd.GeoDataFrame(geometry=[geom], crs='EPSG:3005')
        geoms = ls.explode().reset_index(drop=True).geometry.values
        parts = [redistribute_vertices(part, distance)
                 for part in geoms]
        print(parts)
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))


def get_pt_indices(region, raster_type, df, acc):
    filename = f'{region}_{DEM_source}_3005_{raster_type}.tif'
    raster_path = os.path.join(DEM_DIR, f'{filename}')
    pt_info = []
    with rio.open(raster_path) as src:
        for i, p in df.iterrows():
            latlon = p.geometry.coords.xy
            x, y = latlon[0].tolist(), latlon[1].tolist()
            idx = acc.sel(x=x[0], y=y[0], method='nearest', tolerance=abs(acc.rio.resolution()[0]))
            accum = idx.data[0]
            rows, cols = rio.transform.rowcol(src.transform, x, y)
            pt_info.append([f'{rows[0]},{cols[0]}', accum, rows[0], cols[0], p.geometry, False])
    
    cols = ['cell_idx', 'acc', 'ix', 'jx', 'geometry', 'in_lake']
    pt_df = pd.DataFrame(pt_info, columns=cols)
    pt_gdf = gpd.GeoDataFrame(pt_df, crs='EPSG:3005')
    return pt_gdf


def process_shoreline_pts(inputs):

    x, y, region_ppts, resolution = inputs    
    pt = Point(x, y)
    
    # lake_check = not_in_this_lake & not_in_any_lake
    # don't let adjacent points both be pour points
    # but avoid measuring distance to points within lakes
    rpt_dists = region_ppts[~region_ppts['in_lake']].distance(pt).min()
    dist_check = rpt_dists <= 4.0 * resolution
    
    # accum_check = accum < 0.95 * max_acc
    accum_check = True
    if accum_check & (~dist_check):
        # check if the potential point is in any of the lakes
        # not_in_any_lake = sum([lg.contains(pt) for lg in lakes_df.geometry]) == 0
        if not lakes_df.contains(pt).any():
            return pt
    return None


def find_lake_inflows(region, region_polygon, lakes_df, region_ppts):
    # add intersections with stream cells at edge of lakes
    acc, crs, affine = retrieve_raster(region, region_polygon, 'accum')
    # stream, crs, affine = retrieve_raster(region, region_polygon, 'stream')
    resolution = abs(acc.rio.resolution()[0])
    # print('resolution, ', resolution)
    
    min_acc_cells = 1E6 / (resolution**2)
    # print(f'    acc threshold = {min_acc_cells}')
        
    n_lakes = len(lakes_df)
    # print(lakes_df)
    print(f' processing {n_lakes} lakes')

    all_new_pts = []
    region_ppts['in_lake'] = False
    n_grps = len(list(set(lakes_df['lake_idx'])))
    if n_grps < 10:
        p_interval = 5
    elif n_grps < 100:
        p_interval = 10
    else:
        p_interval = 25
        
    for i, lake_grp in lakes_df.groupby('lake_idx'):
        region_ppts.loc[lake_grp['index_right'].values, 'in_lake'] = True
        
    filtered_region_ppts = region_ppts[region_ppts['in_lake']].copy()
        
    n = 0
    tot_pts = 0
    tb = time.time()
    for _, lake_grp in lakes_df.groupby('lake_idx'):
        n += 1
        if n % p_interval == 0:
            print(f'   Processing lake group {n}/{n_grps}')        
        
        # get the unique lake polygon (geoms in lake_grp are duplicates)
        lake_geom = list(set(lake_grp.geometry))[0].simplify(resolution)
        
        shoreline = lake_geom.buffer(resolution*2)
        # create_geom = True
        # if lake_geom.area > 10000000:
        #     foo = gpd.GeoDataFrame(geometry=[shoreline], crs='EPSG:3005')
        #     if create_geom:
        #         foo.to_file('shore_foo.geojson')
        #         create_geom = False
        
        # resample the shoreline vector to prevent missing confluence points
        resampled_shoreline = redistribute_vertices(shoreline.exterior, 5).coords.xy
        xs = resampled_shoreline[0].tolist()
        ys = resampled_shoreline[1].tolist()
        
        px_pts = acc.sel(x=xs, y=ys, method='nearest', tolerance=resolution/2)
        latlon = list(set(zip(px_pts.x.values, px_pts.y.values)))
        tot_pts += len(latlon)

        inputs = []
        for x, y in latlon:
            acc_val = px_pts.sel(x=x, y=y).drop_duplicates(dim=...).squeeze()
            accum = acc_val.item()
            if (accum > min_acc_cells):
                inputs.append((x, y, filtered_region_ppts, resolution))
                            
        # filter for empty pts
        # td = time.time()
        # print(f'    time to filter {len(latlon)} input pts: {td-tc:.2f}s')
                
        p = mp.Pool()
        pts = p.map(process_shoreline_pts, inputs)
        tc = time.time()
        all_new_pts += [e for e in pts if e is not None]

    tc = time.time()
    print(f'     time to iterate through {tot_pts} points to yield {len(all_new_pts)} pts: {tc-tb:.2f}')
    
    rpts = region_ppts[['geometry']].copy()
    all_pts_filtered = []
    for pt in all_new_pts:
        dists = rpts.distance(pt)
        if (dists > 4 * resolution).all():
            ptg = gpd.GeoDataFrame(geometry=[pt], crs='EPSG:3005')
            # append the new point to the reference point dataframe to
            # update the set of points checked against.
            rpts = gpd.GeoDataFrame(pd.concat([rpts, ptg]), crs='EPSG:3005')
            all_pts_filtered.append(pt)
                    
    new_pts = gpd.GeoDataFrame(geometry=all_pts_filtered, crs='EPSG:3005')    
    new_pt_gdf = get_pt_indices(region, 'accum', new_pts, acc)
    return new_pt_gdf

        
regions = list(set([e.split('_')[0] for e in os.listdir(os.path.join(DATA_DIR, 'merged_basin_groups/region_polygons'))]))

regions = ['08P']

for region in regions:
    output_fname = f'{region}_pour_pts_filtered.geojson'
    output_fpath = os.path.join(DATA_DIR, output_fname)
    if os.path.exists(output_fpath):
        print(f'   {region} already processed: {output_fname}')
        continue
    t0 = time.time()
    print(f'Processing {region}.')
    region_polygon = get_region_polygon(region)
    region_lakes_fpath = os.path.join(DATA_DIR, f'region_waterbodies/{region}_waterbodies.geojson')
    if not os.path.exists(region_lakes_fpath):
        print('    Creating region water bodies layer.')
        t1 = time.time()
        ldf = gpd.read_file(lakes_fpath, mask=region_polygon)
        t2 = time.time()
        print(f'    Lakes layer opened in {t2-t1:.0f}s')
        print(f'    Creating lakes geometry file for {region}')
        region_lakes = mask_lakes_by_region(region, region_polygon, ldf)
        region_lakes = region_lakes.to_crs(3005)
        region_lakes.to_file(region_lakes_fpath)
        n_lakes = len(region_lakes)
        print(f'    File saved.  There are {n_lakes} water body objects in {region}.')
    else:
        region_lakes = gpd.read_file(region_lakes_fpath)
        
    ppts_fpath = os.path.join(DATA_DIR, f'pour_points/{region}/{region}_pour_pts_CONF_RC.geojson') 
    region_ppts = gpd.read_file(ppts_fpath)
    
    lakes_df = filter_ppts_by_lakes_geom(region, region_lakes, region_ppts)            

    new_ppts = find_lake_inflows(region, region_polygon, lakes_df, region_ppts)    
            
    n_pts0 = len(region_ppts)
    filtered_ppts = region_ppts[~region_ppts['in_lake']].copy()
    n_pts1 = len(filtered_ppts)
    
    concat_df = pd.concat([filtered_ppts, new_ppts], join='outer', axis=0)
    final_pts = gpd.GeoDataFrame(concat_df, crs='EPSG:3005')
    n_final = len(final_pts)
    
    print(f'    {n_pts0-n_pts1} points eliminated (fall within lakes)')
    print(f'    {len(new_ppts)} points added for lake inflows.')
    print(f'    {n_final} points after filter and merge. ({n_pts0-n_final} difference)')
    
    
    final_pts.to_file(output_fpath)
    te = time.time()
    utime = n_final / (te-t0)
    print(f'      {region} processed in {te-t0:.0f}s ({utime:.2f}pts/s)')  
    print('')

    
