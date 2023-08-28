# generate basins
import os

import time

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
# import rasterio as rio

import multiprocessing as mp

from shapely.geometry import Point, LineString, Polygon

DEM_source = 'USGS_3DEP'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')
DEM_DIR = os.path.join(BASE_DIR, 'processed_data/processed_dem')

output_path = os.path.join(DATA_DIR, f'pour_points/')
if not os.path.exists(output_path):
    os.mkdir(output_path)

# this should correspond with the threshold accumulation
# set in "derive_flow_accumulation.py"
min_basin_area = 2 # km^2

lakes_dir = os.path.join(BASE_DIR, 'input_data/BasinATLAS')
lakes_fpath = os.path.join(lakes_dir, 'HydroLAKES_clipped.gpkg')
# the HydroLAKES CRS is EPSG:4326
lakes_crs = 4326

if not os.path.exists(lakes_fpath):
    err_msg = f'HydroLAKES file not found at {lakes_fpath}.  Download from https://www.hydrosheds.org/products/hydrolakes.  See README for details.'
    raise Exception(err_msg)


def retrieve_raster(region, raster_type):
    filename = f'{region}_{DEM_source}_3005_{raster_type}.tif'
    raster_path = os.path.join(DEM_DIR, f'{filename}')
    raster = rxr.open_rasterio(raster_path, mask_and_scale=True)
    crs = raster.rio.crs
    affine = raster.rio.transform(recalc=False)
    clipped = clip_raster_with_polygon(region, raster, crs, raster.rio.resolution())
    return clipped, crs, affine


def get_region_polygon(region):
    polygon_path = os.path.join(BASE_DIR, 'input_data/region_polygons/')
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


def trim_appendages(row):
    g = gpd.GeoDataFrame(geometry=[row['geometry']], crs='EPSG:3005')
    geom = g.explode()
    geom['area'] = geom.geometry.area
    if len(geom) > 1:
        # return only the largest geometry by area
        return geom.loc[geom['area'].idxmax(), 'geometry']
    return row['geometry']
               

def filter_lakes(lakes_df, ppts, resolution):
    """
    Permanency code:
    -1 unknown
    0 no value available
    1 permanent
    2 intermittent

    Args:
        wb_df (geodataframe): Water body geometries.
        ppts (geodataframe): Pour points.
        
    water_definition Label Definition
    ----------------------------- ---- ----------
    None            0       No Waterbody Type value available.
    Canal           1       An artificial watercourse serving as a navigable waterway or to
                            channel water.
    Conduit         2       An artificial system, such as an Aqueduct, Penstock, Flume, or
                            Sluice, designed to carry water for purposes other than
                            drainage.
    Ditch           3       Small, open manmade channel constructed through earth or
                            rock for the purpose of conveying water.
    *Lake           4       An inland body of water of considerable area.
    *Reservoir      5       A wholly or partially manmade feature for storing and/or
                            regulating and controlling water.
    Watercourse     6       A channel on or below the earth's surface through which water
                            may flow.
    Tidal River     7       A river in which flow and water surface elevation are affected by
                            the tides.
    *Liquid Waste   8       Liquid waste from an industrial complex.
    """    
    lakes_df = lakes_df.to_crs(ppts.crs)
    
    # reproject to projected CRS before calculating area
    lakes_df['area'] = lakes_df.geometry.area
    lakes_df['lake_id'] = lakes_df.index.values
        
    # filter lakes smaller than 0.1 km^2
    min_area = 100000
    lakes_df = lakes_df[lakes_df['area'] > min_area]
    lakes_df = lakes_df.dissolve().explode(index_parts=False).reset_index(drop=True)
    lake_cols = lakes_df.columns
    
    # filter out Point type geometries
    lakes_df = lakes_df[~lakes_df.geometry.type.isin(['Point', 'LineString'])]
    # find and fill holes in polygons    
    lakes_df.geometry = [Polygon(p.exterior) for p in lakes_df.geometry]
        
    # find the set of lakes that contain confluence points
    lakes_with_pts = gpd.sjoin(lakes_df, ppts, how='left', predicate='intersects')
    
    # the rows with index_right == nan are lake polygons containing no points
    lakes_with_pts = lakes_with_pts[~lakes_with_pts['index_right'].isna()]
    lakes_with_pts = lakes_with_pts[[c for c in lakes_with_pts.columns if 'index_' not in c]]
    # drop all duplicate indices
    lakes_with_pts = lakes_with_pts[~lakes_with_pts.index.duplicated(keep='first')]
    lakes_with_pts.area = lakes_with_pts.geometry.area
        
    # use negative and positive buffers to remove small "appendages"
    # that tend to add many superfluous inflow points
    distance = 100  # metres
    lakes_with_pts.geometry = lakes_with_pts.buffer(-distance).buffer(distance * 1.5).simplify(resolution/np.sqrt(2))
    lakes_with_pts['geometry'] = lakes_with_pts.apply(lambda row: trim_appendages(row), axis=1)
    return lakes_with_pts
    
    
def interpolate_line(inputs):
    geom, n, num_vertices = inputs
    d = n / num_vertices
    return (n, geom.interpolate(d, normalized=True))
 
    
def redistribute_vertices(geom, distance):
    """Evenly resample along a linestring
    See this SO post:
    https://gis.stackexchange.com/a/367965/199640
    
    Args:
        geom (polygon): lake boundary geometry
        distance (numeric): distance between points in the modified linestring

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if geom.geom_type in ['LineString', 'LinearRing']:
        num_vertices = int(round(geom.length / distance))
        
        if num_vertices == 0:
            num_vertices = 1
        # print(f'total distance = {geom.length:.0f} m, n_vertices = {num_vertices}')
        inputs = [(geom, float(n), num_vertices) for n in range(num_vertices + 1)]
        pool = mp.Pool()
        results = pool.map(interpolate_line, inputs)
        pool.close()
        
        df = pd.DataFrame(results, columns=['n', 'geometry'])
        df = df.sort_values(by='n').reset_index(drop=True)
        return LineString(df['geometry'].values)
    
    elif geom.geom_type == 'MultiLineString':
        ls = gpd.GeoDataFrame(geometry=[geom], crs='EPSG:3005')
        geoms = ls.explode().reset_index(drop=True).geometry.values
        parts = [redistribute_vertices(part, distance)
                 for part in geoms]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))


def find_link_ids(target):
    x, y = target
    stream_loc = stream.sel(x=x, y=y).squeeze()
    link_id = stream_loc.item()
    if ~np.isnan(link_id):
        i, j = np.argwhere(stream.x.values == x)[0], np.argwhere(stream.y.values == y)[0]
        return [i[0], j[0], Point(x, y), link_id]
    else:
        nbr = stream.rio.clip_box(x-resolution, y-resolution, x+resolution,y+resolution)
        
        if np.isnan(nbr.data).all():
            return None
        
        raster_nonzero = nbr.where(nbr > 0, drop=True)
        
        # Check surrounding cells for nonzero link_ids
        xs, ys = raster_nonzero.x.values, raster_nonzero.y.values
        for x1, y1 in zip(xs, ys):
            link_id = nbr.sel(x=x1, y=y1, method='nearest', tolerance=resolution).squeeze().item()
            ix, jx = np.argwhere(stream.x.values == x1)[0], np.argwhere(stream.y.values == y1)[0]
            
            # check if point is valid
            if np.isnan(ix) | np.isnan(jx):
                print(x, y, xs, ys, link_id)
                print(ix, jx)
            if ~np.isnan(link_id):
                return [ix[0], jx[0], Point(x1, y1), link_id]
            
    return None


def add_lake_inflows(lakes_df, ppts, stream, acc):
    
    n = 0
    tot_pts = 0
    resolution = abs(stream.rio.resolution()[0])
    crs = stream.rio.crs.to_epsg()

    points_to_check = []
    # used_links = []
    # all_links = []
    # test_pts = []
    for _, row in lakes_df.iterrows():
        n += 1
        if n % 10 == 0:
            print(f'   Processing lake group {n}/{len(lakes_df)}.')
        
        lake_geom = row['geometry']        
        # resample the shoreline vector to prevent missing confluence points
        resampled_shoreline = redistribute_vertices(lake_geom.exterior, resolution).coords.xy
        xs = resampled_shoreline[0].tolist()
        ys = resampled_shoreline[1].tolist()

        # find the closest cell to within 1 pixel diagonal of the lake polygon boundary
        # this is the problem here.
        # what's happening is for each interpolated point on the line, 
        # we look for the nearest pixel in the stream raster
        # we should iterate through and find the nearest *stream pixel* 
        # and record it if 
        #           i)  it's not in a lake and 
        #           ii) not on a stream link already recorded
        px_pts = stream.sel(x=xs, y=ys, method='nearest', tolerance=resolution)
        latlon = list(set(zip(px_pts.x.values, px_pts.y.values)))
        if len(latlon) == 0:
            print('skip')
            continue
        
        # the line interpolation misses some cells,
        # so check around each point for stream cells
        # that aren't inside the lake polygon
        pl = mp.Pool()
        results = pl.map(find_link_ids, latlon)
        results = [r for r in results if r is not None]
        pl.close()
        
        pts = pd.DataFrame(results, columns=['ix', 'jx', 'geometry', 'link_id'])
        # drop duplicate link_ids
        pts['CONF'] = True
        pts = pts[~pts['link_id'].duplicated(keep='first')]
        pts.dropna(subset='geometry', inplace=True)
            
        points_to_check += [pts]
                                
    print(f'    {len(points_to_check)} points identified as potential lake inflows')
           
    n = 0
    all_pts = []
    acc_vals = []
    all_points_df = gpd.GeoDataFrame(pd.concat(points_to_check, axis=0), crs=f'EPSG:{crs}')
    all_points_df.reset_index(inplace=True, drop=True)

    for i, row in all_points_df.iterrows():
        
        pt = row['geometry']
        n += 1
        if n % 250 == 0:
            print(f'{n}/{len(points_to_check)} points checked.')
        
        # index_right is the lake id the point is contained in
        # don't let adjacent points both be pour points
        # but avoid measuring distance to points within lakes
        nearest_neighbour = ppts.distance(pt).min()

        # check the point is not within some distance (in m) of an existing point
        min_spacing = 500        
        if nearest_neighbour > min_spacing:
            all_pts.append(i)
            x, y = pt.x, pt.y
            acc_val = acc.sel(x=x, y=y, method='nearest').item()
            acc_vals.append(acc_val)
            
    all_points = all_points_df.iloc[all_pts].copy()
    all_points['acc'] = acc_vals
    return all_points
    
region_polygon_dir = os.path.join(BASE_DIR, 'input_data/region_polygons')
regions = sorted(list(set([e.split('_')[0] for e in os.listdir(region_polygon_dir)])))
# regions = ['HGW']

for region in regions:
    
    ppt_folder = os.path.join(DATA_DIR, f'pour_points/{region}')
    if not os.path.exists(ppt_folder):
        os.mkdir(ppt_folder)
    
    output_fname = f'{region}_pour_pts_filtered.geojson'
    output_fpath = os.path.join(ppt_folder, output_fname)
    
    if os.path.exists(output_fpath):
        print(f'    {region} pour points already processed.')
        continue 
    else:
        print(f'Processing {region}.')
    
    # import the stream link raster
    stream, _, _ = retrieve_raster(region, 'link')
    acc, _, _ = retrieve_raster(region, 'accum')
        
    resolution = abs(stream.rio.resolution()[0])
    
    # import pour points 
    ppts_fpath = os.path.join(DATA_DIR, f'pour_points/{region}/{region}_pour_pts.geojson') 
    region_ppts = gpd.read_file(ppts_fpath)
    
    t0 = time.time()
    # if the NHN features haven't been clipped to the region polygon, do so now
    region_polygon = get_region_polygon(region)
    lakes_output_dir = os.path.join(lakes_dir, 'clipped_lakes')
    lakes_df_fpath = os.path.join(lakes_output_dir, f'{region}_lakes.geojson')
    if not os.path.exists(lakes_df_fpath):
        if not os.path.exists(lakes_output_dir):
            os.mkdir(lakes_output_dir)
        print('    Creating region water bodies layer.')
        t1 = time.time()
        
        # import the NHN water body features
        bbox_geom = tuple(region_polygon.to_crs(lakes_crs).bounds.values[0])
        lake_features_box = gpd.read_file(lakes_fpath, 
                                      bbox=bbox_geom)
        # clip features to the region polygon
        region_polygon = region_polygon.to_crs(lake_features_box.crs)
        lake_features = gpd.clip(lake_features_box, region_polygon, keep_geom_type=False)
        t2 = time.time()
        print(f'    Lakes layer opened in {t2-t1:.0f}s')
        print(f'    Creating lakes geometry file for {region}')
        lakes_df = filter_lakes(lake_features, region_ppts, resolution)
        lakes_df = lakes_df[~lakes_df.geometry.is_empty]
        lakes_df.to_file(lakes_df_fpath)
        n_lakes = len(lakes_df)
        print(f'    File saved.  There are {n_lakes} water body objects in {region}.')
    else:
        lakes_df = gpd.read_file(lakes_df_fpath)
    
    lake_ppts = region_ppts.clip(lakes_df)
    
    filtered_ppts = region_ppts[~region_ppts['cell_idx'].isin(lake_ppts['cell_idx'])]
        
    print(f'    {len(filtered_ppts)}/{len(region_ppts)} confluence points are not in lakes ({len(region_ppts) - len(filtered_ppts)} points removed).')   
    new_pts = add_lake_inflows(lakes_df, filtered_ppts, stream, acc)
    
    output_ppts = gpd.GeoDataFrame(pd.concat([filtered_ppts, new_pts], axis=0), crs=f'EPSG:{stream.rio.crs.to_epsg()}')
    n_pts0, n_pts1, n_final = len(region_ppts), len(filtered_ppts), len(output_ppts)
        
    print(f'    {n_pts0-n_pts1} points eliminated (fall within lakes)')
    print(f'    {len(new_pts)} points added for lake inflows.')
    print(f'    {n_final} points after filter and merge. ({n_pts0-n_final} difference)')
    output_ppts['region_code'] = region
    # create separate columns for geometry latitude and longitude
    output_ppts['centroid_lon_m_3005'] = output_ppts.geometry.x
    output_ppts['centroid_lat_m_3005'] = output_ppts.geometry.y
    output_ppts.drop(labels=['cell_idx', 'ix', 'jx'], axis=1, inplace=True)
    
    output_ppts.to_file(output_fpath)
    te = time.time()
    utime = n_final / (te-t0)
    print(f'{region} processed in {te-t0:.0f}s ({utime:.2f}pts/s)') 
    print('-------------------------------------------------------') 
    print('')
    
