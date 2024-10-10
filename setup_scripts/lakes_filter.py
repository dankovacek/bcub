# generate basins
import os

import time
import itertools

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from rasterio.features import geometry_mask

import multiprocessing as mp

import basin_processing_functions as bpf

from shapely.geometry import Point, LineString, Polygon
gpd.options.io_engine = "pyogrio"

t0 = time.time()
DEM_source = 'USGS_3DEP'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')
DEM_DIR = os.path.join(BASE_DIR, 'processed_data/processed_dem')

output_path = os.path.join(DATA_DIR, f'pour_points/')
if not os.path.exists(output_path):
    os.mkdir(output_path)

# this should correspond with the threshold accumulation
# set in "derive_flow_accumulation.py"
min_basin_area = 1 # km^2

lakes_dir = os.path.join(BASE_DIR, 'input_data/BasinATLAS')
lakes_fpath = os.path.join(lakes_dir, 'HydroLAKES_clipped.gpkg')
# the HydroLAKES CRS is EPSG:4326
lakes_crs = 4326

# import the NALCMS "perennial ice" polygons and intersect with the region polygon
ice_polygon_fpath = os.path.join(BASE_DIR, 'input_data/NALCMS/NALCMS_2020_ice_mass_polygons_4269.shp')
ice_polygons = gpd.read_file(ice_polygon_fpath)

if not os.path.exists(lakes_fpath):
    err_msg = f'HydroLAKES file not found at {lakes_fpath}.  Download from https://www.hydrosheds.org/products/hydrolakes.  See README for details.'
    raise Exception(err_msg)


def get_region_polygon(region):
    polygon_path = os.path.join(BASE_DIR, 'input_data/region_polygons/')
    file = f'{region}_covering_basins_R0.geojson'
    fpath = os.path.join(polygon_path, file)
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


def retrieve_raster(region, raster_type):
    filename = f'{region}_{DEM_source}_3005_{raster_type}.tif'
    raster_path = os.path.join(DEM_DIR, f'{filename}')
    raster = rxr.open_rasterio(raster_path, mask_and_scale=True)
    crs = raster.rio.crs
    affine = raster.rio.transform(recalc=False)
    clipped = clip_raster_with_polygon(region, raster, crs, raster.rio.resolution())
    return clipped, crs, affine

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
            # for finding points in lakes, need to use projected crs

    if region_ppts.crs != lakes_df.crs:
        lakes_df = lakes_df.to_crs(ppts.crs)
    
    # reproject to projected CRS before calculating area
    lakes_df['area'] = lakes_df.geometry.area
    lakes_df['lake_id'] = lakes_df.index.values
        
    # filter lakes smaller than 0.1 km^2
    min_area = 0.1 * 1E6
    lakes_df = lakes_df[lakes_df['area'] > min_area]
    lakes_df = lakes_df.dissolve().explode(index_parts=False).reset_index(drop=True)
    
    # filter out Point type geometries
    lakes_df = lakes_df[~lakes_df.geometry.type.isin(['Point', 'LineString'])]
    # find and fill holes in polygons
    lakes_df.geometry = bpf.fill_holes(lakes_df.geometry)
        
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
    n, num_vertices, geom = inputs
    d = n / num_vertices
    return (n, geom.interpolate(d, normalized=True))
 

def prepare_line_inputs(geom, distance):
    """Evenly resample along a linestring
    See this SO post:
    https://gis.stackexchange.com/a/367965/199640
    
    Args:
        geom (polygon): lake boundary geometry
        distance (numeric): distance between points in the modified linestring
    """
    if geom.geom_type in ['LineString', 'LinearRing']:
        num_vertices = int(round(geom.length / distance))
        
        if num_vertices == 0:
            raise Exception('No vertices found in linestring.')
        inputs = [(n, num_vertices, geom) for n in range(num_vertices)]
        return inputs
    
    elif geom.geom_type == 'MultiLineString':
        print('      ...$$$$$$$$$$$$$$$$$$  Multilinestring found')
        ls = gpd.GeoDataFrame(geometry=[geom], crs='EPSG:3005')
        geoms = ls.explode().reset_index(drop=True).geometry.values
        parts = [prepare_line_inputs(part, distance) for part in geoms if not part.is_empty]
        return list(itertools.chain.from_iterable(parts))
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))


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
        return inputs
    
    elif geom.geom_type == 'MultiLineString':
        print('here??????????????????????????????????')
        ls = gpd.GeoDataFrame(geometry=[geom], crs='EPSG:3005')
        geoms = ls.explode().reset_index(drop=True).geometry.values
        return type(geom)([redistribute_vertices(part, distance)
                 for part in geoms if not part.is_empty])
        # return type(geom)([p for p in parts if not p.is_empty])
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


def mask_stream_raster(stream, lakes_df):
    # Get the geometry of the polygons
    geometries = [geom for geom in lakes_df.geometry]

    # Create a mask with the same shape as the raster
    out_shape = (stream.shape[1], stream.shape[2])  # assuming stream has dimensions (band, y, x)
    mask = geometry_mask(geometries, transform=stream.rio.transform(), invert=True, out_shape=out_shape)
    # Apply the mask to set values inside polygons to NaN
    masked_stream = np.where(mask, np.nan, stream.data)
    # Update the stream DataArray with the masked data
    stream.data = masked_stream
    return stream


def check_point_validity(region, xs, ys, stream, resample_distance):
    # Get the bounds of the stream DataArray
    x_min, x_max = stream.x.min().item(), stream.x.max().item()
    y_min, y_max = stream.y.min().item(), stream.y.max().item()

    xs, ys = np.array(xs), np.array(ys)

    # Filter xs and ys to ensure they are within bounds
    valid_xs = np.logical_and(xs >= x_min, xs <= x_max)
    valid_ys = np.logical_and(ys >= y_min, ys <= y_max)
    valid_indices = np.logical_and(valid_xs, valid_ys)

    filtered_xs = xs[valid_indices]
    filtered_ys = ys[valid_indices]

    # Collect out-of-bounds points
    out_of_bounds_xs = xs[~valid_indices]
    out_of_bounds_ys = ys[~valid_indices]

    # Perform selection if there are valid points
    if len(filtered_xs) > 0 and len(filtered_ys) > 0:
        px_pts = stream.sel(x=filtered_xs, y=filtered_ys, method='nearest', tolerance=1.01 * resample_distance)
    else:
        raise Exception('No valid points found.')

    # Create GeoDataFrame for out-of-bounds points
    out_of_bounds_points = [Point(x, y) for x, y in zip(out_of_bounds_xs, out_of_bounds_ys)]
    out_of_bounds_gdf = gpd.GeoDataFrame(geometry=out_of_bounds_points)

    # Save out-of-bounds points to a GeoJSON file
    folder = os.path.join(BASE_DIR, 'processed_data', 'pour_points', region)
    fname = f'{region}_out_of_bounds_points.geojson'
    out_of_bounds_gdf.to_file(os.path.join(folder, fname), driver='GeoJSON')
    return px_pts


def process_batch_results(region, results, stream, resample_distance, pool):
    df = pd.DataFrame(results, columns=['n', 'geometry'])
    df = df.sort_values(by='n').reset_index(drop=True)
    resampled_shoreline = LineString(df['geometry'].values).coords.xy
    xs = resampled_shoreline[0].tolist()
    ys = resampled_shoreline[1].tolist()

    # check validity of the shoreline pixels
    px_pts = check_point_validity(region, xs, ys, stream, resample_distance)
    # find the closest cell to within 1 pixel diagonal of the lake polygon boundary
    # For each interpolated point on the line, 
    # we look for the nearest pixel in the stream raster
    # we should iterate through and find the nearest *stream pixel* 
    # and record it if 
    #           i)  it's not in a lake and 
    #           ii) not on a stream link already recorded
    # first, filter the stream raster to delete stream pixels within any lake polygon
    # then, for each point on the lake boundary, find the nearest stream pixel    
    latlon = list(set(zip(px_pts.x.values, px_pts.y.values)))

    if len(latlon) == 0:
        raise Exception('No lakes found.')
    
    # the line interpolation misses some cells,
    # so check around each point for stream cells
    # that aren't inside the lake polygon
    results = pool.map(find_link_ids, latlon)
    results = [r for r in results if r is not None]
    return results


def add_lake_inflows(region, lakes_df, ppts, stream, acc):
    
    n = 0
    resolution = abs(stream.rio.resolution()[0])
    crs = stream.rio.crs.to_epsg()

    # points_to_check = []
    print(f'   Processing {len(lakes_df)} lake shorelines.')
    # set the interval (distance) to interpolate along the linestring
    resample_distance = resolution * 1.5
    resampled_shoreline = list(itertools.chain.from_iterable(
        prepare_line_inputs(row['geometry'].exterior, resample_distance) for _, row in lakes_df.iterrows()
    ))
    print(f'     {len(resampled_shoreline)} points to check.')
    # resample the shoreline vector to prevent missing confluence points
    all_points = []
    n_chunks = len(resampled_shoreline) / 25000
    chunks = np.array_split(resampled_shoreline, int(n_chunks) + 1)
    pool = mp.Pool()
    
    b = 0
    for chunk in chunks:
        b += 1
        print(f'    Processing batch {b}/{len(chunks)}')
        results = pool.map(interpolate_line, chunk)
        batch_points = process_batch_results(region, results, stream, resample_distance, pool)
        all_points+= batch_points
    pool.close()
    
    # convert to geodataframe and drop duplicate link_ids
    pts = gpd.GeoDataFrame(all_points, columns=['i_row', 'j_col', 'geometry', 'link_id'], crs=f'EPSG:{crs}')
    pts['CONF'] = True
    pts = pts[~pts['link_id'].duplicated(keep='first')]
    pts.dropna(subset='geometry', inplace=True)                                        
    print(f'    {len(pts)} points identified as potential lake inflows')
         
    all_pts = []
    acc_vals = []
    pts.reset_index(inplace=True, drop=True)
    pts_idx = pts.sindex
    ppts_idx = ppts.sindex
    for i, row in pts.iterrows():        
        pt = row['geometry']
        if i % 250 == 0:
            print(f'{i}/{len(pts)} points checked.')
        
        # index_right is the lake id the point is contained in
        # don't let adjacent points both be pour points
        # but avoid measuring distance to points within lakes
        nearest_neighbour = ppts.distance(pt).min()

        # check the point is not within some distance (in m) of an existing point
        min_spacing = 250
        if nearest_neighbour > min_spacing:
            all_pts.append(i)
            x, y = pt.x, pt.y
            acc_val = acc.sel(x=x, y=y, method='nearest').item()
            acc_vals.append(acc_val)
            
    pts = pts.iloc[all_pts].copy()
    pts['acc'] = acc_vals
    return pts
    
# open region polygon dataframe
region_gdf = gpd.read_file(os.path.join(BASE_DIR, 'input_data/BCUB_regions_merged_R0.geojson'))
print(f'    ...loaded resources in {time.time()-t0:.1f}s.')
for region in sorted(list(set(region_gdf['region_code'].values))):
    
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

    region_polygon = region_gdf[region_gdf['region_code'] == region].copy()
            
    # import the stream link raster
    stream, _, _ = retrieve_raster(region, 'link')
    acc, _, _ = retrieve_raster(region, 'accum')
        
    resolution = abs(stream.rio.resolution()[0])
    
    # import pour points 
    ppts_fpath = os.path.join(DATA_DIR, f'pour_points/{region}/{region}_pour_pts.geojson') 
    region_ppts = gpd.read_file(ppts_fpath)
    
    t0 = time.time()
    # if the NHN features haven't been clipped to the region polygon, do so now
    
    lakes_output_dir = os.path.join(lakes_dir, 'clipped_lakes')
    lakes_df_fpath = os.path.join(lakes_output_dir, f'{region}_lakes.geojson')

    if not os.path.exists(lakes_df_fpath):
        if not os.path.exists(lakes_output_dir):
            os.mkdir(lakes_output_dir)
        print('    Creating region water bodies layer.')
        t1 = time.time()
        
        # import the NHN water body features
        lakes_poly = region_polygon.copy().to_crs(lakes_crs)

        # bbox must be in minx, miny, maxx, maxy order
        region_bbox = tuple(lakes_poly.total_bounds)
        lake_features_clip = gpd.read_file(lakes_fpath, 
                                      bbox=region_bbox)
        t2 = time.time()
        print(f'    Lakes layer opened in {t2-t1:.0f}s ({len(lake_features_clip)} features).')
        # filter the lakes for those that are contained in the region polygon

        filtered_lakes = gpd.sjoin(lake_features_clip, lakes_poly, how='inner', op='within')
        filtered_lake_features = lake_features_clip.loc[filtered_lakes.index]
        print(f'    Creating lakes geometry file for {region}')
        lakes_df = filter_lakes(filtered_lake_features, region_ppts, resolution)
        lakes_df = lakes_df[~lakes_df.geometry.is_empty]
        lakes_df.to_file(lakes_df_fpath)

        n_lakes = len(lakes_df)
        print(f'    File saved.  There are {n_lakes} water body objects in {region}.')
    else:
        lakes_df = gpd.read_file(lakes_df_fpath)
    
    lake_ppts = region_ppts.clip(lakes_df)
    filtered_ppts = region_ppts[~region_ppts['cell_idx'].isin(lake_ppts['cell_idx'])]
        
    print(f'    {len(filtered_ppts)}/{len(region_ppts)} confluence points are not in lakes ({len(region_ppts) - len(filtered_ppts)} points removed).')   

    # set the stream pixels to nan where they fall within lake polygons
    stream = mask_stream_raster(stream, lakes_df)
    new_pts = add_lake_inflows(region, lakes_df, filtered_ppts, stream, acc)
    
    output_ppts = gpd.GeoDataFrame(pd.concat([filtered_ppts, new_pts], axis=0), crs=f'EPSG:{stream.rio.crs.to_epsg()}')
    n_pts0, n_pts1, n_final = len(region_ppts), len(filtered_ppts), len(output_ppts)
        
    print(f'    {n_pts0-n_pts1} points eliminated (fall within lakes)')
    print(f'    {len(new_pts)} points added for lake inflows.')
    print(f'    {n_final} points after filter and merge. ({n_pts0-n_final} difference)')
    output_ppts['region_code'] = region
    # create separate columns for geometry latitude and longitude
    output_ppts['centroid_lon_m_3005'] = output_ppts.geometry.x
    output_ppts['centroid_lat_m_3005'] = output_ppts.geometry.y
    
    if not region_polygon.crs == output_ppts.crs:
        region_polygon = region_polygon.to_crs(output_ppts.crs)
    # mark which pour points are in perennial ice polygons
    # get the intersecting ice_mask polygons using sjoin
    if not ice_polygons.crs == region_polygon.crs:
        ice_polygons = ice_polygons.to_crs(region_polygon.crs)

    # creating a label to indicate whether the point falls within an area of perennial ice
    intersecting = gpd.sjoin(ice_polygons, region_polygon, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')
    # get the indices of the rows in the ice_df that intersect with the region polygon
    ice_df = ice_polygons[ice_polygons.index.isin(list(intersecting.index.values))].copy()
    # ice_df.to_file(os.path.join(BASE_DIR, 'input_data/NALCMS', f'{region}_ice.geojson'))

    # Perform a spatial join between the points and the filtered ice polygons
    joined = output_ppts.sjoin(ice_df, how='inner', predicate='within')

    # get the indices of the ppts that are in
    # Create a boolean column based on the spatial join results
    output_ppts['in_perennial_ice'] = output_ppts.index.isin(list(joined.index.values))
    n_in_ice = output_ppts['in_perennial_ice'].sum()
    print(f'    {n_in_ice} points are contained in perennial ice polygons.')
    
    output_ppts.to_file(output_fpath)
    te = time.time()
    utime = n_final / (te-t0)
    print(f'{region} processed in {te-t0:.0f}s ({utime:.2f}pts/s)') 
    print('-------------------------------------------------------') 
    print('')
    
