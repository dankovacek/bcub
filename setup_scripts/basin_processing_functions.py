import os

from shapely.validation import make_valid

import pandas as pd
import numpy as np
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import rioxarray as rxr
from scipy.stats.mstats import gmean

from whitebox.whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()
wbt.verbose = False

def retrieve_raster(fpath):
    rds = rxr.open_rasterio(fpath, masked=True, mask_and_scale=True)
    affine = rds.rio.transform()
    return rds, rds.rio.crs, affine


def filter_and_explode_geoms(gdf, min_area):
    gdf.geometry = gdf.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)    
    gdf = gdf.explode()
    gdf['area'] = round(gdf.geometry.area / 1E6, 2)
    gdf = gdf[gdf['area'] >= min_area * 0.95]    
    return gdf

                
def dump_poly(inputs):
    """Take the polygon batches and create raster clips with each polygon.
    Save the individual polygons as geojson files for later use.

    Args:
        inputs (array): raster file, vector file, temporary file folder, and the raster crs.

    Returns:
        string: list of filepaths for the clipped rasters.
    """
    region, layer, i, row, crs, raster_fpath, temp_folder = inputs

    bdf = gpd.GeoDataFrame(geometry=[row['geometry']], crs=crs)

    basin_fname = f'basin_temp_{i:05d}.geojson'
    basin_fpath = os.path.join(temp_folder, basin_fname)
    raster_fname = f'{region}_basin_{layer}_temp_{int(i):05}.tif'
    fpath_out = os.path.join(temp_folder, raster_fname)

    if (not os.path.exists(basin_fpath)):
        bdf.geometry = bdf.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)    
        bdf.to_file(basin_fpath, driver='GeoJSON')

    # New filename. Assumes input raster file has '.tif' extension
    # Might need to change how you build the output filename         
    if not os.path.exists(fpath_out):
        # Do the actual clipping
        command = f'gdalwarp -s_srs {crs} -cutline {basin_fpath} -crop_to_cutline -multi -of gtiff {raster_fpath} {fpath_out} -wo NUM_THREADS=ALL_CPUS'
        print('doing gdalwarp')
        print(command)
        print('   ...')
        os.system(command)

    g = None
    
    return fpath_out


def match_ppt_to_polygons_by_order(ppt_batch_path, polygon_df, resolution):
    """The WBT "unnest_basins" function does not preserve order of polygons & pour points,
    however the output contains a VALUE that can be used to match the polygon order
    with the input pour point dataframe.  This function does that matching and returns
    a dataframe with the polygon VALUE as the index.

    Args:
        ppt_batch (geodataframe): original batch of pour points
        polygon_df (geodataframe): dataframe of basin polygons
        resolution (tuple): source raster resolution for calculating area 

    Returns:
        geodataframe: geodataframe ordered by polygon VALUE to match the input pour point dataframe
    """
    ppt_batch = gpd.read_file(ppt_batch_path)
    try:
        assert len(ppt_batch) == len(polygon_df)
    except Exception as e:
        print(f' mismatched df lengths: ppt vs. polygon_df')
        print(len(ppt_batch), len(polygon_df))
        print('')

    polygon_df['acc_polygon'] = (polygon_df.geometry.area / (resolution[0] * resolution[1])).astype(int)
    polygon_df['ppt_acc'] = [e if (e > 0) else None for e in ppt_batch['acc'].values.astype(int)]
    
    # create latitude and longitude columns from the ppt_batch dataframe
    polygon_df['ppt_lon_m_3005'] = ppt_batch['geometry'].x
    polygon_df['ppt_lat_m_3005'] = ppt_batch['geometry'].y    
    # polygon_df.reset_index(inplace=True, drop=True)
    polygon_df['VALUE'] = polygon_df['VALUE'].astype(int)
    polygon_df.set_index('VALUE', inplace=True)
    polygon_df.sort_index(inplace=True)
    polygon_df.index.name = 'ID'
    return polygon_df


def check_and_repair_geometries(in_feature):

    # avoid changing original geodf
    in_feature = in_feature.copy(deep=True)

    # drop any missing geometries
    in_feature = in_feature[~(in_feature.is_empty)]
    
    # Repair broken geometries
    for index, row in in_feature.iterrows(): # Looping over all polygons
        if row['geometry'].is_valid:
            next
        else:
            fix = make_valid(row['geometry'])
            try:
                in_feature.loc[[index],'geometry'] =  fix # issue with Poly > Multipolygon
            except ValueError:
                in_feature.loc[[index],'geometry'] =  in_feature.loc[[index], 'geometry'].buffer(0)
    return in_feature


def process_basin_elevation(clipped_raster):
    # evaluate masked raster data
    values = clipped_raster.data.flatten()
    mean_val = np.nanmean(values)
    median_val = np.nanmedian(values)
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    return mean_val, median_val, min_val, max_val


def check_lulc_sum(i, data):
    checksum = round(sum(list(data.values())), 2)
    lulc_check = 1-checksum
    if abs(lulc_check) >= 0.05:
        print(f'    ...checksum flag on {i}: {checksum:.3f}')   
    return lulc_check


def recategorize_lulc(data):    
    forest = ('Land_Use_Forest_frac', [1, 2, 3, 4, 5, 6])
    shrub = ('Land_Use_Shrubs_frac', [7, 8, 11])
    grass = ('Land_Use_Grass_frac', [9, 10, 12, 13, 16])
    wetland = ('Land_Use_Wetland_frac', [14])
    crop = ('Land_Use_Crops_frac', [15])
    urban = ('Land_Use_Urban_frac', [17])
    water = ('Land_Use_Water_frac', [18])
    snow_ice = ('Land_Use_Snow_Ice_frac', [19])
    lulc_dict = {}
    for label, p in [forest, shrub, grass, wetland, crop, urban, water, snow_ice]:
        prop_vals = round(sum([data[e] if e in data.keys() else 0 for e in p]), 2)
        lulc_dict[label] = prop_vals
    return lulc_dict
    

def get_value_proportions(data):
    # vals = data.data.flatten()
    all_vals = data.data.flatten()
    vals = all_vals[~np.isnan(all_vals)]
    n_pts = len(vals)
    unique, counts = np.unique(vals, return_counts=True)
    # create a dictionary of land cover values by coverage proportion
    # assuming raster pixels are equally sized, we can keep the
    # raster in geographic coordinates and just count pixel ratios
    prop_dict = {int(k): round(1.0*v/n_pts, 4) for k, v in zip(list(unique), list(counts)) if k != -9999}
    
    prop_dict = recategorize_lulc(prop_dict)
    return prop_dict


def process_lulc(inputs):
    row, nalcms_path, nalcms_crs, temp_folder = inputs
    
    geom_id = row['ID']
        
    temp_polygon = gpd.GeoDataFrame(geometry=[row['basin_geometry']], crs=nalcms_crs) 
    temp_raster = os.path.join(temp_folder, f'temp_nalcms_{geom_id:05d}.tif')
    # .geojson format gives the gdalwarp function issues
    # maybe because of the crs?  using .shp works correctly.
    temp_polygon_fpath = os.path.join(temp_folder, f'temp_polygon_{geom_id:05d}.shp')
    if not os.path.exists(temp_polygon_fpath):
            temp_polygon.to_file(temp_polygon_fpath)

    if not os.path.exists(temp_raster): 
        command = f"gdalwarp -s_srs '{nalcms_crs.to_wkt()}' -cutline {temp_polygon_fpath} -crop_to_cutline -multi -of gtiff {nalcms_path} {temp_raster} -wo NUM_THREADS=ALL_CPUS"
        os.system(command)
        
    clipped_raster, _, _ = retrieve_raster(temp_raster)
    
    # checksum verifies proportions sum to 1
    prop_dict = get_value_proportions(clipped_raster)
    lulc_check = check_lulc_sum(geom_id, prop_dict)
    prop_dict['lulc_check'] = lulc_check
    prop_dict['ID'] = row['ID']
    # os.remove(temp_raster)
    # os.remove(temp_polygon_fpath)
    return prop_dict


def get_soil_properties(merged, col):
    # dissolve polygons by unique parameter values

    geometries = check_and_repair_geometries(merged)

    df = geometries[[col, 'geometry']].copy().dissolve(by=col, aggfunc='first')
    df[col] = df.index.values
    # re-sum all shape areas
    df['Shape_Area'] = df.geometry.area
    # calculuate area fractions of each unique parameter value
    df['area_frac'] = df['Shape_Area'] / df['Shape_Area'].sum()
    # check that the total area fraction = 1
    total = round(df['area_frac'].sum(), 1)
    sum_check = total == 1.0
    if not sum_check:
        print(f'    Area proportions do not sum to 1: {total:.2f}')
        if np.isnan(total):
            return np.nan
        elif total < 0.9:
            return np.nan
    
    # area_weighted_vals = df['area_frac'] * df[col]
    if 'Permeability' in col:
        # calculate geometric mean
        # here we change the sign (all permeability values are negative)
        # and add it back at the end by multiplying by -1 
        # otherwise the function tries to take the log of negative values
        return gmean(np.abs(df[col]), weights=df['area_frac']) * -1
    else:
        # calculate area-weighted arithmetic mean
        # for porosity
        return (df['area_frac'] * df[col]).sum()
    

def process_glhymps(inputs):
    
    row, crs, glhymps_fpath = inputs
    
    basin = gpd.GeoDataFrame(geometry=[row['basin_geometry']], crs=crs)
    
    basin.rename(columns={'basin_geometry': 'geometry'}, inplace=True)
        
    soil_data = {}
    soil_data['ID'] = row['ID']

    # returns INTERSECTION
    gdf = gpd.read_file(glhymps_fpath, mask=basin)
    
    # now clip to the basin polygon bounds
    clipped_soil = gpd.clip(gdf, mask=basin)
    
    # now reproject to reduce spatial distortion for calculating areas
    clipped_soil = clipped_soil.to_crs(3005)
    
    porosity = get_soil_properties(clipped_soil, 'Porosity')
    permeability = get_soil_properties(clipped_soil, 'Permeability_no_permafrost')
    
    soil_data['Permeability_logk_m2'] = permeability
    soil_data['Porosity_frac'] = porosity
    
    return soil_data


# def matrix_mult(ddx, ddy):
#     pdx = np.power(ddx, 2.0)
#     pdy = np.power(ddy, 2.0)
#     sum_dd = np.add(pdx, pdy)
#     S = np.sqrt(sum_dd)
#     return (180/np.pi)*np.arctan(np.nanmean(S))


# calculate circular mean aspect
def calculate_circular_mean_aspect(ddx, ddy):
    """
    Calculate the circular mean of slope directions given 
    a matrix of slopes. Return circular mean aspect in degrees.
    """
    A = (180 / np.pi)* np.arctan2(ddy, ddx)
    n_angles = np.count_nonzero(~np.isnan(A))
    sine_mean = np.divide(np.nansum(np.sin(np.radians(A))), n_angles)
    cosine_mean = np.divide(np.nansum(np.cos(np.radians(A))), n_angles)
    vector_mean = np.arctan2(sine_mean, cosine_mean)
    aspect_degrees = np.degrees(vector_mean)
    if aspect_degrees + 180 > 360:
        return aspect_degrees - 180
    else:
        return aspect_degrees + 180


def process_terrain_attributes(raster_path):
    
    basin_id = int(raster_path.split('_')[-1].split('.')[0])

    raster, _, _ = retrieve_raster(raster_path)
    basin_data = {}
    basin_data['ID'] = basin_id

    _, median_el, _, _ = process_basin_elevation(raster)
    basin_data['Elevation_m'] = round(median_el, 1)
    
    basin_data['Aspect_deg'] = int(wbt_aspect(raster_path))
    basin_data['Slope_deg'] = round(wbt_slope(raster_path), 1)
    
    return basin_data


def circular_mean_angle(A):
    A = np.where(A < 90, 90-A, 360 - (A-90))
    x_mean = np.nanmean(np.cos(np.radians(A)))
    y_mean = np.nanmean(np.sin(np.radians(A)))
    # mean angle (degrees, [-180, 180], CCW from east)
    mean_angle = np.degrees(np.arctan2(y_mean, x_mean))
    # convert back to CW from north
    mean_angle = 90 - mean_angle
    mean_angle = (mean_angle + 360) % 360
    return mean_angle


def wbt_aspect(raster_path):
    """Calculate the overall orientation of the basin.

    Args:
        raster_path (string): Raster cropped to the drainage basin.

    Returns:
    float: circular mean angle of all pixel slope direction.
    """
    out_path = raster_path.replace('.tif', '_aspect.tif')
    wbt.aspect(
        raster_path, 
        out_path, 
        zfactor=None, 
        # callback=default_callback
    )
    raster, _, _ = retrieve_raster(out_path)
    A = raster.data[0]
    # this is to convert to math convention (+ve CCW from due east)
    # leave commented to use CW from North
    # A = A.where(A < 90, 90-A, 360 - (90-A))
    # need to convert to math convention to calculate
    # the circular mean, then convert back to geographic convention
    mean_angle = circular_mean_angle(A)

    os.remove(out_path)

    return mean_angle


def wbt_slope(raster_path):
    """
    Values are slightly different between methods: 
    WBT uses 5x5 matrix and taylor polynomial fit.
    Convolution is just 3x3 window.
    Values are consistent but slightly lower for WBT avg. slope.
    """
    out_path = raster_path.replace('.tif', '_slope.tif')
    wbt.slope(
        raster_path, 
        out_path, 
        zfactor=None,
        units='degrees' 
        # callback=default_callback
    )
    raster, _, _ = retrieve_raster(out_path)
    S = raster.data[0]
    os.remove(out_path)
    return np.nanmean(S)


def calc_slope_aspect(raster_path):
    # t0 = time.time()
    i = int(raster_path.split('_')[-1].split('.')[0])
    
    basin_data = {}
    basin_data['ID'] = i
    
    wb_aspect = wbt_aspect(raster_path)
    wb_slope = wbt_slope(raster_path)
    # print(f'aspect, slope: {aspect:.1f} {slope:.2f} ')
    basin_data['Val_Slope_deg'] = round(wb_slope, 1)
    basin_data['Val_Aspect_deg'] = round(wb_aspect, 1)

    return basin_data


# concatenate all batches into a single geojson file
def merge_geojson_files(files, output_fpath, temp_folder):

    print('    Merging batch outputs to final output file.')
    print(f'creating output parquet file: {output_fpath}')
    batch_dfs = []

    files = sorted(list(set(files)))

    for file in files:
        fpath = os.path.join(temp_folder, file)
        layer = gpd.read_file(fpath)
        batch_dfs.append(layer)
        
    return gpd.GeoDataFrame(pd.concat(batch_dfs), crs=layer.crs)


        

