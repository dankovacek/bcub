import os

# from osgeo import gdal

# high-performance array library and tools for GPU
import jax.numpy as jnp

from shapely.validation import make_valid

import pandas as pd
import numpy as np

import rioxarray as rxr

from scipy.stats.mstats import gmean

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

from whitebox.whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()
wbt.verbose = False


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.join(ROOT_DIR, 'basin_generator')

ext_media_path = '/media/danbot/Samsung_T51/large_sample_hydrology'
SOURCE_DATA_DIR = os.path.join(ext_media_path, 'common_data')

# porosity and permeability sources
# glhymps_fpath = os.path.join(ext_media_path, 'GLHYMPS/GLHYMPS.gdb')
glhymps_fpath = os.path.join(SOURCE_DATA_DIR, 'GLHYMPS/GLHYMPS_clipped_4326.gpkg')

# import NALCMS raster
# land use / land cover
nalcms_fpath = os.path.join(SOURCE_DATA_DIR, 'NALCMS_NA/NA_NALCMS_2010_v2_land_cover_30m/NA_NALCMS_2010_v2_land_cover_30m.tif')
reproj_nalcms_path = os.path.join(SOURCE_DATA_DIR, 'NALCMS_NA/NA_NALCMS_2010_4326.tif')

def retrieve_raster(fpath):
    rds = rxr.open_rasterio(fpath, masked=True, mask_and_scale=True)
    affine = rds.rio.transform()
    return rds, rds.rio.crs, affine

if not os.path.exists(reproj_nalcms_path):
    nalcms, nalcms_crs, nalcms_affine = retrieve_raster(nalcms_fpath)
    print(f'   ...NALCMS imported, crs = {nalcms.rio.crs.to_epsg()}')
    print('Reproject NALCMS raster')

    warp_command = f'gdalwarp -q -s_srs "{nalcms.rio.crs.to_proj4()}" -t_srs EPSG:4326 -of gtiff {nalcms_fpath} {reproj_nalcms_path} -r bilinear -wo NUM_THREADS=ALL_CPUS'    
    os.system(warp_command)

# 
print(f'Opening NALCMS raster:')
nalcms, nalcms_crs, nalcms_affine = retrieve_raster(reproj_nalcms_path)
print(f'   ...NALCMS imported, crs = {nalcms.rio.crs.to_epsg()}')


DEM_source = 'USGS_3DEP'
DEM_DIR = os.path.join(SOURCE_DATA_DIR, f'DEM_data/processed_dem/{DEM_source}/')


def filter_and_explode_geoms(gdf, min_area):
    gdf.geometry = gdf.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)    
    gdf = gdf.explode()
    gdf['area'] = gdf.geometry.area / 1E6
    gdf = gdf[gdf['area'] >= min_area * 0.95]    
    return gdf

                
def dump_poly(inputs):
    raster_fpath, vector_fname, temp_folder, crs = inputs
    
    polygons = gpd.read_file(vector_fname)
    
    output_fnames = []
    for i, _ in polygons.iterrows():

        bdf = polygons.iloc[[i]]
        id = int(bdf['VALUE'].values[0])
        
        if crs == 4326:
            bdf = bdf.to_crs(crs)
            bdf.geometry = bdf.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)    
        else:
            # bdf.geometry = bdf.geometry.buffer(0.0)
            # bdf.geometry = bdf.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)    
            crs = 3005
            
        basin_fname = f'basin_temp_{id:05d}_{crs}.geojson'
        basin_fpath = os.path.join(temp_folder, basin_fname)
        bdf.to_file(basin_fpath)
             
        # New filename. Assumes input raster file has '.tif' extension
        # Might need to change how you build the output filename        
        raster_fname = raster_fpath.split('/')[-1]
        raster_fname = raster_fname.replace(".tif", f"_{int(id):05}.tif")
        fpath_out = os.path.join(temp_folder, raster_fname)
        
        # Do the actual clipping
        command = f'gdalwarp -s_srs epsg:{crs} -cutline {basin_fpath} -crop_to_cutline -multi -of gtiff {raster_fpath} {fpath_out} -wo NUM_THREADS=ALL_CPUS'
        
        os.system(command)
        # g = gdal.Warp(fpath_out, raster_fpath, format="GTiff",
        #                 cutlineDSName=basin_fpath,
        #                 cropToCutline=True)
        # # Return the filename
        output_fnames.append(fpath_out)

    g = None
    
    return output_fnames


def match_ppt_to_polygons_by_order(ppt_batch, polygon_df, resolution):

    try:
        assert len(ppt_batch) == len(polygon_df)
    except Exception as e:
        print(f' mismatched df lengths: ppt vs. polygon_df')
        print(len(ppt_batch), len(polygon_df))
        print('')

    polygon_df['acc_polygon'] = (polygon_df.geometry.area / (resolution[0] * resolution[1])).astype(int)
    polygon_df['ppt_acc'] = ppt_batch['acc'].values
    polygon_df['acc_diff'] = polygon_df['ppt_acc'] - polygon_df['acc_polygon']
    polygon_df['FLAG_acc_match'] = polygon_df['acc_diff'].abs() > 2

    polygon_df['ppt_x'] = ppt_batch.geometry.x.values
    polygon_df['ppt_y'] = ppt_batch.geometry.y.values
    polygon_df['cell_idx'] = ppt_batch['cell_idx'].values

    # polygon_df.reset_index(inplace=True, drop=True)
    polygon_df.set_index('VALUE', inplace=True)
    polygon_df.sort_index(inplace=True)
    polygon_df.index.name = 'ID'


    # Do I save ppt and polygon pair as individual files?  
    # is there better way to couple them for faster read/write?
    # for i, polygon in polygon_df[:5].iterrows():

    #     # check polygon validity
    #     if not polygon.geometry.is_valid:
    #         polygon.geometry = make_valid(polygon.geometry)

    #     basin = polygon_df.iloc[[i]].copy()
    #     ppt = ppt_batch.iloc[[i]].copy()

    #     # to see that the ordering matches up, save ppt/basin geometry pairs
    #     foo = gpd.GeoDataFrame(pd.concat([ppt, basin]), crs='EPSG:3005')
    #     foo.to_file(os.path.join(temp_folder, f'{i}_temp_ppt_polygon_pair.geojson'))

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


def calculate_gravelius_and_perimeter(polygon):    
    perimeter = polygon.geometry.length.values[0]
    area = polygon.geometry.area.values[0]
    if area == 0:
        return np.nan, perimeter
    else:
        perimeter_equivalent_circle = jnp.sqrt(4 * np.pi * area)
        gravelius = perimeter / perimeter_equivalent_circle
    
    return gravelius, perimeter / 1000

def check_lulc_sum(data):
    checksum = sum(list(data.values())) 
    lulc_check = 1-checksum
    if abs(lulc_check) >= 0.05:
        print(f'    ...lulc failed checksum: {checksum:.3f}')   
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
    unique, counts = jnp.unique(vals, return_counts=True)
    # create a dictionary of land cover values by coverage proportion
    # assuming raster pixels are equally sized, we can keep the
    # raster in geographic coordinates and just count pixel ratios
    prop_dict = {k: 1.0*v/n_pts for k, v in zip(unique, counts)}
    prop_dict = recategorize_lulc(prop_dict)
    return prop_dict


def process_lulc(raster_path):
    # basin_proj = basin.copy().to_crs(nalcms_crs)
    i = int(raster_path.split('_')[-1].split('.')[0])
    clipped_raster, _, _ = retrieve_raster(raster_path)
    # raster_loaded, lu_raster_clipped = clip_raster_to_basin(basin_proj, nalcms)
    # checksum verifies proportions sum to 1
    prop_dict = get_value_proportions(clipped_raster)
    lulc_check = check_lulc_sum(prop_dict)
    prop_dict['lulc_check'] = lulc_check
    prop_dict['ID'] = i
    os.remove(raster_path)
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
        return (df['area_frac'] * df[col]).sum()
    

def process_glhymps(inputs):
    
    i, row = inputs
        
    soil_data = {}
    soil_data['ID'] = i        
    basin = gpd.GeoDataFrame(geometry=[row['geometry']], crs='EPSG:3005')
    # ensure basin is projected to same CRS as GLHYMPS (4326)
    basin_proj = basin.to_crs(4326)

    # returns INTERSECTION
    gdf = gpd.read_file(glhymps_fpath, mask=basin_proj)
    # now clip to the basin polygon bounds
    clipped_soil = gpd.clip(gdf, mask=basin_proj)
    # now reproject to minimize spatial distortion
    clipped_soil = clipped_soil.to_crs(3005)
    
    porosity = get_soil_properties(clipped_soil, 'Porosity')
    permeability = get_soil_properties(clipped_soil, 'Permeability_no_permafrost')
    
    soil_data['Permeability_logk_m2'] = permeability
    soil_data['Porosity_frac'] = porosity
    
    return soil_data


def warp_raster(stn, new_proj, c, temp_dem_folder, temp_raster_path_in):
        
    if c != 'LAEA':
        t_srs = f'EPSG:{c}'
        proj_code = c
    else:
        t_srs = f"'{new_proj}'" 
        proj_code = 'LAEA'
    
    temp_raster_path_out = os.path.join(temp_dem_folder, f'{stn}_temp_{proj_code}.tif')

    warp_command = f'gdalwarp -q -s_srs EPSG:4326 -t_srs {t_srs} -of gtiff {temp_raster_path_in} {temp_raster_path_out} -wo NUM_THREADS=ALL_CPUS'    

    try:
        # print(warp_command)
        os.system(warp_command)            
        return True, temp_raster_path_out
    except Exception as ex:
        print('')
        print(f'Raster reprojection failed for {stn}.')
        print('')
        print(ex)
        print('')
        return False, None


def mat_mult(ddx, ddy):
    pdx = np.power(ddx, 2.0)
    pdy = np.power(ddy, 2.0)
    sum_dd = np.add(pdx, pdy)
    S = np.sqrt(sum_dd)
    return (180/np.pi)*np.arctan(np.nanmean(S))


# calculate circular mean aspect
def calculate_circular_mean_aspect(ddx, ddy):
    """
    Calculate the circular mean of slope directions given 
    a matrix of slopes. Return circular mean aspect in degrees.
    """
    A = (180 / np.pi)* np.arctan2(ddy, ddx)
    n_angles = jnp.count_nonzero(~np.isnan(A))
    sine_mean = jnp.divide(jnp.nansum(jnp.sin(jnp.radians(A))), n_angles)
    cosine_mean = jnp.divide(jnp.nansum(jnp.cos(jnp.radians(A))), n_angles)
    vector_mean = jnp.arctan2(sine_mean, cosine_mean)
    aspect_degrees = jnp.degrees(vector_mean)
    if aspect_degrees + 180 > 360:
        return aspect_degrees - 180
    else:
        return aspect_degrees + 180


def process_basin_terrain_attributes(raster_path):
    i = int(raster_path.split('_')[-1].split('.')[0])
    raster, _, _ = retrieve_raster(raster_path)
    basin_data = {}
    basin_data['ID'] = i
        
    _, median_el, _, _ = process_basin_elevation(raster)
    basin_data['Elevation_m'] = round(median_el, 1)

    return basin_data


def circular_mean_angle(A):
    A = np.where(A < 90, 90-A, 360 - (A-90))
    x_mean = jnp.nanmean(np.cos(np.radians(A)))
    y_mean = jnp.nanmean(np.sin(np.radians(A)))
    # mean angle (degrees, [-180, 180], CCW from east)
    mean_angle = jnp.degrees(np.arctan2(y_mean, x_mean))
    # convert back to CW from north
    mean_angle = 90 - mean_angle
    mean_angle = (mean_angle + 360) % 360
    return mean_angle


def wbt_aspect(raster_path):
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
    values are slightly different, WBT uses 5x5 matrix
    and taylor polynomial expansion fit.
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
    return jnp.nanmean(S)


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

    return basin_data#, ta, tb, tc, wb_ms, mean_aspect


def process_basin_shape_attributes(inputs):
    i, row = inputs
        
    basin_data = {}
    basin_data['ID'] = i
    
    existing_info = row.to_dict()
    basin_data.update(existing_info)        
    basin = gpd.GeoDataFrame(geometry=[row['geometry']], crs='EPSG:3005')
    
    gravelius, perimeter = calculate_gravelius_and_perimeter(basin)
    basin_data['Perimeter'] = perimeter
    basin_data['Gravelius'] = gravelius
    basin_data['Drainage_Area_km2'] = basin.area.values[0] / 1E6

    return basin_data



# concatenate all batches into a single geojson file
def merge_geojson_files(files, output_fpath, temp_folder):

    print('    Merging batch outputs to final output file!')
    
    batch_dfs = []

    files = sorted(list(set(files)))

    for file in files:
        fpath = os.path.join(temp_folder, file)
        layer = gpd.read_file(fpath)
        batch_dfs.append(layer)
        
    all_data = gpd.GeoDataFrame(pd.concat(batch_dfs), crs=layer.crs)
    all_data.to_file(output_fpath)
    all_data.to_file(
        output_fpath.replace('.geojson', '.gpkg'), 
        driver='GPKG', 
        layer=f'pour_points')
        
    # clean up temporary files
    for f in list(set(files)):
        os.remove(os.path.join(temp_folder, f))
        

