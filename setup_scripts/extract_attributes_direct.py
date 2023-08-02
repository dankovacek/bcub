# generate basins
import os
import time
import psutil
import random

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import geopandas as gpd
import pandas as pd
import rioxarray as rxr

import multiprocessing as mp

from shapely.geometry import Point
from shapely.validation import make_valid

from whitebox.whitebox_tools import WhiteboxTools

import basin_processing_functions as bpf

wbt = WhiteboxTools()
wbt.verbose = False

DEM_source = 'USGS_3DEP'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEM_folder = os.path.join(BASE_DIR, 'input_data/processed_dem/')
region_files = os.listdir(DEM_folder)
region_codes = sorted(list(set([e.split('_')[0] for e in region_files])))

#########################
# update these file paths
#########################
common_data = '/home/danbot2/code_5820/large_sample_hydrology/common_data'
nalcms_dir = os.path.join(BASE_DIR, 'input_data/NALCMS/')
glhymps_dir = os.path.join(BASE_DIR, 'input_data/GLHYMPS/')
nalcms_fpath = os.path.join(nalcms_dir, 'NA_NALCMS_2010_v2_land_cover_30m.tif')
glhymps_fpath = os.path.join(glhymps_dir, 'GLHYMPS_clipped_4326.gpkg')

# we need to reproject the NALCMS raster
# and clip it to the region bounds
reproj_nalcms_path = os.path.join(BASE_DIR, 'input_data/NALCMS/NA_NALCMS_landcover_2010_3005_clipped.tif')
mask_path = os.path.join(BASE_DIR, 'input_data/region_polygons/region_bounds.geojson')



if not os.path.exists(reproj_nalcms_path):
    nalcms, nalcms_crs, nalcms_affine = bpf.retrieve_raster(nalcms_fpath)
    raster_wkt = nalcms_crs.to_wkt()
        
    reproj_bounds = gpd.read_file(mask_path).to_crs(nalcms_crs)
    reproj_bounds_path = os.path.join(BASE_DIR, 'input_data/region_polygons/region_bounds_reproj.shp')
    reproj_bounds.to_file(reproj_bounds_path)
    
    # first clip the raster, then reproject to EPSG 3005
    print('Clipping NALCMS raster to region bounds.')
    clipped_nalcms_path = os.path.join(BASE_DIR, 'input_data/NALCMS/NA_NALCMS_landcover_2010_clipped.tif')
    clip_command = f"gdalwarp -s_srs '{raster_wkt}' -cutline {reproj_bounds_path} -crop_to_cutline -multi -of gtiff {nalcms_fpath} {clipped_nalcms_path} -wo NUM_THREADS=ALL_CPUS"
    os.system(clip_command)
    
    print('\nReprojecting clipped NALCMS raster.')
    warp_command = f"gdalwarp -q -s_srs '{raster_wkt}' -t_srs EPSG:3005 -of gtiff {clipped_nalcms_path} {reproj_nalcms_path} -r bilinear -wo NUM_THREADS=ALL_CPUS"
    os.system(warp_command) 
    
    # remove the intermediate step
    if os.path.exists(clipped_nalcms_path):
        os.remove(clipped_nalcms_path)
        

nalcms, nalcms_crs, nalcms_affine = bpf.retrieve_raster(reproj_nalcms_path)


DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')

def retrieve_raster(region):
    filename = f'{region}_USGS_3DEP_3005.tif'
    fpath = os.path.join(DEM_folder, filename)
    raster = rxr.open_rasterio(fpath, mask_and_scale=True)
    crs = raster.rio.crs
    affine = raster.rio.transform(recalc=False)
    return raster, crs, affine


def get_region_area(region):
    polygon_path = os.path.join(DATA_DIR, 'merged_basin_groups/region_polygons/')
    poly_files = os.listdir(polygon_path)
    file = [e for e in poly_files if e.startswith(region)]
    if len(file) != 1:
        raise Exception; 'Region shape file not found.'
    fpath = os.path.join(polygon_path, file[0])
    gdf = gpd.read_file(fpath)
    gdf = gdf.to_crs(3005)
    return gdf['geometry'].area.values[0] / 1E6


def retrieve_polygon(fname):
    shp_num = fname.split('_')[-1].split('.')[0]
    return (shp_num, gpd.read_file(fname))


def check_polygon_df(region):
    poly_path = os.path.join(DATA_DIR, f'derived_basins/{region}_derived_basin_sample.geojson')
    if os.path.exists(poly_path):
        return True
    else:
        return False


def create_pour_point_gdf(stream, acc, pts, crs, n_chunks=2):
    """Break apart the list of stream pixels to avoid memory 
    allocation issue when indexing large rasters.

    Args:
        stream (_type_): _description_
        confluences (_type_): _description_
        n_chunks (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    
    n_chunks = int(10 * np.log(len(pts)))

    # print(f'{len(pts)} points, breaking into {n_chunks} chunks.')

    conf_chunks = np.array_split(np.asarray(pts), indices_or_sections=n_chunks)

    point_array = []
    acc_array = []

    for chunk in conf_chunks:

        xis = [int(c[0]) for c in chunk]
        yis = [int(c[1]) for c in chunk]
        acc_array += [acc.data[0][c[0], c[1]] for c in chunk]

        ppts = stream[0, xis, yis]
        coords = tuple(map(tuple, zip(ppts.coords['x'].values, ppts.coords['y'].values)))
        point_array += [Point(p) for p in coords]

    df = pd.DataFrame()
    df['num_acc_cells'] = acc_array
    df['pt_id'] = list(range(len(df)))

    gdf = gpd.GeoDataFrame(df, geometry=point_array, crs=crs)
    # print(f'Created dataframe with {len(gdf)} pour points')
    del point_array, df, conf_chunks
    return gdf


def check_for_ppt_batches(batch_folder):    
    if not os.path.exists(batch_folder):
        os.mkdir(batch_folder)
        return False
    existing_batches = os.listdir(batch_folder)
    return len(existing_batches) > 0


def create_ppt_file_batches(df, filesize, temp_ppt_filepath):    

    # divide the dataframe into chunks for batch processing
    # save the pour point dataframe to temporary filRes
    # and limit temporary raster files to ?GB / batch
    # batch_limit = 3E6
    batch_limit = 5E5
    # batch_limit = 5E5
    n_batches = int(filesize * len(df) / batch_limit) + 1
    print(f'        ...running {n_batches} batch(es) on {filesize:.1f}MB raster.')
    
    batch_paths = []
    n = 0
    if len(df) * filesize < batch_limit:
        temp_fpath = temp_ppt_filepath.replace('.shp', f'_{n:04}.shp')
        df.to_file(temp_fpath)
        batch_paths.append(temp_fpath)
        # idx_batches.append(df.index.values)
    else:
        # randomly shuffle the indices 
        # to split into batches
        indices = df.index.values
        random.shuffle(indices)
        batches = np.array_split(np.array(indices), indices_or_sections=n_batches)
        for batch in batches:
            n += 1
            batch_gdf = df.iloc[batch].copy()
            # idx_batches.append(batch)
            temp_fpath = temp_ppt_filepath.replace('.shp', f'_{n:04}.shp')
            batch_gdf.to_file(temp_fpath)
            # keep just the filename
            batch_paths.append(temp_fpath)

    # return list(zip(batch_paths, idx_batches))
    return batch_paths


def batch_basin_delineation(fdir_path, ppt_batch_path, temp_raster_path):

    wbt.unnest_basins(
        fdir_path, 
        ppt_batch_path, 
        temp_raster_path,
        esri_pntr=False, 
        # callback=default_callback
    )


def match_ppt_to_polygons_by_geometry(ppt_batch, polygon_df, resolution, region, temp_folder):
    """
    Start with smallest polygons such that by process of elimination
    we obtain unique polygon-pourpt pairs.
    """
    assert ppt_batch.crs == polygon_df.crs

    for i, polygon in polygon_df.iterrows():

        # check polygon validity
        if not polygon.geometry.is_valid:
            polygon.geometry = make_valid(polygon.geometry)

        t0 = time.time()
        acc = int(polygon.geometry.area / (resolution[0] * resolution[1]))

        # basin = polygon_df.iloc[[i]]
        
        polygon_df.loc[i, 'FLAG_acc_match'] = True
        polygon_df.loc[i, 'FLAG_no_ppt'] = False
        ta = time.time()
        found_pts = ppt_batch[ppt_batch.within(polygon.geometry)].copy()
        tb = time.time()
        # found_pts1 = gpd.sjoin(ppt_df, basin, how='inner', predicate='within')
        # tc = time.time()
        
        if tb - ta > 10:
            print(f'FIND TIME > 10s!! ({tb-ta:.1f}s), len={len(ppt_batch)}')
        # print(f'   time a = {tb-ta:.3f}s tb={tc-tb:.3f}s')
        if len(found_pts) == 0:
            msg_area = polygon_df.loc[i, 'area'] /1E6
            print(f'    NO PPT FOUND IN GEOMETRY for area {msg_area:.1f}!!  i={i}')
            # print('')
            polygon_df.loc[[i]]
            # print('')
            polygon_df.loc[i, 'FLAG_no_ppt'] = True
        else:
            found_pts['acc_diff'] = found_pts['acc'] - acc
            closest_match_idx = found_pts['acc_diff'].abs().idxmin()
            best_pt = found_pts[found_pts.index == closest_match_idx]

            best_pt_cell_idx = best_pt['cell_idx'].values[0]
            if best_pt_cell_idx == None:
                print(best_pt)

            # update the polygon df with ppt index info (xy  coords, cell idx, flag)
            polygon_df.loc[i, 'ppt_x'] = best_pt.geometry.x.values[0]
            polygon_df.loc[i, 'ppt_y'] = best_pt.geometry.y.values[0]
            polygon_df.loc[i, 'cell_idx'] = best_pt_cell_idx
            polygon_df.loc[i, 'FLAG_acc_match'] = False

            a_diff_cells = best_pt['acc_diff'].values[0]
            a_diff_pct = a_diff_cells / best_pt['acc'].values[0]
            polygon_df.loc[i, 'acc_diff_cells'] = a_diff_cells
            if (a_diff_cells > 2) & (a_diff_pct > 0.01):
                # point could possibly be just outside??
                # create a test to find out
                # #     
                print(f'    Area difference {a_diff_cells} > 5 cells & {a_diff_pct:.2f} > 1%')
                busted_pt = polygon_df.loc[[i]].copy()
                idx = busted_pt['cell_idx'].values[0]
                fname = f'{region}_unmatched_{idx}.geojson'
                busted_pt.to_csv(os.path.join(temp_folder, fname))
                polygon_df.loc[i, 'FLAG_acc_match'] = True         
                # raise Exception(f'Area difference {a_diff_cells} > expectation (5 cells) & {a_diff_pct:.2f} > 1%')

    return polygon_df


def clean_up_temp_files(temp_folder, batch_rasters):    
    temp_files = [f for f in os.listdir(temp_folder) if 'temp' in f]
    nalcms_files = [e for e in os.listdir(temp_folder) if e.startswith('NA_NALCMS')]
    raster_clips = [e for e in os.listdir(temp_folder) if DEM_source in e]
    all_files = batch_rasters + nalcms_files + raster_clips + temp_files
    for f in list(set(all_files)):
        os.remove(os.path.join(temp_folder, f))


def raster_to_vector_basins_batch(input):

    raster_fname, raster_crs, resolution, min_area, temp_folder = input
    raster_path = os.path.join(temp_folder, raster_fname)
    raster_no = int(raster_fname.split('_')[-1].split('.')[0])
    polygon_path = os.path.join(temp_folder, f'temp_polygons_{raster_no:05}.shp')
    
    # this function creates rasters of ordered 
    # sets of non-overlapping basins
    wbt.raster_to_vector_polygons(
        raster_path,
        polygon_path,
    )

    gdf = gpd.read_file(polygon_path, crs=raster_crs)

    # simplify the polygon geometry to avoid self-intersecting polygons
    simplify_dim = 0.5 * np.sqrt(resolution[0]**2 + resolution[1]**2)
    simplify_dim = abs(resolution[0])
    buffer_dim = 10
    gdf.geometry = gdf.geometry.buffer(buffer_dim)
    gdf.geometry = gdf.geometry.simplify(simplify_dim)
    gdf.geometry = gdf.geometry.buffer(-1.0 * buffer_dim)
    
    gdf = bpf.filter_and_explode_geoms(gdf, min_area)
    if not (gdf.geometry.geom_type == 'Polygon').all():
        gdf = bpf.filter_and_explode_geoms(gdf, min_area)

    assert (gdf.geometry.geom_type == 'Polygon').all()
    
    return gdf


region_codes = [
    '08P', 
    # '08O',
    # '07U', 
    # '07G', 
    # '07O', 
    # '08G', 
    # '08H', '08E', '08A', '08D', 
    # '09A', '08F', '08B', '08C', 
    # '08N',
    # 'ERockies', 
    # 'Fraser', 
    # 'Peace', 
    # 'Liard', 
    ]

idx_dict = {}
tracker_dict = {}
def main():    
    out_crs = 3005
    min_basin_area = 1.0 # km^2
    rn = 0
    n_processed_basins = 0
    for region in region_codes:
        pass
        # # create raster subsets for extracting raster terrain features
        # rfname = f'{region}_USGS_3DEP_3005.tif'
        # regional_raster_fpath = os.path.join(DEM_folder, rfname)
        # ct0 = time.time()
        
        # temp_polygon_files = [f'temp_polygons_{int(n):05}.shp' for n in range(1, len(batch_rasters) + 1)]
        # output_polygon_paths = [os.path.join(temp_folder, f) for f in temp_polygon_files]         
                    
        # p = mp.Pool()
        # clipping_inputs = [(regional_raster_fpath, opp, temp_folder, region_raster_crs) for opp in output_polygon_paths]
        # clipped_raster_feature_sets = p.map(bpf.dump_poly, clipping_inputs)
        # ct1 = time.time()
        # print(f'    {(ct1-ct0)/60:.1f}min to create {n_polygons} clipped rasters.')
        # p.close()
            
        # extract basin terrain attributes
        # p = mp.Pool()
        # tr0 = time.time()
        # clipped_raster_fnames = [item for sublist in clipped_raster_feature_sets for item in sublist]
        # terrain_data = p.map(bpf.process_basin_terrain_attributes, clipped_raster_fnames)
        # terrain_df = pd.DataFrame.from_dict(terrain_data)
        # terrain_df.set_index('ID', inplace=True)
        # terrain_df.sort_index(inplace=True)
        # tr1 = time.time()
        # print(f'    {(tr1-tr0)/60:.1f}min to process terrain attributes from {n_polygons} clipped rasters.')
        # p.close()
        # tr0 = time.time()
        
        # p = mp.Pool()
        # as_data = p.map(bpf.calc_slope_aspect, clipped_raster_fnames)
        # as_df = pd.DataFrame.from_dict(as_data)
        # as_df.set_index('ID', inplace=True)
        # as_df.sort_index(inplace=True)
        # tr1 = time.time()
        # p.close()
        # print(f'    {(tr1-tr0)/60:.1f}min to process mean aspect and slope attributes from {n_polygons} clipped rasters.')
        
        # process lulc (NALCMS))
        # ct0 = time.time()
        # nalcms_clip_inputs = ((reproj_nalcms_path, opp, temp_folder, nalcms_crs) for opp in output_polygon_paths)
        # p = mp.Pool()
        # clipped_lulc_sets = p.map(bpf.dump_poly, nalcms_clip_inputs)
        # lulc_fnames = (item for sublist in clipped_lulc_sets for item in sublist)
        # lulc_inputs = [f for f in lulc_fnames if not os.path.exists(f.replace('.tif', '.shp'))]
        # p.close()
        # p = mp.Pool()
        # lulc_data = p.map(bpf.process_lulc, lulc_inputs)
        # lulc_df = pd.DataFrame.from_dict(lulc_data)
        # lulc_df.set_index('ID', inplace=True)
                    
        # ct1 = time.time()
        # p.close()
        # print(f'    {(ct1-ct0)/60:.2f}min to create {n_polygons} clipped NALCMS rasters.')

        # bp_inputs = [(i, row, glhymps_fpath) for i, row in batch_polygons.iterrows()]
        # n_processed_basins += n_polygons
        
        # # extract basin polygon geometry attributes 
        # ts = time.time()
        # p = mp.Pool()
        # shape_results = p.map(bpf.process_basin_shape_attributes, bp_inputs)
        # shape_df = pd.DataFrame.from_dict(shape_results)
        # shape_df.set_index('ID', inplace=True)
        # shape_df.sort_index(inplace=True)
        # ts1 = time.time()
        # print(f'    {(ts1-ts):.1f}s to process shape attributes for {len(shape_df)} basins.')
        # p.close()
        # # process soil properties (GLHYMPS)
        # ct0 = time.time()
        # p = mp.Pool()
        # glhymps_data = p.map(bpf.process_glhymps, bp_inputs)
        # glhymps_df = pd.DataFrame.from_dict(glhymps_data)
        # glhymps_df.set_index('ID', inplace=True)
        # p.close()
        # del batch_polygons, bp_inputs
        
        # ct1 = time.time()
        # print(f'    {ct1-ct0:.2f}s to process GLHYMPS attributes.')
        
        # # combine all the attribute results
        # all_data = pd.concat([shape_df, terrain_df, as_df, lulc_df, glhymps_df], join='inner', axis=1)
        # comb_df = gpd.GeoDataFrame(all_data, crs=region_raster_crs)
        # comb_df.drop(labels=['FID'], inplace=True, axis=1)
        # del shape_df, terrain_df, as_df, lulc_df, glhymps_df
        
        # t_end = time.time()
        # unit_time = (t_end - t_batch_start) / n_polygons
        # comb_df.to_file(batch_output_fpath)
        # del comb_df
        # print(f'    ...batch processed in {(t_end-t_batch_start)/60:.1f} min ({unit_time:.2f})s/basin...{region}\n')
                    
            

if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print('')
    print('###################################################')
    print('')
    print(f'Script completed in {t1-t0:.2f}s.')
    print('__________________________________________________')
