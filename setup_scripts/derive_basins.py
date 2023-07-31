# generate basins


import os
import time
# import glob
# import json


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import geopandas as gpd
import pandas as pd
import rioxarray as rxr

import random

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

DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')

glhymps_fpath = os.path.join(SOURCE_DATA_DIR, 'GLHYMPS/GLHYMPS_clipped_4326.gpkg')
# import NALCMS raster
# land use / land cover
nalcms_fpath = os.path.join(SOURCE_DATA_DIR, 'NALCMS_NA/NA_NALCMS_2010_v2_land_cover_30m/NA_NALCMS_2010_v2_land_cover_30m.tif')
reproj_nalcms_path = os.path.join(SOURCE_DATA_DIR, 'NALCMS_NA/NA_NALCMS_2010_4326.tif')
if not os.path.exists(reproj_nalcms_path):
    nalcms, nalcms_crs, nalcms_affine = bpf.retrieve_raster(nalcms_fpath)
    print(f'   ...NALCMS imported, crs = {nalcms.rio.crs.to_epsg()}')
    print('Reproject NALCMS raster')

    warp_command = f'gdalwarp -q -s_srs "{nalcms.rio.crs.to_proj4()}" -t_srs EPSG:3005 -of gtiff {nalcms_fpath} {reproj_nalcms_path} -r bilinear -wo NUM_THREADS=ALL_CPUS'    
    os.system(warp_command)
    

DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')

def retrieve_raster(region):
    filename = f'{region}_USGS_3DEP_3005_res1.tif'
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
    acc_array, id_array = [], []

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
    batch_limit = 1E5
    n_batches = int(filesize * len(df) / batch_limit) + 1
    print(f'        ...running {n_batches} batch(es) on {filesize:.1f}MB raster.')
    
    # batch_paths, idx_batches = [], []
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


def raster_to_vector_basins_multi(temp_folder, raster_batch_file, output_polygon_path):
    raster_batch_path = os.path.join(temp_folder, raster_batch_file)
    # this function creates rasters of ordered sets of 
    # non-overlapping basins
    wbt.raster_to_vector_polygons(
        raster_batch_path,
        output_polygon_path,
    )


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

    polygon_df.reset_index(inplace=True, drop=True)

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
        
    gdf.drop(labels=['area'], inplace=True, axis=1)
    gdf.to_file(polygon_path.replace('.shp', '.geojson'))
    
    return gdf


region_codes = [
    '08P', 
    # '08O', 
    # '07G','07U',  
    # '07O', 
    # '08G', 
    # '08H', '08E', '08A', '08D', # done
    # '09A', '08F', '08B', '08C', # done
    # '08N',
    # 'ERockies', 
    # 'Peace',  
    # 'Fraser', 
    # 'Liard', 
    ]

# use this boolean dict to track selection of pour
# points and avoid duplicate basin delineations
combo_df = pd.DataFrame()

idx_dict = {}
tracker_dict = {}
def main():    

    min_basin_area = 1.0 # km^2
    rn = 0
    n_processed_basins = 0
    for region in region_codes:
        t0 = time.time()
        rn += 1
        idx_dict[region] = {}
        tracker_dict[region] = []

        fdir_file = f'{region}_{DEM_source}_3005_fdir.tif'
        fdir_path = os.path.join(DEM_folder, fdir_file)

        # get the file size in MB
        filesize = (os.path.getsize(fdir_path) >> 20 )
        
        # set up a folder for temporary files
        
        temp_folder = os.path.join(DATA_DIR, f'derived_basins/{region}/temp/')        
        ppt_folder = os.path.join(DATA_DIR, f'pour_points/{region}/')
        output_polygon_folder = os.path.join(DATA_DIR, f'derived_basins/{region}')

        for f in [temp_folder, output_polygon_folder]:
            if not os.path.exists(f):
                os.makedirs(f)

        print(f'Processing {region} ({rn}/{len(region_codes)})')

        region_raster, region_raster_crs, _ = retrieve_raster(region)
        raster_resolution = tuple(abs(e) for e in region_raster.rio.resolution())
        
        print(f'    {region} raster crs = {region_raster_crs}.')
          
        # Break the pour points into unique sets
        # to avoid duplicate delineations
        ppt_file = os.path.join(ppt_folder, f'{region}_pour_pts_filtered.geojson')
        
        ppt_gdf = gpd.read_file(ppt_file)
        
        ppt_crs = ppt_gdf.crs
        print(f'    ppt crs = {ppt_crs}.')

        # check for an output file to allow for partial updating
        # by tracking (unique) ppt indices
        output_folder = os.path.join(DATA_DIR, f'derived_basins/{region}/')
        output_fpath = os.path.join(output_folder, f'{region}_processed_attributes.geojson')
        if os.path.exists(output_fpath):
            print('Output file exists, skipping.')
            continue

        # check if the ppt batches have already been created
        batch_folder = os.path.join(temp_folder, 'ppt_batches/')
        batches_exist = check_for_ppt_batches(batch_folder)

        # break the pour point dataframe into batches to limit the RAM used in 
        # processing basins for each pour point.
        temp_ppt_filepath = os.path.join(batch_folder, f'temp_ppts.shp')

        if not batches_exist:
            print(f'    Generating new batch paths.')
            batch_ppt_paths = create_ppt_file_batches(ppt_gdf, filesize, temp_ppt_filepath)
        else:
            print(f'    Retrieving existing batch paths.')
            batch_ppt_files = list(sorted([f for f in os.listdir(batch_folder) if f.endswith('.shp')]))
            batch_ppt_paths = [os.path.join(batch_folder, f) for f in batch_ppt_files]

        n_batch_files = len(batch_ppt_paths)

        batch_output_files = sorted([f for f in os.listdir(output_folder) if f.startswith(output_fpath.split('/')[-1].split('.')[0])])
        if len(batch_output_files) > 0:
            batch_matches = [b.replace('.geojson', '.shp').split('_')[-1] for b in batch_output_files]
            batch_ppt_paths = list(sorted([f for f in batch_ppt_paths if f.split('_')[-1] not in batch_matches]))

        for ppt_batch_path in batch_ppt_paths:
            t_batch_start = time.time()
            batch_no = int(ppt_batch_path.split('_')[-1].split('.')[0])
            print(f'  Starting processing {region} batch {batch_no}/{n_batch_files}')
            temp_fname = f'{region}_temp_raster.tif'
            temp_basin_raster_path = os.path.join(temp_folder, temp_fname)            
            batch_output_fpath = output_fpath.replace('.geojson', f'_{batch_no:04d}.geojson')
            
            # creates the minimum set of rasters with non-overlapping polygons
            batch_basin_delineation(fdir_path, ppt_batch_path, temp_basin_raster_path)
            batch_rasters = sorted([e for e in sorted(os.listdir(temp_folder)) if e.endswith('.tif')])

            # retrieve the batch pour point dataframe
            ppt_batch = gpd.read_file(ppt_batch_path)

            tb1 = time.time()
      
            print(f'    Basins delineated for {region} ppt batch {batch_no} in {(tb1-t_batch_start)/60:.1f}min. {len(batch_rasters)} raster sub-batches to process ({len(ppt_batch)}).')

            # process the raster batches in parallel
            crs_array = [f'EPSG:{region_raster_crs}'] * len(batch_rasters)
            # min_A_array = [min_basin_area] * len(batch_raster_fpaths)
            resolution_array = [raster_resolution] * len(batch_rasters)
            temp_folder_array = [temp_folder] * len(batch_rasters)
            min_areas = [min_basin_area] * len(batch_rasters)
            path_inputs = list(zip(batch_rasters, crs_array, resolution_array, min_areas, temp_folder_array))
            
            # Adjust the number of processes to your own system specs
            batch_size_GB = len(batch_rasters) * filesize / 1E3
            if batch_size_GB < 15:
                n_procs = 28
            elif batch_size_GB < 20:
                n_procs = 22
            elif batch_size_GB < 25:
                n_procs = 18
            else:
                n_procs = 12
            print(f'    Setting n_procs={n_procs:.0f} For a batch size of {batch_size_GB:.1f}GB')

            p = mp.Pool(n_procs)
            trv0 = time.time()
            all_polygons = p.map(raster_to_vector_basins_batch, path_inputs)
            trv1 = time.time()
            print(f'    {trv1-trv0:.2f}s to convert {len(path_inputs)} rasters to vector basins.')
            
            # concatenate polygons
            trc0 = time.time()
            batch_polygons = gpd.GeoDataFrame(pd.concat(all_polygons), crs=region_raster_crs)
            # sort by VALUE to maintain ordering from ppt batching / raster basin unnesting operation
            batch_polygons.sort_values(by='VALUE', inplace=True)
            batch_polygons.reset_index(inplace=True, drop=True)
            trc1 = time.time()
            print(f'    {trc1-trc0:.2f}s to concatenate basin vectors.')
                                              
            # create raster subsets for extracting raster terrain features
            rfname = f'{region}_USGS_3DEP_3005_res1.tif'
            regional_raster_fpath = os.path.join(DEM_folder, rfname)
            ct0 = time.time()
            
            temp_polygon_files = [f'temp_polygons_{int(n):05}.geojson' for n in range(1, len(batch_rasters) + 1)]
            output_polygon_paths = [os.path.join(temp_folder, f) for f in temp_polygon_files]
            
            # create raster clips for each polygon
            p = mp.Pool()
            clipping_inputs = [(regional_raster_fpath, opp, temp_folder, region_raster_crs) for opp in output_polygon_paths]
            clipped_raster_feature_sets = p.map(bpf.dump_poly, clipping_inputs)
            clipped_raster_fnames = [item for sublist in clipped_raster_feature_sets for item in sublist]
            ct1 = time.time()
            print(f'    {ct1-ct0:.1f}s to create {len(clipped_raster_fnames)} clipped rasters.')
            
            # extract basin terrain attributes
            p = mp.Pool()
            tr0 = time.time()
            terrain_data = p.map(bpf.process_basin_terrain_attributes, clipped_raster_fnames)
            terrain_df = pd.DataFrame.from_dict(terrain_data)
            terrain_df.set_index('ID', inplace=True)
            terrain_df.sort_index(inplace=True)
            tr1 = time.time()
            print(f'    {(tr1-tr0)/60:.1f}min to process terrain attributes from {len(clipped_raster_fnames)} clipped rasters.')

            tr0 = time.time()
            
            p = mp.Pool()
            as_data = p.map(bpf.calc_slope_aspect, clipped_raster_fnames)
            as_df = pd.DataFrame.from_dict(as_data)
            as_df.set_index('ID', inplace=True)
            as_df.sort_index(inplace=True)
            tr1 = time.time()
            print(f'    {(tr1-tr0)/60:.1f}min to process mean aspect and slope attributes from {len(clipped_raster_fnames)} clipped rasters.')
            
            
            # process lulc (NALCMS))
            ct0 = time.time()
            reproj_polygon_paths = [e.replace('.shp', '_4326.geojson') for e in output_polygon_paths]
            nalcms_clip_inputs = [(nalcms_fpath, opp, temp_folder, 4326) for opp in reproj_polygon_paths] 
            p = mp.Pool()
            clipped_lulc_sets = p.map(bpf.dump_poly, nalcms_clip_inputs)
            clipped_lulc_fnames = [item for sublist in clipped_lulc_sets for item in sublist]
            
            p = mp.Pool()
            lulc_data = p.map(bpf.process_lulc, clipped_lulc_fnames)
            lulc_df = pd.DataFrame.from_dict(lulc_data)
            lulc_df.set_index('ID', inplace=True)
                        
            ct1 = time.time()
            p = None
            print(f'    {(ct1-ct0)/60:.2f}min to create {len(clipped_lulc_fnames)} clipped NALCMS rasters.')
            
            # if ordering is correct, we don't have to do this step
            # we can instead just iterate over polygons, derive attributes,
            # assign values to ppt dataframe.
            # add the pour point location info to each polygon
            batch_polygons = bpf.match_ppt_to_polygons_by_order(ppt_batch, batch_polygons, raster_resolution)           
            bp_inputs = [(i, row) for i, row in batch_polygons.iterrows()]  
            n_processed_basins += len(batch_polygons)
            
            # extract basin polygon geometry attributes 
            ts = time.time()           
            p = mp.Pool()
            shape_results = p.map(bpf.process_basin_shape_attributes, bp_inputs)
            shape_df = pd.DataFrame.from_dict(shape_results)
            shape_df.set_index('ID', inplace=True)
            shape_df.sort_index(inplace=True)
            ts1 = time.time()
            print(f'    {(ts1-ts):.1f}s to process {len(shape_df)} shape attributes.')
            
            # process soil properties (GLHYMPS)
            ct0 = time.time()
            p = mp.Pool()
            glhymps_data = p.map(bpf.process_glhymps, bp_inputs)
            glhymps_df = pd.DataFrame.from_dict(glhymps_data)
            glhymps_df.set_index('ID', inplace=True)
            
            ct1 = time.time()
            print(f'    {ct1-ct0:.2f}s to process GLHYMPS attributes.')
            
            # combine all the attribute results
            all_data = pd.concat([shape_df, terrain_df, as_df, lulc_df, glhymps_df], join='inner', axis=1)
            comb_df = gpd.GeoDataFrame(all_data, crs=region_raster_crs)
            comb_df.drop(labels=['FID'], inplace=True, axis=1)
            
            t_end = time.time()
            unit_time = (t_end - t_batch_start) / (len(batch_polygons))
            comb_df.to_file(batch_output_fpath)
            print(f'    ...batch processed in {(t_end-t_batch_start)/60:.1f} min ({unit_time:.2f})s/basin...{region}\n')
            
            batch_output_files.append(batch_output_fpath)
            
            # remove all temporary files (don't delete the raster batches yet!)
            clean_up_temp_files(temp_folder, batch_rasters)
            
            
        
        bpf.merge_geojson_files(batch_output_files, output_fpath, output_polygon_folder)
        t_n = time.time()
        n_processed_basins = max(1, n_processed_basins)
        ut = (t_n - t0) / n_processed_basins
        print(f'Total processing time for {region}: {t_n-t0:.1f}s ({ut:.2f}/basin).')

        print('')
        print('      ------------------------------------')
        print('')
            
            

if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print('')
    print('###################################################')
    print('')
    print(f'Script completed in {t1-t0:.2f}s.')
    print('__________________________________________________')