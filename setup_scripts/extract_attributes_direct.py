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

##################################################
# update file paths to geospatial data sources
##################################################
nalcms_dir = os.path.join(BASE_DIR, 'input_data/NALCMS/')
glhymps_dir = os.path.join(BASE_DIR, 'input_data/GLHYMPS/')
nalcms_fpath = os.path.join(nalcms_dir, 'NA_NALCMS_landcover_2010_3005_clipped.tif')
glhymps_fpath = os.path.join(glhymps_dir, 'GLHYMPS_clipped_3005.gpkg')


# if not os.path.exists(reproj_nalcms_path):
#     nalcms, nalcms_crs, nalcms_affine = bpf.retrieve_raster(nalcms_fpath)
#     raster_wkt = nalcms_crs.to_wkt()
        
#     reproj_bounds = gpd.read_file(mask_path).to_crs(nalcms_crs)
#     reproj_bounds_path = os.path.join(BASE_DIR, 'input_data/region_polygons/region_bounds_reproj.shp')
#     reproj_bounds.to_file(reproj_bounds_path)
    
#     # first clip the raster, then reproject to EPSG 3005
#     print('Clipping NALCMS raster to region bounds.')
#     clipped_nalcms_path = os.path.join(BASE_DIR, 'input_data/NALCMS/NA_NALCMS_landcover_2010_clipped.tif')
#     clip_command = f"gdalwarp -s_srs '{raster_wkt}' -cutline {reproj_bounds_path} -crop_to_cutline -multi -of gtiff {nalcms_fpath} {clipped_nalcms_path} -wo NUM_THREADS=ALL_CPUS"
#     os.system(clip_command)
    
#     print('\nReprojecting clipped NALCMS raster.')
#     warp_command = f"gdalwarp -q -s_srs '{raster_wkt}' -t_srs EPSG:3005 -of gtiff {clipped_nalcms_path} {reproj_nalcms_path} -r bilinear -wo NUM_THREADS=ALL_CPUS"
#     os.system(warp_command) 
    
#     # remove the intermediate step
#     if os.path.exists(clipped_nalcms_path):
#         os.remove(clipped_nalcms_path)
        

nalcms, nalcms_crs, nalcms_affine = bpf.retrieve_raster(nalcms_fpath)


DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')

def retrieve_dem_raster(region):
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


def clean_up_temp_files(temp_folder):    
    temp_files = [f for f in os.listdir(temp_folder) if 'temp' in f]
    nalcms_files = [e for e in os.listdir(temp_folder) if e.startswith('NA_NALCMS')]
    raster_clips = [e for e in os.listdir(temp_folder) if DEM_source in e]
    all_files = nalcms_files + raster_clips + temp_files
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


def create_batches(basin_data, region):
    filename = f'{region}_USGS_3DEP_3005.tif'
    fpath = os.path.join(DEM_folder, filename)
    # convert the file size from bytes to MB by shifting 20 positions
    filesize = (os.path.getsize(fpath) >> 20 )
    
    batch_limit = 20
    n_batches = int(np.ceil(filesize / batch_limit))
    
    return np.array_split(basin_data['ID'].values, indices_or_sections=n_batches)
    
    
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
    # out_crs = 3005
    # min_basin_area = 1.0 # km^2
    t_batch_start = time.time()
    for region in region_codes:
        temp_folder = os.path.join(BASE_DIR, f'processed_data/derived_basins/{region}/temp/')
        region_raster, region_raster_crs, _ = retrieve_dem_raster(region)
        
        raster_filename = f'{region}_USGS_3DEP_3005.tif'
        raster_fpath = os.path.join(DEM_folder, raster_filename)
        temp_folder = os.path.join(DATA_DIR, f'derived_basins/{region}/temp/')        
        if not os.path.exists(os.path.join(temp_folder)):
            os.mkdir(os.path.join(temp_folder))
    
        batch_output_folder = os.path.join(DATA_DIR, f'basin_attributes/')
        if not os.path.exists(batch_output_folder):
            os.mkdir(batch_output_folder)
            
        output_fpath = os.path.join(batch_output_folder, f'{region}_basin_attributes.geojson')
        output_batch_paths = []
        skip_save = False
        if os.path.exists(output_fpath):
            skip_save = True
            continue
        
        basin_data = gpd.read_parquet(os.path.join(DATA_DIR, f'derived_basins/{region}/{region}_basins.parquet'))
        batches = create_batches(basin_data, region)
        batch_no = 0
        
        for batch_idxs in batches:
            basins = basin_data[basin_data['ID'].isin(batch_idxs)].copy()
            batch_output_fpath = os.path.join(batch_output_folder, f'{region}_attributes_batch_{batch_no:04}.geojson')
            
            ct0 = time.time()
            # open the parquet file containing basin geometry
            # all geometries should be in EPSG 3005 CRS
            basin_polygons = basins.set_geometry('basin_geometry')[['ID', 'basin_geometry']].copy()
            basin_polygons = basin_polygons.to_crs(region_raster_crs)

            polygon_inputs = [(region, 'dem', i, row, region_raster_crs, raster_fpath, temp_folder) for i, row in basin_polygons.iterrows()]
            p = mp.Pool()
            temp_raster_paths = p.map(bpf.dump_poly, polygon_inputs)
            ct1 = time.time()
            print(f'    {(ct1-ct0)/60:.1f}min to create {len(polygon_inputs)} clipped rasters.')
            p.close()
                                
            # extract basin terrain attributes
            p = mp.Pool()
            tr0 = time.time()
            terrain_data = p.map(bpf.process_terrain_attributes, temp_raster_paths)
            terrain_df = pd.DataFrame.from_dict(terrain_data)
            terrain_df.set_index('ID', inplace=True)
            terrain_df.sort_index(inplace=True)
            tr1 = time.time()
            print(f'    {(tr1-tr0)/60:.1f}min to process terrain attributes from {len(temp_raster_paths)} clipped rasters.')
            p.close()        
            
            # process lulc (NALCMS))
            ct0 = time.time()
            basin_polygons = basin_polygons.to_crs(nalcms_crs)

            nalcms_clip_inputs = [(row, nalcms_fpath, nalcms_crs, temp_folder) for i, row in basin_polygons.iterrows()]
            p = mp.Pool()
            lulc_data = p.map(bpf.process_lulc, nalcms_clip_inputs)
            lulc_df = pd.DataFrame.from_dict(lulc_data)
            lulc_df.set_index('ID', inplace=True)
            ct1 = time.time()
            p.close()
            print(f'    {(ct1-ct0)/60:.2f}min to create {len(lulc_df)} clipped NALCMS rasters.')

            # process soil properties (GLHYMPS)
            # glhymps is in 4326, so we need to reproject the basin polygons
            basin_polygons = basin_polygons.to_crs(4326)
            bp_inputs = [(row, 4326, glhymps_fpath) for i, row in basin_polygons.iterrows()]
            ct0 = time.time()
            p = mp.Pool()
            glhymps_data = p.map(bpf.process_glhymps, bp_inputs)
            glhymps_df = pd.DataFrame.from_dict(glhymps_data)
            glhymps_df.set_index('ID', inplace=True)
            p.close()
            
            ct1 = time.time()
            print(f'    {ct1-ct0:.2f}s to process GLHYMPS attributes.')
            
            # combine all the attribute results
            # we have to drop the geometry columns except for the pour point.
            # we want to keep the centroid geometry in the final output 
            # so split the centroid coordinates into x and y components
            out_data = basin_data.copy()
            out_data['centroid_x'] = basin_data['centroid_geometry'].x
            out_data['centroid_y'] = basin_data['centroid_geometry'].y
            out_data.drop(labels=['basin_geometry', 'centroid_geometry'], axis=1, inplace=True)
            out_data.set_index('ID', inplace=True)
            
            all_data = pd.concat([out_data, terrain_df, lulc_df, glhymps_df], join='inner', axis=1)
            all_data.drop(labels=['FID'], inplace=True, axis=1)        
            comb_df = gpd.GeoDataFrame(all_data, geometry='geometry', crs=3005)
            t_end = time.time()
            unit_time = (t_end - t_batch_start) / len(all_data)
            
            # save the results
            comb_df.to_file(batch_output_fpath)
            output_batch_paths.append(batch_output_fpath)
            del terrain_df, lulc_df, glhymps_df
            del comb_df
            print(f'    ...batch processed in {(t_end-t_batch_start)/60:.1f} min ({unit_time:.2f})s/basin...{region}\n')
            
            clean_up_temp_files(temp_folder)
    
    # merge batch outputs
    
    if not skip_save:
        results = []
        for fpath in output_batch_paths:
            results.append(gpd.read_file(fpath))
        results = pd.concat(results, axis=0)
        results_gdf = gpd.GeoDataFrame(results, geometry='geometry', crs=3005)
        results_gdf.to_file(output_fpath, driver='GeoJSON')
        for fpath in output_batch_paths:
            if os.path.exists(fpath):
                os.remove(fpath)
    else:
        foo = gpd.read_file(output_fpath)
        print(foo.head())
        
if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()
    print('')
    print('###################################################')
    print('')
    print(f'Script completed in {t1-t0:.2f}s.')
    print('__________________________________________________')
