
import os
import time
import shutil
import json
import requests
import zipfile
import warnings
from shapely import make_valid
warnings.filterwarnings('ignore')

from osgeo import ogr, osr, gdal

# import numpy as np
import numpy as np
import geopandas as gpd
import pandas as pd
import rioxarray as rxr
import multiprocessing as mp
from shapely.geometry import shape
import basin_processing_functions as bpf


import basin_processing_functions as bpf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NALCMS_DIR = os.path.join(BASE_DIR, 'input_data', 'NALCMS/')
DEM_folder = os.path.join(BASE_DIR, 'processed_data/processed_dem/')
region_files = os.listdir(DEM_folder)
region_codes = sorted(list(set([e.split('_')[0] for e in region_files])))

#########################
# input file paths
#########################
# nalcms_dir = os.path.join(BASE_DIR, 'input_data/NALCMS/')
BasinATLAS_dir = os.path.join(BASE_DIR, 'input_data/BasinATLAS')

DEM_source = 'USGS_3DEP'
mosaic_path = os.path.join(BASE_DIR, f'processed_data/{DEM_source}_DEM_mosaic_4269.vrt')
dem_crs, (w_res, h_res) = bpf.get_crs_and_resolution(mosaic_path)

# masks used to clip the geospatial layers
mask_path = os.path.join(BASE_DIR, 'input_data/region_bounds/BCUB_convex_hull.geojson')
reproj_bounds_path_4326 = os.path.join(BASE_DIR, 'input_data/region_bounds/convex_hull_4326.shp')
reproj_bounds_path_4269 = os.path.join(BASE_DIR, 'input_data/region_bounds/convex_hull_4269.shp')
region_mask_path = os.path.join(BASE_DIR, 'input_data/BCUB_regions_merged_R0.geojson')

region_cvx = gpd.read_file(mask_path)
if not os.path.exists(reproj_bounds_path_4269):
    print('     Creating convex hull for the region bounds clipping mask in EPSG:4269 and 4326.')
    mask = region_cvx.copy().to_crs('EPSG:4269')
    mask.to_file(reproj_bounds_path_4269)
    mask = region_cvx.copy().to_crs('EPSG:4326')
    mask.to_file(reproj_bounds_path_4326)

# use the mask geometry to clip the GLHYMPS vector file
hydroLakes_fpath = os.path.join(BASE_DIR, 'input_data/BasinATLAS/HydroLAKES_polys_v10.gdb/HydroLAKES_polys_v10.gdb')
hydroLakes_clipped_fpath = os.path.join(BASE_DIR, 'input_data/BasinATLAS/HydroLAKES_clipped.gpkg')

glhymps_dir = os.path.join(BASE_DIR, 'input_data/GLHYMPS/')
glhymps_fpath = os.path.join(glhymps_dir, 'GLHYMPS.zip')
glhymps_clipped_fpath = os.path.join(glhymps_dir, 'GLHYMPS_clipped.gpkg')
reproj_glhymps_fpath = os.path.join(glhymps_dir, 'GLHYMPS_clipped_3005.geojson')
reproj_mask_glhymps_crs = os.path.join(BASE_DIR, 'input_data/region_bounds/convex_hull_glhymps_crs.shp')
reproj_mask_nalcms_crs = os.path.join(BASE_DIR, 'input_data/region_bounds/convex_hull_nalcms_crs.shp')
ak_waterbodies = os.path.join(BASE_DIR, 'input_data/AK_waterbodies/AK_waterbodies_combined.geojson')

def get_SE_alaska_waterbodies(service_url):
    """
    Query waterbodies in Alaska from the USGS National Map service.

    There is a list of files that were generated from the map downloader tool.

    A polygon was drawn over the intersection of the study region with alaska,
    and the map downloader generates a list of covering tiles.

    Returns:
    dict: JSON response containing waterbodies data.
    """
    files_list = 'se_ak_file_list.csv'
    if not os.path.exists(os.path.join(BASE_DIR, 'input_data/AK_waterbodies')):
        os.makedirs(os.path.join(BASE_DIR, 'input_data/AK_waterbodies'))
    
    files_fpath = os.path.join(BASE_DIR, 'input_data/file_lists', files_list)
    files_df = pd.read_csv(files_fpath)
    
    files_df.columns = list(range(len(files_df.columns)))
    files_df = files_df.drop_duplicates(subset=[14])
    # column 14 contains urls
    urls = files_df[14].values
    fnames = [e.split('/')[-1] for e in urls]
    save_paths = [f'{BASE_DIR}/input_data/AK_waterbodies/{e}' for e in fnames]
    
    dl_files = [e for e in save_paths if not os.path.exists(e)]
    if len(dl_files) > 0:
        print(f'Downloading {len(dl_files)} NHN files.')
        with mp.Pool(8) as pool:
            _ = pool.map(bpf.download_file, list(zip(urls, dl_files)))
    
    file_results = os.listdir(f'{BASE_DIR}/input_data/AK_waterbodies')
    return file_results

def get_NHN_layer(filepath, layer_name='NHDWaterbody'):
    # for info on the NHN feature classes
    # https://www.usgs.gov/ngp-standards-and-specifications/national-hydrography-dataset-nhd-data-dictionary-feature-classes
    ds = ogr.Open(filepath)
    if ds is None:
        return None
    layer = ds.GetLayerByName(layer_name)
    if layer:
        geometries = []
        attributes = []
        # Iterate over features in the layer
        for feature in layer:
            # Convert geometry to Shapely geometry
            geom_json = json.loads(feature.GetGeometryRef().ExportToJson())  # Correct method to get JSON
            geom = shape(geom_json)
            geometries.append(geom)

            # Collect attributes
            attrs = {}
            for field in feature.keys():
                attrs[field] = feature.GetField(field)
            attributes.append(attrs)

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=layer.GetSpatialRef().ExportToWkt())
        return gdf        

# the NHN doesn't cover alaska, so we retrieve water features from the USGS
# if not os.path.exists(ak_waterbodies):
#     # Example usage
#     service_url = 'https://maps.matsugov.us/map/rest/services/OpenData/Environment_WaterbodiesLn_MSB/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json'
#     file_results = get_SE_alaska_waterbodies(service_url)
    
#     if file_results:
#         print("Queries successful:")
#         # unzip all the files to the existing location
#         # for f in file_results:
#         #     # unzip the file
#         #     fpath = os.path.join(BASE_DIR, 'input_data/AK_waterbodies', f)
#         #     with zipfile.ZipFile(fpath, 'r') as zip_ref:
#         #         zip_ref.extractall(os.path.join(BASE_DIR, 'input_data/AK_waterbodies'))
#     else:
#         print("Failed to fetch data.")

#     ak_fpaths = [os.path.join(f'{BASE_DIR}/input_data/AK_waterbodies', e) for e in file_results if e.endswith('.gdb')]

#     layers = []
#     n_layer = 1
#     for f in ak_fpaths:
#         print(f'Processing layer {n_layer}/{len(ak_fpaths)}')
#         layer = get_NHN_layer(f)
#         layers.append(layer)
#         ak_crs = layer.crs
#         n_layer += 1

#     ak_gdf = gpd.GeoDataFrame(pd.concat(layers), crs=ak_crs)
#     # filter out the 'Ice Mass' feature type
#     ak_gdf = ak_gdf[ak_gdf['ftype'] != 378]
#     ak_gdf.to_file(ak_waterbodies)
# else:
#     ak_gdf = gpd.read_file(ak_waterbodies)

# clip and reproject the GLHYMPS vector data. 
if not os.path.exists(reproj_glhymps_fpath):
    # if not os.path.exists(glhymps_fpath):
    #     raise Exception('Download and unzip the GLHYMPS data, see the README for details.')
    layer_name = 'GLHYMPS'
    
    glhymps_original_wkt = """
    PROJCRS["World_Cylindrical_Equal_Area",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["Degree",0.0174532925199433]]],
    CONVERSION["World_Cylindrical_Equal_Area",
        METHOD["Lambert Cylindrical Equal Area",
            ID["EPSG",9835]],
        PARAMETER["Latitude of 1st standard parallel",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8823]],
        PARAMETER["Longitude of natural origin",0,
            ANGLEUNIT["Degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["False easting",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",0,
            LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,
            ORDER[1],
            LENGTHUNIT["metre",1]],
        AXIS["(N)",north,
            ORDER[2],
            LENGTHUNIT["metre",1]],
    USAGE[
        SCOPE["Not known."],
        AREA["World."],
        BBOX[-90,-180,90,180]],
    ID["ESRI",54034]]
    """
    
    # get the mask geometry and reproject it using the original GLHYMPS projection
    region_mask = gpd.read_file(region_mask_path)
    region_mask = region_mask.to_crs(glhymps_original_wkt)
    bbox = tuple(region_mask.total_bounds)

    print('Clipping GLHYMPS vector file to region bounds.')
    glhymps_clipped = gpd.read_file(glhymps_fpath,  bbox=bbox)
    glhymps_reproj = glhymps_clipped.to_crs('EPSG:3005')
    glhymps_reproj.to_file(reproj_glhymps_fpath, driver='GeoJSON')
    print(f'GLHYMPS vector file clipped and reprojected to {reproj_glhymps_fpath}.')
    
# clip and reproject the NALCMS raster data
nalcms_dir = '/home/danbot2/code_5820/large_sample_hydrology/common_data/NALCMS_NA'
for y in [2010, 2015, 2020]:
    if y == 2010:
        fname = 'NA_NALCMS_2010_v2_land_cover_30m.tif'
    elif y == 2015:
        fname = 'NA_NALCMS_landcover_2015v2_30m.tif'
    elif y == 2020:
        fname = 'NA_NALCMS_landcover_2020_30m.tif'
    nalcms_fpath = os.path.join(nalcms_dir, f'land_cover_{y}_30m', fname)

    reproj_nalcms_path = os.path.join(BASE_DIR, f'input_data/NALCMS/NA_NALCMS_landcover_{y}_3005_clipped.tif')
    if not os.path.exists(reproj_nalcms_path):
        if not os.path.exists(nalcms_fpath):
            raise Exception('Download and unzip the NALCMS data, see the README for details.')
        
        nalcms_data = rxr.open_rasterio(nalcms_fpath)
        nalcms_wkt = nalcms_data.rio.crs.wkt
        
        # get the mask geometry and reproject it using the original NALCMS projection
        if not os.path.exists(reproj_mask_nalcms_crs):
            mask = gpd.read_file(mask_path).to_crs(nalcms_wkt)
            mask = mask.convex_hull
            mask.to_file(reproj_mask_nalcms_crs)
        
        # first clip the raster, 
        print('Clipping NALCMS raster to (convex hull) of region bounds.')
        clipped_nalcms_path = os.path.join(BASE_DIR, f'input_data/NALCMS/NA_NALCMS_landcover_{y}_clipped.tif')
        clip_command = f"gdalwarp -s_srs '{nalcms_wkt}' -cutline {reproj_mask_nalcms_crs} -crop_to_cutline -multi -of gtiff {nalcms_fpath} {clipped_nalcms_path} -wo NUM_THREADS=ALL_CPUS"
        if not os.path.exists(clipped_nalcms_path):
            os.system(clip_command)
        
        print('\nReprojecting clipped NALCMS raster.')
        # insert  "-co COMPRESS=LZW" in the command below to use compression (slower but much smaller file size)
        warp_command = f"gdalwarp -q -s_srs '{nalcms_wkt}' -t_srs EPSG:3005 -of gtiff {clipped_nalcms_path} {reproj_nalcms_path} -r bilinear -wo NUM_THREADS=ALL_CPUS"
        if not os.path.exists(reproj_nalcms_path):
            os.system(warp_command) 
        
        # remove the intermediate step
        if os.path.exists(clipped_nalcms_path):
            os.remove(clipped_nalcms_path)

# clip and reproject the HydroLAKES layer
if not os.path.exists(hydroLakes_clipped_fpath):
    mask = gpd.read_file(reproj_bounds_path_4326)
    bounds_string = ' '.join([str(int(e)) for e in mask.bounds.values[0]])
    print('Clipping GLHYMPS vector file to region bounds.')
    command = f'ogr2ogr -nlt PROMOTE_TO_MULTI -f GPKG -clipsrc {bounds_string} {hydroLakes_clipped_fpath} {hydroLakes_fpath}'
    print(command)
    os.system(command)


def polygonize_raster(output_fpath, crs, layer_name, nalcms_path, target_value=0, overlap_size=1, tile_size=500, bbox=(-2351328.0, 2447625.0, -676154.0, 305779.0)):
    print(f'    Processing the NALCMS data to extract vector polygons representing {layer_name} from the NALCMS raster.')
        
    # crop DEM to the Hydrobasins polygons after we add a buffer
    # in order to prevent restricting the basin delineation to the bounds,
    # since in many places the higher resolution 3DEP DEM will     
    # Open the raster
    src_ds = gdal.Open(nalcms_path)
    src_ds = gdal.Translate(f'temp_{layer_name}_file.tif', src_ds, projWin=bbox)
    srcband = src_ds.GetRasterBand(1)
    src_crs = src_ds.GetProjection()

    # Prepare spatial reference from the source projection
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_crs)

    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize
    n_x_tiles = int(np.ceil(x_size / tile_size))
    n_y_tiles = int(np.ceil(y_size / tile_size))

    x_res = src_ds.GetGeoTransform()[1]
    y_res = src_ds.GetGeoTransform()[5]
    diag_res = np.sqrt(x_res**2 + y_res**2)
    
    # Create an output datasource
    drv = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(output_fpath):
        drv.DeleteDataSource(output_fpath)
    dst_ds = drv.CreateDataSource(output_fpath)
    dst_layer = dst_ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPolygon)

    # Add a single ID field
    fd = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("DN")

    n_tile = 0    
    # having to do buffer steps in the subsequent step.
    # Process in tiles to manage memory
    for i in range(0, n_x_tiles * tile_size, tile_size):
        n_tile += 1
        if n_tile % 25 == 0:
            print(f'   ...processing tile {n_tile} of {n_x_tiles}.')
        for j in range(0, n_y_tiles * tile_size, tile_size):
            x_off = i
            y_off = j
            x_size_tile = min(tile_size + overlap_size, x_size - x_off)
            y_size_tile = min(tile_size + overlap_size, y_size - y_off)
            
            # Read raster data as an array
            data = srcband.ReadAsArray(x_off, y_off, x_size_tile, y_size_tile)
            mask = np.where(data == target_value, 1, 0)

             # Create a memory dataset for the mask
            mem_drv = gdal.GetDriverByName('MEM')
            mask_ds = mem_drv.Create('', x_size_tile, y_size_tile, 1, gdal.GDT_Byte)
            mask_ds.SetGeoTransform((src_ds.GetGeoTransform()[0] + x_off * src_ds.GetGeoTransform()[1],
                                      src_ds.GetGeoTransform()[1],
                                      src_ds.GetGeoTransform()[2],
                                      src_ds.GetGeoTransform()[3] + y_off * src_ds.GetGeoTransform()[5],
                                      src_ds.GetGeoTransform()[4],
                                      src_ds.GetGeoTransform()[5]))
            mask_ds.SetProjection(src_ds.GetProjection())
            mask_band = mask_ds.GetRasterBand(1)
            mask_band.WriteArray(mask)

            # Polygonize the mask
            gdal.Polygonize(mask_band, None, dst_layer, dst_field, [], callback=None)

    srcband = None
    src_ds = None
    dst_ds = None

    # DN == 1 is the target geometry value, filter and re-save
    print('   ...filtering and re-saving the geometries.')
    gdf = gpd.read_file(output_fpath)
    gdf = gdf[gdf['DN'] == 1]
    # gdf = gdf.dissolve(by='DN')
    gdf['area'] = gdf.geometry.area / 1e6
    gdf.geometry = gdf.simplify(diag_res/2)
    gdf.geometry = bpf.fill_holes(gdf.geometry)
    
    gdf = gdf[gdf['area'] > 0.1]
    gdf.sort_values('area', ascending=False, inplace=True)
    print(f'   ...{len(gdf)} {layer_name} polygons found.')
    
    gdf = gdf.to_crs(f'EPSG:{crs}')
    print(list(set(gdf['DN'].values)))
    gdf.to_file(output_fpath, driver='ESRI Shapefile')
    print(f'   ...{layer_name} polygons re-saved.')
    # remove the temp raster
    os.remove(f'temp_{layer_name}_file.tif')
    return src_crs

# extract the ocean mask for using to clip the
# DEM and form the coastline of region polygons
nalcms_path = os.path.join(BASE_DIR, 'input_data/NALCMS/NA_NALCMS_landcover_2020_3005_clipped.tif')
nalcms_crs, (w_res, h_res) = bpf.get_crs_and_resolution(nalcms_path)
ocean_polygon_fpath = os.path.join(NALCMS_DIR, f'NALCMS_2020_ocean_polygon_{dem_crs}.shp')
if not os.path.exists(ocean_polygon_fpath):
    nalcms_crs = polygonize_raster(ocean_polygon_fpath, nalcms_crs, 'ocean', nalcms_path,
                      target_value=0, overlap_size=1, tile_size=500,
                      bbox=(-41000.0, 1871000, 1331000, 120000))


ocean_mask = gpd.read_file(ocean_polygon_fpath)
ocean_sindex = ocean_mask.sindex

all_regions = region_cvx.to_crs(nalcms_crs)
sr_bbox = all_regions.total_bounds
# note that the bbox for gdal needs to switch the 
# order of the coordinates to be (xmin, ymax, xmax, ymin)
study_region_bbox = (sr_bbox[0], sr_bbox[3], sr_bbox[2], sr_bbox[1])

# extract perennial ice mass polygons
ice_polygon_fpath = os.path.join(NALCMS_DIR, f'NALCMS_2020_ice_mass_polygons_{dem_crs}.shp')
if not os.path.exists(ice_polygon_fpath):
    nalcms_crs = polygonize_raster(ice_polygon_fpath, dem_crs, 'ice_mask', nalcms_path, 
                      target_value=19, overlap_size=1, tile_size=500,
                      bbox=study_region_bbox)

# extract lakes polygons from the NALCMS raster to use for filtering pour points
lakes_polygon_fpath = os.path.join(NALCMS_DIR, f'NALCMS_2020_lakes_polygons_{dem_crs}.shp')
if not os.path.exists(lakes_polygon_fpath):
    nalcms_crs = polygonize_raster(lakes_polygon_fpath, dem_crs, 'lakes_mask', nalcms_path,
                      target_value=18, overlap_size=1, tile_size=500,
                      bbox=study_region_bbox)


    

     