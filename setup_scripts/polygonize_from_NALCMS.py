from time import time
import os
# os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import dask_geopandas as dgpd
import numpy as np
from multiprocessing import Pool

from osgeo import gdal, ogr, osr

from shapely import Polygon
from basin_processing_functions import get_crs_and_resolution

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NALCMS_DIR = os.path.join(BASE_DIR, 'input_data/', 'NALCMS/')
DATA_DIR = os.path.join(BASE_DIR, 'input_data/BasinATLAS/HydroBASINS/')
id_list_fpath = os.path.join(DATA_DIR, 'HYDRO_basin_ids.csv')
output_folder = os.path.join(BASE_DIR, 'input_data/region_bounds')


DEM_source = 'USGS_3DEP'
mosaic_path = os.path.join(BASE_DIR, f'processed_data/{DEM_source}_DEM_mosaic_4269.vrt')
dem_crs, (w_res, h_res) = get_crs_and_resolution(mosaic_path)

region_df = gpd.read_file(os.path.join(BASE_DIR, 'region_bounds'))


def polygonize_raster(output_fpath, crs, layer_name, target_value=0, overlap_size=1, tile_size=500, bbox=(-2351328.0, 2447625.0, -676154.0, 305779.0)):
    print(f'    Processing the NALCMS data to extract vector polygons representing {layer_name} from the NALCMS raster.')
        
    # we crop each DEM to the Hydrobasins polygons after we add a buffer
    # in order to prevent restricting the basin delineation to the bounds,
    # since in many places the higher resolution 3DEP DEM will 
    nalcms_path = os.path.join(BASE_DIR, 'input_data', 'NALCMS', 'NA_NALCMS_landcover_2020_30m.tif')
    
    # Open the raster
    src_ds = gdal.Open(nalcms_path)
    src_ds = gdal.Translate('temp_file.tif', src_ds, projWin=bbox)
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
        if n_tile % 100 == 0:
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
    gdf['area'] = gdf.geometry.area / 1e6

    gdf['geometry'] = gdf.simplify(diag_res/2)
    
    gdf = gdf[gdf['area'] > 0.1]
    gdf.sort_values('area', ascending=False, inplace=True)
    print(f'   ...{len(gdf)} {layer_name} polygons found.')
    # gdf = gdf.dissolve()
    gdf = gdf.to_crs(f'EPSG:{crs}')
    gdf.to_file(ocean_polygon_fpath, driver='ESRI Shapefile')
    
# extract the ocean mask for using to clip the DEM and form the 
# coastline of region polygons
ocean_polygon_fpath = os.path.join(BASE_DIR, NALCMS_DIR, f'NALCMS_ocean_polygon_{dem_crs}.shp')
if not os.path.exists(ocean_polygon_fpath):    
    polygonize_raster(ocean_polygon_fpath, dem_crs, 'ocean_mask', 
                      target_value=0, overlap_size=1, tile_size=500,
                      bbox=(-2351328.0, 2447625.0, -676154.0, 305779.0))

# extract perennial ice mass polygons 
lakes_polygon_fpath = os.path.join(BASE_DIR, NALCMS_DIR, f'NALCMS_ice_mass_polygons_{dem_crs}.shp')
if not os.path.exists(ocean_polygon_fpath):    
    polygonize_raster(ocean_polygon_fpath, dem_crs, 'ocean_mask', 
                      target_value=19, overlap_size=1, tile_size=500,
                      bbox=)
    
# extract lakes polygons from the NALCMS raster to use for filtering pour points
lakes_polygon_fpath = os.path.join(BASE_DIR, NALCMS_DIR, f'NALCMS_lakes_polygons_{dem_crs}.shp')
if not os.path.exists(ocean_polygon_fpath):    
    polygonize_raster(ocean_polygon_fpath, dem_crs, 'ocean_mask', 
                      target_value=18, overlap_size=1, tile_size=500,
                      bbox=)
    
