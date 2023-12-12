import os
import time 

import pandas as pd
import geopandas as gpd
import numpy as np

# import xarray as xr

from shapely.geometry import Point

from multiprocessing import Pool

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import hysets dataset attributes
HYSETS_DIR = '/home/danbot2/code_5820/large_sample_hydrology/common_data/HYSETS_data/'
hysets_df = pd.read_csv(os.path.join(HYSETS_DIR, 'HYSETS_watershed_properties.txt'), sep=';')

# import hysets dataset basin geometries
hysets_basins_path = os.path.join(HYSETS_DIR, 'HYSETS_watershed_boundaries/HYSETS_watershed_boundaries_20200730.shp')
hs_basins = gpd.read_file(hysets_basins_path)
hs_basins = hs_basins.set_crs(4326)

# compare HYSETS basin polygons to the (newer) July 2022 version of the WSC basin polygons
# https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/HydrometricNetworkBasinPolygons/
wsc_basins_path = '/home/danbot2/code_5820/large_sample_hydrology/common_data/WSC_data/'

# find stations in common between HYSETS and updated WSC
common_stns = []
for i, row in hs_basins[hs_basins['Source'] == 'HYDAT'].iterrows():
    stn = row['OfficialID']
    stn_prefix = stn[:2]
    basin_path = os.path.join(wsc_basins_path, f'{stn_prefix}/{stn}/{stn}_DrainageBasin_BassinDeDrainage.shp')
    if os.path.exists(basin_path):
        common_stns.append(stn)
        if len(common_stns) % 100 == 0:
            print(f'{len(common_stns)} basins in common found.')


def calculate_gravelius_and_perimeter(polygon):
    
    perimeter = polygon.geometry.length.values[0]
    area = polygon.geometry.area.values[0]
    if area == 0:
        return np.nan, perimeter
    else:
        gravelius = perimeter / np.sqrt(4 * np.pi * area)
    
    return gravelius, perimeter / 1000


def check_accuracy(stn, hs_polygon, wsc_polygon):
    """
    Computes the accuracy index of the intersection of two polygons.
    The accuracy index is the ratio of the intersection area to the union area.
    If the polygon match is >= 90%, the polygon is considered a match.
    """
    hs = hs_polygon.geometry.values[0]
    wsc = wsc_polygon.geometry.values[0]
    intersection = hs.intersection(wsc).area
    union = hs.union(wsc).area

    if union == 0:
        return False, 0
    
    accuracy_index = intersection / union
    print(f'    {stn} accuracy index: {accuracy_index:.2f}')
    return accuracy_index >= 0.9, accuracy_index
    

def check_divergence(polygon, baseline_perimeter):
    """
    Checks if the perimeter of a polygon is within 1% of the baseline perimeter.
    """
    diff = polygon.geometry.length.values[0] / 1E3 - baseline_perimeter
    return (np.abs(diff) <= 0.01 * baseline_perimeter) | (diff < 0)


def simplify_polygon(stn, polygon, baseline_perimeter, output_fpath=None):
    # Simplifies a polygon geometry by progressively increasing its tolerance.
    # The tolerance is increased until the perimeter of the simplified polygon
    # is within 1% of the baseline (published) perimeter.
    # http://shapely.readthedocs.io/en/latest/manual.html#object.simplify 

    tolerance = 0 #initial smoothing tolerance (meters)
    if output_fpath is not None:
        if not os.path.exists(output_fpath):
            os.makedirs(output_fpath)
    
    tolerance, area, perimeter = tolerance, polygon.area.values[0] / 1E6, polygon.length.values[0] / 1E3
    n = 0
    increment = 5  # simplification tolerance increment (meters)
    while check_divergence(polygon, baseline_perimeter) == False:
        tolerance += increment
        polygon = polygon.simplify(tolerance)
        area = polygon.area.values[0] / 1e6
        perimeter = polygon.length.values[0] / 1e3
        if output_fpath is not None:
            polygon.to_file(os.path.join(output_fpath, f'{stn}_{tolerance}.geojson'))
        n += 1
                
        if n == 10:
            increment *= 2
            # print(f'   ...increment increased to {increment} at n={n}.')
        elif n == 20:
            increment *= 5 
            # print(f'   ...increment increased to {increment} at n={n}.')
        elif n == 50:
            increment *= 2
            # print(f'   ...increment increased to {increment} at n={n}.')
        if n == 100:
            diff = (perimeter - baseline_perimeter)
            print(f'    ...n={n}: iteration limit exceeded. {diff:.1f} length difference remaining.')
            return tolerance, area, perimeter, True
    
    
    return tolerance, area, perimeter, False

    
def process_basin(basin_id):

    hs_polygon = hs_basins[hs_basins['OfficialID'] == basin_id].copy().to_crs(3005)
    stn = hs_polygon['OfficialID'].values[0]
    wsc_basin_fpath = os.path.join(wsc_basins_path, f'{stn[:2]}/{stn}/{stn}_DrainageBasin_BassinDeDrainage.shp')
    updated_wsc_polygon = gpd.read_file(wsc_basin_fpath)
    
    # reproject the hysets polygon to the projection of the updated wsc polygon
    # wsc polygons are equal area conic projections, so we need to reproject the hysets polygon
    # to preserve area for comparison
    wsc_crs = updated_wsc_polygon.crs
    hs_polygon = hs_polygon.to_crs(wsc_crs)

    assert hs_polygon.crs == updated_wsc_polygon.crs
    polygon_match, accuracy = check_accuracy(basin_id, hs_polygon, updated_wsc_polygon)

    if polygon_match:
        hysets_info = hysets_df[hysets_df['Official_ID'] == basin_id].copy()
        hs_area = hysets_info['Drainage_Area_km2'].values[0]
                      
        output_fpath = os.path.join(BASE_DIR, f'processed_data/simplified_polygons/{basin_id}')
    
        hs_perim = hysets_info['Perimeter'].values[0]                
        val_perim = updated_wsc_polygon.geometry.length.values[0] / 1E3
        
        tolerances, smoothed_area, perimeter, flag = simplify_polygon(basin_id, updated_wsc_polygon, hs_perim, output_fpath=None)
        
        results = {
            'Official_ID': str(basin_id),
            'HYSETS_perimeter': hs_perim,
            'HYSETS_area_km2': hs_area,
            'tolerances': tolerances,
            'original_area': updated_wsc_polygon.geometry.area.values[0] / 1E6, # updated WSC polygon area, convert to km2    
            'simplified_area': smoothed_area,  # area after perimeter simplified
            'updated_perimeter': val_perim, # the updated WSC polygon perimeter
            'smoothed_perimeter': perimeter,# the updated WSC polygon perimeter after simplification
            'flag': flag,
            'accuracy': accuracy,
        }
        return results        
    else:
        return {'Official_ID': str(basin_id), 'accuracy': accuracy}
        

which_set = 'updated_wsc'


results_file = os.path.join(BASE_DIR, 'validation/polygon_simplified_results.csv')

# if os.path.exists(results_file):
#     existing_stns = pd.read_csv(results_file)['Official_ID'].values


chunks = np.array_split(common_stns, 250)

i = 0
for chunk in chunks:
    i += 1
    p = Pool()
    print(f'Starting batch {i}/{len(chunks)} ({len(chunk)} basins).')
    t0 = time.time()
    # for c in chunk:
    #     rr = process_basin(c)
    results = p.map(process_basin, chunk)
    # results = [e for e in results if e is not None]
    t1 = time.time()
    tt = t1 - t0
    ut = tt / len(chunk)
    print(f'    ...batch {i} completed in {tt:.1f}s. ({ut:.2f}/basin)')
    new_results_df = pd.DataFrame.from_dict(results, orient='columns')
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        results_df = pd.concat([existing_results, new_results_df])
    else:
        results_df = new_results_df.copy()

    # drop rows with any empty cell
    # foo = results_df[results_df.isnull().any(axis=1)].copy()
    n_mismatch = len(results_df[results_df['accuracy'] < 0.9].copy())
    print(f'final sample size = {len(results_df)}, {n_mismatch} basins with accuracy < 90%.')

    results_df.to_csv(results_file, index=False)



