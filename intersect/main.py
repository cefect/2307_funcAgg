'''
Created on Jul. 25, 2023

@author: cefect


intersecting building data with hazard rasters
'''
import os
import pandas as pd
import geopandas as gpd
import rasterstats
from rasterstats import zonal_stats

from intersect.osm import retrieve_osm_buildings

from definitions import wrk_dir
from hp import init_log, today_str

#===============================================================================
# file indexers
#===============================================================================
#country tiles
index_country_fp_d = {
    'BGD':'BGD_tindex_0725.gpkg'}

index_country_fp_d = {k:os.path.join(r'l:\10_IO\2307_funcAgg\ins\indexes', v) for k,v in index_country_fp_d.items()}

#hazard tiles
index_hazard_fp_d ={
    '500_fluvial':r'500_fluvial\tileindex_500_fluvial.gpkg'}

index_hazard_fp_d = {k:os.path.join(r'd:\05_DATA\2307_funcAgg\fathom\global3', v) for k,v in index_hazard_fp_d.items()}




def run_samples_on_country(country_key, hazard_key,
                           out_dir=None):
    #===========================================================================
    # defaults
    #===========================================================================
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'samples')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(name=f'samp', fp=os.path.join(out_dir, today_str+'.log'))
    log.info(f'on {country_key} x {hazard_key}')
    #===========================================================================
    # #load tiles
    #===========================================================================
    #country 
    gdf = gpd.read_file(index_country_fp_d[country_key])
    log.info(f'loaded country tiles w/ {len(gdf)}')
    
    #hazard
    haz_tile_gdf = gpd.read_file(index_hazard_fp_d[hazard_key])
    log.info(f'loaded hazard tiles w/ {len(haz_tile_gdf)}')
    
    
    #===========================================================================
    # #loop through each tile 
    #===========================================================================
    for i, row in gdf.to_crs(epsg=4326).iterrows():
        log.info(f'building for polygon %i'%row['id'])
        
        #get the geojson file
        """this retrieves precompiled files if they are available"""
        poly_fp = retrieve_osm_buildings(country_key, row.geometry.bounds)
        
        #retrieve the ratser
        bx = haz_tile_gdf.to_crs(epsg=4326).geometry.intersects(row.geometry.centroid)
        assert bx.sum()==1, f'no intersect'
        rlay_fp = haz_tile_gdf[bx]['location'].values[0]
        assert os.path.exists(rlay_fp)
        
        
        
        #compute hte stats
        #print(f'computing stats on {len(gdf)} feats')
        print(rasterstats.utils.VALID_STATS)
        zs = zonal_stats(poly_fp, rlay_fp, nodata=-32768, stats=['min', 'max', 'mean'], all_touched=False)

        raise IOError('stopped here')
if __name__ == '__main__':
    
    run_samples_on_country('BGD', '500_fluvial')