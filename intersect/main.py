'''
Created on Jul. 25, 2023

@author: cefect


intersecting building data with hazard rasters
'''
import os, hashlib
print('\n'.join(os.environ['PATH'].split(';')))
import psutil
from datetime import datetime
import pandas as pd
import fiona
import geopandas as gpd
import rasterstats
from rasterstats import zonal_stats

from intersect.osm import retrieve_osm_buildings

from definitions import wrk_dir, lib_dir
from definitions import temp_dir as temp_dirM
 
from hp import (
    init_log, today_str, get_log_stream, get_raster_point_samples, get_directory_size,
    dstr
    )



#===============================================================================
# file indexers
#===============================================================================
#country tiles
index_country_fp_d = {
    'BGD':'BGD_tindex_0725.gpkg',
    'AUS':'AUS_tindex_0824.gpkg',
    'ZAF':'ZAF_tindex_0824.gpkg',
    'BRA':'BRA_tindex_0824.gpkg',
    'CAN':'CAN_tindex_0824.gpkg',
    'DEU':'DEU_tindex_0824.gpkg',    
    }

index_country_fp_d = {k:os.path.join(r'l:\10_IO\2307_funcAgg\ins\indexes', v) for k,v in index_country_fp_d.items()}

#hazard tiles
index_hazard_fp_d ={
    '500_fluvial':r'500_fluvial\tileindex_500_fluvial.gpkg',
    '500_pluvial':r'500_pluvial\tileindex_500_pluvial.gpkg',
    '100_pluvial':r'100_pluvial\tileindex_100_pluvial.gpkg',
    '100_fluvial':r'100_fluvial\tileindex_100_fluvial.gpkg',
    '050_pluvial':r'050_pluvial\tileindex_050_pluvial.gpkg',
    '050_fluvial':r'050_fluvial\tileindex_050_fluvial.gpkg',
    '010_pluvial':r'010_pluvial\tileindex_010_pluvial.gpkg',
    '010_fluvial':r'010_fluvial\tileindex_010_fluvial.gpkg',    
    }

index_hazard_fp_d = {k:os.path.join(r'd:\05_DATA\2307_funcAgg\fathom\global3', v) for k,v in index_hazard_fp_d.items()}


def get_osm_bldg_cent(country_key, bounds, log=None,out_dir=None,
                      ):
    """intelligent retrival of building centroids"""
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    if log is None: log = get_log_stream()
    if out_dir is None: out_dir=os.path.join(lib_dir, 'bldg_cent')
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    #get record
    uuid = hashlib.shake_256(f'{country_key}_{bounds}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)    
    ofp = os.path.join(out_dir, f'{country_key}_{uuid}.geojson')
    
    if not os.path.exists(ofp):
        """this retrieves precompiled files if they are available"""
        log.info(f'retriving OSM building footprints for {country_key} from bounds: {bounds}')
        poly_fp = retrieve_osm_buildings(country_key, bounds)
        
        #drop to centroid                
        poly_gdf = gpd.read_file(poly_fp)
        if len(poly_gdf)==0:
            log.warning(f'for {country_key}.{bounds} got no polygons... skipping ')
            return None
        
        log.info(f'converting {len(poly_gdf)} polys to centroids')
        
        #add area (Equal Area Cylindrical CRS). drop to centroid 
        cent_gdf = gpd.GeoDataFrame(
            poly_gdf.geometry.to_crs(6933).area.rename('area')
            ).set_geometry(poly_gdf.geometry.centroid)
 
        
        cent_gdf.to_file(ofp)
        
        log.info(f'wrote {len(cent_gdf)} to \n    {ofp}')
        
        #===========================================================================
        # wrap
        #===========================================================================
        meta_d = {
                        'tdelta':(datetime.now()-start).total_seconds(),
                        'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                        'file_GB':os.path.getsize(ofp)/(1024**3),
                        #'output_MB':os.path.getsize(ofp)/(1024**2)
                        }
        log.info(meta_d)
    
        
    else:
        log.info(f'record exists for {country_key}.{bounds}\n    {ofp}')
        
    return ofp


def _sample_igrid(country_key, hazard_key, haz_tile_gdf, row, area_thresh, epsg_id, out_dir, log=None):
    
    if log is None: log=get_log_stream()
    i = row['id']
    
    #===========================================================================
    # get record
    #===========================================================================
    fnstr = f'{country_key}_{hazard_key}_{i}'
    uuid = hashlib.shake_256(f'{fnstr}_{epsg_id}_{area_thresh}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.geojson')
    
    if not os.path.exists(ofp):
        #=======================================================================
        # #get OSM building footprints
        #=======================================================================
        bldg_fp = get_osm_bldg_cent(country_key, row.geometry.bounds, log=log)
        if bldg_fp is None:
            return None
        bldg_pts_gdf = gpd.read_file(bldg_fp)
        #apply filter
        bx = bldg_pts_gdf['area'] > area_thresh
        bldg_pts_gser = bldg_pts_gdf[bx].geometry
        
        log.info(f'    filtered {bx.sum()}/{len(bx)} w/ area_tresh={area_thresh}')
        #=======================================================================
        # #retrieve the corresponding hazard raster
        #=======================================================================
        bx = haz_tile_gdf.to_crs(epsg=epsg_id).geometry.intersects(row.geometry.centroid)
        assert bx.sum() == 1, f'no intersect'
        rlay_fp = haz_tile_gdf[bx]['location'].values[0]
        assert os.path.exists(rlay_fp)
        log.info(f'    for grid {i} got hazard raster {os.path.basename(rlay_fp)}')
        
        #=======================================================================
        # #compute hte stats
        #=======================================================================
        log.info(f'    computing {len(bldg_pts_gser)} samples on {os.path.basename(rlay_fp)}')
        samp_pts = get_raster_point_samples(bldg_pts_gser, rlay_fp, colName=hazard_key, nodata=-32767)
        assert len(samp_pts) == len(bldg_pts_gser)
        log.debug(f'got counts\n' + str(samp_pts.iloc[:, 0].value_counts(dropna=False)))
        
        #write
    
        log.info(f'    writing to \n    {ofp}')
        samp_pts.to_file(ofp)
    else:
        log.info(f'    record exists: {ofp}')
    
    return ofp

def run_samples_on_country(country_key, hazard_key,
                           out_dir=None,
                           temp_dir=None,
                           epsg_id=4326,
                           area_thresh=50,
                           ):
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'samples')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if temp_dir is None:
        temp_dir = os.path.join(temp_dirM, 'samples', today_str)
    
    if not os.path.exists(temp_dir):os.makedirs(temp_dir)
    
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
    # #loop through each tile in the country grid 
    #===========================================================================
    res_d, err_d=dict(), dict()
    cnt=0
    for i, row in gdf.to_crs(epsg=epsg_id).iterrows():
        log.info(f'{i+1}/{len(gdf)} building for polygon %i'%row['id'])
        
        try:
            res_d[i] = _sample_igrid(country_key, hazard_key, haz_tile_gdf, row, area_thresh, epsg_id, out_dir, log)
 
        except Exception as e:
            err_d[i] = row.copy()
            err_d[i]['error'] = str(e)            
            log.error(f'failed on {country_key}.{hazard_key}.{i} w/\n    {e}')
        #print(f'computing stats on {len(gdf)} feats')
        """
        print(rasterstats.utils.VALID_STATS)
        """
        #=======================================================================
        # start_i = datetime.now()
        # with fiona.open(poly_fp, mode='r') as src:
        #     log.info(f'computing stats on {len(src)} polys')            
        #     zs = zonal_stats(src, rlay_fp, nodata=-32768, stats=['min', 'max', 'mean'], all_touched=False)
        # 
        # res_df = pd.DataFrame.from_dict(zs).dropna(axis=0)
        # tdelta = (datetime.now() - start_i).total_seconds()
        # log.info(f'got {len(res_df)} feats w/ valid stats in {tdelta:.2f} secs')
        #=======================================================================
        cnt+=1
    
        
 
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {cnt}')
    #write errors
    if len(err_d)>0:
        err_ofp = os.path.join(out_dir, f'errors_{today_str}_{country_key}_{hazard_key}.gpkg')
        
        err_gdf = pd.concat(err_d, axis=1).T
 
        log.error(f'writing {len(err_d)} error summaries to \n    {err_ofp}\n{err_gdf}')
        err_gdf.set_geometry(err_gdf['geometry']).to_file(err_ofp)
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
        
 
 
if __name__ == '__main__':
    
    run_samples_on_country('BGD', '500_fluvial')
    
    
    
    
    
    
    
    
    
    
    