'''
Created on Jul. 25, 2023

@author: cefect


intersecting building data with hazard rasters
'''
import os, hashlib, sys

 
 
import psutil
from datetime import datetime
import pandas as pd
import fiona
import geopandas as gpd
import rasterstats
from rasterstats import zonal_stats

from concurrent.futures import ProcessPoolExecutor

from intersect.osm import retrieve_osm_buildings

from definitions import wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d
from definitions import temp_dir as temp_dirM

#===============================================================================
# print('\n\nPATH:\n')
# print('\n'.join(os.environ['PATH'].split(';')))
# print('\n\nPYTHONPATH:\n')
# print('\n'.join(os.environ['PYTHONPATH'].split(';')))
#===============================================================================
 
from hp import (
    init_log, today_str, get_log_stream, get_raster_point_samples, get_directory_size,
    dstr
    )



#===============================================================================
# file indexers
#===============================================================================



def get_osm_bldg_cent(country_key, bounds, log=None,out_dir=None, pfx='',
                      ):
    """intelligent retrival of building centroids"""
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    if log is None: log = get_log_stream()
    if out_dir is None: out_dir=os.path.join(lib_dir,'bldg_cent', country_key)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    #get record
    uuid = hashlib.shake_256(f'{country_key}_{bounds}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)    
    ofp = os.path.join(out_dir, f'{pfx}_{uuid}.geojson')
    
    #===========================================================================
    # build
    #===========================================================================
    if not os.path.exists(ofp):
        """this retrieves precompiled files if they are available"""
        log.info(f'retriving OSM building footprints for {country_key} from bounds: {bounds}')
        poly_fp = retrieve_osm_buildings(country_key, bounds, logger=log)
        
        #=======================================================================
        # #drop to centroid                
        #=======================================================================
        log.info(f'loading osm building poly file {os.path.getsize(poly_fp)/(1024**3): .2f} GB \n    {poly_fp}')
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


def _sample_igrid(country_key, hazard_key, haz_tile_gdf, row, area_thresh, epsg_id, out_dir, log=None, haz_base_dir=None):
    
    if log is None: log=get_log_stream()
    i = row['id']
    
    #===========================================================================
    # get record
    #===========================================================================
    fnstr = f'{country_key}_{hazard_key}_{i}'
    uuid = hashlib.shake_256(f'{fnstr}_{epsg_id}_{area_thresh}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir,f'{fnstr}_{uuid}.geojson')
    
    if not os.path.exists(ofp):
        #=======================================================================
        # #get OSM building footprints
        #=======================================================================
        bldg_fp = get_osm_bldg_cent(country_key, row.geometry.bounds, pfx=fnstr, log=log)
        if bldg_fp is None:
            return None
        
        log.info(f'loading osm building file {os.path.getsize(bldg_fp)/(1024**3)}GB \n    {bldg_fp}')
        bldg_pts_gdf = gpd.read_file(bldg_fp)
        
        #apply filter
        log.debug(f'    applying filter on {len(bldg_pts_gdf)}')
        bx = bldg_pts_gdf['area'] > area_thresh
        bldg_pts_gser = bldg_pts_gdf[bx].geometry
        
        log.debug(f'    filtered {bx.sum()}/{len(bx)} w/ area_tresh={area_thresh}')
        #=======================================================================
        # #retrieve the corresponding hazard raster
        #=======================================================================
        bx = haz_tile_gdf.to_crs(epsg=epsg_id).geometry.intersects(row.geometry.centroid)
        assert bx.sum() == 1, f'no intersect'
        
        #get filepath
        """the tile_indexers give absolute filepaths (from when the index was created)"""
        rlay_fp = os.path.join(haz_base_dir, 'raw', os.path.basename(haz_tile_gdf[bx]['location'].values[0]))
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

def _multi_sample_igrid(i, row, country_key, hazard_key, haz_tile_gdf, area_thresh, epsg_id, out_dir, haz_base_dir):
    #log.info(f'{i+1}/{len(gdf)} on grid %i'%row['id'])
    try:
        res = _sample_igrid(country_key, hazard_key, haz_tile_gdf, row, area_thresh, epsg_id, out_dir,   haz_base_dir=haz_base_dir, log=None)
        return (i, res, None)
    except Exception as e:
        err = row.copy()
        err['error'] = str(e)
        #log.error(f'failed on {country_key}.{hazard_key}.{i} w/\n    {e}')
        return (i, None, err)


def run_samples_on_country(country_key, hazard_key,
                           out_dir=None,
                           temp_dir=None,
                           epsg_id=4326,
                           area_thresh=50,
                           max_workers=None,
                           ):
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    assert hazard_key in index_hazard_fp_d, hazard_key
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'samps', country_key, hazard_key)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if temp_dir is None:
        temp_dir = os.path.join(temp_dirM, 'samples', today_str)
    
    if not os.path.exists(temp_dir):os.makedirs(temp_dir)
    
    log = init_log(name=f'samp.{country_key}.{hazard_key}', fp=os.path.join(out_dir, today_str+'.log'))
    log.info(f'on {country_key} x {hazard_key}')
    
    #===========================================================================
    # #load tiles
    #===========================================================================
    #country 
    gdf = gpd.read_file(index_country_fp_d[country_key])
    log.info(f'loaded country tiles w/ {len(gdf)}')
    
    #add index
    if not 'id' in gdf:
        gdf['id'] = gdf.index
        
    
    #hazard
    haz_tile_gdf = gpd.read_file(index_hazard_fp_d[hazard_key])
    log.info(f'loaded hazard tiles w/ {len(haz_tile_gdf)}')
    
    haz_base_dir = os.path.dirname(index_hazard_fp_d[hazard_key])
    #===========================================================================
    # #loop through each tile in the country grid 
    #===========================================================================
 
    res_d, err_d=dict(), dict()
    cnt=0
    
    #===========================================================================
    # single thread
    #===========================================================================
    log.info(f'intersecting buildings and hazard per tile \n\n')
    if max_workers is None:
        for i, row in gdf.to_crs(epsg=epsg_id).iterrows():
            log.info(f'{i+1}/{len(gdf)} on grid %i'%row['id'])
            
            try:
                res_d[i] = _sample_igrid(country_key, hazard_key, haz_tile_gdf, row, area_thresh, epsg_id, out_dir, log, haz_base_dir)
     
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
    # MULTI thread
    #===========================================================================
    else:
        gdf = gdf.iloc[0:8, :]
        log.info(f'running {len(gdf)} w/ max_workers={max_workers}')
        args = (country_key, hazard_key, haz_tile_gdf, area_thresh, epsg_id, out_dir, haz_base_dir)
        
        #run once to prime
        """reduce conflicts...
        get_tag_filter() for example applies to the whole country"""
        
        for i, row in gdf.to_crs(epsg=epsg_id).iterrows():
            _multi_sample_igrid(i, row, *args)
            break
 
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_multi_sample_igrid, i, row, *args) for i, row in gdf.to_crs(epsg=epsg_id).iterrows()]
            for future in futures:
                i, res, err = future.result()
                if res is not None:
                    res_d[i] = res
                if err is not None:
                    err_d[i] = err
        
    
        
 
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {cnt}')
    #write errors
    if len(err_d)>0:
        err_ofp = os.path.join(out_dir, f'errors_{today_str}_{country_key}_{hazard_key}.gpkg')
        
        err_gdf = pd.concat(err_d, axis=1).T
 
        log.error(f'writing {len(err_d)} error summaries to \n    {err_ofp}\n'+err_gdf['error'])
        err_gdf.set_geometry(err_gdf['geometry']).to_file(err_ofp)
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
        
 
 
if __name__ == '__main__':
    
    run_samples_on_country('AUS', '100_fluvial', max_workers=None)
    
    
    
    
    
    
    
    
    
    
    