'''
Created on Jul. 25, 2023

@author: cefect


intersecting agg grids with hazard rasters
    see 'intersect._01_sample.py' for intersecting buildings
'''
#===============================================================================
# IMPORTS--------
#===============================================================================
import os, hashlib, sys, subprocess

 
 
import psutil
from datetime import datetime
import pandas as pd
import numpy as np
import fiona
import geopandas as gpd
from osgeo import ogr
import rasterstats
from rasterstats import zonal_stats


from concurrent.futures import ProcessPoolExecutor

import psycopg2
from sqlalchemy import create_engine, URL

from tqdm import tqdm

from agg.coms_agg import get_conn_str, pg_getCRS

from definitions import (
    wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d, postgres_d, 
    equal_area_epsg, fathom_vals_d
    )
from definitions import temp_dir as temp_dirM

#whitebox
 
from whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()
wbt.set_compress_rasters(False) 
wbt.set_max_procs(1)
wbt.set_verbose_mode(True)

#===============================================================================
# print('\n\nPATH:\n')
# print('\n'.join(os.environ['PATH'].split(';')))
# print('\n\nPYTHONPATH:\n')
# print('\n'.join(os.environ['PYTHONPATH'].split(';')))
#===============================================================================
 
from coms import (
    init_log, today_str, get_log_stream, get_raster_point_samples, get_directory_size,
    dstr
    )


#from intersect._01_sample import get_osm_bldg_cent
 
    


 

#===============================================================================
# EXECUTORS--------
#===============================================================================
def _sql_to_gdf_from_spatial_intersect(schema, tableName, row_gdf, conn_d=postgres_d):
    """load a filtered table to geopanbdas"""
    
    #===========================================================================
    # get search geometry
    #===========================================================================
    epsg_id = pg_getCRS(schema, tableName, conn_d=conn_d)
    wkt_str = row_gdf.to_crs(epsg_id).geometry.iloc[0].wkt #reproject and get
    
    
    #===========================================================================
    # execute
    #===========================================================================
    conn =  psycopg2.connect(get_conn_str(conn_d))
    #set engine for geopandas
    engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
    try:
        cmd_str = f"""
                SELECT ST_Centroid(geom) AS geom, country_key, grid_size, i, j
                    FROM {schema}.{tableName}
                    WHERE ST_Intersects(geom, ST_GeomFromText('{wkt_str}', {epsg_id}))"""
                    
        result = gpd.read_postgis(cmd_str, engine, geom_col='geom') 
        
    except Exception as e:
        raise IOError(f'failed query w/ \n    {e}')
    finally:
        # Dispose the engine to close all connections
        engine.dispose()
        # Close the connection
        conn.close()
        

    return result
 
 

def _worker_get_intersect_grid_rlay_sample(i,row_raw, *args):
    #log.info(f'{i+1}/{len(gdf)} on grid %i'%row['id'])
    log=get_log_stream()
    try:
        res = get_intersect_grid_rlay_sample(row_raw, i, *args, log=log)
        return (i, res, None)
    except Exception as e:
        err = row_raw.copy()
        err['error'] = str(e)
        #log.error(f'failed on {country_key}.{hazard_key}.{i} w/\n    {e}')
        return (i, None, err)



def _get_intersect_rlay_fp(gser, haz_tile_gdf, haz_base_dir):
    """retrieve the rlay from the grid with a geoemtry lookup"""
    
    """better to use a projected CRS... haz_tile_gdf should be reprojected already"""
    bx = haz_tile_gdf.geometry.intersects(gser.to_crs(haz_tile_gdf.crs).geometry.iloc[0].centroid)
    
    
    assert bx.sum() == 1, f'no intersect'
    #get filepath
    """the tile_indexers give absolute filepaths (from when the index was created)"""
    rlay_fp = os.path.join(haz_base_dir, 'raw', os.path.basename(haz_tile_gdf[bx]['location'].values[0]))
    assert os.path.exists(rlay_fp)
    return rlay_fp


def get_intersect_grid_rlay_sample(row_raw, i, out_dir, haz_tile_gdf, haz_base_dir, crs,  
                                hazard_key, country_key, grid_size, log=None):
    
    #===========================================================================
    # get filepath
    #===========================================================================
    fnstr = f'{country_key}_{hazard_key}_{grid_size}_{i}'
    uuid = hashlib.shake_256(f'{fnstr}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.gpkg')
    
    log = log.getChild(fnstr)
    #===========================================================================
    # build
    #===========================================================================
    if not os.path.exists(ofp):
        log.info(f'{country_key}.{hazard_key} on grid %i'%row_raw['id'])
        row_gdf = gpd.GeoDataFrame([row_raw], crs=crs) #reconstruct geopandas object
        #=======================================================================
        # #retrieve the corresponding hazard raster
        #=======================================================================
        rlay_fp = _get_intersect_rlay_fp(row_gdf, haz_tile_gdf, haz_base_dir)
        log.debug(f'    for grid {i} got hazard raster {os.path.basename(rlay_fp)}')
        #===================================================================
        # retrieve the aggregated grid centroids
        #===================================================================
        agg_gridsC_gdf = _sql_to_gdf_from_spatial_intersect('grids', f'agg_{country_key.lower()}_{grid_size:07d}', row_gdf)
        log.debug(f'    got {len(agg_gridsC_gdf)} agg grids')
        #=======================================================================
        # sample
        #=======================================================================
        log.debug(f'    computing {len(agg_gridsC_gdf)} samples on {os.path.basename(rlay_fp)}')
        agg_gridsC_samp_gdf = _wbt_sample(rlay_fp, agg_gridsC_gdf,  hazard_key, log)
        
        #=======================================================================
        # write
        #=======================================================================
        agg_gridsC_samp_gdf.to_file(ofp)
        log.info(f'    finished on {agg_gridsC_samp_gdf.shape} and wrote to\n    {ofp}')
        
    else:
        log.info(f'intersect exists... skipping')
        
    
        
    return ofp


def _wbt_sample(rlay_fp, gdf_pts,  hazard_key, log, out_dir=None,
                nodata_l=None,
                ):
    """ sample points with WBT"""
    #===========================================================================
    # defaults
    #===========================================================================
    if out_dir is None:
        out_dir = os.path.join(temp_dirM, 'agg', 'wbt_sample')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if nodata_l is None: nodata_l = list(fathom_vals_d.keys())
    
    
    #===========================================================================
    # #write to file
    #===========================================================================
    """WBT requires a shape file..."""    
    hash_data = gdf_pts['i'].unique().tolist()+ gdf_pts['j'].unique().tolist()
    
    uuid = hashlib.shake_256(f'{rlay_fp}_{hash_data}_{hazard_key}_{datetime.now()}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir, f'{hazard_key}_{uuid}.shp')
   
    gdf_pts.to_crs(4326).geometry.to_file(ofp)
    
    #===========================================================================
    # execute
    #===========================================================================
    def wbt_callback(value):
        if not "%" in value:
            log.debug(value)
            
    wbt.extract_raster_values_at_points(
        rlay_fp, 
        ofp, 
        out_text=False, 
        callback=wbt_callback)
    
    #===============================================================================
    # #clean and convert
    #===============================================================================
    log.debug(f'loading and cleaning wbt result file: {ofp}')
    samp_gdf = gpd.read_file(ofp).to_crs(gdf_pts.crs)
    
    #join data back
    samp_gdf2 = samp_gdf.rename(columns={'VALUE1':hazard_key}).set_index('FID').join(gdf_pts.drop('geom', axis=1))
    samp_gdf2.index.name = None
    
    #fix nodata type
    nodata_bx = samp_gdf2[hazard_key].astype(float).isin(nodata_l)
    if nodata_bx.any():
        log.debug(f'    set {nodata_bx.sum()}/{len(nodata_bx)} nodata vals to nan')
        samp_gdf2.loc[nodata_bx, hazard_key] = np.nan
        
    if not nodata_bx.all():
        assert samp_gdf2[hazard_key].min()>-9998, 'still some nulls?'
    else:
        log.warning(f'got all nulls')
    
 
    samp_gdf2.dtypes
    return samp_gdf2

def run_agg_samples_on_country(
                        country_key, 
                               hazard_key,
                               grid_size=1020,
                           out_dir=None,
                           temp_dir=None,
                           #epsg_id=4326,
                           area_thresh=50,
                           max_workers=None,
                           ):
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    assert hazard_key in index_hazard_fp_d, hazard_key
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'agg','04_sample', country_key, hazard_key, f'{grid_size:05d}')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if temp_dir is None:
        temp_dir = os.path.join(temp_dirM, 'agg','04_sample', today_str)    
    if not os.path.exists(temp_dir):os.makedirs(temp_dir)
    
    log = init_log(name=f'aggSamp.{country_key}.{hazard_key}.{grid_size}', fp=os.path.join(out_dir, today_str+'.log'))
    log.info(f'on {country_key} x {hazard_key} x {grid_size}')
    
    keys_d = {'country_key':country_key, 'hazard_key':hazard_key, 'grid_size':grid_size}
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
    haz_tile_gdf = gpd.read_file(index_hazard_fp_d[hazard_key]).to_crs(equal_area_epsg)
    log.info(f'loaded hazard tiles w/ {len(haz_tile_gdf)}')
    
    haz_base_dir = os.path.dirname(index_hazard_fp_d[hazard_key])
    #===========================================================================
    # #loop through each tile in the country grid 
    #===========================================================================
 
    res_d, err_d=dict(), dict()
    cnt=0
    args = (out_dir, haz_tile_gdf, haz_base_dir, gdf.crs, hazard_key, country_key, grid_size)
    #===========================================================================
    # single thread
    #===========================================================================
    log.info(f'intersecting buildings and hazard per tile \n\n')
    if max_workers is None:     
 
        for i, row_raw in tqdm(gdf.iterrows()):          
            
            res_d[i] = get_intersect_grid_rlay_sample(row_raw, i, *args, log=log)            
            
            cnt+=1
            
            
 
    #===========================================================================
    # MULTI thread
    #===========================================================================
    else:
        #gdf = gdf.iloc[0:20, :]
        log.info(f'running {len(gdf)} w/ max_workers={max_workers}')
 
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker_get_intersect_grid_rlay_sample, i, row, *args) for i, row in gdf.iterrows()]
            for future in tqdm(futures):
                i, res, err = future.result()
                if res is not None:
                    res_d[i] = res
                if err is not None:
                    err_str = err['error']
                    log.error(f'{i} returned error:\n{err_str}')
                    err_d[i] = err
                cnt+=1
                
 
         
    log.info(f'finished w/ {len(res_d)}')
    #===========================================================================
    # add to meta
    #===========================================================================
    #compute
    meta_d =dict()
    for i, fp in res_d.items():
        ser = gpd.read_file(fp, include_fields=[hazard_key], ignore_geometry=True).iloc[:,0]
        d = {k:getattr(ser, k)() for k in ['min', 'max', 'mean', 'std']}
        meta_d[i] = {**d, **{'null_cnt':ser.isna().sum(), 'zero_cnt':(ser==0).sum(), 'len':len(ser), 'fp':fp}}
 
    #collect
    meta_df = pd.DataFrame.from_dict(meta_d, orient='index')
    
    #add indexers
    for k,v in keys_d.items():
        meta_df[k]=v
 
    #join back onto grid and write
    ofp = os.path.join(out_dir, f'aggSamp_meta_{country_key}_{hazard_key}_{grid_size}_{today_str}.gpkg')
    gdf.join(meta_df, how='inner').to_file(ofp)
    log.info(f'wrote {len(meta_df)} meta to \n    {ofp}')
 
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
    
    run_agg_samples_on_country('DEU', '500_fluvial', max_workers=2)
    
    
    
    
    
    
    
    
    
    
    