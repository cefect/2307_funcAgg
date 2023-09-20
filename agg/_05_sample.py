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

from osgeo import ogr
import fiona
import shapely.geometry
import shapely.wkt

import geopandas as gpd

#import rasterstats
#from rasterstats import zonal_stats

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import psycopg2
from sqlalchemy import create_engine, URL

from tqdm import tqdm

from coms import view, clean_geodataframe, pd_ser_meta, init_log_worker
from agg.coms_agg import get_conn_str, pg_getCRS

from definitions import (
    wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d, postgres_d, 
    equal_area_epsg, fathom_vals_d, aoi_wkt_str
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
def _sql_to_gdf_from_spatial_intersect(schema, tableName, row_gdf, conn_str=None):
    """load a filtered table to geopanbdas"""
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    #===========================================================================
    # get search geometry
    #===========================================================================
    epsg_id = pg_getCRS(schema, tableName, conn_str=conn_str)
    wkt_str = row_gdf.to_crs(epsg_id).geometry.iloc[0].wkt #reproject and get
    
    
    #===========================================================================
    # execute
    #===========================================================================
    conn =  psycopg2.connect(conn_str)
    #set engine for geopandas
    engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
    try:
        cmd_str = f"""
                SELECT ST_Centroid(geom) AS geom, country_key, grid_size, i, j
                    FROM {schema}.{tableName}
                    WHERE ST_Intersects(ST_Centroid(geom), ST_GeomFromText('{wkt_str}', {epsg_id}))"""
                    
        #print(cmd_str)
        result = gpd.read_postgis(cmd_str, engine, geom_col='geom') 
        
    except Exception as e:
        raise IOError(f'failed query w/ \n    {e}')
    finally:
        # Dispose the engine to close all connections
        engine.dispose()
        # Close the connection
        conn.close()
        

    return result
 
 

#===============================================================================
# def _worker_get_intersect_grid_rlay_sample(i,row_raw, *args):
#     #log.info(f'{i+1}/{len(gdf)} on grid %i'%row['id'])
#     log=get_log_stream()
#     try:
#         res = get_intersect_grid_rlay_sample(row_raw, i, *args, log=log)
#         return (i, res, None)
#     except Exception as e:
#         err = row_raw.copy()
#         err['error'] = str(e)
#         #log.error(f'failed on {country_key}.{hazard_key}.{i} w/\n    {e}')
#         return (i, None, err)
#===============================================================================



#===============================================================================
# def _get_intersect_rlay_fp(gser, haz_base_dir):
#     """retrieve the rlay from the grid with a geoemtry lookup"""
#     
#     """better to use a projected CRS... haz_tile_gdf should be reprojected already"""
#     bx = haz_tile_gdf.geometry.intersects(gser.to_crs(haz_tile_gdf.crs).geometry.iloc[0].centroid)
#     
#     
#     assert bx.sum() == 1, f'no intersect'
#     #get filepath
#     """the tile_indexers give absolute filepaths (from when the index was created)"""
#     rlay_fp = os.path.join(haz_base_dir, 'raw', os.path.basename(haz_tile_gdf[bx]['location'].values[0]))
#     assert os.path.exists(rlay_fp)
#     return rlay_fp
#===============================================================================





def _wbt_sample(rlay_fp, gdf_pts,  fnstr, 
                log=None, out_dir=None,  rlay_crs=4326,
                ):
    """ sample points with WBT"""
    #===========================================================================
    # defaults
    #===========================================================================
    if out_dir is None:
        out_dir = os.path.join(temp_dirM, 'agg', 'wbt_sample')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    
    
    
    #===========================================================================
    # #write to file
    #===========================================================================
    """WBT requires a shape file..."""    
    hash_data = gdf_pts['i'].unique().tolist()+ gdf_pts['j'].unique().tolist()
    
    """no cache... always writing"""
    uuid = hashlib.shake_256(f'{rlay_fp}_{hash_data}_{fnstr}_{datetime.now()}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.shp')
   
    log.debug(f'writing {gdf_pts.shape} to \n    {ofp}')
    gdf_pts.to_crs(rlay_crs).geometry.to_file(ofp)
    
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
    

    log.debug(f'finished')
    return ofp


def sample_rlay_from_gdf_wbt(rlay_fp, gdf, hazard_key, pfx, log=None, nodata_l=None, **kwargs):
    """wrap around wbt sample with some post cleaning"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    log=log.getChild(hazard_key)
    if nodata_l is None: nodata_l = list(fathom_vals_d.keys())
    log.debug(f'on {rlay_fp} w/ {len(gdf)}')
    #===========================================================================
    # sampe to shp with wbt
    #===========================================================================
    samp_fp = _wbt_sample(rlay_fp, gdf,pfx, log=log, **kwargs)
    #===============================================================================
    # #clean and convert
    #===============================================================================
    log.debug(f'loading and cleaning wbt result file: {samp_fp}')
    samp_gdf = gpd.read_file(samp_fp).to_crs(gdf.crs)
    
    #join data back
    samp_gdf2 = samp_gdf.rename(columns={'VALUE1':hazard_key}).set_index('FID').join(gdf.drop('geom', axis=1))
    samp_gdf2.index.name = None
    
    #fix nodata type
    nodata_bx = samp_gdf2[hazard_key].astype(float).isin(nodata_l)
    if nodata_bx.any():
        log.debug(f'    set {nodata_bx.sum()}/{len(nodata_bx)} nodata vals to nan')
        samp_gdf2.loc[nodata_bx, hazard_key] = np.nan
    if not nodata_bx.all():
        assert samp_gdf2[hazard_key].min() > -9998, 'still some nulls?'
    else:
        log.warning(f'got all nulls')
    return samp_gdf2


 


def agg_samples_on_rlay(kdi, row_gdf, gridTable, haz_base_dir, out_dir, temp_dir, 
                        log=None, use_cache=False, dev=False,
                        grid_schema='grids'):
    """get samples for grids on a single raster tile"""
    
    #===========================================================================
    # setup
    #===========================================================================
    i = kdi.pop('tile_id')
    pfx_i = '_'.join([str(e) for e in kdi.values()]) + f'_{i:05d}'
    log = log.getChild(str(i))
    
    if dev: 
        use_cache=False
        grid_schema='dev'
    #=======================================================================
    # get hazard filepath
    #=======================================================================
    rlay_fp = os.path.join(haz_base_dir, os.path.basename(row_gdf['location'].iloc[0]))
    assert os.path.exists(rlay_fp), f'hazard tile file does not exist for {i}\n    {rlay_fp}'
    
    #===========================================================================
    # get filepath
    #===========================================================================
    
    
    uuid = hashlib.shake_256(f'{gridTable}'.encode("utf-8"), usedforsecurity=False).hexdigest(8)    
    ofp = os.path.join(out_dir, f'{pfx_i}_{uuid}.gpkg')
    
    log.debug(f' on {pfx_i}')
    #===========================================================================
    # build
    #===========================================================================
    if not os.path.exists(ofp) or (not use_cache):
        log.debug(f'    sampling {pfx_i}')
        #===================================================================
        # retrieve the aggregated grid centroids that intersect
        #===================================================================
        
        
        agg_gridsC_gdf = _sql_to_gdf_from_spatial_intersect(grid_schema, gridTable, row_gdf)
        log.debug(f'    got {len(agg_gridsC_gdf)} agg grids intersecting haz tile')
        #=======================================================================
        # sample
        #=======================================================================
        log.debug(f'    computing {len(agg_gridsC_gdf)} samples on {os.path.basename(rlay_fp)}')
        agg_gridsC_samp_gdf = sample_rlay_from_gdf_wbt(rlay_fp, agg_gridsC_gdf, kdi['hazard_key'], pfx_i, 
            log=log, out_dir=os.path.join(temp_dir, 'wbt_samp'))
    
        if not len(agg_gridsC_samp_gdf)>0:
            """just deal with this during collection"""
            log.warning(f'got no features')
            return 'empty', {}
        #=======================================================================
        # add some indexers
        #=======================================================================
        for k, v in kdi.items():
            agg_gridsC_samp_gdf[k] = v            
     
        #=======================================================================
        # write
        #=======================================================================
        
        clean_geodataframe(agg_gridsC_samp_gdf).to_file(ofp)
        log.debug(f'    finished on {agg_gridsC_samp_gdf.shape} and wrote to\n    {ofp}')
        
    else:
        log.debug(f'loading from cache')
        log.debug(ofp)
        use_cache=True
        #load just the data
        agg_gridsC_samp_gdf = gpd.read_file(ofp, ignore_geometry=True, include_fields=[kdi['hazard_key']])
    
    #=======================================================================
    # meta
    #=======================================================================
    meta_d = pd_ser_meta(agg_gridsC_samp_gdf[kdi['hazard_key']])
    meta_d.update({'geometry':row_gdf.geometry.iloc[0], 'use_cache':use_cache,'rlay_fp':rlay_fp, 'ofp':ofp})
    return ofp, meta_d


def process_row(i, row_raw, keys_d, crs, *args):
    log = init_log_worker()
    
    kdi = {**keys_d, **{'tile_id':i}}
 
    try:
        res_d, meta_d = agg_samples_on_rlay(kdi, gpd.GeoDataFrame([row_raw], crs=crs), *args, log=log)
        return i, res_d, meta_d, None
    except Exception as e:
        log.error(f'for {i} got error\n    {e}')
        err_d = {**{'error': str(e), 'now':datetime.now()}, **row_raw.to_dict()}
        return i, None, None, err_d

def run_agg_samples_on_country(
                        country_key, 
                               hazard_key,
                               grid_size,
                               
                               haz_index_fp=None,
                               
                           out_dir=None,
                           temp_dir=None,
                           #epsg_id=4326,
                           #area_thresh=50,
                           max_workers=None,
                           crs=equal_area_epsg,
                           dev=False,
                           ):
    """sample fathom rasters with aggregation grids
    
    uses these postgres tables:
        grids: f'agg_{country_key.lower()}_{grid_size:07d}_wbldg
        
    
    Returns
    --------
    writes a gpkg file for each country, haz, grid_size, and haz_grid with sampled values
    
    """
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    assert hazard_key in index_hazard_fp_d, hazard_key
    
    if out_dir is None:
        root_fldr = '05_sample'
        if dev: root_fldr+='_dev'
        out_dir = os.path.join(wrk_dir, 'outs', 'agg',root_fldr, country_key, hazard_key, f'{grid_size:04d}')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if temp_dir is None:
        temp_dir = os.path.join(temp_dirM, 'agg','05_sample', today_str)    
    if not os.path.exists(temp_dir):os.makedirs(temp_dir)
    
    if haz_index_fp is None:
        haz_index_fp=index_hazard_fp_d[hazard_key]
    
    log = init_log(name=f'aggSamp.{country_key}.{hazard_key}.{grid_size}', fp=os.path.join(out_dir, today_str+'.log'))
    
    
    keys_d = {'country_key':country_key, 'hazard_key':hazard_key, 'grid_size':grid_size}
    pfx = '_'.join([str(e) for e in keys_d.values()])
    
    log.info(keys_d)
    #===========================================================================
    # #load hazard tiles
    #=========================================================================== 
 
    #hazard
    log.info(f'loading hazard tiles and reprojectiong ({crs})\n    {haz_index_fp}')
    haz_tile_gdf_raw = gpd.read_file(haz_index_fp).to_crs(crs)    
    haz_base_dir = os.path.join(os.path.dirname(haz_index_fp),'raw') #need thsi for relative pathing
    
    #get country slice
    """the hazard tiles are global... could also add a country_key column upstream"""
    if not dev:
        aoi_bbox = shapely.geometry.box(*gpd.read_file(index_country_fp_d[country_key]).to_crs(crs).geometry.total_bounds)
    else:
        aoi_bbox = shapely.wkt.loads(aoi_wkt_str)
    bx = haz_tile_gdf_raw.geometry.intersects(aoi_bbox)
    if not bx.any():
        raise AssertionError(f'failed to find any hazard tiles that intersect with {country_key}')
    
    haz_tile_gdf=haz_tile_gdf_raw.loc[bx, :]
    log.info(f'loaded hazard tiles for \'{country_key}\' ({bx.sum()}/{len(bx)})')
    
    
    
    """
    view(haz_tile_gdf)
    """
    #===========================================================================
    # #loop through each tile
    #===========================================================================
 
    cnt=0
    meta_lib, res_d, err_d=dict(), dict(), dict()
    args = (f'agg_{country_key.lower()}_{grid_size:07d}_wbldg', haz_base_dir, out_dir, temp_dir)
    #===========================================================================
    # single 
    #===========================================================================
    log.info(f'intersecting buildings and hazard per tile \n\n')
   
    if max_workers is None:
        for i, row_raw in haz_tile_gdf.iterrows():
     
            
            log.info(f'{country_key}.{hazard_key} on grid {i}')
     
            try:
     
                res_d[i], meta_lib[i] = agg_samples_on_rlay({**keys_d, **{'tile_id':i}}, 
                                                   gpd.GeoDataFrame([row_raw], crs=crs), 
                                                   *args, log=log, dev=dev)
            except Exception as e:
                log.error(f'for {i} got error\n    {e}')
                err_d[i] = {**{'error': str(e), 'now':datetime.now()}, **row_raw.to_dict()}
                            
            
            cnt+=1
         
    #===========================================================================
    # MULTI PROCESS---------
    #===========================================================================
    else:
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_row, i, row_raw, keys_d, crs, *args): i for i, row_raw in haz_tile_gdf.iterrows()}
            for future in tqdm(futures):
                i, res, meta_d, err = future.result()
                if res is not None:
                    res_d[i], meta_lib[i] = res, meta_d
                elif err is not None: 
                    log.error(f'{i} returned error:\n%s'%err['error'])
                    err_d[i] = err
                cnt+=1
        
 
        
 
 
    
    #===========================================================================
    # wrap meta
    #===========================================================================
    
    meta_gdf = clean_geodataframe(pd.DataFrame.from_dict(meta_lib, orient='index'), crs=crs)
     
    meta_ofp = os.path.join(out_dir, f'_meta_agg05samp_{pfx}_{today_str}.gpkg')
    meta_gdf.to_file(meta_ofp) 
    log.info(f'wrote meta {meta_gdf.shape} to \n    {meta_ofp}')
    """
    view(meta_df)
    """
 
    #===========================================================================
    # errors
    #===========================================================================
    #log.info(f'finished on {cnt}')
    #write errors
    if len(err_d)>0:
        err_ofp = os.path.join(out_dir, f'_errors_agg05samp_{pfx}_{today_str}.gpkg')
        
        err_gdf = clean_geodataframe(pd.DataFrame.from_dict(err_d, orient='index'), crs=crs)
 
        log.error(f'writing {len(err_d)} error summaries to \n    {err_ofp}')
        err_gdf.to_file(err_ofp)
    
    #===========================================================================
    # wrap
    #===========================================================================
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(f'finished w/ \n{meta_d}')
        
 
 
if __name__ == '__main__':
    
    run_agg_samples_on_country('DEU', '050_fluvial',60, max_workers=2, dev=False)
    
    
    
    
    
    
    
    
    
    
    