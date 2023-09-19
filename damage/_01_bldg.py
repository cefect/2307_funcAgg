'''
Created on Sep. 18, 2023

@author: cefect

compute building losses
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
import geopandas as gpd

#import rasterstats
#from rasterstats import zonal_stats

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import psycopg2
from sqlalchemy import create_engine, URL

from tqdm import tqdm

from coms import (
    view, clean_geodataframe, pd_ser_meta, init_log_worker, init_log, today_str,
    get_directory_size,
    )


from definitions import (
    wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d, postgres_d, 
    equal_area_epsg, fathom_vals_d
    )


from damage.coms_dmg import get_rloss
from funcMetrics.func_prep import get_funcLib
from funcMetrics.coms_fm import slice_serx, force_max_depth, force_zero_zero, force_monotonic

from agg.coms_agg import get_conn_str, pg_getCRS, pg_exe





def write_loss_haz_chunk(gdf, func_d, wd_scale, out_dir, fnstr, log=None, use_cache=True):
    """compute loss for this hazard chunk on all functions"""
    #===========================================================================
    # defaults
    #===========================================================================
    
    log.debug(f'w/ {gdf.shape} and  {len(func_d)} funcs')
    
    
    #===========================================================================
    # get ofp
    #===========================================================================
    uuid = hashlib.shake_256(f'{func_d}_{gdf.columns}_{gdf.index}_{wd_scale}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')
    
    #===========================================================================
    # build
    #===========================================================================
    if not os.path.exists(ofp) or (not use_cache):
        log.debug(f'building on {len(gdf)}')
        wd_ar = gdf.T.values[0] * wd_scale
        #=======================================================================
        # loop on each function
        #=======================================================================
        d = dict()
        for df_id, dd_ar in func_d.items():
            log.debug(df_id)
            #get loss
            loss_ar = get_rloss(dd_ar, wd_ar, 
                prec=None) #depths in table are rounded enough
            #append index and collect
            d[df_id] = pd.Series(loss_ar, index=gdf.index, name=df_id)
    
        #===================================================================
        # #collect and wirte
        #===================================================================
        loss_df = pd.concat(d, axis=1)
        
        loss_df.to_pickle(ofp)
        
        log.debug(f'wrote {loss_df.shape} to \n    {ofp}')
        
    #===========================================================================
    # cache
    #===========================================================================
    else:
        log.info(f'file exists... skipping')
        log.debug(ofp)
        
    return ofp

def run_bldg_loss(
        
        country_key, 
        
        max_depth=None,
        fserx=None,
       haz_coln_l=None,
       wd_scale=0.01, 
 
       
       #index_fp=None,
                               
        out_dir=None,
        conn_str=None,
        schema='damage', 
        chunksize=1e5,
 
        ):
    """join grid and osm intersections
    
    goes pretty fast acutally with the zero filter
    
    postgres:
        inters.{country_key}: building depths with columns for hazard
    
    Params
    ------
    fsers: pd.Series
        relative loss functions cleaned and prepped
        
    wd_scale: float
        scaling wd values (to convert to m)
        
    Writes
    ----------
    pd.Series
        index: asset id
        columns: loss for each function on one hazard
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()   
    
    country_key=country_key.lower() 
    
 
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'damage','01_bldg', country_key)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    #log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    if haz_coln_l is None: 
        from definitions import haz_coln_l
        #haz_coln_l = [e[1:] for e in haz_coln_l] #remove the f prefix
    assert isinstance(haz_coln_l, list)
    
    
    if max_depth is None:
        from funcMetrics.coms_fm import max_depth
        
    log = init_log(name=f'rlBldg', fp=os.path.join(out_dir, today_str+'.log'))
    
    log.info(f'on {country_key}') 
    
    
    #===========================================================================
    # prep loss functions
    #===========================================================================
    if fserx is None: fserx = get_funcLib() #select functions
    
    #===========================================================================
    # no need as the interp just uses max loss anyway
    # #extend
    # """using full index as we are changing the index (not just adding values"""
    # fserx_extend = force_max_depth(fserx, max_depth, log).rename('rl')
    #===========================================================================
 
    #drop meta and add zero-zero
    fserx_s = force_monotonic(
        force_zero_zero(slice_serx(fserx, xs_d=None), log=log), 
        log=log)
    
    """
    view(fserx_s)
    """
    #collapse to dictinoary of wd-rl
    func_d = {df_id: gserx.droplevel(list(range(gserx.index.nlevels-1))).reset_index().T.values for df_id, gserx in fserx_s.groupby('df_id')}
        
 
    #===========================================================================
    # loop on hazards
    #===========================================================================
    """looping on haz columns gives more control and allows more filtering (e.g., faster)"""
    cnt=0
    res_lib=dict()
    log.info(f'computing on {len(haz_coln_l)} hazards for {country_key}')
    for haz_coln in haz_coln_l:
        
        log.info(f'on {haz_coln} w/ {len(func_d)} funcs')
        #===========================================================================
        # loop and load from postgis
        #===========    ================================================================
        """could do everything in postgis... but I'm not sure this is faster and I'd have to figure it out"""
        conn =  psycopg2.connect(conn_str)
        engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
        
        
        cmd_str = f'SELECT id, {haz_coln} \nFROM inters.{country_key}'
        
        #exclude empties
        cmd_str += f'\nWHERE {haz_coln} >0 AND {haz_coln} IS NOT NULL'
        #=======================================================================
        # #loop through chunks of the table
        # cmd_str = f'SELECT id, ' + ', '.join(haz_coln_l) + f' \nFROM inters.{country_key}'
        # 
        # #only non-zeros
        # cmd_str+=f'WHERE '+ ' >0 AND '.join(haz_coln_l)
        #=======================================================================
        
        #dev limter (see i break below also)
        #cmd_str+= f'\nLIMIT {chunksize*10}'
        #log.info(cmd_str)
        res_d=dict()
        
 
        for i, gdf in enumerate(pd.read_sql(cmd_str, engine, index_col=['id'], chunksize=int(chunksize))):
            log.info(f'{i} on {gdf.shape}')
            res_d[i] = write_loss_haz_chunk(gdf, func_d, wd_scale, out_dir, f'bldg_{country_key}_{haz_coln}_{i:07d}',log=log)
            
            #wrap chunk
            #if i>4:break
            cnt+=1
            
        #wrap haz_coln loop
        log.info(f'finished {haz_coln} w/ {len(res_d)}\n\n')
        res_lib[haz_coln] = res_d
        
 
 
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {len(res_lib)}')
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'outdir_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    return 





if __name__ == '__main__':
    #run_grids_to_postgres()
    
    
    run_bldg_loss('DEU')
    
 

        