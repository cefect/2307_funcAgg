'''
Created on Sep. 18, 2023

@author: cefect

compute losses from depths using depth-damage functions
'''


#===============================================================================
# IMPORTS--------
#===============================================================================
import os, hashlib, sys, subprocess, copy

 
 
import psutil
from datetime import datetime
import pandas as pd
import numpy as np

#===============================================================================
# from osgeo import ogr
# import fiona
# import shapely.geometry
# import geopandas as gpd
#===============================================================================

#import rasterstats
#from rasterstats import zonal_stats

#===============================================================================
# import concurrent.futures
# from concurrent.futures import ProcessPoolExecutor
#===============================================================================

import psycopg2
from sqlalchemy import create_engine, URL

from tqdm import tqdm

from coms import (
    view, clean_geodataframe, pd_ser_meta, init_log_worker, init_log, today_str,
    get_directory_size,
    )


from definitions import (
    wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d, postgres_d, 
    equal_area_epsg, fathom_vals_d, gridsize_default_l
    )


from _03damage.coms_dmg import get_rloss
from funcMetrics.func_prep import get_funcLib
from funcMetrics.coms_fm import slice_serx, force_max_depth, force_zero_zero, force_monotonic

from _02agg.coms_agg import get_conn_str, pg_getCRS, pg_exe





def write_loss_haz_chunk(ser, func_d, wd_scale, out_dir, fnstr, log=None, use_cache=False, dev=False):
    """compute loss for this hazard chunk on all functions"""
    #===========================================================================
    # defaults
    #===========================================================================
    
    log.debug(f'w/ {ser.shape} and  {len(func_d)} funcs')
    
    assert isinstance(ser, pd.Series)
    
    if dev: use_cache=False
    #===========================================================================
    # get ofp
    #===========================================================================
    uuid = hashlib.shake_256(f'{func_d}_{ser.name}_{ser.index}_{wd_scale}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')
    
    #===========================================================================
    # build
    #===========================================================================
    if not os.path.exists(ofp) or (not use_cache):
        log.debug(f'building on {len(ser)}')
        wd_ar = ser.values * wd_scale
        #=======================================================================
        # loop on each function
        #=======================================================================
        d = dict()
        for df_id, dd_ar in func_d.items():
 
            log.debug(df_id)
            #get loss
 
            try:
                loss_ar = get_rloss(dd_ar.copy(), wd_ar, prec=None) #depths in table are rounded enough
            except Exception as e:
                raise IOError(f'failed to compute losse w/ {df_id}\n    {e}')
 
            #loss_ar = np.full(len(wd_ar), float(df_id))
            
            #append index and collect
            d[df_id] = pd.Series(loss_ar, index=ser.index, name=df_id, dtype=np.float32)
            
            #print(d[df_id].to_dict())
    
        #===================================================================
        # #collect and wirte
        #===================================================================
        loss_df = pd.concat(d, axis=1)
        """
        view(loss_df)
        """
        
        #add indexer
        loss_df['haz_key'] = ser.name
        loss_df=loss_df.set_index('haz_key', append=True).swaplevel(-1, -2).swaplevel(-2, -3)
        loss_df.columns.name='df_id'
        
        #write
        loss_df.to_pickle(ofp)        
        log.debug(f'wrote {loss_df.shape} to \n    {ofp}')
        
    #===========================================================================
    # cache
    #===========================================================================
    else:
        log.info(f'file exists... skipping')
        log.debug(ofp)
        
    return ofp

def loss_calc_country_assetType(
        
        country_key,
        asset_schema='inters',
        tableName=None,        
        max_depth=None,
        fserx=None,
       haz_coln_l=None,
       wd_scale=0.01, 
       
       
 
       
       #index_fp=None,
                               
        out_dir=None,
        conn_str=None,
        #schema='damage', 
        chunksize=1e5,
        log=None,
        dev=False,
 
        ):
    """use asset wd to compute losses for each function and each hazard
    one grid_size (or bldgs)
    
    goes pretty fast acutally with the zero filter
    
 
    
    Params
    ------
    asset_schema: str
        name of schema containing asset depth samples
        
    tableName: str
        name of table with asset depth samples 
        
    fsers: pd.Series
        relative loss functions cleaned and prepped
        
    wd_scale: float
        scaling wd values (to convert to m)
        
    Writes
    ----------
    pd.Series to pickle
        index: asset id
        columns: loss for each function on one hazard
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()   
    
    country_key=country_key.lower()
    
    #asset data
    if tableName is None:
        #only the buildings have simple table names
        if  asset_schema=='inters':
            tableName=country_key
            
    assert isinstance(tableName, str)
    
    asset_type = {'inters':'bldgs', 'inters_grid':'grid', 'inters_agg':'grid'}[asset_schema]
    
    if dev:
        asset_schema='dev'
    
 
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'damage','01_rl', asset_schema, tableName)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    #log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    if haz_coln_l is None: 
        from definitions import haz_coln_l
        #haz_coln_l = [e[1:] for e in haz_coln_l] #remove the f prefix
        
        
    assert isinstance(haz_coln_l, list)
    
    
    if max_depth is None:
        from funcMetrics.coms_fm import max_depth
        
    
    
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
    func_d = dict()    
    for df_id, gserx in fserx_s.groupby('df_id'):
        func_d[df_id] = gserx.droplevel(list(range(gserx.index.nlevels-1))).reset_index().T.astype(float).values
 
    #===========================================================================
    # loop on hazards
    #===========================================================================
    """looping on haz columns gives more control and allows more filtering (e.g., faster)"""
    cnt=0
    res_lib=dict()
    log.info(f'computing on {len(haz_coln_l)} hazards for {country_key} from {tableName}')
    for haz_coln in haz_coln_l:
        
        log.info(f'on {haz_coln} w/ {len(func_d)} funcs and chunksize={chunksize}')
        #===========================================================================
        # loop and load from postgis
        #===========    ================================================================
        """could do everything in postgis... but I'm not sure this is faster and I'd have to figure it out"""
        conn =  psycopg2.connect(conn_str)
        engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
        
        #=======================================================================
        # build query
        #=======================================================================
        if asset_type=='bldgs':            
            index_col=['country_key','gid','id']
        elif asset_type=='grid': 
            index_col=['country_key','grid_size', 'i', 'j']
        else:
            raise IOError(asset_schema)
        
        cmd_str = f'SELECT ' + ', '.join(index_col) +f', {haz_coln}'        
        cmd_str +=f'\nFROM {asset_schema}.{tableName}'         
        cmd_str += f'\nWHERE {haz_coln} >0 AND {haz_coln} IS NOT NULL' #exclude empties
        cmd_str +=f'\nORDER BY '+ ', '.join(index_col) #needed to get consistent pulls?
        #=======================================================================
        # #loop through chunks of the table
        # cmd_str = f'SELECT id, ' + ', '.join(haz_coln_l) + f' \nFROM inters.{country_key}'
        # 
        # #only non-zeros
        # cmd_str+=f'WHERE '+ ' >0 AND '.join(haz_coln_l)
        #=======================================================================
 
        log.info(cmd_str)
        res_d=dict()        
        #=======================================================================
        # chunk loop
        #=======================================================================
        for i, gdf in enumerate(pd.read_sql(cmd_str, engine, index_col=index_col, chunksize=int(chunksize), dtype={haz_coln:np.float32})):
            log.info(f'{i} on {gdf.shape}')
            
            if len(gdf)==0:
                log.warning(f'for chunk {i} got no rows... skipping') #not sure why this would happen
                continue
            assert len(gdf.columns)==1
            fnstr = f'rl_{country_key}_{haz_coln}_{i:03d}'
            
            ser = gdf.iloc[:,0].astype(np.float32)
            ser = ser.rename(ser.name.replace('_bmean', ''))
            
            res_d[i] = write_loss_haz_chunk(ser, copy.deepcopy(func_d), wd_scale, out_dir, fnstr,log=log, dev=dev)
 
            cnt+=1
            
        #wrap haz_coln loop
        log.info(f'finished {haz_coln} w/ {len(res_d)}\n\n')
        res_lib[haz_coln] = {k:os.path.basename(v) for k,v in res_d.items()}
        
        engine.dispose()
        conn.close()
 
    #===========================================================================
    # write meta
    #===========================================================================
    log.debug('meta')
    meta_df = pd.DataFrame.from_dict(res_lib)
    
    meta_d = dict(country_key=country_key, asset_schema=asset_schema, tableName=tableName, out_dir=out_dir)
    for k,v in meta_d.items():
        meta_df[k]=v
        
    meta_df = meta_df.set_index(list(meta_d.keys()), append=True)
    meta_df.index.set_names('chunk', level=0, inplace=True)
    meta_df.columns.name='haz_key'
    
    ofp = os.path.join(out_dir, f'_meta_rl_{country_key}_{asset_schema}_{tableName}_{len(meta_df):03d}_{today_str}.pkl')
    meta_df.to_pickle(ofp)
    log.info(f'wrote meta {meta_df.shape} to \n    {ofp}')
 
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


def run_agg_loss(country_key, grid_size_l=None, sample_type='grid_cent', **kwargs):
    """compute losses from agg grid centroid samples"""
    if grid_size_l is None: grid_size_l=gridsize_default_l
    log = init_log(name=f'rlAgg')
    
    

    
    d=dict()
    log.info(f'on {len(grid_size_l)} grids')
    for grid_size in grid_size_l:
        
        
        #===========================================================================
        # set source table based on sampling type
        #===========================================================================
        if sample_type=='grid_cent':
            asset_schema='inters_grid'
            tableName=f'agg_samps_{country_key}_{grid_size:04d}'
            haz_coln_l=None
            
        elif sample_type=='bldg_mean':
            asset_schema = 'inters_agg'
            tableName=f'agg_wd_bmean_{country_key}_{grid_size:04d}'
            haz_coln_l=['f010_fluvial_bmean', 'f050_fluvial_bmean', 'f100_fluvial_bmean', 'f500_fluvial_bmean']
 
        #=======================================================================
        # run
        #=======================================================================
        d[grid_size] = loss_calc_country_assetType(country_key,asset_schema=asset_schema, 
                                                   tableName=tableName, log=log,
                                                   haz_coln_l=haz_coln_l,
                                                   **kwargs)
        
    log.info(f'finished on {len(d)}')
    
    return d

def run_bldg_loss(*args, **kwargs):
    return loss_calc_country_assetType(*args, log = init_log(name=f'rlBldg'), **kwargs)


if __name__ == '__main__':
 
    
    
    #run_bldg_loss('deu', dev=False)
    
 
    run_agg_loss('deu', dev=False, sample_type='bldg_mean')
    
    
    print('done')
 

        