'''
Created on Sep. 18, 2023

@author: cefect

port relative losses to postgis
'''
#===============================================================================
# IMPORTS-------
#===============================================================================
import os, hashlib, sys, subprocess
import psutil
from datetime import datetime


import numpy as np
from numpy import dtype
import pandas as pd
import geopandas as gpd

import psycopg2
print('psycopg2.__version__=' + psycopg2.__version__)

 

from sqlalchemy import create_engine, URL

from tqdm import tqdm

from _02agg.coms_agg import (
    get_conn_str, pg_vacuum, pg_spatialIndex, pg_getCRS, pg_exe, pg_register, pg_getcount,
    pg_comment
    )

from coms import (
    init_log, today_str, get_log_stream,  get_directory_size,
    dstr, view, get_filepaths, clean_geodataframe
    )

from definitions import wrk_dir, lib_dir, postgres_d, index_country_fp_d, temp_dir, gridsize_default_l

#===============================================================================
# funcs
#===============================================================================


def df_to_sql(schema, tableName, df, first, conn_str=None, log=None):
    """load a dataframe into a postgres table"""
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    start = datetime.now()
 
    assert isinstance(df, pd.DataFrame)
    #===========================================================================
    # execute
    #===========================================================================
    conn =  psycopg2.connect(conn_str)
    #set engine for geopandas
    engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
    try: 
        
         
        log.debug(f'porting {df.shape} to {tableName}')
        if_exists = 'replace' if first else 'append'
         
        df.to_sql(tableName, engine, schema=schema, 
                                           if_exists=if_exists, 
                                           index=False, 
                                           #index_label=dxcol.index.names,
                                           )
                
                
        
    except Exception as e:
        raise IOError(f'failed query w/ \n    {e}')
    finally:
        # Dispose the engine to close all connections
        engine.dispose()
        # Close the connection
        conn.close()
        

    log.debug(f'finished in %.2f secs'%((datetime.now()-start).total_seconds()))
    return 


 

def rl_to_post(
        
        country_key, asset_schema, asset_tableName,
 
       haz_coln_l=None,
       search_dir=None,
 
        conn_str=None,
 
        schema='damage',
         log=None,
         dev=False,
 
        ):
    """concat loss chunk .pkls and port to post
    
    NOTE: see da_loss.plot_rl_raw for some diagnostic plots
    
    
    Returns
    ---------
    table:
        damage.rl_{country_key}_{asset_type}
        """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()    
    country_key=country_key.lower() 
 
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    
    #hazard columns
    if haz_coln_l is None: 
        from definitions import haz_coln_l
        haz_coln_l = [e[1:] for e in haz_coln_l] #remove the f prefix        
    assert isinstance(haz_coln_l, list)
    
    #asset type
    asset_type = {'inters':'bldgs', 'inters_grid':'grid', 'inters_agg':'grid_bmean', 'expo':'bldgs'}[asset_schema]
    
    keys_d=dict(country_key=country_key, asset_type=asset_type)
    log.info(f'on {keys_d}')
    
    if 'grid' in asset_type:
        grid_size=int(asset_tableName.split('_')[-1])
        keys_d['grid_size'] = grid_size
        keys_l = ['country_key', 'grid_size', 'i', 'j', 'haz_key']
    #===========================================================================
    # elif asset_type=='grid_bmean':
    #     grid_size=int(asset_tableName.split('_')[-1])
    #     keys_d['grid_size'] = grid_size
    #     keys_l = ['country_key', 'grid_size', 'i', 'j', 'haz_key']
    #===========================================================================
        
    elif asset_type=='bldgs':
        keys_l = ['country_key', 'gid', 'id', 'haz_key']

    if dev:
        schema='dev'
        asset_schema='dev'
    #===========================================================================
    # load indexers
    #===========================================================================
    """because we have 1 index per hazard column"""
 
    
    if search_dir is None: 
        search_dir = os.path.join(r'l:\10_IO\2307_funcAgg\outs\damage\01_rl', asset_schema, asset_tableName)
        
    #collect metas
    meta_fp = get_filepaths(search_dir, '_meta', ext='.pkl', recursive=True, single=True)    
    
    #['chunk', 'country_key', 'asset_schema', 'tableName', 'haz_key']
    serx = pd.read_pickle(meta_fp).stack().droplevel('out_dir').rename('fp')
 
    #base directory for relative pathing
    base_dir = os.path.dirname(meta_fp)
 
    chunk_l = serx.index.unique('chunk').tolist()
    log.info(f'loaded index w/ {serx.shape} and {len(chunk_l)} grids for \n    {keys_d}')
    
    #===========================================================================
    # setup table
    #===========================================================================
    tableName=f'rl_{country_key}_{asset_type}'
    
    if 'grid' in asset_type:
        tableName+=f'_{grid_size:04d}'

    
    """writing all hazard keys to one table"""
    
    pg_exe(f'DROP TABLE IF EXISTS {schema}.{tableName}')
        
    
    #===========================================================================
    # createa a table for each hazard key
    #===========================================================================
    cnt=0
    first=True
    for haz_key, gserx in serx.groupby('haz_key'):
        log.info(f'{tableName} on {haz_key} w/ {gserx.shape}')

        #=======================================================================
        # loop and upload each
        #=======================================================================
        
        for keys, fileName in tqdm(gserx.items()):
            data_fp = os.path.join(base_dir, fileName)
            log.debug(f' for {keys} loading from \n    {data_fp} ')
            
            #===================================================================
            # #load
            #===================================================================
            df_raw = pd.read_pickle(data_fp)
            
            #===================================================================
            # #clean up
            #===================================================================
            df1 = df_raw.copy().round(1).astype(np.float32)
            df1 = df1.rename(columns={e:f'dfid_{e:04d}' for e in df1.columns})
            df1.columns.name=None
            
            df2 = df1.reset_index()
            
            if 'id' in df2.columns:
                df2 = df2.sort_values('id')
            
            df2.loc[:, 'country_key'] = df2['country_key'].str.lower()
 
            #===================================================================
            # port
            #===================================================================
            df_to_sql(schema, tableName, df2, first, conn_str=conn_str, log=log.getChild(haz_key)) 
 
        
            first=False
            cnt+=1
        
        log.info(f'finished {haz_key}\n\n')
        
 

    #===========================================================================
    # clean-----
    #===========================================================================
    log.info(f'cleaning {schema}.{tableName}')
    
 
    
    keys_str = ', '.join(keys_l)
    pg_exe(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
    
    cmt_str = f'port of {cnt} .pkl rl results for \'{asset_type}\' loaded from {base_dir}\n'
    cmt_str += f'built with {os.path.realpath(__file__)} at '+datetime.now().strftime("%Y.%m.%d: %H.%M.%S")
    pg_comment(schema, tableName, cmt_str)
    
    log.info(f'cleaning {tableName} w/ {pg_getcount(schema, tableName)} rows')
    try:
        pg_vacuum(schema, tableName)
        """table is a-spatial"""
        #pg_spatialIndex(schema, tableName, columnName='geometry')
        #pg_register(schema, tableName)
    except Exception as e:
        log.error(f'failed cleaning w/\n    {e}')
            
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on \'{tableName}\'')
    
    meta_d = {
                    'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    #'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    return  tableName
    
def run_agg_rl_topost(country_key, grid_size_l=None, 
                      sample_type='bldg_mean', log=None,
                      **kwargs):
    
    if grid_size_l is None: grid_size_l=gridsize_default_l
    if log is None: log = init_log(name=f'rlAgg')
    
 
    d=dict()
    log.info(f'on {len(grid_size_l)} grids')
    for grid_size in grid_size_l:
        #=======================================================================
        # get params forthis type
        #=======================================================================
        if sample_type=='grid_cent':
            asset_schema='inters_grid'
            tableName=f'agg_samps_{country_key}_{grid_size:04d}'
            
        elif sample_type=='bldg_mean':
            asset_schema = 'inters_agg'
            tableName=f'agg_wd_bmean_{country_key}_{grid_size:04d}'
        else:
            raise IOError(sample_type)
    
        #=======================================================================
        # run
        #=======================================================================
        d[grid_size] = rl_to_post(country_key, asset_schema, tableName, 
                   log=log.getChild(str(grid_size)), **kwargs)
        
    log.info(f'finished w/ \n    {d}')
    
    return d

        
def run_bldg_rl_topost(country_key, filter_cent_expo=False, log=None, **kwargs):
    """port all the rl pickles to one table 'damage.rl_deu_bldgs'
    this now contains many zeros
    smaller grid sizes have many unlinked neighbours (as our selection was from the 1020)
    
    """
    if log is None: log = init_log(name=f'rlBldg')
    
    #select source table  by exposure filter strategy
    if filter_cent_expo:
        #wd for doubly exposed grids w/ polygon geometry. see _02agg._07_views.run_view_grid_wd_wgeo()
 
        expo_str = '2x'
    else:
        #those grids with building exposure (using similar layer as for the centroid sampler) 
        expo_str = '1x'
    
    asset_schema='expo'
    tableName=f'bldgs_grid_link_{expo_str}_{country_key}_bwd'
    
    return rl_to_post(country_key, asset_schema,tableName, log=log, **kwargs)
        
    
def run_all(country_key='deu',   **kwargs):
    
 
    log = init_log(name=f'rl_topost')
    
    #===========================================================================
    # buildings
    #===========================================================================
    run_bldg_rl_topost(country_key, log=log, **kwargs)
    
    run_agg_rl_topost(country_key, log=log, **kwargs)
    
    

if __name__ == '__main__':
 
    
    #run_bldg_rl_topost('deu', dev=True)
    #run_agg_rl_topost('deu', dev=True)
    
    
    
    
    run_all(dev=True) #takes a few mins
    
    
    
    
    
    
    
    print('done')
    
    
    
    