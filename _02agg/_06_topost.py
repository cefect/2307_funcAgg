'''
Created on Sep. 18, 2023

@author: cefect

port grid samples (05_sample) to postgis
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
    get_conn_str, pg_vacuum, pg_spatialIndex, pg_getCRS, pg_exe, pg_register, pg_comment
    )

from coms import (
    init_log, today_str, get_log_stream,  get_directory_size,
    dstr, view, get_filepaths, clean_geodataframe
    )

from definitions import wrk_dir, lib_dir, postgres_d, gridsize_default_l, temp_dir

#===============================================================================
# funcs
#===============================================================================


def to_post(schema, tableName, df, first, conn_str=None, log=None):
    """load a table to geopanbdas"""
    if conn_str is None: conn_str=get_conn_str(postgres_d)
 
    #===========================================================================
    # load
    #===========================================================================
 
    
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
        

    return 


def concat_on_hazard( index_df, log=None, out_dir=None, use_cache=True, dev=False):
    """concat a set agg samples on hazard key using the index frame for a single index grid"""
    
    """
    view(index_df)
    view(gdf)
    """
    
    #===========================================================================
    # defaultgs
    #===========================================================================
    if out_dir is None: out_dir=os.path.join(temp_dir, 'agg','topost', 'concat_haz_key')
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    if dev: use_cache=False
    #===========================================================================
    # get ofp
    #===========================================================================
    uuid = hashlib.shake_256(f'{index_df}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)    
    ofp = os.path.join(out_dir, f'concat_haz_{uuid}.pkl')
    log.debug(f'on {ofp}')
    #===========================================================================
    # build
    #===========================================================================
    if not os.path.exists(ofp) or (not use_cache):
        
        #loop on each hazard key and join
        first = True
        for haz_key, row in index_df.droplevel(1).iterrows():
            log.debug(haz_key)
            #take all the data from the first
            if first:
                df = gpd.read_file(row['ofp'], ignore_fields=['hazard_key'], ignore_geometry=True)
            else:
                df = gpd.read_file(row['ofp'], include_fields=[haz_key], ignore_geometry=True).join(df) #just the sample data
            first = False
        
        #clean columns for postgis
 
 
        df1 = df.rename(columns={k:'f'+k for k in index_df.index.unique('haz_key').tolist()})
        
        df1.loc[:, 'country_key'] = df['country_key'].str.lower()
        
        df1.to_pickle(ofp)

        log.debug(f'concated to get {str(df1.shape)} and wrote to \n    {ofp}')
        
    #===========================================================================
    # load
    #===========================================================================
    else:
        
        log.debug(f'using cache')
        df1 = pd.read_pickle(ofp)
        
        
    return df1

def run_agg_samps_to_post(
        
        country_key, 
       grid_size,
       haz_coln_l=None,
       
       #index_fp=None,
                               
 
        conn_str=None,
        schema='inters_grid', 
        dev=False,
        log=None,
 
        ):
    """merge hazard columns then add grids to postgis
    
    
    Returns
    -------
    
    postgres table inters_grid.agg_samps_{country_key}_{grid_size:04d}
    """
     
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()   
    
    country_key=country_key.lower()
    
    if dev:
        schema='dev' 
    
    if log is None: log = init_log(name=f'toPostG')
    #log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    if haz_coln_l is None: 
        from definitions import haz_coln_l
        haz_coln_l = [e[1:] for e in haz_coln_l] #remove the f prefix
    assert isinstance(haz_coln_l, list)
    keys_d=dict(country_key=country_key, grid_size=grid_size)
    log.info(f'on {keys_d}')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    #===========================================================================
    # setup table
    #===========================================================================
    tableName=f'agg_samps_{country_key}_{grid_size:04d}'
    
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
    #===========================================================================
    # build indexer
    #===========================================================================
    """because we have 1 index per hazard column"""
    root_fldr = '05_sample'
    if dev: root_fldr+='_dev'
    
    d=dict()
    base_dir = os.path.join(r'l:\10_IO\2307_funcAgg\outs\agg', root_fldr,  country_key.upper() )
    for haz_coln in haz_coln_l:

 
        search_dir = os.path.join(base_dir, haz_coln, f'{grid_size:04d}')
        index_fp = get_filepaths(search_dir, '_meta', ext='.gpkg')
        
        d[haz_coln] = gpd.read_file(index_fp, ignore_geometry=True)
        
    index_dx = pd.concat(d, names=['haz_key', 'grid_id'])
    """
    view(pd.concat(d))
    view(index_dx)
    """
    gid_l = index_dx.index.unique('grid_id').tolist()
    log.info(f'loaded index w/ {index_dx.shape} and {len(gid_l)} grids')
    
    #===========================================================================
    # upload each grid
    #===========================================================================
    first = True
    cnt=0
    for i, index_df in tqdm(index_dx.groupby('grid_id')):
        log.debug(f'on {i} w/ {index_df.shape}')
        
        #=======================================================================
        # concat on hazar dkeys
        #=======================================================================
        """drop all the redundant columns"""
        agg_samp_df = concat_on_hazard(index_df, log=log, dev=dev)
 
        
        #=======================================================================
        # load this into post
        #=======================================================================
        """using geopandas... could also use ogr2ogr"""
        to_post(schema, tableName, agg_samp_df, first, conn_str=conn_str, log=log)
        
        first=False
        cnt+=1
        
    #===========================================================================
    # post
    #===========================================================================
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY (country_key, grid_size, i,j)')
    
    #add comment
    cmt_str = f'port of {cnt} .gpkg sample files on grid centroids from {base_dir}\n'
    cmt_str += f'built with {os.path.realpath(__file__)} at '+datetime.now().strftime("%Y.%m.%d.%S")
    pg_comment(schema, tableName, cmt_str)
 
    
    #===========================================================================
    # clean
    #===========================================================================
    log.info(f'cleaning')
    try:
        pg_vacuum(schema, tableName)
        #pg_spatialIndex(schema, tableName, columnName='geometry')
        #pg_register(schema, tableName)
    except Exception as e:
        log.error(f'failed cleaning w/\n    {e}')
            
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on tableName for {len(index_dx)} \non {keys_d}')
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    #'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    return 
    
 

        
 
def run_all(ck, **kwargs):
    log = init_log(name='occu')
    
    for grid_size in gridsize_default_l:
        run_agg_samps_to_post(ck, grid_size, log=log, **kwargs)
    
    

if __name__ == '__main__':
    run_all('deu', dev=True)
    
    
    #run_agg_samps_to_post('deu', 1020, dev=True)
    
    
    
    
    