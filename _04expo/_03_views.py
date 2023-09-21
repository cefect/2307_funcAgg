'''
Created on Sep. 21, 2023

@author: cefect
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
 
 


from concurrent.futures import ProcessPoolExecutor

import psycopg2
from sqlalchemy import create_engine, URL

from tqdm import tqdm



from definitions import (
    wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d, postgres_d, 
    equal_area_epsg, fathom_vals_d, gridsize_default_l
    )
from definitions import temp_dir as temp_dirM
 


from _02agg.coms_agg import (
    get_conn_str, pg_getCRS, pg_to_df, pg_exe, pg_getcount, pg_spatialIndex, pg_get_column_names,
    pg_vacuum, pg_comment, pg_register
    )

from _02agg._07_views import create_view_join_grid_geom

from coms import (
    init_log, today_str,  dstr, view
    )


def create_view_merge_stats(country_key, haz_key,
                         grid_size_l=None,
         dev=False, conn_str=None,
         with_geom=False,
        ):
    
    """create a view by unioning all the grid stats and selecting wet_cnts from one haz
    
    useful for joins later"""
    

    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()  
    
    log = init_log(name=f'view')
    
    if grid_size_l is None: grid_size_l = gridsize_default_l
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #===========================================================================
    # table params
    #===========================================================================
    keys_l = ['country_key', 'grid_size', 'i', 'j', 'haz_key']
    if dev:
        schema = 'dev'
    else:
        schema='expo'
 
    tableName=f'grid_bldg_stats_{country_key}_{haz_key}'
    
    #===========================================================================
    # query
    #===========================================================================
    sql(f'DROP VIEW IF EXISTS {schema}.{tableName}')
    
    cmd_str = f'CREATE VIEW {schema}.{tableName} AS \n'
    
    first = True
    for grid_size in grid_size_l:        
        tableName_i = f'grid_bldg_stats_{country_key}_{grid_size:04d}'
                
        if not first:
            cmd_str += 'UNION\n'
        else:
            #build column selection (only want 1 hazard)
            full_coln_l = pg_get_column_names(schema, tableName_i)
            haz_col = [e for e in full_coln_l  if haz_key in e][0] #column name of wet counts we want
            keep_coln_l = [e for e in full_coln_l  if not e.endswith('wetcnt')] #other columns            
            
            #assemble
            cols = ', '.join(keep_coln_l)
            cols+=f', {haz_col} as wet_cnt'
 
        
        cmd_str += f'SELECT {cols}\n'
        cmd_str += f'FROM {schema}.{tableName_i} \n'
        
        """no haz_key column. each row has wet stats for each haz key
        cmd_str += f'WHERE {tableName_i}.haz_key=\'{haz_key}\' \n'"""
        
        
        # filters        
        first = False
    
    cmd_str+=f'ORDER BY grid_size, i, j\n'
    sql(cmd_str)
    
    #===========================================================================
    # join geometry
    #===========================================================================
    if with_geom:
        create_view_join_grid_geom(schema, tableName, country_key, log=log, dev=dev, conn_str=conn_str)
    
    #===========================================================================
    # wrap
    #===========================================================================
        #meta
    meta_d = {
        'tdelta':(datetime.now() - start).total_seconds(), 
        'RAM_GB':psutil.virtual_memory()[3] / 1000000000, 
        #'postgres_GB':get_directory_size(postgres_dir)}
        #'output_MB':os.path.getsize(ofp)/(1024**2)
        }
    log.info(f'finished on {tableName} w/ \n{meta_d}')
    
    return tableName


def create_view_join_stats_to_rl(
        country_key, haz_key,
 
        log=None,
        conn_str=None,
        dev=False,
 
 
        with_geom=False,
        ):
    """merge rl, rl_mean, wd views (run_view_join_depths()) with expo stats (create_view_merge_stats())"""
        

    #if grid_size_l is None: grid_size_l = gridsize_default_l
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    start = datetime.now()   
    
    country_key = country_key.lower() 
 
    if conn_str is None: conn_str = get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'jstats')
        
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #===========================================================================
    # talbe params
    #===========================================================================
 
    #grids w/ grid RL, mean RL, grid WD ... see run_view_join_depths()
    table_left=f'rl_mean_grid_{country_key}_{haz_key}_wd'
    
    #grids w/ bldg_cnt and wet_cnt
    table_right=f'grid_bldg_stats_{country_key}_{haz_key}' #wd for all grids. see _02agg._07_views()
    
    tableName = f'grid_rl_wd_bstats_{country_key}_{haz_key}'
 
    
    if dev:
        schema='dev'
        schema_left, schema_right=schema, schema        

    else:
        schema='damage'  #makes more sense here I think
        schema_left='damage'
        schema_right='expo'       
        
    
    keys_l = ['country_key', 'grid_size', 'i', 'j']
 
    
    #===========================================================================
    # join depths
    #===========================================================================
    sql(f'DROP MATERIALIZED VIEW IF EXISTS {schema}.{tableName}')    
    
    
    cmd_str = f'CREATE MATERIALIZED VIEW {schema}.{tableName} AS \n'
    
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_l]) 
    cmd_str+= f"""
        SELECT tleft.*, tright.bldg_cnt, tright.wet_cnt
            FROM {schema_left}.{table_left} AS tleft
                LEFT JOIN {schema_right}.{table_right} AS tright
                    ON {link_cols}
            """
    
    sql(cmd_str)
    
    #===========================================================================
    # join geometry
    #===========================================================================
    if with_geom:
        create_view_join_grid_geom(schema, tableName, country_key, log=log, dev=dev, conn_str=conn_str)
    
    
    
    
    #===========================================================================
    # wrap
    #===========================================================================
        #meta
    meta_d = {
        'tdelta':(datetime.now() - start).total_seconds(), 
        'RAM_GB':psutil.virtual_memory()[3] / 1000000000, 
        #'postgres_GB':get_directory_size(postgres_dir)}
        #'output_MB':os.path.getsize(ofp)/(1024**2)
        }
    log.info(f'finished on {tableName} w/ \n{meta_d}')
    
    return tableName




def get_grid_rl_dx(
        country_key, haz_key,
 
        log=None,
        conn_str=None,
        dev=False,
        use_cache=True,
 
        out_dir=None,
        limit=None,
 
        ):
    
    """helper to retrieve results from run_view_join_depths() as a dx
    
 
    """ 
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'expo','03_views', country_key, haz_key)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    
    if log is None:
        log = init_log(name=f'grid_rl')
    
    if dev: use_cache=False
    #===========================================================================
    # cache
    #===========================================================================
    fnstr = f'grid_rl_{country_key}_{haz_key}'
    uuid = hashlib.shake_256(f'{fnstr}_{dev}_{limit}'.encode("utf-8"), usedforsecurity=False).hexdigest(8)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')
    
    if (not os.path.exists(ofp)) or (not use_cache):
        
        #===========================================================================
        # talbe params
        #===========================================================================
        #see create_view_join_stats_to_rl()
        tableName = f'grid_rl_wd_bstats_{country_key}_{haz_key}' 
        
        if dev:
            schema = 'dev'
    
        else:
            schema = 'damage' 
            
        keys_l = ['country_key', 'grid_size','haz_key', 'i', 'j']
        
        #===========================================================================
        # download
        #===========================================================================
        conn =  psycopg2.connect(conn_str)
        engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
        
        #row_cnt=0
        
        """only ~600k rows"""
        
        cmd_str = f'SELECT * FROM {schema}.{tableName}'
        
        if not limit is None:
            cmd_str+=f'\n    LIMIT {limit}'
 
        log.info(cmd_str)
        df_raw = pd.read_sql(cmd_str, engine, index_col=keys_l)
        """
        view(df_raw.head(100))        
        """    
        
        engine.dispose()
        conn.close()
        
        log.info(f'finished w/ {len(df_raw)} total rows')
        
        #===========================================================================
        # clean up
        #===========================================================================
        #exposure meta
        expo_colns = ['bldg_expo_cnt', 'grid_wd', 'bldg_cnt', 'wet_cnt']
        df1 = df_raw.copy()
        df1.loc[:, expo_colns] = df1.loc[:, expo_colns].fillna(0.0)        
        df1=df1.set_index(expo_colns, append=True)
        
        #split bldg and grid losses
        col_bx = df1.columns.str.contains('_mean') 
        
        grid_dx = df1.loc[:, ~col_bx]
        rnm_d = {k:int(k.split('_')[1]) for k in grid_dx.columns.values}
        grid_dx = grid_dx.rename(columns=rnm_d).sort_index(axis=1)
        grid_dx.columns = grid_dx.columns.astype(int)
        
        
        bldg_dx = df1.loc[:, col_bx]
        rnm_d = {k:int(k.split('_')[1]) for k in bldg_dx.columns.values}
        bldg_dx = bldg_dx.rename(columns=rnm_d).sort_index(axis=1)
        bldg_dx.columns = bldg_dx.columns.astype(int)
        
        assert np.array_equal(grid_dx.columns, bldg_dx.columns)
     
        
        dx = pd.concat({
            'bldg_mean':bldg_dx, 
            'grid_cent':grid_dx, 
            #'expo':df.loc[:, expo_colns].fillna(0.0)
            }, 
            names = ['rl_type', 'df_id'], axis=1).dropna(how='all') 
        
        #===========================================================================
        # write
        #===========================================================================
        """
        view(dx.head(100))
        """
        
 
        log.info(f'writing {dx.shape} to \n    {ofp}')
        dx.sort_index(sort_remaining=True).sort_index(sort_remaining=True, axis=1).to_pickle(ofp)
    
    else:
        log.info(f'loading from cache:\n    {ofp}')
        dx = pd.read_pickle(ofp)
 
 
    log.info(f'got {dx.shape}')
    return dx


if __name__ == '__main__':
    
    #create_view_join_stats_to_rl('deu', 'f500_fluvial', dev=True, with_geom=False)
    
    get_grid_rl_dx('deu', 'f500_fluvial', dev=True, limit=None)



    
 