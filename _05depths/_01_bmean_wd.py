"""

mean asset loss per grid
"""


#===============================================================================
# IMPORTS-----
#===============================================================================
import os, hashlib, sys, subprocess, psutil
from datetime import datetime
from itertools import product

import psycopg2
from sqlalchemy import create_engine, URL
#print('psycopg2.__version__=' + psycopg2.__version__)



from tqdm import tqdm

import pandas as pd
import numpy as np


from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l, haz_label_d
    )


from coms import (
    init_log, today_str, get_directory_size,dstr, view,  
    ) 

from _02agg.coms_agg import (
    get_conn_str, pg_vacuum, pg_spatialIndex, pg_exe, pg_get_column_names, pg_register, pg_getcount,
    pg_comment, pg_table_exists, pg_get_nullcount
    )
 
from _02agg._07_views import create_view_join_grid_geom




def run_bldg_wd_group_stats(
        
        country_key, grid_size,
        agg_func='AVG',
 
        conn_str=None,
 
         log=None,
         dev=False,
         add_geom=False,
 
        ):
    """calc mean buildilng depths
    
 
        
    this script is similar to _03damage._03_rl_mean_bldg
    
    
    Returns
    -------
    postgres table
        inters_agg.wd_mean_{country_key}_{grid_size:04d} 

    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    start = datetime.now()   
    
    country_key = country_key.lower() 
 
    # log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
    if conn_str is None: conn_str = get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'wd_mean')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #===========================================================================
    # talbe params-------
    #===========================================================================
    #source table keys
    keys_d = { 
        'bldg':['country_key', 'gid', 'id'],
        'grid':['country_key', 'grid_size', 'i', 'j']        
    }
    
    if agg_func=='AVG':
        col_sfx='bmean'
    else:
        col_sfx = agg_func.replace('_', '').lower() 
        
        
    tableName=f'agg_wd_{col_sfx}_{country_key}_{grid_size:04d}' #output
 
    table_bldg = f'{country_key}' #building dephts
    
    #gid:bldg links for buildings in grids with some building expousre
    #see _04expo._01_full_links
    table_link =f'bldgs_grid_link_1x_{country_key}_{grid_size:04d}' 
    
    
    if dev: 
        schema='dev'
        schema_bldg=schema        
        schema_link=schema
        
    else:
        schema='inters_agg'
        schema_bldg='inters' 
        schema_link='expo'
 
    assert pg_table_exists(schema_bldg, table_bldg)
    assert pg_table_exists(schema_link, table_link), f'{schema_link}.{table_link} does not exist\n    see _04expo._01_full_links'
    #assert pg_table_exists(schema_grid, table_grid), f'{schema_grid}.{table_grid} does not exist'
    #===========================================================================
    # join buidling wd to grid links
    #===========================================================================
    #setup
 
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
    
    #build query
    
    g_cols = ', '.join([e for e in keys_d['grid']])
    
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_d['bldg']])  
    
    haz_cols = [e for e in pg_get_column_names(schema_bldg, table_bldg) if e.startswith('f')]
    cols = ', '.join([f'tleft.{e}' for e in keys_d['grid']]) + ', '
    cols+= ', '.join([f'tright.{e}' for e in haz_cols])
    
    
    cols2 = ', '.join(keys_d['grid'])+', '
    

    
    cols2+= ', '.join([f'CAST({agg_func}({e}) AS real) AS {e}_{col_sfx}' for e in haz_cols])
    
    
    #execute (using a sub-query)
    #the subquery is similar to _04expo._01_full_links.run_merge_expo_bldgs_wd()
    #but here we select only buildings with exposure to the particular grid size
    sql(f"""
    CREATE TABLE {schema}.{tableName} AS
        SELECT {cols2}
            FROM (SELECT {cols}
                    FROM {schema_link}.{table_link} as tleft
                    LEFT JOIN {schema_bldg}.{table_bldg} as tright
                        ON {link_cols}
                ) AS t
                    GROUP BY {g_cols}
    
    """) 
    #===========================================================================
    # post
    #===========================================================================
    
    keys_str = ', '.join(keys_d['grid'])
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
    
    assert pg_get_nullcount(schema, tableName, haz_cols[0]+f'_{col_sfx}')==0, f'bad link?'
 
    #add comment 
    source_d = dict(tableName=tableName, table_bldg=table_bldg, table_link=table_link )
    
    cmt_str = f'compute mean building depths (grouped by grids) \n from tables: {source_d}\n'
    cmt_str += f'built with {os.path.realpath(__file__)} at '+datetime.now().strftime("%Y.%m.%d: %H.%M.%S")
    pg_comment(schema, tableName, cmt_str)
    
    log.info(f'cleaning {tableName} ')
    
    try:
        pg_vacuum(schema, tableName)
        """table is a-spatial"""
        #pg_spatialIndex(schema, tableName, columnName='geometry')
        #pg_register(schema, tableName)
    except Exception as e:
        log.error(f'failed cleaning w/\n    {e}')
        
    #===========================================================================
    # add view to join grid depths
    #===========================================================================
    """would be useful for comparing centroid to bmean depths"""
    #params
    table_left = f'agg_samps_{country_key}_{grid_size:04d}'
    viewName = table_left+f'_j{col_sfx}'
    
    if dev:
        schema_left='dev'
    else:
        schema_left = f'inters_grid'
    
    #set up
    sql(f'DROP MATERIALIZED VIEW IF EXISTS {schema}.{viewName} CASCADE')
    
    #build query
    cols = 'tleft.*, '
    cols+= ', '.join([f'tright.{e}_{col_sfx}' for e in haz_cols])
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_d['grid']]) 
    
    #execute
    sql(f"""
        CREATE MATERIALIZED VIEW {schema}.{viewName} AS
            SELECT {cols}
                FROM {schema_left}.{table_left} as tleft
                    LEFT JOIN {schema}.{tableName} as tright
                        ON {link_cols}
                        """)
    
    #===========================================================================
    # add grid polygons to the view
    #===========================================================================
    if add_geom:
        create_view_join_grid_geom(schema, viewName, country_key, log=log, dev=dev, conn_str=conn_str)
        
    
    #===========================================================================
    # wrap
    #===========================================================================
        #meta
    meta_d = {
        'tdelta':(datetime.now() - start).total_seconds(), 
        'RAM_GB':psutil.virtual_memory()[3] / 1000000000, 
        'postgres_GB':get_directory_size(postgres_dir)}
        #'output_MB':os.path.getsize(ofp)/(1024**2)
    log.info(f'finished w/ {schema}.{tableName}\n{meta_d}')
    
    return tableName


def run_all(country_key='deu', grid_size_l=None, **kwargs):
    log = init_log(name='mean')
    if grid_size_l is None: grid_size_l = gridsize_default_l
    
    res_d = dict()
    for grid_size in grid_size_l: 
        res_d[grid_size] = run_bldg_wd_group_stats(country_key, grid_size, log=log, **kwargs)
        
    log.info(f'finished w/ \n{dstr(res_d)}')
    
    return res_d
    
 

def get_grid_wd_dx(
        country_key, haz_key,
 
        log=None,conn_str=None,dev=False,use_cache=True,out_dir=None,
        limit=None,
 
        ):
    
    """helper to retrieve results from run_view_join_depths() as a dx
    
     WARNING: this relies on _04expo.create_view_join_stats_to_rl()
    """ 
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'damage','05_means', country_key, haz_key)
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
        #see _04expo._03_views.create_view_join_stats_to_rl()
        tableName = f'grid_rl_wd_bstats_{country_key}_{haz_key}' 
        
        if dev:
            schema = 'dev'
    
        else:
            schema = 'damage' 
            
        assert pg_table_exists(schema, tableName, asset_type='matview'), f'missing table dependency \'{tableName}\'\n    see _04expo._03_views.create_view_join_stats_to_rl()'
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
            'bldg':bldg_dx, 
            'grid':grid_dx, 
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
    run_all( dev=True)
    #run_bldg_wd_group_stats('deu', 1020, dev=True, add_geom=False)
    
 
    
    #run_extract_haz('deu', 'f500_fluvial', dev=False)
    
        
    
