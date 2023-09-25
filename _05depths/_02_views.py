'''
Created on Sep. 24, 2023

@author: cefect


see also _04expo._03_views
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
 
import psycopg2
#print('psycopg2.__version__=' + psycopg2.__version__)
from sqlalchemy import create_engine, URL

 

from definitions import (
    wrk_dir,   postgres_d, 
    equal_area_epsg, fathom_vals_d, gridsize_default_l
    )
 

from _02agg.coms_agg import (
    get_conn_str, pg_getCRS, pg_to_df, pg_exe, pg_getcount, pg_spatialIndex, pg_get_column_names,
    pg_vacuum, pg_comment, pg_register, pg_table_exists
    )

from _02agg._07_views import create_view_join_grid_geom

from coms import (
    init_log, today_str,  dstr, view
    )


def create_view_merge_bmeans(country_key='deu', 
                             #haz_key='f500_fluvial',
                         grid_size_l=None,
         dev=False,
          conn_str=None,
         with_geom=False,
         log=None,
         include_gcent=False,
        ):
    
    """create a view by unioning all the child building mean depths
    
    useful for joins later"""
    

    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()  
    
    if log is None: log = init_log(name=f'view')
    
    if grid_size_l is None: grid_size_l = gridsize_default_l
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #===========================================================================
    # table params
    #===========================================================================
    keys_l = ['country_key', 'grid_size', 'i', 'j', 'haz_key']

 
 
    #both options are created with _05depths._01_bmean_wd.run_bldg_wd_means()
    if include_gcent:
        tableName=f'agg_samps_{country_key}_jbmean'
        schema = f'inters_agg'
        asset_type='matview'
    else:
        #just the child mean depths
        tableName=f'agg_wd_bmean_{country_key}'
        schema='inters_agg'
        asset_type='table'
    
    if dev:
        schema = 'dev'
    

    #===========================================================================
    # query
    #===========================================================================
    sql(f'DROP VIEW IF EXISTS {schema}.{tableName} CASCADE')
    
    cmd_str = f'CREATE VIEW {schema}.{tableName} AS \n'
    
    first = True
    for grid_size in grid_size_l:
        if include_gcent:
            tableName_i = f'agg_samps_{country_key}_{grid_size:04d}_jbmean'
        else:  
            tableName_i = f'agg_wd_bmean_{country_key}_{grid_size:04d}'
            
        assert pg_table_exists(schema, tableName_i, asset_type=asset_type), f'missing {schema}.{tableName_i}'
 
        
                
        if not first:
            cmd_str += 'UNION\n'
        else: 
            cols = '*'
 
        
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
    log.info(f'finished on {schema}.{tableName} w/ \n{meta_d}\n\n')
    
    return tableName


def create_view_join_stats_to_bmeans(
        country_key='deu', haz_key='f500_fluvial',
 
        log=None,
        conn_str=None,
        dev=False,
 
 
        with_geom=False,
        ):
    """building means with expo stats (create_view_merge_stats())"""
        

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
 
    #grids w/ centroid wd, bmean wd, see create_view_merge_bmeans()
    table_left=f'agg_samps_{country_key}_jbmean'
    
    #grids w/ bldg_cnt and wet_cnt
    #see _02agg._07_views.create_view_merge_stats()
    table_right=f'grid_bldg_stats_{country_key}_{haz_key}' 
    
    tableName = f'grid_wd_bmean_bstats_{country_key}_{haz_key}'
 
    
    if dev:
        schema='dev'
        schema_left, schema_right=schema, schema        

    else:
        schema='inters_agg'   
        schema_left='inters_agg'
        schema_right='expo'       
        
    
    #check dependencies
    assert pg_table_exists(schema_left, table_left, asset_type='matview'), \
        f'missing left: %s.%s'%(schema_left, table_left)
        
    assert pg_table_exists(schema_right, table_right, asset_type='table'), \
        f'missing right: %s.%s'%(schema_right, table_right)
        
    keys_l = ['country_key', 'grid_size', 'i', 'j']
 
    log.info(f'creating view from {table_left} and {table_right}')
    #===========================================================================
    # join depths
    #===========================================================================
    sql(f'DROP MATERIALIZED VIEW IF EXISTS {schema}.{tableName} CASCADE')    
    
    #prep
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_l])
    
    cols = ', '.join([f'tleft.{e}' for e in keys_l])
    cols +=f', tleft.{haz_key} as grid_wd'
    cols +=f', tleft.{haz_key}_bmean as bmean_wd'
    cols += f', tright.bldg_cnt, tright.wet_cnt' 
    cols += f', \'{haz_key}\' as haz_key'
    
    cmd_str = f"""
    CREATE MATERIALIZED VIEW {schema}.{tableName} AS
        SELECT {cols}
            FROM {schema_left}.{table_left} AS tleft
                LEFT JOIN {schema_right}.{table_right} AS tright
                    ON {link_cols}
            """
    
    
    #exe
    sql(cmd_str)
    
    
    #===========================================================================
    # #===========================================================================
    # # add haz key
    # #===========================================================================
    # sql(f"""ALTER MATERIALIZED VIEW {schema}.{tableName} ADD COLUMN haz_key VARCHAR(255) DEFAULT \'{haz_key}\'""")
    # sql(f'REFRESH MATERIALIZED VIEW {schema}.{tableName}')
    #===========================================================================
    
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


def get_grid_wd_dx(
        country_key='deu', haz_key='f500_fluvial',
 
        log=None,conn_str=None,dev=False,use_cache=True,out_dir=None,
        limit=None,
 
        ):
    
    """download dx from create_view_join_stats_to_bmeans()
    
    see also _03damage._05_mean_bins.get_grid_rl_dx()
    """ 
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'depths','02_views', country_key, haz_key)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    
    if log is None:
        log = init_log(name=f'grid_wd')
    
    if dev: use_cache=False
    #===========================================================================
    # cache
    #===========================================================================
    fnstr = f'grid_wd_{country_key}_{haz_key}'
    uuid = hashlib.shake_256(f'{fnstr}_{dev}_{limit}'.encode("utf-8"), usedforsecurity=False).hexdigest(8)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')
    
    if (not os.path.exists(ofp)) or (not use_cache):
        
        #===========================================================================
        # talbe params
        #===========================================================================
        #see create_view_join_stats_to_bmeans()
        tableName = f'grid_wd_bmean_bstats_{country_key}_{haz_key}' 
        
        if dev:
            schema = 'dev'
    
        else:
            schema = 'inters_agg' 
            
        assert pg_table_exists(schema, tableName, asset_type='matview'), f'missing table dependency \'{tableName}\'\n    see create_view_join_stats_to_bmeans()'
        keys_l = ['country_key', 'grid_size','haz_key', 'i', 'j', 'bldg_cnt', 'wet_cnt']
        
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
        dx_raw = pd.read_sql(cmd_str, engine, index_col=keys_l)
        """
        view(df_raw.head(100))        
        """    
        
        engine.dispose()
        conn.close()
        
        log.info(f'finished w/ {len(dx_raw)} total rows')
        
        
        #=======================================================================
        # clean
        #=======================================================================
        
        """some nulls on grid wd because of the left join"""
        dx = dx_raw.fillna(0.0)
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
        
        """
        dx.isna().sum()
        dx.index.to_frame().isna().sum()
        """
 
 
    log.info(f'got {dx.shape}')
    return dx
 

def run_all(use_cache=False, **kwargs):
    log = init_log(name=f'depths.views')
    
    create_view_merge_bmeans(log=log, **kwargs)
    
    create_view_join_stats_to_bmeans(log=log, **kwargs)
    
    get_grid_wd_dx(log=log, use_cache=use_cache, **kwargs)
 
    

if __name__ == '__main__':
    #create_view_merge_bmeans(dev=False, include_gcent=False)
    
    #create_view_join_stats_to_bmeans(dev=False, with_geom=False)
    
 
    
    #get_grid_wd_dx(dev=False, use_cache=False)
    
    
    run_all(dev=True)
    
    
    
    
    print('done')
    
 
    
    
    
    
    
    
    
    



    
 