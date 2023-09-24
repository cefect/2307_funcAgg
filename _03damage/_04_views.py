'''
Created on Sep. 21, 2023

@author: cefect

prepare data concats/pivot views for plotting
'''


#===============================================================================
# IMPORTS-----
#===============================================================================
import os, hashlib, sys, subprocess, psutil
from datetime import datetime
from itertools import product

#import psycopg2
#print('psycopg2.__version__=' + psycopg2.__version__)

#from sqlalchemy import create_engine, URL

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
    pg_table_exists
    )
 
from _02agg._07_views import create_view_join_grid_geom




def run_view_merge_grid(country_key, haz_key,
                         grid_size_l=None,
         dev=False, conn_str=None, log=None,
        ):
    
    """create a view by unioning all the grid losses
    
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
    if dev:
        schema = 'dev'
    else:
        schema='damage'
        
    """could also use the occupied tables"""
    tableName=f'rl_mean_grid_{country_key}_{haz_key}'
    
    
    
    
    #===========================================================================
    # buidl query
    #===========================================================================
    sql(f'DROP MATERIALIZED VIEW IF EXISTS {schema}.{tableName} CASCADE')
    
    cmd_str = f'CREATE MATERIALIZED VIEW {schema}.{tableName} AS \n'
    
    first = True
    for grid_size in grid_size_l:
        
        tableName_i = f'rl_mean_{country_key}_{grid_size:04d}'
                
        if not first:
            cmd_str += 'UNION\n'
 
        
        cmd_str += f'SELECT *\n'
        cmd_str += f'FROM {schema}.{tableName_i} \n'
        cmd_str += f'WHERE {tableName_i}.haz_key=\'{haz_key}\' \n'
        
        # filters        
        first = False
    
 
    sql(cmd_str)
    
    #===========================================================================
    # clean
    #===========================================================================
    keys_str = ', '.join(keys_l)
    
    """not working...."""
    #sql(f'ALTER MATERIALIZED VIEW {schema}.{tableName} ADD PRIMARY KEY ({keys_str})') #doesnt work for views
    #sql(f'ALTER MATERIALIZED VIEW {schema}.{tableName} ADD CONSTRAINT unique_{tableName} UNIQUE ({keys_str})')
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

    
        
        
    

def run_view_join_depths(
        country_key, haz_key,
 
        log=None,
        conn_str=None,
        dev=False,
 
 
        with_geom=False,
        ):
    """for plotting, merge the grid_sizes and slice to a single haz_key
    
    
    needed by _03damage._05_mean_bins.get_grid_rl_dx()
    """
        

    #if grid_size_l is None: grid_size_l = gridsize_default_l
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    start = datetime.now()   
    
    country_key = country_key.lower() 
 
    if conn_str is None: conn_str = get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'rl_mean')
        
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #===========================================================================
    # talbe params
    #===========================================================================
 
    
    table_left=f'rl_mean_grid_{country_key}_{haz_key}' #losses for all grrid sizes merged... see run_view_merge_grid()
    tableName = table_left+'_wd'
    table_right=f'agg_samps_{country_key}_{haz_key}' #wd for all grids. see _02agg._07_views()
 
    
    if dev:
        schema='dev'
        schema_left, schema_right=schema, schema        

    else:
        schema='damage' 
        schema_left='damage'
        schema_right='inters_grid'       
        
    
    keys_l = ['country_key', 'grid_size', 'i', 'j']
 
    
    #===========================================================================
    # join depths
    #===========================================================================
    sql(f'DROP MATERIALIZED VIEW IF EXISTS {schema}.{tableName}')    
    
    
    cmd_str = f'CREATE MATERIALIZED VIEW {schema}.{tableName} AS \n'
    
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_l]) 
    cmd_str+= f"""
        SELECT tleft.*, tright.{haz_key} AS grid_wd
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
 
    log.info(f'creating view from {table_left} and {table_right}')
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




def run_all(ck='deu', haz_key='f500_fluvial', **kwargs):
    log = init_log(name=f'dmg_views')
    
    run_view_merge_grid('deu', 'f500_fluvial',log=log, **kwargs)
    run_view_join_depths('deu', 'f500_fluvial',log=log, **kwargs)
    


if __name__ == '__main__':
    #run_view_merge_grid('deu', 'f500_fluvial',dev=False)
    #run_view_join_depths('deu', 'f500_fluvial', dev=False, with_geom=False)
    
    
    run_all()
    
 
        
        
        
    
