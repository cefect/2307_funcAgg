"""

create a view that includes all of the grid depths for one hazard
pivot and slice agg grid samples to get all grid levels for a single hazard
"""


#===============================================================================
# IMPORTS-----
#===============================================================================
import os, hashlib, sys, subprocess, psutil
from datetime import datetime
from itertools import product

import psycopg2
#print('psycopg2.__version__=' + psycopg2.__version__)

from sqlalchemy import create_engine, URL

from tqdm import tqdm

import pandas as pd
import numpy as np

from coms import (
    init_log, today_str, get_directory_size,dstr, view,  
    ) 

from _02agg.coms_agg import (
    get_conn_str, pg_vacuum, pg_spatialIndex, pg_exe, pg_get_column_names, pg_register, pg_getcount,
    pg_table_exists
    )
 

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l, haz_label_d
    )

 

def create_view_join_grid_geom(schema, table_left, country_key,
                               log=None, dev=False, 
                               conn_str=None):
    """create a new vew of the passed table that joins the grid geometry"""
    
 
    
    #===========================================================================
    # defaults
    #===========================================================================
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'grid_geom')
        
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    
    log.info(f'creating a view with geometry from \'{table_left}\'')
    
    #===========================================================================
    # table params
    #===========================================================================
    table_right = f'agg_{country_key}' #merge of all grid geometry... see run_view_grid_geom_union()
    if dev:
        schema_right = 'dev'
    else:
        schema_right = 'grids'
    assert pg_table_exists(schema_right, table_right, asset_type='matview'), f'{schema_right}.{table_right} view must exist'
    
    keys_l = ['country_key', 'grid_size', 'i', 'j']
    #=======================================================================
    # setup
    #=======================================================================
    tableName = table_left + '_wgeo'
    sql(f'DROP VIEW IF EXISTS {schema}.{tableName} CASCADE')
    #=======================================================================
    # build query
    #=======================================================================
    cmd_str = f'CREATE VIEW {schema}.{tableName} AS'
    
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_l])
    
    #add an arbitrary indexer for QGIS viewing
    cols =f'ROW_NUMBER() OVER (ORDER BY tleft.i, tleft.j) as fid, ' 
    cols +=f'tleft.*, tright.geom' 
    cmd_str+= f"""
        SELECT {cols}
            FROM {schema}.{table_left} AS tleft
                LEFT JOIN {schema_right}.{table_right} AS tright
                    ON {link_cols}
            """
            
    sql(cmd_str)
 
    pg_register(schema, tableName)
    
    log.info(f'finished on {schema}.{tableName}')
    
    return tableName

def run_view_grid_samp_pivot(
        country_key, haz_key,
        grid_size_l=None,
        log=None,
        conn_str=None,
        dev=False,
        with_geom=False,
        ):
    """create a view that includes all of the grid depths for one hazard
    
    Returns
    -----------
    postgres view: inters_grid.agg_samps_{country_key}_{haz_key}"""
        

    
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()  
    
    if grid_size_l is None: grid_size_l = gridsize_default_l
    
    
    if dev:
        schema = 'dev'
    else:
        schema='inters_grid'
        
    tableName=f'agg_samps_{country_key}_{haz_key}'
     
    
    country_key=country_key.lower() 
 
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'grid')
        
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
        
    #===========================================================================
    # if out_dir is None:
    #     out_dir = os.path.join(wrk_dir, 'outs', 'damage','03_mean', country_key, haz_key)
    # if not os.path.exists(out_dir):os.makedirs(out_dir)
    #===========================================================================
        
    sql(f'DROP MATERIALIZED VIEW IF EXISTS {schema}.{tableName} CASCADE')
    #===========================================================================
    # build zuery
    #===========================================================================
    #===========================================================================
    # no... dont want a new table... just download
    # tableName = f'rl_mean_{country_key}_{haz_key}'
    # sql(f'DROP TABLE IF EXISTS damage.{tableName}')
    #===========================================================================
    
    cmd_str = f'CREATE MATERIALIZED VIEW {schema}.{tableName} AS \n'
 
    log.info(f'on {grid_size_l}')
    
    first = True
    for grid_size in grid_size_l:
        
        tableName_i = f'agg_samps_{country_key}_{grid_size:04d}'
        
        if not first:
            cmd_str += 'UNION\n'
 
        cmd_str += f'SELECT country_key, grid_size, i, j, {haz_key}\n'
        cmd_str += f'FROM {schema}.{tableName_i} \n'
        
        # filters
        cmd_str += f'    WHERE {tableName_i}.{haz_key}>0\n'
        
        first = False
    
 
    sql(cmd_str)
    
    log.info(f'finished w/ {tableName}')
    #===========================================================================
    # add geometry-------
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
    log.info(f'finishedw/ \n{meta_d}')
    
    return 
        
        

def run_view_grid_geom_union(country_key,
                         grid_size_l=None,
         dev=False, conn_str=None,
        ):
    
    """create a view by unioning all the grid geometry together
    
    useful for joins later"""
    

    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()  
    
    log = init_log(name=f'view')
    
    if grid_size_l is None: grid_size_l = gridsize_default_l
    
    if dev:
        schema = 'dev'
    else:
        schema='grids'
        
    """could also use the occupied tables"""
    tableName=f'agg_{country_key}'
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    sql(f'DROP VIEW IF EXISTS {schema}.{tableName}')
    #===========================================================================
    # buidl query
    #===========================================================================
    
    cmd_str = f'CREATE VIEW {schema}.{tableName} AS \n'
    
    first = True
    for grid_size in grid_size_l:
        
        tableName_i = f'agg_{country_key}_{grid_size:07d}'
        
        if not first:
            cmd_str += 'UNION\n'
 
        cmd_str += f'SELECT country_key, grid_size, i, j, geom\n'
        cmd_str += f'FROM {schema}.{tableName_i} \n'
        
        # filters        
        first = False
    
 
    sql(cmd_str)
    
    
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
    log.info(f'finished w/ {tableName} \n{meta_d}')
    
    return 
    
        
def create_view_grid_wd_wgeo(
        country_key, grid_size,
 
        log=None,
        conn_str=None,
        dev=False,
        with_geom=False,
        haz_key='f500_fluvial'
        ):
    """create a view of of exposed/occupied grids with polygon geometry
        each result cell should have centroid exposure and some buidling exposure
    
    needed by _04expo._01_full_links
    
    similar to create_view_join_grid_geom()... but we drop unexposed grids
    
    Returns
    -----------
    postgres view: inters_grid.agg_samps_{country_key}_{haz_key}"""
 
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()  
 
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'grid')
        
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #===========================================================================
    # table params
    #===========================================================================
    if dev:
        schema = 'dev'
        schema_right=schema
    else:
        schema='inters_grid'
        schema_right='grids'
        
    table_left = f'agg_samps_{country_key}_{grid_size:04d}'
    
    #raw grids w/ polygon geometry. see _02agg._01_grids.run_build_agg_grids()
    table_right=f'agg_{country_key}_{grid_size:07d}'
    tableName=f'agg_expo_{country_key}_{grid_size:04d}_poly'
        
 
    keys_l = ['country_key', 'grid_size', 'i', 'j']
    #===========================================================================
    # join polygons
    #===========================================================================
    sql(f'DROP MATERIALIZED VIEW IF EXISTS {schema}.{tableName}') 
    
    cmd_str = f'CREATE MATERIALIZED VIEW {schema}.{tableName} AS \n'
    
    cols = ', '.join([f'tleft.{e}' for e in keys_l]) 
    cols +=f', tright.geom'
    #cols =f'tleft.*, tright.geom'
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_l]) 
    cmd_str+= f"""
        SELECT {cols}
            FROM {schema}.{table_left} AS tleft
                LEFT JOIN {schema_right}.{table_right} AS tright
                    ON {link_cols}
                        WHERE tleft.{haz_key}>0
            """
    
    sql(cmd_str)
    
    log.info(f'finished w/ {tableName}')
    
    #===========================================================================
    # wrap
    #===========================================================================
    pg_spatialIndex(schema, tableName)
    
        #meta
    meta_d = {
        'tdelta':(datetime.now() - start).total_seconds(), 
        'RAM_GB':psutil.virtual_memory()[3] / 1000000000, 
        #'postgres_GB':get_directory_size(postgres_dir)}
        #'output_MB':os.path.getsize(ofp)/(1024**2)
        }
    log.info(f'finished w/ {tableName} \n{meta_d}')
 
 
    
    return tableName


def run_view_grid_wd_wgeo(ck, grid_size_l=None, **kwargs):
    """ takes a few mins"""
    log = init_log(name=f'grid')
    
    if grid_size_l is None: grid_size_l = gridsize_default_l
    
    d=dict()
    for g in grid_size_l:
        d[g] = create_view_grid_wd_wgeo(ck, g, log=log, **kwargs)
        
    log.info(f'finished on \n    {d}')   
        
if __name__ == '__main__':
    run_view_grid_geom_union('deu', dev=False)
    #run_view_grid_samp_pivot('deu','f500_fluvial', dev=False, with_geom=False)
    
    #run_view_grid_wd_wgeo('deu', dev=False)
    
    
    
 
    
        
    
