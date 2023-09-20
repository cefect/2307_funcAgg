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

from agg.coms_agg import (
    get_conn_str, pg_vacuum, pg_spatialIndex, pg_exe, pg_get_column_names, pg_register, pg_getcount,
    pg_table_exists
    )
 

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l, haz_label_d
    )

 
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
    
 
    #log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
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
    
    #===========================================================================
    # add geometry
    #===========================================================================
    if with_geom:
        
        """materlized view doesn't seem to be working in QGIS"""
        log.info(f'creating a view with geometry')
        
        geom_table=f'agg_{country_key}'
        if dev:
            geom_schema='dev'
        else:
            geom_schema='grids'
        
        assert pg_table_exists(geom_schema, geom_table, asset_type='matview'), f'{geom_schema}.{geom_table} view must exist'
        
        #=======================================================================
        # setup
        #=======================================================================
        tableName1 = tableName+'_wgeo'
        
        sql(f'DROP TABLE IF EXISTS {schema}.{tableName1}')
        
        
        #=======================================================================
        # build query
        #=======================================================================
 
        cmd_str = f"""
            CREATE TABLE {schema}.{tableName1} AS
                SELECT ltab.*, rtab.geom
                    FROM {schema}.{tableName} as ltab
                        LEFT JOIN {geom_schema}.{geom_table} as rtab
                            ON ltab.i=rtab.i 
                            AND ltab.j=rtab.j 
                            AND ltab.grid_size=rtab.grid_size
                            AND ltab.country_key=rtab.country_key
                            """
                            
        sql(cmd_str)
        
        sql(f'ALTER TABLE {schema}.{tableName1} ADD PRIMARY KEY (country_key, grid_size, i,j)')
        pg_register(schema, tableName1)
        
        
 
    
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
        schema='grid'
        
    """could also use the occupied tables"""
    tableName=f'agg_{country_key}'
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    sql(f'DROP MATERIALIZED VIEW IF EXISTS {schema}.{tableName}')
    #===========================================================================
    # buidl query
    #===========================================================================
    
    cmd_str = f'CREATE MATERIALIZED VIEW {schema}.{tableName} AS \n'
    
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
    log.info(f'finishedw/ \n{meta_d}')
    
    return 
    
        
        
        
        
if __name__ == '__main__':
    #run_view_grid_geom_union('deu', dev=True)
    run_view_grid_samp_pivot('deu','f500_fluvial', dev=True)
    
    
    
 
    
        
    
