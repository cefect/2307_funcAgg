"""

slice and concat grid depths
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
    get_conn_str, pg_vacuum, pg_spatialIndex, pg_exe, pg_get_column_names, pg_register, pg_getcount
    )
 

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l, haz_label_d
    )

 
def run_pg_slice_concat_grid_samples(
        country_key, haz_key,
        grid_size_l=None,
        log=None,
        conn_str=None,
        dev=False,
        out_dir=None,
        chunksize=1e6,
        ):
    """cut a single hazard and merge across grid_sizes"""
        

    if grid_size_l is None: grid_size_l = gridsize_default_l
    
    #===========================================================================
    # defaults
    #===========================================================================
    schema='inters_grid'
    tableName=f'grids_wd_{country_key}_{haz_key}'
    start=datetime.now()   
    
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
        
    sql(f'DROP VIEW IF EXISTS {schema}.{tableName}')
    #===========================================================================
    # build zuery
    #===========================================================================
    #===========================================================================
    # no... dont want a new table... just download
    # tableName = f'rl_mean_{country_key}_{haz_key}'
    # sql(f'DROP TABLE IF EXISTS damage.{tableName}')
    #===========================================================================
    
    cmd_str = f'CREATE VIEW {schema}.{tableName} AS \n'
 
    log.info(f'on {grid_size_l}')
    
    first = True
    for grid_size in grid_size_l:
        
        tableName_i = f'pts_fathom_{country_key}_grid_{grid_size:04d}'
        
        #log.info(f'on {grid_size} w/ \'{tableName_i}\'')
        
        if not first:
            cmd_str += 'UNION\n'
 
        cmd_str += f'SELECT country_key, grid_size, i, j, {haz_key}\n'
        cmd_str +=f'FROM {schema}.{tableName_i} \n'
            
        
        #filters
        cmd_str += f'    WHERE {tableName_i}.{haz_key}>0'
        
 
 
        
        
        first = False
    
    if dev:
        cmd_str += f'        LIMIT 100\n'
        
    #===========================================================================
    # execute
    #===========================================================================
    print(cmd_str)
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
    run_pg_slice_concat_grid_samples('deu','f500_fluvial', dev=False)
    
 
    
        
    
