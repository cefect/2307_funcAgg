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

 

def create_view_join_bldg_geom(schema, table_left, country_key='deu',
                               log=None, dev=False, 
                               conn_str=None):
    """create a new vew of the passed table that joins the building geometry"""
    
 
    
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
    table_right = f'{country_key}' #merge of all grid geometry... see run_view_grid_geom_union()
    if dev:
        schema_right = 'dev'
    else:
        schema_right = 'inters'
    assert pg_table_exists(schema_right, table_right, asset_type='table'), f'{schema_right}.{table_right} table must exist'
    
    keys_l = ['country_key', 'gid', 'id']
    #=======================================================================
    # setup
    #=======================================================================
    tableName = table_left + '_wgeo'
    sql(f'DROP MATERIALIZED VIEW IF EXISTS {schema}.{tableName} CASCADE')
    #=======================================================================
    # build query
    #=======================================================================
    cmd_str = f'CREATE MATERIALIZED VIEW {schema}.{tableName} AS'
    
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_l])
    
    #add an arbitrary indexer for QGIS viewing
    cols =f'ROW_NUMBER() OVER (ORDER BY tleft.i, tleft.j) as fid, ' 
    cols +=f'tleft.*, tright.geometry as geom' 
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

 
 
 
    
    
    
 
    
        
    
