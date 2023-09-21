'''
Created on Sep. 21, 2023

@author: cefect


link exposed agg grids to all child buildings

NOTE: _02agg._03_joins is similar, but leaner as it only includes exposed buildings
'''
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

from coms import (
    init_log, today_str, get_directory_size,dstr, view,  
    ) 

from _02agg.coms_agg import (
    get_conn_str, pg_vacuum, pg_spatialIndex, pg_exe, pg_get_column_names, pg_register,
    pg_comment, pg_getCRS, pg_register
    )
 

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l, haz_label_d
    )



def run_agg_bldg_full_links(
         country_key, grid_size,
 
        conn_str=None, 
        log=None,
        epsg_id=equal_area_epsg,
 
 
        dev=False,
        with_geo=False,
 
        ):
    """spatially join (exposed) grid ids to each bldg using grid polygons
    inters_grid.agg_expo_{country_key}_{grid_size:04d}_poly only includes doubly exposed grids
    
    
    Params
    ----------
    haz_key: str
        column with grid wd used to identify grids as 'exposed' (want most extreme)
        
        
    """
    
 
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
 
    tableName = f'bldgs_grid_link_full_{country_key}_{grid_size:04d}'
    
    #wd for all buildings. see _01intersect._03_topost
    table_left=f'{country_key}' 
    
    #wd for all grids w/ polygon geometry. see _02agg._07_views.run_view_grid_wd_wgeo()
    table_grid=f'agg_expo_{country_key}_{grid_size:04d}_poly' 
 
    
    if dev:
        schema='dev'
        schema_left, schema_grid=schema, schema        

    else:
        schema='expo' 
        schema_left='inters'
        schema_grid='inters_grid'       
        
    
    keys_l = ['country_key', 'grid_size', 'i', 'j']
    

    assert pg_getCRS(schema_grid, table_grid)==epsg_id
    #===========================================================================
    # spatially join grid keys
    #===========================================================================
    """
    NOTES:
    want to include all points (even dry ones)
    want an inner join so we only get those that intersect
    
    """
    
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName}')    
    
 
    
    cols = 'LOWER(pts.country_key) as country_key, pts.gid, pts.id, polys.grid_size, polys.i, polys.j'
    if with_geo: cols+=', pts.geometry as geom'
    

    cmd_str=f"""
    CREATE TABLE {schema}.{tableName} AS
        SELECT {cols}
            FROM {schema_left}.{table_left} AS pts
                JOIN {schema_grid}.{table_grid} AS polys 
                    ON ST_Intersects(polys.geom, ST_Transform(pts.geometry, {epsg_id}))
                    """
    
    sql(cmd_str)
            
    #===========================================================================
    # #clean up
    #===========================================================================
    log.info(f'cleaning')
    pg_exe(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY (country_key, gid, id)')
    
    cmt_str = f'join grid ({table_grid}) i,j to points ({table_left}) \n'
    cmt_str += f'built with {os.path.realpath(__file__)} at '+datetime.now().strftime("%Y.%m.%d.%S")
    pg_comment(schema, tableName, cmt_str)
    
 
    if with_geo:
        pg_register(schema, tableName)
    pg_vacuum(schema, tableName)
    
 
            
            
    #===========================================================================
    # #wrap
    #===========================================================================
    meta_d = {
        'tdelta':(datetime.now() - start).total_seconds(), 
        'RAM_GB':psutil.virtual_memory()[3] / 1000000000, 
        'postgres_GB':get_directory_size(postgres_dir)}
        #'output_MB':os.path.getsize(ofp)/(1024**2)
    log.info(f'finishedw/ \n{meta_d}')
    
    return tableName
 
 
if __name__ == '__main__':
    
    run_agg_bldg_full_links('deu', 1020, dev=True, with_geo=True)
