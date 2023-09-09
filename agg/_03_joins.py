'''
Created on Sep. 5, 2023

@author: cefect

join agg grids to sample poitns
'''
#===============================================================================
# IMPORTS-----
#===============================================================================
import os, hashlib, sys, subprocess, psutil
from datetime import datetime
from itertools import product

import psycopg2
print('psycopg2.__version__=' + psycopg2.__version__)

from sqlalchemy import create_engine, URL

from tqdm import tqdm

import pandas as pd

from coms import (
    init_log, today_str, get_directory_size,dstr, view,  
    ) 

from agg.coms_agg import get_conn_str, pg_vacuum, pg_spatialIndex

from definitions import index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir




tableName_grid='agg'
out_schema = 'inters_agg'
            
#===============================================================================
# FUNCS---------
#===============================================================================

def run_join_agg_grids(
        country_l = ['bgd'],
        grid_size_l=[
            100000, 
            #2e5, #big for testing
            ],
        
        conn_d=postgres_d,
        out_dir=None,
        tableBaseName = 'pts_osm_fathom'
        
 
        ):
    """join intersect points to agg grids
    
 
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()    
 
 
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'agg', '01_join')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    
    log = init_log(name=f'jgrid', fp=os.path.join(out_dir, today_str+'.log'))
    
    if country_l is  None: country_l=[e.lower() for e in index_country_fp_d.keys()]
    #if epsg_id is None: epsg_id=equal_area_epsg
    
    log.info(f'on \n    {country_l}\n    {conn_d}')
    
    #===========================================================================
    # loop and join
    #===========================================================================    
    for i, (grid_size, country_key) in enumerate(product([int(e) for e in grid_size_l], country_l)):
        tableName=f'{tableBaseName}_{country_key}_{grid_size:07d}'
        log.info(f'on {i}: {tableName}') 
        
        _build_grid_inters_join(grid_size, country_key, tableName, conn_d, log) 
        
    #===========================================================================
    # wrap
    #===========================================================================
        #meta
    meta_d = {
        'tdelta':(datetime.now() - start).total_seconds(), 
        'RAM_GB':psutil.virtual_memory()[3] / 1000000000, 
        'postgres_GB':get_directory_size(postgres_dir)}
        #'output_MB':os.path.getsize(ofp)/(1024**2)
    log.info(f'finishedw/ \n{meta_d}')
    
    return
        

def _build_grid_inters_join(
        grid_size, country_key, tableName, conn_d, log,
        epsg_id=equal_area_epsg
        ):
    """build a table with the spatial join results"""
    
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
        #remove if it exists
        with conn.cursor() as cur:
            cur.execute(f"""DROP TABLE IF EXISTS {out_schema}.{tableName}""")
            
            
    #perform the join
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
        with conn.cursor() as cur:
            cmd_str=f"""
                CREATE TABLE {out_schema}.{tableName} AS
                    SELECT pts.country_key, pts.gid, pts.id, polys.grid_size, polys.I, polys.J, ST_Transform(pts.geometry, {epsg_id}) as geom 
                        FROM inters.{country_key} AS pts
                        JOIN grids.{tableName_grid} AS polys
                    ON ST_Contains(polys.geom, ST_Transform(pts.geometry, {epsg_id}))
                        WHERE pts.country_key=%s AND polys.grid_size=%s
 
                    
                    """
            print(cmd_str)
            cur.execute(cmd_str, (country_key.upper(),grid_size))
            
    #clean up
    pg_spatialIndex(conn_d, out_schema, tableName)
    pg_vacuum(conn_d, f'{out_schema}.{tableName}')
    
            
    #get stats
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT COUNT(*) FROM {out_schema}.{tableName}""")
            print(cur.fetchall())
 





if __name__ == '__main__':
    run_join_agg_grids()
    
    
    
    
    