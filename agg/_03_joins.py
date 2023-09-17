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
#print('psycopg2.__version__=' + psycopg2.__version__)

from sqlalchemy import create_engine, URL

from tqdm import tqdm

import pandas as pd

from coms import (
    init_log, today_str, get_directory_size,dstr, view,  
    ) 

from agg.coms_agg import get_conn_str, pg_vacuum, pg_spatialIndex, pg_exe
 

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l
    )





out_schema = 'inters_agg'
            
#===============================================================================
# FUNCS---------
#===============================================================================

def run_join_agg_grids(
        country_l = ['deu'],
        grid_size_l=[
            1020,  #412.6 secs (using views)
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
    
    if grid_size_l is None: grid_size_l=gridsize_default_l
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
    #===========================================================================
    # defautls
    #===========================================================================
    start=datetime.now() 
    
    #==================================================================
    # using views is too slow (400secs)
    # tableName_grid='agg'
    # tableName_inters='pts_osm_fathom'
    #==================================================================
    
    #use the base tables (44 secs)
    tableName_grid = f'agg_{country_key}_{grid_size:07d}'
    tableName_inters=f'{country_key.lower()}'
 
    #===========================================================================
    # setup
    #===========================================================================
    
    
    pg_exe(f"""DROP TABLE IF EXISTS {out_schema}.{tableName}""")
    
    #get the column names
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = %s
                        ORDER BY ordinal_position;
                        """, (tableName_inters,))
            
            coln_l = [e[0] for e in cur.fetchall()]
            
    print(f'columns\n    {coln_l}')
 
    
    
            
            
    #perform the join
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
        with conn.cursor() as cur:
            #build the query
            cmd_str=f"""CREATE TABLE {out_schema}.{tableName} AS
                            SELECT """
            
            #get all the columns  
            for e in [e for e in coln_l if not e=='geometry']:
                cmd_str+=f'pts.{e}, '
                            
 

            cmd_str+=f"""polys.grid_size, polys.I, polys.J, ST_Transform(pts.geometry, {epsg_id}) as geom 
                        FROM inters.{tableName_inters} AS pts
                        JOIN grids.{tableName_grid} AS polys
                    ON ST_Contains(polys.geom, ST_Transform(pts.geometry, {epsg_id}))
                        WHERE pts.country_key=%s AND polys.grid_size=%s AND polys.country_key=%s
 
                    
                    """
            print(cmd_str)
            cur.execute(cmd_str, (country_key.upper(), grid_size, country_key))
            
    #clean up
    log.info(f'cleaning')
    pg_spatialIndex(out_schema, tableName)
    pg_vacuum(out_schema, tableName)
    
            
    #get stats
    print(pg_exe(f"""SELECT COUNT(*) FROM {out_schema}.{tableName}""", log=log, return_fetch=True))
 
            
            
    #wrap
    meta_d = {
        'tdelta':(datetime.now() - start).total_seconds(), 
        'RAM_GB':psutil.virtual_memory()[3] / 1000000000, 
        'postgres_GB':get_directory_size(postgres_dir)}
        #'output_MB':os.path.getsize(ofp)/(1024**2)
    log.info(f'finishedw/ \n{meta_d}')
 





if __name__ == '__main__':
    run_join_agg_grids()
    
    
    
    
    