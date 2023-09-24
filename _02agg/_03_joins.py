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

from _02agg.coms_agg import (
    get_conn_str, pg_vacuum, pg_spatialIndex, pg_exe, pg_get_column_names, pg_register,
    pg_comment
    )
 

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l, haz_label_d
    )






            
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
        #out_dir=None,
        #tableBaseName = 'bldgs_',
        dev=False,
        
 
        ):
    """join grid ids to building points (with some exposure)
    
    uses inters.{country_key}

    Returns
    ----------
    Postgres table [agg_bldg.bldgs_grid_link_{country_key}_{grid_size:04d}]
        bulding to grid links (only buidlings with some depths on f500_fluvial)
 
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start = datetime.now()    
    
    log = init_log(name=f'jgrid')
    
    if grid_size_l is None: grid_size_l = gridsize_default_l
    if country_l is  None: country_l = [e.lower() for e in index_country_fp_d.keys()]
    # if epsg_id is None: epsg_id=equal_area_epsg
    
    log.info(f'on \n    {country_l}\n    {conn_d}')
    
    #===========================================================================
    # loop and join
    #===========================================================================    
    for i, (grid_size, country_key) in enumerate(product([int(e) for e in grid_size_l], country_l)):
        tableName=f'bldgs_grid_link_{country_key}_{grid_size:04d}'
        log.info(f'on {i}: {tableName}') 
        
        _build_grid_inters_join(grid_size, country_key, tableName, conn_d, log, dev=dev) 
        
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
        epsg_id=equal_area_epsg,
        out_schema = 'agg_bldg',
        dev=False,
        haz_key='f500_fluvial',
        ):
    """build a table with the spatial join results"""
    
 
    #===========================================================================
    # defautls
    #===========================================================================
    start=datetime.now() 
    if dev: out_schema='dev'
    
    #==================================================================
    # using views is too slow (400secs)
    # tableName_grid='agg'
    # tableName_inters='pts_osm_fathom'
    #==================================================================
    
    #use the base tables (44 secs)
    tableName_grid = f'agg_{country_key}_{grid_size:07d}'
    tableName_inters=f'{country_key.lower()}'
 
    if not dev:
        pts_schema='inters'
        grid_schema='grids'
    else:
        pts_schema='dev'
        grid_schema='dev'
        out_schema='dev'
        #tableName_grid+='_wbldg'
    #===========================================================================
    # setup
    #===========================================================================
    
    
    pg_exe(f"""DROP TABLE IF EXISTS {out_schema}.{tableName}""")
    
    #===========================================================================
    # build query
    #===========================================================================
    #get the column names
    coln_l = pg_get_column_names('inters', tableName_inters)            
    print(f'columns\n    {coln_l}')
 
    cols = 'LOWER(pts.country_key) as country_key, pts.gid, pts.id, polys.grid_size, polys.i, polys.j'
    
    
    cmd_str=f"""
    CREATE TABLE {out_schema}.{tableName} AS
        SELECT {cols}
            FROM {pts_schema}.{tableName_inters} AS pts
                LEFT JOIN {grid_schema}.{tableName_grid} AS polys 
                    ON ST_Intersects(polys.geom, ST_Transform(pts.geometry, {epsg_id}))
                        WHERE pts.country_key=%s AND polys.grid_size=%s AND polys.country_key=%s
                            AND pts.{haz_key} > 0"""
    print(cmd_str)
    
    #===========================================================================
    # execute   
    #===========================================================================
            
    #perform the join
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
        with conn.cursor() as cur:
            cur.execute(cmd_str, (country_key.upper(), grid_size, country_key))
            
    #clean up
    log.info(f'cleaning')
    
    cmt_str = f'join grid ({tableName_grid}) i,j to points ({tableName_inters}) pts.{haz_key} > 0 \n'
    cmt_str += f'built with {os.path.realpath(__file__)} at '+datetime.now().strftime("%Y.%m.%d.%S")
    pg_comment(out_schema, tableName, cmt_str)
    
    #set keys
    pg_exe(f'ALTER TABLE {out_schema}.{tableName} ADD PRIMARY KEY (country_key, gid, id)')
    
    #pg_spatialIndex(out_schema, tableName)
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
 

def xxxrun_drop_haz_inters_agg(
        conn_d=postgres_d,
        country_l = ['deu'], 
        grid_size_l=[1020, 240, 60],
        haz_coln_l = None,
        conn_str=None,
        ):
    """cleaning of existing tables to drop the ahzard columns
    
    and create views by joing back the point geometry"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    #create if it exists
    if grid_size_l is None: grid_size_l=gridsize_default_l.copy()
    if country_l is  None: country_l=[e.lower() for e in index_country_fp_d.keys()]
    if haz_coln_l is None: haz_coln_l=list(haz_label_d.keys())
    if conn_str is None: conn_str = get_conn_str(conn_d)
    
    log = init_log(name=f'hdrop')
 
    schema='inters_agg'
    
    get_tbl = lambda x: f'pts_osm_fathom_{country_key}_{x:07d}'
 
 
    print(f'creating view on {len(country_l)}')
    
    for country_key in country_l:        
 
        #=======================================================================
        # clean columns in table
        #=======================================================================
        log.info(f'for {country_key} dropping columns')
        for grid_size in grid_size_l:
            tableName = get_tbl(grid_size)
                 
            # start the query with the first table
            cmd_str = f"""ALTER TABLE {schema}.{tableName} DROP COLUMN geom"""
 
            # get the column names
            coln_l = list(set(haz_coln_l).intersection(pg_get_column_names(schema, tableName)))
            
            if len(coln_l)==0:
                log.warning(f'no columns to drop... skipping')
                continue
 
            for coln in coln_l:
                cmd_str += f', DROP COLUMN {coln}'
        
            pg_exe(cmd_str, conn_str=conn_str, log=log)
 
            #===========================================================================
            # clean up
            #=========================================================================== 
            pg_vacuum(schema, tableName, conn_str) 
            
        #===================================================================
        # create a view----
        #===================================================================
        """view is too slow to be helpful"""
        for grid_size in grid_size_l:
            tableName = get_tbl(grid_size)
            
            
            viewName=tableName+'_v'
            pg_exe(f'DROP VIEW IF EXISTS {schema}.{viewName}', conn_str=conn_str, log=log)
            
            #===================================================================
            #  query
            #===================================================================
            
            cmd_str=f"""CREATE VIEW {schema}.{viewName} AS\n   SELECT """
            
            #get columns from left table
            coln_l = pg_get_column_names('inters', country_key)
            cmd_str+=', '.join([f'pts.{k}' for k in coln_l])
            
            cmd_str+=f', agg.grid_size, agg.i, agg.j'
            
            cmd_str+=f"""
            FROM inters.{country_key} as pts
                JOIN {schema}.{tableName} as agg
                     ON pts.id=agg.id
                 """
                                    
 
            pg_exe(cmd_str, conn_str=conn_str, log=log)
            
            #===================================================================
            # wrap
            #===================================================================
            pg_register(schema, viewName)
            
 
                
    print(f'finished')
    
      
    
 

if __name__ == '__main__':
    run_join_agg_grids(grid_size_l=None, dev=True)
    #run_drop_haz_inters_agg()
    
    
    
    
    
