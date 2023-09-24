'''
Created on Jul. 25, 2023

@author: cefect


grids with buildings and the counts
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
from sqlalchemy import create_engine, URL

from tqdm import tqdm



from definitions import (
    wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d, postgres_d, 
    equal_area_epsg, fathom_vals_d, gridsize_default_l
    )
from definitions import temp_dir as temp_dirM
 
from coms import (
    init_log, today_str, get_directory_size,dstr
    )

from _02agg.coms_agg import (
    get_conn_str, pg_getCRS, pg_to_df, pg_exe, pg_getcount, pg_spatialIndex, pg_get_column_names,
    pg_vacuum, pg_comment, pg_register
    )

from _02agg._07_views import create_view_join_grid_geom




 



#===============================================================================
# EXECUTORS--------
#===============================================================================
 
def run_grids_occupied(
                        country_key, 
                           #hazard_key,
                               grid_size,
                           out_dir=None,
                           dev=False,
                           conn_str=None,
                           epsg_id=equal_area_epsg,
                           log=None,
                           geom_type='point',
 
                           ):
    """build sampling table from grid centroids for those grids selected by:
        bldgs_grid_link (_02agg._03_joins):
        
    NOTE: because we mostly use this to get centroids, we create a new table
        but we now also use this for polygons... which would make more sense as a view
    
 
    Params
    ---------
    geom_type: str
        type of geometry to include
        
    Returns
    ---------
    postgres table [agg_bldg.agg_occupied_{country_key}_{grid_size:04d}]
        grid to building links
    """
    

    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    country_key=country_key.lower()
    #assert hazard_key in index_hazard_fp_d, hazard_key
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'agg','04_occu', country_key,  f'{grid_size:05d}')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    if log is None:
        log = init_log(name=f'occu.{country_key}.{grid_size}', fp=os.path.join(out_dir, today_str+'.log'))
    
    
    keys_d = {'country_key':country_key, 
              #'hazard_key':hazard_key, 
              'grid_size':grid_size}
    log.info(f'on {keys_d}')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    #===========================================================================
    # get table names
    #===========================================================================
    link_tableName=f'bldgs_grid_link_{country_key}_{grid_size:04d}'    
    grid_tableName=f'agg_{country_key}_{grid_size:07d}'    
    new_tableName=f'agg_occupied_{country_key}_{grid_size:04d}'
    #bldg_expo_tn = country_key.lower()
    
    if dev:
        out_schema='dev'
        link_schema='dev'
        grid_schema='dev'
        #bldg_expo_sch='dev'
        
    else:
        link_schema='agg_bldg'
        out_schema='agg_bldg'
        grid_schema='grids'
        #bldg_expo_sch='inters'
            
    #===========================================================================
    # create a temp table of unique indexers
    #===========================================================================
    """use the building-grid links to construct an index of i,j values with exposed buildings
    
    NOTE: this means we lose grids with centroid exposure (but dry or no buildigns)
        not a bad thing
    """  
    schema1 = 'temp'  
    tableName1= new_tableName+'_exposed'
    if dev: tableName1+='_dev'
    log.info(f'creating \'{schema1}.{tableName1}\' from unique i,j columns from {link_tableName}')     
    sql(f"DROP TABLE IF EXISTS {schema1}.{tableName1}")
      
 
      
    #get exposed indexers and their feature counts
    sql(f'''CREATE TABLE {schema1}.{tableName1} AS 
                SELECT LOWER(ltab.country_key) AS country_key, ltab.grid_size, ltab.i, ltab.j, COUNT(*) as bldg_expo_cnt
                    FROM {link_schema}.{link_tableName} AS ltab
                            GROUP BY ltab.country_key, ltab.grid_size, ltab.i, ltab.j''')
      
    #add the primary key
    sql(f"ALTER TABLE {schema1}.{tableName1} ADD PRIMARY KEY (country_key, i, j)")
 
    #===========================================================================
    # join grid geometryu
    #===========================================================================
    log.info(f'\n\njoining grid geometry')
    
    
    
    
    
    #prep query
    cols = f'ltab.*, '
    if geom_type=='point':
        cols+=f'ST_Centroid(rtab.geom) as geom'
    elif geom_type=='poly':
        cols+=f'rtab.geom'
        new_tableName+='_poly'
        
    #prep
    sql(f"DROP TABLE IF EXISTS {out_schema}.{new_tableName}")
      
    #get exposed indexers and their feature counts
    sql(f'''CREATE TABLE {out_schema}.{new_tableName} AS 
                SELECT {cols}
                    FROM {schema1}.{tableName1} AS ltab
                        LEFT JOIN {grid_schema}.{grid_tableName} as rtab
                            ON ltab.i=rtab.i AND ltab.j=rtab.j AND ltab.grid_size=rtab.grid_size AND ltab.country_key=rtab.country_key''')
    
 
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'\n\nwrap')
    

    #key
    pg_exe(f'ALTER TABLE {out_schema}.{new_tableName} ADD PRIMARY KEY (country_key, grid_size, i, j)')
    
    #comment
    cmt_str = f'grids with exposed buildings, building counts, and wet counts per hazard\n'
    cmt_str += f'built with {os.path.realpath(__file__)} at '+datetime.now().strftime("%Y.%m.%d.%S")
    pg_comment(out_schema, new_tableName, cmt_str)
    
    #spatisl
    pg_register(out_schema, new_tableName)
    assert pg_getCRS(out_schema, new_tableName)==epsg_id
    pg_spatialIndex(out_schema, new_tableName)
    
    #clean up
    pg_vacuum(out_schema, new_tableName)
    
    #drop the temps
    sql(f"DROP TABLE IF EXISTS temp.{new_tableName}")
    #sql(f"DROP TABLE IF EXISTS {schema1}.{tableName2}")
    
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(f'finished on \'{new_tableName}\' w/ \n    {meta_d}')
    
    return
        
def run_all(country_key='deu', **kwargs):
    log = init_log(name='occu')
    
    for grid_size in gridsize_default_l:
        run_grids_occupied(country_key, grid_size, log=log, **kwargs)
        
#===============================================================================
# def create_poly_views(country_key='deu', grid_size_l=None, log=None,conn_str=None, dev=False):
#     """create a view of the occupied layer but with polygons instead of centroids"""
#     
#     #===========================================================================
#     # defaults
#     #===========================================================================
#     start=datetime.now()
#  
#     if grid_size_l is None: grid_size_l=gridsize_default_l.copy()
#     
#  
#     if log is None:
#         log = init_log(name=f'occu_poly')
#  
#     sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
#     
#     #===========================================================================
#     # loop on grid size
#     #===========================================================================
#     for grid_size in grid_size_l:
#         
#         #===========================================================================
#         # get table names
#         #===========================================================================
#  
#         source_table=f'agg_occupied_{country_key}_{grid_size:04d}'
#         #bldg_expo_tn = country_key.lower()
#         
#         if dev:
#             source_schema='dev'
#             
#         else:
#             source_schema='agg_bldg'
#             
#         schema=source_schema
#         
#         create_view_join_grid_geom
#===============================================================================
        
    
 
if __name__ == '__main__':
    
    #run_grids_occupied('deu', 60, dev=True, geom_type='poly')
    
    run_all('deu', dev=True, geom_type='poly')
    
    
    
    
    
    
    
    
    
    
    