'''
Created on Jul. 25, 2023

@author: cefect


compute the total buildng counts and wet counts for each grid
'''
#===============================================================================
# IMPORTS--------
#===============================================================================
import os, hashlib, sys, subprocess

 
 
import psutil
from datetime import datetime
import pandas as pd
import numpy as np
 



from definitions import (
    wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d, postgres_d, 
    equal_area_epsg, fathom_vals_d, gridsize_default_l
    )
from definitions import temp_dir as temp_dirM
 


from _02agg.coms_agg import (
    get_conn_str, pg_getCRS, pg_to_df, pg_exe, pg_getcount, pg_spatialIndex, pg_get_column_names,
    pg_vacuum, pg_comment, pg_register
    )

from _02agg._07_views import create_view_join_grid_geom

from coms import (
    init_log, today_str, get_directory_size,dstr
    )


 



#===============================================================================
# EXECUTORS--------
#===============================================================================
 
def run_expo_stats_grouped(
                        country_key, 
                           #hazard_key,
                               grid_size,
                           out_dir=None,
                           dev=False,
                           conn_str=None,
                           epsg_id=equal_area_epsg,
                           log=None,
                           add_view=False,
                           ):
    """building exposure stats per grid
 
        
        
    Returns
    ---------
 
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
    # join building samples to full grid link------
    #===========================================================================
    
    """need this to compute exposure per hazard grid
    decided to use temprary tables as we should only need each of these once
    """
    
    #===========================================================================
    # params
    #===========================================================================
 
    #gid:bldg links for buildings in grids with some building expousre
    #see _04expo._01_full_links
    table_left =f'bldgs_grid_link_1x_{country_key}_{grid_size:04d}'
    #table_left = f'bldgs_grid_link_full_{country_key}_{grid_size:04d}'
    
    #sample values for all haz_keys
    table_right = f'{country_key}'
    
    #temp table output
    schema1 = 'temp'
    pts_table = f'bldgs_wd_{country_key}_{grid_size:04d}_linkd'
    
    if dev:
 
        schema_left, schema_right = 'dev', 'dev'
    else: 
        schema_left='expo'        
        schema_right = 'inters'
        
 
    keys_d = { 
        'bldg':['country_key', 'gid', 'id'],
        'grid':['country_key', 'grid_size', 'i', 'j']        
    }
 
    #===========================================================================
    # query
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS {schema1}.{pts_table}') 
    
    cmd_str = f'CREATE TABLE {schema1}.{pts_table} AS \n'
    
    #cols = ', '.join([f'tleft.{e}' for e in keys_l]) 
    #cols +=f', tright.geom'
    cols =f'tleft.*, '
    
    #get hazard columns from right table
    haz_coln_l = [e for e in pg_get_column_names(schema_right, table_right) if not e in keys_d['bldg'] + ['geometry']]
    #haz_coln_l = set(pg_get_column_names(schema_right, table_right)).difference(keys_d['bldg'] + ['geometry'])
        
    cols+=', '.join([f'CAST(tright.{e} AS real) as {e}' for e in haz_coln_l])
    
    """no need for geometry... just use the grid index keys"""
    #add point geom
    #cols +=f', ST_Transform(tright.geometry, {epsg_id}) as geom'
    
    #link columns
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_d['bldg']])
    
    #assemble 
    cmd_str+= f"""
        SELECT {cols}
            FROM {schema_left}.{table_left} AS tleft
                LEFT JOIN {schema_right}.{table_right} AS tright
                    ON {link_cols}
                        WHERE tright.f500_fluvial IS NOT NULL
            """
    
    sql(cmd_str)
    
    sql(f"ALTER TABLE {schema1}.{pts_table} ADD PRIMARY KEY (country_key, gid, id)")
    #check
    """should match all points... pretty sure this query works"""
    null_cnt = pg_exe(f"""SELECT COUNT(*) FROM {schema1}.{pts_table} WHERE f500_fluvial IS NULL""", return_fetch=True)[0][0]              
    if not null_cnt==0:
        raise IOError(f'got {null_cnt} f500_fluvial nulls') #usually happens when the left join fails
 
    
    
    log.info(f'finished w/ {pts_table}')
 
    #===========================================================================
    # calc bldg stats------
    #===========================================================================    
    #===========================================================================
    # params
    #===========================================================================
    tableName = f'grid_bldg_stats_{country_key}_{grid_size:04d}'
    
    if dev:
        schema='dev'        
    else:
        schema='expo'
        
    #===========================================================================
    # query
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
    
    cmd_str = f'CREATE TABLE {schema}.{tableName} AS'
    
    #columns
    cols =', '.join(keys_d['grid']) #indexers
    cols+=', COUNT(id) as bldg_cnt, '
    
 
    cols+=', \n'.join([f'COUNT(CASE WHEN {e} > 0 THEN 1 ELSE NULL END) as {e}_wetcnt' for e in haz_coln_l])
    #cols+= ', '.join([f'CAST(AVG({e}) AS int) AS {e}_mean' for e in haz_coln_l])
    
    #groupers
    gcols = ', '.join(keys_d['grid'])
    
    cmd_str+= f"""
        SELECT {cols}
            FROM {schema1}.{pts_table}
                GROUP BY {gcols}
            """
    
    sql(cmd_str)
       
    #===========================================================================
    # post
    #===========================================================================
    key_str = ', '.join(keys_d['grid'])
    sql(f"ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({key_str})")
    
    #comment
    cmt_str = f'grids with exposed buildings, building counts, and wet counts per hazard\n'
    cmt_str+=f'table_left={table_left}\ntable_right={table_right}\n'
    cmt_str += f'built with {os.path.realpath(__file__)} at '+datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pg_comment(schema, tableName, cmt_str)
            
     
    #clean up
    pg_vacuum(schema, tableName)
    
    #drop the temps
    sql(f"DROP TABLE IF EXISTS {schema1}.{pts_table}")
    
    #===========================================================================
    # add a view with the geometry-----
    #===========================================================================
    if add_view:
        create_view_join_grid_geom(schema, tableName, country_key, log=log, dev=dev, conn_str=conn_str)
    
    #===========================================================================
    # wrap-------
    #===========================================================================
    log.info(f'\n\nwrap')
    

   
    
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(f'finished on \'{tableName}\' w/ \n    {meta_d}')
    
    return tableName


 
    
        
def run_all(ck, **kwargs):
    log = init_log(name='occu')
    
    for grid_size in gridsize_default_l:
        run_expo_stats_grouped(ck, grid_size, log=log, **kwargs)
        

if __name__ == '__main__':
    
    #run_expo_stats_grouped('deu', 1020, dev=False)
    
    run_all('deu', dev=False)
 
    
    
    
    
    
    
    
    
    
    
    