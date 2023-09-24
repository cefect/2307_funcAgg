'''
Created on Sep. 21, 2023

@author: cefect
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
 
 

 

from definitions import (
    wrk_dir,   postgres_d, 
    equal_area_epsg, fathom_vals_d, gridsize_default_l
    )
 

from _02agg.coms_agg import (
    get_conn_str, pg_getCRS, pg_to_df, pg_exe, pg_getcount, pg_spatialIndex, pg_get_column_names,
    pg_vacuum, pg_comment, pg_register
    )

from _02agg._07_views import create_view_join_grid_geom

from coms import (
    init_log, today_str,  dstr, view
    )


def create_view_merge_stats(country_key, haz_key,
                         grid_size_l=None,
         dev=False, conn_str=None,
         with_geom=False,
         log=None,
        ):
    
    """create a view by unioning all the grid stats and selecting wet_cnts from one haz
    
    useful for joins later"""
    

    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()  
    
    if log is None: log = init_log(name=f'view')
    
    if grid_size_l is None: grid_size_l = gridsize_default_l
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #===========================================================================
    # table params
    #===========================================================================
    keys_l = ['country_key', 'grid_size', 'i', 'j', 'haz_key']
    if dev:
        schema = 'dev'
    else:
        schema='expo'
 
    tableName=f'grid_bldg_stats_{country_key}_{haz_key}'
    
    #===========================================================================
    # query
    #===========================================================================
    sql(f'DROP VIEW IF EXISTS {schema}.{tableName} CASCADE')
    
    cmd_str = f'CREATE VIEW {schema}.{tableName} AS \n'
    
    first = True
    for grid_size in grid_size_l:        
        tableName_i = f'grid_bldg_stats_{country_key}_{grid_size:04d}'
                
        if not first:
            cmd_str += 'UNION\n'
        else:
            #build column selection (only want 1 hazard)
            full_coln_l = pg_get_column_names(schema, tableName_i)
            haz_col = [e for e in full_coln_l  if haz_key in e][0] #column name of wet counts we want
            keep_coln_l = [e for e in full_coln_l  if not e.endswith('wetcnt')] #other columns            
            
            #assemble
            cols = ', '.join(keep_coln_l)
            cols+=f', {haz_col} as wet_cnt'
 
        
        cmd_str += f'SELECT {cols}\n'
        cmd_str += f'FROM {schema}.{tableName_i} \n'
        
        """no haz_key column. each row has wet stats for each haz key
        cmd_str += f'WHERE {tableName_i}.haz_key=\'{haz_key}\' \n'"""
        
        
        # filters        
        first = False
    
    cmd_str+=f'ORDER BY grid_size, i, j\n'
    sql(cmd_str)
    
    #===========================================================================
    # join geometry
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
    log.info(f'finished on {tableName} w/ \n{meta_d}\n\n')
    
    return tableName





def run_all(ck='deu', e='f500_fluvial', **kwargs):
    log = init_log(name=f'expo.views')
    
    create_view_merge_stats(ck, e, log=log, **kwargs)
    
 
    

if __name__ == '__main__':
 
    #create_view_merge_stats('deu', 'f500_fluvial', dev=True)
    
    #create_view_join_stats_to_rl('deu', 'f500_fluvial', dev=False, with_geom=False)
    
    #get_grid_rl_dx('deu', 'f500_fluvial', dev=False, use_cache=False, limit=None)
    
    
    run_all(dev=False)
    
    
    
    
    
    
    
    
    



    
 