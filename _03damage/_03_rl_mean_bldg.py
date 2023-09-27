"""

mean asset loss per grid
"""


#===============================================================================
# IMPORTS-----
#===============================================================================
import os, hashlib, sys, subprocess, psutil
from datetime import datetime
from itertools import product

import psycopg2
from sqlalchemy import create_engine, URL
#print('psycopg2.__version__=' + psycopg2.__version__)



from tqdm import tqdm

import pandas as pd
import numpy as np

from coms import (
    init_log, today_str, get_directory_size,dstr, view,  
    ) 

from _02agg.coms_agg import (
    get_conn_str, pg_vacuum, pg_spatialIndex, pg_exe, pg_get_column_names, pg_register, pg_getcount,
    pg_comment, pg_table_exists, pg_get_nullcount
    )
 

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l, haz_label_d
    )



def run_bldg_rl_means(
        
        country_key, grid_size,
 
        conn_str=None,
 
         log=None,
         dev=False,
         sample_type='bldg_mean'
        ):
    """join mean building losses (grouped by grids) to the grid losses
    
    NOTE: the building loss means ignore nulls and zeros 
        because we pre-filtered these in _02agg.join 
    
    
    Returns
    -------
    postgres table
        damage.rl_mean_{country_key}_{grid_size:04d} 

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
    #source table keys
    keys_d = { 
        'bldg':['country_key', 'gid', 'id'],
        'grid':['country_key', 'grid_size', 'i', 'j']        
    }
 
        
    tableName=f'rl_mean_{country_key}_{grid_size:04d}' #output    
    
    table_bldg = f'rl_{country_key}_bldgs' #building losses
    
    #Link all buildings to  exposed grids for: 1x) grids w/ exposed buildings
    table_link = f'bldgs_grid_link_1x_{country_key}_{grid_size:04d}'
    
    if dev: 
        schema='dev'
        schema_bldg=schema        
        schema_link=schema
        
    else:
        schema='damage'
        schema_bldg='damage' 
        schema_link='expo'
        
    schema_grid = schema_bldg
    
    
    #grid losses by type
    if sample_type=='grid_cent':
        #centroid based losses
        table_grid=f'rl_{country_key}_grid_{grid_size:04d}' 
 
        
    elif sample_type=='bldg_mean':
        #building mean based losses
        table_grid=f'rl_{country_key}_grid_bmean_{grid_size:04d}' #grid losses
        
    else:
        raise IOError(sample_type)
    
 
    assert pg_table_exists(schema_bldg, table_bldg)
    assert pg_table_exists(schema_link, table_link)
    assert pg_table_exists(schema_grid, table_grid)
    #===========================================================================
    # join buidling losses to grid links------
    #===========================================================================
    #building losses table contains all buidlings selected by 1020 grid exposure, 
    raise IOError('building losses table is missing some entries')
    #join onto the link tables, which should shorten the list (for all but the 1020)
    #want to include all the zeros however
    
    #setup
    table_link1 = f'{table_bldg}_linkd'
    sql(f'DROP TABLE IF EXISTS temp.{table_link1} CASCADE')
    
    #execute
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_d['bldg']])
    
       
    sql(f"""
    CREATE TABLE temp.{table_link1} AS
        SELECT tleft.grid_size, tleft.i, tleft.j, tright.* 
            FROM {schema_link}.{table_link} as tleft
                LEFT JOIN {schema_bldg}.{table_bldg} as tright
                    ON {link_cols}
    
    """)
    
    keys_str = ', '.join(keys_d['bldg'] + ['haz_key'])
    sql(f'ALTER TABLE temp.{table_link1} ADD PRIMARY KEY ({keys_str})')
    
    #check
    null_cnt= pg_get_nullcount('temp', table_link1, 'dfid_0026')
    if not null_cnt==0:
        raise AssertionError(f'got {null_cnt} nulls on left join result')
    #===========================================================================
    # join average building loss to the grid losses------
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')

    kl = keys_d['grid'] + ['haz_key']
    #retrieve function column names
    func_coln_l = list(set(pg_get_column_names(schema_grid, table_grid)).difference(kl))
    
    #build columns
    
    #grid keys
    #cols =', '.join([f'tleft.{e}' for e in kl])
    cols = 'tleft.*'
     
    cols+=f', COUNT(tright.id) AS bldg_expo_cnt, '  #we only have exposed buildings so this isnt very useful
    cols+= ', '.join([f'CAST(AVG(COALESCE(tright.{e}, 0)) AS real) AS {e}_mean' for e in func_coln_l])
    
    cols +=f', \'{sample_type}\' as sample_type'
 
    """not too worried about nulls as we filter for those with high wetted fraction"""
    
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in kl])
    grp_cols = ', '.join([f'tleft.{e}' for e in kl])
    
    #get grouper columns (i,j) from inters_agg, then use these to compute grouped means
    cmd_str=f"""
        CREATE TABLE {schema}.{tableName} AS
            SELECT {cols}
                FROM {schema_grid}.{table_grid} AS tleft
                    LEFT JOIN temp.{table_link1} AS tright
                        ON {link_cols}
                            GROUP BY {grp_cols}
    """

    sql(cmd_str) 
    #===========================================================================
    # clean
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS temp.{table_link1}')
 
    #check the keys are unique
    keys_str = ', '.join(kl)
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
    
    
    #add comment
 
    source_d = dict(tableName=tableName,table_grid=table_grid, table_bldg=table_bldg, table_link=table_link )
    
    cmt_str = f'join mean building losses (grouped by grids) to the grid losses (sample_type={sample_type}) \n from tables: {source_d}\n'
    cmt_str += f'built with {os.path.realpath(__file__)} at '+datetime.now().strftime("%Y.%m.%d: %H.%M.%S")
    pg_comment(schema, tableName, cmt_str)
    
    log.info(f'cleaning {tableName} ')
    
    try:
        pg_vacuum(schema, tableName)
        """table is a-spatial"""
 
    except Exception as e:
        log.error(f'failed cleaning w/\n    {e}')
    
    #===========================================================================
    # wrap
    #===========================================================================
        #meta
    meta_d = {
        'tdelta':(datetime.now() - start).total_seconds(), 
        'RAM_GB':psutil.virtual_memory()[3] / 1000000000, 
        'postgres_GB':get_directory_size(postgres_dir)}
        #'output_MB':os.path.getsize(ofp)/(1024**2)
    log.info(f'finished w/ {schema}.{tableName}\n{meta_d}')
    
    return tableName


def run_all(ck, grid_size_l=None, **kwargs):
    log = init_log(name='mean')
    if grid_size_l is None: grid_size_l = gridsize_default_l
    
    res_d = dict()
    for grid_size in grid_size_l: 
        res_d[grid_size] = run_bldg_rl_means(ck, grid_size, log=log, **kwargs)
        
    log.info(f'finished w/ \n{dstr(res_d)}')
    
    return res_d
    

    
        
        
        
if __name__ == '__main__':
    run_all('deu', dev=True)
    #run_bldg_rl_means('deu', 1020, dev=True, sample_type='bldg_mean')
    
    #run_extract_haz('deu', 'f500_fluvial', dev=False)
    
        
    
