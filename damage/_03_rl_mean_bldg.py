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
#print('psycopg2.__version__=' + psycopg2.__version__)

from sqlalchemy import create_engine, URL

from tqdm import tqdm

import pandas as pd

from coms import (
    init_log, today_str, get_directory_size,dstr, view,  
    ) 

from agg.coms_agg import (
    get_conn_str, pg_vacuum, pg_spatialIndex, pg_exe, pg_get_column_names, pg_register, pg_getcount
    )
 

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l, haz_label_d
    )



def run_bldg_rl_means(
        
        country_key, grid_size,
 
 
       
       #index_fp=None,
                               
        out_dir=None,
        conn_str=None,
 
         log=None,
         dev=False,
 
        ):
    """join mean building losses (grouped by grids) to the grid losses"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    schema='damage'
    start=datetime.now()   
    
    country_key=country_key.lower() 
    
 
    #log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'rl_mean')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    #===========================================================================
    # setup table
    #===========================================================================
    tableName=f'rl_mean_{country_key}_{grid_size:04d}'
    
 
    
    """includes multiple df_id and hazard types"""
    
    func_coln_l = pg_get_column_names('damage', f'rl_{country_key}_bldgs' )
    
    func_coln_l = list(set(func_coln_l).difference(['haz_key', 'country_key', 'id']))
    
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName}')
    sql(f'DROP TABLE IF EXISTS temp.{tableName}')
    
    #===========================================================================
    # calc average building loss per grid
    #===========================================================================
    table_left = f'damage.rl_{country_key}_bldgs'
    if dev:        
        table_left = f'(SELECT * FROM {table_left} LIMIT 100)'
 
        
        
    #===========================================================================
    # coln_l = 'tleft.id, tleft.country_key, tleft.haz_key, tleft.' + ', tleft.'.join(func_coln_l) + f', tright.i, tright.j'
    # 
    # """I dont think ALTER TABLE works (also probably a bad idea)"""
    # cmd_str=f"""
    #     CREATE TABLE {schema}.{tableName} AS
    #         SELECT {coln_l}
    #             FROM {table_left} AS tleft
    #                 LEFT JOIN inters_agg.pts_osm_fathom_{country_key}_{grid_size:07d} AS tright
    #                     ON tleft.id=tright.id
    # """
    #===========================================================================
    
    coln_l = 'tleft.country_key, tright.grid_size, tleft.haz_key, tright.i, tright.j, ' + ', '.join([f'AVG({e}) AS {e}_mean' for e in func_coln_l])
    
    
    """I dont think ALTER TABLE works (also probably a bad idea)"""
    """this query os slow"""
    
    #get grouper columns (i,j) from inters_agg, then use these to compute grouped means
    cmd_str=f"""
        CREATE TABLE temp.{tableName} AS
            SELECT {coln_l}
                FROM {table_left} AS tleft
                    LEFT JOIN inters_agg.pts_osm_fathom_{country_key}_{grid_size:07d} AS tright
                        ON tleft.id=tright.id
                            GROUP BY tleft.country_key,grid_size, haz_key, i, j
    """
    
    sql(cmd_str)
    
    row_cnt = pg_getcount('temp', tableName)
    log.info(f'built table of averages w/ {row_cnt} entries in  %.2f secs'%(datetime.now() - start).total_seconds())
    
    #===========================================================================
    # join grid losses to this
    #===========================================================================
    """this results in many nulls as grid centroids hit less often t han the buildings
    
    using a normal join so we capture nulls on either
    """
    """ NO... use the above which is already sliced as the left
    table_left = f'damage.rl_{country_key}_grid_{grid_size:04d}'
    if dev:        
        table_left = f'(SELECT * FROM {table_left} LIMIT 100)'"""
    
    start_i = datetime.now()
 
    coln_l = f'tleft.*, tright.' + ', tright.'.join(func_coln_l)
    
    cmd_str = f"""
        CREATE TABLE {schema}.{tableName} AS
            SELECT {coln_l}
                FROM temp.{tableName} AS tleft
                    JOIN damage.rl_{country_key}_grid_{grid_size:04d} AS tright
                        ON tleft.i=tright.i AND tleft.j=tright.j
        """
    
    #print(cmd_str)
    sql(cmd_str)
    row_cnt = pg_getcount('temp', tableName)
    log.info(f'joined grid centroid losses w/ {row_cnt} entries in  %.2f secs'%(datetime.now() - start_i).total_seconds())
    
    #===========================================================================
    # clean
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS temp.{tableName}')
    
    log.info(f'cleaning {tableName} w/ {row_cnt} rows')
    try:
        pg_vacuum(schema, tableName)
        """table is a-spatial"""
        #pg_spatialIndex(schema, tableName, columnName='geometry')
        #pg_register(schema, tableName)
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
    log.info(f'finishedw/ \n{meta_d}')
    
    return
    

def run_extract_haz(
        country_key, haz_key,
        grid_size_l=None,
        log=None,
        conn_str=None,
        dev=True,
        out_dir=None,
        chunksize=1e6,
        ):
    """for plotting, we need to slice to a single hazard, then join across tables"""
        

    if grid_size_l is None: grid_size_l = gridsize_default_l
    
    #===========================================================================
    # defaults
    #===========================================================================
    schema='damage'
    start=datetime.now()   
    
    country_key=country_key.lower() 
    
 
    #log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'rl_mean')
        
        
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'damage','03_mean', country_key, haz_key)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
        
        
    #===========================================================================
    # build zuery
    #===========================================================================
    #===========================================================================
    # no... dont want a new table... just download
    # tableName = f'rl_mean_{country_key}_{haz_key}'
    # sql(f'DROP TABLE IF EXISTS damage.{tableName}')
    #===========================================================================
    
    cmd_str = ''
    
    
 
    log.info(f'on {grid_size_l}')
    
    first = True
    for grid_size in grid_size_l:
        
        
        tableName_i=f'rl_mean_{country_key}_{grid_size:04d}'
 
        
        log.info(f'on {grid_size} w/ \'{tableName_i}\'')
        
        if not first:
            cmd_str+='UNION\n'
        

 
        cmd_str +=f'SELECT * FROM {schema}.{tableName_i}\n'
            
        cmd_str +=f'    WHERE {tableName_i}.haz_key=\'{haz_key}\'\n'
        

        
        first=False
    
    if dev:
        cmd_str+=f'        LIMIT 100\n'
        
    #===========================================================================
    # download
    #===========================================================================
    conn =  psycopg2.connect(conn_str)
    engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
    
    #row_cnt=0
    
    """only ~600k rows"""
    #===========================================================================
    # for i, gdf in enumerate(pd.read_sql(cmd_str, engine, index_col=None, chunksize=int(chunksize))):
    #     log.info(f'{i} w/ {len(gdf)}')
    #     row_cnt+=len(gdf)
    #===========================================================================
    log.info(cmd_str)
    df = pd.read_sql(cmd_str, engine, 
                     index_col=['country_key', 'haz_key', 'i', 'j'],
                     )
    """
    view(df)
    """    
    
    engine.dispose()
    conn.close()
    
    log.info(f'finished w/ {len(df)} total rows')
    
    #===========================================================================
    # clean up
    #===========================================================================
 
    
    col_bx = df.columns.str.contains('_mean')
    
    dx = pd.concat({
        'bldg_mean':df.loc[:, col_bx].rename(columns={k:int(k.split('_')[1]) for k in df.columns[col_bx].values}), 
        'grid_cent':df.loc[:, ~col_bx].rename(columns={k:int(k.split('_')[1]) for k in df.columns[~col_bx].values}), 
        }, 
        names = ['rl_type', 'df_id'], axis=1)
    
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'rl_mean_{country_key}_{haz_key}_{len(df)}_{today_str}.pkl')
    dx.sort_index(sort_remaining=True).to_pickle(ofp)
    
    log.info(f'wrote {df.shape} to \n    {ofp}')
 
 
    
    #===========================================================================
    # wrap
    #===========================================================================
        #meta
    meta_d = {
        'tdelta':(datetime.now() - start).total_seconds(), 
        'RAM_GB':psutil.virtual_memory()[3] / 1000000000, 
        #'postgres_GB':get_directory_size(postgres_dir)}
        'output_MB':os.path.getsize(ofp)/(1024**2)
        }
    log.info(f'finishedw/ \n{meta_d}')
    
    return ofp
        
        
        
        
        
        
        
if __name__ == '__main__':
    run_bldg_rl_means('deu', 60, dev=True)
    
    #run_extract_haz('deu', 'f500_fluvial')
    
        
    