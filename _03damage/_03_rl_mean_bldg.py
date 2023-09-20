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
import numpy as np

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
    if dev:
        tableName+='_dev'
    
 
    
    """includes multiple df_id and hazard types"""
    
    func_coln_l = pg_get_column_names('damage', f'rl_{country_key}_bldgs' )
    
    func_coln_l = list(set(func_coln_l).difference(['haz_key', 'country_key', 'id']))
    
    
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
    """because join the grid-id table to the rl table (w/ a left join) we can only query wet buildings?"""
    
    cols = 'tleft.country_key, tright.grid_size, tleft.haz_key, tright.i, tright.j, COUNT(tleft.id) AS wet_cnt, ' 
    cols+= ', '.join([f'AVG(tleft.{e}) AS {e}_mean' for e in func_coln_l])
    
    """not too worried about nulls as we filter for those with high wetted fraction"""
    
    raise IOError(f'i think we need to pre-compute a dedicated talbe of bldg and wet counts')
    
    
    """I dont think ALTER TABLE works (also probably a bad idea)"""
    """this query can be slow"""
    
    #get grouper columns (i,j) from inters_agg, then use these to compute grouped means
    cmd_str=f"""
        CREATE TABLE temp.{tableName} AS
            SELECT {cols}
                FROM {table_left} AS tleft
                    LEFT JOIN inters_agg.pts_osm_fathom_{country_key}_{grid_size:07d} AS tright
                        ON tleft.id=tright.id
                            GROUP BY tleft.country_key,grid_size, haz_key, i, j
    """
    
    #print(cmd_str)
    sql(cmd_str)
    
    row_cnt = pg_getcount('temp', tableName)
    log.info(f'built table of averages w/ {row_cnt} entries in  %.2f secs'%(datetime.now() - start).total_seconds())
    
    #check the keys are unique
    sql(f'ALTER TABLE temp.{tableName} ADD PRIMARY KEY (haz_key, i,j)')
    
    #===========================================================================
    # join bldg counts
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS temp.{tableName}2')
    
    table_right = f'inters_agg.pts_osm_fathom_{country_key}_{grid_size:07d}'
    if dev:        
        table_right = f'(SELECT * FROM {table_right} LIMIT 1000)'
    
    cmd_str=f"""
        CREATE TABLE temp.{tableName}2 AS
            SELECT tleft.*, COUNT(tright.id) AS bldg_cnt
                FROM temp.{tableName} AS tleft
                    LEFT JOIN {table_right} AS tright
                        ON tleft.i=tright.i AND tleft.j=tright.j AND tleft.country_key=LOWER(tright.country_key)
                            GROUP BY tleft.country_key, tleft.grid_size,  tleft.haz_key, tleft.i, tleft.j
                        
            """
    sql(cmd_str)
    
    sql(f'ALTER TABLE temp.{tableName}2 ADD PRIMARY KEY (haz_key, i,j)')
    #===========================================================================
    # join grid losses to this
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName}')
    """
    NOTES:
        using FULL OUTER so we capture both wet/dry combinations of assets/grids
        
        during dev (because we do not limit the outer join), this gives many grid centroid nulls
    
    
    """
 
    
    start_i = datetime.now()
    
    #get left columns
    index_coln_l = ['haz_key', 'i', 'j', 'country_key', 'grid_size']
    coln_l = [e for e in pg_get_column_names('temp', f'{tableName}2') if not e in index_coln_l]
    
    
    
    #cols = ', '.join([f'COALESECE(tleft.{e}, tright.{e}) as {e}' for e in index_coln_l])
    cols="""COALESCE(CAST(tleft.i AS INTEGER), CAST(tright.i AS INTEGER)) as i,
        COALESCE(CAST(tleft.j AS INTEGER), CAST(tright.j AS INTEGER)) as j,
        COALESCE(tleft.haz_key, tright.haz_key) as haz_key,
        COALESCE(tleft.country_key, tright.country_key) as country_key,
        COALESCE(tleft.grid_size, tright.grid_size) as grid_size
        """
    cols+=', tleft.'+', tleft.'.join(coln_l)
    cols +=', tright.' + ', tright.'.join(func_coln_l)
    
    cmd_str = f"""
        CREATE TABLE {schema}.{tableName} AS
            SELECT {cols}
                FROM temp.{tableName}2 AS tleft
                    FULL OUTER JOIN damage.rl_{country_key}_grid_{grid_size:04d} AS tright
                        ON tleft.i=tright.i AND tleft.j=tright.j AND tleft.haz_key=tright.haz_key
 
        """
    
    #print(cmd_str)
    sql(cmd_str)
    row_cnt = pg_getcount('temp', tableName)
    log.info(f'joined grid centroid losses w/ {row_cnt} entries in  %.2f secs'%(datetime.now() - start_i).total_seconds())
    
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY (haz_key, i,j)')
 
    """wet counts are bad"""
    #===========================================================================
    # clean
    #===========================================================================
    #sql(f'DROP TABLE IF EXISTS temp.{tableName}')
    #sql(f'DROP TABLE IF EXISTS temp.{tableName}2')
    
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
        dev=False,
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
        
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    #===========================================================================
    # concat
    #===========================================================================
 
    tableName = f'rl_mean_{country_key}_{haz_key}'
    sql(f'DROP TABLE IF EXISTS temp.{tableName}')
    
    cmd_str = f'CREATE TABLE temp.{tableName} AS\n'
 
    log.info(f'on {grid_size_l}')
    
    #union and slice
    first = True
    for grid_size in grid_size_l:
        
        tableName_i = f'rl_mean_{country_key}_{grid_size:04d}'
        
        log.debug(f'on {grid_size} w/ \'{tableName_i}\'')
        
        if not first:
            cmd_str += 'UNION\n'
 
        cmd_str += f'SELECT * FROM {schema}.{tableName_i}\n'
            
        
        #filters
        cmd_str += f'    WHERE {tableName_i}.haz_key=\'{haz_key}\' '
        #those with at least 2 wet houses
        cmd_str +=f'AND {tableName_i}.wet_cnt>1 '
        #at least some grid damage
        cmd_str +=f'AND {tableName_i}.dfid_0026>0 \n'
        
        
        first = False
        
    #join depths
    #cmd_str +=f'LEFT JOIN inters_grid.grid_wd_{country_key}_{haz_key} ON '
    
    if dev:
        cmd_str += f'        LIMIT 100\n'
        
        
    sql(cmd_str)
    
    """bad counts"""
    
    #===========================================================================
    # join water depths
    #===========================================================================
    cmd_str = f"""
        SELECT tleft.*, tright.{haz_key} AS grid_wd
            FROM temp.{tableName} AS tleft
                LEFT JOIN inters_grid.grids_wd_{country_key}_{haz_key} AS tright
                    ON tleft.i=tright.i 
                        AND tleft.j=tright.j 
                        AND tleft.country_key=LOWER(tright.country_key)
                        AND tleft.grid_size=tright.grid_size
            """
        
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
    df_raw = pd.read_sql(cmd_str, engine, 
                     index_col=['country_key', 'grid_size','haz_key', 'i', 'j'],
                     )
    """
    view(df_raw.head(100))
 
    
    """    
    
    engine.dispose()
    conn.close()
    
    log.info(f'finished w/ {len(df_raw)} total rows')
    
    #===========================================================================
    # clean up
    #===========================================================================
    #exposure meta
    expo_colns = ['wet_cnt', 'bldg_cnt', 'grid_wd']
    df1 = df_raw.copy()
    df1.loc[:, expo_colns] = df1.loc[:, expo_colns].fillna(0.0)
    
    df1=df1.set_index(expo_colns, append=True)
    
    #split bldg and grid losses
    col_bx = df1.columns.str.contains('_mean') 
    
    grid_dx = df1.loc[:, ~col_bx]
    rnm_d = {k:int(k.split('_')[1]) for k in grid_dx.columns.values}
    grid_dx = grid_dx.rename(columns=rnm_d)
    grid_dx.columns = grid_dx.columns.astype(int)
    
    
    bldg_dx = df1.loc[:, col_bx]
    rnm_d = {k:int(k.split('_')[1]) for k in bldg_dx.columns.values}
    bldg_dx = bldg_dx.rename(columns=rnm_d)
    bldg_dx.columns = bldg_dx.columns.astype(int)
    
    assert np.array_equal(grid_dx.columns, bldg_dx.columns)
 
    
    dx = pd.concat({
        'bldg_mean':bldg_dx, 
        'grid_cent':grid_dx, 
        #'expo':df.loc[:, expo_colns].fillna(0.0)
        }, 
        names = ['rl_type', 'df_id'], axis=1).dropna(how='all').fillna(0.0)
    
    #===========================================================================
    # write
    #===========================================================================
    
    ofp = os.path.join(out_dir, f'rl_mean_{country_key}_{haz_key}_{len(dx)}_{today_str}.pkl')
    log.info(f'writing {dx.shape} to \n    {ofp}')
    dx.sort_index(sort_remaining=True).to_pickle(ofp)
    
 
 
 
    
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
    
    #run_extract_haz('deu', 'f500_fluvial', dev=False)
    
        
    
