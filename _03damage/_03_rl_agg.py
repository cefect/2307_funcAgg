"""

mean asset loss per grid
"""


#===============================================================================
# IMPORTS-----
#===============================================================================
import os, hashlib, sys, subprocess, psutil, winsound
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
    pg_comment, pg_table_exists, pg_get_nullcount, pg_get_nullcount_all, pg_to_df, pg_get_meta
    )
 

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l, haz_label_d
    )


def create_table_join_links(country_key='deu',conn_str=None,log=None,dev=False):
    """join links to building losses
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    
    country_key = country_key.lower() 
 
    if conn_str is None: conn_str = get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'rl_mean')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #===========================================================================
    # table params
    #===========================================================================
    #source table keys
    keys_d = {
        'bldg':['country_key', 'gid', 'id'], 
        'grid':['country_key', 'grid_size', 'i', 'j']}
    
    
    
    table_bldg = f'rl_{country_key}_bldgs' #building losses
    
    tableName = table_bldg+'_link'
    
 
    table_link = f'a01_links_1x_{country_key}'
    if dev:
 
        schema_bldg = 'dev'
        schema_link = schema_bldg
    else:
 
        schema_bldg = 'damage'
        schema_link = 'wd_bstats'
    schema_grid = schema_bldg
    assert pg_table_exists(schema_bldg, table_bldg)
    assert pg_table_exists(schema_link, table_link)
    schema='temp'
 
    #===========================================================================
    # #setup
    #===========================================================================
 
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
#execute
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_d['bldg']])
    
    sql(f"""
    CREATE TABLE {schema}.{tableName} AS
        SELECT tleft.grid_size, tleft.i, tleft.j, tright.*
                FROM {schema_link}.{table_link} as tleft
                    LEFT JOIN {schema_bldg}.{table_bldg} as tright
                        ON {link_cols}""")
    
    
    keys_str = ', '.join(keys_d['bldg'] + ['haz_key', 'grid_size'])
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
#check
    assert pg_get_nullcount_all(schema, tableName).max() == 0
    
    log.info(f'finished on {schema}.{tableName}')
    return schema, tableName


def create_table_agg_bldg_rl(schema_right, table_right,  country_key='deu',conn_str=None,log=None,dev=False):
    """join grid lossese and aggregate 
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    
    country_key = country_key.lower() 
 
    if conn_str is None: conn_str = get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'rl_mean')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #===========================================================================
    # table params
    #===========================================================================
 
    #source table keys
    keys_d = {
        'bldg':['country_key', 'gid', 'id'], 
        'grid':['country_key', 'grid_size', 'i', 'j']}
    if dev:
        schema = 'dev'
        schema_left = 'dev'
    else:
        schema = 'damage'
        schema_left = 'damage'
        
    tableName = f'rl_{country_key}_agg'
    table_left = f'rl_{country_key}_grid'
    
    #===========================================================================
    # prep
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
    kl = keys_d['grid'] + ['haz_key']
    
#retrieve function column names
    func_coln_l = list(set(pg_get_column_names(schema_right, table_right)).difference(kl + keys_d['bldg']))
#build columns
#grid keys
#cols =', '.join([f'tleft.{e}' for e in kl])
    cols = 'tleft.*'
    cols += f', COUNT(tright.id) AS bldg_expo_cnt, ' #we only have exposed buildings so this isnt very useful
    cols += ', '.join([f'CAST(AVG(COALESCE(tright.{e}, 0)) AS real) AS {e}_mean' for e in func_coln_l])
    """not too worried about nulls as we filter for those with high wetted fraction"""
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in kl])
    grp_cols = ', '.join([f'tleft.{e}' for e in kl])
    
#get grouper columns (i,j) from inters_agg, then use these to compute grouped means
    cmd_str = f"""
    CREATE TABLE {schema}.{tableName} AS
        SELECT {cols}
            FROM {schema_left}.{table_left} AS tleft
                LEFT JOIN {schema_right}.{table_right} AS tright
                    ON {link_cols}
                        GROUP BY {grp_cols}"""
    sql(cmd_str)
#===========================================================================
# clean
#===========================================================================
#check the keys are unique
    keys_str = ', '.join(kl)
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
    
    
    return schema, tableName


def create_table_join_wd_gstats(schema_left, table_left,  country_key='deu',conn_str=None,log=None,dev=False):
    """join depth group stats
    """
    raise IOError('wont work as the gstats table needs to be melted (on haz_key)... hard to do this in postgres')
    #===========================================================================
    # defaults
    #===========================================================================
    
    
    country_key = country_key.lower() 
 
    if conn_str is None: conn_str = get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'rl_mean')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #===========================================================================
    # table params
    #===========================================================================
    
    keys_d = {
        'bldg':['country_key', 'gid', 'id'], 
        'grid':['country_key', 'grid_size', 'i', 'j']}
 
    if dev:
        schema = 'dev'
        schema_right = 'dev'
    else:
        schema = 'damage'
        schema_right = 'wd_bstats'

    #depth stats grouped on grids
    #_05depths._03_gstats.run_pg_build_gstats()
    table_right =f'a03_gstats_1x_{country_key}'

    tableName=f'rl_{country_key}_agg'
        
 
    #===========================================================================
    # #set up
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
    #===========================================================================
    # #build query
    #===========================================================================
    #depth
    wd_cols = [e for e in pg_get_column_names(schema_right, table_right) if e.startswith('f')]
 
    cols = 'tleft.*, '
    cols += ', '.join([f'tright.{e}' for e in schema_right])
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_d['grid']])
#===========================================================================
# #execute
#===========================================================================
    sql(f"""
    CREATE TABLE {schema}.{tableName} AS
        SELECT {cols}
            FROM {schema_left}.{table_left} as tleft
                LEFT JOIN {schema_right}.{table_right} as tright
                    ON {link_cols}
                    """)
#===========================================================================
# clean
#===========================================================================
#check the keys are unique
    keys_str = ', '.join(keys_d['grid'])
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
    
    
    return schema, tableName



def run_join_agg_rl(
        
        country_key='deu',  
 
        conn_str=None,
 
         log=None,
         dev=False,
 
        ):
    """join mean building losses (grouped by grids) to the grid losses via aggregation
    
    
    for a similar routine against the depths, see _05depths._03_gstats
    

    
    
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
    
    skwargs = dict(log=log, dev=dev, conn_str=conn_str, country_key=country_key)
    
    res_d = dict()
    #===========================================================================
    # join links to building relative losses
    #===========================================================================
    s,t = create_table_join_links(**skwargs)
    res_d['linkd'] = {'schema':s, 'table':t}
    
    #===========================================================================
    # aggregate
    #===========================================================================
    schema, tableName = create_table_agg_bldg_rl(s, t, **skwargs)
    res_d['agg'] = {'schema':s, 'table':t}
    
    #add comment
    
    #===========================================================================
    # join building group depth stats
    #===========================================================================
    """ just do this in pandas
    create_table_join_wd_gstats(s, t, **skwargs)"""

    
    
    #===========================================================================
    # post
    #===========================================================================
 
    
    cmt_str = f'join mean building losses (grouped by grids) to the grid losses\n'
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


def load_rl_dx(
        country_key='deu', 
 
        log=None,conn_str=None,dev=False,use_cache=True,out_dir=None,
 
        use_aoi=False,
        
        ):
    """load the relative loss dx
    
 
    """
    
    #===========================================================================
    # defaults
    #===========================================================================    
    country_key = country_key.lower() 
 
    if conn_str is None: conn_str = get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'rl_mean')
    
    #sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'damage','03_rl_agg', country_key)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    
    keys_l = ['country_key', 'haz_key', 'grid_size', 'i', 'j']
 
    #===========================================================================
    # #table params
    #===========================================================================
    if dev:
        schema='dev'
    else:
        schema='damage'
        
    tableName=f'rl_{country_key}_agg'
    
    #===========================================================================
    # cache
    #===========================================================================
    meta = pg_get_meta(schema, tableName)
    
        
    fnstr = f'rl_agg_{country_key}'
    uuid = hashlib.shake_256(f'{fnstr}_{dev}_{meta}'.encode("utf-8"), usedforsecurity=False).hexdigest(12)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')
    
    if (not os.path.exists(ofp)) or (not use_cache):
        #===========================================================================
        # #load
        #===========================================================================
        dx_raw = pg_to_df(f"""SELECT * FROM {schema}.{tableName}""", conn_str=conn_str, index_col=keys_l)
        
        """
        rl_dx.columns
        """
        #===========================================================================
        # clean up
        #===========================================================================
        dx1 = dx_raw.drop('bldg_expo_cnt', axis=1) #not needed anymore
     
     
        
        #split bldg and grid losses
        col_bx = dx1.columns.str.contains('_mean') 
        
        #grid losses
        grid_dx = dx1.loc[:, ~col_bx]
        rnm_d = {k:int(k.split('_')[1]) for k in grid_dx.columns.values}
        grid_dx = grid_dx.rename(columns=rnm_d).sort_index(axis=1)
        grid_dx.columns = grid_dx.columns.astype(int)
        
        
        bldg_dx = dx1.loc[:, col_bx]
        rnm_d = {k:int(k.split('_')[1]) for k in bldg_dx.columns.values}
        bldg_dx = bldg_dx.rename(columns=rnm_d).sort_index(axis=1)
        bldg_dx.columns = bldg_dx.columns.astype(int)
        
        assert np.array_equal(grid_dx.columns, bldg_dx.columns)
        
        
        dx2 = pd.concat({
            'bldg':bldg_dx, 
            'grid':grid_dx, 
            #'expo':df.loc[:, expo_colns].fillna(0.0)
            }, 
            names = ['rl_type', 'df_id'], axis=1).dropna(how='all') 
        
        #===========================================================================
        # wrap
        #===========================================================================
     
        res_dx = dx2.sort_index(sort_remaining=True).sort_index(sort_remaining=True, axis=1)
        log.info(f'finished on {res_dx.shape}')
        
    else:
        log.info(f'loading from cache:\n    {ofp}')
        res_dx = pd.read_pickle(ofp)
    
    return res_dx 
    
    """
    view(rl_dx.head(100))
    """
    

    
        
        
        
if __name__ == '__main__':
    #run_all('deu', dev=True)
    #run_join_agg_rl(dev=False)
    
    load_rl_dx(dev=True)
    
    #run_extract_haz('deu', 'f500_fluvial', dev=False)
    
    print('done')
    winsound.Beep(440, 500)
    
        
    
