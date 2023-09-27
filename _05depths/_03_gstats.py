"""

computing group stats on building depths

made this to try and simplify and consolidate as it became too confusing to work with all the tables
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


from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l, haz_label_d
    )


from coms import (
    init_log, today_str, get_directory_size,dstr, view,  
    ) 

from _02agg.coms_agg import (
    get_conn_str, pg_vacuum, pg_spatialIndex, pg_exe, pg_get_column_names, pg_register, pg_getcount,
    pg_comment, pg_table_exists, pg_get_nullcount, pg_get_meta, pg_getCRS, pg_get_nullcount_all,
    pg_to_df
    )
 
from _02agg._07_views import create_view_join_grid_geom


#===============================================================================
# globals-------
#===============================================================================
keys_d = { 
        'bldg':['country_key', 'gid', 'id'],
        'grid':['country_key', 'grid_size', 'i', 'j']        
    }


def create_table_links_merge(grid_size_l, country_key, expo_str, keys_l, dev, log, conn_str=None):
    """merge all the links tables"""
    #===========================================================================
    # defaults
    #===========================================================================
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    

    tableName = f'a01_links_{expo_str}_{country_key}' # output
    if dev:
        schema = 'dev'
        source_schema='dev'
    else:
        source_schema='expo'
        schema = 'wd_bstats'
    #===========================================================================
    # prep
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
    #===========================================================================
    # build
    #===========================================================================
    cmd_str = f'CREATE TABLE {schema}.{tableName} AS \n'
    first = True
    source_d = dict()
    for grid_size in grid_size_l:
        
        #buidling to grid links
        #see _04expo._01_full_links.run_agg_bldg_full_links()
        
        tableName_i = f'bldgs_grid_link_{expo_str}_{country_key}_{grid_size:04d}'
        source_d[grid_size] = tableName_i
        assert pg_table_exists(source_schema, tableName_i, asset_type='table'), f'missing {schema}.{tableName_i}'
        if not first:
            cmd_str += 'UNION\n'
        else:
            cols = '*'
        cmd_str += f'SELECT {cols}\n'
        cmd_str += f'FROM {source_schema}.{tableName_i} \n'
        # filters
        first = False
    
    cmd_str += f'ORDER BY grid_size, i, j\n'
#===========================================================================
# exe
#===========================================================================
    sql(cmd_str)
#===========================================================================
# post
#===========================================================================
    keys_str = ', '.join(keys_l)
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
#add comment
    cmt_str = f'merged links from\n    {dstr(source_d)} \n'
    cmt_str += f'built with {os.path.realpath(__file__)} at ' + datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pg_comment(schema, tableName, cmt_str)
    log.info(f'cleaning {tableName} ')
    try:
        pg_vacuum(schema, tableName)
        """table is a-spatial"""
    #pg_spatialIndex(schema, tableName, columnName='geometry')
    #pg_register(schema, tableName)
    except Exception as e:
        log.error(f'failed cleaning w/\n    {e}')
    log.info(f'finished building {schema}.{tableName}')
    return schema,  tableName


def create_table_joinL_bldg_wd(tableName, table_left, country_key, dev, keys_l, 
                               haz_key_l=None, conn_str=None, log=None):
    """expand building water depths onto the merged link table"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    if dev:
        schema = 'dev'
        schema_bldg = schema
    else:
        schema = 'wd_bstats'
        schema_bldg = 'expo'

    #buildings that occur in any of the link tables (w/ water depths joined
    #_04expo._01_full_links.run_expo_bldg()
    table_bldg = f'bldg_expo_wd_{country_key}'

    
    if haz_key_l is None: 
        haz_key_l = [e for e in pg_get_column_names(schema_bldg, table_bldg) if e.startswith('f')]
        
    log.info(f'w/ haz_key_l:\n    {haz_key_l}')
    #===========================================================================
    # #set up
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
    #===========================================================================
    # #build query
    #===========================================================================
 
    cols = 'tleft.*, '
    cols += ', '.join([f'tright.{e}' for e in haz_key_l])
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_d['bldg']])
#===========================================================================
# #execute
#===========================================================================
    sql(f"""
    CREATE TABLE {schema}.{tableName} AS
        SELECT {cols}
            FROM {schema}.{table_left} as tleft
                LEFT JOIN {schema_bldg}.{table_bldg} as tright
                    ON {link_cols}
                    """)
    
    #===========================================================================
    # post
    #===========================================================================
    keys_str = ', '.join(keys_l)
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
    
    #check for nulls
    ser = pg_get_nullcount_all(schema, tableName)
    assert (ser==0).all()
    
    #add comment
    cmt_str = f'left join water depths {table_bldg} to merge of links {table_left} \n'
    cmt_str += f'built with {os.path.realpath(__file__)} at ' + datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pg_comment(schema, tableName, cmt_str)
    log.info(f'cleaning {tableName} ')
    try:
        pg_vacuum(schema, tableName)
        """table is a-spatial"""
    #pg_spatialIndex(schema, tableName, columnName='geometry')
    #pg_register(schema, tableName)
    except Exception as e:
        log.error(f'failed cleaning w/\n    {e}')
    log.info(f'finished building {schema}.{tableName}')
    return schema,  tableName


def create_table_aggregate(tableName, table_big, agg_func_l,  dev=False, conn_str=None, log=None):
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    if dev:
        schema = 'dev'
    else:
        schema = 'wd_bstats'
        
    keys_l = keys_d['grid']
    #===========================================================================
    # #setup
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
    #===========================================================================
    # #build query
    #===========================================================================
    g_cols = ', '.join([e for e in keys_d['grid']])
    haz_cols = [e for e in pg_get_column_names(schema, table_big) if e.startswith('f')]
    cols = ', '.join(keys_d['grid']) + ', '
#aggrecation columns
    for agg_func in agg_func_l:
        col_sfx = agg_func.lower().replace('_', '')
        cols += ', '.join([f'CAST({agg_func}({e}) AS real) AS {e}_{col_sfx}' for e in haz_cols]) + ', '
    
#wet counts
    cols += ', '.join([f'COUNT(*) FILTER (WHERE {e} > 0) as {e}_wetcnt' for e in haz_cols])
#dry count
    cols += ', COUNT(id) as bldg_cnt'
#null counts (using most extreme)
    cols += f', COUNT(*) FILTER (WHERE {haz_cols[-1]} IS NULL) as null_cnt'
    print('\n'.join(cols.split(',')))
    #===========================================================================
    # exe
    #===========================================================================
 
    sql(f"""
            CREATE TABLE {schema}.{tableName} AS
                SELECT {cols}
                    FROM {schema}.{table_big}
                            GROUP BY {g_cols}
            """)
    #===========================================================================
    # post
    #===========================================================================
 
    keys_str = ', '.join(keys_l)
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
    
    #nulls
    nc_ser = pg_to_df(f"""SELECT SUM(null_cnt) FROM {schema}.{tableName} GROUP BY grid_size""")
    assert nc_ser.iloc[:,0].max()==0, 'shouldnt get nulls anymore'    
    assert pg_get_nullcount_all(schema, tableName, conn_str=conn_str).max()==0
    
    #add comment
    cmt_str = f'group stats on {table_big} w/ functions: {agg_func_l} \n'
    cmt_str += f'built with {os.path.realpath(__file__)} in %.1f secs at %s '%(
        (datetime.now()-start).total_seconds(), datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    pg_comment(schema, tableName, cmt_str)
    log.info(f'cleaning {tableName} ')
    try:
        pg_vacuum(schema, tableName)
 
    except Exception as e:
        log.error(f'failed cleaning w/\n    {e}')
    log.info(f'finished building {schema}.{tableName}')
    return schema,  tableName


def create_view_wgeo_slice(

        country_key='deu',
        grid_size=1020,  
        expo_str='1x',
                           dev=False, conn_str=None, log=None):
    """create a view of the aggregated stats with some slicing"""
    
    
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    
    if log is None:
        log = init_log(name=f'stats')
        
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    
    
    #===========================================================================
    # table params
    #===========================================================================
    table_right =  f'agg_{country_key}_{grid_size:07d}'
    table_left=f'a03_gstats_{expo_str}_{country_key}'
    tableName = table_left+f'_{grid_size:04d}_wgeo'
    
        
    if dev:
        schema='dev'
        schema_right = 'dev'
    else:
        schema_right = 'grids'
        schema='wd_bstats'
    #assert pg_table_exists(schema_right, table_right, asset_type='view'), f'{schema_right}.{table_right} view must exist'
    
    keys_l = ['country_key', 'grid_size', 'i', 'j']
    #=======================================================================
    # setup
    #=======================================================================
 
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
    #=======================================================================
    # build query
    #=======================================================================
    cmd_str = f'CREATE TABLE {schema}.{tableName} AS'
    
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_l])
    
    #add an arbitrary indexer for QGIS viewing
    cols =f'ROW_NUMBER() OVER (ORDER BY tleft.i, tleft.j) as fid, ' 
    cols +=f'tleft.*, tright.geom' 
    cmd_str+= f"""
        SELECT {cols}
            FROM {schema}.{table_left} AS tleft
                LEFT JOIN {schema_right}.{table_right} AS tright
                    ON {link_cols}
                        WHERE tleft.grid_size={grid_size}
            """
            
    sql(cmd_str)
 
    #===========================================================================
    # post
    #===========================================================================
    keys_str = ', '.join(keys_l)
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
    
        #add comment
    cmt_str = f'join {table_right} to {table_left} grid geometry \n'
    cmt_str += f'built with {os.path.realpath(__file__)} in %.1f secs at %s '%(
        (datetime.now()-start).total_seconds(), datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    pg_comment(schema, tableName, cmt_str)
    log.info(f'cleaning {tableName} ')
    
    pg_register(schema, tableName)
    pg_spatialIndex(schema, tableName)
    pg_vacuum(schema, tableName)
    
    log.info(f'finished on {schema}.{tableName}')
    
    return tableName
         
def create_table_aoi_select(
        #L:\02_WORK\NRC\2307_funcAgg\04_CALC\aoi\aoi02_20230926.gpkg
        aoi_str = 'ST_GeomFromText(\'POLYGON((668789.8125 5610018, 760321.6875 5610018, 760321.6875 5646507, 668789.8125 5646507, 668789.8125 5610018))\', 6933)',
        
        country_key='deu',

        expo_str='1x',
         conn_str=None, log=None):
    """slice to aoi"""
    
    
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    
    if log is None:
        log = init_log(name=f'stats')
        
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    
    #===========================================================================
    # table params
    #===========================================================================
    #merge of grid geometries... see _02agg._07_views.run_view_grid_geom_union()
    table_right =  f'agg_{country_key}'
    
    #depth child stats. see create_table_aggregate()
    table_left=f'a03_gstats_{expo_str}_{country_key}'
    tableName = table_left+f'_aoi'
    
    
    #===========================================================================
    # no... these have only one grid size
    #     #group stats for 1 grid size w/ geometry. see create_view_wgeo_slice()
    # table_big = f'a03_gstats_{expo_str}_{country_key}_{grid_size:04d}_wgeo'
    # tableName =  f'a03_gstats_{expo_str}_{country_key}_{grid_size:04d}_aoi01'
    #===========================================================================
    
 
    schema_right = 'grids'
    schema='wd_bstats'
    #assert pg_table_exists(schema_right, table_right, asset_type='view'), f'{schema_right}.{table_right} view must exist'
    
    keys_l = ['country_key', 'grid_size', 'i', 'j']
    #=======================================================================
    # setup
    #=======================================================================
    epsg_id = pg_getCRS(schema_right, table_right)
    
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
    #=======================================================================
    # build query
    #=======================================================================
    cmd_str = f'CREATE TABLE {schema}.{tableName} AS'
    
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_l])
    
    #add an arbitrary indexer for QGIS viewing
    cols =f'ROW_NUMBER() OVER (ORDER BY tleft.i, tleft.j) as fid, ' 
    cols +=f'tleft.*, tright.geom' 
    cmd_str+= f"""
        SELECT {cols}
            FROM {schema}.{table_left} AS tleft
                LEFT JOIN {schema_right}.{table_right} AS tright
                    ON {link_cols}
                        WHERE ST_Intersects(ST_Centroid(geom),ST_Transform({aoi_str}, {epsg_id}));
            """
            
    sql(cmd_str)
    
    
 
    #===========================================================================
    # post
    #===========================================================================
    keys_str = ', '.join(keys_l)
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
    
        #add comment
    cmt_str = f'aoi select using geometry from {table_right} left join to {table_left} with: \n{aoi_str}'
    cmt_str += f'built with {os.path.realpath(__file__)} in %.1f secs at %s '%(
        (datetime.now()-start).total_seconds(), datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    pg_comment(schema, tableName, cmt_str)
    log.info(f'cleaning {tableName} ')
    
    pg_register(schema, tableName)
    pg_spatialIndex(schema, tableName)
    pg_vacuum(schema, tableName)
    
    log.info(f'finished on {schema}.{tableName}')
    
    return tableName
         
 

def run_pg_build_gstats(
        
        country_key='deu', 
        grid_size_l=None,
        haz_key_l=None,
 
 
        conn_str=None,
        
        filter_cent_expo=False,
        agg_func_l=['AVG','stddev_pop'],
 
         log=None,
         dev=False,
         add_geom=False,
 
        ):
    """ build table with stats for agg groups
    
 
    
    
    Returns
    -------
    postgres table
        inters_agg.wd_mean_{country_key}_{grid_size:04d} 

    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    start = datetime.now()   
    
    country_key = country_key.lower() 
    if grid_size_l is None: grid_size_l=gridsize_default_l
 
    # log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
    if conn_str is None: conn_str = get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'gstats')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    if filter_cent_expo:
        # wd for doubly exposed grids w/ polygon geometry. see _02agg._07_views.run_view_grid_wd_wgeo()
        expo_str = '2x'
    else:
        expo_str = '1x' # those grids with building exposure (using similar layer as for the centroid sampler)
    

    #source table keys
    table_d=dict()
    
    #===========================================================================
    # merge all of the link tables 
    #===========================================================================
 
    keys_l = keys_d['bldg']+['grid_size']
    
    _, table_d['links_merge'] = create_table_links_merge(grid_size_l, country_key, expo_str, keys_l, dev, log, conn_str=conn_str)
        
    #===========================================================================
    # join buidling wd to grid links
    #===========================================================================
 
    tableName = f'a02_links_{expo_str}_{country_key}_wd'
    table_left = table_d['links_merge']
    
    _, table_d['links_wd'] = create_table_joinL_bldg_wd(tableName, table_left, country_key, dev, keys_l, 
                                                        haz_key_l=haz_key_l, conn_str=conn_str, log=log)
    
    
    
    #===========================================================================
    # compute group stats
    #===========================================================================
    tableName=f'a03_gstats_{expo_str}_{country_key}'
    table_big = table_d['links_wd']
    
    schema, table_d['gstats'] = create_table_aggregate(tableName, table_big, agg_func_l, dev=dev, conn_str=conn_str, log=log) 
 
 
    
    #===========================================================================
    # add grid polygons to the view
    #===========================================================================
    if add_geom:
        create_view_join_grid_geom(schema, table_d['gstats'], country_key, log=log, dev=dev, conn_str=conn_str)
        
    
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




def get_a03_gstats_1x(
        country_key='deu', 
        expo_str='1x',
        log=None,conn_str=None,dev=False,use_cache=True,out_dir=None,
        limit=None,
        use_aoi=False,
        ):
    
    """helper to retrieve results from run_pg_build_gstats() as a dx
    
 
    """ 
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'depths','03_gstats', country_key)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    
    if log is None:
        log = init_log(name=f'dl')
    
    if dev: use_cache=False
    
    #===========================================================================
    # talbe params
    #===========================================================================
    fnstr = f'gstats_{country_key}'
    #see _04expo._03_views.create_view_join_stats_to_rl()
    if use_aoi:
        assert not dev
        tableName=f'a03_gstats_{expo_str}_{country_key}_aoi'
        fnstr+='_aoi'
    else:
        tableName = f'a03_gstats_{expo_str}_{country_key}'
    
    if dev:
        schema = 'dev'

    else:
        schema = 'wd_bstats' 
        
    #load meta
    assert pg_table_exists(schema, tableName, asset_type='table'), f'missing table dependency \'{schema}.{tableName}\''
    meta_df = pg_get_meta(schema, tableName)
    #===========================================================================
    # cache
    #===========================================================================

    
    
    
    
    uuid = hashlib.shake_256(f'{fnstr}_{dev}_{limit}_{meta_df}'.encode("utf-8"), usedforsecurity=False).hexdigest(8)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')
    
    if (not os.path.exists(ofp)) or (not use_cache):
 
        keys_l = ['country_key', 'grid_size', 'i', 'j']
        
        #===========================================================================
        # download
        #===========================================================================
        conn =  psycopg2.connect(conn_str)
        engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
        
        #row_cnt=0
        
        """only ~600k rows"""
        
        cmd_str = f'SELECT * FROM {schema}.{tableName}'
        
        if not limit is None:
            cmd_str+=f'\n    LIMIT {limit}'
 
        log.info(cmd_str)
        df_raw = pd.read_sql(cmd_str, engine, index_col=keys_l)
        """
        view(df_raw.head(100))        
        """    
        
        engine.dispose()
        conn.close()
        
        log.info(f'loaded {df_raw.shape} from {tableName}')
        
        #===========================================================================
        # clean up
        #===========================================================================
        #exposure meta
        expo_colns = ['bldg_cnt', 'null_cnt']
        df1 = df_raw.copy()
        df1.loc[:, expo_colns] = df1.loc[:, expo_colns].fillna(0.0)        
        df1=df1.set_index(expo_colns, append=True)
        
        #multi-index the columns
        #split by aggregate function

        col_df = df1.columns.str.split('_', expand=True).to_frame().reset_index(drop=True)
        
        col_mdex = pd.MultiIndex.from_frame(pd.concat({
            'haz_key':col_df.iloc[:,0].str.cat(others= col_df.iloc[:,1], sep='_'),
            'agg_func':col_df.iloc[:,2]
            }, axis=1))
        
        dx = pd.DataFrame(df1.values, index=df1.index, columns=col_mdex).rename(columns={'wetcnt':'wet_cnt'})
 
        """
        col_mdex.to_frame().reset_index(drop=True).join(pd.Series(df1.columns.values, name='og'))
        
 
        """
 
        
        #===========================================================================
        # write
        #===========================================================================
        """
        view(dx2.head(100))
        """
        
 
        log.info(f'writing {dx.shape} to \n    {ofp}')
        dx.sort_index(sort_remaining=True).sort_index(sort_remaining=True, axis=1).to_pickle(ofp)
    
    else:
        log.info(f'loading from cache:\n    {ofp}')
        dx = pd.read_pickle(ofp)
 
 
    log.info(f'got {dx.shape}')
    return dx
 

        
if __name__ == '__main__':
 
    run_pg_build_gstats(dev=False, 
                        #haz_key_l=['f500_fluvial'], 
                        add_geom=False)
    
    
 
    
    #get_a03_gstats_1x(dev=False, use_aoi=False)
    
    print('done')
    winsound.Beep(440, 500)
    
        
    
