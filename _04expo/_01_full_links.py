'''
Created on Sep. 21, 2023

@author: cefect


link exposed agg grids to all child buildings

NOTE: _02agg._03_joins is similar, but leaner as it only includes exposed buildings
'''
#===============================================================================
# IMPORTS-----
#===============================================================================
import os, hashlib, sys, subprocess, psutil, winsound
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
    pg_comment, pg_getCRS, pg_register, pg_table_exists, pg_get_nullcount, pg_getcount, pg_get_nullcount_all
    )
 
 
from _01intersect._04_views import create_view_join_bldg_geom

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l, haz_label_d
    )



def run_agg_bldg_full_links(
         country_key, 
         grid_size,
         
         filter_bldg_expo=False,
         filter_cent_expo=False,
 
        conn_str=None, 
        log=None,
        epsg_id=equal_area_epsg,
 
 
        dev=False,
        with_geo=False,
 
        ):
    """spatially join (exposed) grid ids to each bldg using grid polygons
    
    inters_grid.agg_expo_{country_key}_{grid_size:04d}_poly only includes doubly exposed grids
    
    
    Params
    ----------
    haz_key: str
        column with grid wd used to identify grids as 'exposed' (want most extreme)
        
    filter_bldg_expo: bool
        only include those buildings with some exposure
    
    filter_cent_expo: bool
        only include buildings in grid cells with some centroid exposure
        
        
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
    
    if filter_bldg_expo:
        """exclude some buildings from the link table"""
        raise NotImplementedError(f'see  _02agg._03_joins')
    else:
        """use grids with some buidling exposure"""
 
    
    
    #wd for all buildings. see _01intersect._03_topost
    table_left=f'{country_key}' 
    

    #select the grid table
    if filter_cent_expo:
        #wd for doubly exposed grids w/ polygon geometry. see _02agg._07_views.run_view_grid_wd_wgeo()
        table_grid=f'agg_expo_{country_key}_{grid_size:04d}_poly'
        schema_grid='inters_grid'
        expo_str = '2x'
    else:
        #those grids with building exposure (using similar layer as for the centroid sampler)
        #see _02agg._04_occupied
        table_grid=f'agg_occupied_{country_key}_{grid_size:04d}_poly'
        schema_grid=f'agg_bldg' 
        expo_str = '1x'
        
    tableName = f'bldgs_grid_link_{expo_str}_{country_key}_{grid_size:04d}'
 
    if dev:
        schema='dev'
        schema_left, schema_grid=schema, schema        

    else:
        schema='expo' 
        schema_left='inters'
        
 
    if with_geo: assert pg_getCRS(schema_grid, table_grid)==epsg_id
    #===========================================================================
    # spatially join grid keys
    #===========================================================================
    """
    NOTES:
    want to include all points (even dry ones)
 
    
    """
    
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE') 
    
    cols = 'LOWER(pts.country_key) as country_key, pts.gid, pts.id, polys.grid_size, polys.i, polys.j'
    if with_geo: cols+=', pts.geometry as geom'
    
    #filter nulls
    haz_cols = [e for e in pg_get_column_names(schema_left, table_left) if e.startswith('f')]
    where_cols = ' AND '.join([f'pts.{e} IS NOT NULL' for e in haz_cols])
    
 

    cmd_str=f"""
    CREATE TABLE {schema}.{tableName} AS
        SELECT {cols}
            FROM {schema_left}.{table_left} AS pts
                JOIN {schema_grid}.{table_grid} AS polys 
                    ON ST_Intersects(polys.geom, ST_Transform(pts.geometry, {epsg_id}))
                        WHERE {where_cols}
                    """
    
    sql(cmd_str)
            
    #===========================================================================
    # #clean up
    #===========================================================================
    
    log.info(f'cleaning')
    pg_exe(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY (country_key, gid, id)')
    
    cmt_str = f'join grid ({table_grid}) i,j to points ({table_left}) \n'
    cmt_str += f'built with {os.path.realpath(__file__)} at '+datetime.now().strftime("%Y.%m.%d: %H.%M.%S")
    pg_comment(schema, tableName, cmt_str)
    
 
    if with_geo:
        pg_register(schema, tableName)
    pg_vacuum(schema, tableName)
    
         
    #===========================================================================
    # #wrap
    #===========================================================================
    meta_d = {
        'tdelta':(datetime.now() - start).total_seconds(), 
        'RAM_GB':psutil.virtual_memory()[3] / 1000000000, 
        'postgres_GB':get_directory_size(postgres_dir)}
        #'output_MB':os.path.getsize(ofp)/(1024**2)
    log.info(f'finishedw/ \n{meta_d}')
    
    return tableName




def create_table_bldg_exposed_union(country_key='deu', 
        expo_str = '1x', 
        conn_str=None, 
         log=None,
         dev=False,
         add_geom=False,
         grid_size_l=None,
         ):
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    start = datetime.now()   
 
    
    if grid_size_l is None: grid_size_l = gridsize_default_l
 
    # log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
    if conn_str is None: conn_str = get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'mLinks')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #source table keys
    keys_d = {
        'bldg':['country_key', 'gid', 'id'], 
        'grid':['country_key', 'grid_size', 'i', 'j']}
    if dev:
        schema = 'dev'
    else:
        schema = 'expo'
    #===========================================================================
    # loop and merge
    #===========================================================================
    d = dict()
    first = True
    schema_left = schema
    for i, grid_size in enumerate(sorted(grid_size_l, reverse=True)):
        log.info(f'grid_size={grid_size}')
        table_res = f'bg_link_{i:02d}'
        table_right = f'bldgs_grid_link_{expo_str}_{country_key}_{grid_size:04d}'
        assert pg_table_exists(schema, table_right)
        if first:
            table_left = table_right
        d[grid_size] = pg_getcount(schema_left, table_left, conn_str=conn_str)
        if first:
            first = False
    #continue
        #=======================================================================
        # build quiery
        #=======================================================================
        sql(f'DROP TABLE IF EXISTS temp.{table_res} CASCADE')
        link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_d['bldg']])
        cols = ', '.join([f'COALESCE(tleft.{e}, tright.{e}) as {e}' for e in keys_d['bldg']])
        cmd_str = f"""
                    CREATE TABLE temp.{table_res} AS
                        SELECT {cols} 
                            FROM {schema_left}.{table_left} as tleft
                                FULL OUTER JOIN {schema}.{table_right} as tright
                                    ON {link_cols}"""
        #=======================================================================
        # exec
        #=======================================================================
        sql(cmd_str)
        #=======================================================================
        # wrap
        #=======================================================================
        keys_str = ', '.join(keys_d['bldg'])
        sql(f'ALTER TABLE temp.{table_res} ADD PRIMARY KEY ({keys_str})')
        schema_left = 'temp'
        table_left = table_res
        assert pg_get_nullcount(schema_left, table_left, 'id') == 0, grid_size
    
    d[grid_size] = pg_getcount(schema_left, table_left, conn_str=conn_str)
    log.info(f'finisehd w/ \n    {d}')
    
    return schema_left, table_left



def create_table_ljoin_depths(schema_left, table_left, 
                              log=None, 
                              country_key='deu',
                              dev=False,conn_str=None,
                              
                              ):
    """join depths to buidlings"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    start = datetime.now()   
    
    country_key = country_key.lower() 
    
 
 
    # log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
    if conn_str is None: conn_str = get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'mLinks')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
 
    
    #===========================================================================
    # talbe params-------
    #===========================================================================
    #source table keys
    keys_d = { 
        'bldg':['country_key', 'gid', 'id'],
        'grid':['country_key', 'grid_size', 'i', 'j']        
    }
    
 
        
        
    tableName=f'bldg_expo_wd_{country_key}' #output
 
    table_right = f'{country_key}' #building dephts
 
    if dev: 
        schema='dev'
        schema_right=schema        
 
        
    else:
        schema='expo'
        schema_right='inters' 
 
 
    assert pg_table_exists(schema_right, table_right)
    assert pg_table_exists(schema_left, table_left)
    
    #===========================================================================
    # prep
    #===========================================================================
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')

    #===========================================================================
    # #build query
    #===========================================================================
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_d['bldg']])
    haz_cols = [e for e in pg_get_column_names(schema_right, table_right) if e.startswith('f')]
    
    
    """not including geometry here"""
    #cols= ', '.join([f'tleft.{e}' for e in keys_d['bldg'] if not e in keys_d['grid']]) + ', '
    cols = ', '.join([f'tleft.{e}' for e in keys_d['bldg']]) + ', '
    cols += ', '.join([f'tright.{e}' for e in haz_cols])


    cmd_str = f"""
            CREATE TABLE {schema}.{tableName} AS            
                SELECT {cols}            
                    FROM {schema_left}.{table_left} as tleft            
                        LEFT JOIN {schema_right}.{table_right} as tright            
                            ON {link_cols}
            
            """
#print(cmd_str)
    sql(cmd_str)
    
    #===========================================================================
    # post
    #===========================================================================
    keys_str = ', '.join(keys_d['bldg'])
    sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
    
    assert pg_get_nullcount(schema, tableName, haz_cols[0]) == 0, f'bad link?'
#add comment
    cmt_str = f'joined buidling depths from \'{table_right}\' to \'{table_left}\' \n'
    cmt_str += f'built with {os.path.realpath(__file__)} create_table_ljoin_depths() at ' + datetime.now().strftime("%Y.%m.%d: %H.%M.%S")
    pg_comment(schema, tableName, cmt_str)
    log.info(f'cleaning {tableName} ')
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
    cnt = pg_getcount(schema, tableName)
    #col_l = pg_get_column_names(schema, tableName)
    ser = pg_get_nullcount_all(schema, tableName)
    log.info(f'finished on {schema}.{tableName} w/ {cnt} and \n{ser}')
    
    return schema, tableName

def run_expo_bldg(
        
        country_key='deu', 
        expo_str = '1x', 
        conn_str=None,
 
         log=None,
         dev=False,
         add_geom=False,
         grid_size_l=None,
 
        ):
    """create table of buildings expoed to any grid cell
    
 
     here we get all the building ids found in any of the link tables
         alternatively, we could have taken the buidling ids from the largest link table
         then purge entries in the other tables that dont show up in the largest
    
    
    Returns
    -------
    postgres mat view
        inters_agg.wd_mean_{country_key}_{grid_size:04d} 

    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    start = datetime.now()   
    
    country_key = country_key.lower() 
    
    if grid_size_l is None: grid_size_l = gridsize_default_l
 
    # log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
    if conn_str is None: conn_str = get_conn_str(postgres_d)
    if log is None:
        log = init_log(name=f'mLinks')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    res_d = dict()
    skwargs = dict(log=log, conn_str=conn_str, dev=dev)
    #===========================================================================
    # union all the link tables
    #===========================================================================
    
    s, t = create_table_bldg_exposed_union( grid_size_l=grid_size_l, **skwargs)
    res_d['union'] = {'schema':s, 'table':t}
    
 
    #===========================================================================
    # join buidling wd 
    #===========================================================================
    #setup
    #tableName1 = tableName+'_bwd'
 
    create_table_ljoin_depths(s, t, **skwargs)
        
    #===========================================================================
    # add geometry
    #===========================================================================
    if add_geom:
        create_view_join_bldg_geom(schema, tableName, log=log, dev=dev, country_key=country_key)
    
        
        
    
    
        
#===============================================================================
# replaced with run_expo_bldg
# def run_merge_expo_bldgs_wd(
#         
#         country_key='deu', 
# 
#         filter_cent_expo=False,
#  
#         conn_str=None,
#  
#          log=None,
#          dev=False,
#          add_geom=False,
#          grid_size_l=None,
#  
#         ):
#     """join  depths to the largest link table
#     
#  
#     
#     
#     Returns
#     -------
#     postgres mat view
#         inters_agg.wd_mean_{country_key}_{grid_size:04d} 
# 
#     """
#     
#     #===========================================================================
#     # defaults
#     #===========================================================================
#     
#     start = datetime.now()   
#     
#     country_key = country_key.lower() 
#     
#     if grid_size_l is None: grid_size_l = gridsize_default_l
#  
#     # log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
#     
#     if conn_str is None: conn_str = get_conn_str(postgres_d)
#     if log is None:
#         log = init_log(name=f'wd_mean')
#     
#     sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
#     
#     #===========================================================================
#     # talbe params-------
#     #===========================================================================
#     #source table keys
#     keys_d = { 
#         'bldg':['country_key', 'gid', 'id'],
#         'grid':['country_key', 'grid_size', 'i', 'j']        
#     }
#     
#     if filter_cent_expo:
#         #wd for doubly exposed grids w/ polygon geometry. see _02agg._07_views.run_view_grid_wd_wgeo()
#  
#         expo_str = '2x'
#     else:
#         #those grids with building exposure (using similar layer as for the centroid sampler) 
#         expo_str = '1x'
#         
#         
#     tableName=f'bldgs_grid_link_{expo_str}_{country_key}_bwd' #output
#  
#     table_bldg = f'{country_key}' #building dephts
#  
#     if dev: 
#         schema='dev'
#         schema_bldg=schema        
#  
#         
#     else:
#         schema='expo'
#         schema_bldg='inters' 
#  
#  
#     assert pg_table_exists(schema_bldg, table_bldg)
#     
#     #===========================================================================
#     # merge all the link tables-----
#     #===========================================================================
#     """no need to materlized this as we only query onces
#     
#     each table has a different length as the larger grids pick up more neighbours:
#         0060: 1479487
#         0240: 2221677
#         1020: 4655823
#         
#         
#     just take the buildings from the largest grid_size
#     
#         
#     """
#     
#     grid_size = max(grid_size_l)
#     table_left = f'bldgs_grid_link_{expo_str}_{country_key}_{grid_size:04d}' 
#  
#  
#  
#  
#     #===========================================================================
#     # join buidling wd 
#     #===========================================================================
#     #setup
#     #tableName1 = tableName+'_bwd'
#  
#     sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE')
#     
#     #build query    
#     
#     link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_d['bldg']])  
#     
#     haz_cols = [e for e in pg_get_column_names(schema_bldg, table_bldg) if e.startswith('f')]
#     
#     """not including geometry here"""
#     #cols= ', '.join([f'tleft.{e}' for e in keys_d['bldg'] if not e in keys_d['grid']]) + ', '
#     cols= ', '.join([f'tleft.{e}' for e in keys_d['bldg']]) + ', '    
#     cols+= ', '.join([f'tright.{e}' for e in haz_cols])
#  
#     #execute (using a sub-query)
#     cmd_str = f"""
#     CREATE TABLE {schema}.{tableName} AS
#         SELECT {cols}
#             FROM {schema}.{table_left} as tleft
#                 LEFT JOIN {schema_bldg}.{table_bldg} as tright
#                     ON {link_cols}
#     
#     """
#     #print(cmd_str)
#     sql(cmd_str) 
#     #===========================================================================
#     # post
#     #===========================================================================    
#     keys_str = ', '.join(keys_d['bldg'])
#     sql(f'ALTER TABLE {schema}.{tableName} ADD PRIMARY KEY ({keys_str})')
#     
#     assert pg_get_nullcount(schema, tableName, 'id')==0, f'bad link?'
#  
#     #add comment 
#     cmt_str = f'joined buidling depths from \'{table_bldg}\' to \'{table_left}\' \n'
#     cmt_str += f'built with {os.path.realpath(__file__)} run_merge_expo_bldgs_wd() at '+datetime.now().strftime("%Y.%m.%d: %H.%M.%S")
#     pg_comment(schema, tableName, cmt_str)
#     
#     log.info(f'cleaning {tableName} ')
#     
#     try:
#         pg_vacuum(schema, tableName)
#         """table is a-spatial"""
#         #pg_spatialIndex(schema, tableName, columnName='geometry')
#         #pg_register(schema, tableName)
#     except Exception as e:
#         log.error(f'failed cleaning w/\n    {e}')
#         
#     #===========================================================================
#     # add geometry
#     #===========================================================================
#     if add_geom:
#         create_view_join_bldg_geom(schema, tableName, log=log, dev=dev, country_key=country_key)
#  
#===============================================================================


def run_all(country_key='deu', **kwargs):
    log = init_log(name='links')
    
    for grid_size in gridsize_default_l:
        run_agg_bldg_full_links(country_key, grid_size, log=log, **kwargs)
        
    run_expo_bldg(country_key=country_key, log=log, **kwargs)
        
if __name__ == '__main__':
    
    #run_agg_bldg_full_links('deu', 1020, dev=False, with_geo=False, filter_cent_expo=False)
 
    #run_merge_expo_bldgs_wd(dev=True, add_geom=False,)
    #run_all('deu', dev=True)
    
    run_expo_bldg(dev=False)
    
    print('done')
    winsound.Beep(440, 500)
    
    
    
    
    
    
    
    
 
