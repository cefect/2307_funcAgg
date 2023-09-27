'''
Created on Sep. 21, 2023

@author: cefect


link exposed agg grids to all child buildings

NOTE: _02agg._03_joins is similar, but leaner as it only includes exposed buildings
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
    pg_comment, pg_getCRS, pg_register, pg_table_exists, pg_get_nullcount
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
    want an inner join so we only get those that intersect
    
    """
    
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName} CASCADE') 
    
    cols = 'LOWER(pts.country_key) as country_key, pts.gid, pts.id, polys.grid_size, polys.i, polys.j'
    if with_geo: cols+=', pts.geometry as geom'
    
 

    cmd_str=f"""
    CREATE TABLE {schema}.{tableName} AS
        SELECT {cols}
            FROM {schema_left}.{table_left} AS pts
                JOIN {schema_grid}.{table_grid} AS polys 
                    ON ST_Intersects(polys.geom, ST_Transform(pts.geometry, {epsg_id}))
                        WHERE pts.f010_fluvial IS NOT NULL
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


def run_all(country_key='deu', **kwargs):
    log = init_log(name='links')
    
    for grid_size in gridsize_default_l:
        run_agg_bldg_full_links(country_key, grid_size, log=log, **kwargs)
        
        

def run_merge_expo_bldgs_wd(
        
        country_key='deu', 

        filter_cent_expo=False,
 
        conn_str=None,
 
         log=None,
         dev=False,
         add_geom=False,
         grid_size_l=None,
 
        ):
    """join  building depths to exposed grid link table
    
 
    
    
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
        log = init_log(name=f'wd_mean')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    #===========================================================================
    # talbe params-------
    #===========================================================================
    #source table keys
    keys_d = { 
        'bldg':['country_key', 'gid', 'id'],
        'grid':['country_key', 'grid_size', 'i', 'j']        
    }
    
    if filter_cent_expo:
        #wd for doubly exposed grids w/ polygon geometry. see _02agg._07_views.run_view_grid_wd_wgeo()
 
        expo_str = '2x'
    else:
        #those grids with building exposure (using similar layer as for the centroid sampler) 
        expo_str = '1x'
        
        
    tableName=f'bldgs_grid_link_{expo_str}_{country_key}' #output
 
    table_bldg = f'{country_key}' #building dephts
 
    if dev: 
        schema='dev'
        schema_bldg=schema        
 
        
    else:
        schema='expo'
        schema_bldg='inters' 
 
 
    assert pg_table_exists(schema_bldg, table_bldg)
    
    #===========================================================================
    # merge all the link tables-----
    #===========================================================================
    """no need to materlized this as we only query onces
    
    each table has a different length as the larger grids pick up more neighbours:
        0060: 1479487
        0240: 2221677
        1020: 4655823
        
        
    just take the buildings from the largest grid_size
    
        
    """
    
    grid_size = max(grid_size_l)
    table_left = f'bldgs_grid_link_{expo_str}_{country_key}_{grid_size:04d}' 
 
 
 
 
    #===========================================================================
    # join buidling wd 
    #===========================================================================
    #setup
    tableName1 = tableName+'_bwd'
 
    sql(f'DROP TABLE IF EXISTS {schema}.{tableName1} CASCADE')
    
    #build query    
    
    link_cols = ' AND '.join([f'tleft.{e}=tright.{e}' for e in keys_d['bldg']])  
    
    haz_cols = [e for e in pg_get_column_names(schema_bldg, table_bldg) if e.startswith('f')]
    
    """not including geometry here"""
    #cols= ', '.join([f'tleft.{e}' for e in keys_d['bldg'] if not e in keys_d['grid']]) + ', '
    cols= ', '.join([f'tleft.{e}' for e in keys_d['bldg']]) + ', '    
    cols+= ', '.join([f'tright.{e}' for e in haz_cols])
 
    #execute (using a sub-query)
    cmd_str = f"""
    CREATE TABLE {schema}.{tableName1} AS
        SELECT {cols}
            FROM {schema}.{table_left} as tleft
                LEFT JOIN {schema_bldg}.{table_bldg} as tright
                    ON {link_cols}
    
    """
    #print(cmd_str)
    sql(cmd_str) 
    #===========================================================================
    # post
    #===========================================================================    
    keys_str = ', '.join(keys_d['bldg'])
    sql(f'ALTER TABLE {schema}.{tableName1} ADD PRIMARY KEY ({keys_str})')
    
    assert pg_get_nullcount(schema, tableName1, 'id')==0, f'bad link?'
 
    #add comment 
    cmt_str = f'joined buidling depths from \'{table_bldg}\' to \'{table_left}\' \n'
    cmt_str += f'built with {os.path.realpath(__file__)} run_merge_expo_bldgs_wd() at '+datetime.now().strftime("%Y.%m.%d: %H.%M.%S")
    pg_comment(schema, tableName1, cmt_str)
    
    log.info(f'cleaning {tableName1} ')
    
    try:
        pg_vacuum(schema, tableName1)
        """table is a-spatial"""
        #pg_spatialIndex(schema, tableName, columnName='geometry')
        #pg_register(schema, tableName)
    except Exception as e:
        log.error(f'failed cleaning w/\n    {e}')
        
    #===========================================================================
    # add geometry
    #===========================================================================
    if add_geom:
        create_view_join_bldg_geom(schema, tableName1, log=log, dev=dev, country_key=country_key)
 
       
        
if __name__ == '__main__':
    
    #run_agg_bldg_full_links('deu', 1020, dev=True, with_geo=True, filter_cent_expo=False)
 
    #run_merge_expo_bldgs_wd(dev=False, add_geom=False,)
    run_all('deu', dev=True)
    
    print('done')
    
    
    
    
    
    
    
    
 
