'''
Created on Sep. 2, 2023

@author: cefect

create aggregation geometries and join to points
'''

import os, hashlib, sys, subprocess, psutil
from datetime import datetime

import psycopg2
print('psycopg2.__version__=' + psycopg2.__version__)

from sqlalchemy import create_engine, URL

from tqdm import tqdm

import pandas as pd

from coms import (
    init_log, today_str, get_log_stream, get_directory_size,
    dstr, view, get_conn_str
    )

 

from definitions import index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg



def build_extents_grid(conn_d, epsg_id, schema, tableName):
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
        #remove if it exists
        with conn.cursor() as cur:
            cur.execute(f"""DROP TABLE IF EXISTS {schema}.{tableName}""")
        conn.commit()
        #create a table with a geometry column
        with conn.cursor() as cur:
            cur.execute(f"""

            CREATE TABLE {schema}.{tableName} (

                country_key text,

                geom geometry(POLYGON, 4326))

                """)
        #=======================================================================
        # #update the CRSID
        # with conn.cursor() as cur:
        #     cur.execute("""SELECT UpdateGeometrySRID('grids', 'country_grids', 'geometry', 4326)""")
        # conn.commit()
        #=======================================================================
        #create the groups
        with conn.cursor() as cur:
            estr = f"""

            INSERT INTO {schema}.{tableName}  

                    SELECT country_key, ST_Extent(geometry) as geom

                        FROM grids.country_grids

                        GROUP BY country_key;"""
            print(estr)
            cur.execute(estr)
        #transform
        with conn.cursor() as cur:
            cur.execute(f"""
 
        ALTER TABLE {schema}.{tableName}  
 
            ALTER COLUMN geom TYPE geometry(POLYGON, {epsg_id})
 
                USING ST_Transform(geom, {epsg_id});
 
                    """)
        conn.commit()
        #check
        with conn.cursor() as cur:
            cur.execute("""SELECT Find_SRID(%s, %s, %s)""", (schema, tableName, 'geom'))
            assert cur.fetchone()[0] == epsg_id, 'crs mismatch'
    return  


def build_agg_grid(grid_size, conn_d, schema, tableName, country_l, log, tableName2):
    """buidl aggregated grids per country"""
    
    #===========================================================================
    # setup
    #===========================================================================
    with psycopg2.connect(get_conn_str(conn_d)) as conn: 
        
        #remove if it exists
        with conn.cursor() as cur:
            cur.execute(f"""DROP TABLE IF EXISTS {schema}.{tableName}""")
        conn.commit()
        
        #populate
        
    
    #===========================================================================
    # #loop on each country
    #===========================================================================
    with psycopg2.connect(get_conn_str(conn_d)) as conn: 
        #build the mesh (need to filter with ST_Intersects still
        with conn.cursor() as cur:
            cmd_str=f"""
                CREATE TABLE {schema}.{tableName} AS
                    SELECT (ST_SquareGrid({grid_size}, geom)).*, country_key, {grid_size} as grid_size
                        FROM {schema}.{tableName2} 
                            """ 
            print(cmd_str)
            cur.execute(cmd_str)
            
        #register the geometry for QGIS
        with conn.cursor() as cur:
            cur.execute(f"""SELECT Populate_Geometry_Columns(%s::regclass)""", (f'{schema}.{tableName}',))
                
 
            
    with psycopg2.connect(get_conn_str(conn_d)) as conn: 
        #query result
        with conn.cursor() as cur:
            cur.execute(f"""
            SELECT country_key, COUNT(*)
                    FROM {schema}.{tableName}
                        GROUP BY country_key;
                    """)
            
            #return series as resuilt
            res = pd.Series(dict(cur.fetchall()))
            log.info(f'built grid={grid_size} w/\n%s'%res)
                
  

def run_join_agg_grids(
        country_l=None,
        out_dir=None,
        conn_d=postgres_d,
        epsg_id=None,
        schema='grids',
        grid_size_l=[100000]
        ):
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()    
 
 
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'agg', '01_jgrid')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    
    log = init_log(name=f'jgrid', fp=os.path.join(out_dir, today_str+'.log'))
    
    if country_l is  None: country_l=[e.lower() for e in index_country_fp_d.keys()]
    if epsg_id is None: epsg_id=equal_area_epsg
    
    log.info(f'on \n    {country_l}\n    {conn_d}')
    
    
    #===========================================================================
    # create extents grid
    #===========================================================================
    tbl_extents='country_grids_extents'
    #build_extents_grid(conn_d, epsg_id, schema, tbl_extents)
    log.info(f'built {tbl_extents}')
 
    #===========================================================================
    # create agg grids
    #===========================================================================
    tbl_agg = 'agg'
    log.info(f'buidling {tbl_agg} w/ \n    {grid_size_l}')
    
    raise IOError('stopped here')
    #set up the table

    for grid_size in grid_size_l:
        build_agg_grid(grid_size, conn_d, schema, tbl_agg, country_l, log, tbl_extents)
    
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished  ')
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    #'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    return
    
 
        

if __name__ == '__main__':
    run_join_agg_grids()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    