'''
Created on Sep. 2, 2023

@author: cefect

create aggregation grids
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

 

from definitions import index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir



def build_extents_grid(conn_d, epsg_id, schema, tableName):
    """build table of textents from original grids"""
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


def build_agg_grid(grid_size_l, country_l, conn_d, schema, tableBaseName, tableName2, epsg_id, log):
    """buidl aggregated grids per country
    
    decided to create 1 table per grid-size as these will be large operations
    """
 
    #===========================================================================
    # #loop on each grid size
    #===========================================================================
    log.info(f'looping on {len(grid_size_l)} grids')
    res_d = dict()
 
    for i, grid_size in enumerate([int(e) for e in grid_size_l]):
        tableName=f'{tableBaseName}_{grid_size:07d}'
        log.info(f'on {i}: {tableName}')
        start_i =datetime.now()
        
        with psycopg2.connect(get_conn_str(conn_d)) as conn: 
            #===========================================================================
            # setup
            #===========================================================================
 
            #remove if it exists
            with conn.cursor() as cur:
                cur.execute(f"""DROP TABLE IF EXISTS {schema}.{tableName}""")            
            conn.commit()
            
            #create table
            with conn.cursor() as cur:          
                cmd_str=f"""
                    CREATE TABLE {schema}.{tableName} AS
                        SELECT a.country_key, {grid_size} as grid_size, (ST_SquareGrid({grid_size}, geom)).* 
                            FROM {schema}.{tableName2} a
                                """
                log.info(cmd_str) 
                cur.execute(cmd_str)
                
                #delete irrellevant grids
                cmd_str=f"""
                    DELETE FROM {schema}.{tableName} a
                        WHERE NOT EXISTS (
                            SELECT 1
                            FROM grids.country_grids b
                            WHERE ST_Intersects(a.geom, ST_Transform(b.geometry, {epsg_id}))
                        )"""
                         
                log.info(cmd_str) 
                cur.execute(cmd_str)
                
                 
                #register the geometry for QGIS
                with conn.cursor() as cur:
                    cur.execute(f"""SELECT Populate_Geometry_Columns(%s::regclass)""", (f'{schema}.{tableName}',))
                    
                #change dtype on grid_size
                with conn.cursor() as cur:
                    cur.execute(f"""
                    ALTER TABLE {schema}.{tableName}
                        ALTER COLUMN grid_size TYPE integer;
                            """)  
                    
                    
                #create spatial index
                with conn.cursor() as cur:
                    cur.execute(f"""
                    CREATE INDEX {tableName}_geom_idx
                        ON {schema}.{tableName}
                            USING GIST (geom);
                            """)  
                
            
 
                
 
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
                assert len(res)>0
                log.info(f'built grid={grid_size} w/\n%s'%res)
                res_d[grid_size] = res
                
        #=======================================================================
        # wrap grid
        #=======================================================================
        meta_d = {
                'tdelta':(datetime.now()-start_i).total_seconds(),
                'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                'postgres_GB':get_directory_size(postgres_dir),
                #'output_MB':os.path.getsize(ofp)/(1024**2)
                }

        log.info(f'finished {grid_size}\n    {dstr(meta_d)}\n\n')
                
 
                
 
    
    log.info(f'finished w/ {len(res_d)}')
    

    
    return res_d
  

def run_join_agg_grids(
        country_l=None,
        out_dir=None,
        conn_d=postgres_d,
        epsg_id=None,
        schema='grids',
        grid_size_l=[
            30*34,30*8, 30*2
            #1e5, 2e5, #big for testing
            ]
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
 
    #grid_size_l, country_l, conn_d, schema, tableName, tableName2, epsg_id
    _ = build_agg_grid(grid_size_l, country_l, conn_d, schema, tbl_agg,  tbl_extents, epsg_id, log)
    
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished  ')
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(postgres_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    return
    
 
        

if __name__ == '__main__':
    run_join_agg_grids()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    