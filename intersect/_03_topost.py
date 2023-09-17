'''
Created on Sep. 3, 2023

@author: cefect

port collect.pkl results to postgres
'''
import os, hashlib, sys, subprocess
import psutil
from datetime import datetime


import numpy as np
from numpy import dtype
import pandas as pd
import geopandas as gpd

import psycopg2
print('psycopg2.__version__=' + psycopg2.__version__)

from sqlalchemy import create_engine, URL

from tqdm import tqdm

from agg.coms_agg import get_conn_str, pg_vacuum, pg_spatialIndex

from coms import (
    init_log, today_str, get_log_stream, get_raster_point_samples, get_directory_size,
    dstr, view
    )

from definitions import wrk_dir, lib_dir, postgres_d, index_country_fp_d



def _get_fp_lib(srch_dir, log):
    country_dirs_d = {d:os.path.join(srch_dir, d) for d in os.listdir(srch_dir) if os.path.isdir(os.path.join(srch_dir, d))}
    assert len(country_dirs_d) > 0, 'no subfolders'
    log.info(f'on {len(country_dirs_d)}')
    fp_lib = dict()
    for country_key, country_dir in country_dirs_d.items():
        log.debug(f'concat {country_key}')
        #get all the files (hazard keys)
        fp_d=dict()
        for fn, fp in {e:os.path.join(country_dir, e) for e in os.listdir(country_dir) if e.endswith('.pkl')}.items():
            
            fp_d[int(fn.split('_')[1])]=fp
            
        fp_lib[country_key] = fp_d
 
        
    return fp_lib

 
def sql_table_exists(tableName, conn):
    """see if a table exists"""     
    with conn.cursor() as cur:     
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM 
                    pg_tables
                WHERE 
                    tablename  = %s)""", (tableName,))
        
        return cur.fetchone()[0]

def sql_schema_exists(schemaName, conn):
    """see if a schema exists"""
    
    with conn.cursor() as cur:
    
        cur.execute("""
            SELECT EXISTS (
                    SELECT schema_name FROM 
                        information_schema.schemata 
                    WHERE 
                        schema_name = %s)""",(schemaName,)
            )
        
        return cur.fetchone()[0]
    


def gpkg_to_postgis(gpkg_path, pg_conn_string, table_name):
    # Construct the ogr2ogr command
    cmd = [
        "ogr2ogr",
        "-f", "PostgreSQL",
        pg_conn_string,
        gpkg_path,
        "-nln", table_name,
        "-overwrite"
    ]

    # Run the ogr2ogr command
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check for errors
    if result.returncode != 0:
        print(f"An error occurred while running ogr2ogr: {result.stderr}")
    else:
        print(f"Successfully copied {gpkg_path} to PostGIS table {table_name}")

 
 

def run_to_postgres(
        srch_dir=None,out_dir=None,
        conn_str=None,
        schemaName='inters',
        country_l = ['AUS', 'BGD', 'BRA', 'CAN', 'DEU', 'ZAF'],
        coln_l=['010_fluvial', '050_fluvial', '100_fluvial', '500_fluvial', 'geometry'],
        ):
    """extract .pkl and load into postgress"""
    
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    
    if srch_dir is None:
        srch_dir = os.path.join(wrk_dir, 'outs', 'inters', '02_collect')
    assert os.path.exists(srch_dir)
 
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'inters', '03_topost')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
 
    
    log = init_log(name=f'toPost', fp=os.path.join(out_dir, today_str+'.log'))
    log.info(f'on \n    {srch_dir}')
    
    
    #===========================================================================
    # collect files
    #===========================================================================
    #get country dirs
    fp_lib = _get_fp_lib(srch_dir, log)
    
 
    
    #===========================================================================
    # set up database
    #===========================================================================
    # Connect to an existing database
    log.info(f'psycopg.connect w/ \n{conn_str}')
    with psycopg2.connect(conn_str) as conn:
  
        #see if our schema exists
        if not sql_schema_exists(schemaName, conn):
            log.info(f'schema \'{schemaName}\' not found... creating')
            with conn.cursor() as cur:
                cur.execute(f"""CREATE SCHEMA {schemaName}""")  
            
 
    #===========================================================================
    # #loop through each country
    #===========================================================================
    tab_d = dict()
    for country_key, fp_d in fp_lib.items():
        if not country_key in country_l:
            log.warning(f'skipping {country_key}')
            continue
        
        log.info(f'on {country_key} w/ {len(fp_d)}')
        tableName=f'{country_key.lower()}'
        
 
        with psycopg2.connect(conn_str) as conn:
            """using a new connection for each country/table
            makes parallelism possible
            guessing this is a nice balance between latency and committing
            """
            #get sqlalchemy engine
  
            engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
             
            #loop through each grid
            first = True
            for gid, fp in tqdm(fp_d.items()):
                log.debug(f'at gid={gid} w/ {os.path.basename(fp)}')
                 
                # load
                dxcol = pd.read_pickle(fp)
                 
                assert set(coln_l).difference(dxcol.columns) == set(), f'column mismatch on \n    {fp}'
                 
                # use geopandas to write
                log.debug(f'to_postgis on {dxcol.shape}') 
                if first:
                    if_exists = 'replace'
                else:
                    if_exists = 'append'
                 
                dxcol.reset_index().to_postgis(tableName, engine, schema=schemaName,
                                               if_exists=if_exists,
                                               index=False,
                                               # index_label=dxcol.index.names,
                                               )
                 
                #addd keys
                if first: 
                    with conn.cursor() as cur:
                        cur.execute(f"""
                        ALTER TABLE {schemaName}.{tableName}
                            ADD PRIMARY KEY (gid, id);""")
                     
  
                first = False
                 
            #close connection. wrap
            engine.dispose()
            log.debug(f'finished {country_key}')
            tab_d[country_key] = tableName
            
        #clean table
        print(f'pg_vacuum')
        pg_vacuum(schemaName, tableName)
        pg_spatialIndex(schemaName, tableName, columnName='geometry', log=log)
            
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on\n{dstr(tab_d)}')
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    #'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    return tab_d
        



def init_schema(schemaName, conn, log):
    if not sql_schema_exists(schemaName, conn):
        log.info(f'schema \'{schemaName}\' not found... creating')
        with conn.cursor() as cur:
            cur.execute(f"""CREATE SCHEMA {schemaName}""")

def run_grids_to_postgres(
        out_dir=None,
        conn_str=None,
        schemaName='grids',tableName='country_grids',
        index_d=index_country_fp_d,
        ):
    """add grids to postgis"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
 
 
    
    log = init_log(name=f'toPostG')
    log.info(f'on \n    {index_d.keys()}\n    {postgres_d}')
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    
 
        
    
    #===========================================================================
    # setup
    #===========================================================================
    #see if our schema exists
    with psycopg2.connect(conn_str) as conn:        
        init_schema(schemaName, conn, log)  
        
        
    #===========================================================================
    # #loop and load each
    #===========================================================================
    with psycopg2.connect(get_conn_str(postgres_d)) as conn: 
        
        #set engine for geopandas
        engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
        
        first=True
        for k, fp in index_d.items():
            log.info(f'{os.path.basename(fp)}')
            
            #use geopandas
            gdf = gpd.read_file(fp)
            
            
            #add country key
            gdf['country_key'] = k.lower()
            
            #port to postgis
            log.info(f'porting {gdf.shape} to postgis')
            if_exists = 'replace' if first else 'append'
            
            gdf.to_postgis(tableName, engine, schema=schemaName, 
                                               if_exists=if_exists, 
                                               index=False, 
                                               #index_label=dxcol.index.names,
                                               )
            
            first=False
     
 
            
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {len(index_d)}')
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    #'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    return 
    
 

        
 
        
    
    
    
    

if __name__ == '__main__':
    #run_grids_to_postgres()
    
    
    run_to_postgres(country_l = ['DEU'])
    
    
    
    
    
    
    
    
    
