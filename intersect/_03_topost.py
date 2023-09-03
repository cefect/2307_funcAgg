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

from coms import (
    init_log, today_str, get_log_stream, get_raster_point_samples, get_directory_size,
    dstr, view
    )

from definitions import wrk_dir, lib_dir, conn_str



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
            def get_conn():
                return conn
            engine = create_engine('postgresql+psycopg2://', creator=get_conn)
            
            #loop through each grid
            first = True
            for gid, fp in tqdm(fp_d.items()):
                log.debug(f'at gid={gid} w/ {os.path.basename(fp)}')
                

                
                #load
                dxcol = pd.read_pickle(fp)
                
                assert set(coln_l).difference(dxcol.columns)==set(), f'column mismatch on \n    {fp}'
            
                
                #use geopandas to write
                log.debug(f'to_postgis on {dxcol.shape}') 
                if first:
                    if_exists='replace'
                else:
                    if_exists='append'
                
                dxcol.reset_index().to_postgis(tableName, engine, schema=schemaName, 
                                               if_exists=if_exists, 
                                               index=False, 
                                               #index_label=dxcol.index.names,
                                               )
                
                #addd keys
                if first: 
                    with conn.cursor() as cur:
                        cur.execute(f"""
                        ALTER TABLE {schemaName}.{tableName}
                            ADD PRIMARY KEY (gid, id);""")
                    

                
                
                first = False
                
            #close connection. wrap
            log.debug(f'finished {country_key}')
            tab_d[country_key] = tableName
            
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
        
            
 

if __name__ == '__main__':
    run_to_postgres(conn_str=conn_str,
                    country_l = [
                        #'AUS', 
                        'BGD', 
                        'BRA', 'CAN', 'DEU', 'ZAF'],
                    )