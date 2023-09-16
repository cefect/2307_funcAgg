'''
Created on Sep. 9, 2023

@author: cefect
'''
#===============================================================================
# IMPORTS--------
#===============================================================================
import os, hashlib, sys, subprocess, psutil
from datetime import datetime
from itertools import product

import psycopg2
print('psycopg2.__version__=' + psycopg2.__version__)

#from sqlalchemy import create_engine, URL

from tqdm import tqdm

import pandas as pd

from coms import (
    init_log, today_str, get_log_stream, get_directory_size,
    dstr, view, 
    )

from agg.coms_agg import get_conn_str, pg_vacuum, pg_spatialIndex, pg_exe

 

from definitions import index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir


def run_merge_agg_grids(
        conn_d=postgres_d,
        tbl_grids_l = [
            #'agg_aus_0000240',
            #'agg_aus_0001020',
            'agg_bgd_0000060',
            'agg_bgd_0000240',
            'agg_bgd_0001020',
            'agg_bgd_0100000',
 
            # 'agg_bra_0000240',
            # 'agg_bra_0001020',
            # 'agg_can_0000240',
            # 'agg_can_0001020',
            # 'agg_deu_0000060',
            # 'agg_deu_0000240',
            # 'agg_deu_0001020',
            # 'agg_zaf_0000240',
            # 'agg_zaf_0001020',
            ],
                
        schema='grids', viewName='agg',
        ):
    """merge the agg grids"""
    
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
        
        #===========================================================================
        # setup
        #===========================================================================
        #remove if it exists
        with conn.cursor() as cur:
            cur.execute(f"""DROP VIEW IF EXISTS {schema}.{viewName}""")
        conn.commit()
        
        #create if it exists

        
        #=======================================================================
        # build
        #=======================================================================
        #start the query with the first table
        cmd_str = f"""
        CREATE VIEW {schema}.{viewName} AS
            SELECT * FROM {schema}.{tbl_grids_l[0]}
        """
        
        #add the rest of the tables to the union statement
        print(f'building union call w/ {len(tbl_grids_l)} tables')
        for tableName_i in tbl_grids_l[1:]:
            cmd_str += f"""
                UNION ALL
                SELECT * FROM {schema}.{tableName_i}
            """
        with conn.cursor() as cur:
            print(cmd_str)
            cur.execute(cmd_str)
        conn.commit()

        #=======================================================================
        # first=True
        # for tableName_i in tqdm(tbl_grids_l):
        #                 
        #     #create it
        #     if first:
        #         with conn.cursor() as cur:
        #             cmd_str=f"""
        #                 CREATE VIEW {schema}.{viewName} AS
        #                     SELECT * FROM {schema}.{tableName_i}
        #                         WHERE false"""
        #             print(cmd_str)
        #             cur.execute(cmd_str)
        #             
        #         conn.commit()
        #         
        #     #populate query
        #     with conn.cursor() as cur:
        #         cmd_str = f"""
        #             CREATE OR REPLACE VIEW {schema}.{viewName} AS
        #                 SELECT * FROM {schema}.{viewName}
        #                 UNION ALL
        #                 SELECT * FROM {schema}.{tableName_i}
        #         """
        #         print(cmd_str)
        #         cur.execute(cmd_str)
        #         
        #     first=False
        #=======================================================================
            
    #===========================================================================
    # report
    #===========================================================================
    cmd_str = """SELECT country_key, grid_size, COUNT(*)
            FROM grids.agg
            GROUP BY country_key, grid_size
            ORDER BY country_key, grid_size"""
    print(cmd_str)
    res = pg_exe(cmd_str)
    
    print(pd.DataFrame(res, columns=['country_key', 'grid_size', 'count']))
 
    #===========================================================================
    # clean up
    #===========================================================================
    """not needed by a view
    pg_spatialIndex(conn_d, schema, tableName)
    pg_vacuum(conn_d, f'{schema}.{tableName}')"""
                
    print(f'finished')
    

def run_clean_inters(
        conn_d=postgres_d,
        country_l = ['aus', 'bra', 'can', 'deu', 'zaf'], 
        schema='inters',  
        ):
    """merge the agg grids"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    #create if it exists
    if country_l is  None: country_l=[e.lower() for e in index_country_fp_d.keys()]
    
    #===========================================================================
    # clean and index
    #===========================================================================
    for tableName in tqdm(country_l):
        print(f'pg_vacuum {tableName}')
        pg_vacuum(conn_d, f'{schema}.{tableName}')
        
        print(f'pg_spatialIndex {tableName}')
        pg_spatialIndex(conn_d, schema, tableName, columnName='geometry')
        
    print(f'finished')
    return
    
def run_merge_inters(
        conn_d=postgres_d,
        country_l = None, 
        schema='inters', viewName='pts_osm_fathom',
        ):
    """merge the agg grids"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    #create if it exists
    if country_l is  None: country_l=[e.lower() for e in index_country_fp_d.keys()]
    
 
        
    
    #===========================================================================
    # create view
    #===========================================================================
    print(f'creating view on {len(country_l)}')
    pg_exe(f"""DROP VIEW IF EXISTS {schema}.{viewName}""")
    
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
 
 
        #=======================================================================
        # build
        #=======================================================================
        #start the query with the first table
        cmd_str = f"""
        CREATE VIEW {schema}.{viewName} AS
            SELECT * FROM {schema}.{country_l[0]}
        """
        
        #add the rest of the tables to the union statement
        print(f'building union call w/ {len(country_l)} tables')
        for tableName_i in country_l[1:]:
            cmd_str += f"""
                UNION ALL
                SELECT * FROM {schema}.{tableName_i}
            """
        with conn.cursor() as cur:
            print(cmd_str)
            cur.execute(cmd_str)

            
    #===========================================================================
    # report
    #===========================================================================
 
    res = pg_exe(f"""SELECT country_key, COUNT(*)
            FROM {schema}.{viewName}
            GROUP BY country_key""", return_fetch=True)
    
    print(pd.DataFrame(res, columns=['country_key', 'count']))
 
    #===========================================================================
    # clean up
    #===========================================================================
    """not needed by a view
    pg_spatialIndex(conn_d, schema, tableName)
    pg_vacuum(conn_d, f'{schema}.{tableName}')"""
                
    print(f'finished')
    
      
    
if __name__ == '__main__':
    run_merge_inters()
    """
      country_key     count
0         AUS   2333745
1         BGD   4647944
2         BRA   6450206
3         CAN   5754035
4         DEU  25994846
5         ZAF    986456
"""
    #run_clean_inters()
    #run_merge_agg_grids()
        
