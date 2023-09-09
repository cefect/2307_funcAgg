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

from sqlalchemy import create_engine, URL

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
            #'agg_bgd_0000060',
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
    res = pg_exe(
        """SELECT country_key, grid_size, COUNT(*)
            FROM grids.agg
            GROUP BY country_key, grid_size
            ORDER BY country_key, grid_size"""
            )
    
    print(pd.DataFrame(res, columns=['country_key', 'grid_size', 'count']))
 
    #===========================================================================
    # clean up
    #===========================================================================
    """not needed by a view
    pg_spatialIndex(conn_d, schema, tableName)
    pg_vacuum(conn_d, f'{schema}.{tableName}')"""
                
    print(f'finished')
    
    
    
    
    
if __name__ == '__main__':
    #run_build_agg_grids()
    run_merge_agg_grids()
        
