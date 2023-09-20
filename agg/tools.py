'''
Created on Sep. 20, 2023

@author: cefect


misc tools
'''

#===============================================================================
# IMPORTS--------
#===============================================================================
import os, hashlib, sys, subprocess

 
 
import psutil
from datetime import datetime
import pandas as pd
import numpy as np
import fiona
import geopandas as gpd
 
 
import psycopg2
from sqlalchemy import create_engine, URL

from tqdm import tqdm



from definitions import (
    wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d, postgres_d, 
    equal_area_epsg, fathom_vals_d, gridsize_default_l
    )
from definitions import temp_dir as temp_dirM
 


from agg.coms_agg import (
    get_conn_str, pg_getCRS, pg_to_df, pg_exe, pg_getcount, pg_spatialIndex, pg_get_column_names,
    pg_vacuum, pg_comment, pg_register
    )

from coms import (
    init_log, today_str, get_log_stream, get_raster_point_samples, get_directory_size,
    dstr
    )


def pg_spatialIndex_vacuum(
        schema_tables=[
            ('grids','agg_deu_0000060'),
            ('grids','agg_deu_0000240'),
            ('grids','agg_deu_0001020'),
            ('inters','deu'),
            ],
        conn_str=None,
        dev=False,
        ):
    
    """rebuild spatial indeex and vacuum tables"""
    #===========================================================================
    # defaults
    #===========================================================================
    
    log = init_log(name='maint')
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    
    start =datetime.now()
    
                    
    #===========================================================================
    # loop and apply
    #===========================================================================
    for schema, tableName in schema_tables:
        """this can be very slow...."""
        if dev: schema='dev'
        start_i = datetime.now()
        args = (schema, tableName)
        #fcnt = pg_getcount(schema, tableName)
        fcnt=0
        
        coln_l = pg_get_column_names(*args)
        
        
        
        gcoln=None
        for k in coln_l:
            if k.startswith('geom'): 
                gcoln=k
                break

        
        assert gcoln in coln_l
        
        log.info(f'on {schema}.{tableName} w/ {fcnt} rows and cols \n    {coln_l}')
        #=======================================================================
        # spatial index
        #=======================================================================
        
        pg_register(*args)
        
        index_name = f'{tableName}_geom_idx'
        sql(f'DROP INDEX IF EXISTS {schema}.{index_name}')
         
        pg_spatialIndex(*args, columnName=gcoln)
        
        #=======================================================================
        # vacuum
        #=======================================================================
        pg_vacuum(*args)
        
        log.info(f'finished in {(datetime.now()-start_i).total_seconds():.2f} secs\n\n')
    
    #===========================================================================
    # wrap
    #===========================================================================
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    #'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(f'finished on {len(schema_tables)} \n    {meta_d}')
    
    return
        
        
        
        
        
        
        
if __name__ == '__main__':
    
    pg_spatialIndex_vacuum(dev=True)
    
    
    
    
    
    
    
    
    
    
    
