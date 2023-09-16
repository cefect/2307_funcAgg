'''
Created on Jul. 25, 2023

@author: cefect


remove grids that do not intersect buildings
    could do this in a few places...
    but at this point we can take advantage of the spatial joins in the inters_agg tables
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
from osgeo import ogr
import rasterstats
from rasterstats import zonal_stats


from concurrent.futures import ProcessPoolExecutor

import psycopg2
from sqlalchemy import create_engine, URL

from tqdm import tqdm



from definitions import (
    wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d, postgres_d, 
    equal_area_epsg, fathom_vals_d
    )
from definitions import temp_dir as temp_dirM
 


from agg.coms_agg import get_conn_str, pg_getCRS, pg_to_df, pg_exe, pg_getcount, pg_spatialIndex

from coms import (
    init_log, today_str, get_log_stream, get_raster_point_samples, get_directory_size,
    dstr
    )


 



#===============================================================================
# EXECUTORS--------
#===============================================================================
 
def run_purge_grids(
                        country_key, 
                           hazard_key,
                               grid_size,
                           out_dir=None,
 
 
                           ):
    """build a new table of agg grids with just those intersecting buildings"""
    
    raise IOError('looks like 2 grids within DEU are missing')
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    assert hazard_key in index_hazard_fp_d, hazard_key
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'agg','04_purge', country_key, hazard_key, f'{grid_size:05d}')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    
    log = init_log(name=f'purge.{country_key}.{hazard_key}.{grid_size}', fp=os.path.join(out_dir, today_str+'.log'))
    log.info(f'on {country_key} x {hazard_key} x {grid_size}')
    
    keys_d = {'country_key':country_key, 'hazard_key':hazard_key, 'grid_size':grid_size}
 
    #===========================================================================
    # get table names
    #===========================================================================
    sample_tableName=f'pts_osm_fathom_{country_key}_{grid_size:07d}'
    grid_tableName=f'agg_{country_key}_{grid_size:07d}'
    new_tableName=grid_tableName+'_wbldg'
            
    #===========================================================================
    # create a temp table of unique indexers
    #===========================================================================    
    log.info(f'creating table {new_tableName} from unique i,j columns from {sample_tableName}')
    
    pg_exe(f"DROP TABLE IF EXISTS temp.{new_tableName}")
    
    pg_exe(f"CREATE TABLE temp.{new_tableName} AS SELECT DISTINCT i, j FROM inters_agg.{sample_tableName}")
    
    #add the primary key
    pg_exe(f"ALTER TABLE temp.{new_tableName} ADD PRIMARY KEY (i, j)")
 
    #report
    ij_expo_cnt = pg_getcount('temp', new_tableName)
    ij_cnt = pg_getcount('grids', grid_tableName)    
    log.info(f'identified {ij_expo_cnt}/{ij_cnt} unique grids with buildings')
    
    #===========================================================================
    # join the grids to this
    #===========================================================================
    log.info(f'joing grids')
    pg_exe(f"DROP TABLE IF EXISTS grids.{new_tableName}")
    pg_exe(f"""
    CREATE TABLE grids.{new_tableName} AS
        SELECT gtab.country_key, gtab.grid_size, bldgs.i,bldgs.j, gtab.geom
            FROM temp.{new_tableName} as bldgs
            JOIN grids.{grid_tableName} as gtab ON bldgs.i = gtab.i AND bldgs.j = gtab.j
    """)
    
    pg_exe(f"DROP TABLE IF EXISTS temp.{new_tableName}")
    
    pg_spatialIndex('grids', new_tableName)
    
    #===========================================================================
    # wrap
    #===========================================================================
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
        
 
 
if __name__ == '__main__':
    
    run_purge_grids('DEU', '500_fluvial', 1020)
    
    
    
    
    
    
    
    
    
    
    