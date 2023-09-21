'''
Created on Jul. 25, 2023

@author: cefect


grids with buildings and the counts
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
 
 


from concurrent.futures import ProcessPoolExecutor

import psycopg2
from sqlalchemy import create_engine, URL

from tqdm import tqdm



from definitions import (
    wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d, postgres_d, 
    equal_area_epsg, fathom_vals_d, gridsize_default_l
    )
from definitions import temp_dir as temp_dirM
 


from _02agg.coms_agg import (
    get_conn_str, pg_getCRS, pg_to_df, pg_exe, pg_getcount, pg_spatialIndex, pg_get_column_names,
    pg_vacuum, pg_comment, pg_register
    )

from coms import (
    init_log, today_str, get_log_stream, get_raster_point_samples, get_directory_size,
    dstr
    )


 



#===============================================================================
# EXECUTORS--------
#===============================================================================
 
def run_grids_occupied_stats(
                        country_key, 
                           #hazard_key,
                               grid_size,
                           out_dir=None,
                           dev=False,
                           conn_str=None,
                           epsg_id=equal_area_epsg,
                           log=None,
 
                           ):
    """grids with building exposure (and some building stats)
    
    NOTE: this function is discretized differently than the other ones...
        would be better to have grid_size as a list/loop
        
        
    Returns
    ---------
    postgres table [agg_bldg.agg_occupied_{country_key}_{grid_size:04d}]
        grid to building links
    """
    

    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    country_key=country_key.lower()
    #assert hazard_key in index_hazard_fp_d, hazard_key
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'agg','04_occu', country_key,  f'{grid_size:05d}')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    if log is None:
        log = init_log(name=f'occu.{country_key}.{grid_size}', fp=os.path.join(out_dir, today_str+'.log'))
    
    
    keys_d = {'country_key':country_key, 
              #'hazard_key':hazard_key, 
              'grid_size':grid_size}
    log.info(f'on {keys_d}')
    
    sql = lambda x:pg_exe(x, conn_str=conn_str, log=log)
    #===========================================================================
    # get table names
    #===========================================================================
    link_tableName=f'bldgs_grid_link_{country_key}_{grid_size:04d}'    
    grid_tableName=f'agg_{country_key}_{grid_size:07d}'    
    new_tableName=f'agg_occupied_{country_key}_{grid_size:04d}'
    #bldg_expo_tn = country_key.lower()
    
    if dev:
        out_schema='dev'
        link_schema='dev'
        grid_schema='dev'
        #bldg_expo_sch='dev'
        
    else:
        link_schema='agg_bldg'
        out_schema='agg_bldg'
        grid_schema='grids'
        #bldg_expo_sch='inters'
            
    #===========================================================================
    # create a temp table of unique indexers
    #===========================================================================
    """use the building-grid links to construct an index of i,j values with exposed buildings
    
    NOTE: this means we lose grids with centroid exposure (but dry or no buildigns)
        not a bad thing
    """  
    schema1 = 'temp'  
    tableName1= new_tableName+'_exposed'
    if dev: tableName1+='_dev'
    log.info(f'creating \'{schema1}.{tableName1}\' from unique i,j columns from {link_tableName}')     
    sql(f"DROP TABLE IF EXISTS {schema1}.{tableName1}")
      
 
      
    #get exposed indexers and their feature counts
    sql(f'''CREATE TABLE {schema1}.{tableName1} AS 
                SELECT LOWER(ltab.country_key) AS country_key, ltab.grid_size, ltab.i, ltab.j, COUNT(*) as bldg_expo_cnt
                    FROM {link_schema}.{link_tableName} AS ltab
                            GROUP BY ltab.country_key, ltab.grid_size, ltab.i, ltab.j''')
      
    #add the primary key
    sql(f"ALTER TABLE {schema1}.{tableName1} ADD PRIMARY KEY (country_key, i, j)")
     
    ##report grid counts
    """something is wrong with the tables... vacuuming and rebuilding indexes now"""
    #===========================================================================
    # this is very slow?
    # log.info(f'finished indexers query')
    # ij_expo_cnt = pg_getcount(schema1, tableName1)
    # ij_cnt = pg_getcount('grids', grid_tableName)    
    #  
    #  
    # #report asset counts
    # grid_asset_cnt = int(pg_exe(f"SELECT SUM(bldg_expo_cnt) as total_fcnt FROM {schema1}.{tableName1}", return_fetch=True)[0][0])
    # asset_cnt = pg_getcount(link_schema, link_tableName)
    #  
    # log.info(f'identified {ij_expo_cnt}/{ij_cnt} unique grids with {asset_cnt} assets')
    #  
    # if not grid_asset_cnt==asset_cnt:
    #     log.warning(f'asset count on grids ({grid_asset_cnt:,}) differs from \'inters_agg\' ({asset_cnt:,})')
    #===========================================================================
        
        
    #===========================================================================
    # join grid geometryu
    #===========================================================================
    log.info(f'\n\njoining grid geometry')
    
    tableName2 = tableName1+'_wgeo'
    if dev: tableName2+='_dev'    
    
    sql(f"DROP TABLE IF EXISTS {out_schema}.{new_tableName}") 
      
    #get exposed indexers and their feature counts
    sql(f'''CREATE TABLE {out_schema}.{new_tableName} AS 
                SELECT ltab.*, ST_Centroid(rtab.geom) as geom
                    FROM {schema1}.{tableName1} AS ltab
                        LEFT JOIN {grid_schema}.{grid_tableName} as rtab
                            ON ltab.i=rtab.i AND ltab.j=rtab.j AND ltab.grid_size=rtab.grid_size AND ltab.country_key=rtab.country_key''')
    
    #post
    #===========================================================================
    # assert pg_getCRS(schema1, tableName2)==epsg_id
    # pg_exe(f'ALTER TABLE {schema1}.{tableName2} ADD PRIMARY KEY (country_key, grid_size, i, j)')
    # pg_register(schema1, tableName2)
    # pg_spatialIndex(schema1, tableName2, log=log)
    #===========================================================================
    
    
    #===========================================================================
    # compute building datats on exposed grid
    #===========================================================================
#===============================================================================
#     """this is too slow"""
#     start_i = datetime.now()
#     fcnt = pg_getcount(schema1, tableName2)
#     
#     log.info(f'\n\ncomputing buidling stats on {fcnt} grids') 
#      
#     sql(f"DROP TABLE IF EXISTS {out_schema}.{new_tableName}")
#     
#     #build columns
#     cols = f'gtab.country_key, gtab.grid_size,gtab.i, gtab.j, COUNT(bldg.id) as bldg_cnt, '
#     
#     #coln_l = pg_get_column_names(bldg_expo_sch, bldg_expo_tn) 
#     
#     #===========================================================================
#     # #add wet counts
#     # """too slow?"""
#     # haz_coln_l = ['f010_fluvial', 'f050_fluvial', 'f100_fluvial', 'f500_fluvial']
#     # 
#     # cols+=', \n'.join([f'COUNT(CASE WHEN bldg.{e} > 0 THEN 1 ELSE NULL END) as {e}_wetCnt' for e in haz_coln_l])
#     #===========================================================================
#     
#  
#     #add geometry
#     """decided to include the centroid here as we'll need it in sample"""
#     #cols+=f', ST_Transform(ST_Centroid(gtab.geom), {epsg_id}) as geom'
#     cols+=f'ST_Centroid(gtab.geom) as geom'
#     
#     sql(f"""
#     CREATE TABLE {out_schema}.{new_tableName} AS
#         SELECT {cols}
#             FROM {schema1}.{tableName2} as gtab
#                 LEFT JOIN {bldg_expo_sch}.{bldg_expo_tn} as bldg
#                     ON ST_Intersects(gtab.geom, ST_Transform(bldg.geometry, {epsg_id}))
#                         GROUP BY gtab.country_key, gtab.grid_size, gtab.i, gtab.j
#     """)
#      
#     
# 
#     log.info(f'finished building stats in {(datetime.now()-start_i).total_seconds():.2f} secs')
#===============================================================================
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'\n\nwrap')
    

    #key
    pg_exe(f'ALTER TABLE {out_schema}.{new_tableName} ADD PRIMARY KEY (country_key, grid_size, i, j)')
    
    #comment
    cmt_str = f'grids with exposed buildings, building counts, and wet counts per hazard\n'
    cmt_str += f'built with {os.path.realpath(__file__)} at '+datetime.now().strftime("%Y.%m.%d.%S")
    pg_comment(out_schema, new_tableName, cmt_str)
    
    #spatisl
    pg_register(out_schema, new_tableName)
    assert pg_getCRS(out_schema, new_tableName)==epsg_id
    pg_spatialIndex(out_schema, new_tableName)
    
    #clean up
    pg_vacuum(out_schema, new_tableName)
    
    #drop the temps
    sql(f"DROP TABLE IF EXISTS temp.{new_tableName}")
    #sql(f"DROP TABLE IF EXISTS {schema1}.{tableName2}")
    
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(f'finished on \'{new_tableName}\' w/ \n    {meta_d}')
    
    return
        
def run_all(ck, **kwargs):
    log = init_log(name='occu')
    
    for grid_size in gridsize_default_l:
        run_grids_occupied_stats(ck, grid_size, log=log, **kwargs)
    
 
if __name__ == '__main__':
    
    #run_grids_occupied_stats('DEU', 60, dev=True)
    
    run_all('deu', dev=True)
    
    
    
    
    
    
    
    
    
    
    