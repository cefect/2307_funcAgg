'''
Created on Sep. 22, 2023

@author: cefect

compute total losses (multiply by building count)
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
#print('psycopg2.__version__=' + psycopg2.__version__)
from sqlalchemy import create_engine, URL
 

from definitions import (
    wrk_dir,   postgres_d, temp_dir,
    equal_area_epsg, fathom_vals_d, gridsize_default_l
    )
 

from _02agg.coms_agg import (
    get_conn_str, pg_getCRS, pg_to_df, pg_exe, pg_getcount, pg_spatialIndex, pg_get_column_names,
    pg_vacuum, pg_comment, pg_register, pg_table_exists
    )

from _02agg._07_views import create_view_join_grid_geom

from coms import (
    init_log, today_str,  dstr, view
    )

from _03damage._05_mean_bins import filter_rl_dx_minWetFrac, get_grid_rl_dx


def get_total_losses(
        country_key='deu', 
        haz_key='f500_fluvial',
        dx_raw=None,
        log=None,dev=False,use_cache=True,out_dir=None,
        ):
    """total losses (building weighted)"""
     
     
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
     
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'damage','06_total', country_key, haz_key)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    
    if log is None:
        log = init_log(name=f'grid_rl')
    
 
    #===========================================================================
    # load
    #===========================================================================
    if dx_raw is None:     
        dx_raw = get_grid_rl_dx(country_key, haz_key, log=log, use_cache=use_cache, dev=dev)
            
    #===========================================================================
    # cache
    #===========================================================================
    fnstr = f'grid_TL_{country_key}_{haz_key}'
    uuid = hashlib.shake_256(f'{fnstr}_{dev}_{dx_raw.shape}_{dx_raw.head()}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')

    if (not os.path.exists(ofp)) or (not use_cache):

            
        dx1 = dx_raw.droplevel('country_key')
        
        #===========================================================================
        # compute
        #===========================================================================
        
        #get total loss
        rdx = dx1.multiply(dx1.index.get_level_values('bldg_cnt'), axis=0)
        
        #=======================================================================
        # write
        #=======================================================================
        log.info(f'got {rdx.shape} writing to \n    {ofp}')
        
        rdx.to_pickle(ofp)
        
    else:
        log.info(f'loading from cache')
        rdx = pd.read_pickle(ofp)
    
    return rdx
    
 
    
    
    
if __name__=='__main__':
    
    
    get_total_losses(dev=False, use_cache=False)
    
    
    
    print('done')



