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

#from _02agg._07_views import create_view_join_grid_geom
from _03damage._03_rl_agg import load_rl_dx
from _05depths._03_gstats import get_a03_gstats_1x

from coms import (
    init_log, today_str,  dstr, view
    )

#from _03damage._05_mean_bins import filter_rl_dx_minWetFrac, get_grid_rl_dx


def get_total_losses(
        country_key='deu', 

        dx_raw=None,
        log=None,dev=False,use_cache=True,out_dir=None, use_aoi=False,
        ):
    """total losses (building weighted)"""
     
     
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
     
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'damage','06_total', country_key)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    
    if log is None:
        log = init_log(name=f'grid_rl')
    
 
    #===========================================================================
    # load
    #===========================================================================
    if dx_raw is None:     
        dx_raw = load_rl_dx(country_key=country_key, log=log, use_cache=use_cache, dev=dev, use_aoi=use_aoi)
            
    #===========================================================================
    # cache
    #===========================================================================
    fnstr = f'grid_TL_{country_key}'
    uuid = hashlib.shake_256(f'{fnstr}_{dev}_{dx_raw.shape}_{dx_raw.head()}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')

    if (not os.path.exists(ofp)) or (not use_cache):
        dx1 = dx_raw
        #===========================================================================
        # load depth group stats and add some to the index
        #===========================================================================
        """joining some stats from here"""
        wdx_raw = get_a03_gstats_1x(country_key=country_key, log=log, use_cache=use_cache)
        
        
        
        wdx1 = wdx_raw.stack(level='haz_key').reset_index(['bldg_cnt', 'null_cnt'])
        wdx1.columns.name=None
        
        #rename
        wdx1 = wdx1.rename(columns={'avg':'grid_wd'})
        
        col_l = ['grid_wd', 'bldg_cnt', 'wet_cnt'] #data to join
        
        #join to index
        dx1.index = pd.MultiIndex.from_frame(dx1.index.to_frame().join(wdx1.loc[:, col_l]))
        
 
        #===========================================================================
        # compute
        #===========================================================================
        
        #get total loss
        rdx = dx1.multiply(dx1.index.get_level_values('bldg_cnt'), axis=0)
        
        
        #=======================================================================
        # sum check
        #=======================================================================
        cdx1 = rdx.xs('bldg', level='rl_type', axis=1).groupby(['grid_size', 'haz_key']).sum().unstack('haz_key')
        
        cser1 = cdx1.round(0).nunique()
        cser2 = cser1[cser1!=1.0]
        
        if len(cser2)>0:
            """something to do with BN_FLEMO not starting at zero"""
            log.warning(f'got {len(cser2)} events with different building totals\n{cser2}')
        
        
        """
        
        cdx2 = pd.concat([cdx1, cdx1.round(1).nunique().to_frame().T])
        view(cdx1.round(1).nunique())
        view(cdx2)
        view(rdx.groupby(['grid_size', 'haz_key']).sum().unstack('haz_key'))
        
        bx= dx1.index.get_level_values('wet_cnt')>0
        view(dx1[bx].head(100))
        """
        
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



