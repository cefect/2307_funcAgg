'''
Created on Sep. 21, 2023

@author: cefect
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
    pg_vacuum, pg_comment, pg_register
    )

from _02agg._07_views import create_view_join_grid_geom

from coms import (
    init_log, today_str,  dstr, view
    )



def get_grid_rl_dx(
        country_key, haz_key,
 
        log=None,conn_str=None,dev=False,use_cache=True,out_dir=None,
        limit=None,
 
        ):
    
    """helper to retrieve results from run_view_join_depths() as a dx
    
     WARNING: this relies on _04expo.create_view_join_stats_to_rl()
    """ 
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'damage','05_means', country_key, haz_key)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    
    if log is None:
        log = init_log(name=f'grid_rl')
    
    if dev: use_cache=False
    #===========================================================================
    # cache
    #===========================================================================
    fnstr = f'grid_rl_{country_key}_{haz_key}'
    uuid = hashlib.shake_256(f'{fnstr}_{dev}_{limit}'.encode("utf-8"), usedforsecurity=False).hexdigest(8)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')
    
    if (not os.path.exists(ofp)) or (not use_cache):
        
        #===========================================================================
        # talbe params
        #===========================================================================
        #see create_view_join_stats_to_rl()
        tableName = f'grid_rl_wd_bstats_{country_key}_{haz_key}' 
        
        if dev:
            schema = 'dev'
    
        else:
            schema = 'damage' 
            
        keys_l = ['country_key', 'grid_size','haz_key', 'i', 'j']
        
        #===========================================================================
        # download
        #===========================================================================
        conn =  psycopg2.connect(conn_str)
        engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
        
        #row_cnt=0
        
        """only ~600k rows"""
        
        cmd_str = f'SELECT * FROM {schema}.{tableName}'
        
        if not limit is None:
            cmd_str+=f'\n    LIMIT {limit}'
 
        log.info(cmd_str)
        df_raw = pd.read_sql(cmd_str, engine, index_col=keys_l)
        """
        view(df_raw.head(100))        
        """    
        
        engine.dispose()
        conn.close()
        
        log.info(f'finished w/ {len(df_raw)} total rows')
        
        #===========================================================================
        # clean up
        #===========================================================================
        #exposure meta
        expo_colns = ['bldg_expo_cnt', 'grid_wd', 'bldg_cnt', 'wet_cnt']
        df1 = df_raw.copy()
        df1.loc[:, expo_colns] = df1.loc[:, expo_colns].fillna(0.0)        
        df1=df1.set_index(expo_colns, append=True)
        
        #split bldg and grid losses
        col_bx = df1.columns.str.contains('_mean') 
        
        grid_dx = df1.loc[:, ~col_bx]
        rnm_d = {k:int(k.split('_')[1]) for k in grid_dx.columns.values}
        grid_dx = grid_dx.rename(columns=rnm_d).sort_index(axis=1)
        grid_dx.columns = grid_dx.columns.astype(int)
        
        
        bldg_dx = df1.loc[:, col_bx]
        rnm_d = {k:int(k.split('_')[1]) for k in bldg_dx.columns.values}
        bldg_dx = bldg_dx.rename(columns=rnm_d).sort_index(axis=1)
        bldg_dx.columns = bldg_dx.columns.astype(int)
        
        assert np.array_equal(grid_dx.columns, bldg_dx.columns)
     
        
        dx = pd.concat({
            'bldg_mean':bldg_dx, 
            'grid_cent':grid_dx, 
            #'expo':df.loc[:, expo_colns].fillna(0.0)
            }, 
            names = ['rl_type', 'df_id'], axis=1).dropna(how='all') 
        
        #===========================================================================
        # write
        #===========================================================================
        """
        view(dx.head(100))
        """
        
 
        log.info(f'writing {dx.shape} to \n    {ofp}')
        dx.sort_index(sort_remaining=True).sort_index(sort_remaining=True, axis=1).to_pickle(ofp)
    
    else:
        log.info(f'loading from cache:\n    {ofp}')
        dx = pd.read_pickle(ofp)
 
 
    log.info(f'got {dx.shape}')
    return dx

def g(df):
    bins = np.linspace(0, 1, 10)
    groups = df.groupby(pd.cut(df.x, bins))
    return groups.mean()

def run_bldg_rl_mean_bins(
        country_key='deu', haz_key='f500_fluvial',
        dx_raw=None,
        log=None,dev=False,use_cache=True,out_dir=None,
        ):
    """compute a binned mean of the bldg RL"""
     
     
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
     
    if log is None:
        log = init_log(name=f'grid_rl')

    #===========================================================================
    # load
    #===========================================================================
    if dx_raw is None: 
        dx_raw = get_grid_rl_dx(country_key, haz_key, log=log, use_cache=True, dev=dev)
        
    #===========================================================================
    # filter
    #===========================================================================
    dx1 = filter_rl_dx_minWetFrac(dx_raw, log=log)
    """no need to preserve grid_wd styled index"""
    
    keys_l =  ['country_key', 'grid_size', 'haz_key'] #only keys we preserve
    #get clean
     
    dx2 = dx1.xs('bldg_mean', level='rl_type', axis=1)
     
    #get a slice with clean index
    dx3 = dx2.reset_index(keys_l+['grid_wd']).reset_index(drop=True).set_index(keys_l+['grid_wd'])
     
    #get the meanned bins
    compute_binned_mean(dx3, log=log)
             
 

def filter_rl_dx_minWetFrac(dx1, min_wet_frac=0.95, log=None):
    mdex = dx1.index
    mdf = mdex.to_frame().reset_index(drop=True)
    assert mdf['wet_cnt'].max() > 0, 'something is wrong with the building stats'
    mdf['wet_frac'] = mdf['wet_cnt'] / mdf['bldg_cnt']
    assert np.all(mdf['wet_frac'].max() <= 1.0)
    bx = (mdf['wet_frac'] > min_wet_frac).values
    assert bx.any()
    log.info(f'selected {bx.sum()}/{len(bx)} w/ min_wet_frac={min_wet_frac}')
 
    return dx1.loc[bx, :].droplevel(['bldg_expo_cnt', 'wet_cnt', 'bldg_cnt'])
        
def compute_binned_mean(dx_raw, log=None, out_dir=None, use_cache=False, bin_cnt=21):
    """calc binned mean"""
    
    if out_dir is None:
        out_dir = os.path.join(temp_dir, 'damage','05_means', 'compute_binned_mean')
    if not os.path.exists(out_dir):os.makedirs(out_dir)  
  
     
    if log is None:
        log = init_log(name=f'grid_rl')
     
 
    #===========================================================================
    # cache
    #===========================================================================
    fnstr = f'grid_rl_meansBin'
    uuid = hashlib.shake_256(f'{dx_raw.head()}_{dx_raw.shape}_{dx_raw.index.shape}'.encode("utf-8"), usedforsecurity=False).hexdigest(8)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')
     
    #===========================================================================
    # biuld
    #===========================================================================
    if (not os.path.exists(ofp)) or (not use_cache):
 
        #prep the data
        serx = dx_raw.stack().swaplevel().sort_index(sort_remaining=True).rename('bldg_mean_rl')
        
        #keys to group iterate over
        keys_l = [e for e in serx.index.names if not 'grid_wd' in e]
        
        #bins
        bins_ar = np.linspace(0, serx.index.unique('grid_wd').max(), bin_cnt)
        #bins_ar = np.linspace(0, 100, bin_cnt)
        
        d = dict()
        for keys, gserx in serx.groupby(keys_l, axis=0):
 
 
            #reshape
            df = gserx.reset_index('grid_wd').reset_index(drop=True).dropna(how='any')
            
            #get categories for this
            categories = pd.cut(df['grid_wd'], bins_ar)
            
            #mean per category
            """this gives the mean of the x values also... which are probably better to plot against?"""
            d[keys] = df.groupby(categories, observed=False).mean().rename(columns = {'grid_wd':'grid_wd_bin', 'bldg_mean_rl':'bldg_mean_rl_bin'})
            
        res_dx = pd.concat(d, names=serx.index.names)
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info(f'writing {res_dx.shape} to \n    {ofp}')
        res_dx.to_pickle(ofp)
        
    else:
        log.info(f'loading from cache...')
        res_dx = pd.read_pickle(ofp)
        
    return res_dx
    

if __name__=='__main__':
    
    
    run_bldg_rl_mean_bins('deu', dev=False)
    
    
    
    
    
    
    
    