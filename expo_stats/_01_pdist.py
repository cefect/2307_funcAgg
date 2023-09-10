'''
Created on Sep. 9, 2023

@author: cefect


fitting probability distributions to the aggregated -intersect datasetes
'''
 
#===============================================================================
# IMPORTS-----
#===============================================================================
import os, hashlib, sys, subprocess, psutil
from datetime import datetime
from itertools import product
import concurrent.futures

import psycopg2
print('psycopg2.__version__=' + psycopg2.__version__)

from sqlalchemy import create_engine, URL

from tqdm import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd
idx = pd.IndexSlice

from scipy.stats import expon


from coms import (
    init_log, today_str, get_directory_size,dstr, view,  get_log_stream
    ) 

from agg.coms_agg import get_conn_str, pg_exe

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l,
    haz_coln_l
    )

schema='inters_agg'


def _futures_buid_grid(grid_size, country_key, i, min_size, out_dir):
    tableName=f'pts_osm_fathom_{country_key}_{grid_size:07d}'
    log = get_log_stream()

    #set filepath
    uuid = hashlib.shake_256(f'{tableName}_{min_size}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir,f'{tableName}_{uuid}.geojson')

    if not os.path.exists(ofp):
        dx = _calc_grid_country(tableName, log, min_size)
        dx.to_pickle(ofp)
        log.info(f'wrote {dx.shape} to \n    {ofp}')
    else:
        log.info(f'    filepath exists... skipping')

    return i, ofp


def run_build_pdist(
        country_l = ['bgd', 'deu'],
        grid_size_l=None,
        
        conn_d=postgres_d,
        out_dir=None,
        min_size=5,
        #nonzero_frac_thresh=0.99,
        max_workers=None,
 
        ):
    """fit pdist to each clump
    
    
    Pars
    ---------
    min_size: int
        minimum clump size to consider
        
    nonzero_frac_thresh: float
        minimum non-zero fraction of clump to accept
        using this to exclude edge/fringe clumps
        
    """
    
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()    
 
 
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'expo_stats', 'pdist')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    
    log = init_log(name=f'pdist', fp=os.path.join(out_dir, today_str+'.log'))
    
    if grid_size_l is None: grid_size_l=gridsize_default_l
    if country_l is  None: country_l=[e.lower() for e in index_country_fp_d.keys()]
    #if epsg_id is None: epsg_id=equal_area_epsg
    
    log.info(f'on \n    {country_l}\n    {conn_d}')
    
    #===========================================================================
    # retrieve dataframe from postgis
    #===========================================================================
    if max_workers is None:
        res_d = dict()
        for i, (grid_size, country_key) in enumerate(product([int(e) for e in grid_size_l], country_l)):
            tableName=f'pts_osm_fathom_{country_key}_{grid_size:07d}'
            log.info(f'on {i}: {tableName}') 
            
            #set filepath
     
            uuid = hashlib.shake_256(f'{tableName}_{min_size}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
            ofp = os.path.join(out_dir,f'{tableName}_{uuid}.geojson')
            
            if not os.path.exists(ofp):
     
                dx = _calc_grid_country(tableName, log, min_size)
                dx.to_pickle(ofp)
                
                log.info(f'wrote {dx.shape} to \n    {ofp}')
                
            else:
                log.info(f'    filepath exists... skipping')
                
            res_d[i] = ofp
            
    #===========================================================================
    # PARALLEL
    #===========================================================================
    else:
        res_d = dict()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_futures_buid_grid, grid_size, country_key, i, min_size, out_dir) 
                       for i, (grid_size, country_key) in enumerate(product([int(e) for e in grid_size_l], country_l))]
        
        for future in concurrent.futures.as_completed(futures):
            i, ofp = future.result()
            res_d[i] = ofp
        
        
        
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {len(res_d)}')
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
            
def _calc_grid_country(tableName, log, min_size):
    """calc pdist for a single table"""
    
    log=log.getChild(tableName)
    start = datetime.now()
    #===========================================================================
    # """slow... loads everything into memory.
    #     alternatively, could use 'chunksize'
    #         cant control how it splits        
    #     gdf = _post_to_gpd('inters_agg', tableName, conn_d=conn_d)"""
    #===========================================================================
    #get iterater of grid indexes
    ij_dx = pd.DataFrame(
        pg_exe(f"""SELECT i, j, COUNT(*) FROM inters_agg.{tableName} GROUP BY i, j""", return_fetch=True), 
        columns=['i', 'j', 'count']).sort_values(['i', 'j'], ignore_index=True)
        
    ij_dx.index.name='gid'
    ij_dx = ij_dx.set_index(['i', 'j'], append=True)
        
    log.info(f'queried {len(ij_dx)} grids on {tableName}')
    
    #=======================================================================
    # #loop on each 'i' group
    #=======================================================================
    """to reduce i/o calls... pulling a column at a time"""
    res_d = dict()
    cnt=0
    bx = ij_dx['count'] >min_size
    for i, ij_gdx0 in ij_dx[bx].groupby('i'):
        
         
        #load group to geop[andas
        pts_dx = _get_i_group(tableName, i).set_index(
            ['country_key', 'gid', 'id', 'grid_size', 'i', 'j'])
        
        """pts_df.columns"""
        
        #group on each cell
 
        log.info(f'computing i={i} ({len(pts_dx)}) on {len(haz_coln_l)} columns')
        for j, ij_gdx1 in ij_gdx0.groupby('j'):
            
            #get the slice            
            keys_d=ij_gdx1.index.to_frame(index=False).to_dict('records')[0]
            
            pts_gdx = pts_dx.xs(j, level='j')     
            
            keys_d['count']=len(pts_gdx)
 
        
            #compute each haz            
            d = dict()
            for haz_coln, ser in pts_gdx.items():
                log.debug(f'{i}.{haz_coln}')
                d[haz_coln] = _get_agg_stats_on_ser(ser)
                
            #clean up
            res_df = pd.DataFrame.from_dict(d).stack().swaplevel().sort_index().rename('val').to_frame()
            res_df.index.set_names(['haz', 'metric'], inplace=True)
            
            #add the indexers
            for k1,v1 in keys_d.items():
                res_df[k1]=v1                
            res_df.set_index(list(keys_d.keys()), append=True, inplace=True)
            
            #store
            res_d[keys_d['gid']] = res_df 
            cnt+=1
            
        #=======================================================================
        # if cnt>3:
        #     break
        #=======================================================================
            
    #===========================================================================
    # wrap
    #===========================================================================
    res_dx = pd.concat(res_d.values()).reorder_levels(['gid', 'i', 'j', 'haz', 'count','metric']).sort_index(sort_remaining=True)
    
    log.info(f'finished {tableName} w/ {len(res_dx)} in {(datetime.now()-start).total_seconds():.2f}secs')
    
    return res_dx
 
        

def plot_exponential_dist(params, data):
    import matplotlib.pyplot as plt
# Plot the raw histogram and the fitted distribution
    _ = plt.hist(data, bins=20, density=True, alpha=0.6, color='g')
    ax = plt.gca()
    ax.set_xlim(0, data.max())
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = expon.pdf(x, *params)
    plt.plot(x, p, 'k', linewidth=2)
    title = f"Fit results: loc = {params[0]:.2f}, scale={params[1]:.2f}"  
    plt.title(title)
    plt.show()

def _get_agg_stats_on_ser(ser):
    """compoute stats on single group series"""
    d = dict()
    #d['cnt'] = len(ser) #gather this higher
    d['zero_cnt'] = (ser==0.0).sum()
    d['null_cnt'] = ser.isna().sum()
    
    ar = ser.dropna().values
    
    """
    ser.hist()
    
    import matplotlib.pyplot as plt
    plt.show()
    ser.value_counts(dropna=False)
    """
 
    
    # Fit an exponential distribution to the data
    if np.all(ar==0):
        d['loc'], d['scale'] = 0,0
    else:
        
        params = expon.fit(ar)
        
        d['loc'], d['scale'] = params[0], params[1]
        d['mean'], d['var'], d['skew'], d['kurt'] = expon.stats(*params, moments='mvsk')
        
        """
        print(d)         
        plot_exponential_dist(params, ar)
        """
    
    return d
    
 

    
    
            
            
            
             

def _get_i_group(tableName, i, conn_d=postgres_d):
    """load a filtered table to geopanbdas"""
    
    with psycopg2.connect(get_conn_str(conn_d)) as conn:
        #set engine for geopandas
        engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
        return pd.read_sql_query(f"""
                SELECT country_key, gid, id, f010_fluvial, f050_fluvial, f100_fluvial, f500_fluvial, grid_size, i, j 
                    FROM {schema}.{tableName}
                    WHERE i={i}""", engine)
        
        #=======================================================================
        # return gpd.read_postgis(f"""
        #         SELECT country_key, gid, id, f010_fluvial, f050_fluvial, f100_fluvial, f500_fluvial, grid_size, i, j 
        #             FROM {schema}.{tableName}
        #             WHERE i={i}""", engine)
        #=======================================================================
              
def _post_to_gpd(schema, tableName, conn_d=postgres_d): 
    """load a table into geopandas"""                                           
    with psycopg2.connect(get_conn_str(conn_d)) as conn:        
 
        
        #set engine for geopandas
        engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
        return gpd.read_postgis(f"SELECT * FROM {schema}.{tableName}", engine)
    


if __name__ == '__main__':
    run_build_pdist(max_workers=4)
    
    
    
    
    
    
    
    
    
    
    
    
    