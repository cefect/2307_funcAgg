'''
Created on Sep. 9, 2023

@author: cefect


fitting probability distributions to the aggregated -intersect datasetes
'''
 
#===============================================================================
# IMPORTS-----
#===============================================================================
import os, hashlib, sys, subprocess, psutil, warnings
from datetime import datetime
from itertools import product
import concurrent.futures

import psycopg2
#print('psycopg2.__version__=' + psycopg2.__version__)

from sqlalchemy import create_engine, URL

from tqdm import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd
idx = pd.IndexSlice


import matplotlib.pyplot as plt

from scipy.stats import expon


from coms import (
    init_log, today_str, get_directory_size,dstr, view,  get_log_stream
    ) 

from _02agg.coms_agg import get_conn_str, pg_exe

from definitions import (
    index_country_fp_d, wrk_dir, postgres_d, equal_area_epsg, postgres_dir, gridsize_default_l,
    haz_coln_l, temp_dir
    )

schema='inters_agg'





def run_build_pdist(
        country_l = ['bgd'],
        grid_size_l=None,
        
        conn_d=postgres_d,
        out_dir=None,
        min_size=5,
        #nonzero_frac_thresh=0.99,
        max_workers=None,
        
        debug_len=None,
        use_icache=True,
 
        ):
    """fit pdist to each clump
    
    
    Pars
    ---------
    min_size: int
        minimum clump size to consider
        
    nonzero_frac_thresh: float
        minimum non-zero fraction of clump to accept
        using this to exclude edge/fringe clumps
        
    use_icache: bool
        whether to use the cached pickles for the 'i' groups (from tmp_dir)
        
    Returns
    ----------
    pd.DataFrame write to .pkl for each grid_size and country_key
    """
    
    raise IOError(f"""this was run on an old table... not sure I trust it.
            should use expo.grid_bldg_stats_{country_key}_{grid_size:04d}
            and join to inters.{country_key} 

        """)
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
    meta_lib=dict()
    res_d = dict()
 
    for en_i, (grid_size, country_key) in enumerate(product([int(e) for e in grid_size_l], country_l)):
        tableName=f'pts_osm_fathom_{country_key}_{grid_size:07d}'
        start_i=datetime.now() 
        
        log.info(f'on {en_i}: {tableName}')
        
        #set filepath
        try:
            uuid = hashlib.shake_256(f'{tableName}_{min_size}_{debug_len}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
            ofp = os.path.join(out_dir,f'{tableName}_{uuid}.pkl')
            
             
            
            if (not os.path.exists(ofp)) or (not debug_len is None):
                
                dx = _calc_grid_country(tableName, log, min_size, max_workers, 
                                        debug_len=debug_len, out_dir=out_dir, use_icache=use_icache)
                
                
                #append indexers 
                for k,v in {'grid_size':grid_size, 'country_key':country_key}.items():
                    dx[k]=v
                dx = dx.set_index(['grid_size', 'country_key'], append=True).reorder_levels(
                    ['grid_size', 'country_key','gid', 'i', 'j', 'haz']).sort_index(sort_remaining=True)
                
 
                #write
                if debug_len is None:
                    dx.to_pickle(ofp)
                
                    log.info(f'wrote {dx.shape} to \n    {ofp}')
                
            else:
                log.info(f'    filepath exists... skipping\n    {ofp}')
                dx = pd.read_pickle(ofp)
                
            
            
            #get some meta
            #cnt_ar = dx.index.get_level_values('count').values
            mdx=dx.xs('metric', level=0, axis=1)
            """
            view(mdx)
            view(dx)
            """
            meta_lib[en_i] = {
                'len':len(dx), 'ofn':os.path.basename(ofp), 'grid_size':grid_size, 'country_key':country_key,
                'count_max':mdx['count'].max(), 'count_mean':mdx['count'].mean(),
                'max':mdx['max'].max(), 'min':mdx['min'].min(),
                'wet_cnt':mdx['wet_cnt'].sum(), 'zero_cnt':mdx['zero_cnt'].sum(),
 
                #'output_MB':os.path.getsize(ofp)/(1024**2),
                'max_workers':max_workers,'debug_len':debug_len,
                'tdelta_secs':(datetime.now()-start_i).total_seconds(),
                'now':datetime.now(),
                }
            
            #store
            res_d[en_i] = ofp
 
        except Exception as e:
            log.error(f'failed on  {en_i}: {tableName} w/ \n    {e}')
            meta_lib[en_i]={
                'error':str(e),
                'tdelta_secs':(datetime.now()-start).total_seconds(),
                'now':datetime.now(),
                }
        
 
            
 
        
 
        
    #===========================================================================
    # write meta
    #===========================================================================
    
    meta_df = pd.DataFrame.from_dict(meta_lib).T
    
    ofp = os.path.join(out_dir, f'meta_{len(meta_df)}_{today_str}.csv')
    meta_df.to_csv(ofp, index=False)
    log.info(f'wrote meta {meta_df.shape} to \n    {ofp}')
        
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
    
    
def _calc_grid_country(tableName, log, min_size, max_workers, debug_len=None, out_dir=None, use_icache=True):
    """calc pdist for a single table"""
    
    raise IOError('this probably doesnt work anymore as inters_agg.{tableName} has changed')
    
    log=log.getChild(tableName)
    start = datetime.now()
    #===========================================================================
    # """slow... loads everything into memory.
    #     alternatively, could use 'chunksize'
    #         cant control how it splits        
    #     gdf = _post_to_gpd('inters_agg', tableName, conn_d=conn_d)"""
    #===========================================================================
    #===========================================================================
    # #get iterater of grid indexes
    #===========================================================================
    ij_dx = pd.DataFrame(
        pg_exe(f"""SELECT i, j, COUNT(*) FROM inters_agg.{tableName} GROUP BY i, j""", return_fetch=True), 
        columns=['i', 'j', 'count']).sort_values(['i', 'j'], ignore_index=True)
        
    ij_dx.index.name='gid'
    ij_dx = ij_dx.set_index(['i', 'j'], append=True)
        
    log.debug(f'queried {len(ij_dx)} grids on {tableName}')
    
    #===========================================================================
    # #slice
    #===========================================================================
    bx = ij_dx['count'] >min_size    
    
    ij_dx_sel = ij_dx.loc[bx, :]
    
    #slice for debugging
    if not debug_len is None:
        assert __debug__
        log.warning(f'trimming to {debug_len} for debugging')
        ij_dx_sel = ij_dx_sel.iloc[:4,:]
    
    
    
    log.info(f'computing on {len(ij_dx_sel)}/{len(ij_dx)} w/ count>{min_size}')
    
    #=======================================================================
    # #loop on each 'i' group
    #=======================================================================
    """to reduce i/o calls... pulling a column of i cells at a time"""
    res_d = dict()

    if max_workers is None:
        for i, ij_gdx0 in ij_dx_sel.groupby('i'):
 
            res_d[i] = _get_stats_igroup(tableName, log, i, ij_gdx0, out_dir=out_dir, use_icache=use_icache)
                
 
            
    #===========================================================================
    # PARALLEL
    #===========================================================================
    else: 
        log.info(f'concurrent.futures.ProcessPoolExecutor(max_workers={max_workers})')
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker_get_stats_igroup, i, ij_gdx0, tableName,out_dir, use_icache): i for i, ij_gdx0 in ij_dx_sel.groupby('i')}
            
            for future in concurrent.futures.as_completed(futures):
                i, res_dx = future.result()
                res_d[i] = res_dx
 
            
    #===========================================================================
    # wrap
    #===========================================================================
    res_dx = pd.concat(res_d.values()).reorder_levels(['gid', 'i', 'j', 'haz']).sort_index(sort_remaining=True)
    
    log.info(f'finishedw/ {len(res_dx)} in {(datetime.now()-start).total_seconds():.2f} secs')
    
    return res_dx
 
def _worker_get_stats_igroup(i, ij_gdx0, tableName, out_dir, use_icache):
    log = get_log_stream(name=f'{tableName}_{str(os.getpid())}')
    res_dx = _get_stats_igroup(tableName, log, i, ij_gdx0, out_dir=out_dir, use_icache=use_icache)
    return i, res_dx
            


def _calc_stats_igroup(ij_gdx0, pts_dx, log, **kwargs):
    """calc the stats for this group of i grids (by iterating over j)"""
    
    #===========================================================================
    # setup
    #===========================================================================

    res_d = dict()
    cnt = 0
    
    #===========================================================================
    # loop and compute each gruop
    #===========================================================================
    """should probably parallelize here instead"""
    for j, ij_gdx1 in ij_gdx0.groupby('j'):
        cnt += 1
        #get the slice
        keys_d = ij_gdx1.index.to_frame(index=False).to_dict('records')[0]
        pts_gdx = pts_dx.xs(j, level='j')
        keys_d['count'] = len(pts_gdx)
        #===================================================================
        # if plot_only:
        #     _write_hist(pts_gdx, keys_d, out_dir=out_dir)
        #
        #     res_d[j]=pd.DataFrame()
        # else:
        #===================================================================
        #compute each haz
        d, hist_d = dict(), dict()
        for haz_coln, ser in pts_gdx.items():
            log.debug(f'{haz_coln}')
            d[haz_coln], hist_d[haz_coln] = _get_agg_stats_on_ser(ser)
        
        #clean up
        res_df = pd.DataFrame.from_dict(d).stack().swaplevel().sort_index().rename('val').to_frame()
        res_df.index.set_names(['haz', 'metric'], inplace=True)
        #add the indexers
        for k1, v1 in keys_d.items():
            res_df[k1] = v1
        
        res_df.set_index(list(keys_d.keys()), append=True, inplace=True)
        #add hist
        res_df1 = res_df.unstack('metric').droplevel(0, axis=1).reset_index(level='count')
        res_df1.columns.name = None
        #join to same index
        hist_df = pd.DataFrame.from_dict(hist_d).T
        hist_df = res_df1.index.to_frame().reset_index(drop=True).join(hist_df, on='haz').set_index(res_df1.index.names)
        #merge
        res_d[j] = pd.concat({'metric':res_df1, 'hist':hist_df}, axis=1)
    
        #wrap j loop
    #===========================================================================
    # wrap
    #===========================================================================
    res_dx = pd.concat(res_d.values())
    return res_dx

def _get_stats_igroup(tableName, log, i, ij_gdx0, use_icache=True, **kwargs):
    """compute stats for an 'i' group of grids"""
    
    #get write info
    uuid = hashlib.shake_256(f'{tableName}_{ij_gdx0}_{i}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    odi = os.path.join(temp_dir, 'pdist', '_get_stats_igroup', tableName)
    if not os.path.exists(odi): os.makedirs(odi)
    
    ofp = os.path.join(odi,f'{tableName}_{i:07d}_{uuid}.pkl')
    
    #===========================================================================
    # build it
    #===========================================================================
    if (not os.path.exists(ofp)) or (not use_icache):
            
        #load group to geop[andas
        pts_dx = _sql_to_df_igroup(tableName, i).set_index(
            ['country_key', 'gid', 'id', 'grid_size', 'i', 'j'])
        """pts_df.columns"""
        
        #group on each cell
        jvals = ij_gdx0.index.unique('j')
        log.info(f'    computing i={i} w/ {len(ij_gdx0)} j vals ({len(pts_dx)}) on {len(haz_coln_l)} columns')
        res_dx = _calc_stats_igroup(ij_gdx0, pts_dx, log, **kwargs)
        
        log.debug(f'writing {res_dx.shape} to \n    {ofp}')
        res_dx.to_pickle(ofp)
        
    #===========================================================================
    # load it
    #===========================================================================
    else:
        assert use_icache
        log.debug(f'i={i} exists... loading from {ofp}')
        res_dx = pd.read_pickle(ofp)
        
            
            
    
    return res_dx

def _get_agg_stats_on_ser(ser):
    """compoute stats on single group series"""
    d = dict()
    #d['cnt'] = len(ser) #gather this higher
    d['zero_cnt'] = (ser==0.0).sum()
    d['null_cnt'] = ser.isna().sum()
    d['wet_cnt'] = len(ser)-( d['zero_cnt']+d['null_cnt'])
    
    
    
    ar = ser.dropna().values
    
    bins = np.linspace(0, 500, 21 )  # Creates 10 bins between 0 and 100
    
    if len(ar)>2:
        d['mean'] = np.mean(ar)
        d['std'] = np.std(ar)
        d['min']=np.min(ar)
        d['max']=np.max(ar)
        
        """
        ser.hist()
        
        import matplotlib.pyplot as plt
        plt.show()
        ser.value_counts(dropna=False)
        """
        
        #===========================================================================
        # compute histogram
        #===========================================================================
        #ar = np.array([0., 0.,   0., 0., 0.,0.])
        
        
        """this will often throw the following warning
        this is safe to ignore as we are not generating the histgraoms for plotting, but for comparing
            numpy\lib\histograms.py:885: RuntimeWarning: invalid value encountered in divide  return n/db/n.sum(), bin_edges
            
        """
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            hist, bin_edges = np.histogram(ar, bins, density=True)
            
    else:
        hist, bin_edges = np.histogram(np.full(1, 0.0), bins, density=True)
        
        hist = np.full(len(hist), np.nan)
 
         
 
    
    #===========================================================================
    # # Fit an exponential distribution to the data
    # """"NO! the data is not exponentially distributed"""
    # if np.all(ar==0):
    #     d['loc'], d['scale'] = 0,0
    # else:
    #     
    #     params = expon.fit(ar)
    #     
    #     d['loc'], d['scale'] = params[0], params[1]
    #     d['mean'], d['var'], d['skew'], d['kurt'] = expon.stats(*params, moments='mvsk')
    #===========================================================================
        
 
    
    return d, pd.Series(np.append(hist, np.nan), index=bin_edges, name='hist')
    
 


def _write_hist(dx, keys_d, out_dir=None, maxd=500):
    """write a plot"""
    
    df = dx.reset_index(drop=True)
    
    if np.all(df==0) or len(df)<5:
        return
    
    #setup plot
    plt.close('all')
    #fig, ax = plt.subplots()
 
    #set up data
    mdex = dx.index
    
    lab_d = {k:mdex.unique(k)[0] for k in ['country_key', 'grid_size', 'gid']}
    
    
    #plot
    df.hist(sharey=True, sharex=True, bins=np.linspace(0, maxd, 20))
    
    
    fig = plt.gcf()
    
    ax = fig.gca()
    
    #text
    tstr= '\n'.join([f'{k}={v}' for k,v in lab_d.items()])
    
    tstr +=f'\ncnt={len(df)}'
    
    ax.text(0.95, 0.05, tstr, 
                            transform=ax.transAxes, va='bottom', ha='right', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
    #===========================================================================
    # post
    #===========================================================================

    
    ax.set_xlim(0, maxd)
    ax.set_xlabel('depth (cm)')
    
    uuid = hashlib.shake_256(f'{lab_d}_{keys_d}_{today_str}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    nstr = '_'.join([str(e) for e in lab_d.values()])
    ofp = os.path.join(out_dir, f'samp_hist_{nstr}_{today_str}_{uuid}.svg')
    fig.savefig(ofp, dpi = 300,   transparent=True)
    
    """
    
    plt.show()
    """
    
    
def plot_exponential_dist(params, data):
    """plotting an exponential distribution (for debugging"""
 
#     Plot the raw histogram and the fitted distribution
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


    
    
            
            
            
             

def _sql_to_df_igroup(tableName, i, conn_d=postgres_d):
    """load a filtered table to geopanbdas"""
    
    conn =  psycopg2.connect(get_conn_str(conn_d))
    #set engine for geopandas
    engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
    try:
        result = pd.read_sql_query(f"""
                SELECT country_key, gid, id, f010_fluvial, f050_fluvial, f100_fluvial, f500_fluvial, grid_size, i, j 
                    FROM {schema}.{tableName}
                    WHERE i={i}""", engine)
        
    except Exception as e:
        raise IOError(f'failed query w/ \n    {e}')
    finally:
        # Dispose the engine to close all connections
        engine.dispose()
        # Close the connection
        conn.close()
        

    return result
        
        #=======================================================================
        # return gpd.read_postgis(f"""
        #         SELECT country_key, gid, id, f010_fluvial, f050_fluvial, f100_fluvial, f500_fluvial, grid_size, i, j 
        #             FROM {schema}.{tableName}
        #             WHERE i={i}""", engine)
        #=======================================================================
              
#===============================================================================
# def _post_to_gpd(schema, tableName, conn_d=postgres_d): 
#     """load a table into geopandas"""                                           
#     with psycopg2.connect(get_conn_str(conn_d)) as conn:        
#  
#         
#         #set engine for geopandas
#         engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
#         return gpd.read_postgis(f"SELECT * FROM {schema}.{tableName}", engine)
#===============================================================================
    


if __name__ == '__main__':
    run_build_pdist(max_workers=None, 
                    grid_size_l=[1020],
                    debug_len=None,
                    #out_dir=r'l:\10_IO\2307_funcAgg\outs\expo_stats\pdist\plots',
                    use_icache=True,
                    )
    
    
    
    
    
    
    
    
    
    
    
    
    