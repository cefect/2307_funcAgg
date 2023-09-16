'''
Created on Sep. 6, 2023

@author: cefect

calculate function aggregation error potential metrics
    trying different methods
'''
#===============================================================================
# IMPORTS-----------
#===============================================================================
import os, hashlib
from datetime import datetime

import numpy as np
import pandas as pd
idx = pd.IndexSlice

import scipy.integrate
#from tqdm import tqdm

from coms import (
    init_log, today_str, _get_filepaths, view
    )

from funcMetrics.coms_fm import slice_serx
from funcMetrics.func_prep import get_funcLib
from expo_stats.coms_exp import load_pdist_concat
 

from definitions import wrk_dir, dfunc_pkl_fp, temp_dir

import matplotlib.pyplot as plt

def force_max_depth(
        serx_raw,
        max_depth, log
        ):
    """add a maximum depth to each function"""
    log = log.getChild('maxDepth')
    log.info(f'on {serx_raw.shape} w/ max_depth={max_depth}')
    
    #===========================================================================
    # #add a flag to the index
    #===========================================================================
    dx = serx_raw.to_frame()
    dx['max_depth_forcing']=False
    serx = dx.set_index('max_depth_forcing', append=True).swaplevel().iloc[:,0]
 
    #===========================================================================
    # #add the max depth to each function group
    #===========================================================================
    d = dict()
    cnt=0
    for df_id, gserx in serx.groupby('df_id'):
        
        wd_vals = gserx.index.get_level_values('wd')
        
        d[df_id] =gserx.copy()
        
        #expand
        if max(wd_vals)<max_depth:
            new_index_vals = list(gserx.index[0])
            new_index_vals[-2]  =True #flag this one as forced
            new_index_vals[-1] = max_depth
            
            d[df_id].loc[tuple(new_index_vals)] = gserx.max()
            
            cnt+=1
            
    #===========================================================================
    # wrap
    #===========================================================================
    res_serx = pd.concat(d.values())
    
    assert len(res_serx) == len(serx) + cnt
    log.info(f'forced {cnt} max depths')
    
    return res_serx
 


def get_depth_weights(search_dir, log=None, min_wet_frac=0.05):
    """calculate the depth weights"""
    
    log = log.getChild('get_depth_weights')
    #===========================================================================
    # get cache filepath
    #=========================================================================== 
    fp_l = _get_filepaths(search_dir)   
    out_dir = os.path.join(temp_dir, 'funcMetrics', 'get_depth_weights')
    if not os.path.exists(out_dir):os.makedirs(out_dir)

    
    uuid = hashlib.shake_256((f'{min_wet_frac}'+'_'.join(fp_l)).encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir,f'dWeights_{len(fp_l)}_{uuid}.pkl')
    
    #===========================================================================
    # build
    #===========================================================================
    if not os.path.exists(ofp):
        
        #=======================================================================
        # load
        #=======================================================================
        log.info(f'loading from \n    {search_dir}')
        dx_raw = load_pdist_concat(search_dir=search_dir).droplevel(['i', 'j'])
        
        log.info(f'loaded {dx_raw.shape}')
        
        #=======================================================================
        # #extract
        #=======================================================================
        #split the data
        metric_df_raw = dx_raw.xs('metric', level=0, axis=1)
        metric_df_raw['wet_frac'] = metric_df_raw['wet_cnt']/metric_df_raw['count']
        """
        view(metric_df_raw)
        """
        
        #apply filter
        bx = metric_df_raw['wet_frac']>min_wet_frac
        
        if not bx.any():
            raise IOError('no valids')
            
        log.info(f'    selected {bx.sum()}/{len(bx)} w/ min_wet_frac={min_wet_frac}')
        
        #metric_df = gdx.xs('metric', level=0, axis=1)[bx]
        #=======================================================================
        # #get mean of grid cells
        #=======================================================================
        """leaving other levels for more transparency"""
        hist_dx = dx_raw[bx].xs('hist', level=0, axis=1).groupby(
            ['grid_size', 'country_key', 'haz']).mean().dropna(axis=1)
        
        hist_dx.columns.name='wsh'
        
        #=======================================================================
        # write
        #=======================================================================
        hist_dx.to_pickle(ofp)
        log.info(f'wrote {hist_dx.shape} to \n    {ofp}')
        
    #===========================================================================
    # load
    #===========================================================================
    else:
        log.info(f'loading from cache')
        hist_dx = pd.read_pickle(ofp)
        
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished w/ {hist_dx.shape}')
    
    return hist_dx
        
        
        
 
def run_depth_weighted_curvature(
        search_dir=r'l:\10_IO\2307_funcAgg\outs\expo_stats\pdist',
        fserx = None,
        #curves_fp=r'l:\10_IO\2307_funcAgg\outs\funcs\lib\dfunc_lib_17_359_20230915.pkl',
        hwd_scale=0.01,
        country_key='deu',
        #df_id=941,
        out_dir=None,
        max_depth=None,
        min_wet_frac=.9,
        ):
    """compute the weighted curvature for a single country and curve
    
    
    Params
    --------
    hwd_scale: float
        value to use to re-scale the wd values on the histogram (convert to m)
        
    """
    raise IOError('instead of wet_frac, use those where the grid centroid is wet')
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()    
 
    
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'funcs', '01_cWeight')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    
    log = init_log(name=f'cWeight', fp=os.path.join(out_dir, today_str+'.log'))
    
    if max_depth is None:
        from funcMetrics.coms_fm import max_depth
    
    #===========================================================================
    # load curves
    #===========================================================================
    if fserx is None: fserx = get_funcLib() #select functions
 
    #extend
    """using full index as we are changing the index (not just adding values"""
    fserx_extend = force_max_depth(fserx, max_depth, log).rename('rl')
    
    #===========================================================================
    # get depth weights
    #===========================================================================
    hist_dx =get_depth_weights(search_dir, log=log, min_wet_frac=min_wet_frac).xs(country_key, level='country_key')
    hist_dx.columns = hist_dx.columns.astype(float)*hwd_scale #convert to meters
    
    hg_keys = ['grid_size', 'haz']
    #===========================================================================
    # compute metric0
    #===========================================================================
    res_d, meta_lib = dict(), dict()
    for df_id, gserx in fserx_extend.groupby('df_id'):
        #prep inputs
        s = slice_serx(gserx, xs_d=None).droplevel(['df_id', 'model_id']).rename('rl')
        
        #compute on each depth histogram population
        res_l=list()
        meta_d=dict()
        for i, ((grid_size, haz), ghdx) in enumerate(hist_dx.groupby(hg_keys)):
            
            #get the area
            #rser = compute_weighted_curvature(s.copy(), ghdx.iloc[0,:], log)
            meta_d[i], rdf = compute_hist_weighted(s.copy(), ghdx.iloc[0,:], log)
            
            #add indexers        
            res_l.append(pd.concat({grid_size:pd.concat({haz:rdf}, names=['haz'])}, names=['grid_size']))            
            meta_d[i].update(dict(zip(hg_keys, (grid_size, haz))))
            
        #=======================================================================
        # #merge
        #=======================================================================
        res_d[df_id] = pd.concat(res_l)
        meta_lib[df_id] = pd.DataFrame.from_dict(meta_d, orient='index')
        
        """
        view(res_d[df_id])
        view(rdx)
        """
    
    #merge
    """leaving all the function metdata off... this could be re-joined later using df_id if necessary"""
    rdx = pd.concat(res_d, names=['df_id'])
    
    #===========================================================================
    # set(rserx.index.names).difference(serx_raw.index.names)
    # 
    # .reorder_levels(serx_raw.index.names + ['grid_size', 'max_depth_forcing', 'haz'])
    #===========================================================================
 
    #===========================================================================
    # #write
    #===========================================================================
    ofp = os.path.join(out_dir, f'cWeight_{len(rdx)}_{today_str}.pkl')
    rdx.to_pickle(ofp)
    
    log.info(f'wrote {len(rdx)} to \n    {ofp}')
    
    return ofp
    
#===============================================================================
# metric methods-------------
#===============================================================================
def compute_weighted_curvature(
        fser,
        hser,
        
        log,
 
        ):
    """try2 using the curvature weighted by typical exposure
    
    Params
    -----------
    fser: pd.Series
        depth damage function
        
    hser: pd.Sers
        histogram of computed exposures
        
    """
    raise IOError('link grid cent, have grouped histograms')
    log = log.getChild('wCurve')
    log.info(f'on {len(fser)} wd values')
    
    if not fser.index.name=='wd':
        raise IOError(fser.index.name)
    
    """no.. this messes up joining
    fser.index = fser.index.values.round(3).astype(float)"""
    
    #===========================================================================
    # compute the mean
    #===========================================================================
    #shift histogram to bucket centers
    hser.index = hser.index+np.diff(hser.index)[0]/2
    density_split = hser.sum()/2
    
    wd_mean  = np.interp(density_split, hser.values, hser.index.values,
                         right=hser.index[0], #mean falls within first bucket
                         )
    
    #===========================================================================
    # #get this on the function
    #===========================================================================
    get_rl = lambda x: np.interp(x, fser.index.values, fser.values)
    rl_mean = get_rl(wd_mean)
    
    log.info(f'got density_split={density_split} and wd_mean={wd_mean}')
    
    """
    hser.plot()
    import matplotlib.pyplot as plt
    plt.show()
    """
    #===========================================================================
    # loop and compute curvature for each discrete function value
    #===========================================================================
    wd_dom=fser.index.values
    #iterate over each value within the function
    res_d = dict()
    for x, y in fser.items():
        
        #=======================================================================
        # #build the straight component of the area
        #=======================================================================
        straight_line = pd.Series({x:y, wd_mean:rl_mean})
        
        #=======================================================================
        # build curved line
        #=======================================================================
        #slice the fser to all values within this
        if x<wd_mean:
            bx = np.logical_and(wd_dom>=x, wd_dom<wd_mean)
        else:
            bx = np.logical_and(wd_dom<=x, wd_dom>=wd_mean)
            
        assert bx.sum()>=1
        
        curved_line = fser[bx]
        
        #append mean points
        curved_line.loc[wd_mean] = rl_mean        
        curved_line = curved_line.sort_index()
 
        
        #=======================================================================
        # #calculate the area between the straight and curved line
        #=======================================================================
        curve_area = scipy.integrate.trapezoid(curved_line.values, x=curved_line.index)
        line_area = scipy.integrate.trapezoid(straight_line.values, x=straight_line.index)

        delta_area = abs(line_area - curve_area)
        
        #=======================================================================
        # weight
        #=======================================================================
        depth_weight = np.interp(x, hser.index.values, hser.values, left=0)
        
        res_d[x] = delta_area*depth_weight
        #=======================================================================
        # #plot
        #=======================================================================
        
        #=======================================================================
        # fig, ax = plt.subplots() 
        # 
        # 
        # ax.plot(fser, color='red', alpha=0.2, linewidth=0.5)
        # ax.plot(curved_line, color='red')
        # ax.plot(straight_line, color='black')
        # ax.plot(wd_mean, rl_mean, marker='x', color='purple')
        #=======================================================================
        

 
        
        """
        plt.close('all')
        plt.show()
        
        fser.plot()
        """
        
    #===========================================================================
    # wrap
    #===========================================================================
    
    rser = pd.Series(res_d, name='lc_area')
    rser.index.name = 'wd'
    
    return rser

def compute_hist_weighted(
        fser,
        hser_raw,
        
        log,
 
        ):
    """try3 calc from histogram them compare to mode
    
    Params
    -----------
    fser: pd.Series
        depth damage function
        
    hser: pd.Sers
        histogram of computed exposures
        
    """
    #===========================================================================
    # defulats
    #===========================================================================
    log = log.getChild('hWeight')
    log.info(f'on {len(fser)} wd values')
    
    if not fser.index.name=='wd':
        raise IOError(fser.index.name)
    
    #===========================================================================
    # prep
    #===========================================================================
    get_rl = lambda x: np.interp(x, fser.index.values, fser.values)
    
    #shift histogram to bucket centers
    hdf = hser_raw.copy().rename('p').to_frame()
    hdf.index = hdf.index+np.diff(hdf.index)[0]/2
 
    
    #===========================================================================
    # get fval at mode
    #===========================================================================
    meta_d = {'mode_wsh':hdf.idxmax().iloc[0]}    
    meta_d['mode_rl'] = get_rl(meta_d['mode_wsh'])
    
    #===========================================================================
    # get fval for hist
    #===========================================================================
    hdf['rl'] = [get_rl(x) for x in hdf.index]
    
    
    hdf['rl_weight'] = hdf['rl']*hdf['p']
    
    
    #(hdf['p']*(1.0/hdf['p'].sum())).sum()
    
    #re-scale so probabilities sum to 1
    hdf['rl_ev'] = hdf['rl_weight']*(1.0/hdf['p'].sum())
    
    #===========================================================================
    # compare
    #===========================================================================
    meta_d['rl_ev'] = hdf['rl_ev'].sum()
    meta_d['diff'] = meta_d['mode_rl'] - meta_d['rl_ev']
    
    meta_d['err_frac'] = abs(meta_d['diff'])/meta_d['rl_ev']
    
    log.info(f'finished w/ \n    {meta_d}')
    
    return meta_d, hdf
 
    
    
    
if __name__=='__main__':
    


    run_depth_weighted_curvature()

    
 
    
    print('finished ')