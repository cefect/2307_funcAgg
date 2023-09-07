'''
Created on Sep. 6, 2023

@author: cefect

calculate function aggregation error potential metrics
'''

import os, hashlib, sys, subprocess, psutil
from datetime import datetime

import numpy as np
import pandas as pd

from tqdm import tqdm

from coms import (
    init_log, today_str, get_log_stream, get_directory_size,
    dstr, view, get_conn_str,  
    )

from funcMetrics.coms_fm import slice_serx
 

from definitions import wrk_dir, dfunc_pkl_fp

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
    for df_id, gserx in tqdm(serx.groupby('df_id')):
        
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
    
 
 
 
            
    

def compute_gradient(
        serx,
        
        log
        ):
    """calculate the gradient of the functions"""
 
    log = log.getChild('gradient')
    log.info(f'on {len(serx.index.unique(1))} funcs')
    
 
    
    #===========================================================================
    # loop and compute
    #===========================================================================
    res_d = dict()
    for df_id, gserx in tqdm(serx.groupby('df_id')):
        #prep
        log.debug(f'df_id={df_id} w/ {len(gserx)}')
        ar = gserx.droplevel([0,1]).reset_index().T.values
        
 
        
        #compute the first derivative
        """
        The np.gradient function calculates the gradient of an N-dimensional array. The gradient is a measure of how the values of the array change, and it is calculated by taking the differences between adjacent elements along each axis of the array. In this case, the input array f is one-dimensional, so the gradient is calculated along the only axis.

        The gradient at each point is calculated using central differences, which means that the value at each point is calculated as the average of the differences between that point and its two neighbors. For example, for the second element in the array (index 1), the gradient is calculated as (f[2] - f[0]) / 2, which gives (4 - 1) / 2 = 1.5. This value represents the slope of a line that passes through points (0, f[0]) and (2, f[2]).
        """
        
        d = {'deriv1':np.gradient(ar[1],ar[0],edge_order=1 )}        
        
        d['deriv2'] = np.gradient(d['deriv1'],ar[0],edge_order=1 )
 
        #collect 
        res_d[df_id] = pd.concat([gserx]+[pd.Series(v, index=gserx.index, name=k) for k,v in d.items()], axis=1)
        
    #===========================================================================
    # collect
    #===========================================================================
    log.info(f'finished w/ {len(res_d)}')
    res_dx = pd.concat(res_d.values(), axis=0).astype(np.float16)
    
    return res_dx
 
    
    
    
    

def run_gradient(
        fp = dfunc_pkl_fp,
        out_dir=None,
        max_depth=None,
        ):
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()    
 
 
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'funcs', '01_deriv')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    
    log = init_log(name=f'deriv', fp=os.path.join(out_dir, today_str+'.log'))
    
    if max_depth is None:
        from funcMetrics.coms_fm import max_depth
    
    #===========================================================================
    # load
    #===========================================================================
    log.info(f'loading from \n    {fp}')
    serx_raw = pd.read_pickle(fp)
    
    
    
    #===========================================================================
    # extend
    #===========================================================================
    """using full index as we are changing the index (not just adding values"""
    serx_extend = force_max_depth(serx_raw, max_depth, log)
    
    
    #===========================================================================
    # compute
    #===========================================================================
 
    res_dx =  compute_gradient(slice_serx(serx_extend, xs_d=None), log)
    
    #add back the big index
    res_dx.index = serx_extend.index
    
    """
    view(res_dx)
    """
    
    #===========================================================================
    # #write
    #===========================================================================
    ofp = os.path.join(out_dir, f'derivs_{len(res_dx)}_{today_str}.pkl')
    res_dx.to_pickle(ofp)
    
    log.info(f'wrote {str(res_dx.shape)} to \n    {ofp}')
    
    return ofp
    
    
    
if __name__=='__main__':
    


    run_gradient()

    
 
    
    print('finished ')