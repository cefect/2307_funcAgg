'''
Created on Sep. 6, 2023

@author: cefect

common functions for working with functions

trying to keep as much function maniuplation as high as possible
    so we don't bury this... more transpaent
'''

import numpy as np
import pandas as pd
from coms import (
    init_log, today_str, get_directory_size,dstr, view
    ) 



#===============================================================================
# VARS--------
#===============================================================================
max_depth=10.0

#===============================================================================
# HELPER FUNCS-------
#===============================================================================




def slice_serx(serx_raw,
               xs_d = {'sector_attribute':'residential'},
               keep_names_l = None,
               ):
    """do some typical slicing and cleaning of the function data"""
    if keep_names_l is None:
        keep_names_l = ['model_id','df_id', 'wd']
        
    #set the cross section
    serx = serx_raw
    if not xs_d is None:
        for lvlName, lvlVal in xs_d.items():
            serx = serx.xs(lvlVal, level=lvlName)
            keep_names_l.append(lvlName)
        
    #drop the levels
    drop_lvl_names = list(set(serx_raw.index.names).difference(keep_names_l))
    
    serx_s = serx.droplevel(drop_lvl_names)

    serx_s.index = serx_s.index.remove_unused_levels()
    return serx_s
    

 
def force_max_depth(
        serx_raw,
        max_depth, log
        ):
    """add a maximum depth to each function
    
    only really needed for plotting. np.interp does this"""
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
 

def force_zero_zero(
        serx_raw,log=None
        ):
    
    """ensure each function has a zero-zero entry
    
    
    see also max_depth"""
    
    log = log.getChild('zero')
    log.info(f'on {serx_raw.shape}')
    
    #===========================================================================
    # #add a flag to the index
    #===========================================================================
    dx = serx_raw.to_frame()
    dx['zero-zero']=False
    serx = dx.set_index('zero-zero', append=True).swaplevel().iloc[:,0]
    
 
    
    #===========================================================================
    # #add zero-zero
    #===========================================================================
    d = dict()
    cnt=0
    fcnt = len(serx.index.unique('df_id'))
    log.info(f'on {fcnt} funcs')
    for df_id, gserx in serx.groupby('df_id'):
        
        wd_vals = gserx.index.get_level_values('wd').values
        
        d[df_id] =gserx.copy()
        
        #expand
        if min(wd_vals)>0.0:
            new_index_vals = list(gserx.index[0]) #take indexers from first row
            new_index_vals[-2]  =True #flag this one as forced
            new_index_vals[-1] = 0.0
            
            #add the entry
            d[df_id].loc[tuple(new_index_vals)] = 0.0
            
            d[df_id].sort_index(level='wd', inplace=True)
            
            cnt+=1
            
    #===========================================================================
    # wrap
    #===========================================================================
    res_serx = pd.concat(d.values())
    
    assert len(res_serx) == len(serx) + cnt
    log.info(f'forced {cnt}/{fcnt} zero-zeros')
    
    return res_serx


def force_monotonic(
        serx_raw,log=None
        ):
    
    """drop non-monotonic values from functions
    
    not sure why some functions have this...
    
    
    see also max_depth"""
    
    log = log.getChild('mono')
    log.info(f'on {serx_raw.shape}')
    
    #===========================================================================
    # #add a flag to the index
    #===========================================================================
    #serx = serx_raw.copy()
    flagColn='non-monotonic'
    dx = serx_raw.to_frame()
    dx[flagColn]=False
    serx = dx.set_index(flagColn, append=True).swaplevel().iloc[:,0]
    
 
    
    #===========================================================================
    # #add zero-zero
    #===========================================================================
    d = dict()
    cnt=0
    fcnt = len(serx.index.unique('df_id'))
    log.info(f'on {fcnt} funcs')
    
    for df_id, gserx in serx.groupby('df_id'):
        
        #get dd_ar
        drop_lvls = list(range(gserx.index.nlevels-1))
        dd_ar = gserx.droplevel(drop_lvls).reset_index().T.values
        
        """
        dd_ar[1]
        """
        
        diff_bool = np.diff(dd_ar)>=0
        
 
        d[df_id] =gserx.copy()
        
        #expand
        if not np.all(diff_bool):
            
            
            #fix x-vals
            if np.any(diff_bool[0]):
                d[df_id].index = d[df_id].index.remove_unused_levels().set_levels(
                    np.maximum.accumulate(
                        d[df_id].index.get_level_values('wd').values
                        ),level='wd')
            #fix y-vals
            if np.any(diff_bool[1]):
                
                d[df_id] = pd.Series(
                    np.maximum.accumulate(d[df_id].values),
                    index=d[df_id].index,
                    name=d[df_id].name)
                
            #flag
            d[df_id].index = d[df_id].index.set_levels([True], level=flagColn) 
            
            cnt+=1
            
        assert d[df_id].shape==gserx.shape
            
    #===========================================================================
    # wrap
    #===========================================================================
    res_serx = pd.concat(d.values())
    
    assert len(res_serx) == len(serx) #+ cnt
    log.info(f'forced {cnt}/{fcnt} monotonic')
    
    return res_serx

def force_and_slice(fserx_raw, max_depth=10, log=None, **kwargs):
    
    fserx1 = slice_serx(fserx_raw, xs_d=None)
    
    fserx2 = force_zero_zero(fserx1,log=log)
    
    fserx3 = force_monotonic(fserx2,log=log)
    
    return force_max_depth(fserx3, max_depth, log=log)

  
