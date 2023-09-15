'''
Created on Sep. 15, 2023

@author: cefect
'''

import os, hashlib
import pandas as pd
import numpy as np
idx = pd.IndexSlice
from datetime import datetime

from coms import _get_filepaths
from definitions import wrk_dir, haz_label_d, temp_dir


def load_pdist_concat(
        search_dir=r'l:\10_IO\2307_funcAgg\outs\expo_stats\pdist',
        infer_keys=False, #temporary because I forgot to add the indexers
        ):
    
    """load pdist results and concat
    
    NOTE: probably need to switch to dask once we add the other countries"""
    
    #===========================================================================
    # retrieve filepaths
    #===========================================================================
    fp_l = _get_filepaths(search_dir)
    print(f'got {len(fp_l)} from \n    {search_dir}')
    
    #===========================================================================
    # get cache filepath
    #===========================================================================    
    out_dir = os.path.join(temp_dir, 'pdist', 'load_pdist_concat')
    if not os.path.exists(out_dir):os.makedirs(out_dir)

    
    uuid = hashlib.shake_256('_'.join(fp_l).encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir,f'pdist_{len(fp_l)}_{uuid}.pkl')
    
    
    
    #===========================================================================
    # build
    #===========================================================================
    if not os.path.exists(ofp):
        df_d = {os.path.basename(fp):pd.read_pickle(fp) for fp in fp_l}
        
        print(f'loading. infer_keys={infer_keys}')
        if infer_keys:
     
            for i, (fn,dx) in enumerate(df_d.items()):
                l = fn.split('_')
                
                d = {'country_key':l[3], 'grid_size': int(l[4])}
                
                for k,v in d.items():
                    dx[k]=v
                    
                dx = dx.set_index(list(d.keys()), append=True)
                
                dx.index = dx.index.reorder_levels(['country_key', 'grid_size','gid', 'i', 'j', 'haz', 'count', 'metric', ])
                
                df_d[fn]=dx.sort_index(sort_remaining=True)
      
        dx = pd.concat(df_d.values()).sort_index(sort_remaining=True)
        
        dx.to_pickle(ofp)
        print(f'wrote {dx.shape} to \n    {ofp}')
        
    else:
        print(f'loading cached from \n    {ofp}')
        dx = pd.read_pickle(ofp)
    
    print(f'finished w/ {len(dx)}')
    
    return dx
    

    




def _resample_df(df_raw, n=2):
    
    df1 = df_raw.T
    
    df1['group'] = df1.reset_index().index//n
    
    df2 = df1.reset_index().set_index('group')
    
    df3 = df2.groupby('group').mean().reset_index(drop=True).set_index(df_raw.columns.name).dropna(axis=0, how='all')
    df3.index = df3.index - df3.index[0] #shfit back to heads
    
    #fix end
    df3.loc[df1.index[-1]] = np.nan
    
    res_df = df3.T
    
    assert len(res_df)==len(df_raw)
    assert len(res_df.columns) == len(df_raw.columns)//n+1
    
    return res_df

def _resample_ser(ser, n=2):
    """resample data through averaging"""
    df = ser.to_frame().reset_index()
    
    #add the group
    df['group'] = df.index//n
    
    #aggregate
    rserx = df.groupby('group').mean().set_index(ser.index.name).iloc[:,0].rename(ser.name).dropna()
    rserx.index = rserx.index - rserx.index[0] #shfit back to heads
    
    #fix end
    rserx.loc[ser.index[-1]] = ser.iloc[-1]
    
    assert len(rserx) == len(ser)//2+1
    
    return rserx
    

 
    