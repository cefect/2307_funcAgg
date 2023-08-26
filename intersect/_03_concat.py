'''
Created on Aug. 26, 2023

@author: cefect

concat collected sims
'''
import os, hashlib, sys, subprocess
import psutil
from datetime import datetime

import pandas as pd
import geopandas as gpd

from tqdm import tqdm

import concurrent.futures

from hp import (
    init_log, today_str, get_log_stream, get_raster_point_samples, get_directory_size,
    dstr, view
    )

from definitions import wrk_dir, lib_dir
from definitions import temp_dir as temp_dirM

 



 


def _load_concat_write(fp_d, out_dir, log, country_key):
    df_l = [pd.read_pickle(fp) for fp in fp_d.values()]
    #concat
    dxind = pd.concat(df_l, axis=1)
    #write
    ofp_i = os.path.join(out_dir, f'samp_concat_{country_key}_{len(dxind)}.pkl')
    log.info(f'writing {str(dxind.shape)} to \n    {ofp_i}')
    dxind.to_pickle(ofp_i)
    return ofp_i

def run_concat_sims(
        srch_dir=None,
        out_dir=None,
        #temp_dir=None,
        max_workers=None,
        ):
    """
    collect and concat samples (drop geoemtry)
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    
    if srch_dir is None:
        srch_dir = os.path.join(wrk_dir, 'outs', 'inters', '02_collect')
    assert os.path.exists(srch_dir)
 
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'inters', '03_concat')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    
    log = init_log(name=f'concat', fp=os.path.join(out_dir, today_str+'.log'))
    log.info(f'on \n    {srch_dir}')
    
    #===========================================================================
    # collect files
    #===========================================================================
    #get country dirs
    country_dirs_d = {d:os.path.join(srch_dir, d) for d in os.listdir(srch_dir) if os.path.isdir(os.path.join(srch_dir, d))}
    
    assert len(country_dirs_d)>0, 'no subfolders'
    log.info(f'on {len(country_dirs_d)}')
    
    ofp_d, err_d=dict(), dict()
    for country_key, country_dir in country_dirs_d.items():
        log.info(f'concat {country_key}')        
        
        #get all the files (hazard keys)
        fp_d = {e:os.path.join(country_dir, e) for e in os.listdir(country_dir) if e.endswith('.pkl')}
        
        log.info(f'got {len(fp_d)} .pkl files...loading')
        
        #load
        try:
 
            ofp_d[country_key] = _load_concat_write(fp_d, out_dir, log, country_key)
        except Exception as e:
            log.error(f'failed to execute {country_key} w/ \n    {e}')
            err_d[country_key] = str(e)
 
        
    
 
    #===========================================================================
    # #write errors
    #===========================================================================
    if len(err_d)>0:
        log.error(f'writing {len(err_d)} errors to file')
        pd.Series(err_d).to_csv(os.path.join(out_dir, f'errors_{len(err_d)}_{today_str}.csv'))
        
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {len(ofp_d)}')
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    
    return ofp_d
 
        
 
 
    
    



if __name__ == '__main__':
    run_concat_sims(max_workers=8)
    
    
    
    
    
    
    
    