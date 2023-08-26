'''
Created on Aug. 26, 2023

@author: cefect

collecting results of intersect routine
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

 




def get_fpserx(srch_dir, log):
    """load structured files into a filepath Multindex series"""
    
    def dict_to_series(d):
        data = []
        for key, values in d.items():
            for i, value in enumerate(values):
                data.append((key, i, value))
        return pd.Series(data=[x[2] for x in data], index=pd.MultiIndex.from_tuples([(x[0], x[1]) for x in data]))
    
    fp_lib = dict()
    for country_key, country_dir in {e:os.path.join(srch_dir, e) for e in os.listdir(srch_dir)}.items():
        #loop through each hazard key
        fp_d = dict()
        for haz_key, haz_dir in {e:os.path.join(country_dir, e) for e in os.listdir(country_dir)}.items():
            log.debug(f'{country_key}.{haz_key}')
            
            #get indexers
            d=dict()
            for fn, fp in {e:os.path.join(haz_dir, e) for e in os.listdir(haz_dir) if e.endswith('.geojson')}.items():
                d[int(fn.replace(country_key+'_', '').replace(haz_key+'_', '').split('_')[0])] = fp
            fp_d[haz_key] = d
        
        #concat into serx
        l=[pd.Series(e, name=k) for k,e in fp_d.items()] 
        fp_lib[country_key] = pd.concat(l, keys=[s.name for s in l], names=['haz_key', 'id']).sort_index()
    
#wrap
    fpserx = pd.concat(fp_lib, names=['country_key'])
    log.info(f'got {len(fpserx)} entries')
    return fpserx

def _get_gdf(id, fp):
    return id, gpd.read_file(fp, 
                             #ignore_fields=['geometry'],
                             ignore_geometry=True,
                             )


def _load_samps_set(gserx, max_workers):
    """load geodataframe from each file in the gserx"""
    if max_workers is None:
        d = dict()
        for id, fp in tqdm(gserx.droplevel([0, 1]).to_dict().items()):
            _, d[id] = _get_gdf(id, fp)
    
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(_get_gdf, *
                        zip(*gserx.droplevel([0, 1]).to_dict().items())), total=len(gserx.droplevel([0, 1]).to_dict().items())))
        d = dict(results)
    return pd.concat(d, names=['gid', 'fid']).sort_index()

def run_collect_sims(
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
        srch_dir = os.path.join(wrk_dir, 'outs', 'inters', '01_sample')
    assert os.path.exists(srch_dir)
 
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'inters', '02_collect')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    #===========================================================================
    # if temp_dir is None:
    #     temp_dir = os.path.join(temp_dirM, 'collect')    
    # if not os.path.exists(temp_dir):os.makedirs(temp_dir)
    #===========================================================================
    
    log = init_log(name=f'collect', fp=os.path.join(out_dir, today_str+'.log'))
    log.info(f'on \n    {srch_dir}')
    
    #===========================================================================
    # collect files
    #===========================================================================
    fpserx = get_fpserx(srch_dir, log)
    
    #===========================================================================
    # loop and load per sim (concat1)
    #===========================================================================
    """here we concat all of the points for the country
    each haz_key should have redundant geometry info
    for some countries, this points file becomes quite large
    for fancier analysis (e.g., anything spatial) should use PostGIS
    
    just extracting the values for  now"""
    
    ofp_d = {k:dict() for k in fpserx.index.unique('country_key')}
    for i, ((country_key, haz_key), gserx) in enumerate(fpserx.groupby(['country_key', 'haz_key'])):
        log.info(f'{i+1} on {country_key}.{haz_key} w/ {len(gserx)}')
        
        #get record
        uuid = hashlib.shake_256(f'{country_key}_{haz_key}_{srch_dir}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
        
        odi = os.path.join(out_dir, country_key)
        if not os.path.exists(odi):os.makedirs(odi) 
        ofp_i = os.path.join(odi, f'{country_key}_{haz_key}_{uuid}.pkl')
        
        if not os.path.exists(ofp_i):
            dxind = _load_samps_set(gserx, max_workers)
            
            log.info(f'writing {str(dxind.shape)} to \n    {ofp_i}')
            dxind.to_pickle(ofp_i)
            
        else:
            log.info(f'record exists... skipping')
 
            
        ofp_d[country_key][haz_key] = ofp_i
 
 
 
        
    #wrap
    log.info(f'finished on {len(ofp_d)}')
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
 
        
 
 
    
    



if __name__ == '__main__':
    run_collect_sims(max_workers=8)
    
    
    
    
    
    
    
    