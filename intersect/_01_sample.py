'''
Created on Jul. 25, 2023

@author: cefect


intersecting building data with hazard rasters
'''
import os, hashlib, sys, subprocess

 
 
import psutil
from datetime import datetime
import pandas as pd
import numpy as np
import fiona
import geopandas as gpd
from osgeo import ogr
import rasterstats
from rasterstats import zonal_stats


from concurrent.futures import ProcessPoolExecutor

from intersect.osm import retrieve_osm_buildings

from definitions import (
    wrk_dir, lib_dir, index_country_fp_d, index_hazard_fp_d, temp_dir, equal_area_epsg,
    fathom_vals_d)
from definitions import temp_dir as temp_dirM

#whitebox
from whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()
wbt.set_compress_rasters(False) 
wbt.set_max_procs(1)
wbt.set_verbose_mode(True)


#===============================================================================
# print('\n\nPATH:\n')
# print('\n'.join(os.environ['PATH'].split(';')))
# print('\n\nPYTHONPATH:\n')
# print('\n'.join(os.environ['PYTHONPATH'].split(';')))
#===============================================================================
 
from coms import (
    init_log, today_str, get_log_stream,   get_directory_size,
    dstr, get_filepaths
    )



#===============================================================================
# BUILDINGS--------
#===============================================================================
def ogr_get_layer_names(fp):
    """Retrieve a list of layer names from a .geojson file"""
    ds = ogr.Open(fp)
    num_layers = ds.GetLayerCount()
    
    # Extract the layer names list
    layer_names_list = []
    for i in range(num_layers):
        layer = ds.GetLayerByIndex(i)
        layer_names_list.append(layer.GetName())
        
    # Close the file
    ds = None
    
    return layer_names_list
    
def ogr_export_geometry(fp, ofp, log=None):
    """use ogr2ogr to extract only the geometry from a file
    
    much faster than geojson"""
    log = log.getChild('ogr2ogr')
    
    #get the layer names
    log.debug(f'retriving layer name')
    layerName = ogr_get_layer_names(fp)[0]
 
    
    #geometry and centroid\
    cmd_str = f'''SELECT ST_Centroid(geometry) AS geometry, 
    ST_Area(ST_Transform(geometry, {equal_area_epsg})) AS area 
    FROM \'{layerName}\'
    WHERE ST_IsValid(geometry)'''
    
    args=['ogr2ogr', '-overwrite','-skipfailures', '-f', 'GeoJSON', '-dialect', 
          'SQLite', '-sql',cmd_str, '-select','area',ofp, fp]
    
    log.debug(f'subprocess on '+'\n    '.join(args))
    p = subprocess.run(args,stderr=sys.stderr, stdout=sys.stdout, check=True)   
    


    assert p.returncode==0  
    
    log.debug(f'finished on {ofp}')
    
    return ofp
    
def ogr_clean_polys(fp, ofp, log=None):
    """use ogr2ogr to clean and simplify the layer"""
    log = log.getChild('ogr2ogr')
 
    log.info(f'    cleaning polygons')
    args = ['ogr2ogr', '-f', 'GeoJSON', '-overwrite','-skipfailures','-simplify','0.00001',
                        '-nlt','POLYGON','-makevalid','-nln','osm_polys_clean',
                        '-select','geometry', ofp, fp]
    
 
    log.debug(f'subprocess on '+'\n    '.join(args))
    
    p = subprocess.run(args,stderr=sys.stderr, stdout=sys.stdout, check=True)   
    


    assert p.returncode==0  
    
    log.debug(f'finished on {ofp}')
    
    return ofp

def get_osm_bldg_cent(country_key, bounds, log=None,out_dir=None, 
                      pfx='', #{country_key}_{hazard_key}_{i}. NOTE: should remove hazard_key next time
                      use_cache=True, 
                      pre_clean_polys=False,
                      manual_l=[
                          'DEU_500_fluvial_8'
                          ],
                      ):
    """intelligent retrival of building centroids"""
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    if log is None: log = get_log_stream()
    if out_dir is None: out_dir=os.path.join(lib_dir,'bldg_cent', country_key)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    log = log.getChild('get_osm')
    
    #===========================================================================
    # #get record
    #===========================================================================
    if not pfx in manual_l:
        uuid = hashlib.shake_256(f'{country_key}_{bounds}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)    
        ofp = os.path.join(out_dir, f'{pfx}_{uuid}.geojson')
    else:
        """spent a few hours on this... some pulls would not complete
        had to manually clean in QGIS"""
        ofp = get_filepaths(out_dir, pfx, ext='.geojson')
        assert os.path.exists(ofp)
    
    log.debug(f'for {country_key} w/ bounds: {bounds} and ofp: {ofp}')
    #===========================================================================
    # build
    #===========================================================================
    if not os.path.exists(ofp) or (not use_cache):
        """this retrieves precompiled files if they are available"""
        log.debug(f'retriving OSM building footprints for {country_key} from bounds: {bounds}')
        poly_fp = retrieve_osm_buildings(country_key, bounds, logger=log, use_cache=True)
        
        if os.path.getsize(poly_fp)<1e3:
            log.error(f'empty osm poly file... skipping')
            return None
        #=======================================================================
        # #drop to centroid                
        #=======================================================================
        log.debug(f'extracting centroid from osm building poly file w/ {os.path.getsize(poly_fp)/(1024**3): .2f} GB \n    {poly_fp}')
 
        #optional cleaning
        if pre_clean_polys:
            """this didn't seem to help... could explore alternate cleaning methods"""            
            poly_clean_fp = ogr_clean_polys(poly_fp, os.path.join(temp_dir, f'{pfx}_clean_{uuid}.geojson'), log=log)
        else:
            poly_clean_fp=poly_fp
        
        #use ogr to export centroids
        ogr_export_geometry(poly_clean_fp, ofp, log=log)
        #_wbt_centroid(poly_fp, ofp, log=log)
        
        #===========================================================================
        # wrap
        #===========================================================================
        meta_d = {
                        'tdelta':(datetime.now()-start).total_seconds(),
                        'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                        'file_GB':os.path.getsize(ofp)/(1024**3),
                        #'output_MB':os.path.getsize(ofp)/(1024**2)
                        }
        log.debug(meta_d)
    
        
    else:
        log.debug(f'record exists ...loading from cache\n    {ofp}')
        
    return ofp

#===============================================================================
# EXECUTORS--------
#===============================================================================

def _wbt_sample(rlay_fp, bldg_pts_gser, ofp, hazard_key,log, nodata_l=None):
    """ sample points with WBT"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    log = log.getChild('wbt_sample')
    if nodata_l is None: nodata_l = list(fathom_vals_d.keys())
    #write to file
    """WBT requires a shape file..."""    
    bldg_pts_filter_fp = os.path.join(temp_dir, os.path.basename(ofp).split('.')[0]+'.shp')
    log.debug(f'wirting shapefile to \n    {bldg_pts_filter_fp}')    
    bldg_pts_gser.to_file(bldg_pts_filter_fp)
     
    #===========================================================================
    # execute
    #===========================================================================
    def wbt_callback(value):
        if not "%" in value:
            log.debug(value)
             
    wbt.extract_raster_values_at_points(
        rlay_fp, 
        bldg_pts_filter_fp, 
        out_text=False, 
        callback=wbt_callback)
    
    #===============================================================================
    # #clean and convert
    #===============================================================================
    log.debug(f'loading and cleaning wbt result file: {bldg_pts_filter_fp}')
    gdf_raw = gpd.read_file(bldg_pts_filter_fp)
    log.debug(f'loaded {gdf_raw.shape}')
    gdf1 = gdf_raw.rename(columns={'VALUE1':hazard_key}).drop('FID', axis=1)
    
    bx = gdf1[hazard_key].astype(float).isin(nodata_l)
    if bx.any():
        log.debug(f'    set {bx.sum()}/{len(bx)} nodata vals to nan')
        gdf1.loc[bx, hazard_key]=np.nan
 
    log.debug(f'    writing {len(gdf1)} samples to: {ofp}')
    gdf1.to_file(ofp)
 
    return ofp


def _wbt_centroid(poly_fp,  ofp,log=None):
    """ sample points with WBT"""
    #write to file
    """WBT requires a shape file..."""    
    #poly_shp_fp = os.path.join(temp_dir, os.path.basename(ofp).split('.')[0]+'.shp')    
    #bldg_pts_gser.to_file(bldg_pts_filter_fp)
    
    shp_ofp = os.path.join(temp_dir, os.path.basename(ofp).split('.')[0]+'.shp')    
    #===========================================================================
    # execute
    #===========================================================================
    def wbt_callback(value):
        if not "%" in value:
            log.debug(value)
            
    log.debug(f'wbt.centroid_vector on {poly_fp}')
    res = wbt.centroid_vector(
        poly_fp, 
        shp_ofp, 
 
        callback=wbt_callback)
    
    assert res==0
    
    #===============================================================================
    # #clean and convert
    #===============================================================================
    #===========================================================================
    # log.debug(f'loading and cleaning wbt result file: {bldg_pts_filter_fp}')
    # bldg_pts_sample_gser = gpd.read_file(bldg_pts_filter_fp)
    #===========================================================================

def _sample_igrid(country_key, hazard_key, haz_tile_gdf, row, area_thresh, 
                  epsg_id, out_dir, log=None, haz_base_dir=None, nodata=-32767):
    
    if log is None: log=get_log_stream()
    
    i = row['id']
    log = log.getChild(str(i))
    #===========================================================================
    # get record
    #===========================================================================
    fnstr = f'{country_key}_{hazard_key}_{i}'
    uuid = hashlib.shake_256(f'{fnstr}_{epsg_id}_{area_thresh}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    ofp = os.path.join(out_dir,f'{fnstr}_{uuid}.geojson')
    
    log.debug(f' w/ {ofp}')
    if not os.path.exists(ofp):
        #=======================================================================
        # #get OSM building footprints
        #=======================================================================
        bldg_fp = get_osm_bldg_cent(country_key, row.geometry.bounds, pfx=fnstr, log=log)
        if bldg_fp is None:
            return None
        
        log.debug(f'loading bldg_cent {os.path.getsize(bldg_fp)/(1024**3): .2f} GB: {bldg_fp}')
        bldg_pts_gdf = gpd.read_file(bldg_fp)
        
        #apply filter
        log.debug(f'    applying filter on {len(bldg_pts_gdf)} w/ {area_thresh}')
        bx = bldg_pts_gdf['area'] > area_thresh
        
        if bx.sum()==0:
            log.warning(f'no valid buildings... skipping')
            return None
        
        bldg_pts_gser = bldg_pts_gdf[bx].geometry
        
        
        
        log.debug(f'    filtered {bx.sum()}/{len(bx)} w/ area_tresh={area_thresh}')
        #=======================================================================
        # #retrieve the corresponding hazard raster
        #=======================================================================
        bx = haz_tile_gdf.to_crs(epsg=epsg_id).geometry.intersects(row.geometry.centroid)
        assert bx.sum() == 1, f'no intersect'
        
        #get filepath
        """the tile_indexers give absolute filepaths (from when the index was created)"""
        rlay_fp = os.path.join(haz_base_dir, 'raw', os.path.basename(haz_tile_gdf[bx]['location'].values[0]))
        assert os.path.exists(rlay_fp)
        log.debug(f'    for grid {i} got hazard raster {os.path.basename(rlay_fp)}')
        
        #=======================================================================
        # #compute hte stats
        #=======================================================================
        log.info(f'    computing {len(bldg_pts_gser)} samples on {os.path.basename(rlay_fp)}')
 
        
        _wbt_sample(rlay_fp, bldg_pts_gser, ofp, hazard_key, log)
 
    else:
        log.info(f'    record exists: {ofp}')
    
    return ofp

def _multi_sample_igrid(i, row, country_key, hazard_key, haz_tile_gdf, area_thresh, epsg_id, out_dir, haz_base_dir):
    #log.info(f'{i+1}/{len(gdf)} on grid %i'%row['id'])
    try:
        res = _sample_igrid(country_key, hazard_key, haz_tile_gdf, row, area_thresh, epsg_id, out_dir,   haz_base_dir=haz_base_dir, log=None)
        return (i, res, None)
    except Exception as e:
        err = row.copy()
        err['error'] = str(e)
        #log.error(f'failed on {country_key}.{hazard_key}.{i} w/\n    {e}')
        return (i, None, err)


def run_samples_on_country(country_key, hazard_key,
                           out_dir=None,
                           temp_dir=None,
                           epsg_id=4326,
                           area_thresh=50,
                           max_workers=None,
                           
                           ):
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
    assert hazard_key in index_hazard_fp_d, hazard_key
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'inters','01_sample', country_key, hazard_key)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    if temp_dir is None:
        temp_dir = os.path.join(temp_dirM, 'sample', today_str)
    
    if not os.path.exists(temp_dir):os.makedirs(temp_dir)
    
    log = init_log(name=f'samp.{country_key}.{hazard_key}', fp=os.path.join(out_dir, today_str+'.log'))
    log.info(f'on {country_key} x {hazard_key}')
    
    #===========================================================================
    # #load tiles
    #===========================================================================
    #country 
    gdf = gpd.read_file(index_country_fp_d[country_key])
    log.info(f'loaded country tiles w/ {len(gdf)}')
    
    #add index
    if not 'id' in gdf:
        gdf['id'] = gdf.index
        
    
    #hazard
    haz_tile_gdf = gpd.read_file(index_hazard_fp_d[hazard_key])
    log.info(f'loaded hazard tiles w/ {len(haz_tile_gdf)}')
    
    haz_base_dir = os.path.dirname(index_hazard_fp_d[hazard_key])
    #===========================================================================
    # #loop through each tile in the country grid 
    #===========================================================================
 
    res_d, err_d=dict(), dict()
    cnt=0
    
    #===========================================================================
    # single thread
    #===========================================================================
    log.info(f'intersecting buildings and hazard per tile \n\n')
    if max_workers is None:
        for i, row in gdf.to_crs(epsg=epsg_id).iterrows():
            log.info(f'{i+1}/{len(gdf)} {country_key}.{hazard_key} on grid %i'%row['id'])
            
            res_d[i] = _sample_igrid(country_key, hazard_key, haz_tile_gdf, row, area_thresh, epsg_id, out_dir, log, haz_base_dir)
     #==========================================================================
     #        try:
     #            
     # 
     #        except Exception as e:
     #            err_d[i] = row.copy()
     #            err_d[i]['error'] = str(e)            
     #            log.error(f'failed on {country_key}.{hazard_key}.{i} w/\n    {e}')
     #==========================================================================
            #print(f'computing stats on {len(gdf)} feats')
            """
            print(rasterstats.utils.VALID_STATS)
            """
            #=======================================================================
            # start_i = datetime.now()
            # with fiona.open(poly_fp, mode='r') as src:
            #     log.info(f'computing stats on {len(src)} polys')            
            #     zs = zonal_stats(src, rlay_fp, nodata=-32768, stats=['min', 'max', 'mean'], all_touched=False)
            # 
            # res_df = pd.DataFrame.from_dict(zs).dropna(axis=0)
            # tdelta = (datetime.now() - start_i).total_seconds()
            # log.info(f'got {len(res_df)} feats w/ valid stats in {tdelta:.2f} secs')
            #=======================================================================
            cnt+=1
    
    #===========================================================================
    # MULTI thread
    #===========================================================================
    else:
        #gdf = gdf.iloc[0:20, :]
        log.info(f'running {len(gdf)} w/ max_workers={max_workers}')
        args = (country_key, hazard_key, haz_tile_gdf, area_thresh, epsg_id, out_dir, haz_base_dir)
        
        #run once to prime
        """reduce conflicts...
        get_tag_filter() for example applies to the whole country"""
        
        for i, row in gdf.to_crs(epsg=epsg_id).iterrows():
            _multi_sample_igrid(i, row, *args)
            break
 
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_multi_sample_igrid, i, row, *args) for i, row in gdf.to_crs(epsg=epsg_id).iterrows()]
            for future in futures:
                i, res, err = future.result()
                if res is not None:
                    res_d[i] = res
                if err is not None:
                    err_str = err['error']
                    log.error(f'{i} returned error:\n{err_str}')
                    err_d[i] = err
                cnt+=1
        
    
        
 
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished on {cnt}')
    #write errors
    if len(err_d)>0:
        err_ofp = os.path.join(out_dir, f'errors_{today_str}_{country_key}_{hazard_key}.gpkg')
        
        err_gdf = pd.concat(err_d, axis=1).T
 
        log.error(f'writing {len(err_d)} error summaries to \n    {err_ofp}\n'+err_gdf['error'])
        err_gdf.set_geometry(err_gdf['geometry']).to_file(err_ofp)
    
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':get_directory_size(out_dir),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
        
 
 
if __name__ == '__main__':
    
    run_samples_on_country('DEU', '500_fluvial', max_workers=None)
    
    
    
    
    
    
    
    
    
    
    