'''
Created on Jul. 25, 2023

@author: cefect
'''

#===============================================================================
# imports
#===============================================================================
import os, hashlib, subprocess, sys, json, shutil
from datetime import datetime
import psutil
import osmium
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from rasterstats import zonal_stats

from definitions import osm_pbf_data, osm_cache_dir

from coms import get_log_stream, dstr



#OSM pbf data directories
if not os.path.exists(osm_cache_dir): os.makedirs(osm_cache_dir)

#===============================================================================
# check osmium
#===============================================================================
assert os.path.exists(shutil.which('osmium')), f'failed to find osmium.exe in the path'

#===============================================================================
# class BuildingFootprints(osmium.SimpleHandler):
#     def __init__(self):
#         super().__init__()
#         
#         self.num_buildings = 0
#     
#     def way(self, w):
#         if 'building' in w.tags:
#             self.num_buildings += 1
#===============================================================================

def get_lib_dir(lib_dir, suffix):
    
    if lib_dir is None: 
        lib_dir = os.path.join(osm_cache_dir, suffix)
        
    if not os.path.exists(lib_dir): os.makedirs(lib_dir)
    
    return lib_dir

def _exe_osmimum(osmium_cmd, *args, log=None):
    if log is None: log=get_log_stream()
    
    log.debug(f'executing \'osmium {osmium_cmd}\'')
    log.debug(f'args:{args}')
    
    try:
        p = subprocess.run(['osmium', osmium_cmd,*args], stderr=sys.stderr, stdout=sys.stdout, check=True)
    except Exception as e:
        raise IOError(f'osmium command failed w/ \n    {e}')        
    if not p.returncode==0:
        raise AssertionError(f'osmium command failed')        
    log.debug(f'completed')
    
    return p
    

def get_tag_filter(
        pbf_raw_fp,
        filter_str='a/building',
        #precompiled_index_fp=r'l:\10_IO\2210_AggFSyn\ins\osm_20230725\tag_filters\index.pkl',
        lib_dir= None,
        overwrite=False,
        logger=None, use_cache=True,
        ):
    """
    retrieve pbf file with tag filter applied
    
    filteres entire country pbf to just get buildings
    
    INPUTS
    ------
    precompiled_index_fp: str
        filepath to index 
    """
    log = logger.getChild('tagFilter')
    log.debug(f'applying filter \'{filter_str}\' to \n    {pbf_raw_fp}')
    lib_dir = get_lib_dir(lib_dir, '01_tag')
    
    #===========================================================================
    # load index4
    #===========================================================================
    #===========================================================================
    # if not os.path.exists(precompiled_index_fp):
    #     index_df = pd.DataFrame(columns=['country_key', 'pbf_raw_fp', 'date', 'uuid'])
    #     
    # else:
    #     index_df= pd.read_pickle(precompiled_index_fp)
    #===========================================================================
        
    #===========================================================================
    # build 
    #===========================================================================
    #get hex
  
    uuid = hashlib.shake_256(f'{pbf_raw_fp}_{filter_str}'.encode("utf-8"), usedforsecurity=False).hexdigest(16)
    fnstr = os.path.basename(pbf_raw_fp).replace('.osm.pbf', '').replace('-latest', '') #nice country string
    filter_fp = os.path.join(lib_dir, f'{fnstr}_{uuid}.pbf') 
    
    log.debug(f'for {filter_fp}')
    
    if (not os.path.exists(filter_fp)) or (not use_cache):   
        log.info(f'applying tags-filter')
        _ = _exe_osmimum('tags-filter', pbf_raw_fp, filter_str, '-o', filter_fp, '--progress', log=log)
 
        
    else:
        log.debug(f'tag_filter already exists...loading from cache')
        
 
    assert os.path.exists(filter_fp)
    assert os.path.getsize(filter_fp)>1e3, f'bad filesize on {filter_fp}'
    
    return filter_fp        


def get_box_filter(
        pbf_fp,
        bounds,
        lib_dir= None,
        logger=None, use_cache=True,
        ):
    """
    filter a pbf file by bounds
    """
    #===========================================================================
    # setup
    #===========================================================================
    log = logger.getChild('boxFilter')
    log.debug(f'applying bounds filter \'{bounds}\' to \n    {pbf_fp}')
    #setup the directory
    fnstr = os.path.basename(pbf_fp).split('_')[0] #nice countyry string
    
    if lib_dir is None: 
        lib_dir = os.path.join(osm_cache_dir, fnstr, '02_box')
        
    if not os.path.exists(lib_dir): os.makedirs(lib_dir)
 
    
    #get the file
    uuid = hashlib.sha256(f'{pbf_fp}_{bounds}'.encode("utf-8")).hexdigest()    
    
    filter_fp = os.path.join(lib_dir, f'{fnstr}_{uuid}.pbf')
 
    
    if (not os.path.exists(filter_fp)) or (not use_cache): 
        log.debug(f'executing extract --bbox on {pbf_fp}')
        _ = _exe_osmimum('extract', pbf_fp, '--bbox', 
                         f'{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}',
                         '-s','simple','-o',filter_fp,'--overwrite',log=log)
        
        log.debug(f'wrote box_filter to {filter_fp}')
    else:
        log.debug(f'box_filter already exists.. .loading from cache\n    {filter_fp}')
        
 
    assert os.path.exists(filter_fp)
    
    return filter_fp       

def export_pbf_to_geojson(pbf_fp,
           lib_dir= None,
           logger=None,use_cache=True,
           ):
    """export a pbf file into a GIS file
    
    might be a way to use this for filtering as well
    couldn't figure out how to exclude fields (i.e., geometry only export)""" 
    #===========================================================================
    # setup
    #===========================================================================
    log = logger.getChild('export')    
    
    #setup the directory
    lib_dir = get_lib_dir(lib_dir, '03_export') 
    
    uuid = hashlib.sha256(f'{pbf_fp}'.encode("utf-8")).hexdigest()    
    fnstr = os.path.basename(pbf_fp).split('_')[0] #nice countyry string
    ofp = os.path.join(lib_dir, f'{fnstr}_{uuid}.geojson')
    
    
    """
    print(ofp)
    """
    
    if (not os.path.exists(ofp)) or (not use_cache):   
        log.debug(f'exporting\n    from:{pbf_fp}\n    to:{ofp}')
        
        _ = _exe_osmimum('export', pbf_fp, '--geometry-types=polygon','-o',ofp, '--overwrite',log=log)
        
    else:
        log.debug(f'exported file already exists...loading from cache\n    {ofp}')
        
 
    assert os.path.exists(ofp)
    
    return ofp  

class AreaHandler(osmium.SimpleHandler):
    def __init__(self):
        super(AreaHandler, self).__init__()
        self.factory = osmium.geom.GeoJSONFactory()
        self.features = []

    def area(self, a):
        try:
            # Create a MultiPolygon geometry from the area
            multipolygon = self.factory.create_multipolygon(a)
            if multipolygon:
                # Load the GeoJSON string into a dictionary
                geometry = json.loads(multipolygon)
                # Create a new dictionary to represent the feature
                feature = {
                    'type': 'Feature',
                    'geometry': geometry,
                    'properties': {}#{tag.k: tag.v for tag in a.tags}
                }
                # Add the feature to the list of features
                self.features.append(feature)
        except:
            pass

        
#===============================================================================
# def pbf_to_geodataframe(pbf_fp):
#     """extract just the polygons from the pbf file"""
#     
#     # Create an instance of the handler and apply it to the PBF file
#     handler = AreaHandler()
#     
#     print(f'parsing areas from {pbf_fp}')
#     handler.apply_file(pbf_fp)
#     
#     gdf = gpd.GeoDataFrame.from_features(handler.features)
#     
#     print(f'finished w/ {len(gdf)}')
#     
#     return gdf
#===============================================================================
    
    #gdf.to_file(r'l:\10_IO\2210_AggFSyn\outs\gdf.gpkg')

#===============================================================================
# class AreaProcessHandler(osmium.SimpleHandler):
#     def __init__(self):
#         super(AreaProcessHandler, self).__init__()
#         self.factory = osmium.geom.GeoJSONFactory()
#         self.features = []
# 
#     def area(self, a):
#         try:
#             # Create a MultiPolygon geometry from the area
#             multipolygon = self.factory.create_multipolygon(a)
#             if multipolygon:
#                 # Load the GeoJSON string into a dictionary
#                 geometry = json.loads(multipolygon)
#                 
#                 shape(geometry)
#                 
#  
#                 # Add the feature to the list of features
#                 #self.features.append(feature)
#         except:
#             pass
#     
# def area_process(pbf_fp):
#     
#         # Create an instance of the handler and apply it to the PBF file
#     handler = AreaProcessHandler()
#     
#     print(f'parsing areas from {pbf_fp}')
#     handler.apply_file(pbf_fp)
#===============================================================================
    
    
        
    

def retrieve_osm_buildings(
        
        country_key,
        bounds,
        #lib_dir=None,      
        logger=None,
        use_cache=True,
        ):
    """retrieve osm buildings
    
    because of the osmium calls, we perform the filter in two layers.
    use pre-compilled pbf files from the cache if available
    
    """
    #===========================================================================
    # setup
    #===========================================================================
    start=datetime.now()
    if logger is None: logger = get_log_stream()
    log = logger.getChild('osm')
 
    log.debug(f'on {country_key} \n    bounds: {bounds}')
    
    
    #===========================================================================
    # #country raw data
    #===========================================================================
    pbf_fp = osm_pbf_data[country_key]
    assert os.path.exists(pbf_fp)
    
    #===========================================================================
    # filters
    #===========================================================================
    #tag filter
    filter_tag_fp = get_tag_filter(pbf_fp, logger=log, use_cache=True) #this one is country-wide
    
    #bounding box filter
    filter_tag_box_fp = get_box_filter(filter_tag_fp, bounds, logger=log, use_cache=use_cache)
    
    #===========================================================================
    # #export data
    #===========================================================================
    osm_filter_fp = export_pbf_to_geojson(filter_tag_box_fp, logger=log, use_cache=use_cache)
 
    
    #===========================================================================
    # wrap
    #===========================================================================
    meta_d = {
                    'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':os.path.getsize(osm_filter_fp)/(1024**3),
                    'use_cache':use_cache,
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
            
    log.debug(f'buildings retrived at \n    {osm_filter_fp}\n    {dstr(meta_d)}')
    
    
    return osm_filter_fp
    
 
    

    

if __name__ == '__main__':
    
    #load tile
    tile_fp = r'l:\10_IO\2210_AggFSyn\ins\indexes\BGD_tindex_0725.gpkg'
    gdf = gpd.read_file(tile_fp)
    
    
    #loop through each tile and retrieve from bounds
    for i, row in gdf.to_crs(epsg=4326).iterrows():
        print(f'building for polygon %i'%row['id'])
        retrieve_osm_buildings('BGD', row.geometry.bounds)
        
        #break
        
        
        
        
        
        
        
