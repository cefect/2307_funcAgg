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

from definitions import osm_pbf_basedir

from hp import get_log_stream, dstr



#OSM pbf data directories

osm_pbf_data = {
    'BGD':'bangladesh-latest.osm.pbf',
    'AUS':'australia-latest.osm.pbf',
    'DEU':'germany-latest.osm.pbf',
    'CAN':'canada-latest.osm.pbf',
    'ZAF':'south-africa-latest.osm.pbf',    
    }

osm_cache_dir = os.path.join(osm_pbf_basedir, 'cache')
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
    
    log.info(f'executing \'osmium {osmium_cmd}\' w/ {args}')
    
    p = subprocess.run(['osmium', osmium_cmd,*args], stderr=sys.stderr, stdout=sys.stdout, check=True)        
    assert p.returncode==0        
    log.debug(f'completed')
    
    return p
    

def get_tag_filter(
        pbf_raw_fp,
        filter_str='a/building',
        #precompiled_index_fp=r'l:\10_IO\2210_AggFSyn\ins\osm_20230725\tag_filters\index.pkl',
        lib_dir= None,
        overwrite=False,
        logger=None
        ):
    """
    retrieve pbf file with tag filter applied
    
    INPUTS
    ------
    precompiled_index_fp: str
        filepath to index 
    """
    log = logger.getChild('tagFilter')
    log.debug(f'applying filter \'{filter_str}\' to \n    {pbf_raw_fp}')
    lib_dir = get_lib_dir(lib_dir, 'tag')
    
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
    uuid = hashlib.sha256(f'{os.path.basename(pbf_raw_fp)}_{filter_str}'.encode("utf-8")).hexdigest()    
    filter_fp = os.path.join(lib_dir, f'{uuid}.pbf') 
    
    log.debug(f'on {filter_fp}')
    
    if not os.path.exists(filter_fp):   
        #=======================================================================
        # cmd_str = f'osmium tags-filter {pbf_raw_fp} {filter_str} -o {filter_fp}'
        # log.debug(f'executing \n    {cmd_str}')
        # result = os.system(cmd_str)
        # 
        # 
        # assert result==0, f'tags-filter failed w/ {result}'
        #=======================================================================
        _ = _exe_osmimum('tags-filter', pbf_raw_fp, filter_str, '-o', filter_fp, '--progress', log=log)
        #=======================================================================
        # log.info(f'executing \'osmium tags-filter\' on {filter_fp}')
        # p = subprocess.run(['osmium', 'tags-filter', pbf_raw_fp, filter_str, '-o', filter_fp, '--progress'], stderr=sys.stderr, stdout=sys.stdout, check=True)        
        # assert p.returncode==0        
        # log.debug(f'filtere applied successfully and retrived: \n    {filter_fp}')
        #=======================================================================
        
    else:
        log.debug(f'tag_filter already exists')
        
 
    assert os.path.exists(filter_fp)
    
    return filter_fp        


def get_box_filter(
        pbf_fp,
        bounds,
        lib_dir= None,
        logger=None,
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
    lib_dir = get_lib_dir(lib_dir, 'box')    
    
    #get the file
    uuid = hashlib.sha256(f'{pbf_fp}_{bounds}'.encode("utf-8")).hexdigest()    
    filter_fp = os.path.join(lib_dir, f'{uuid}.pbf')
 
    
    if not os.path.exists(filter_fp):   
 #==============================================================================
 #        cmd_str = f'osmium extract {pbf_fp} -b {bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]} -s simple -o {filter_fp}'
 #        log.debug(f'executing \n    {cmd_str}')
 # 
 #        result = os.system(cmd_str)
 #        assert result==0, f'tags-filter failed w/ {result}'
 #        
 #        #p = subprocess.run(['osmium', 'tags-filter', pbf_raw_fp, filter, '-o', filter_fp], stderr=sys.stderr, stdout=sys.stdout, check=True)
 #        
 #        log.debug(f'finished on {filter_fp}')
 #==============================================================================
        
        _ = _exe_osmimum('extract', pbf_fp, '--bbox', f'{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}','-s','simple','-o',filter_fp, '--progress', log=log)
        
    else:
        log.debug(f'box_filter already exists')
        
 
    assert os.path.exists(filter_fp)
    
    return filter_fp       

def export_pbf_to_geojson(pbf_fp,
           lib_dir= osm_cache_dir,
           logger=None,
           ):
    """export a pbf file into a GIS file""" 
    #===========================================================================
    # setup
    #===========================================================================
    log = logger.getChild('export')    
    
    uuid = hashlib.sha256(f'{pbf_fp}'.encode("utf-8")).hexdigest()    
    ofp = os.path.join(lib_dir, f'{uuid}.geojson')
    
    log.debug(f'exporting\n    from:{pbf_fp}\n    to:{ofp}')
    """
    print(ofp)
    """
    
    if not os.path.exists(ofp):   
 #==============================================================================
 #        cmd_str = f'osmium export {pbf_fp} --geometry-types=polygon -o {ofp}'
 #        log.debug(f'executing \n    {cmd_str}')
 #        result = os.system(cmd_str)
 #        assert result==0, f'tags-filter failed w/ {result}'
 # 
 #        
 #        log.debug(f'finished on {ofp}')
 #==============================================================================
        
        _ = _exe_osmimum('export', pbf_fp, '--geometry-types=polygon','-o',ofp, '--progress', log=log)
        
    else:
        log.debug(f'exported file already exists')
        
 
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

        
def pbf_to_geodataframe(pbf_fp):
    """extract just the polygons from the pbf file"""
    
    # Create an instance of the handler and apply it to the PBF file
    handler = AreaHandler()
    
    print(f'parsing areas from {pbf_fp}')
    handler.apply_file(pbf_fp)
    
    gdf = gpd.GeoDataFrame.from_features(handler.features)
    
    print(f'finished w/ {len(gdf)}')
    
    return gdf
    
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
    pbf_fp = os.path.join(osm_pbf_basedir, osm_pbf_data[country_key])
    assert os.path.exists(pbf_fp)
    
    #===========================================================================
    # filters
    #===========================================================================
    #tag filter
    filter_tag_fp = get_tag_filter(pbf_fp, logger=log)
    
    #bounding box filter
    filter_tag_box_fp = get_box_filter(filter_tag_fp, bounds, logger=log)
    
    #===========================================================================
    # #export data
    #===========================================================================
    osm_filter_fp = export_pbf_to_geojson(filter_tag_box_fp, logger=log)
 
    
    #===========================================================================
    # wrap
    #===========================================================================
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    'file_GB':os.path.getsize(osm_filter_fp)/(1024**3),
                    #'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
            
    log.info(f'buildings retrived at \n    {osm_filter_fp}\n    {dstr(meta_d)}')
    
    
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
        
        
        
        
        
        
        
