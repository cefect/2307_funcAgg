'''
Created on Jul. 25, 2023

@author: cefect
'''

#===============================================================================
# imports
#===============================================================================
import os, hashlib, subprocess, sys, json
import osmium
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from rasterstats import zonal_stats

from definitions import root_dir

#OSM pbf data directories
osm_pbf_basedir=r'l:\10_IO\2210_AggFSyn\ins\osm_20230725'
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

def get_tag_filter(
        pbf_raw_fp,
        filter_str='a/building',
        #precompiled_index_fp=r'l:\10_IO\2210_AggFSyn\ins\osm_20230725\tag_filters\index.pkl',
        lib_dir= None,
        overwrite=False,
        ):
    """
    retrieve pbf file with tag filter applied
    
    INPUTS
    ------
    precompiled_index_fp: str
        filepath to index 
    """
    
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
    #filter_fp=os.path.join(lib_dir, f'{os.path.basename(pbf_raw_fp)}_{filter}.pbf'.replace(r'/', '_'))
    
    if not os.path.exists(filter_fp):   
        cmd_str = f'osmium tags-filter {pbf_raw_fp} {filter_str} -o {filter_fp}'
        print(f'executing \n    {cmd_str}')
        result = os.system(cmd_str)
        assert result==0, f'tags-filter failed w/ {result}'
        
        #p = subprocess.run(['osmium', 'tags-filter', pbf_raw_fp, filter, '-o', filter_fp], stderr=sys.stderr, stdout=sys.stdout, check=True)
        
        print(f'finished on {filter_fp}')
        
    else:
        print(f'tag_filter already exists')
        
 
    assert os.path.exists(filter_fp)
    
    return filter_fp        


def get_box_filter(
        pbf_fp,
        bounds,
        lib_dir= None,
        ):
    """
    filter a pbf file by bounds
    """
    #setup the directory
    lib_dir = get_lib_dir(lib_dir, 'box')

    
    
    uuid = hashlib.sha256(f'{pbf_fp}_{bounds}'.encode("utf-8")).hexdigest()
    
    filter_fp = os.path.join(lib_dir, f'{uuid}.pbf')
 
    
    if not os.path.exists(filter_fp):   
        cmd_str = f'osmium extract {pbf_fp} -b {bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]} -s simple -o {filter_fp}'
        print(f'executing \n    {cmd_str}')
        result = os.system(cmd_str)
        assert result==0, f'tags-filter failed w/ {result}'
        
        #p = subprocess.run(['osmium', 'tags-filter', pbf_raw_fp, filter, '-o', filter_fp], stderr=sys.stderr, stdout=sys.stdout, check=True)
        
        print(f'finished on {filter_fp}')
        
    else:
        print(f'tag_filter already exists')
        
 
    assert os.path.exists(filter_fp)
    
    return filter_fp       

def export_pbf_to_geojson(pbf_fp,
           lib_dir= osm_cache_dir,
           ):
    """export a pbf file into a GIS file""" 
    
    uuid = hashlib.sha256(f'{pbf_fp}'.encode("utf-8")).hexdigest()
    
    ofp = os.path.join(lib_dir, f'{uuid}.geojson')
    
    if not os.path.exists(ofp):   
        cmd_str = f'osmium export {pbf_fp} --geometry-types=polygon -o {ofp}'
        print(f'executing \n    {cmd_str}')
        result = os.system(cmd_str)
        assert result==0, f'tags-filter failed w/ {result}'
 
        
        print(f'finished on {ofp}')
        
    else:
        print(f'tag_filter already exists')
        
 
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
        ):
    """retrieve osm buildings
    
    because of the osmium calls, we perform the filter in two layers.
    use pre-compilled pbf files from the cache if available
    
    """
    #setup directory
    #lib_dir = get_lib_dir(lib_dir, country_key)
    
    #country raw data
    pbf_fp = os.path.join(osm_pbf_basedir, osm_pbf_data[country_key])
    assert os.path.exists(pbf_fp)
    
    #tag filter
    filter_tag_fp = get_tag_filter(pbf_fp)
    
    #bounding box filter
    filter_tag_box_fp = get_box_filter(filter_tag_fp, bounds)
    
    #export data
    osm_filter_fp = export_pbf_to_geojson(filter_tag_box_fp)
    #extract_geometries(filter_tag_box_fp)
    
    
    
    #sample rasters
    #gdf = pbf_to_geodataframe(filter_tag_box_fp)
    #area_process(filter_tag_box_fp)
    
    
    print(f'finished on \n    {osm_filter_fp}')
    
    
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
        
        
        
        
        
        
        
