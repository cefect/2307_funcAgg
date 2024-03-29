'''
Created on Aug. 24, 2023

@author: cefect
'''

#===============================================================================
# IMPORTS-------
#===============================================================================
import os, logging, pprint, webbrowser, sys, glob
import logging.config
from datetime import datetime

import numpy as np
import pandas as pd

import rasterio as rio

import shapely
import geopandas as gpd



from definitions import wrk_dir, logcfg_file, temp_dir

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


today_str = datetime.now().strftime('%Y%m%d')

#===============================================================================
# vars
#===============================================================================


#===============================================================================
# loggers-----------
#===============================================================================
log_format_str = '%(levelname)s.%(name)s.%(asctime)s:  %(message)s'
def init_root_logger( 
        log_dir = wrk_dir,
        ):
    """set up the root logger and config from file
    
    typically, our config file has these handlers:
        handler_consoleHandler: level=INFO
        handler_consoleHandlerError: level=WARNING
        handler_fileHandler1: level=DEBUG
        handler_fileHandler2: level=WARNING
        
    """
        
        
        
        
    logger = logging.getLogger() #get the root logger
    
    logging.config.fileConfig(logcfg_file,
                              defaults={'logdir':str(log_dir).replace('\\','/')},
                              disable_existing_loggers=True,
                              ) #load the configuration file 
    
    logger.info(f'root logger initiated and configured from file: {logcfg_file}\n    logdir={log_dir}')
    
    return logger

def get_new_file_logger(
        name='r',
        level=logging.DEBUG,
        fp=None, #file location to log to
        logger=None,
        ):
    
    #===========================================================================
    # configure the logger
    #===========================================================================
    if logger is None:
        logger = logging.getLogger(name)
        
    if fp is None:
        return logger
        
    #logger.setLevel(level)
    
    #===========================================================================
    # configure the handler
    #===========================================================================
    
    assert fp.endswith('.log')
    
    formatter = logging.Formatter(log_format_str)        
    handler = logging.FileHandler(fp, mode='w') #Create a file handler at the passed filename 
    handler.setFormatter(formatter) #attach teh formater object
    handler.setLevel(level) #set the level of the handler
    
    logger.addHandler(handler) #attach teh handler to the logger
    
    if not os.path.exists(fp):
        logger.info('built new file logger  here \n    %s'%(fp))
    
    return logger
 
def init_log(
 
        log_dir=wrk_dir,
        **kwargs):
    """wrapper to setup the root loger and create an additional local file logger"""
    
    root_logger = init_root_logger(log_dir=log_dir) 
    
    #set up the file logger
    return get_new_file_logger(**kwargs)

def init_log_worker(name=None,
                    stream_level=logging.INFO,
                    file_level=logging.DEBUG,
                    fp=None,):
    """setup logging for a new processing worker... for concurrent.futures"""
    
    if name is None: 
        name=str(os.getpid())
        
    if fp is None:
        fp = os.path.join(temp_dir, 'log',f'{today_str}_{name}.log')
        
    if not os.path.exists(os.path.dirname(fp)):
        os.makedirs(os.path.dirname(fp))
    
    #setup the logger
    
    logger = logging.getLogger(name)
    
    #add the stream handler
    logger = get_log_stream(name=name, logger=logger, level=stream_level)
    
    #add teh file handler
 
 
    
        
    return get_new_file_logger(name=name, logger=logger, level=file_level, fp=fp)
    


def get_log_stream(name=None, 
                   level=logging.INFO, 
                   logger=None,
                   ):
    """get a logger with stream handler
    seems to be disconnecting from the root log..."""
    if name is None: 
        name=str(os.getpid())
        
    if level is None:
        if __debug__:
            level=logging.DEBUG
        else:
            level=logging.INFO
    
    if logger is None:
        logger = logging.getLogger(name)
        

    
    #see if it has been configured
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        logger.setLevel(level)
        handler = logging.StreamHandler(
            stream=sys.stdout, #send to stdout (supports colors)
            ) 
        formatter = logging.Formatter(log_format_str, datefmt='%H:%M:%S') 
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


#===============================================================================
# FILES-------
#===============================================================================
def get_directory_size(directory):
    total_size = 0
    for path, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024**3)


def _get_filepaths(search_dir, pattern='*.pkl'):
    # Use os.path.join to ensure the path is constructed correctly for the OS
    search_pattern = os.path.join(search_dir, '**', pattern)

    # Use glob.glob with recursive set to True to find all .pkl files in the directory
 

    return glob.glob(search_pattern, recursive=True)


def get_filepaths(search_dir, pattern, single=True, ext=None, recursive=False):
    
    if not recursive:
        l = [os.path.join(search_dir, e) for e in os.listdir(search_dir) if pattern in e]
    else:
        
        l = list()
        for dirpath, _, fns in os.walk(search_dir):
            l = l+[os.path.join(dirpath, e) for e in fns if pattern in e]
                
 

    
    if not ext is None:
        l = [e for e in l if e.endswith(ext)]
    
    assert len(l)>0, f'failed to locate any files w/ patter = \'{pattern}\' from \n    {search_dir}'
    if single:
        assert len(l)==1, l
        return l[0]
    
    return l
    
    
    
#===============================================================================
# PANDAS-----
#===============================================================================

def pd_mdex_append_level(
        mdex, d
        ):
    """add a simple level from the d"""
    
    mdf = mdex.to_frame().reset_index(drop=True)
    
    for k,v in d.items():
        mdf[k]=v
        
    return pd.MultiIndex.from_frame(mdf)
    
def pd_ser_meta(ser):
    return {'len':len(ser), 'null_cnt':ser.isna().sum(), 'zero_cnt':(ser==0).sum(), 'max':ser.max(), 'min':ser.min(), 'std':ser.std()}
#===============================================================================
# GEOPANDAS-------
#===============================================================================
def get_raster_point_samples(gser, rlay_fp, colName=None, nodata=None,
                             ):
    """sample a raster with some points"""
    
    assert isinstance(gser, gpd.geoseries.GeoSeries)
    assert np.all(gser.geom_type=='Point')
    
    with rio.open(rlay_fp) as rlay_ds:
 
        #defaults
        if colName is None: 
            colName = os.path.basename(rlay_ds.name)
            
        if nodata is None:
            nodata=rlay_ds.nodata
        
        #get points
        coord_l = [(x,y) for x,y in zip(gser.x , gser.y)]
        samp_l = [x[0] for x in rlay_ds.sample(coord_l)]
     
        
        #replace nulls
        samp_ar = np.where(np.array([samp_l])==nodata, np.nan, np.array([samp_l]))[0]        
        
        
    return gpd.GeoDataFrame(data={colName:samp_ar}, index=gser.index, geometry=gser)
 
def clean_geodataframe(gdf_raw, gcoln='geometry', **kwargs):
    
    # Ensure 'geometry' column is recognized as a geometry column
    #===========================================================================
    # df[gcoln] = df[gcoln].apply(shapely.wkt.loads)
    # gdf = gpd.GeoDataFrame(df, geometry=gcoln)
    #===========================================================================
    if not isinstance(gdf_raw, gpd.GeoDataFrame):
        gser = gdf_raw.pop(gcoln)
        gdf = gpd.GeoDataFrame(gdf_raw, geometry=gser.values, **kwargs)
    else:
        gdf = gdf_raw.drop(gcoln, axis=1).set_geometry(gdf_raw[gcoln])
    
    # Move gcoln column to the end
    cols = list(gdf.columns.values)
    cols.pop(cols.index(gcoln))
 
    return gdf[cols+[gcoln]]
#===============================================================================
# MISC-----------
#===============================================================================
def dstr(d,
         width=100, indent=0.3, compact=True, sort_dicts =False,
         ):
    return pprint.pformat(d, width=width, indent=indent, compact=compact, sort_dicts =sort_dicts)

def view(df):
    """view a DataFrame in the system default web browser"""
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    import webbrowser
    #import pandas as pd
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        #type(f)
        df.to_html(buf=f)
        
    webbrowser.open(f.name)