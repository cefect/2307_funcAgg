'''
Created on Aug. 24, 2023

@author: cefect
'''

#===============================================================================
# IMPORTS-------
#===============================================================================
import os, logging, pprint, webbrowser, sys
import logging.config
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd

from definitions import wrk_dir, logcfg_file


today_str = datetime.now().strftime('%Y%m%d')

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
        
    logger.setLevel(level)
    
    #===========================================================================
    # configure the handler
    #===========================================================================
    assert fp.endswith('.log')
    
    formatter = logging.Formatter(log_format_str)        
    handler = logging.FileHandler(fp, mode='w') #Create a file handler at the passed filename 
    handler.setFormatter(formatter) #attach teh formater object
    handler.setLevel(level) #set the level of the handler
    
    logger.addHandler(handler) #attach teh handler to the logger
    
    logger.info('built new file logger  here \n    %s'%(fp))
    
    return logger
 
def init_log(
 
        log_dir=wrk_dir,
        **kwargs):
    """wrapper to setup the root loger and create an additional local file logger"""
    
    root_logger = init_root_logger(log_dir=log_dir) 
    
    #set up the file logger
    return get_new_file_logger(**kwargs)


def get_log_stream(name=None, level=None):
    """get a logger with stream handler"""
    if name is None: name=str(os.getpid())
    if level is None:
        if __debug__:
            level=logging.DEBUG
        else:
            level=logging.INFO
    
    logger = logging.getLogger(name)
    
    #see if it has been configured
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(
            stream=sys.stdout, #send to stdout (supports colors)
            ) #Create a file handler at the passed filename 
        formatter = logging.Formatter(log_format_str) 
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
 
    
#===============================================================================
# MISC-----------
#===============================================================================
def dstr(d,
         width=100, indent=0.3, compact=True, sort_dicts =False,
         ):
    return pprint.pformat(d, width=width, indent=indent, compact=compact, sort_dicts =sort_dicts)

def view(df):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    import webbrowser
    #import pandas as pd
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(delete=False, suffix='.html', mode='w') as f:
        #type(f)
        df.to_html(buf=f)
        
    webbrowser.open(f.name)