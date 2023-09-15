'''
Created on Sep. 6, 2023

@author: cefect

common functions for working with functions
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

    return serx.droplevel(drop_lvl_names)
    

 



  
