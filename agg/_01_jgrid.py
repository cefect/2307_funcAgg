'''
Created on Sep. 2, 2023

@author: cefect

create aggregation geometries and join to points
'''

import os, hashlib, sys, subprocess
from datetime import datetime

import psycopg2
print('psycopg2.__version__=' + psycopg2.__version__)


from tqdm import tqdm

from coms import (
    init_log, today_str, get_log_stream, get_raster_point_samples, get_directory_size,
    dstr, view
    )

from definitions import index_country_fp_d, wrk_dir, postgres_d


def run_join_agg_grids(
        country_l=None,
        out_dir=None,
        ):
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()    
 
 
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'agg', '01_jgrid')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
 
    
    log = init_log(name=f'toPost', fp=os.path.join(out_dir, today_str+'.log'))
    log.info(f'on \n    {country_l}')
    
    
    #===========================================================================
    # loop on each country
    #===========================================================================
    for country_key in country_l:
        log.info(f'on {country_key}')
        
        #compute the grid extent
        """SELECT ST_AsText(ST_Extent(geometry))
        FROM grids.country_grids
        WHERE country_key=BGD;"""
        
        #build the mesh (need to filter with ST_Intersects still
        """WITH grid AS (
SELECT (ST_SquareGrid(1, ST_Transform(geom,4326))).*
FROM admin0 WHERE name = 'Canada'
)
  SELEcT ST_AsText(geom)
  FROM grid
  """
        

if __name__ == '__main__':
    run_join_agg_grids()