'''
Created on Aug. 24, 2023

@author: cefect
'''
import argparse
from agg._01_grids import run_build_agg_grids, gridssize_default_l

from definitions import index_country_fp_d, equal_area_epsg


 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run build agg grids')
    #parser.add_argument('--out-dir', type=str, help='Output directory')
    #parser.add_argument('--conn', type=dict, default=None, help='Connection dictionary')
    parser.add_argument('--epsg', type=int, default=equal_area_epsg, help='EPSG ID')
    parser.add_argument('--schema', type=str, default='grids', help='Schema')
    parser.add_argument('--gridsizes', nargs='+', type=int, default=gridssize_default_l, help='Grid size list')
    parser.add_argument('--countries', nargs='+', type=str, 
                        default=[e.lower() for e in index_country_fp_d.keys()], 
                        help='Country list')

    args = parser.parse_args()
    
    run_build_agg_grids(
        #out_dir=args.out_dir,
        #conn_d=args.conn,
        epsg_id=args.epsg,
        schema=args.schema,
        grid_size_l=args.gridsizes,
        country_l=args.countries
    )
