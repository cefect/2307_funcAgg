'''
Created on Aug. 24, 2023

@author: cefect
'''
import argparse
from agg._03_joins import run_join_agg_grids


from definitions import index_country_fp_d, gridsize_default_l


 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='join grids to points')
    #parser.add_argument('--out-dir', type=str, help='Output directory')
    #parser.add_argument('--conn', type=dict, default=None, help='Connection dictionary')
    #parser.add_argument('--epsg', type=int, default=equal_area_epsg, help='EPSG ID')
    #parser.add_argument('--schema', type=str, default='grids', help='Schema')
    parser.add_argument('--gridsizes', nargs='+', type=int, default=gridsize_default_l, help='Grid size list')
    parser.add_argument('--countries', nargs='+', type=str, 
                        default=[e.lower() for e in index_country_fp_d.keys()], 
                        help='Country list')

    args = parser.parse_args()
    
    run_join_agg_grids(
        #out_dir=args.out_dir,
        #conn_d=args.conn,
 
        grid_size_l=args.gridsizes,
        country_l=args.countries
    )
