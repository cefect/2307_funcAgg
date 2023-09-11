'''
Created on Aug. 24, 2023

@author: cefect
'''
import argparse
from expo_stats._01_pdist import run_build_pdist, gridsize_default_l


from definitions import index_country_fp_d, equal_area_epsg


 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute pdist on each agg group')
    parser.add_argument('--maxworkers', type=int,default=4, help='workers ofr processing pool')
 
 
    parser.add_argument('--gridsizes', nargs='+', type=int, default=gridsize_default_l, help='Grid size list')
    parser.add_argument('--countries', nargs='+', type=str, 
                        default=[e.lower() for e in index_country_fp_d.keys()], 
                        help='Country list')

    args = parser.parse_args()
    
    run_build_pdist(
        #out_dir=args.out_dir,
        #conn_d=args.conn,
        max_workers=args.maxworkers,
        grid_size_l=args.gridsizes,
        country_l=args.countries
    )
