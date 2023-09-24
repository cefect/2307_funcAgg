'''
Created on Aug. 24, 2023

@author: cefect
'''
import argparse
from _02agg._04_occupied import run_all as func

 

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sample agg grids')
    parser.add_argument('--country_key', required=False,default='deu', help='Country key')
    #parser.add_argument('--hazard_key', required=True, help='Hazard key')
    parser.add_argument('--geom_type', type=str, default='point', help='geometry type')
    #===========================================================================
    # parser.add_argument('--out_dir', default=None, help='Output directory (default: None)')
    # parser.add_argument('--temp_dir', default=None, help='Temporary directory (default: None)')
    # parser.add_argument('--area_thresh', type=int, default=50, help='Area threshold (default: 50)')
    #===========================================================================
    #parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of workers (default: None)')

    args = parser.parse_args()
    
    func(country_key=args.country_key.upper(),   geom_type=args.geom_type)