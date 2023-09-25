'''
Created on Aug. 24, 2023

@author: cefect
'''
import argparse
from _05depths._03_gstats import run_pg_build_gstats as func

 

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run building RL means.')
    parser.add_argument('--country_key', default='deu', help='Country key')
    parser.add_argument('--haz_key_l', default=None, help='hazards')
    #parser.add_argument('--grid_size', required=True, type=int, help='Grid size')
 

    args = parser.parse_args()

    func(
        country_key=args.country_key,
        haz_key_l=args.haz_key_l,
        )
