'''
Created on Aug. 24, 2023

@author: cefect
'''
import argparse
from _04expo._01_full_links import run_agg_bldg_full_links as func

 

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run building RL means.')
    parser.add_argument('--country_key', required=True, help='Country key')
    parser.add_argument('--grid_size', required=True, type=int, help='Grid size')
 

    args = parser.parse_args()

    func(
         args.country_key,
         args.grid_size,
 
    )