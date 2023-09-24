'''
Created on Aug. 24, 2023

@author: cefect
'''
import argparse
from _04expo._01_full_links import run_all as func

 
"""not sure parallelization helps as there is only 1 country table for hte poitns"""
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run building RL means.')
    parser.add_argument('--country_key', required=False, default='deu', help='Country key')
    #parser.add_argument('--filter_cent_expo', type=bool, default=False, help='filter schem e')
 

    args = parser.parse_args()
    
    print(args)

    func(
         country_key = args.country_key,
         #filter_cent_expo = args.filter_cent_expo,
 
    )
