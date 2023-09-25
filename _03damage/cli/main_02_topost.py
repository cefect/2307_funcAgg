'''
Created on Aug. 24, 2023

@author: cefect
'''
import argparse
from _03damage._02_topost import run_all as func

 

 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run building RL means.')
    parser.add_argument('--country_key', required=False, default='deu', help='Country key')
    #parser.add_argument('--grid_size', required=True, type=int, help='Grid size')
 

    args = parser.parse_args()

    func(
         country_key=args.country_key,
 
 
    )
