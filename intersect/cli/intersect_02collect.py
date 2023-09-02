'''
Created on Aug. 26, 2023

@author: cefect
'''

import argparse
from intersect._02_collect import run_collect_sims


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run collect sims')
    parser.add_argument('--srch_dir', type=str, default=None, help='search directory')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory')
    parser.add_argument('--max_workers', type=int, default=10, help='maximum number of workers')
    
    args = parser.parse_args()
    
    run_collect_sims(args.srch_dir, args.out_dir, args.max_workers)