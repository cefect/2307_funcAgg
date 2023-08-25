'''
Created on Aug. 24, 2023

@author: cefect
'''
import argparse
from intersect.main import run_samples_on_country


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run samples on country')
    parser.add_argument('country_key', type=str, help='Country key')
    parser.add_argument('hazard_key', type=str, help='Hazard key')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--temp_dir', type=str, default=None, help='Temporary directory')
    parser.add_argument('--epsg_id', type=int, default=4326, help='EPSG ID')
    parser.add_argument('--area_thresh', type=int, default=50, help='Area threshold')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of workers')

    args = parser.parse_args()

    run_samples_on_country(args.country_key, args.hazard_key,
                           out_dir=args.out_dir,
                           temp_dir=args.temp_dir,
                           epsg_id=args.epsg_id,
                           area_thresh=args.area_thresh,
                           max_workers=args.max_workers)