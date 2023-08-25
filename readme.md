# Aggregation of Damage Functions
scripts for computation and data analysis of potential errors arising from aggregation of different loss functions



## install
build conda environment from ./environment.yml

create a ./definitions.py file similar to that shown below

 

## Related

- [2210_AggFSyn](https://github.com/cefect/2210_AggFSyn): preliminary work (for ICFM9)

- [2112_Agg](https://github.com/cefect/2112_Agg): original aggregation study (from which the curves work spanwed). This is now limited to grid aggregation.

- [figueiredo2018](https://github.com/cefect/figueiredo2018/tree/cef): damage curve library



### definitions.py
```
import os, sys

src_dir = os.path.dirname(os.path.abspath(__file__))
src_name = os.path.basename(src_dir)

# default working directory
wrk_dir = r'l:\10_IO\2307_funcAgg'

lib_dir = os.path.join(wrk_dir, 'lib')

temp_dir = os.path.join(os.path.expanduser('~'), 'py', 'temp', src_name)
 

# logging configuration file
logcfg_file = os.path.join(src_dir, 'logger.conf')
 
#OSM data
osm_pbf_basedir=os.path.join(wrk_dir, r'ins\osm_20230725')

osm_pbf_data = { #relative
    'BGD':'bangladesh-latest.osm.pbf',
    'AUS':'australia-latest.osm.pbf',
    'DEU':'germany-latest.osm.pbf',
    'CAN':'canada-latest.osm.pbf',
    'ZAF':'south-africa-latest.osm.pbf',    
    }

osm_pbf_data = {k:os.path.join(osm_pbf_basedir, v) for k,v in osm_pbf_data.items()}

osm_cache_dir = os.path.join(osm_pbf_basedir, 'cache')

#add osmium to path
os.environ['PATH'] += r";l:\09_REPOS\02_JOBS\2307_funcAgg\env\funcAgg3\Library\bin"
```