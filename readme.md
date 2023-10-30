[![DOI](https://zenodo.org/badge/682497962.svg)](https://zenodo.org/badge/latestdoi/682497962)
# Flood Damage Function Aggregation Processing Pipeline and Data Analysis Scripts
scripts for computation and data analysis of potential errors arising from aggregation of different loss functions



## Use

### figures

Figure 4: Relative losses for three grid aggregations for the 500-year fluvial undefended hazard scenario and the four selected flood
damage functions (Table 1).
`_03damage.da_loss.plot_rl_agg_v_bldg(dfid_l=dfunc_curve_l)`

Figure S2: Relative losses for three grid aggregations and four idealized functions 
`_03damage.da_loss.plot_rl_agg_v_bldg(dfid_l=hill_curve_l)`

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
 
#===============================================================================
# indexing data
#===============================================================================
#country tiles
index_country_fp_d = {
    'BGD':'BGD_tindex_0725.gpkg',
    'AUS':'AUS_tindex_0824.gpkg',
    'ZAF':'ZAF_tindex_0824.gpkg',
    'BRA':'BRA_tindex_0824.gpkg',
    'CAN':'CAN_tindex_0824.gpkg',
    'DEU':'DEU_tindex_0824.gpkg',    
    }

index_country_fp_d = {k:os.path.join(r'l:\10_IO\2307_funcAgg\ins\indexes', v) for k,v in index_country_fp_d.items()}
 
#===============================================================================
# #OSM data
#===============================================================================
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

#===============================================================================
# HAZARD data
#===============================================================================


#hazard tiles
"""these indexes should have a 'location' field with the absolute path to each raster file
the relative path is extracted from this assuming the raster files are in ./raw (relative to the index file)
"""
index_hazard_fp_d ={
    '500_fluvial':r'500_fluvial\tileindex_500_fluvial.gpkg',
    '500_pluvial':r'500_pluvial\tileindex_500_pluvial.gpkg',
    '100_pluvial':r'100_pluvial\tileindex_100_pluvial.gpkg',
    '100_fluvial':r'100_fluvial\tileindex_100_fluvial.gpkg',
    '050_pluvial':r'050_pluvial\tileindex_050_pluvial.gpkg',
    '050_fluvial':r'050_fluvial\tileindex_050_fluvial.gpkg',
    '010_pluvial':r'010_pluvial\tileindex_010_pluvial.gpkg',
    '010_fluvial':r'010_fluvial\tileindex_010_fluvial.gpkg',    
    }

index_hazard_fp_d = {k:os.path.join(r'd:\05_DATA\2307_funcAgg\fathom\global3', v) for k,v in index_hazard_fp_d.items()}


#===============================================================================
# WhiteBoxTools
#===============================================================================
 
sys.path.append(r'l:\09_REPOS\04_FORKS\whitebox-tools\WBT')
```
