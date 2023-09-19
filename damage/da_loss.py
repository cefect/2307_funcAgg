'''
Created on Mar. 28, 2023

@author: cefect

plot relative loss of bldgs vs. agg
'''

#===============================================================================
# PLOT ENV------
#===============================================================================

#===============================================================================
# setup matplotlib----------
#===============================================================================
env_type = 'draft'
cm = 1 / 2.54

if env_type == 'journal': 
    usetex = True
elif env_type == 'draft':
    usetex = False
elif env_type == 'present':
    usetex = False
else:
    raise KeyError(env_type)

 
 
  
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')
font_size=8
def set_doc_style():
 
    
    matplotlib.rc('font', **{'family' : 'serif','weight' : 'normal','size': font_size})
    matplotlib.rc('legend',fontsize=font_size)
     
    for k,v in {
        'axes.titlesize':font_size,
        'axes.labelsize':font_size,
        'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,
        'figure.titlesize':font_size+2,
        'figure.autolayout':False,
        'figure.figsize':(17.7*cm,18*cm),#typical full-page textsize for Copernicus (with 4cm for caption)
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v

#===============================================================================
# journal style
#===============================================================================
if env_type=='journal':
    set_doc_style() 
 
    env_kwargs=dict(
        output_format='pdf',add_stamp=False,add_subfigLabel=True,transparent=True
        )            
#===============================================================================
# draft
#===============================================================================
elif env_type=='draft':
    set_doc_style() 
 
    env_kwargs=dict(
        output_format='svg',add_stamp=True,add_subfigLabel=True,transparent=True
        )          
#===============================================================================
# presentation style    
#===============================================================================
elif env_type=='present': 
 
    env_kwargs=dict(
        output_format='svg',add_stamp=True,add_subfigLabel=False,transparent=False
        )   
 
    font_size=12
 
    matplotlib.rc('font', **{'family' : 'sans-serif','sans-serif':'Tahoma','weight' : 'normal','size':font_size})
     
     
    for k,v in {
        'axes.titlesize':font_size+2,
        'axes.labelsize':font_size+2,
        'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,
        'figure.titlesize':font_size+4,
        'figure.autolayout':False,
        'figure.figsize':(34*cm,19*cm), #GFZ template slide size
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)


#===============================================================================
# imports---------
#===============================================================================
import os, math
import pandas as pd
idx = pd.IndexSlice
import numpy as np


from datetime import datetime


from coms import init_log, today_str, view
from coms_da import get_matrix_fig, _get_cmap, _hide_ax
from funcMetrics.func_prep import get_funcLib

from definitions import wrk_dir, clean_names_d, haz_label_d

#===============================================================================
# data
#===============================================================================
 


 
def plot_rl_agg_v_bldg(
        pick_fp=r'l:\10_IO\2307_funcAgg\outs\damage\03_mean\deu\f500_fluvial\rl_mean_deu_f500_fluvial_345824_20230919.pkl',
        country_key='deu',
        haz_key=f'f500_fluvial',
        out_dir=None,
        figsize=None,
        min_wet_frac=0.9,
        ):
    
    """plot relative loss from grid centroids and building means    
    
    
    
    Params
    -------------
    pick_fp: str
        filepath to pd.DataFrame. 
        pre-filtered so we only have grids with some centroid and building loss
        see _03_rl_mean_bldg.run_extract_haz (only 1 haz_key)
    """
    #===========================================================================
    # defaults
    #===========================================================================
    raise IOError('something is wrong with the data. wet_cnt/dry_cnt is bad. grid_cent loss labels seem to be mixed')
  
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'funcMetrics', 'da', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
     
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='rl_agg')
    
    if figsize is None:
        if env_type=='present':
            figsize=(34*cm,10*cm)
            
 
    
    #===========================================================================
    # load data
    #===========================================================================
    dx_raw = pd.read_pickle(pick_fp).xs(country_key, level='country_key')
    dx_raw=dx_raw.sample(int(1e4))
    """
    view(dx_raw.head(100))
    view(dx_raw.head(100).stack())
    view(dx.head(100))
    dx['grid_size']=999
    dx.set_index('grid_size', append=True, inplace=True)
    
    """
    
    dx1 = dx_raw.stack()
    mdex = dx1.index
    
    assert len(mdex.unique('haz_key'))==1
    assert haz_key in mdex.unique('haz_key')
    
    #===========================================================================
    # filter data
    #===========================================================================
    """want to exclude partials"""
    mdf = mdex.to_frame().reset_index(drop=True)
    mdf['wet_frac'] = mdf['wet_cnt']/mdf['bldg_cnt']
    
    """waiting for this column to be fixed"""
    #assert mdf['wet_frac'].max()<=1.0
    bx = mdf['wet_frac']>(min_wet_frac+1)
    
    log.info(f'selected {bx.sum()}/{len(bx)} w/ min_wet_frac={min_wet_frac}')
    
    dx2 = dx1.loc[bx.values, :].droplevel(['wet_cnt', 'bldg_cnt'])
    
    mdex = dx2.index
    #===========================================================================
    # setup indexers
    #===========================================================================        
    keys_d = {'row':'df_id',  'col':'grid_size', 'color':'haz_key'}
    kl = list(keys_d.values())     
    log.info(f' loaded {len(dx1)}')
    
    #===========================================================================
    # setup figure
    #===========================================================================
    row_keys, col_keys, color_keys = [mdex.unique(e).tolist() for e in keys_d.values()]
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=False, 
                               constrained_layout=False,
                               sharex=True, sharey=True, add_subfigLabel=False, figsize=figsize)
    
    rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
    
    color_d = _get_cmap(color_keys, name='viridis')
    
    #add tinws
    #===========================================================================
    # twin_ax_d=dict()
    # for row_key, col_key, ax in rc_ax_iter:
    #     ax.set_xlim((0, 6.0))
    #     
    #     if not row_key in twin_ax_d:
    #         twin_ax_d[row_key] = dict()
    #         
    #     if not col_key in twin_ax_d[row_key]:
    #         ax2 = ax.twinx()
    #         ax2.set_ylim(0, 0.12)
    #         twin_ax_d[row_key][col_key] = ax2
    #===========================================================================
        
    
    #===========================================================================
    # loop and plot
    #===========================================================================
    #letter=list(string.ascii_lowercase)[j]
    
 
    for (row_key, col_key), gdx0 in dx2.groupby(kl[:2]):
        log.info(f'df_id:{row_key} x grid_size:{col_key}')
        ax = ax_d[row_key][col_key]
        
        gdx0 = gdx0.droplevel(kl[:2])
        #=======================================================================
        # plot function 
        #=======================================================================
        #=======================================================================
        # wd_rl_df = gserx0.xs(color_keys[0], level=keys_d['color']).reset_index().drop(serx.name, axis=1)
        # xar, yar = wd_rl_df['wd'], wd_rl_df['rl']
        # 
        # ax.plot(xar, yar, color='black', marker='o', linestyle='dashed', alpha=0.5, markersize=3,fillstyle='none')
        #=======================================================================
        
 
        
        for color_key, gdx1 in gdx0.groupby(kl[2]):
            """setup for multiple hazards... but this is too busy"""
            #c=color_d[color_key]
            c='black'
            
            #prep data
            df= gdx1.droplevel(kl[2]).reset_index('grid_wd').reset_index(drop=True).set_index('grid_wd')
            
            #plot grid cent
            ax.plot(df['grid_cent'], color='red', alpha=0.3, marker='o', linestyle='none', markersize=5, fillstyle='none',
                    label='grid')
            
            #plot bldg_mean
            ax.plot(df['bldg_mean'], color='black', alpha=0.3,   marker='.', linestyle='none', markersize=3,
                    label='building')
            
    #===========================================================================
    # get some function meta
    #===========================================================================
    #get model-dfid lookup
 #==============================================================================
 #    fserx = get_funcLib()
 #    meta_df = fserx.index.to_frame().reset_index(drop=True).loc[:, ['df_id', 'model_id', 'abbreviation']
 #                                      ].drop_duplicates().set_index('df_id') 
 # 
 #    for i, row in meta_df.iterrows():
 #        if not row['model_id'] in clean_names_d:
 #            clean_names_d[row['model_id']] = row['abbreviation']
 #==============================================================================
    #===========================================================================
    # post
    #===========================================================================    
    for row_key, col_key, ax in rc_ax_iter:
 
        #ax.grid()
        
        # first row
        if row_key == row_keys[0]:
            ax.set_title(f'{col_key}m grid')
        
        # last row
        if row_key == row_keys[-1]: 
            pass 
            # ax.set_xlabel(f'WSH (cm)')
             
        # first col
        if col_key == col_keys[0]: 
            ax.set_ylabel(f'function {row_key}')
             
        #=======================================================================
        # #last col
        # if col_key==col_keys[-1]:
        #     #ax2.set_ylabel('A*P')
        #     pass
        # else:
        #     ax2.set_yticklabels([])
        #=======================================================================
            
            
    #macro labelling
    #plt.subplots_adjust(left=1.0)
    macro_ax = fig.add_subplot(111, frame_on=False)
    _hide_ax(macro_ax) 
    macro_ax.set_ylabel(f'relative loss', labelpad=20)
    macro_ax.set_xlabel(f'WSH (m)')
    
    #===========================================================================
    # macro_ax2 = macro_ax.twinx()
    # _hide_ax(macro_ax2)
    # macro_ax2.set_ylabel('A*P', rotation=-90, labelpad=20)
    #===========================================================================
    """
    plt.show()
    """
            
        
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'rl_{env_type}_{len(col_keys)}x{len(row_keys)}_{today_str}.svg')
    fig.savefig(ofp, dpi = 300,   transparent=True)
    
    plt.close('all')
    
    
 
    #===========================================================================
    # meta_d = {
    #                 'tdelta':(datetime.now()-start).total_seconds(),
    #                 'RAM_GB':psutil.virtual_memory () [3]/1000000000,
    #                 #'file_GB':get_directory_size(out_dir),
    #                 'output_MB':os.path.getsize(ofp)/(1024**2)
    #                 }
    # 
    # log.info(meta_d)
    #===========================================================================
    
    log.info(f'wrote to \n    {ofp}')
     
    return ofp
        
 
   
if __name__=='__main__':
    


    #plot_dfunc_matrix()
    
    #plot_gradient_matrix(fp=r'l:\10_IO\2307_funcAgg\outs\funcs\01_deriv\derivs_3266_20230907.pkl')
    
    plot_rl_agg_v_bldg()

    
 
    
    print('finished ')
    
    
    
    
