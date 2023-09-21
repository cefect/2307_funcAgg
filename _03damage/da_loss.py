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

import psycopg2
from sqlalchemy import create_engine, URL

from scipy.stats import gaussian_kde

from definitions import wrk_dir, clean_names_d, haz_label_d, postgres_d

from coms import init_log, today_str, view
from coms_da import get_matrix_fig, _get_cmap, _hide_ax
from funcMetrics.func_prep import get_funcLib

from _02agg.coms_agg import (
    get_conn_str,   
    )

from _03damage._05_mean_bins import get_grid_rl_dx, compute_binned_mean, filter_rl_dx_minWetFrac
 



#===============================================================================
# data
#===============================================================================
 
def plot_rl_raw(
        tableName='rl_deu_grid_0060',
        schema='dev',
        out_dir=None,
        figsize=None,
        conn_str=None,
        limit=None,
 
        ):
    
    """plot simple histograms of RL values (for checking)
    
    
    
 
    """
    #===========================================================================
    # defaults
    #===========================================================================
 
  
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'damage', 'da', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
     
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='rl_raw')
    if conn_str is None: conn_str=get_conn_str(postgres_d)
    
    if 'bldgs' in tableName:
        asset_type='bldg'
    else:
        asset_type='grid'
        
    if asset_type=='grid': 
        keys_l = ['country_key', 'grid_size', 'i', 'j', 'haz_key']
    elif asset_type=='bldg':
        keys_l = ['country_key', 'gid', 'id', 'haz_key']
    
    
    
    
    #===========================================================================
    # download
    #===========================================================================
    conn =  psycopg2.connect(conn_str)
    engine = create_engine('postgresql+psycopg2://', creator=lambda:conn)
    
    #row_cnt=0
    
    """only ~600k rows"""
 
    cmd_str = f"""SELECT * FROM {schema}.{tableName}"""
    if not limit is None:
        cmd_str+=f'\n    LIMIT {limit}'
    log.info(cmd_str)
    df_raw = pd.read_sql(cmd_str, engine, index_col=keys_l)
    
    df_raw.columns.name='df_id'
    serx = df_raw.stack().droplevel([0])
    
    mdex = serx.index
    #===========================================================================
    # setup indexers
    #===========================================================================        
    keys_d = {'row':'df_id',  'col':'haz_key'}
    kl = list(keys_d.values())     
    log.info(f' loaded {len(serx)}')
    
    #===========================================================================
    # setup figure
    #===========================================================================
    row_keys, col_keys = [mdex.unique(e).tolist() for e in keys_d.values()]
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=True, 
                               constrained_layout=False,
                               sharex=True, sharey=True, add_subfigLabel=False, figsize=figsize)
    
    rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
    
    #color_d = _get_cmap(color_keys, name='viridis')
    
    
    #===========================================================================
    # loop and plot---------
    #===========================================================================
 
    for (row_key, col_key), gdx0 in serx.groupby(kl[:2]):
        log.info(f'df_id:{row_key} x {col_key} w/ ({len(gdx0)})')
        ax = ax_d[row_key][col_key]
        
        ar = gdx0.droplevel(kl[:2]).reset_index(drop=True).values
        
        #ax.set_xlim((0, 500))
        #=======================================================================
        # post hist
        #=======================================================================
        ax.hist(ar, bins=10, color='black', range=(0, 100), alpha=0.8)
        
        #=======================================================================
        # add text
        #=======================================================================
        zero_cnt = (ar==0).sum()
        tstr = f'cnt: {len(ar)}\nmean:{ar.mean():.1f}\nzeros:{zero_cnt}'
        ax.text(0.95, 0.05, tstr, 
                            transform=ax.transAxes, va='bottom', ha='right', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
        
        
    #===========================================================================
    # post------
    #===========================================================================
    fig.suptitle(tableName)
    
  
    for row_key, col_key, ax in rc_ax_iter:
 
        #ax.grid()
        
        # first row
        #=======================================================================
        # if row_key == row_keys[0]:
        #     ax.set_title(f'{col_key}m grid')
        #=======================================================================
        
        # last row
        if row_key == row_keys[-1]: 
             
            ax.set_xlabel(f'WSH (cm)')
             
        # first col
        if col_key == col_keys[0]: 
            ax.set_ylabel(f'function {row_key}')
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'rl_raw_{env_type}_{tableName}_{len(col_keys)}x{len(row_keys)}_{today_str}.svg')
    fig.savefig(ofp, dpi = 300,   transparent=True)
    
    plt.close('all')
    
    log.info(f'wrote to \n    {ofp}')
    return ofp
        
        
    
 

 

 

def plot_rl_agg_v_bldg(
        dx_raw=None,
        country_key='deu',
        haz_key='f500_fluvial',
        out_dir=None,
        figsize=None,
        min_wet_frac=0.95,
        samp_frac=0.1,
        dev=False,
        ):
    
    """plot relative loss from grid centroids and building means    
    
    
    
    Params
    -------------
 
    """
    #raise IOError('looks pretty good. need to filter partials, add running-mean for buildings, add colorscale')
    
    #===========================================================================
    # defaults
    #===========================================================================
 
  
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'damage', 'da', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
     
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='rl_agg')
    
    if figsize is None:
        if env_type=='present':
            figsize=(34*cm,10*cm)
            
 
    
    #===========================================================================
    # load data
    #===========================================================================
    if dx_raw is None:
        #load from postgres view damage.rl_mean_grid_{country_key}_{haz_key}_wd and do some cleaning
        dx_raw = get_grid_rl_dx(country_key, haz_key, log=log, use_cache=True, dev=dev)
    
    if not dev:
        dx_raw=dx_raw.sample(int(len(dx_raw)*samp_frac))
 
 
    """
    view(dx_raw.head(100))
    view(dx1.head(100))
 
    
    """
    
    dx1 = dx_raw.stack().droplevel('country_key')
    mdex = dx1.index
    
    assert len(mdex.unique('haz_key'))==1
    assert haz_key in mdex.unique('haz_key')
    
    #===========================================================================
    # filter data---------
    #===========================================================================
    """want to exclude partials
    
    view(mdf.head(100))
 
    """
    dx2 = filter_rl_dx_minWetFrac(dx1, min_wet_frac=min_wet_frac, log=log)
    
 
    # get binned means 
    mean_bin_dx = compute_binned_mean(dx2)
    raise IOError('stopped here')
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
    
    #color_d = _get_cmap(color_keys, name='viridis')
    
 
        
    
    #===========================================================================
    # loop and plot-----
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
            

            
            #===================================================================
            # #plot bldg_mean
            #===================================================================
            #ax.plot(df['bldg_mean'], color='black', alpha=0.3,   marker='.', linestyle='none', markersize=3,label='building')
            #as density
            x,y = df.index.values, df['bldg_mean'].values
            xy = np.vstack([x,y])
            z = gaussian_kde(xy)(xy)
            
            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            cax = ax.scatter(x, y, c=z, s=20, label='building')
            
            """
            plt.show()
            """
            
            #===================================================================
            # #plot grid cent
            #===================================================================
            ax.plot(df['grid_cent'], color='black', alpha=0.1, marker='.', linestyle='none', markersize=1, fillstyle='none',
                    label='grid')
            
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
    # post----
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
            ax.set_ylabel(f'function \'{row_key}\'')
             
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
    ofp = os.path.join(out_dir, f'rl_{env_type}_{len(col_keys)}x{len(row_keys)}_{today_str}.png')
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
    


    plot_rl_agg_v_bldg(country_key='deu', haz_key='f500_fluvial', dev=False)
    
    #plot_rl_raw( tableName='rl_deu_grid_0060', schema='dev')
   # plot_rl_raw( tableName='rl_deu_bldgs', schema='dev')

    
 
    
    print('finished ')
    
    
    
    
