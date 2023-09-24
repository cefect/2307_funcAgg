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
font_size=6
dpi=300
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
        'figure.figsize':(18*cm,18*cm),#typical full-page textsize for Copernicus and wiley (with 4cm for caption)
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v

#===============================================================================
# journal style
#===============================================================================
if env_type=='journal':
    set_doc_style() 
    dpi=1000
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
import scipy.stats

from definitions import wrk_dir, clean_names_d, haz_label_d, postgres_d

from coms import init_log, today_str, view
from coms_da import get_matrix_fig, _get_cmap, _hide_ax
from funcMetrics.func_prep import get_funcLib
from funcMetrics.coms_fm import (
    slice_serx, force_max_depth, force_zero_zero, force_monotonic, force_and_slice
    )


from palettable.colorbrewer.sequential import PuBu_9, RdPu_3

from _03damage._05_mean_bins import filter_rl_dx_minWetFrac
from _05depths._02_views import get_grid_wd_dx 

#===============================================================================
# data
#===============================================================================
 
 

def plot_rlMean_v_gCentroid(
        dx_raw=None,
         
        country_key='deu',
        haz_key='f500_fluvial',
        out_dir=None,
        figsize=None,
        min_wet_frac=0.95,
 
        samp_frac=0.0001, #
        dev=False,
        ylab_d = clean_names_d,
        cmap='PuRd_r',
        ):
    
 
    
    """grid centroids vs. child building means 
    
    not sure we need thsi for hte paper... but a nice check
    
    
    Params
    -------------
 
        
    samp_frac: float
        for reducing the data sent to gaussian_kde
        relative to size of complete data set (not the gruops)
 
    """
 
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
  
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'depths', 'da', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
     
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='bmeans')
    
    if figsize is None:
        if env_type=='present':
            figsize=(34*cm,10*cm)
            
 
    
    #===========================================================================
    # load data--------
    #===========================================================================
    if dx_raw is None:
        #load from postgres view damage.rl_mean_grid_{country_key}_{haz_key}_wd and do some cleaning
        dx_raw = get_grid_wd_dx(country_key=country_key, haz_key=haz_key, log=log, use_cache=True, dev=dev)
    
    """no.. better to just sample the kde
    if not dev:
        dx_raw=dx_raw.sample(int(len(dx_raw)*samp_frac))"""
 
 
    """
    view(dx_raw.head(100))
    view(dx1.head(100))
 
    
    """
    
    dx1 = dx_raw.droplevel('country_key')
    mdex = dx1.index
    
    assert len(mdex.unique('haz_key'))==1
    assert haz_key in mdex.unique('haz_key')
    
    #===========================================================================
    # filter data
    #===========================================================================
    """want to exclude partials
    
    view(dx1.head(100))
    
    mdex.unique('df_id')
 
    """
 
    
    dx2 = filter_rl_dx_minWetFrac(dx1, min_wet_frac=min_wet_frac, log=log)
    
 
 
    #===========================================================================
    # setup indexers
    #===========================================================================        
    keys_d = {'row':'haz_key',  'col':'grid_size'}
    kl = list(keys_d.values())     
    log.info(f' loaded {len(dx1)}')
    
 
    
    #===========================================================================
    # setup figure
    #===========================================================================
    row_keys, col_keys = [mdex.unique(e).tolist() for e in keys_d.values()]
 
    
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=False, 
                               constrained_layout=False, #needs to be unconstrainted for over label to work
                               figsize_scaler=3,
                               sharex=True, sharey='row', add_subfigLabel=True, figsize=figsize)
    
    rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
    
    #color_d = _get_cmap(color_keys, name='viridis')
    
 
    
    
    #===========================================================================
    # loop and plot-----
    #===========================================================================
    #letter=list(string.ascii_lowercase)[j]
    meta_lib=dict()
 
    for (row_key, col_key), gdx0 in dx2.groupby(kl[:2]):
        log.info(f'df_id:{row_key} x grid_size:{col_key}')
        ax = ax_d[row_key][col_key]
        
        gdx0 = gdx0.droplevel(kl[:2]).dropna()
        
        #assert (gdx0==0).sum().sum()==0
        
 
        xar, yar = gdx0['bmean_wd'], gdx0['grid_wd']
        #===================================================================
        # #plot bldg_mean scatter
        #===================================================================
        
        #geet a sample of hte data
        df_sample = gdx0.sample(min(int(len(dx2)*samp_frac), len(gdx0)))
        
        log.info(f'    w/ {gdx0.size} and sample {df_sample.size}')
        
        #ax.plot(df['bldg_mean'], color='black', alpha=0.3,   marker='.', linestyle='none', markersize=3,label='building')
        #as density
        x,y = df_sample['bmean_wd'].values, df_sample['grid_wd'].values
        xy = np.vstack([x,y])
        
        """need to compute this for each set... should have some common color scale.. but the values dont really matter"""
        pdf = gaussian_kde(xy)
        z = pdf(xy) #Evaluate the estimated pdf on a set of points.
        
        # Sort the points by density, so that the densest points are plotted last
        indexer = z.argsort()
        x, y, z = x[indexer], y[indexer], z[indexer]
        cax = ax.scatter(x, y, c=z, s=5, cmap=cmap, alpha=0.3, marker='.', edgecolors='none', rasterized=True)
        
                #===================================================================
        # plot 1:1
        #===================================================================
        c = [0,gdx0.max().max()]
        ax.plot(c,c, color='black', linestyle='dashed', linewidth=0.5)
        
 
        #=======================================================================
        # linear regression
        #=======================================================================
 
        
        #regression
        lm = scipy.stats.linregress(xar, yar)
         
        predict = lambda x:np.array([lm.slope*xi + lm.intercept for xi in x])            
        ax.plot(np.array(c), predict(np.array(c)), color='red', linestyle='solid', label='regression', marker=None)
         
        print({'rvalue':lm.rvalue, 'slope':lm.slope, 'intercept':lm.intercept})
 
            
        #===================================================================
        # text-------
        #===================================================================
 
        bmean, gmean = gdx0.mean()
        #tstr = f'count: {len(gdx0)}\n'
        tstr ='$\overline{\overline{RL_{bldg,j}}}$: %.2f'%bmean
        tstr+='\n$\overline{RL_{grid,j}}$: %.2f'%gmean
        
        rmse = np.sqrt(np.mean((xar - yar)**2))
        tstr+='\nRMSE: %.2f'%rmse
        
        bias = gmean/bmean
        tstr+='\nbias: %.2f'%(bias)
        
        tstr+='\nr: %.2f'%(lm.rvalue)
        
        
        #tstr+='\n$\overline{wd}$: %.2f'%(gdx0.index.get_level_values('grid_wd').values.mean())
        
        coords = (0.8, 0.05)
 
 
        
        ax.text(*coords, tstr, size=6,
                            transform=ax.transAxes, va='bottom', ha='center', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
        
        #=======================================================================
        # meta
        #=======================================================================
        if not row_key in meta_lib: meta_lib[row_key] = dict()
        meta_lib[row_key][col_key] = {'bldg_pop_mean':bmean, 'grid_pop_mean':gmean, 'bias':bias}
        
 
        
 
 
    #===========================================================================
    # post----
    #===========================================================================    
    for row_key, col_key, ax in rc_ax_iter:
        pass
        #ax.grid()
        
        # first row
#===============================================================================
#         if row_key == row_keys[0]:
#             ax.set_xlim(0,1000)
#             ax.set_title(f'{col_key}m grid')
#             
#         #second row
#         if row_key==row_keys[1]:
# 
#             if col_key==col_keys[0]:
#                 ax.legend(ncol=1, loc='upper right', frameon=False)
#                 
#         
#         # last row
#         if row_key == row_keys[-1]: 
#             pass 
#             # ax.set_xlabel(f'WSH (cm)')
#              
#         # first col
#         if col_key == col_keys[0]: 
#             if not row_key=='hist':
#                 ax.set_ylabel(ylab_d1[row_key])
#             else:
#                 ax.set_ylabel(f'density of WSH')
#===============================================================================
 
            
    #===========================================================================
    # #macro labelling
    #===========================================================================
    #plt.subplots_adjust(left=1.0)
    macro_ax = fig.add_subplot(111, frame_on=False)
    _hide_ax(macro_ax) 
    macro_ax.set_ylabel(f'grid centroid water depth in cm', labelpad=20)
    macro_ax.set_xlabel(f'child depth mean in cm')
    
    """doesnt help
    fig.tight_layout()"""
    
    #===========================================================================
    # #add colorbar
    #===========================================================================
    #create the axis
    fig.subplots_adjust(bottom=0.25)
    leg_ax = fig.add_axes([0.07, 0, 0.9, 0.08], frameon=True)
    leg_ax.set_visible(False)    
    
    #color scaling from density values
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(min(z), max(z)), cmap=cmap)
    
    #add the colorbar
    cbar = fig.colorbar(sm,
                     ax=leg_ax,  # steal space from here (couldnt get cax to work)
                     extend='both', #pointed ends
                     format = matplotlib.ticker.FuncFormatter(lambda x, p:'%.1e' % x),
                     label='gaussian kernel-density estimate of $\overline{RL_{bldg,j}}$', 
                     orientation='horizontal',
                     fraction=.99,
                     aspect=50, #make skinny
 
                     )
 
    #===========================================================================
    # tighten up
    #===========================================================================
    fig.subplots_adjust(top=0.95, right=0.95)
 
    #===========================================================================
    # meta
    #===========================================================================
    meta_df = pd.concat({k:pd.DataFrame.from_dict(v) for k,v in meta_lib.items()},
                        names=['df_id', 'stat'])
    
    mdf1 = meta_df.stack().unstack(level='stat')
    
    #mdf1['bias'] = mdf1['grid_rl_pop_mean']/mdf1['bldg_rl_pop_mean']
    
    
    
    
    log.info(f'meta w/ {meta_df.shape}\n%s'%mdf1['bias'])
        
    #===========================================================================
    # write-------
    #===========================================================================
    
    
    ofp = os.path.join(out_dir, f'wd_{env_type}_{len(col_keys)}x{len(row_keys)}_{today_str}.svg')
    fig.savefig(ofp, dpi = dpi,   transparent=True)
    
    plt.close('all')
    
    
 
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    #'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    #'file_GB':get_directory_size(out_dir),
                    'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
     
    log.info(meta_d)
    
    log.info(f'wrote to \n    {ofp}')
     
    return ofp
 
 
    
if __name__=='__main__':
    
 
    plot_rlMean_v_gCentroid(dev=False, samp_frac=0.01)

    
 
    
    print('finished ')
    
    
    
    
