'''
Created on Mar. 28, 2023

@author: cefect

plot building grouped stats
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
    usetex = False #need to use word anyway
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
from matplotlib import gridspec
 
#set teh styles
plt.style.use('default')
font_size=6
dpi=300
transparent=True

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
           
#===============================================================================
# draft
#===============================================================================
elif env_type=='draft':
    set_doc_style() 
 
         
#===============================================================================
# presentation style    
#===============================================================================
elif env_type=='present': 
    transparent=False 
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
import os, math, hashlib, string
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
from _05depths._03_gstats import get_a03_gstats_1x

#===============================================================================
# data
#===============================================================================
 
def plot_gstats(
        dx_raw=None,
         
        country_key='deu',
        xcoln='avg', 
        ycoln='stddevpop',
        #haz_key='f500_fluvial',
        out_dir=None,
        #figsize=None,
        figsize_scaler=2,
        #min_wet_frac=0.95,
        min_bldg_cnt=2,
 
        samp_frac=0.0001, #
        dev=False,
 
        cmap='viridis',
        
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
     
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='gstats')
    
 
    if dev:
        samp_frac=1.0
        
    ylab_d = haz_label_d.copy()
    
 
    #===========================================================================
    # load data--------
    #===========================================================================
    if dx_raw is None:
        #load from postgres view damage.rl_mean_grid_{country_key}_{haz_key}_wd and do some cleaning
        dx_raw = get_a03_gstats_1x(country_key=country_key, log=log, use_cache=True, dev=dev)
        """
        samp_size = int(1e4)
        dx_raw.groupby(['grid_size']).sample(samp_size).to_pickle(os.path.join(out_dir, f'dev_dx_raw_{samp_size}_{today_str}.pkl'))
        """
    
 
    #===========================================================================
    # filter data
    #===========================================================================
    dx1 = dx_raw.xs(country_key, level='country_key')#.xs(haz_key, level='haz_key', axis=1).
    
    #===========================================================================
    # NO.... wet_cnt is based on the 500-year
    # #need some wets
    # bx = dx1.index.get_level_values('wet_cnt')>0
    # log.info(f'selected {bx.sum()}/{len(bx)} w/ some exposed buildings')
    # dx2 = dx1.loc[bx, :]
    #===========================================================================
    
    
 
    #building count
    bx = dx1.index.get_level_values('bldg_cnt')>=min_bldg_cnt
    log.info(f'selected {bx.sum():,}/{len(bx):,} w/ min_bldg_cnt={min_bldg_cnt}')
    dx2=dx1[bx]
    
    #by wet_frac. No... this is specific to each hazard scenario 
 #==============================================================================
 #    dx1.index = pd.MultiIndex.from_frame(
 #                    dx1.index.to_frame().join(
 #                        dx1.xs('f500_fluvial', level='haz_key', axis=1)['wet_cnt'].astype(int)
 #                        ))
 # 
 #    #mdf['wet_frac'] = mdf['wet_cnt'] / mdf['bldg_cnt']
 #    
 #    dx2 = filter_rl_dx_minWetFrac(dx1[bx], min_wet_frac=min_wet_frac, log=log)
 #==============================================================================
 
    #stack
    dx3 = dx2.stack(level='haz_key')#.drop('wet_cnt', axis=1)
    
    #drop zeros
    bx = dx3['wet_cnt']>0
    log.info(f'selected {bx.sum()}/{len(bx)} w/ some exposed buildings')
    dx4 = dx3.loc[bx, :].copy()
    
    #add wet frac
    dx4.loc[:, 'wet_frac'] = dx4['wet_cnt']/dx4.index.get_level_values('bldg_cnt')
    
    """
    view(dx.head(100))
    view
    """
 
    dx=dx4.reorder_levels(['haz_key', 'grid_size', 'i', 'j', 'bldg_cnt', 'null_cnt']).sort_index(sort_remaining=True)
    mdex = dx.index
    log.info(f' filtered to {dx.shape}')
    #===========================================================================
    # setup indexers
    #===========================================================================
    if xcoln=='avg':
        xlims = (0, 1000)
    else:
        xlims = (0, max(dx[xcoln])*1.05)
    ylims = (0, max(dx[ycoln]))
            
    keys_d = {'row':'haz_key',  'col':'grid_size'}
    kl = list(keys_d.values())     
 
    
 
    binx = np.linspace(0, xlims[1], 21)
    biny = np.linspace(0, ylims[1], 21)
    #===========================================================================
    # setup figure-------
    #===========================================================================
    row_keys, col_keys = [mdex.unique(e).tolist() for e in keys_d.values()]
 
 
    nc, nr = len(col_keys), len(row_keys)+1
    
    fig = plt.figure(
            #layout='tight' #no effect
            #layout='constrained', #breaks the histograms for some reason
                     #figsize=(nc*figsize_scaler, nr*figsize_scaler),
                     )
    
 
    
    
    gsM = gridspec.GridSpec(nr, nc , 
                               height_ratios=[1 for _ in range(nr-1)]+[0.2], #even then the color bar
                               #width_ratios=[4,1],
                               figure=fig,
                               wspace=0.1, hspace=0.2)
            
    ax_d = {r:dict() for r in row_keys} #container for axis
    i_d = {e:i for i, e in enumerate(row_keys)}
    j_d = {e:i for i, e in enumerate(col_keys)}
    
    #color_d = _get_cmap(color_keys, name='viridis')
    
 
    log.info(f'on \n    cols:{col_keys}\n    rows:{row_keys}')
    
    #===========================================================================
    # loop and plot-----
    #===========================================================================
 
    meta_lib=dict()
    ax_main_previous=None
 
    for (row_key, col_key), gdx0 in dx.loc[:, (xcoln, ycoln)].groupby(kl[:2]):
        
        i, j = i_d[row_key], j_d[col_key]
        
        log.info(f'%s:{row_key} ({i}) x %s:{col_key} ({j})'%(keys_d['row'], keys_d['col']))
         
        #=======================================================================
        # setup subfigure
        #=======================================================================
        #subfig = subfig_d[row_key][col_key]
        
        # Define the gridspec (2x2)
        gs_i = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gsM[i,j],
                                                height_ratios=[1,4], width_ratios=[4,1],
                                                wspace=0.0, hspace=0.0                                                
                                                )
        
        # Scatter plot (lower left) 
        ax_main = fig.add_subplot(gs_i[1, 0], sharex=ax_main_previous, sharey=ax_main_previous)        
        #=======================================================================
        # prep data
        #=======================================================================
        gdx0 = gdx0.droplevel(kl[:2]).dropna()
 
        xar, yar = gdx0[xcoln], gdx0[ycoln]
        #===================================================================
        # #plot density scatter-------
        #===================================================================
        
        #geet a sample of hte data
        df_sample = gdx0.sample(min(int(len(dx)*samp_frac), len(gdx0)))
        
        log.info(f'    w/ {gdx0.size} and sample {df_sample.size}')
    
        
        #ax.plot(df['bldg_mean'], color='black', alpha=0.3,   marker='.', linestyle='none', markersize=3,label='building')
        #as density
        x,y = df_sample[xcoln].values, df_sample[ycoln].values
        
        """cant log transform zeros?"""
        #xy = np.vstack([np.log(x),np.log(y)]) #log transformed
        xy = np.vstack([x,y])
        
        """need to compute this for each set... should have some common color scale.. but the values dont really matter"""
        pdf = gaussian_kde(xy)
        z = pdf(xy) #Evaluate the estimated pdf on a set of points.
        
        # Sort the points by density, so that the densest points are plotted last
        indexer = z.argsort()
        x, y, z = x[indexer], y[indexer], z[indexer]
        cax = ax_main.scatter(x, y, c=z, s=5, cmap=cmap, alpha=1.0, marker='.', edgecolors='none', rasterized=True)
        
        #ax_main.set_aspect('equal')
        ax_main.set_xlim(xlims)
        ax_main.set_ylim(ylims)
        #=======================================================================
        # right (y) histogram
        #=======================================================================
        hist_kwargs = dict(color='black', alpha=0.5)
        # Histogram of y values (lower right)
        ax_right = fig.add_subplot(gs_i[1,1], 
                                      sharey=ax_main, #cant use this with turning off/on the histograms
                                      )
        
        ax_right.hist(yar, orientation='horizontal',bins=biny, **hist_kwargs)
        
        ax_right.axis('off')
 
        
        #=======================================================================
        # top (x) histogram
        #=======================================================================
        # Histogram of x values (top)
        ax_top = fig.add_subplot(gs_i[0,0],sharex=ax_main)
        
        ax_top.hist(xar, bins=binx, **hist_kwargs)
        
        ax_top.axis('off')
 
        #===================================================================
        # text-------
        #===================================================================
 
        xmean, ymean= gdx0.mean()
        #tstr = f'count: {len(gdx0)}\n'
        tstr ='$\overline{\sigma}$: %.2f'%ymean
        #tstr+='\n$\overline{\mu}$: %.2f'%xmean
        
 
        xq, yq = gdx0.quantile(0.75)
        tstr+='\n$Q_{0.75}[\sigma]$: %.2f'%yq
        #tstr+='\n$Q_{0.99}[\mu]$: %.2f'%xq
        
        tstr+='\nn: %.2e'%len(gdx0)
        
        #tstr+=f'\n{row_key}.{col_key}'
 
         
        coords = (0.9, 0.9)         
        ax_main.text(*coords, tstr, size=6,
                            transform=ax_main.transAxes, va='top', ha='right', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
         
        #=======================================================================
        # panel label
        #=======================================================================
        letter=list(string.ascii_lowercase)[j]
        ax_main.text(0.05, 0.95, 
                '(%s%s)'%(letter, i), 
                transform=ax_main.transAxes, va='top', ha='left',
                size=matplotlib.rcParams['axes.titlesize'],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                )
 
 
        #=======================================================================
        # wrap
        #=======================================================================
        ax_d[row_key][col_key]=ax_main
        ax_main_previous=ax_main

        
 
    """
    plt.show()
    """
 
    rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
    #===========================================================================
    # post----
    #===========================================================================    
    for row_key, col_key, ax in rc_ax_iter:
        pass
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
            ax.set_ylabel(ylab_d[row_key])
 
            
    #===========================================================================
    # #macro labelling
    #===========================================================================
    lab_d = {
        'stddevpop':r'standard deviation of child depths in cm ($\sigma$)',         #($\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$)
        'avg':r'child depths mean in cm ($\mu$)',
        'wet_frac':'fraction of child buildings flooded'
        }
    
    #plt.subplots_adjust(left=1.0)
    macro_ax = fig.add_subplot(gsM[:nr-1, :], frame_on=False)
    _hide_ax(macro_ax) 
    macro_ax.set_ylabel(lab_d[ycoln],labelpad=20, size=font_size+2)
    macro_ax.set_xlabel(lab_d[xcoln], size=font_size+2) # ($\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$
    
    """doesnt help
    fig.tight_layout()"""
    
    #===========================================================================
    # #add colorbar
    #===========================================================================
    #create the axis
    #fig.subplots_adjust(bottom=0.25)
    #leg_ax = fig.add_axes([0.07, 0, 0.9, 0.08], frameon=True)
    leg_ax = fig.add_subplot(gsM[nr-1, :]) #span bottom
    leg_ax.set_visible(False)    
    
    #color scaling from density values
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(min(z), max(z)), cmap=cmap)
    
    #add the colorbar
    cbar = fig.colorbar(sm,
                        #cax=leg_ax,
                     ax=leg_ax,  # steal space from here (couldnt get cax to work)
                     extend='both', #pointed ends
                     format = matplotlib.ticker.FuncFormatter(lambda x, p:'%.1e' % x),
                     label='gaussian kernel-density estimate', 
                     orientation='horizontal',
                     location='bottom',
                     #pad=0.4,
                     fraction=0.3,
                     aspect=60, #make skinny
                     #shrink=0.5,makes narrower
 
                     )
 
    #===========================================================================
    # tighten up
    #===========================================================================
    fig.subplots_adjust(top=0.99, right=0.99, left=0.1, bottom=0.07)
    
    #===========================================================================
    # fig.patch.set_linewidth(10)  # set the line width of the frame
    # fig.patch.set_edgecolor('cornflowerblue')  # set the color of the frame
    #===========================================================================
 
        
    #===========================================================================
    # write-------
    #===========================================================================
    
    
    ofp = os.path.join(out_dir, f'gstats_{xcoln}-{ycoln}_{env_type}_{len(col_keys)}x{len(row_keys)}_{today_str}.svg')
    fig.savefig(ofp, dpi = dpi,   transparent=True, 
                #edgecolor=fig.get_edgecolor(),
                )
    
    plt.close('all')
    
    
 
    meta_d = {
                    'tdelta':'%.2f secs'%(datetime.now()-start).total_seconds(),
                    #'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    #'file_GB':get_directory_size(out_dir),
                    'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
     
    log.info(meta_d)
    
    log.info(f'wrote to \n    {ofp}')
     
    return ofp
 
 
    
if __name__=='__main__':
    
 
    plot_gstats(dev=False, 
                samp_frac=0.05,
                #xcoln='wet_frac', #not a strong relation
                #dx_raw = pd.read_pickle(r'l:\\10_IO\\2307_funcAgg\\outs\\depths\\da\\20230926\\dev_dx_raw_10000_20230926.pkl')
                )

    
 
    
    print('finished ')
    
    
    
    
