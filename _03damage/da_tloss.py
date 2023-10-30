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
env_type = 'present'
cm = 1 / 2.54

if env_type == 'journal': 
    usetex = False
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
present=False
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
# presentation style--------    
#===============================================================================
elif env_type=='present': 
 
    env_kwargs=dict(
        output_format='svg',add_stamp=True,add_subfigLabel=False,transparent=False
        )   
 
    font_size=14
    present=True
 
    matplotlib.rc('font', **{'family' : 'sans-serif','sans-serif':'Tahoma','weight' : 'normal','size':font_size})
     
     
    for k,v in {
        'axes.titlesize':font_size+2,
        'axes.labelsize':font_size+2,
        'xtick.labelsize':font_size,
        'ytick.labelsize':font_size,
        'figure.titlesize':font_size+4,
        'figure.autolayout':False,
        'figure.figsize':(22*cm,18*cm), #GFZ template slide size
        'legend.title_fontsize':'large',
        'text.usetex':usetex,
        }.items():
            matplotlib.rcParams[k] = v
  
print('loaded matplotlib %s'%matplotlib.__version__)


#===============================================================================
# imports---------
#===============================================================================
import os, math, hashlib
import pandas as pd
idx = pd.IndexSlice
import numpy as np


from datetime import datetime

import psycopg2
from sqlalchemy import create_engine, URL

from scipy.stats import gaussian_kde
import scipy.stats

from definitions import wrk_dir, clean_names_d, haz_label_d, postgres_d, temp_dir

from coms import init_log, today_str, view
from coms_da import get_matrix_fig, _get_cmap, _hide_ax
from misc.func_prep import get_funcLib
from funcMetrics.coms_fm import (
    slice_serx, force_max_depth, force_zero_zero, force_monotonic, force_and_slice
    )


from palettable.colorbrewer.sequential import PuBu_9, RdPu_3

from _02agg.coms_agg import (
    get_conn_str,   
    )

from _03damage._03_rl_agg import load_rl_dx
from _03damage._05_mean_bins import compute_binned_mean
 
from _03damage._06_total import get_total_losses

from _05depths._03_gstats import get_a03_gstats_1x


#===============================================================================
# data
#===============================================================================
 
 

def _load_and_filter(dx_raw, country_key, haz_key, min_bldg_cnt, dfid_l, 
                     dev, use_aoi, log,use_cache=True,
                     out_dir=None,):
    """laod and filter relative loss data"""
    
    #===========================================================================
    # defautls
    #===========================================================================
    if out_dir is None:
        out_dir = os.path.join(temp_dir, 'da_loss', '_load_and_filter')
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    
    #===========================================================================
    # cache
    #===========================================================================
    fnstr=f'tl_dx_filter_{country_key}_{haz_key}_{len(dfid_l)}'
    uuid = hashlib.shake_256(
        ' '.join([str(e) for e in [fnstr,  min_bldg_cnt, dfid_l,use_cache, dev, use_aoi]]).encode("utf-8"), 
        usedforsecurity=False).hexdigest(8)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')
    
    if (not os.path.exists(ofp)) or (not use_cache):
        #===========================================================================
        # load loss data
        #===========================================================================
        
        if dx_raw is None:
            
            dx_raw = get_total_losses(country_key=country_key,
                                      log=log,dev=dev,use_cache=use_cache,
                                       use_aoi=use_aoi,)
 
            
        """
        view(dx_raw.head(100))
        """
     
        #===========================================================================
        # slice
        #===========================================================================
        dx1 = dx_raw.xs(haz_key, level='haz_key').xs(country_key, level='country_key')
        log.info(f'sliced to get {dx1.shape}')
        
 
        #===========================================================================
        # filter data------
        #===========================================================================
        dx2 = dx1.stack()
     
        #===========================================================================
        # #drop bad cells
        #===========================================================================
        mdf = dx2.index.to_frame().reset_index(drop=True)
        bx = np.logical_and(
            (mdf['df_id'] == 946).values, 
            np.invert(dx2['bldg'].round(1) == dx2['grid'].round(1)).values)
        
        if bx.any():
            """not sure what this is from"""
            log.warning(f'dropping {bx.sum()}/{len(bx)} from dataset w/ bad linear relation')
            dx3 = dx2.loc[~bx, :]
        else:
            dx3 = dx2
        mdex = dx3.index
            
        #===========================================================================
        # #select functions
        #===========================================================================
        if not dfid_l is None:
            bx = mdex.to_frame().reset_index(drop=True)['df_id'].isin(dfid_l).values
            assert bx.any()
            log.info(f'w/ {len(dfid_l)} df_ids selected {bx.sum()}/{len(bx)}')
            dx4 = dx3.loc[bx, :]
        else:
            dx4=dx3
            
        mdex = dx4.index
            
            
        #===========================================================================
        # #building count
        #===========================================================================
        bx = mdex.get_level_values('bldg_cnt') >= min_bldg_cnt
        log.info(f'selected {bx.sum():.2e}/{len(bx):.2e} w/ min_bldg_cnt={min_bldg_cnt}')
        dx5 = dx4.loc[bx, :]
     
 
        #===========================================================================
        # #all zero
        #===========================================================================
        bx = dx5.sum(axis=1) == 0
        if bx.any():
            log.info(f'got {bx.sum():.2e}/{len(bx):.2e} w/ zero loss...dropping')
            log.info(dx5.loc[bx, :].index.to_frame().reset_index(drop=True)['df_id'].value_counts().to_dict())
            dx6 = dx5.loc[~bx, :]
        else:
            dx6=dx5
            
        #===========================================================================
        # #aoi selection
        #===========================================================================
        if use_aoi:
            #just those in the aoi
            assert not dev
 
            #this function loads all the data.. but we only need the indexers
            """would make more sense to do the slicing above now... but this still works"""
            sel_mdex = get_a03_gstats_1x(country_key=country_key, log=log, use_aoi=use_aoi).droplevel(['country_key', 'bldg_cnt', 'null_cnt']).index
     
            #identify overlap
            bx_aoi = dx6.index.to_frame().reset_index(drop=True).set_index(sel_mdex.names).index.isin(sel_mdex)
            #slice
            dx7 = dx6.loc[bx_aoi, :]
            #check
            check_mdex = dx7.unstack([c for c in dx7.index.names if not c in ['grid_size', 'i', 'j']]).index
            """I guess the selection is shorter because of the filters above"""
            log.info(f'w/ use_aoi={use_aoi} selected {len(check_mdex)} aggregate assets (from the aois {len(sel_mdex)})')
        else:
            dx7 = dx6
            
        #violating bldg_means?
        #===========================================================================
        # #write
        #===========================================================================
        dx = dx7
        log.info(f'writing {dx.shape} to \n    {ofp}')
        dx.to_pickle(ofp)
    else:
        log.info(f'loading from cache\n    {ofp}')
        dx = pd.read_pickle(ofp)
        
    #===========================================================================
    # wrap
    #===========================================================================
    log.info(f'finished data prep w/ {dx.shape}')
    return dx 

    

def plot_TL_agg_v_bldg(
        dx_raw=None,fserx_raw=None,
        country_key='deu',
        haz_key='f500_fluvial',
        dfid_l=None,
 
        figsize=None,
 
        samp_frac=0.008, #
        use_aoi=False,
        dev=False,
        min_bldg_cnt=0,
 
        use_cache=True,
        out_dir=None,
        ):
    
    """plot TOTAL loss from grid centroids and building means    
    
    
    
    Params
    -------------
    fserx_raw: pd.Series
        loss functions
        
    samp_frac: float
        for reducing the data sent to gaussian_kde
        relative to size of complete data set (not the gruops)
 
    """
    #raise IOError('looks pretty good. need to filter partials, add running-mean for buildings, add colorscale')
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
  
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'damage', 'da', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
     
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='tl_agg')
    
 
    ylab_d = clean_names_d.copy()
    cmap = 'cividis'
    
    if dev: samp_frac=1.0
    #===========================================================================
    # load data--------
    #===========================================================================
    dx = _load_and_filter(dx_raw, country_key, haz_key, min_bldg_cnt, dfid_l,   dev, use_aoi, log, use_cache=use_cache)
    mdex=dx.index 
    
    """
    view(dx.head(100))
    """
    
    

    #===========================================================================
    # setup indexers
    #===========================================================================        
    keys_d = {'row':'df_id',  'col':'grid_size'}
    kl = list(keys_d.values())     
    log.info(f' loaded {dx.shape}')
    
    
    #===========================================================================
    # prep loss functions---------
    #===========================================================================
    if fserx_raw is None: 
        fserx_raw = get_funcLib() #select functions
 
 
    #drop meta and add zero-zero
    #fserx = force_monotonic(force_zero_zero(slice_serx(fserx_raw, xs_d=None), log=log),log=log)
    
    #===========================================================================
    # setup figure
    #===========================================================================
    row_keys, col_keys = [mdex.unique(e).tolist() for e in keys_d.values()]
    
 
    
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=False, 
                               constrained_layout=False, #needs to be unconstrainted for over label to work
                               sharex=True, sharey=True, add_subfigLabel=np.invert(present), 
                               figsize=figsize)
    
    rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
    
    #color_d = _get_cmap(color_keys, name='viridis')
 
    #===========================================================================
    # loop and plot-----
    #===========================================================================
    #letter=list(string.ascii_lowercase)[j]
    
 
    for (row_key, col_key), gdx0 in dx.groupby(kl[:2]):
        log.info(f'df_id:{row_key} x grid_size:{col_key}')
        ax = ax_d[row_key][col_key]
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        gdx0 = gdx0.droplevel(kl[:2])
        
        assert (gdx0==0).sum().sum()==0
 
 
        #===================================================================
        # prep
        #===================================================================
        #c=color_d[color_key]
        gdx1=gdx0
        
        
        #prep data
        df= gdx1.reset_index('grid_wd').reset_index(drop=True).set_index('grid_wd')
        
        #geet a sample of hte data
        df_sample = df.copy().sample(min(int(len(dx)*samp_frac), len(df)))
        
        log.info(f'    w/ {df.size} and sample {df_sample.size}')
        
        #===================================================================
        # plot scatter------
        #===================================================================
        #ax.plot(df['bldg'], color='black', alpha=0.3,   marker='.', linestyle='none', markersize=3,label='building')
        #as density
        x,y = df_sample['grid'].values, df_sample['bldg'].values
        xy = np.vstack([np.log(x),np.log(y)]) #log transformed
        
        pdf = gaussian_kde(xy)
        z = pdf(xy) #Evaluate the estimated pdf on a set of points.
        
        # Sort the points by density, so that the densest points are plotted last
        indexer = z.argsort()
        x, y, z = x[indexer], y[indexer], z[indexer]
        
        scatter_kwargs=dict(s=5, cmap=cmap, alpha=0.9, marker='.', edgecolors='none', rasterized=True)
        
        if present:
            scatter_kwargs['s']=20
            
        
        cax = ax.scatter(x, y, c=z, **scatter_kwargs)
        
        #===================================================================
        # plot 1:1
        #===================================================================
        c = [0,10e5]
        
        
        ax.plot(c,c, color='black', linestyle='dashed', linewidth=0.5)
        
        #===================================================================
        # plot interp----
        #===================================================================
        #==============================================================================
        #            this doesnt work well with the log plot
        #            #get data
        #            x,y = df['grid'].values, df['bldg'].values
        #            xar = np.array([0, 10e5])
        #            
        # 
        #            #regression
        #            lm = scipy.stats.linregress(x, y)
        #            
        #            predict = lambda x:np.array([lm.slope*xi + lm.intercept for xi in x])            
        #            ax.plot(xar, predict(xar), color='red', linestyle='solid', label='regression', marker='x')
        #            
        #            print({'rvalue':lm.rvalue, 'slope':lm.slope, 'intercept':lm.intercept})
        #==============================================================================
 
            
        #===================================================================
        # text--------
        #===================================================================
        if not present:
            bsum, gsum = gdx0.sum()
            #tstr = f'count: {len(gdx0)}\n'
            tstr ='$\sum{\overline{RL_{bldg,j}}}*B_{j}}$: %.2e\n'%bsum
            tstr+='$\sum{RL_{grid,j}*B_{j}}$: %.2e\n'%gsum
            
            rmse = np.sqrt(np.mean((gdx0['bldg'] - gdx0['grid'])**2))
            tstr+='RMSE: %.2f\n'%rmse
            
            bias =gdx0['grid'].sum()/gdx0['bldg'].sum()
            tstr+='bias: %.2f'%bias
             
            coords = (0.7, 0.05)
     
             
            ax.text(*coords, tstr, 
                                transform=ax.transAxes, va='bottom', ha='center', 
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                                )
        
        #ax.set_aspect('equal')
        
        ax.set_xlim(1,10e4)
        ax.set_ylim(1,10e4)
 
    #===========================================================================
    # get some function meta
    #===========================================================================
    #get model-dfid lookup
    
    meta_df = fserx_raw.index.to_frame().reset_index(drop=True).loc[:, ['df_id', 'model_id', 'abbreviation']
                                      ].drop_duplicates().set_index('df_id') 
  
    #add defaults and convert to df_id
    ylab_d1=dict()
    for i, row in meta_df.iterrows():
        if not row['model_id'] in ylab_d:
            ylab_d1[i] = row['abbreviation']
        else:
            ylab_d1[i] = ylab_d[row['model_id']]
 
    #===========================================================================
    # post----
    #===========================================================================    
    for row_key, col_key, ax in rc_ax_iter:
 
        # first row
        if row_key == row_keys[0]:
            ax.set_title(f'{col_key}m grid')
 
        # last row
        if row_key == row_keys[-1]: 
            pass 
            # ax.set_xlabel(f'WSH (cm)')
              
        # first col
        if col_key == col_keys[0]: 
            if not row_key=='hist':
                ax.set_ylabel(ylab_d1[row_key])
 
 
            
    #===========================================================================
    # #macro labelling
    #===========================================================================
    if present:
        fig.subplots_adjust(left=0.10)
        labelpad=30
    else:
        labelpad=20
    #plt.subplots_adjust(left=1.0)
    macro_ax = fig.add_subplot(111, frame_on=False)
    _hide_ax(macro_ax) 
    macro_ax.set_ylabel('grid relative loss ($RL_{grid,j}$) * child building count ($B_{j}$)', 
                        labelpad=labelpad, size=font_size+2)
    macro_ax.set_xlabel('child relative loss mean ($\overline{RL_{bldg,j}}$) * child building count ($B_{j}$)', 
                        size=font_size+2)
    
    """doesnt help
    fig.tight_layout()"""
    
    #===========================================================================
    # #add colorbar
    #===========================================================================
    #create the axis
    if not present:
        fig.subplots_adjust(bottom=0.15, top=0.95, right=0.95)
        leg_ax = fig.add_axes([0.05, 0.00, 0.9, 0.08], frameon=True) #left, bottom, width, height
        leg_ax.set_visible(False)    
        
        #color scaling from density values
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(min(z), max(z)), cmap=cmap)
        
        #add the colorbar
        cbar = fig.colorbar(sm,
                         ax=leg_ax,  # steal space from here (couldnt get cax to work)
                         extend='both', #pointed ends
                         format = matplotlib.ticker.FuncFormatter(lambda x, p:'%.1e' % x),
                         label='log-transformed gaussian kernel-density estimate of relative losses (RL) * child building count ($B_{j}$)', 
                         orientation='horizontal',
                         fraction=.99,
                         aspect=50, #make skinny
     
                         )
        
    else:
        fig.subplots_adjust(top=0.95, right=0.95, left=0.13)
 
 
 
    #===========================================================================
    # write----------
    #===========================================================================
    fig.patch.set_linewidth(10)  # set the line width of the frame
    fig.patch.set_edgecolor('cornflowerblue')  # set the color of the frame
    
    
    ofp = os.path.join(out_dir, f'TL_{env_type}_{len(col_keys)}x{len(row_keys)}_{today_str}.svg')
    fig.savefig(ofp, dpi = dpi,   transparent=True,
                #edgecolor='black',
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
    
    hill_curve_l = [942,  944, 945, 946]
    dfunc_curve_l = [26, 380, 402, 941]
    
    
    #plot_rl_raw(tableName='rl_deu_grid_bmean_1020')
 
   
    plot_TL_agg_v_bldg(samp_frac=0.001, dfid_l=dfunc_curve_l,
                       dev=False, use_cache=True)
 
 

    
 
    
    print('done ')
    
    
    
    
