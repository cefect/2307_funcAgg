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
env_type = 'journal'
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
import os, math, hashlib
import pandas as pd
idx = pd.IndexSlice
import numpy as np


from datetime import datetime

import psycopg2
from sqlalchemy import create_engine, URL

from scipy.stats import gaussian_kde
 
from definitions import wrk_dir, clean_names_d, haz_label_d, postgres_d, temp_dir

from coms import init_log, today_str, view
from coms_da import get_matrix_fig, _get_cmap, _hide_ax
from funcMetrics.func_prep import get_funcLib
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
 
def plot_rl_raw(
        tableName='rl_deu_grid_0060', #rl_{country_key}_{asset_type}
        schema='damage',
        out_dir=None,
        figsize=None,
        figsize_scaler=2,
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
                               sharex=True, sharey=True, add_subfigLabel=False, 
                               figsize=figsize,
                               figsize_scaler=figsize_scaler)
    
    rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
    
    #color_d = _get_cmap(color_keys, name='viridis')
    
    
    #===========================================================================
    # loop and plot---------
    #===========================================================================
 
    for (row_key, col_key), gdx0 in serx.groupby(kl[:2]):
        log.info(f'df_id:{row_key} x {col_key} w/ ({len(gdx0)})')
        ax = ax_d[row_key][col_key]
        
        ar = gdx0.droplevel(kl[:2]).reset_index(drop=True).values
        
 
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
    fnstr=f'rl_dx_filter_{country_key}_{haz_key}_{len(dfid_l)}'
    uuid = hashlib.shake_256(
        ' '.join([str(e) for e in [fnstr,  min_bldg_cnt, dfid_l,use_cache, dev, use_aoi]]).encode("utf-8"), 
        usedforsecurity=False).hexdigest(8)
    ofp = os.path.join(out_dir, f'{fnstr}_{uuid}.pkl')
    
    if (not os.path.exists(ofp)) or (not use_cache):
        #===========================================================================
        # load loss data
        #===========================================================================
        if dx_raw is None:
            #load from postgres view damage.rl_mean_grid_{country_key}_{haz_key}_wd and do some cleaning
            dx_raw = load_rl_dx(country_key=country_key, log=log, use_cache=use_cache, dev=dev)
            
        """
        view(dx_raw.head(100))
        """
     
        #===========================================================================
        # slice
        #===========================================================================
        dx1 = dx_raw.xs(haz_key, level='haz_key').xs(country_key, level='country_key')
        log.info(f'sliced to get {dx1.shape}')
        
        #===========================================================================
        # load depth group stats and add some to the index
        #===========================================================================
        """joining some stats from here"""
        wdx_raw = get_a03_gstats_1x(country_key=country_key, log=log, use_cache=use_cache)
        
        wdx1 = wdx_raw.xs(haz_key, level='haz_key', axis=1).xs(country_key, level='country_key').reset_index(['bldg_cnt', 'null_cnt'])
        wdx1.columns.name=None
        
        #rename
        wdx1 = wdx1.rename(columns={'avg':'grid_wd'})
        
        col_l = ['grid_wd', 'bldg_cnt', 'wet_cnt'] #data to join
        
        #join to index
        dx1.index = pd.MultiIndex.from_frame(dx1.index.to_frame().join(wdx1.loc[:, col_l]))
        
 
     
        log.info(f'joined cols from depth stats\n    {col_l}')
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

def plot_rl_agg_v_bldg(
        dx_raw=None,fserx_raw=None,
        country_key='deu',
        haz_key='f500_fluvial',
        out_dir=None,
        figsize=None,
        #min_wet_frac=1.0,
        min_bldg_cnt=2,
        dfid_l = None, 
        samp_frac=0.0001, #
        dev=False,
        ylab_d = clean_names_d,
        cmap='PuRd_r',
        use_cache=True,
        use_aoi=False,
        ):
    

    
    """plot relative loss agg vs. bldg
    
    
    
    Params
    -------------
    fserx_raw: pd.Series
        loss functions
        
    samp_frac: float
        for reducing the data sent to gaussian_kde
        relative to size of complete data set (not the gruops)
 
     use_aoi: bool
         load the spatially sliced grid ids and only include these
    """
 
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
  
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'damage', 'da', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
     
    log = init_log(fp=os.path.join(out_dir, today_str + '.log'), name='rl_agg')
            
    if dev: 
        samp_frac = 1.0
    
    #===========================================================================
    # load data--------
    #===========================================================================
    dx = _load_and_filter(dx_raw, country_key, haz_key, min_bldg_cnt, dfid_l,   dev, use_aoi, log, use_cache=use_cache)
    mdex=dx.index 

    #===========================================================================
    # # get binned means    
    #===========================================================================
    
    keys_l =  ['grid_size',   'df_id', 'grid_wd'] #only keys we preserve   
 
    #get a slice with clean index
    sx1 = dx['bldg'].reset_index(keys_l).reset_index(drop=True).set_index(keys_l)
     
    mean_bin_dx = compute_binned_mean(sx1, log=log, use_cache=use_cache)
    
    """
    view(dx.head(100))
    """
 
 
    #===========================================================================
    # setup indexers
    #===========================================================================        
    keys_d = {'row':'df_id',  'col':'grid_size'}
    kl = list(keys_d.values())     
    log.info(f' loaded {len(dx)}')
    
    
    #===========================================================================
    # prep loss functions---------
    #===========================================================================
    if fserx_raw is None: 
        fserx_raw = get_funcLib() #select functions
 
 
    #drop meta and add zero-zero
    fserx = force_and_slice(fserx_raw, log=log)
    
    #===========================================================================
    # setup figure
    #===========================================================================
    row_keys, col_keys = [mdex.unique(e).tolist() for e in keys_d.values()]
 
    
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=False, 
                               constrained_layout=False, #needs to be unconstrainted for over label to work
                               sharex=True, sharey='row', add_subfigLabel=True, figsize=figsize)
    
    rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
    
 
    #===========================================================================
    # loop and plot-----
    #===========================================================================
 
    meta_lib=dict()
 
    for (row_key, col_key), gdx0 in dx.groupby(kl[:2]):
        log.info(f'df_id:{row_key} x grid_size:{col_key}')
        ax = ax_d[row_key][col_key]
        
        gdx0 = gdx0.droplevel(kl[:2])
        
 
        
        
        #=======================================================================
        # plot function 
        #=======================================================================
        wd_rl_df = fserx.xs(row_key, level='df_id').reset_index('wd').reset_index(drop=True)
        xar, yar = wd_rl_df['wd']*100, wd_rl_df['rl']
         
        ax.plot(xar, yar, color='black', marker='o', linestyle='solid', alpha=0.8, 
                markersize=3,fillstyle='none', linewidth=0.5, label=f'$f(WSH)$')
        
 
        
        
        #prep data
        gdx1 = gdx0
        df= gdx1.reset_index('grid_wd').reset_index(drop=True).set_index('grid_wd')
        
        #===================================================================
        # scatter -------
        #===================================================================
        
        #geet a sample of hte data
        df_sample = df.copy().sample(min(int(len(dx)*samp_frac), len(df)))
        assert len(df_sample)>0
        
        log.info(f'    w/ {df.size} and sample {df_sample.size}')
        
        #ax.plot(df['bldg'], color='black', alpha=0.3,   marker='.', linestyle='none', markersize=3,label='building')
        #as density
        x,y = df_sample.index.values, df_sample['bldg'].values
        
        if not len(df_sample) < 3:
            #xy = np.vstack([x,y])
            xy = np.vstack([np.log(x),np.log(y)]) #log transformed
            
            """need to compute this for each set... should have some common color scale.. but the values dont really matter"""
            try:
                pdf = gaussian_kde(xy)
            except Exception as e:
                raise IOError(f'failed to build gaussian on\n    {xy} w/\n    w/ {e}')
            z = pdf(xy) #Evaluate the estimated pdf on a set of points.
            
            # Sort the points by density, so that the densest points are plotted last
            indexer = z.argsort()
            x, y, z = x[indexer], y[indexer], z[indexer]
        else:
            #just use dummy values
            z = np.full(x.shape, 1.0)
        cax = ax.scatter(x, y, c=z, s=10, cmap=cmap, alpha=0.3, marker='.', edgecolors='none', rasterized=True)
        
        #===================================================================
        # plot binned bldg_mean lines
        #===================================================================
        bin_serx = mean_bin_dx.loc[idx[col_key, row_key, :], :].reset_index(drop=True
                                       ).set_index('grid_wd_bin').iloc[:,0]
                           
        ax.plot(bin_serx, color='blue', alpha=1.0, marker=None, linestyle='dashed', 
                markersize=2, linewidth=1.0,
                label='$\overline{RL_{bldg,j}}(WSH)$')
        
        #===================================================================
        # plot grid-----
        #===================================================================
        #usaeful for debugging
        #ax.plot(df.index.values, df['grid'], color='green', marker='.', alpha=0.3, markersize=1, fillstyle='none', linestyle='none')
        """
        plt.show()
        """
 
            
        #===================================================================
        # text-------
        #===================================================================
 
        bmean, gmean = gdx0.mean()
        #tstr = f'count: {len(gdx0)}\n'
        tstr ='$\overline{\overline{RL_{bldg,j}}}$: %.2f'%bmean
        tstr+='\n$\overline{RL_{grid,j}}$: %.2f'%gmean
        
        rmse = np.sqrt(np.mean((gdx0['bldg'] - gdx0['grid'])**2))
        tstr+='\nRMSE: %.2f'%rmse
        
        bias = gmean/bmean
        tstr+='\nbias: %.2f'%(bias)
        
        #tstr+='\n$\overline{wd}$: %.2f'%(gdx0.index.get_level_values('grid_wd').values.mean())
        
        coords = (0.8, 0.05)
        #=======================================================================
        # if row_key==402:
        #     coords = (coords[0], 0.5) #move it up
        #=======================================================================
 
        
        ax.text(*coords, tstr, size=6,
                            transform=ax.transAxes, va='bottom', ha='center', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
        
        #=======================================================================
        # meta
        #=======================================================================
        if not row_key in meta_lib: meta_lib[row_key] = dict()
        meta_lib[row_key][col_key] = {'bldg_rl_pop_mean':bmean, 'grid_rl_pop_mean':gmean, 'bias':bias}
        
 
        
    #===========================================================================
    # plot histograms---------
    #===========================================================================
 
 #==============================================================================
 #    wd_df = dx.unstack(level='df_id').index.to_frame().reset_index(drop=True).drop(['i', 'j'], axis=1)
 #    for grid_size, gdf in wd_df.groupby('grid_size'):
 #        ax = ax_d['hist'][grid_size]
 #        
 #        ax.hist(gdf['grid_wd'], bins = np.linspace(0, 1000, 31 ), color='black', alpha=0.5, density =True)
 #        
 #        #turn off some spines
 #        ax.spines['top'].set_visible(False)
 #        ax.spines['right'].set_visible(False)
 #        ax.spines['left'].set_visible(False)
 #        
 #        #===================================================================
 #        # text
 #        #===================================================================
 # 
 #        tstr = f'n= {len(gdf):.2e}\n'
 #        tstr +='$\overline{WSH}$= %.2f'%gdf['grid_wd'].mean()
 #        #tstr +='\n$\sigma^2$= %.2f'%gdf['grid_wd'].var() #no... we want the asset variance
 #        ax.text(0.6, 0.5, tstr, 
 #                            transform=ax.transAxes, va='bottom', ha='left', 
 #                            #bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
 #                            )
 #        
 #        raise IOError('add building counts')
 #==============================================================================
 
        
        
            
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
            ax.set_xlim(0, 1010)
 
            ax.set_title(f'{col_key}m grid') 

            if col_key==col_keys[0]:
                ax.legend(ncol=1, loc='center right', frameon=False)
                
        
        # last row
        if row_key == row_keys[-1]: 
            pass 
            # ax.set_xlabel(f'WSH (cm)')
             
        # first col
        if col_key == col_keys[0]: 
            if not row_key=='hist':
                ax.set_ylabel(ylab_d1[row_key])
            else:
                ax.set_ylabel(f'density of WSH')
 
            
    #===========================================================================
    # #macro labelling
    #===========================================================================
    #plt.subplots_adjust(left=1.0)
    macro_ax = fig.add_subplot(111, frame_on=False)
    _hide_ax(macro_ax) 
    macro_ax.set_ylabel(f'relative loss in % (RL)', labelpad=20, size=font_size+2)
    macro_ax.set_xlabel(f'water depth in cm (WSH)', size=font_size+2)
 
    #===========================================================================
    # #add colorbar
    #===========================================================================
    #create the axis
    if len(row_keys)==3:
        fig.subplots_adjust(bottom=0.15, top=0.95, right=0.98)
        leg_ax = fig.add_axes([0.07, 0, 0.9, 0.08], frameon=True)
 
 
    
        leg_ax.set_visible(False)
                
        
        #color scaling from density values
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(min(z), max(z)), cmap=cmap)
        
        #add the colorbar
        cbar = fig.colorbar(sm,
                         ax=leg_ax,  # steal space from here (couldnt get cax to work)
                         extend='both', #pointed ends
                         format = matplotlib.ticker.FuncFormatter(lambda x, p:'%.1e' % x),
                         label='log-transformed gaussian kernel-density estimate of $\overline{RL_{bldg,j}}[WSH]$', 
                         orientation='horizontal',
                         fraction=.99,
                         aspect=50, #make skinny
     
                         )

    else:
        fig.subplots_adjust(right=0.98, bottom=0.18)
 
 
 
    #===========================================================================
    # meta
    #===========================================================================
    meta_df = pd.concat({k:pd.DataFrame.from_dict(v) for k,v in meta_lib.items()},
                        names=['df_id', 'stat'])
    
    mdf1 = meta_df.stack().unstack(level='stat')
 
    log.info(f'meta w/ {meta_df.shape}\n%s'%mdf1['bias'])
        
    #===========================================================================
    # write-------
    #===========================================================================
    fig.patch.set_linewidth(10)  # set the line width of the frame
    fig.patch.set_edgecolor('cornflowerblue')  # set the color of the frame
    
    
    
    uuid = hashlib.shake_256(f'{dfid_l}_{dev}'.encode("utf-8"), usedforsecurity=False).hexdigest(6)
    fnstr=f'{env_type}_{len(col_keys)}x{len(row_keys)}'
    if use_aoi:fnstr+='_aoi'
    ofp = os.path.join(out_dir, f'rl_{fnstr}_{today_str}_{uuid}.svg')
    fig.savefig(ofp, dpi = dpi,   transparent=True,
                #edgecolor='black'
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
 
   
    #plot_TL_agg_v_bldg(samp_frac=0.01)
    #===========================================================================
    # plot_rl_agg_v_bldg(dev=False,  use_cache=True,  
    #                    samp_frac=0.01,
    #                    dfid_l=dfunc_curve_l,                
    #                    )
    #===========================================================================
    
    #Koblenz focal area
    plot_rl_agg_v_bldg(use_cache=True,samp_frac=1.0,dfid_l=[26],  use_aoi=True,
                       figsize=(18*cm,6*cm))

    
 
    
    print('done ')
    
    
    
    
