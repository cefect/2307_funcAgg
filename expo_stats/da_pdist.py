'''
Created on Sep. 13, 2023

@author: cefect

plot for pdist results
'''

#===============================================================================
# PLOT ENV------
#===============================================================================

#===============================================================================
# setup matplotlib 
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

def set_doc_style():
 
    font_size=8
    matplotlib.rc('font', **{'family' : 'serif','weight' : 'normal','size'   : font_size})
     
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
# imports-----
#===============================================================================
import matplotlib.patches as mpatches


import os, glob, hashlib, string, itertools, psutil
import pandas as pd
import numpy as np
idx = pd.IndexSlice
from datetime import datetime

#from scipy.stats import expon

from coms import init_log, today_str, view
from da.hp import get_matrix_fig, _get_cmap, _set_violinparts_style, _get_markers
from expo_stats.coms_exp import load_pdist_concat, _resample_df, _resample_ser

from definitions import wrk_dir, haz_label_d, temp_dir

#===============================================================================
# data
#===============================================================================
 
#===============================================================================
# funcs--------
#===============================================================================


#===============================================================================
# def plot_pdist_metric_v_count(
#  
#         out_dir=None,
#         yval='loc'
#         ):
#     """scatter plots of pdist data vs. count"""
#     
#     #===========================================================================
#     # defaults
#     #===========================================================================
#     
#     if out_dir is None:
#         out_dir=os.path.join(wrk_dir, 'outs', 'da', 'pdist', today_str)
#     if not os.path.exists(out_dir):os.makedirs(out_dir)
#     
#     log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='pdist')
#     
#  
#     #===========================================================================
#     # load data
#     #===========================================================================
#     serx = load_pdist_concat().droplevel(['i', 'j']) #.loc[idx[:,:,:,:,:,yval]].rename(yval)
#     mdex = serx.index
#     
#     #get a data grouper
#     keys_l = ['haz',  'grid_size', 'country_key']    
#     #serx_grouper = serx.groupby(keys_l)
#     
#     
#     log.info(f' loaded {serx.index.shape} ')
#     
#     
#     #===========================================================================
#     # setup figure
#     #===========================================================================
#     row_keys, col_keys, color_keys = [mdex.unique(e).tolist() for e in keys_l]
#     fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=True, sharex='col', sharey=True)
#     
#     rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
#     
#     #color map
#     #retrieve the color map
#     cmap = plt.cm.get_cmap(name='Set1')    
#     ik_d = dict(zip(color_keys, np.linspace(0, 1, len(color_keys))))
#     hex = lambda x:matplotlib.colors.rgb2hex(x)
#     color_d = {k:hex(cmap(ni)) for k, ni in ik_d.items()}
#     
#     
# 
#     #===========================================================================
#     # loop and plot
#     #===========================================================================
#     cnt=0
#     for (row_key, col_key, color_key), gserxR in serx.groupby(keys_l):
#         log.info(f'{row_key} x {col_key} x {color_key}')
#         ax = ax_d[row_key][col_key]
#  
#         #get the data
#         gserx = gserxR.droplevel(keys_l).unstack('metric').reset_index(level='count')
#         
#         #remove all zeros
#         bx = gserx['count']>(gserx['zero_cnt']+gserx['null_cnt'])
#         
#         gserx1 = gserx[bx]
#         
#         """
#         gserx.hist()
#         plt.show()
#         """
#  
#         
#         ax.plot(gserx1['count'], gserx1[yval],color=color_d[color_key], alpha=0.5,
#                  marker='.', linestyle='none', label=color_key)
#         
#         #ax.hist(gserx.index.get_level_values('count'))
#         
#  
#         
#         cnt+=1
#             
#     #===========================================================================
#     # text
#     #===========================================================================
# 
#     serx_grouper = serx.groupby([keys_l[0], keys_l[1]])
#     for row_key, col_key, ax in rc_ax_iter:
#         
#         gserx0 = serx_grouper.get_group((row_key, col_key))
#         
#         tstr=''
#         
#         for color_key, gserx1 in gserx0.groupby(keys_l[2]):
#             gdx=gserx1.droplevel(keys_l).unstack('metric').reset_index(level='count')
#             bx = gdx['count']>(gdx['zero_cnt']+gdx['null_cnt'])
#             
#             tstr += f'{color_key}_cnt={len(bx)}\n{color_key}_dry_cnt={len(bx)-bx.sum()}\n'
#              
#             #f'sector=%s\n'%mdf['sector_attribute'][0]
#             #f'real_frac={bx.sum()/len(bx):.4f}'
#              
#         ax.text(0.95, 0.05, tstr, 
#                             transform=ax.transAxes, va='bottom', ha='right', 
#                             bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
#                             )
#         
#         
#     #===========================================================================
#     # post
#     #===========================================================================
# 
#     
#     for row_key, col_key, ax in rc_ax_iter:
#  
#         #last row
#         if row_key==row_keys[-1]:
#  
#             ax.set_xlabel('agg. size')
#             
#         #first col
#         if col_key==col_keys[0]:
#             ax.set_ylabel(f'{yval} (cm)')
#             
#             if row_key==row_keys[0]:
#                 ax.legend()
#                 
#     
#     #===========================================================================
#     # write
#     #===========================================================================
#     ofp = os.path.join(out_dir, f'pdist_scatter_count-{yval}_{len(col_keys)}x{len(row_keys)}_{today_str}.png')
#     fig.savefig(ofp, dpi = 300,   transparent=True)
#     log.info(f'wrote to \n    %s'%ofp)
#     
#     return ofp
#     """
#     plt.show()
#     """
#         
#  
#  
# 
# def plot_pdist_metric_violin(
#  
#         out_dir=None,
#         yval='loc'
#         ):
#     """scatter plots of pdist data vs. count"""
#     
#     #===========================================================================
#     # defaults
#     #===========================================================================
#     
#     if out_dir is None:
#         out_dir=os.path.join(wrk_dir, 'outs', 'da', 'pdist', today_str)
#     if not os.path.exists(out_dir):os.makedirs(out_dir)
#     
#     log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='pdist')
#     
#  
#     #===========================================================================
#     # load data
#     #===========================================================================
#     serx = load_pdist_concat().droplevel(['i', 'j']) #.loc[idx[:,:,:,:,:,yval]].rename(yval)
#     mdex = serx.index
#     
#     #get a data grouper
#     keys_d = {'row':'haz',  'col':'grid_size', 'color':'country_key'}
#     kl = list(keys_d.values())    
#     #serx_grouper = serx.groupby(keys_l)
#     
#     
#     log.info(f' loaded {serx.index.shape} for yval={yval}')
#     
#     
#     #===========================================================================
#     # setup figure
#     #===========================================================================
#     row_keys, col_keys, color_keys = [mdex.unique(e).tolist() for e in keys_d.values()]
#     fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=True, sharex=True, sharey=True)
#     
#     rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
#     
#     #color map
#     #retrieve the color map
#     color_d= _get_cmap(color_keys)
#     
#     
# 
#     #===========================================================================
#     # loop and plot
#     #===========================================================================
#     cnt=0
#     for (row_key, col_key), gserx0 in serx.groupby(kl[:2]):
#         log.info(f'{row_key} x {col_key}')
#         ax = ax_d[row_key][col_key]
#  
#         #make a violin plot for each color on the same axis
#         cl = list()
#         for i, (color_key, gserx1) in enumerate(gserx0.groupby(kl[2])):
#             cl.append(color_key)
#             color=color_d[color_key]
#             #get the data
#             gdx = gserx1.droplevel(kl).unstack('metric').reset_index(level='count')
#         
#             #remove all zeros
#             bx = gdx['count']>(gdx['zero_cnt']+gdx['null_cnt'])
#             
#             gserx2 = gdx.loc[bx,yval]
#             
#             #voilin plot
#             violin_parts = ax.violinplot(gserx2.values, positions=[i], 
#                                          showmeans=True, showmedians=False, showextrema=False,
#  
#                                          )
#             
#             
#             
#             # Change the color of each part of the violin plot
#             for pc in violin_parts.pop('bodies'):
#                 pc.set_facecolor(color)
#                 #pc.set_edgecolor('black')
#                 pc.set_alpha(.8)
#                 
#             for k, line in violin_parts.items():
#                 line.set_color('black')
#                 
#             cnt+=1
#                 
#                 
#         #fix ticks
#         ax.set_xticks(np.arange(len(cl)))
#         ax.set_xticklabels(cl)
#  
#     #===========================================================================
#     # text
#     #===========================================================================
# 
#     #===========================================================================
#     # serx_grouper = serx.groupby([keys_l[0], keys_l[1]])
#     # for row_key, col_key, ax in rc_ax_iter:
#     #     
#     #     gserx0 = serx_grouper.get_group((row_key, col_key))
#     #     
#     #     tstr=''
#     #     
#     #     for color_key, gserx1 in gserx0.groupby(keys_l[2]):
#     #         gdx=gserx1.droplevel(keys_l).unstack('metric').reset_index(level='count')
#     #         bx = gdx['count']>(gdx['zero_cnt']+gdx['null_cnt'])
#     #         
#     #         tstr += f'{color_key}_cnt={len(bx)}\n{color_key}_dry_cnt={len(bx)-bx.sum()}\n'
#     #          
#     #         #f'sector=%s\n'%mdf['sector_attribute'][0]
#     #         #f'real_frac={bx.sum()/len(bx):.4f}'
#     #          
#     #     ax.text(0.95, 0.05, tstr, 
#     #                         transform=ax.transAxes, va='bottom', ha='right', 
#     #                         bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
#     #                         )
#     #===========================================================================
#         
#         
#     #===========================================================================
#     # post
#     #===========================================================================
#     
#     #build legend
#     legend_handles = [mpatches.Patch(color=c, label=k) for k,c in color_d.items()]
#     
#     for row_key, col_key, ax in rc_ax_iter:
#  
#         #last row
#         if row_key==row_keys[-1]:
#  
#             ax.set_xlabel(kl[2])
#             
#         #first col
#         if col_key==col_keys[0]:
#             ax.set_ylabel(f'{yval} (cm)')
#             
#             
#         #last col
#         if col_key==col_keys[-1]:   
#             if row_key==row_keys[0]:
#                 ax.legend(handles=legend_handles)
#                 
#     
#     #===========================================================================
#     # write
#     #===========================================================================
#     ofp = os.path.join(out_dir, f'pdist_violin_{yval}_{len(col_keys)}x{len(row_keys)}_{today_str}.svg')
#     fig.savefig(ofp, dpi = 300,   transparent=True)
#     log.info(f'wrote to \n    %s'%ofp)
#     plt.close('all')
#     
#     return ofp
#     """
#     plt.show()
#     """
#         
#  
#     
# def plot_pdist_paramterized(
#  
#         out_dir=None,
#         std_dev_multiplier=2,
#  
#         ):
#     """use average parameter values to plot pdist for each group"""
#     
#     #===========================================================================
#     # defaults
#     #===========================================================================
#     
#     if out_dir is None:
#         out_dir=os.path.join(wrk_dir, 'outs', 'da', 'pdist', today_str)
#     if not os.path.exists(out_dir):os.makedirs(out_dir)
#     
#     log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='pdist')
#     
#  
#     #===========================================================================
#     # load data
#     #===========================================================================
#     serx = load_pdist_concat().droplevel(['i', 'j']) #.loc[idx[:,:,:,:,:,yval]].rename(yval)
#     mdex = serx.index
#     
#     #get a data grouper
#     keys_d = {'row':'haz',  'col':'grid_size', 'color':'country_key'}
#     kl = list(keys_d.values())    
#     #serx_grouper = serx.groupby(keys_l)
#     
#     
#     log.info(f' loaded {serx.index.shape}')
#     
#     #===========================================================================
#     # setup figure
#     #===========================================================================
#     row_keys, col_keys, color_keys = [mdex.unique(e).tolist() for e in keys_d.values()]
#     fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=True, sharex=True, sharey=True)
#     
#     rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
#     
#     #color map
#  
#     color_d = _get_cmap(color_keys)
#     
#     #===========================================================================
#     # loop and plot
#     #===========================================================================
#     xar = np.linspace(0, 500, 100) #dummy xrange
#     cnt=0
#     for (row_key, col_key), gserx0 in serx.groupby(kl[:2]):
#         log.info(f'{row_key} x {col_key}')
#         ax = ax_d[row_key][col_key]
#  
#         #make a violin plot for each color on the same axis
#         cl = list()
#         for i, (color_key, gserx1) in enumerate(gserx0.groupby(kl[2])):
#             cl.append(color_key)
#             color=color_d[color_key]
#             #get the data
#             gdx = gserx1.droplevel(kl).unstack('metric').reset_index(level='count')
#         
#             #remove all zeros
#             bx = gdx['count']>(gdx['zero_cnt']+gdx['null_cnt'])
#             
#             gdx1 = gdx.loc[bx,['loc', 'scale']].sort_values('loc', ignore_index=True)
#             
#             #get location indexers
#             s = gdx1['loc']
#             def get_idx(search_val):
#                 return (s - search_val).abs().idxmin()
#             
#             std_dev = s.std()            
#             
#             #plot each
#             for k, index_val, line_kwargs in [
#                 ('mean', get_idx(s.mean()), dict(alpha=0.8, label=color_key)),
#                 ('upper_std',get_idx(s.mean()+std_dev*std_dev_multiplier),dict(alpha=0.2)),
#                 ('lower_std',get_idx(s.mean()-std_dev*std_dev_multiplier), dict(alpha=0.2)),
#                 ]:
#                 
#                 loc, scale = gdx1.loc[index_val, 'loc'],  gdx1.loc[index_val, 'scale']
#                 pdist_ar = expon.pdf(xar, loc, scale)
#                 ax.plot(xar, pdist_ar, color=color, **line_kwargs)
#                 
#             log.debug(f"finished on {i}")
#             
#             """
#             plt.show()
#             """
#     
#===============================================================================


def plot_hist_combine_country_violin(
        country_key='bgd',
        out_dir=None,
        #std_dev_multiplier=1,
        min_wet_frac=0.05,
        
        bw_method=0.1,
        sample_frac=1.0,
        ):
    """plot the combined histograms for a single country as violines per bar
    
    
    Params
    -------
    min_wet_frac: float
        fitler to exclude cells with very little exposure
        
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'da', 'pdist', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name=f'pdist.{country_key}')
    
 
    #===========================================================================
    # load data
    #===========================================================================
    dx = load_pdist_concat().droplevel(['i', 'j'])
    mdex = dx.index
    
    #get a data grouper
    keys_d = {'row':'haz',  'col':'grid_size', 'color':'country_key'}
    kl = list(keys_d.values())    
    #serx_grouper = serx.groupby(keys_l)
    
    
    log.info(f' loaded {dx.shape}')
    
    #===========================================================================
    # setup figure
    #===========================================================================
    row_keys, col_keys, color_keys = [mdex.unique(e).tolist() for e in keys_d.values()]
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=False, sharex=True, sharey=True, add_subfigLabel=False)
    
    rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
    
    #color map 
    color_d = _get_cmap(color_keys)
    
    #===========================================================================
    # loop and plot
    #===========================================================================
    #letter=list(string.ascii_lowercase)[j]
 
    for (row_key, col_key), gserx0 in dx.xs(country_key, level='country_key').groupby(kl[:2]):
        log.info(f'{row_key} x {col_key}')
        ax = ax_d[row_key][col_key] 
 
        color_key=country_key
        color=color_d[color_key]
        
        #===================================================================
        # #get the data
        #===================================================================
        gdx = gserx0.droplevel(kl[:2])
        gdx = gdx.sample(n=int(len(gdx)*sample_frac))
        
        #split the data
        metric_df_raw = gdx.xs('metric', level=0, axis=1)
        metric_df_raw['wet_frac'] = metric_df_raw['wet_cnt']/metric_df_raw['count']
        """
        view(metric_df_raw)
        """
        
        #apply filter
        bx = metric_df_raw['wet_frac']>min_wet_frac
        
        if not bx.any():
            log.warning(f'    no values exceed min_wet_frac={min_wet_frac}')
            continue
            
        log.info(f'    selected {bx.sum()}/{len(bx)} w/ min_wet_frac={min_wet_frac}')
        
        #metric_df = gdx.xs('metric', level=0, axis=1)[bx]
        
        hist_df_raw = gdx.xs('hist', level=0, axis=1)[bx]
        
        hist_df_raw.columns.name='wsh'
        
        #ressample to reduce x discretization
        log.info(f'    resampling w/ {hist_df_raw.shape}')
        hist_df = _resample_df(hist_df_raw).dropna(axis=1, how='all')
        
        #=======================================================================
        # violin plots plot
        #=======================================================================
        log.info(f'    building violin plots on {hist_df.shape} w/ bw_method={bw_method}')
        gd = {k:v.dropna().values.reshape(-1) for k,v in hist_df.items()}
        
        violin_parts = ax.violinplot(gd.values(), widths=1.0, showmeans=True, showextrema=False,  bw_method=bw_method)
        
        # Change the color of each part of the violin plot
        _set_violinparts_style(violin_parts, color)
 
 
        #===================================================================
        # text
        #===================================================================
        
        tstr = f'cnt={len(bx)}\nselect_cnt={bx.sum()}\n'
 
              
        ax.text(0.95, 0.95, tstr, 
                            transform=ax.transAxes, va='top', ha='right', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
            
 
        log.debug(f"finished")
            
        """
        ax.clear()
        plt.show()
        """
        
    #===========================================================================
    # post
    #===========================================================================
    fig.suptitle(country_key)
     
    for row_key, col_key, ax in rc_ax_iter:
        #ax.grid()
        
        #first row
        if row_key==row_keys[0]:
            ax.set_title(f'{col_key}m grid')
        
        #last row
        if row_key==row_keys[-1]:
  
            ax.set_xlabel(f'WSH (cm)')
            
            #ax.get_ticks()
            ax.set_xticklabels([int(e) for e in gd.keys()])  # rotation is optional
             
        #first col
        if col_key==col_keys[0]:
            ax.set_ylabel(f'{haz_label_d[row_key]} density')
             
 
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'hist_combine_violin_{country_key}_{len(col_keys)}x{len(row_keys)}_{today_str}.svg')
    fig.savefig(ofp, dpi = 300,   transparent=True)
    log.info(f'wrote to \n    %s'%ofp)
    plt.close('all')
     
    return ofp
 

def plot_hist_combine_mean_line(
 
        out_dir=None,
        std_dev_multiplier=1,
        min_wet_frac=0.05,
        
        bw_method=0.1,
        sample_frac=1.0,
        ):
    """plot mean lines of hist bars
    
    
    Params
    -------
    min_wet_frac: float
        fitler to exclude cells with very little exposure
        
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'da', 'pdist', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name=f'pdist')
    start=datetime.now()
    
    #===========================================================================
    # load data
    #===========================================================================
    dx = load_pdist_concat().droplevel(['i', 'j'])
    mdex = dx.index
    
    #get a data grouper
    keys_d = {'row':'haz',  'col':'grid_size', 'color':'country_key'}
    kl = list(keys_d.values())    
 
    log.info(f' loaded {dx.shape}')
    
    #===========================================================================
    # setup figure
    #===========================================================================
    row_keys, col_keys, color_keys = [mdex.unique(e).tolist() for e in keys_d.values()]
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=False, sharex=True, sharey=True, add_subfigLabel=False)
    
    rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
    
    #color map 
    color_d = _get_cmap(color_keys)
    
    marker_d = _get_markers(color_keys, markers=['s', 'o'])
    
    #===========================================================================
    # loop and plot
    #===========================================================================
    #letter=list(string.ascii_lowercase)[j]
 
    for (row_key, col_key), gdx0 in dx.groupby(kl[:2]):
        log.info(f'{row_key} x {col_key}')
        ax = ax_d[row_key][col_key] 
        
        tstr=''
        for color_key, gdx1 in gdx0.groupby(kl[2]):
            marker=marker_d[color_key]
            color=color_d[color_key]
            
            #===================================================================
            # #get the data
            #===================================================================
            gdx = gdx1.droplevel(kl)
            if not sample_frac is None:
                #log.warning(f'sampling w/ sample_frac={sample_frac}')
                gdx = gdx.sample(n=int(len(gdx)*sample_frac))
            
            #split the data
            metric_df_raw = gdx.xs('metric', level=0, axis=1)
            metric_df_raw['wet_frac'] = metric_df_raw['wet_cnt']/metric_df_raw['count']
            """
            view(metric_df_raw)
            """
            
            #apply filter
            bx = metric_df_raw['wet_frac']>min_wet_frac
            
            if not bx.any():
                log.warning(f'    no values exceed min_wet_frac={min_wet_frac}')
                continue
                
            log.info(f'    selected {bx.sum()}/{len(bx)} w/ min_wet_frac={min_wet_frac}')
            
            #metric_df = gdx.xs('metric', level=0, axis=1)[bx]
            
            hist_df_raw = gdx.xs('hist', level=0, axis=1)[bx]
            
            hist_df_raw.columns.name='wsh'
            

 
            #===================================================================
            # plot mean line
            #===================================================================
            #ressample to reduce x discretization
            log.info(f'    resampling w/ {hist_df_raw.shape}')
            
            hist_mean_ser = hist_df_raw.mean(axis=0).dropna()
            
            xar, yar = hist_mean_ser.index.astype(float).values, hist_mean_ser.values
            
            line_kwargs = dict(color=color, marker = marker, markersize=3, fillstyle='none')
            ax.plot(xar, yar,  alpha=0.8, label=color_key, **line_kwargs)
            
            #===================================================================
            # plot error fills
            #===================================================================
            hist_std_ser = hist_df_raw.std(axis=0).dropna()
            
            # Calculate upper and lower bounds
            upper_bound = yar + hist_std_ser.values*std_dev_multiplier
            lower_bound = np.maximum(0, yar - hist_std_ser.values*std_dev_multiplier)
            
            assert xar.shape==upper_bound.shape
            
            # Plot error fills
            ax.fill_between(xar, lower_bound, upper_bound, color=color, alpha=0.1)
            
            #add markers
            for bnd_ar in [lower_bound, upper_bound]:
                ax.plot(xar, bnd_ar,alpha=0.1,**line_kwargs)
     
     
            #===================================================================
            # text
            #===================================================================
            
            tstr += f'{color_key.upper()}: {bx.sum():,}/{len(bx):,}\n'
 
              
        ax.text(0.4, 0.95, tstr, 
                            transform=ax.transAxes, va='center', ha='left', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
            
 
        log.debug(f"finished")
            
        """
        ax.clear()
        plt.show()
        """
        
    #===========================================================================
    # post
    #===========================================================================
 
     
    for row_key, col_key, ax in rc_ax_iter:
        #ax.grid()
        
        #first row
        if row_key==row_keys[0]:
            ax.set_title(f'{col_key}m grid')
            
            if col_key==col_keys[-1]:
                ax.legend()
        
        #last row
        if row_key==row_keys[-1]:
  
            ax.set_xlabel(f'WSH (cm)')
 
             
        #first col
        if col_key==col_keys[0]:
            ax.set_ylabel(f'{haz_label_d[row_key]} density')
             
 
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'hist_combine_mean_{len(col_keys)}x{len(row_keys)}_{today_str}.svg')
    fig.savefig(ofp, dpi = 300,   transparent=True)
    
    plt.close('all')
    
    
 
    meta_d = {
                    'tdelta':(datetime.now()-start).total_seconds(),
                    'RAM_GB':psutil.virtual_memory () [3]/1000000000,
                    #'file_GB':get_directory_size(out_dir),
                    'output_MB':os.path.getsize(ofp)/(1024**2)
                    }
    
    log.info(meta_d)
    
    log.info(f'wrote to \n    %s\n    {meta_d}'%ofp)
     
    return ofp
 
    
if __name__=='__main__':
    


    #plot_pdist_metric_v_count()
    #plot_pdist_metric_violin(yval='loc')
    #plot_pdist_metric_violin(yval='scale')
    
    
    #plot_pdist_paramterized()
    #===========================================================================
    # for k in ['bgd', 'deu']:
    #     plot_hist_combine_country_violin(country_key=k, sample_frac=1.0)
    #===========================================================================
    
    plot_hist_combine_mean_line(sample_frac=0.05)

 
    
    print('finished ')
    
    