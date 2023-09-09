'''
Created on Mar. 28, 2023

@author: cefect

plot damage functions
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
# imports---------
#===============================================================================
import os, math
import pandas as pd
idx = pd.IndexSlice
import numpy as np


from datetime import datetime

from coms import init_log, today_str, view
from da.hp import get_matrix_fig

from definitions import wrk_dir, dfunc_pkl_fp, clean_names_d

#===============================================================================
# data
#===============================================================================
def transform_1d_to_2d(arr, cols, constant_values=-9999):
    n = len(arr)
    rows = math.ceil(n / cols)
    padded_arr = np.pad(arr, (0, rows*cols - n), constant_values=constant_values)
    return np.resize(padded_arr, (rows, cols))



def _get_matrix_plot_map_df(model_l, figsize, ncols):
    """take a list and reshape it onto 2D for setting up a matrix plot"""
    #build a dataframe to map the model_id onto the grid
    model_df = pd.DataFrame(transform_1d_to_2d(np.array(model_l), ncols))
    model_df.columns = [f'c{e}' for e in model_df.columns]
    model_df.index = [f'r{e}' for e in model_df.index]
#get figsize (holding the width)
    if figsize is None:
        figsize_width = matplotlib.rcParams['figure.figsize'][0]
        figsize_height = figsize_width * (model_df.shape[0] / model_df.shape[1])
        figsize = figsize_width, figsize_height
#setup the figure with this
    col_keys, row_keys = model_df.columns.tolist(), model_df.index.tolist()
    return row_keys, col_keys, figsize, model_df


def _update_fancy_names(model_names_d, serx):
    for model_id in serx.index.unique('model_id'):
        if not model_id in model_names_d:
            model_names_d[model_id] = serx.xs(model_id, level='model_id').index.unique('abbreviation')[0].replace(' et al.', '.')
    
    return 

def plot_dfunc_matrix(
        fp=None,
        out_dir=None,
        figsize=None,
        model_l=None,
        ncols=5,
        model_names_d=clean_names_d.copy(),
        ):
    """plot damage function groups"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    if fp is  None: fp = dfunc_pkl_fp
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'funcs', 'da', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='dfunc')
    
 
    #===========================================================================
    # load data from first
    #===========================================================================
    
    serx = pd.read_pickle(fp).xs('residential', level='sector_attribute')
    
    if model_l is None:
        model_l = serx.index.unique('model_id').tolist()
        
 
    #update the names
    _update_fancy_names(model_names_d, serx)
 
        
    
    """
    dxind.index.unique('model_id')
    """
    log.info(f' loaded {serx.shape}')
    
    dfid_cnt = len(serx.index.unique('df_id'))
    log.info(f'plotting {dfid_cnt} functions')
    
 
    #===========================================================================
    # setup figure
    #===========================================================================
    row_keys, col_keys, figsize, model_df = _get_matrix_plot_map_df(model_l, figsize, ncols)
    
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=False, 
                               sharex=True, sharey=True, figsize=figsize, add_subfigLabel=False)
 
 
    #uncessary levels
    drop_lvl_names = list(set(serx.index.names).difference(['model_id','df_id', 'wd']))
    #===========================================================================
    # plot loop    
    #===========================================================================
    cnt=0
    rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
    for row_key, col_key, ax in rc_ax_iter:
 
        #load this data
        model_id = model_df.loc[row_key, col_key]
        if model_id==-9999:
            #ax.axis('off') #hide it
            continue
        fancy_name = model_names_d[model_id]
        
        #slice to this model and remove unecessary indexers
        serx_i = serx.xs(model_id, level='model_id').droplevel(drop_lvl_names)
 
 
        #plot all the curves
        func_cnt = len(serx_i.index.unique('df_id'))
        log.info(f'plotting %i \'df_id\' values for model_id=\'{model_id}\' ({fancy_name})'%(func_cnt))
        for df_id, gserx in serx_i.groupby('df_id', group_keys=False):
            
            ar = gserx.droplevel('df_id').reset_index().values.swapaxes(0,1)
            ax.plot(ar[0], ar[1], color='black', linewidth=0.5, alpha=0.5)
            
            
        #plot the mean
        if func_cnt>1:
            ar = serx_i.groupby('wd').mean().reset_index().values.swapaxes(0,1)
            ax.plot(ar[0], ar[1], color='red', linewidth=1.0,  linestyle='dashed', label='mean')
 
        """
        plt.show()
        view(mdf)
        """
        
        #add text
        mdf = serx.xs(model_id, level='model_id').droplevel(['df_id', 'wd']).index.to_frame().reset_index(drop=True).drop_duplicates(keep='first').dropna(axis=0, how='all')
        
 
 
        tstr = f'%s\n'%fancy_name+f'model={model_id}\n'+f'cnt={func_cnt}'
            
            #f'sector=%s\n'%mdf['sector_attribute'][0]
            #f'real_frac={bx.sum()/len(bx):.4f}'
            
        ax.text(0.95, 0.05, tstr, 
                            transform=ax.transAxes, va='bottom', ha='right', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
        
        cnt+=1
        
 
 
    log.info(f'plot built w/ {cnt:,.0f} records')
    #===========================================================================
    # post
    #===========================================================================
    for row_key, col_key, ax in rc_ax_iter:
        
        #set limits
        ax.set_ylim(0,110)
        ax.set_xlim(0,10.0)
            
 
        #last row
        if row_key==row_keys[-1]:
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: '%i'%(x*100)))
            ax.set_xlabel('depth (cm)')
            
        #first col
        if col_key==col_keys[0]:
            ax.set_ylabel('relative loss (pct)')
            
            if row_key==row_keys[0]:
                ax.legend()
    
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'dfuncs_{len(col_keys)}x{len(row_keys)}.svg')
    fig.savefig(ofp, dpi = 300,   transparent=True)
    log.info(f'wrote to \n    %s'%ofp)
    
    return ofp
    
    

def plot_gradient_matrix(
        fp=None,
        out_dir=None,
        figsize=None,
        dfid_l=[49, 798, 811, 449, 481, 402],
        #dfid_l=None,
 
        model_names_d=clean_names_d.copy(),
        line_kwargs_d = {
            'rl':dict(linestyle='solid', marker='o', fillstyle='none'),
            'deriv1':dict(linestyle='dashed', marker='v'),
            'deriv2':dict(linestyle='dotted', marker='^')
            }
        ):
    """plot function, gradient1, and gradient2
    only using eaxmple functions
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
 
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'funcs', 'da', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='grad')
    
 
    #===========================================================================
    # load data from first
    #===========================================================================
    
    dx_raw = pd.read_pickle(fp)#.xs('residential', level='sector_attribute')
    
    if dfid_l is None:
        #dfid_l = dx_raw.index.unique('df_id').tolist()
        
 
        
        #all non-flemo
        tvals = tuple(set(dx_raw.index.unique('model_id')).difference([3]))
        
        bx = np.isin(dx_raw.index.get_level_values('model_id'), tvals)
        
        dfid_l = dx_raw.loc[bx, :].index.unique('df_id').tolist()[40:60]
        #=======================================================================
        # print()
        # 
        # dx_raw.loc[bx, :].groupby('df_id').plot()
        #=======================================================================
 
        
    #slice
    dx = dx_raw.loc[np.isin(dx_raw.index.get_level_values('df_id'),dfid_l), :]
 
    #update the names
    _update_fancy_names(model_names_d, dx)
 
        
    
    """
    dxind.index.unique('model_id')
    """
    log.info(f' loaded {dx.shape}')
    
    dfid_cnt = len(dx.index.unique('df_id'))
    log.info(f'plotting {dfid_cnt} functions')
    
 
    #===========================================================================
    # setup figure
    #===========================================================================
    row_keys = ['']
    col_keys = dfid_l
 
    if figsize is None:
        #figsize_width = matplotlib.rcParams['figure.figsize'][0]
        figsize_width = dfid_cnt*3
        figsize_height = figsize_width / dfid_cnt
        
        figsize = figsize_width, figsize_height
    
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=False, 
                               sharex=True, sharey=True, figsize=figsize, add_subfigLabel=False)
 
 
    #uncessary levels
    drop_lvl_names = list(set(dx.index.names).difference(['df_id', 'wd']))
    #===========================================================================
    # plot loop    
    #===========================================================================
    cnt=0
    rc_ax_iter = [(row_key, col_key, ax) for row_key, ax_di in ax_d.items() for col_key, ax in ax_di.items()]
    for row_key, col_key, ax in rc_ax_iter:
 
        #load meta        
        mdex = dx.xs(col_key, level='df_id').index #get index data        
        model_id = mdex.unique('model_id')[0] 
        fancy_name = model_names_d[model_id]
        
        
        #get data
        df = dx.xs(col_key, level='df_id').droplevel(drop_lvl_names) 
        
        #plot each
        for name, col in df.items():
            col.plot(ax=ax, 
                     #color='black', 
                     **line_kwargs_d[name])
 
 
 
 
        """
        plt.show()
        view(mdf)
        """
        
        #add text
        #mdf = serx.xs(model_id, level='model_id').droplevel(['df_id', 'wd']).index.to_frame().reset_index(drop=True).drop_duplicates(keep='first').dropna(axis=0, how='all')
        
 
 
        tstr = f'%s\n'%fancy_name+f'model={model_id}\n'+f'func={col_key}'
            
            #f'sector=%s\n'%mdf['sector_attribute'][0]
            #f'real_frac={bx.sum()/len(bx):.4f}'
            
        ax.text(0.95, 0.05, tstr, 
                            transform=ax.transAxes, va='bottom', ha='right', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
        
        cnt+=1
        
 
 
    log.info(f'plot built w/ {cnt:,.0f} records')
    #===========================================================================
    # post
    #===========================================================================
    for row_key, col_key, ax in rc_ax_iter:
        
        #set limits
        ax.set_ylim(-100,100)
        ax.set_xlim(0,10.0)
        ax.grid()
            
        #first row
        if row_key==row_keys[0]:
            if col_key==col_keys[-1]: #last col
                ax.legend()
            
        #last row
        if row_key==row_keys[-1]:
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: '%i'%(x*100)))
            ax.set_xlabel('depth (cm)')
            
        #first col
        if col_key==col_keys[0]:
            ax.set_ylabel('relative loss (pct)')
            

            

    
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'dfuncs_gradient_{len(col_keys)}x{len(row_keys)}.svg')
    fig.savefig(ofp, dpi = 300,   transparent=True)
    log.info(f'wrote to \n    %s'%ofp)
    
    return ofp
    
    
   
if __name__=='__main__':
    


    #plot_dfunc_matrix()
    
    plot_gradient_matrix(fp=r'l:\10_IO\2307_funcAgg\outs\funcs\01_deriv\derivs_3266_20230907.pkl')

    
 
    
    print('finished ')
    
    
    
    