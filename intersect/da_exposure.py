'''
Created on Mar. 28, 2023

@author: cefect

data analysis on intersect (exposure) results
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
# imports
#===============================================================================
import os
import pandas as pd
import numpy as np

from datetime import datetime

from coms import init_log, today_str
from da.hp import get_matrix_fig

from definitions import wrk_dir, haz_label_d

#===============================================================================
# data
#===============================================================================
    #results from 03_concat
concat_fp_d = {
    'ZAF': 'samp_concat_ZAF_986456.pkl',
    'DEU': 'samp_concat_DEU_25994846.pkl',
    'CAN': 'samp_concat_CAN_5754035.pkl',
    'BRA': 'samp_concat_BRA_6450206.pkl',
    'BGD': 'samp_concat_BGD_4668905.pkl',
    'AUS': 'samp_concat_AUS_2333745.pkl'}

concat_fp_d = {k:os.path.join(r'l:\10_IO\2307_funcAgg\outs\inters\03_concat', v) for k,v in concat_fp_d.items()}


def plot_wd_hist_country(
        fp,country_key,
        out_dir=None,
        ):
    """plot a histogram of the sample from a country"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'da', 'exposure', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='hist')
    
 
    #===========================================================================
    # load data
    #===========================================================================
    dxind = pd.read_pickle(fp)
    log.info(f' loaded {dxind.shape} for {country_key} from \n    {fp}')
    
    
 
    #===========================================================================
    # setup figure
    #===========================================================================
    row_keys, col_keys = dxind.columns.tolist(), [' ']
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=False, sharex=True, sharey=True)
     
    ax_d = {k:v[' '] for k,v in ax_d.items()} #drop level
    
    #===========================================================================
    # plot loop
    #===========================================================================]
    for haz_key, ax in ax_d.items():
        #drop zeros
        bx = dxind[haz_key]!=0
        ar = dxind.loc[bx, haz_key].values
        
        #add histogram
        log.info(f'adding histogram for {haz_key} w/ {bx.sum()}/{len(bx)}')
        bvals, bins, patches = ax.hist(ar, density=True, bins=30, color='black', alpha=0.5)
        
        
        #add text
        tstr = f'{haz_key}\ntcnt={len(bx)}\nzero_cnt={len(bx)-bx.sum()}\nnon-zero_frac={bx.sum()/len(bx):.4f}'
        ax.text(0.95, 0.95, tstr, 
                            transform=ax.transAxes, va='top', ha='right', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
 
    #===========================================================================
    # post
    #===========================================================================
    for row_key, ax in ax_d.items():
        ax.set_title(haz_label_d[row_key])
        #ax.set_ylabel(row_key)
        
 
            
        #last row
        if row_key==row_keys[-1]:
            ax.set_xlabel('water depth (cm)')
    
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'wd_hist_{country_key}_{len(dxind)}.svg')
    fig.savefig(ofp, dpi = 300,   transparent=True)
    log.info(f'wrote to \n    %s'%ofp)
    
    return ofp
    
    

def plot_wd_hist_combine(
        fp_d=None,
        out_dir=None,
        figsize=(20*cm,18*cm),
        ):
    """plot a histogram of the sample from a country"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    if fp_d is  None: fp_d = concat_fp_d
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'da', 'exposure', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='hist2')
    
 
    #===========================================================================
    # load data from first
    #===========================================================================
    
    dxind = pd.read_pickle(list(fp_d.values())[0])
    log.info(f' loaded {dxind.shape}')
    
    
 
    #===========================================================================
    # setup figure
    #===========================================================================
    col_keys, row_keys  = dxind.columns.tolist(), list(fp_d.keys())
    fig, ax_d = get_matrix_fig(row_keys, col_keys, log=log, set_ax_title=False, sharex=True, sharey=True, figsize=figsize)
     
 
    #===========================================================================
    # plot loop    
    #===========================================================================]
    cnt=0
    for country_key, fp in fp_d.items():
        #load this data
        dxind = pd.read_pickle(fp)
        cnt+=len(dxind)
        log.info(f'for {country_key} loaded {dxind.shape}')
        
        
        #loop on hazard key
        for haz_key, col in dxind.items():
            log.info(f'building {country_key}.{haz_key}')
            
            ax = ax_d[country_key][haz_key]
            
            
            #drop zeros and null
            bx = np.logical_and(
                col!=0, col.notnull())
            ar = col[bx].values
            
            #add histogram
            log.debug(f'adding histogram for {haz_key} w/ {bx.sum()}/{len(bx)}')
            bvals, bins, patches = ax.hist(ar, density=True, bins=30, color='black', alpha=0.5)
            
            #add mean line
            ax.axvline(np.mean(ar), color='black', linestyle='dashed', linewidth=0.75)
            
            
            #add text
            tstr = f'count={len(bx):.1e}\n'+\
                f'zeros={(col!=0).sum():.1e}\n'+\
                f'nulls={col.isnull().sum():.1e}\n'
                #f'real_frac={bx.sum()/len(bx):.4f}'
                
            ax.text(0.95, 0.95, tstr, 
                                transform=ax.transAxes, va='top', ha='right', 
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                                )
 
    log.info(f'plot built w/ {cnt:,.0f} records')
    #===========================================================================
    # post
    #===========================================================================
    for row_key, ax_di in ax_d.items():
        for col_key, ax in ax_di.items():
            
            #first row
            if row_key==row_keys[0]:
                ax.set_title(haz_label_d[col_key])
 
                
            #last row
            if row_key==row_keys[-1]:
                ax.set_xlabel('water depth (cm)')
                
            #first col
            if col_key==col_keys[0]:
                ax.set_ylabel(row_key)
    
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'wd_hist_{len(col_keys)}x{len(row_keys)}.svg')
    fig.savefig(ofp, dpi = 300,   transparent=True)
    log.info(f'wrote to \n    %s'%ofp)
    
    return ofp
    
    
     
if __name__=='__main__':
    


    plot_wd_hist_combine()

    
    
    #===========================================================================
    # plot_wd_hist_country(
    #     r'l:\10_IO\2307_funcAgg\outs\inters\03_concat\samp_concat_AUS_2333745.pkl','AUS',
    #     )
    #===========================================================================
 
    
    
    print('finished ')
    
    
    
    