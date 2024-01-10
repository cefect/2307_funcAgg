'''
Created on Sep. 28, 2023

@author: cefect

computing the enveelop of loss functions
'''

#===============================================================================
# setup matplotlib
#===============================================================================

import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
 
#set teh styles
plt.style.use('default')
font_size=8
dpi=300
cm = 1 / 2.54

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
        'figure.figsize':(6*cm,6*cm),#typical full-page textsize for Copernicus and wiley (with 4cm for caption)
        'legend.title_fontsize':'large',
        'text.usetex':False,
        }.items():
            matplotlib.rcParams[k] = v
            
set_doc_style()


#===============================================================================
# imports---------
#===============================================================================
import os, math, hashlib
import pandas as pd
idx = pd.IndexSlice
import numpy as np


from datetime import datetime

import scipy.integrate
from scipy.stats import gaussian_kde
 
from definitions import wrk_dir, clean_names_d, haz_label_d, postgres_d, temp_dir

from coms import init_log, today_str, view
from coms_da import get_matrix_fig, _get_cmap, _hide_ax
from misc.func_prep import get_funcLib
from funcMetrics.coms_fm import (
    slice_serx, force_max_depth, force_zero_zero, force_monotonic, force_and_slice
    )

def run_calc_envelopes(fserx_raw=None,
                       out_dir=None,
                       dfid_l = None,
                       fscale=4*cm,
                       ):
    """calculate the depth weights"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'funcMetrics', 'envelope', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    log = init_log(fp=os.path.join(out_dir, today_str+'.log'), name='envelope')
 
    ylab_d = clean_names_d
    #===========================================================================
    # load curves
    #===========================================================================
    if fserx_raw is None: 
        fserx_raw = get_funcLib() #select functions
 
 
    #drop meta and add zero-zero
    fserx = force_and_slice(fserx_raw, log=log)
    
    #slice
    if not dfid_l is None:
        fserx = fserx.loc[fserx.index.get_level_values('df_id').isin(dfid_l)]
        
    #drop zeros
    fserx = fserx.loc[fserx.index.get_level_values('wd')>=0]
    
    #convert to cm
    index_df = fserx.index.to_frame().reset_index(drop=True)
    index_df['wd']=index_df['wd']*100
    fserx.index = pd.MultiIndex.from_frame(index_df)
 
 
    
    """
    view(fserx)
    """
    
    #dfid_l = fserx.index.unique('df_id').tolist()
    log.info(f'computing on {len(dfid_l)}')
    
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
    # setup figure--------
    #===========================================================================
    #create a figure with subplots (arranged vertically) one for each entry in dfidl
    fig, ax_ar = plt.subplots(1,len(dfid_l), figsize=( fscale*len(dfid_l),fscale ), layout='constrained')
    ax_d = dict(zip(dfid_l, ax_ar))

 
        
        

    #===========================================================================
    # loop and plot
    #===========================================================================
    res_d = dict()
    for df_id, gserx in fserx.groupby('df_id'):
        
        #sertup
        ser = gserx.droplevel(list(range(5))).sort_index()
        
        xar, yar = ser.index, ser.values
        
        ax = ax_d[df_id]
        
        #plot the functino
        ax.plot(xar, yar, color='black')
        
        #plot the envelope bottom  line
        #line from 0,0 to xmax, ymax
        linef = lambda x: yar[0] + (yar[-1] - yar[0]) * (x - xar[0]) / (xar[-1] - xar[0]) 
        
        ax.plot(xar, linef(xar), color='red')
        
        # draw the hatch between the function and the envelope bottom line
 
        
        ax.fill_between(xar, yar, 
                        y2=linef(xar), color='blue', alpha=0.1,
                        #where=(yar>=straight_line), 
                        )
        
        # compute the area of the envelope
        curve_area = scipy.integrate.trapz(yar, xar, dx=0.1)
        line_area = scipy.integrate.trapz(linef(xar), xar, dx=0.1)
        
        #=======================================================================
        # text
        #=======================================================================
        tstr = '$A_{f}$: %.2f'%curve_area
        tstr+='\n$A_{line}$: %.2f'%line_area
        tstr+='\n$A_{envelope}$: %.2f'%(curve_area-line_area)
        
        
        ax.text(0.5, 0.5, tstr, size=6,
                            transform=ax.transAxes, va='bottom', ha='center', 
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
        
        #=======================================================================
        # meta        
        #=======================================================================
        res_d[df_id] = {'curve':curve_area, 'line':line_area}
        
        #=======================================================================
        # post
        #=======================================================================
        
        ax.set_title(ylab_d1[df_id])
        ax.set_ylabel('RL (%)')
        ax.set_xlabel('WSH (cm)')
        
    #===========================================================================
    # wrap
    #===========================================================================
    df = pd.DataFrame.from_dict(res_d).T
    df['envelope'] = df['curve'] - df['line']
    
    
    ofp = os.path.join(out_dir, f'envelopes_{len(df)}_{today_str}')
    
    log.info(f'finished w/ {df.shape} and wrote to \n    {ofp}.svg')
    
    
    fig.savefig(ofp+'.svg', dpi = 300,   transparent=True,
                #edgecolor='black'
                )
    
    df.to_csv(ofp+'.csv')
 
 
    
    
    
if __name__=='__main__':
    dfunc_curve_l = [26, 380, 402, 941]
    
    run_calc_envelopes(dfid_l=dfunc_curve_l)