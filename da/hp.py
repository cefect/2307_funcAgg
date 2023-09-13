'''
Created on Aug. 26, 2023

@author: cefect

plot helpers
'''


import os, string
from datetime import datetime


import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

cm = 1 / 2.54


def _get_cmap(color_keys, name='Set1'):
    cmap = plt.cm.get_cmap(name=name)
    ik_d = dict(zip(color_keys, np.linspace(0, 1, len(color_keys))))
    hex = lambda x:matplotlib.colors.rgb2hex(x)
 
    return {k:hex(cmap(ni)) for k, ni in ik_d.items()}

def get_matrix_fig(  
                       row_keys, #row labels for axis
                       col_keys, #column labels for axis (1 per column)
                       
                       fig_id=0,
                       figsize=None, #None: calc using figsize_scaler if present
                       figsize_scaler=None,
                        #tight_layout=False,
                        constrained_layout=True,
                        set_ax_title=True, #add simple axis titles to each subplot
                        log=None,
                        add_subfigLabel=True,
                        fig=None,
                        **kwargs):
        
        """get a matrix plot with consistent object access
        
        Parameters
        ---------
        figsize_scaler: int
            multipler for computing figsize from the number of col and row keys
            
        add_subfigLabel: bool
            add label to each axis (e.g., A1)
            
        Returns
        --------
        dict
            {row_key:{col_key:ax}}
            
        """
        
        
        #=======================================================================
        # defautls
        #=======================================================================
 
        #special no singluar columns
        if col_keys is None: ncols=1
        else:ncols=len(col_keys)
        
        log.info(f'building {len(row_keys)}x{len(col_keys)} fig\n    row_keys:{row_keys}\n    col_keys:{col_keys}')
        
        #=======================================================================
        # precheck
        #=======================================================================
        """needs to be lists (not dict keys)"""
        assert isinstance(row_keys, list)
        #assert isinstance(col_keys, list)
        #=======================================================================
        # build figure
        #=======================================================================
        # populate with subplots
        if fig is None:
            if figsize is None: 
                if figsize_scaler is None:
                    figsize=matplotlib.rcParams['figure.figsize']
                else:
                    
                    figsize = (len(col_keys)*figsize_scaler, len(row_keys)*figsize_scaler)
                    
                    #fancy diagnostic p rint
                    fsize_cm = tuple(('%.2f cm'%(e/cm) for e in figsize))                    
                    log.info(f'got figsize={fsize_cm} from figsize_scaler={figsize_scaler:.2f} and col_cnt={len(col_keys)}')
                    
 
                
        
            fig = plt.figure(fig_id,
                figsize=figsize,
                #tight_layout=tight_layout,
                constrained_layout=constrained_layout,
 
                )
        else:
            #check the user doesnt expect to create a new figure
            assert figsize_scaler is None
            assert figsize is None
            assert constrained_layout is None
            assert fig_id is None
        

        #=======================================================================
        # add subplots
        #=======================================================================
        ax_ar = fig.subplots(nrows=len(row_keys), ncols=ncols, **kwargs)
        
        #convert to array
        if not isinstance(ax_ar, np.ndarray):
            assert len(row_keys)==len(col_keys)
            assert len(row_keys)==1
            
            ax_ar = np.array([ax_ar])
            
        
        #=======================================================================
        # convert to dictionary 
        #=======================================================================
        ax_d = dict()
        for i, row_ar in enumerate(ax_ar.reshape(len(row_keys), len(col_keys))):
            ax_d[row_keys[i]]=dict()
            for j, ax in enumerate(row_ar.T):
                ax_d[row_keys[i]][col_keys[j]]=ax
        
                #=======================================================================
                # post format
                #=======================================================================
                if set_ax_title:
                    if col_keys[j] == '':
                        ax_title = row_keys[i]
                    else:
                        ax_title='%s.%s'%(row_keys[i], col_keys[j])
                    
                    ax.set_title(ax_title)
                    
                    
                if add_subfigLabel:
                    letter=list(string.ascii_lowercase)[j]
                    ax.text(0.05, 0.95, 
                            '(%s%s)'%(letter, i), 
                            transform=ax.transAxes, va='top', ha='left',
                            size=matplotlib.rcParams['axes.titlesize'],
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", lw=0.0,alpha=0.5 ),
                            )
 
                
            
 
        log.info('built %ix%i w/ figsize=%s'%(len(col_keys), len(row_keys), figsize))
        return fig, ax_d

#===============================================================================
# PLOTERS----------
#===============================================================================


