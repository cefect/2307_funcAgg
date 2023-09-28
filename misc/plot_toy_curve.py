'''
Created on Sep. 28, 2023

@author: cefect

plot of toy example function
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
        'text.usetex':True,
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

 
 

from definitions import wrk_dir,  temp_dir

from coms import init_log, today_str, view



def hill(x, h, xmax, ymax):
    """modfieid hill function
    
    NOTE: throws a divide by zero warning and returns a zero
    """ 


    return  (ymax*2) / (1 + (xmax / x)**h)

def plot_toy_func(
        out_dir=None,
        wd_d = {
            1:0, 2:1.0, 3:0.25
            },
        func=hill,
        ):
    """plot a toy example of aggregation
    
    params
    -------
    wd_d: dict
        water depths
        
    func: callable
        loss function
    """
    
    #===========================================================================
    # defaults
    #===========================================================================
    start=datetime.now()
  
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'misc', 'toy', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
     
    log = init_log(fp=os.path.join(out_dir, today_str + '.log'), name='toy')
    
    wd_ser = pd.Series(wd_d).sort_values()
    #===========================================================================
    # setup the plot
    #===========================================================================
    fig, ax = plt.subplots(layout='constrained')
    """
    plt.show()
    """
    ax.set_ylabel('relative loss')
    ax.set_xlabel('water depth')
    #===========================================================================
    # plot the function with some typical parameters
    #===========================================================================
    
    
    h = 0.5  # hill coefficient
    xmax = wd_ser.max()  # x value at which function reaches half its maximum
    ymax = 100  # maximum y value
    
    x = np.linspace(0.0, xmax, 100)
    
    rl = lambda x: func(x, h, xmax, ymax)

 
    
    ax.plot(x, rl(x), label='$f(WSH)$', color='black')
    
    #===========================================================================
    # # Add hatching between rl(x) and straight line from (0,0) to (xmax, ymax)
    #===========================================================================
    straight_line = np.linspace(0, ymax, len(x))
    ax.fill_between(x, rl(x), straight_line, where=(rl(x)>=straight_line), 
                    interpolate=True, hatch=None, alpha=0.1, color='blue', label='envelope', linewidth=0)
    
    
    #===========================================================================
    # add the house losses
    #===========================================================================
    wd_rl_ser = pd.Series(rl(wd_ser.values), index=wd_ser.values)
    ax.plot(wd_rl_ser, color='orange', linestyle='none', marker='o', markersize=5, label='$asset_{i}$')
    
    xmean = wd_ser.mean()

    
    #===========================================================================
    # add the aggregate
    #===========================================================================
    
    ax.plot(xmean, rl(xmean), color='black', linestyle='none', marker='s', markersize=5, label='$asset_{j}$')   
    
    #===========================================================================
    # mean lines
    #===========================================================================
    linek = dict(color='black', linestyle='dashed', linewidth=0.75)
    
    #vertical (xmean)
    ax.axvline(xmean,   **linek)
    
    ax.text(xmean, 0.2, '$\overline{WSH_{i}}$', rotation=90, transform=ax.transAxes, ha='right')
    
    #horizontal (ymean)    
    ax.axhline(wd_rl_ser.mean(),  **linek)
    
    ax.text(0, wd_rl_ser.mean(), '$\overline{RL_{i}}$',    va='bottom', ha='left')
    
    #===========================================================================
    # label the gap
    #===========================================================================
    point1=(xmean, rl(xmean))
    point2=(point1[0], wd_rl_ser.mean())
    
    #add the lines
    d=0.5
    ax.annotate('', xy=(point1[0], point2[1]), xytext=(point1[0], point1[1]),
            arrowprops=dict(arrowstyle='<->', mutation_scale=d*20))
    
    
    #label
    ax.text(point1[0]*1.1, (point1[1]+point2[1])/2, f'Jensen\'s Gap',
        #rotation=90, verticalalignment='center',
        )
    
    #===========================================================================
    # post
    #===========================================================================
    ax.legend(frameon=False)
    
    #===========================================================================
    # write
    #===========================================================================
    ofp = os.path.join(out_dir, f'toy_wd_rl_{today_str}.svg')
    fig.savefig(ofp, dpi = dpi,   transparent=True,
                #edgecolor='black'
                )
    
    log.info(f'wrote to \n    {ofp}')
    
    return ofp
    
    
    
    
    
    



if __name__ == '__main__':
    plot_toy_func()
    
    
    
    
    