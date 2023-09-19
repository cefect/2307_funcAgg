'''
Created on Sep. 18, 2023

@author: cefect
'''
import numpy as np
import pandas as pd


def get_rloss(dd_ar,
              xar,
              prec=None, #precision for water depths
              #from_cache=True,
              ):
    """compute relative loss for a function on an array of x vals
    
    speed ups:
        only interpolates unique x values (round with precision), then joins
    
    this should be the base function that can be parallelized if needed    
    2023-09-18: copied from       C:\LS\09_REPOS\02_JOBS\2210_AggFSyn\aggF\coms\scripts.py
    
    some functions like to return zero rloss
    
    Params
    -----------
    dd_ar: np.array
        function: dep_ar, dmg_ar = dd_ar[0], dd_ar[1]
        
    xar: np.array
        depths on which to compute loss
    
    """
    
    #===========================================================================
    # precheck
    #===========================================================================
    assert isinstance(xar, np.ndarray)
    
    if not np.all(np.diff(dd_ar)>=0):
        raise AssertionError(f'passed dd_ar is not monotonic\n{dd_ar}')
    assert dd_ar.shape[0]==2
    
    #=======================================================================
    # prep xvals
    #=======================================================================
    """using a frame to preserve order and resolution"""
    rdf = pd.Series(xar, name='wd_raw').to_frame().sort_values('wd_raw')
    
    if not prec is None:
        rdf['wd_round'] = rdf['wd_raw'].round(prec)
    else:
        rdf['wd_round'] = rdf['wd_raw']
        

    
    #=======================================================================
    # identify xvalues to interp
    #=======================================================================
    #get unique values
    xuq = rdf['wd_round'].unique()
 
 
    #filter thoe out of bounds
    #===========================================================================
    # bool_ar = np.full(len(xuq), True)
    # xuq1 = xuq[bool_ar]
    #===========================================================================
    #=======================================================================
    # interploate
    #=======================================================================
 
    res_ar = np.apply_along_axis(lambda x:np.interp(x,
                                dd_ar[0], #depths (xcoords)
                                dd_ar[1], #damages (ycoords)
                                left=0, #depth below range
                                right=max(dd_ar[1]), #depth above range
                                ),
                        0, xuq)
    

    #=======================================================================
    # plug back in
    #=======================================================================
    """may be slower.. but easier to use pandas for the left join here"""
    rdf = rdf.join(pd.Series(res_ar, index=xuq, name='loss'), on='wd_round')
    
    """"
    
    ax =  self.plot()
    
    ax.scatter(rdf['wd_raw'], rdf['rl'], color='red', s=30, marker='x')
    """
    
    #return to the original index order
    return rdf.sort_index()['loss'].values









