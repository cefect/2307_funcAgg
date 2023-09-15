'''
Created on Sep. 15, 2023

@author: cefect

pre-processing on functions
'''
import os
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt


from coms import (
    init_log, today_str, get_directory_size,dstr, view
    ) 


from definitions import wrk_dir, haz_label_d, temp_dir



def _get_mdex(vidnm, df_d):
    # join some tabs
    df1 = df_d['damage_function'].set_index(vidnm)
    meta_d = dict()
    link_colns = set() # columns for just linking
    for tabName in [
        'damage_format', 
        'function_format', 
        'coverage', 
        'sector', 
        'unicede_occupancy', # property use type
        'construction_material', 
        'building_quality', 
        'number_of_floors', 
        'basement_occurance', 
        'precaution']:
        #===================================================================
        # #get id frame
        #===================================================================
        jdf = df_d[tabName].copy()
        # index
        assert vidnm == jdf.columns[0]
        if not jdf[vidnm].is_unique:
            raise AssertionError('non-unique indexer \'%s\' on \'%s\'' % (vidnm, tabName))
        jdf = jdf.set_index(vidnm)
        link_colns.update(jdf.columns)
        #===================================================================
        # #add descriptions
        #===================================================================
        container_tabName = '%s_container' % tabName
        assert container_tabName in df_d, container_tabName
        container_jdf = df_d[container_tabName]
        container_jcoln = container_jdf.columns[0]
        assert container_jcoln in jdf.columns
        jdf1 = jdf.join(container_jdf.set_index(container_jcoln), on=container_jcoln)
        #===================================================================
        # join to main
        #===================================================================
        df1 = df1.join(jdf1, on=vidnm)
        #===================================================================
        # meta
        #===================================================================
        meta_d[tabName] = {'shape':str(jdf1.shape), 'jcoln':vidnm, 'columns':jdf1.columns.tolist(), 
            'container_tabn':container_tabName, 
            'desc_colns':container_jdf.columns.tolist(), 
            'link_colns':jdf.columns.tolist()}
    
    # drop link columns
    df2 = df1.drop(link_colns, axis=1)
    
 
    
    return df2.reset_index()

def load_dfunc_serx(
        dfunc_fp = None,
        vidnm = 'df_id',
        ):
    
    """load the serx of all damage functions from tabular file
    
    see also C:\LS\09_REPOS\02_JOBS\2210_AggFSyn\aggF\coms\scripts.py
    """
    
    if dfunc_fp is None: 
        from definitions import dfunc_fp
        
    #===========================================================================
    # #load spreadsheet
    #===========================================================================
    print(f'loading from \n    {dfunc_fp}')
    df_d_raw = pd.read_excel(dfunc_fp, sheet_name=None)
    
    #clean empties
    df_d = {k:v for k,v in df_d_raw.items() if len(v)>0}
    
    print(f'removed empites {len(df_d_raw)-len(df_d)}/{len(df_d_raw)}')
    
    #===========================================================================
    # build index
    #===========================================================================
    mdex_df_raw = _get_mdex(vidnm, df_d)
    
    #apply pre-filters
    """this filter gives consistent data structures"""
    bx = np.logical_and(
        np.logical_and(
            mdex_df_raw['function_formate_attribute']=='discrete',
            mdex_df_raw['damage_formate_attribute']=='relative',
            ),
        mdex_df_raw['coverage_attribute']=='building',
        )
    
    mdex_df = mdex_df_raw[bx]
    print(f'filtered curves to get {bx.sum()}/{len(mdex_df_raw)}')
    
    #===========================================================================
    # def get_wContainer(tabName):
    #     df1 = df_d[tabName].copy()
    #     
    #     #=======================================================================
    #     # l = set(vid_df.index).difference(df1[vidnm])
    #     # df1[vidnm].unique()
    #     # if not len(l)==0:
    #     #     raise AssertionError('missing %i keys in \'%s\''%(len(l), tabName))
    #     #=======================================================================
    #     
    # 
    #     
    #     jdf = df_d[tabName+'_container'].copy()
    #     
    #     jcoln = jdf.columns[0]
    #     
    #     return df1.join(jdf.set_index(jcoln), on=jcoln) 
    # 
    # lkp_d = {k:get_wContainer(k) for k in ['wd', 'relative_loss']}
    # 
    # 
    # 
    # 
    # def get_by_vid(vid, k):
    #     try:
    #         df1 = lkp_d[k].groupby(vidnm).get_group(vid).drop(vidnm, axis=1).reset_index(drop=True)
    #     except Exception as e:
    #         raise AssertionError(e)
    #     #drop indexers
    #     bxcol = df1.columns.str.contains('_id')
    #     return df1.loc[:, ~bxcol]
    # 
    # #get_by_vid(26, 'wd')
    # get_by_vid(26, 'relative_loss')
    #===========================================================================
    #===========================================================================
    # build depth-damage frame
    #===========================================================================
    """I guess RL and WD are merged based on position (or maybe I'm missing some indexer?)"""
    #join water depths
    """because df_id is non-unique... need to use merge"""
    wd_df = df_d['wd'].join(
        df_d['wd_container'].set_index('wd_cont_id'), how='left', on='wd_cont_id'
        ).drop(['wd_cont_id', 'wd_id'], axis=1) 
        
        
    assert wd_df['wd_value'].notna().all()
    assert set(df_d['wd']['wd_cont_id']).difference(df_d['wd_container']['wd_cont_id'])==set()
    
    #add indexer
    wd_df['wd_rl_id'] = wd_df.groupby('df_id').cumcount()
    
    
    relative_loss_df = df_d['relative_loss'].join(
        df_d['relative_loss_container'].set_index('relative_loss_cont_id'), how='left', on='relative_loss_cont_id'
        ).drop(['relative_loss_cont_id', 'relative_loss_id'], axis=1)
    
    #add indexer
    relative_loss_df['wd_rl_id'] = relative_loss_df.groupby('df_id').cumcount()
    
    #merge with unique wd_rl_id and df_id columns
    wd_rl_df = pd.merge(wd_df, relative_loss_df, on=['df_id', 'wd_rl_id']).drop('wd_rl_id', axis=1).rename(columns={'wd_value':'wd', 'relative_loss_value':'rl'})
    
    
    #===========================================================================
    # join with index
    #===========================================================================
    
    df3=pd.merge(mdex_df,  wd_rl_df, on=vidnm, how='left')
    
    assert set(mdex_df['model_id']).difference(df3['model_id'])==set()
 
    # create mdex
 
    serx = df3.set_index(mdex_df.columns.to_list()+['wd']).swaplevel(i=0, j=1).sort_index(level=['model_id', 'df_id','wd'], sort_remaining=False).iloc[:,0]
    print(f'finished w/ {len(serx)} w/ model_id\n    %s'%serx.index.unique('model_id'))
    
    """
    view(dx.tail(1000))
    """
    
    return serx
    
 
 
def write_dfunc_serx(**kwargs):
    
    serx = load_dfunc_serx(**kwargs)
    
    serx.to_pickle(r'l:\10_IO\2307_funcAgg\ins\figueiredo2018\dfunc_lib_figu2018_20230906.pkl')
    

def prep_wagenaar2018(
        csv_fp = r'l:\10_IO\2307_funcAgg\ins\funcs\BN_lookup.csv',
        out_dir=None,
        
        ):
    
    """load and prep wagenaar_2018"""
    
    #===========================================================================
    # defaults
    #===========================================================================
    if out_dir is None:
        out_dir=os.path.join(wrk_dir, 'outs', 'funcMetrics', 'wagenaar2018', today_str)
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    
    #===========================================================================
    # load
    #===========================================================================
    
    df_raw = pd.read_csv(csv_fp, index_col=0)
    
    print(f'loaded {df_raw.shape} from \n    {csv_fp}')
    
    #===========================================================================
    # #build selector for all devaults
    #===========================================================================
    bx = pd.Series(True, index=df_raw.index)
    for coln in ['rp_range', 'd_range', 'pre_range', 'fe_range', 'ba_range','bt_range']:
        print(df_raw[coln].value_counts(dropna=False))
        
        #select rows of interest
        """nan values are used when no other info is available"""
        bx_i = df_raw[coln].isna() 
        
        #append slicer
        bx = np.logical_and(bx, bx_i)
        
        print(f'for \'{coln}\' selected {bx_i.sum()} entries with combined selection of {bx.sum()}')
        
    print(f'finished selection w/ {bx.sum()}/{len(bx)}')
    
    df1 = df_raw[bx].dropna(axis=1, how='all')
        
    #===========================================================================
    # clean
    #===========================================================================
    df2 = df1.dropna(subset=['wd_range'], axis=0).drop(['level', 'lower', 'upper'], axis=1)
    
    #split water depth
    wd_df = df2['wd_range'].str.replace('[','').str.replace(')','').str.replace(']',''
                                    ).str.split(',', expand=True).astype(float)
                                    
    wd_df.columns = ['wd_lower', 'wd_upper']
    
    
                                    
    df3 = df2.drop('wd_range', axis=1).join(wd_df.mean(axis=1).rename('wd_mean')
                            ).sort_values('wd_mean').rename(columns={'interpolation':'rl'})
    
    #===========================================================================
    # compute expected value for each depth 
    #===========================================================================
 
    wd_ev_df = df3['wd_mean'].to_frame().join((df3['rl']*df3['prob']
                               ).rename('rl (ev)')).groupby('wd_mean').sum()
                               
                               
    #write
    ofp = os.path.join(out_dir, f'wagenaar2018_{len(wd_ev_df)}_{today_str}.pkl')
    wd_ev_df.to_pickle(ofp)
    print(f'wrote {wd_ev_df.shape} to \n    {ofp}')
    #===========================================================================
    # plot
    #===========================================================================
    fig, ax = plt.subplots()
    
    scatter=ax.scatter(df3['wd_mean'], df3['rl'], s=df3['prob']*1000, alpha=0.5, label='relative loss (prob)', c='blue')
    
    ax.plot(wd_ev_df, color='red', label='relative loss (EV)')
    
    ax.set_xlabel('WSH (cm)')
    ax.set_ylabel('relative loss (frac)')
    
    
    # Create a legend
    #===========================================================================
    # # Generate a list of markers for the legend
    # markers = []
    # pser= df3['prob']
    # for prob in np.linspace(pser.min(), pser.max(), num=5):
    #     markers.append(plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='b', markersize=prob*100, alpha=0.5))
    # 
    # # Generate a list of labels for the legend
    # labels = np.round(np.linspace(pser.min(), pser.max(), num=5), 2)
    # 
    # # Add the legend to the plot
    # plt.legend(markers, labels, title="prob values", labelspacing=1.2)
    #===========================================================================
    # produce a legend with the unique colors from the scatter
    #===========================================================================
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                     loc="lower left", title="Probability")
    # ax.add_artist(legend1)
    #===========================================================================
    
    # produce a legend with a cross-section of sizes from the scatter
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5, color='b')
    
    new_lab=list()
    for lab in labels:
        sfx = str(int(lab.split('{')[1][:-2])/1000)+'}$'
        new_lab.append(lab[:14]+sfx)
        
    legend2 = ax.legend(handles, new_lab, loc="upper right", title="Probability")
    
    #write
    ofp = os.path.join(out_dir, f'wagenaar2018_{len(wd_ev_df)}_{today_str}.svg')
    fig.savefig(ofp, dpi = 300,   transparent=True)
    
    print(f'wrote to \n    {ofp}')
    
    """
    plt.show()
    df_raw.columns
    view(df_raw[bx])
    view(df1)
    view(df3)
    """
    


#===============================================================================
# def generate_scatter_plot(df):
#     
#  
# 
#     # Create a scatter plot
#     plt
#     
#     # Set the title and labels
#     plt.title('Scatter plot: wd_mean vs interpolation')
#     plt.xlabel('wd_mean')
#     plt.ylabel('interpolation')
#     
#     # Show the plot
#     plt.show()
#===============================================================================

 



    
    
    
if __name__ == '__main__':
    
    prep_wagenaar2018()  
