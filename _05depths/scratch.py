'''
Created on Sep. 25, 2023

@author: cefect
'''
#===============================================================================
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import numpy as np
# 
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import numpy as np
# 
# def create_subfigures(m, n, data):
#     fig = plt.figure(figsize=(n*6, m*6))
# 
#     for i in range(m):
#         for j in range(n):
#             x = data[i][j][0]
#             y = data[i][j][1]
# 
#             gs = gridspec.GridSpec(4, 4, figure=fig, left=j/n, right=(j+1)/n, bottom=(m-i-1)/m, top=(m-i)/m)
#             ax_main = fig.add_subplot(gs[1:4, :3])
#             ax_xDist = fig.add_subplot(gs[0, :3], sharex=ax_main)
#             ax_yDist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
# 
#             # scatterplot on main ax
#             ax_main.scatter(x, y, alpha=0.2)
# 
#             # histogram on the attached axes
#             ax_xDist.hist(x, 40, histtype='stepfilled', orientation='vertical', color='gray')
#             ax_xDist.invert_yaxis()
# 
#             ax_yDist.hist(y, 40, histtype='stepfilled', orientation='horizontal', color='gray')
#             ax_yDist.invert_xaxis()
# 
#     plt.show()
# 
# # example usage:
# m, n = 2, 2
# data = [[(np.random.rand(100), np.random.rand(100)) for _ in range(n)] for _ in range(m)]
# create_subfigures(m, n, data)
#===============================================================================

#===============================================================================
# import matplotlib.pyplot as plt
# 
# import matplotlib.gridspec as gridspec
# 
# 
# def format_axes(fig):
#     for i, ax in enumerate(fig.axes):
#         ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
#         ax.tick_params(labelbottom=False, labelleft=False)
# 
# 
# # gridspec inside gridspec
# fig = plt.figure()
# 
# gs0 = gridspec.GridSpec(1, 2, figure=fig)
# 
# gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])
# 
# ax1 = fig.add_subplot(gs00[:-1, :])
# ax2 = fig.add_subplot(gs00[-1, :-1])
# ax3 = fig.add_subplot(gs00[-1, -1])
# 
# # the following syntax does the same as the GridSpecFromSubplotSpec call above:
# gs01 = gs0[1].subgridspec(3, 3)
# 
# ax4 = fig.add_subplot(gs01[:, :-1])
# ax5 = fig.add_subplot(gs01[:-1, -1])
# ax6 = fig.add_subplot(gs01[-1, -1])
# 
# plt.suptitle("GridSpec Inside GridSpec")
# format_axes(fig)
# 
# plt.show()
#===============================================================================

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np


 

#===============================================================================
# np.random.seed(19680808)
# # gridspec inside gridspec
# fig = plt.figure(layout='constrained', figsize=(10, 10))
# subfigs = fig.subfigures(3, 3, wspace=0.07)
# 
# 
# for i, sf_l in enumerate(subfigs):
#     for j, subfig in enumerate(sf_l):
#         print(i,j) 
#         
#         # Define the gridspec (2x2)
#         gs = gridspec.GridSpec(2, 2, 
#                                height_ratios=[1,4], width_ratios=[4,1],
#                                )
#         
#         #main (lower left)
#         ax_main = subfig.add_subplot(gs[1, 0])
#         ax_main.pcolormesh(np.random.randn(3, 3), vmin=-2.5, vmax=2.5)
#  
#         
#         #hist x-vals (lower right)
#         ax_right = subfig.add_subplot(gs[1,1])
#         ax_right.hist(np.random.randn(100), orientation='horizontal', color='blue')
#         ax_right.set_xlabel('right')
#         
#         #hist y-vals
#         ax_top = subfig.add_subplot(gs[0,0])
#         ax_top.hist(np.random.randn(100), color='orange')
#         ax_top.set_xlabel('top')
#===============================================================================

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_scatter_wHist(data):
    """create an m x n scatter plot with histograms on the x and y axis"""
    fig = plt.figure(constrained_layout=True, figsize=(8, 8))
    subfigs = fig.subfigures(3, 3, wspace=0.07)
 
    for i, sf_l in enumerate(subfigs):
        for j, subfig in enumerate(sf_l):
            x = data[i][j][0]
            y = data[i][j][1]
 
            # Define the gridspec (2x2)
            gs = gridspec.GridSpec(2, 2, 
                                   height_ratios=[1,4], width_ratios=[4,1],
                                   figure=subfig,
                                   wspace=0.0, hspace=0.0)
 
            # Scatter plot (lower left)
            ax_main = subfig.add_subplot(gs[1, 0])
            ax_main.scatter(x, y)
 
            # Histogram of y values (lower right)
            ax_right = subfig.add_subplot(gs[1,1], 
                                          #sharey=ax_main, #cant use this with turning off/on the histograms
                                          )
            ax_right.hist(y, orientation='horizontal', color='blue')
            #ax_right.axis('off')
            ax_right.spines['left'].set_visible(False)
            ax_right.spines['right'].set_visible(False)
            ax_right.spines['top'].set_visible(False)
            ax_right.set_yticks([])
  
 
            # Histogram of x values (top)
            ax_top = subfig.add_subplot(gs[0,0], 
                                        #sharex=ax_main
                                        )
            ax_top.hist(x, color='orange')
            
            #ax_top.axis('off')
            ax_top.spines['bottom'].set_visible(False)
            ax_top.spines['top'].set_visible(False)
            ax_top.spines['right'].set_visible(False)
            ax_top.set_xticks([])
  
 
    plt.show()
 
# example usage:
data = [[(np.random.rand(100), np.random.rand(100)) for _ in range(3)] for _ in range(3)]
plot_scatter_wHist(data)

 
 
 












