"""playing with the hill function"""


import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

xar = np.linspace(0, 500, 50)

ka = max(xar)
rl_max = 100
def hill(l, n, ka=ka):
    """
    Calculates the fraction of receptor bound by ligand using the Hill function.
    
    Args:
    l: Free ligand concentration.
    n: Hill coefficient.
    ka: Dissociation constant. Defaults to 0.75.
    
    Returns:
    The fraction of receptor bound by ligand.
    """
    
    return (rl_max*2) / (1 + (ka / l)**n)

# Define the Hill coefficients
hill_coefficients = [0.2, 0.5,0.9, 1,1.1, 2] #+ np.linspace(1,2,5).tolist()



def hill_line(l):
    
    return l*hill(ka, 1.0)/ka
 


# Create a plot of the dose-response curves
plt.figure(figsize=(10,10))
for hill_coefficient in hill_coefficients:
    
    yar = hill(xar, hill_coefficient)
    plt.plot(xar, yar, label=f"Hill coefficient = {hill_coefficient}")
    
    
 
plt.plot(xar, hill_line(xar), label=f'line', color='black', linestyle='dashed')

# Add a dashed line at the Kd value
#===============================================================================
# plt.plot([0.001, ka, ka], [0.5, 0.5, 0], color="blue", linestyle="dashed")
# plt.annotate("Kd", xy=(ka * 1.1, 0.1), color="blue")
#===============================================================================
ax = plt.gca()
#ax.set_aspect('equal')
# Set the plot limits and labels
plt.xlim(-0.01, ka)
#plt.ylim(-0.01, ka)
plt.xlabel("Free Ligand Concentration")
plt.ylabel("Fraction of Receptor Bound by Ligand")

# Add a legend and title
plt.legend()
plt.title(f"Hill Function Dose-Response Curves ka={ka}")

# Show the plot

fig = plt.gcf()

ofp = os.path.join(r'l:\10_IO\2307_funcAgg\outs\damage\da', 'hill_funcs.svg')
fig.savefig(ofp,   transparent=True)

print(f'wrote to\n    {ofp}')





