from numpy.random import rand

B_0 = 1.0           # Prob. that content produced from outside sources is of interest to content consumers
ALPHA = 1.0         # Delay sensitivity for content consumers
R_P = 1.0           # Rate at which content producers create new content
R_0 = 1.0           # Rate at which sources outside the community create new content


# Some helper functions
f = lambda x: 1 - x
g = lambda x: 1 - x