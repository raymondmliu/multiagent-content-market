from numpy.random import rand

MEMBERS_COUNT = 25  # Number of community members
M = 10.0             # Rate budget for content consumers
M_INFL = 50.0        # Rate budget for influencers
B_0 = 1.0           # Prob. that content produced from outside sources is of interest to content consumers
ALPHA = 1.0         # Delay sensitivity for content consumers
R_P = 1.0           # Rate at which content producers create new content
R_0 = 1.0           # Rate at which sources outside the community create new content
MODEL_STEPS = 25    # Number of steps to run the model

# Main topic of interest for each member
MAIN_TOPS = rand(MEMBERS_COUNT)