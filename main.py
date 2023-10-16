from agents import MemberAgent, InfluencerAgent
from model import MultiAgentModel

# Number of community members
MEMBERS_COUNT = 15
# Rate budget for content consumers
M = 1.0
# Rate budget for influencers
M_INFL = 1.0
# Prob. that content produced from outside sources is of interest to content consumers
B_0 = 1.0
# Delay sensitivity for content consumers
ALPHA = 1.0
# Rate at which content producers create new content
R_P = 1.0
# Rate at which sources outside the community create new content
R_0 = 1.0


print('hi I\'m here!')