import matplotlib.pyplot as plt
import numpy as np

# from mesa import Agent, Model
# from mesa.datacollection import DataCollector

from model_classes import ContentMarketModel
from constants import *

# Ensure results are reproducible
np.random.seed(0)

num_steps = 25
num_members = 50
M = 10.0
M_INFL = 50.0
mems_alloc = np.random.rand(num_members, num_members + 2)
infl_alloc = np.random.rand(num_members)
prod_topics = np.random.rand(num_members)

perf_model = ContentMarketModel(mems_alloc, infl_alloc, prod_topics, num_steps, num_members, M, M_INFL, True)
lim_model = ContentMarketModel(mems_alloc, infl_alloc, prod_topics, num_steps, num_members, M, M_INFL, False)

for _ in range(num_steps):
    perf_model.step()
    lim_model.step()

perf_agent_df = perf_model.datacollector.get_agent_vars_dataframe()
perf_model_df = perf_model.datacollector.get_model_vars_dataframe()

lim_agent_df = lim_model.datacollector.get_agent_vars_dataframe()
lim_model_df = lim_model.datacollector.get_model_vars_dataframe()

plt.figure(figsize=(10, 5))
plt.plot(perf_agent_df["Influencer Utility"].loc[(slice(None),num_members)], label="Influencer Utility - Perfect Information")
plt.plot(lim_agent_df["Influencer Utility"].loc[(slice(None),num_members)], label="Influencer Utility - Limited Information")
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()
plt.title("Trend of Influencer Utility Over Time")
plt.show()