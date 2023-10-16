import numpy as np
import warnings

from mesa import Agent, Model
from mesa.time import RandomActivation
from scipy.optimize import minimize
from mesa.datacollection import DataCollector

from constants import *

"""Helper functions
"""
f = lambda x: 1 - x
p = lambda content_item, cons_id: f(np.absolute(content_item - MAIN_TOPS[cons_id]))
g = lambda x: 1 - x
q = lambda content_item, prod_id: g(np.absolute(content_item - MAIN_TOPS[prod_id]))

class MemberAgent(Agent):
  def __init__(self, unique_id, model):
    self.unique_id = unique_id
    self.model = model

    self.consumer_utility = 0.0
    self.producer_utility = 0.0
  
  def optimize_cons_util(self):
    # x: the rate allocation for this agent
    def calc_util(x):
      # Utility obtained directly from other agents
      def calc_direct_util(prod_id):
          return self.model.B[prod_id, self.unique_id] * np.exp(-ALPHA / x[prod_id]) \
            if x[prod_id] else 0

      # Utility obtained from other agents through influencer
      def calc_inf_util(prod_id):
        return self.model.B[prod_id, self.unique_id] * np.exp(-ALPHA *  (1/x[-2] + 1/self.model.infl_alloc[prod_id])) \
          if (x[-2] and self.model.infl_alloc[prod_id]) else 0

      direct_util = R_P * sum(calc_direct_util(i) for i in range(MEMBERS_COUNT) if i != self.unique_id)
      inf_util = R_P * sum(calc_inf_util(i) for i in range(MEMBERS_COUNT) if i != self.unique_id)

      # Utility from outside sources
      out_util = R_0 * B_0 * np.exp(- ALPHA / x[-1]) if x[-1] else 0

      return - (direct_util + inf_util + out_util)
    
    x_0 = self.model.mems_alloc[self.unique_id]

    constraints = (
      {'type': 'ineq', 'fun': lambda x: M - np.sum(x)}
    )
    bounds = ((0.0, M) for _ in range(MEMBERS_COUNT + 2))

    result = minimize(calc_util, x_0, constraints=constraints, bounds=bounds)
    self.consumer_utility = -result.fun
    self.model.mems_alloc[self.unique_id] = result.x

  def optimize_prod_util(self):
    # x: the topic this agent produces content on
    def calc_util(x):
      def B(cons_id):
        return q(x[0], self.unique_id) * p(x[0], cons_id)
      
      # Social support received directly from consumers
      def calc_direct_util(cons_id):
        return B(cons_id) * np.exp(-ALPHA / self.model.mems_alloc[cons_id, self.unique_id]) \
          if self.model.mems_alloc[cons_id, self.unique_id] else 0


      # Social support received from influencer
      def calc_inf_util(cons_id):
        return B(cons_id) * np.exp(-ALPHA * \
                                   (1/self.model.infl_alloc[self.unique_id] + 1/self.model.mems_alloc[cons_id,-2])) \
          if (self.model.infl_alloc[self.unique_id] and self.model.mems_alloc[cons_id,-2]) else 0

      direct_util = R_P * sum(calc_direct_util(i) for i in range(MEMBERS_COUNT) if i != self.unique_id)
      inf_util = R_P * sum(calc_inf_util(i) for i in range(MEMBERS_COUNT) if i != self.unique_id)
      return (- direct_util - inf_util)
    
    x_0 = [self.model.prod_topics[self.unique_id]]

    constraints = ()
    bounds = [(0.0, 1.0)]

    result = minimize(calc_util, x_0, constraints=constraints, bounds=bounds)
    self.producer_utility = -result.fun
    self.model.prod_topics[self.unique_id] = result.x[0]
  
  def step(self):
    self.optimize_cons_util()
    self.optimize_prod_util()
    self.model.populate_B()



class InfluencerAgent(Agent):
  def __init__(self, unique_id, model):
    self.unique_id = unique_id
    self.model = model

    self.utility = 0.0
  
  def optimize(self):
    def calc_util(x):
      def calc_inf_util(prod_id, cons_id):
        return self.model.B[prod_id, cons_id] * np.exp(-ALPHA * (1/x[prod_id] + 1/self.model.mems_alloc[cons_id, -2])) \
          if (x[prod_id] and self.model.mems_alloc[cons_id, -2]) else 0

      inf_util = sum(sum(calc_inf_util(z, y) for z in range(MEMBERS_COUNT) if z != y) for y in range(MEMBERS_COUNT))
      return - inf_util

    x_0 = self.model.infl_alloc

    constraints = (
      {'type': 'ineq', 'fun': lambda x: M_INFL - np.sum(x)}
    )
    bounds = ((0.0, M_INFL) for _ in range(MEMBERS_COUNT))

    result = minimize(calc_util, x_0, constraints=constraints, bounds=bounds)

    self.utility = -result.fun
    self.model.infl_alloc = result.x

  def step(self):
    self.optimize()


class MultiAgentModel(Model):
  def __init__(self, mems_alloc, infl_alloc, prod_topics):
    """
    Set up rate allocations
    """
    # mems_alloc[i, j] = rate allocation of consumer i to producer j
    #   - The first MEMBERS_COUNT elements represent rate allocation for following content producers
    #   - Second last element: allocation for following influencer
    #   - Last element: allocation for following outside sources
    self.mems_alloc = mems_alloc
    for i in range(MEMBERS_COUNT):
      # An agent should not follow themself
      self.mems_alloc[i, i] = 0.0
      # Normalize the rate allocations to have less than sum M
      if np.sum(self.mems_alloc[i]) > M:
        self.mems_alloc[i] = self.mems_alloc[i] * (M / np.sum(self.mems_alloc[i]))

    # infl_alloc[i] = rate allocation of influencer to producer i
    self.infl_alloc = infl_alloc
    if np.sum(self.mems_alloc) > M:
      self.infl_alloc = self.infl_alloc * (M_INFL / np.sum(self.infl_alloc))

    # prod_topics[i] = topic that member i produces content on
    self.prod_topics = prod_topics

    self.B = np.zeros((MEMBERS_COUNT, MEMBERS_COUNT))

    self.schedule = RandomActivation(self)

    for i in range(MEMBERS_COUNT):
      community_member = MemberAgent(i, self)
      self.schedule.add(community_member)
    
    influencer = InfluencerAgent(MEMBERS_COUNT, self)
    self.schedule.add(influencer)
    
  def populate_B(self):
    for prod_id in range(MEMBERS_COUNT):
      for cons_id in range(MEMBERS_COUNT):
        self.B[prod_id, cons_id] = q(self.prod_topics[prod_id], prod_id) \
                                * p(self.prod_topics[prod_id], cons_id)


  def step(self):
    print(f"Doing step {self.schedule.steps}")
    self.schedule.step()

# Ensure results are reproducible
np.random.seed(0)

mems_alloc = np.random.rand(MEMBERS_COUNT, MEMBERS_COUNT + 2)
infl_alloc = np.random.rand(MEMBERS_COUNT)
prod_topics = np.random.rand(MEMBERS_COUNT)

model = MultiAgentModel(mems_alloc, infl_alloc, prod_topics)

for _ in range(MODEL_STEPS):
    model.step()