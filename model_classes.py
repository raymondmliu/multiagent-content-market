import matplotlib.pyplot as plt
import numpy as np

from mesa import Agent, Model
from scipy.optimize import minimize
from mesa.datacollection import DataCollector

from scheduler import AlternatingScheduler
from constants import *


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

      direct_util = R_P * sum(calc_direct_util(i) for i in range(self.model.num_members) if i != self.unique_id)
      inf_util = R_P * sum(calc_inf_util(i) for i in range(self.model.num_members) if i != self.unique_id)

      # Utility from outside sources
      out_util = R_0 * B_0 * np.exp(- ALPHA / x[-1]) if x[-1] else 0

      return - (direct_util + inf_util + out_util)
    
    x_0 = self.model.mems_alloc[self.unique_id]

    constraints = (
      {'type': 'ineq', 'fun': lambda x: self.model.M - np.sum(x)}
    )
    bounds = ((0.0, self.model.M) for _ in range(self.model.num_members + 2))

    result = minimize(calc_util, x_0, constraints=constraints, bounds=bounds, method=self.model.min_params['method'])
    self.consumer_utility = -result.fun
    self.model.mems_alloc[self.unique_id] = result.x

  def optimize_prod_util(self):
    """Calculate producer social support
    Given: this agent produces content on topic x[0]
    """
    # x[0]: the topic this agent produces content on
    def calc_util(x):
      def B(cons_id):
        return self.model.q(x[0], self.unique_id) * self.model.p(x[0], cons_id)
      
      # Social support received directly from consumers
      def calc_direct_util(cons_id):
        return B(cons_id) * np.exp(-ALPHA / self.model.mems_alloc[cons_id, self.unique_id]) \
          if self.model.mems_alloc[cons_id, self.unique_id] else 0

      # Social support received from influencer
      def calc_inf_util(cons_id):
        return B(cons_id) * np.exp(-ALPHA * \
                                   (1/self.model.infl_alloc[self.unique_id] + 1/self.model.mems_alloc[cons_id,-2])) \
          if (self.model.infl_alloc[self.unique_id] and self.model.mems_alloc[cons_id,-2]) else 0

      direct_util = R_P * sum(calc_direct_util(i) for i in range(self.model.num_members) if i != self.unique_id)
      inf_util = R_P * sum(calc_inf_util(i) for i in range(self.model.num_members) if i != self.unique_id)

      if self.model.is_perf: # Perfect information
        return (- direct_util - inf_util)
      else: # Limited information: only take into account social support received from influencer
        return (- inf_util)
    
    x_0 = [self.model.prod_topics[self.unique_id]]

    constraints = ()
    bounds = [(0.0, 1.0)]

    result = minimize(calc_util, x_0, constraints=constraints, bounds=bounds, method=self.model.min_params['method'])
    self.producer_utility = -result.fun
    self.model.prod_topics[self.unique_id] = result.x[0]
  
  def step(self):
    self.optimize_cons_util()
    self.optimize_prod_util()
    self.model.populate_B()
    if self.model.v:
      print(f"  Updated member {self.unique_id}")



class InfluencerAgent(Agent):
  def __init__(self, unique_id, model):
    self.unique_id = unique_id
    self.model = model

    self.utility = 0.0
  
  def optimize(self):
    """Calculate total social support received from content consumers who follow this influencer
    Given: influencer rate allocation x
    """
    def calc_util(x):
      # Calculate social support given a specific producer and consumer
      def calc_inf_util(prod_id, cons_id):
        return self.model.B[prod_id, cons_id] * np.exp(-ALPHA * (1/x[prod_id] + 1/self.model.mems_alloc[cons_id, -2])) \
          if (x[prod_id] and self.model.mems_alloc[cons_id, -2]) else 0

      inf_util = sum(sum(calc_inf_util(z, y) for z in range(self.model.num_members) if z != y) \
                     for y in range(self.model.num_members))
      return - inf_util

    x_0 = self.model.infl_alloc

    constraints = (
      {'type': 'ineq', 'fun': lambda x: self.model.M_INFL - np.sum(x)}
    )
    bounds = ((0.0, self.model.M_INFL) for _ in range(self.model.num_members))

    result = minimize(calc_util, x_0, constraints=constraints, bounds=bounds, method=self.model.min_params['method'])

    self.utility = -result.fun
    self.model.infl_alloc = result.x

  def step(self):
    self.optimize()
    if self.model.v:
      print("  Updated influencer")


class ContentMarketModel(Model):
  def __init__(self, params, main_topics, prod_topics, mems_alloc, infl_alloc, min_params, f, g):
    self.num_members = params['num_members']
    self.M = params['M']
    self.M_INFL = params['M_INFL']
    self.is_perf = params['is_perfect'] # whether model is perfect information
    self.v = params['verbose']
    self.f = f
    self.g = g

    self.min_params = min_params

    self.main_topics = main_topics
    self.initialize_rate_allocs(mems_alloc, infl_alloc, prod_topics)

    self.B = np.zeros((self.num_members, self.num_members))

    self.members = [MemberAgent(i, self) for i in range(self.num_members)]
    self.influencer = InfluencerAgent(self.num_members, self)

    self.schedule = AlternatingScheduler(self, params['influencer_update_frequency'])
    # create and schedule agents
    for m in self.members:
      self.schedule.add(m)
    self.schedule.add(self.influencer)

    self.datacollector = DataCollector(
      agent_reporters = {
        "Consumer Utility": lambda agent: agent.consumer_utility if isinstance(agent, MemberAgent) else None,
        "Producer Social Support": lambda agent: agent.producer_utility if isinstance(agent, MemberAgent) else None,
        "Influencer Utility": lambda agent: agent.utility if isinstance(agent, InfluencerAgent) else None,
      },
      model_reporters={"Total Social Welfare": lambda m: m.total_welfare}
    )
  
  """
  Set up rate allocations
  """
  def initialize_rate_allocs(self, mems_alloc, infl_alloc, prod_topics):
    # mems_alloc[i, j] = rate allocation of consumer i to producer j
    #   - The first model.num_members elements represent rate allocation for following content producers
    #   - Second last element: allocation for following influencer
    #   - Last element: allocation for following outside sources
    self.mems_alloc = mems_alloc
    for i in range(self.num_members):
      # An agent should not follow themself
      self.mems_alloc[i, i] = 0.0
      # Normalize the rate allocations to have less than sum M
      if np.sum(self.mems_alloc[i]) > self.M:
        self.mems_alloc[i] = self.mems_alloc[i] * (self.M / np.sum(self.mems_alloc[i]))

    # infl_alloc[i] = rate allocation of influencer to producer i
    self.infl_alloc = infl_alloc
    if np.sum(self.mems_alloc) > self.M:
      self.infl_alloc = self.infl_alloc * (self.M_INFL / np.sum(self.infl_alloc))

    # prod_topics[i] = topic that member i produces content on
    self.prod_topics = prod_topics
    
  def populate_B(self):
    for prod_id in range(self.num_members):
      for cons_id in range(self.num_members):
        self.B[prod_id, cons_id] = self.q(self.prod_topics[prod_id], prod_id) \
                                * self.p(self.prod_topics[prod_id], cons_id)

  def step(self):
    self.calc_total_social_welfare()
    self.datacollector.collect(self)
    if self.v:
      print(f'Step {self.schedule.steps}:')
    self.schedule.step()
  
  def calc_total_social_welfare(self):
    self.total_welfare = sum(m.consumer_utility for m in self.members)
    return self.total_welfare
  
  def p(self, content_item, cons_id):
    return self.f(np.absolute(content_item - self.main_topics[cons_id]))

  def q(self, content_item, prod_id):
    return self.g(np.absolute(content_item - self.main_topics[prod_id]))

