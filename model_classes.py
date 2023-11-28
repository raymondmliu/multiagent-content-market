import matplotlib.pyplot as plt
import numpy as np

from enum import IntEnum
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from scipy.optimize import minimize

from scheduler import CustomScheduler

class InfoAccessEnum(IntEnum):
  PERFECT = 1
  LIMITED = 2

class ProducerAgent(Agent):
  def __init__(self, unique_id, mem_id, model):
    self.unique_id = unique_id
    self.mem_id = mem_id
    self.model = model

    self.utility = 0.0

  def optimize_prod_util(self):
    """Calculate producer social support
    Given: this agent produces content on topic x[0]
    """
    # x[0]: the topic this agent produces content on
    def calc_util(x):
      def B(cons_id):
        return self.model.q(x[0], self.mem_id) * self.model.p(x[0], cons_id)
      
      # Social support received directly from consumers
      def calc_direct_util(cons_id):
        return B(cons_id) * np.exp(-self.model.ALPHA / self.model.mems_alloc[cons_id, self.mem_id]) \
          if self.model.mems_alloc[cons_id, self.mem_id] else 0

      # Social support received from influencer
      def calc_inf_util(cons_id):
        return B(cons_id) * np.exp(-self.model.ALPHA * \
                                   (1/self.model.infl_alloc[self.mem_id] + 1/self.model.mems_alloc[cons_id,-2])) \
          if (self.model.infl_alloc[self.mem_id] and self.model.mems_alloc[cons_id,-2]) else 0

      direct_util = self.model.R_P * sum(calc_direct_util(i) for i in range(self.model.num_members) if i != self.mem_id)
      inf_util = self.model.R_P * sum(calc_inf_util(i) for i in range(self.model.num_members) if i != self.mem_id)

      if self.model.info_access == InfoAccessEnum.PERFECT: # Perfect information
        return (- direct_util - inf_util)
      elif self.model.info_access == InfoAccessEnum.LIMITED: # Limited information: only take into account social support received from influencer
        return (- inf_util)
      else:
        raise ValueError("Something went wrong here!")
    
    x_0 = [self.model.prod_topics[self.mem_id]]

    constraints = ()
    bounds = [(-1.0, 1.0)]

    result = minimize(calc_util, x_0, constraints=constraints, bounds=bounds, method=self.model.min_params['method'])
    # self.producer_utility = -result.fun
    self.model.prod_topics[self.mem_id] = result.x[0]

  
  def optimize_prod_util_limited(self):
    # x[0]: the topic this agent produces content on
    def calc_value(x):
      cur_prod_topics = self.model.prod_topics.copy()
      cur_prod_topics[self.mem_id] = x[0]

      B = self.model.calc_B(cur_prod_topics)

      """Optimize the utility of the influencer given the current content production allocation"""
      def optimize():
        # x: influencer rate allocation
        def calc_util(x):
          # Calculate social support given a specific producer and consumer
          def calc_inf_util(prod_id, cons_id):
            return B[prod_id, cons_id] * np.exp(-self.model.ALPHA * (1/x[prod_id] + 1/self.model.mems_alloc[cons_id, -2])) \
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

        utility = -result.fun
        infl_alloc = result.x
        return utility, infl_alloc
      
      utility, infl_alloc = optimize()
      value_maximized = np.exp(-self.model.ALPHA / infl_alloc[self.mem_id])  if infl_alloc[self.mem_id] else 0

      return value_maximized
                               
    x_0 = [self.model.prod_topics[self.mem_id]]

    constraints = ()
    bounds = [(-1.0, 1.0)]

    # result = minimize(calc_value, x_0, constraints=constraints, bounds=bounds, method=self.model.min_params['method'])
    # self.model.prod_topics[self.mem_id] = result.x[0]
    step = 0.01
    vals = [calc_value([x]) for x in np.arange(-1.0, 1.0, step)]
    result = np.argmax(vals)
    if vals[result] > 0:
      self.model.prod_topics[self.mem_id] = -1.0 + step*result
  
  def calc_prod_util(self):
    self.model.B = self.model.calc_B(self.model.prod_topics)

    def B(cons_id):
      return self.model.q(self.model.prod_topics[self.mem_id], self.mem_id) * self.model.p(self.model.prod_topics[self.mem_id], cons_id)
    
    # Social support received directly from consumers
    def calc_direct_util(cons_id):
      return B(cons_id) * np.exp(-self.model.ALPHA / self.model.mems_alloc[cons_id, self.mem_id]) \
        if self.model.mems_alloc[cons_id, self.mem_id] else 0

    # Social support received from influencer
    def calc_inf_util(cons_id):
      return B(cons_id) * np.exp(-self.model.ALPHA * \
                                  (1/self.model.infl_alloc[self.mem_id] + 1/self.model.mems_alloc[cons_id,-2])) \
        if (self.model.infl_alloc[self.mem_id] and self.model.mems_alloc[cons_id,-2]) else 0

    direct_util = self.model.R_P * sum(calc_direct_util(i) for i in range(self.model.num_members) if i != self.mem_id)
    inf_util = self.model.R_P * sum(calc_inf_util(i) for i in range(self.model.num_members) if i != self.mem_id)

    return direct_util + inf_util

  def step(self):
    if self.model.info_access == InfoAccessEnum.LIMITED:
      self.optimize_prod_util_limited()
      # self.optimize_prod_util()
    else:
      self.optimize_prod_util()
    
    self.utility = self.calc_prod_util()

    self.model.B = self.model.calc_B(self.model.prod_topics)
    if self.model.v:
      print(f"  Updated producer {self.mem_id}")
  


class ConsumerAgent(Agent):
  def __init__(self, unique_id, mem_id, model):
    self.unique_id = unique_id
    self.mem_id = mem_id
    self.model = model

    self.utility = 0.0
  
  def optimize_cons_util(self):
    # x: the rate allocation for this agent
    def calc_util(x):
      # Utility obtained directly from other agents
      def calc_direct_util(prod_id):
          return self.model.B[prod_id, self.mem_id] * np.exp(-self.model.ALPHA / x[prod_id]) \
            if x[prod_id] else 0

      # Utility obtained from other agents through influencer
      def calc_inf_util(prod_id):
        return self.model.B[prod_id, self.mem_id] * np.exp(-self.model.ALPHA *  (1/x[-2] + 1/self.model.infl_alloc[prod_id])) \
          if (x[-2] and self.model.infl_alloc[prod_id]) else 0

      direct_util = self.model.R_P * sum(calc_direct_util(i) for i in range(self.model.num_members) if i != self.mem_id)
      inf_util = self.model.R_P * sum(calc_inf_util(i) for i in range(self.model.num_members) if i != self.mem_id)

      # Utility from outside sources
      out_util = self.model.R_0 * self.model.B_0 * np.exp(- self.model.ALPHA / x[-1]) if x[-1] else 0

      return - (direct_util + inf_util + out_util)
    
    x_0 = self.model.mems_alloc[self.mem_id]

    constraints = (
      {'type': 'ineq', 'fun': lambda x: self.model.M - np.sum(x)}
    )
    bounds = ((0.0, self.model.M) for _ in range(self.model.num_members + 2))

    result = minimize(calc_util, x_0, constraints=constraints, bounds=bounds, method=self.model.min_params['method'])
    self.utility = -result.fun
    self.model.mems_alloc[self.mem_id] = result.x


  def calc_cons_util(self):
    # Utility obtained directly from other agents
    def calc_direct_util(prod_id):
        return self.model.B[prod_id, self.mem_id] * np.exp(-self.model.ALPHA / self.model.mems_alloc[self.mem_id, prod_id]) \
          if self.model.mems_alloc[self.mem_id, prod_id] else 0

    # Utility obtained from other agents through influencer
    def calc_inf_util(prod_id):
      return self.model.B[prod_id, self.mem_id] * np.exp(-self.model.ALPHA *  (1/self.model.mems_alloc[self.mem_id, -2] + 1/self.model.infl_alloc[prod_id])) \
        if (self.model.mems_alloc[self.mem_id, -2] and self.model.infl_alloc[prod_id]) else 0

    direct_util = self.model.R_P * sum(calc_direct_util(i) for i in range(self.model.num_members) if i != self.mem_id)
    inf_util = self.model.R_P * sum(calc_inf_util(i) for i in range(self.model.num_members) if i != self.mem_id)

    # Utility from outside sources
    out_util = self.model.R_0 * self.model.B_0 * np.exp(- self.model.ALPHA / self.model.mems_alloc[self.mem_id, -1]) \
                  if (self.model.mems_alloc[self.mem_id, -1]) else 0

    return direct_util + inf_util + out_util

  
  def step(self):
    self.optimize_cons_util()

    if self.model.v:
      print(f"  Updated consumer {self.mem_id}")



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
        return self.model.B[prod_id, cons_id] * np.exp(-self.model.ALPHA * (1/x[prod_id] + 1/self.model.mems_alloc[cons_id, -2])) \
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
    for c in self.model.consumers:
      c.utility = c.calc_cons_util()
    if self.model.v:
      print("  Updated influencer")


class ContentMarketModel(Model):
  def __init__(self, params, main_topics, prod_topics, mems_alloc, infl_alloc, min_params, f, g, conv_steps):
    self.num_members = params['num_members']
    self.M = params['M']
    self.M_INFL = params['M_INFL']
    self.info_access = params['info_access'] # whether model is perfect or limited information
    self.v = params['verbose']
    self.B_0 = params['B_0']
    self.ALPHA = params['ALPHA']
    self.R_P = params['R_P']
    self.R_0 = params['R_0']
    self.f = f
    self.g = g

    self.min_params = min_params

    self.main_topics = main_topics
    self.initialize_rate_allocs(mems_alloc, infl_alloc, prod_topics)

    self.B = np.zeros((self.num_members, self.num_members))

    self.producers = [ProducerAgent(f'P_{i}', i, self) for i in range(self.num_members)]
    self.consumers = [ConsumerAgent(f'C_{i}', i, self) for i in range(self.num_members)]
    self.influencer = InfluencerAgent('infl', self)

    self.schedule = CustomScheduler(self)
    # create and schedule agents
    for p in self.producers:
      self.schedule.add(p)
    for c in self.consumers:
      self.schedule.add(c)
    self.schedule.add(self.influencer)

    self.B = self.calc_B(self.prod_topics)
    for c in self.consumers:
      c.utility = c.calc_cons_util()
    self.calc_total_social_welfare()

    self.datacollector = DataCollector(
      agent_reporters = {
        "Consumer Utility": lambda agent: agent.utility if isinstance(agent, ConsumerAgent) else None,
        "Producer Social Support": lambda agent: agent.utility if isinstance(agent, ProducerAgent) else None,
        "Influencer Social Support": lambda agent: agent.utility if isinstance(agent, InfluencerAgent) else None,
      },
      model_reporters={"Total Social Welfare": lambda m: m.total_welfare}
    )

    self.conv_steps = conv_steps
    self.convergence_counter = 0
  
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
        print("Normalizing consumer allocation")
        self.mems_alloc[i] = self.mems_alloc[i] * (self.M / np.sum(self.mems_alloc[i]))

    # infl_alloc[i] = rate allocation of influencer to producer i
    self.infl_alloc = infl_alloc
    if np.sum(self.infl_alloc) > self.M_INFL:
      print("Normalizing influencer allocation")
      self.infl_alloc = self.infl_alloc * (self.M_INFL / np.sum(self.infl_alloc))

    # prod_topics[i] = topic that member i produces content on
    self.prod_topics = prod_topics
    
  def calc_B(self, prod_topics):
    result = np.zeros((self.num_members, self.num_members))
    for prod_id in range(self.num_members):
      for cons_id in range(self.num_members):
        result[prod_id, cons_id] = self.q(prod_topics[prod_id], prod_id) \
                                * self.p(prod_topics[prod_id], cons_id)

    return result

  def step(self):
    self.datacollector.collect(self)
    if self.v:
      print(f'Step {self.schedule.steps}:')

    model_converged = False
    step_converged = self.schedule.step()
    if step_converged:
      self.convergence_counter += 1
      if self.convergence_counter == self.conv_steps:
        model_converged = True
    else:
      self.convergence_counter = 0

    self.calc_total_social_welfare()

    return model_converged
  
  def calc_total_social_welfare(self):
    self.total_welfare = sum(c.utility for c in self.consumers)
    return self.total_welfare
  
  def p(self, content_item, cons_id):
    val = self.f(np.absolute(content_item - self.main_topics[cons_id]))
    if val > 1.0 or val < 0.0:
      raise ValueError()
    return val

  def q(self, content_item, prod_id):
    val = self.g(np.absolute(content_item - self.main_topics[prod_id]))
    if val > 1.0 or val < 0.0:
      raise ValueError()
    return val

