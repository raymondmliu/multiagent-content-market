import numpy as np

from mesa import Agent, Model
from mesa.time import RandomActivation

class MemberAgent:
  def __init__(self, unique_id, main_topic):
    self.unique_id = unique_id
    self.main_topic = main_topic

    """
    Rate allocation vector
    - The first MEMBERS_COUNT elements represent rate allocation for following
      content producers
    - Second last element: allocation for following influencer
    - Last element: allocation for following outside sources
    """
    self.rate_alloc = np.random.rand(MEMBERS_COUNT + 2)
    self.rate_alloc[self.unique_id] = 0.0 # An agent should not follow themself
    # Normalize the rate allocation based on the rate budget:
    self.rate_alloc = self.rate_alloc * (M / np.sum(self.rate_alloc))
    """
    Utilities
    """
    self.consumer_utility = 0.0
    self.producer_utility = 0.0


  def update_consumer_utility(self):
    def calc_direct_utility(producer_id):
      rate = self.rate_alloc[producer_id]
      if rate <= 0:
        return 0
      return R_P * B(producer_id, self.unique_id) * np.exp(-ALPHA/rate) * np.sign(rate)

    def calc_influencer_utility(producer_id):
      rate_infl_to_prod = influencer.rate_alloc[producer_id]
      rate_cons_to_infl = self.rate_alloc[-2]
      if (rate_infl_to_prod <= 0 or rate_cons_to_infl <= 0):
        return 0

      return R_P * B(producer_id, self.unique_id) * np.exp(-ALPHA * (1/rate_infl_to_prod) + (1/rate_cons_to_infl)) * np.sign(rate_cons_to_infl) * np.sign(rate_infl_to_prod)

    def calc_outside_utility():
      rate = self.rate_alloc[-1]
      if rate <= 0:
        return 0

      return R_0 * B_0 * np.exp(-ALPHA/rate) * np.sign(rate)

    res = 0
    for z in range(MEMBERS_COUNT):
      res += calc_direct_utility(z) + calc_influencer_utility(z)

    self.consumer_utility = res + calc_outside_utility()

  def constraint(self):
    np.sum(self.rate_alloc) < M

  def update_producer_utility(self):
    def calc_direct_utility(consumer_id):
      rate = members[consumer_id].rate_alloc[self.unique_id]
      if rate <= 0:
        return 0
      return R_P * B(self.unique_id, consumer_id) * np.exp(-ALPHA/rate) * np.sign(rate)

    def calc_influencer_utility(consumer_id):
      rate_infl_to_prod = influencer.rate_alloc[self.unique_id]
      rate_cons_to_infl = members[consumer_id].rate_alloc[-2]
      if (rate_infl_to_prod <= 0 or rate_cons_to_infl <= 0):
        return 0

      return R_P * B(self.unique_id, consumer_id) * np.exp(-ALPHA * (1/rate_infl_to_prod) + (1/rate_cons_to_infl)) * np.sign(rate_cons_to_infl) * np.sign(rate_infl_to_prod)

    res = 0
    for y in range(MEMBERS_COUNT):
      res += calc_direct_utility(y) + calc_influencer_utility(y)
    self.producer_utility = res

class InfluencerAgent:
  def __init__(self):
    # Rate allocation for following content producers
    self.rate_alloc = np.random.rand(MEMBERS_COUNT)
    # Normalize the rate allocation based on the rate budget:
    self.rate_alloc = self.rate_alloc * (M_INFL / np.sum(self.rate_alloc))

    self.social_support = 0.0

  def update_social_support(self):
    def calc_one_social_support(producer_id, consumer_id):
      rate_infl_to_prod = influencer.rate_alloc[producer_id]
      rate_cons_to_infl = members[consumer_id].rate_alloc[-2]
      if (rate_infl_to_prod <= 0 or rate_cons_to_infl <= 0):
        return 0

      return R_P * B(producer_id, consumer_id) * np.exp(-ALPHA * (1/rate_infl_to_prod) + (1/rate_cons_to_infl)) * np.sign(rate_cons_to_infl) * np.sign(rate_infl_to_prod)

    res = 0
    for y in range(MEMBERS_COUNT):
      for z in range(MEMBERS_COUNT):
        if z != y:
          res += calc_one_social_support(z, y)

    self.social_support = res

  def constraint(self):
    np.sum(self.rate_alloc) < M_INFL