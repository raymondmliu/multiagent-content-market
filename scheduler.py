from mesa.time import BaseScheduler
from enum import IntEnum
import random
import numpy as np

class AgentType(IntEnum):
  CONSUMER = 1
  INFLUENCER = 2
  PRODUCER = 3


class CustomScheduler(BaseScheduler):
    def __init__(self, model):
        super().__init__(model)

        self.agent_type_to_update = AgentType.CONSUMER

    """
    Update one class of agents in one step: consumers, influencer, or producer.
    Return whether the updated agents are within the convergence criteria of the previous agents
    """
    def step(self):
        if self.model.v:
            print(f'Updating {self.agent_type_to_update.name}')
        
        step_converged = False

        if (self.agent_type_to_update == AgentType.CONSUMER):
            cur_alloc = self.model.mems_alloc.copy()
            for n in range(self.model.num_members):
                self.model.consumers[n].step()
            step_converged = self.converged(cur_alloc, self.model.mems_alloc) 

        elif (self.agent_type_to_update == AgentType.INFLUENCER):
            cur_alloc = self.model.infl_alloc.copy()
            self.model.influencer.step()
            step_converged = self.converged(cur_alloc, self.model.infl_alloc)

        elif (self.agent_type_to_update == AgentType.PRODUCER):
            cur_alloc = self.model.prod_topics.copy()
            for n in range(self.model.num_members):
                self.model.producers[n].step()
            step_converged = self.converged(cur_alloc, self.model.prod_topics)
        else:
            raise ValueError()

        self.agent_type_to_update = AgentType((self.agent_type_to_update) % len(AgentType) + 1)
        
        self.steps += 1
        self.time += 1

        return step_converged

    def converged(self, arr1, arr2):
        rtol=1e-02 # default: 1e-05
        atol=1e-03 # default: 1--08
        return np.allclose(arr1, arr2, rtol=rtol, atol=atol)

        