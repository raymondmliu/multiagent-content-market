from mesa.time import BaseScheduler
import random

class AlternatingScheduler(BaseScheduler):
    def __init__(self, model, frequency):
        super().__init__(model)

        self.frequency = frequency # number of agents to be updated before influencer is updated
        self.counter_to_next_inf_step = self.frequency 
        self.cur_member_id = 0

        random.seed(0)

    def step(self):
        """Step the agents in an alternating pattern.
        For each member: update the member, then update the influencer."""

        if (self.counter_to_next_inf_step == 0): 
            # update influencer!
            self.model.influencer.step()
            self.counter_to_next_inf_step = self.frequency
        else: # update the next member agent
            if (self.cur_member_id == self.model.num_members):
                # reset to the first member
                self.cur_member_id = 0
            
            self.model.members[self.cur_member_id].step()
            
            self.cur_member_id += 1
            self.counter_to_next_inf_step -= 1
        
        self.steps += 1
        self.time += 1

        