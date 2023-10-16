from mesa import Agent, Model
from mesa.time import RandomActivation

class MultiAgentModel(Model):
    def __init__(self, num_agents_A, num_agents_B, learning_rate, max_iterations):
        self.num_agents_A = num_agents_A
        self.num_agents_B = num_agents_B
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.schedule = RandomActivation(self)

        # Create Type A agents
        for i in range(self.num_agents_A):
            agent = AgentA(i, self)
            self.schedule.add(agent)

        # Create Type B agents
        for i in range(self.num_agents_A, self.num_agents_A + self.num_agents_B):
            agent = AgentB(i, self)
            self.schedule.add(agent)

    def step(self):
        self.schedule.step()