import numpy as np

from model_classes import MultiAgentModel
from constants import MEMBERS_COUNT, M, M_INFL, B_0, ALPHA, R_P, R_0, MAIN_TOPS

""" Create the model
"""

model = MultiAgentModel()

for _ in range(10):
    model.step()
