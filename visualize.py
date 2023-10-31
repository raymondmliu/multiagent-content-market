"""
VISUALIZING THE CHANGE IN ALLOCATIONS OVER TIME
"""

import networkx as nx
from pyvis.network import Network
import ipywidgets as widgets
from IPython.display import display

import numpy as np
from model_classes import ContentMarketModel

from time import sleep

np.random.seed(0)
num_steps = 100

num_steps = 100

default_min_params = {
    'method': 'SLSQP',
}

default_params = {
    'num_members': 35,
    'M': 5.0,
    'M_INFL': 25.0,
    'is_perfect': True,
    'verbose': False,
    'influencer_update_frequency': 5
}

default_f = lambda x: np.exp(-x)
default_g = default_f

default_main_topics = np.random.rand(default_params['num_members'])
default_prod_topics = np.random.rand(default_params['num_members'])
default_infl_alloc = np.random.rand(default_params['num_members'])
default_mems_alloc = np.random.rand(default_params['num_members'], default_params['num_members'] + 2)

params = default_params.copy()
params['verbose'] = False
params['M'] = 5.0
params['M_INFL'] = 25.0
params['num_members'] = 15
params['influencer_update_frequency'] = 2
min_params = default_min_params.copy()

main_topics = np.random.rand(params['num_members'])
prod_topics = np.random.rand(params['num_members'])
infl_alloc = np.random.rand(params['num_members'])
mems_alloc = np.random.rand(params['num_members'], params['num_members'] + 2)

model = ContentMarketModel(
    params = params,
    main_topics = main_topics,
    prod_topics = prod_topics,
    infl_alloc = infl_alloc,
    mems_alloc = mems_alloc,
    min_params = min_params,
    f = default_f,
    g = default_g
)

graph_data = []
prod_topics_arr = []

for step_num in range(num_steps):
    step_dict = {}
    for m in range(params['num_members']):
        for n in range(params['num_members'] + 1):
            if m != n:
                step_dict[(m, n)] = model.mems_alloc[m, n]
    
    for n in range(params['num_members']):
        step_dict[(params['num_members'], n)] = model.infl_alloc[n]

    prod_topics_arr.append(list(model.prod_topics))
        
    graph_data.append(step_dict)

    model.step()


def plot_graph(timestep=0):
    G = nx.DiGraph()
    for edge, weight in graph_data[timestep].items():
        G.add_edge(edge[0], edge[1], weight=weight)

    net = Network(notebook=False)
    net.repulsion()

    net.from_nx(G)

    net.show('example.html')

# for i in range(0, num_steps):
plot_graph(0)

#steps = len(graph_data)
#step_slider = widgets.IntSlider(value=0, min=0, max=steps-1, step=1, description='Step:')
#widgets.interactive(plot_graph, step=step_slider)
widgets.interact(plot_graph, timestep=(0, len(graph_data) - 1))