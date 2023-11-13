import numpy as np
from model_classes import ContentMarketModel, InfoAccessEnum
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(0)

default_min_params = {
    'method': 'SLSQP',
}

default_params = {
    'num_members': 10,          # Number of community members
    'M': 5.0,                   # Rate budget for consumer
    'M_INFL': 10.0,             # Rate limit for influencer
    'info_access': InfoAccessEnum.PERFECT,         # Perfect information or not
    'verbose': False, 
    'B_0': 0.75,                 # Prob. that content produced from outside sources is of interest to content consumers
    'ALPHA': 1.0,               # Delay sensitivity
    'R_P': 1.0,                 # Rate at which content producers create new content
    'R_0': 1.0,                 # Rate at which sources outside the community create new content
}

default_f = lambda x: 1 - 0.5 * x
default_g = lambda x: 1 - 0.5 * x

#default_f = lambda x: np.exp(-x)
#default_g = default_f

# default_main_topics = np.linspace(-1, 1, default_params['num_members'])
# default_prod_topics = np.zeros(default_params['num_members'])
# default_infl_alloc = np.ones(default_params['num_members']) / (default_params['M_INFL'] + 1)
# default_mems_alloc = np.ones((default_params['num_members'], default_params['num_members'] + 2)) / (default_params['M'] + 3)

"""Runs the model given the parameters"""
def run_model(num_members = default_params['num_members'], 
                M = default_params['M'],
                M_INFL = default_params['M_INFL'],
                info_access = default_params['info_access'],
                verbose = default_params['verbose'],
                B_0 = default_params['B_0'],
                ALPHA = default_params['ALPHA'],
                R_P = default_params['R_P'],
                R_0 = default_params['R_0'],
                f = default_f,
                g = default_g,
                is_random_init = False,
                max_num_steps = 100,
                convergence_steps = 10,
                ):
    
    params = {
        'num_members': num_members,          # Number of community members
        'M': M,                   # Rate budget for consumer
        'M_INFL': M_INFL,             # Rate limit for influencer
        'info_access': info_access,         # Perfect information or not
        'verbose': verbose, 
        'B_0': B_0,                 # Prob. that content produced from outside sources is of interest to content consumers
        'ALPHA': ALPHA,               # Delay sensitivity
        'R_P': R_P,                 # Rate at which content producers create new content
        'R_0': R_0,                 # Rate at which sources outside the community create new content
    }

    main_topics = np.linspace(-1.0, 1.0, num_members)

    if is_random_init:
        prod_topics = np.random.rand(num_members)
        infl_alloc = np.random.rand(num_members) 
        mems_alloc = np.random.rand(num_members, num_members + 2)
    else:
        # prod_topics = np.zeros(num_members)
        prod_topics = main_topics.copy()
        infl_alloc = np.ones(num_members) * (M_INFL / (num_members + 0))
        mems_alloc = np.ones((num_members, num_members + 2))  * (M / (num_members + 2))


    model = ContentMarketModel(
        params = params,
        main_topics = main_topics,
        prod_topics = prod_topics,
        infl_alloc = infl_alloc,
        mems_alloc = mems_alloc,
        min_params = default_min_params,
        f = f,
        g = g,
        conv_steps = convergence_steps,
    )


    model_converged = False

    for i in range(max_num_steps):
        total_welfare = model.total_welfare
        model_converged = model.step()
        if model_converged:
            break
    
    if not model_converged:
        print("Did not converge!")
    
    return model


def plot_topics(model, title, threshold=0.005, file=None):
    """Plot scalar values on a number line with clustering for close values and annotate with indices."""

    # First, sort the values along with their indices
    values = model.prod_topics.copy()
    indexed_values = sorted(enumerate(values), key=lambda x: x[1])

    clusters = []
    current_cluster = [indexed_values[0]]

    for i, v in indexed_values[1:]:
        # If the current value is close to the last value in the current cluster
        if v - current_cluster[-1][1] <= threshold:
            current_cluster.append((i, v))
        else:
            clusters.append(current_cluster)
            current_cluster = [(i, v)]
    clusters.append(current_cluster)

    plt.figure(figsize=(10, 2))
    plt.yticks([])  # Hide y-axis
    plt.xlim(min(values) - 0.5, max(values) + 0.5)
    plt.axhline(0, color='black', linewidth=0.5)  # Draw number line

    prev_text_position = None

    for cluster in clusters:
        center = sum(val for _, val in cluster) / len(cluster)  # Calculate the center of the cluster
        plt.plot(center, 0, 'ro', markersize=5 + 2 * len(cluster))  # Adjust marker size based on cluster size
        
        indices_str = ", ".join(str(index) for index, _ in sorted(cluster))
        
        # Adjusting text position to avoid overlap
        if prev_text_position is None or abs(center - prev_text_position) >= threshold:
            text_position = 0.01
        else:
            text_position += 0.015

        plt.text(center, text_position, indices_str, horizontalalignment='center', verticalalignment='bottom')
        prev_text_position = center

    plt.title(f"Producer topic allocation with {title}, social welfare = {model.total_welfare:.2f}")
    if file:
        plt.savefig(f'paper/figures/{file}_topics.jpg')
    plt.show()


def graph_model(model, title, file=None):
    # Create a new directed graph
    G = nx.DiGraph()

    """NODES
    """
    nodes = [f'C_{i}' for i in range(model.num_members)] # consumer nodes
    nodes.extend([f'P_{i}' for i in range(model.num_members)]) # producer nodes
    nodes.extend(['infl', 'out']) # influencer and outside sources nodes
    G.add_nodes_from(nodes)

    """EDGES
    """
    edges = []
    edge_colors = []
    # Add the edges consumer -> outside source
    for i in range(model.num_members):
        G.add_edge(f'C_{i}', 'out', weight=model.mems_alloc[i, -1], color='magenta')

    # Add the edges consumer -> influencer
    for i in range(model.num_members):
        G.add_edge(f'C_{i}', 'infl', weight=model.mems_alloc[i, -2], color='green')

    # Add the edges consumer -> producer
    for cons in range(model.num_members):
        for prod in range(model.num_members):
            G.add_edge(f'C_{cons}', f'P_{prod}', weight=model.mems_alloc[cons, prod], color='gray')

    # Add the edges influencer -> producer
    for i in range(model.num_members):
        G.add_edge(f'infl', f'P_{i}', weight=model.infl_alloc[i], color='red')

    """NODE POSITIONS IN GRAPH
    """
    # Consumers
    pos = {}
    scale = model.num_members * 0.5

    for i in range(model.num_members):
        pos[f'C_{i}'] = (-scale, i*scale)
        pos[f'P_{i}'] = (scale, i*scale)
    pos['infl'] = (0, model.num_members*scale/2)
    pos['out'] = (-scale*2, model.num_members*scale/2)

    plt.figure(figsize=(scale*1.5, scale*1.5))
    plt.title(f'Rate allocations with {title}, social welfare = {model.total_welfare:.2f}')

    nx.draw(G, pos, 
            with_labels=True, 
            node_size = 1000, 
            node_color='skyblue', 
            width=[data['weight'] * 3 for _, _, data in G.edges(data=True)], 
            edge_color=[data['color'] for _, _, data in G.edges(data=True)], 
            connectionstyle="arc3,rad=0")

    if file:
        plt.savefig(f'paper/figures/{file}_allocs.jpg')
    plt.show()

    plot_topics(model, title=title, file=file)
    
    #plot_number_line_with_indexed_clustering(model.prod_topics, threshold=0.005)