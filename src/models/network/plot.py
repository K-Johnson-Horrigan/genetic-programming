import networkx as nx
from matplotlib import pyplot as plt


def plot_network_problem(ax=None, scale=1, title='Network', save=True, show=False, **kwargs):
    """Plot a TM as a graph"""

    verts = kwargs['nodes']
    edges = kwargs['links']

    if ax is None:
        fig, ax = plt.subplots()

    # Create networkxs graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(verts)))
    G.add_edges_from(edges)
    pos = verts
    # connectionstyle = None
    connectionstyle = [f"arc3,rad={r}" for r in [.25, .75]]
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=range(len(kwargs['nodes'])),
        node_color='white',
        edgecolors='black',
        node_size=600 * scale,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        labels={key: vert for key, vert in enumerate(verts)},
        font_color='black',
        font_size=10 * scale,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrowstyle="-|>",
        edgelist=edges, # Specify edge order
        connectionstyle=connectionstyle,
        arrowsize=20 * scale,
        # edge_color = edge_props,
        # edge_cmap = plt.cm.tab10,
        # edge_vmax = 9,
        width=2 * scale,
        # alpha=0.5,
        node_size=600 * scale,
    )
    # nx.draw_networkx_edge_labels(
    #     G,
    #     pos,
    #     ax=ax,
    #     connectionstyle=connectionstyle,
    #     # edge_labels = {edges[key]: label for key,label in enumerate(edge_labels)},
    #     edge_labels=edge_labels,
    #     alpha=0.5,
    #     # labe
    plt.title(title)
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    if show:
        plt.show()


def plot_network(org, ax=None, scale=1, title='Network', save=True, show=False, **kwargs):
    """Plot a TM as a graph"""

    verts = kwargs['nodes']
    edges = kwargs['links']
    edges = tuple([(int(u),int(v)) for u,v in edges])

    if ax is None:
        fig, ax = plt.subplots()

    # Create networkxs graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(verts)))
    G.add_edges_from(edges)
    pos = verts
    # connectionstyle = None
    connectionstyle = [f"arc3,rad={r}" for r in [.25, .75]]
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=range(len(kwargs['nodes'])),
        node_color='white',
        edgecolors='black',
        node_size=600 * scale,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        labels={key: vert for key, vert in enumerate(verts)},
        font_color='black',
        font_size=10 * scale,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrowstyle="-|>",
        edgelist=edges, # Specify edge order
        connectionstyle=connectionstyle,
        arrowsize=20 * scale,
        # edge_color = edge_props,
        # edge_cmap = plt.cm.tab10,
        # edge_vmax = 9,
        width=2 * scale,
        # alpha=0.5,
        node_size=600 * scale,
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        ax=ax,
        connectionstyle=connectionstyle,
        edge_labels = {edges[i]: org[i] for i,link in enumerate(org)},
        # edge_labels=tuple(int(i) for i in org),
        alpha=0.5,
    )
    plt.title(title)
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    if show:
        plt.show()