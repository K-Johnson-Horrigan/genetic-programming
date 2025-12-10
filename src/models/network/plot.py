import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as tri


def plot_network(org=None, ax=None, scale=1, title='Network', save=True, show=False, **kwargs):
    """Plot a TM as a graph"""

    verts = kwargs['nodes']
    verts = [(x,y) for y,x in verts]
    edges = kwargs['links']
    edges = tuple([(int(u),int(v)) for u,v in edges])

    if ax is None:
        fig, ax = plt.subplots()

    # Create networkxs graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(verts)))
    G.add_edges_from(edges)
    pos = verts
    # connectionstyle = ''
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
        # arrowstyle="-|>",
        arrowstyle="-",
        edgelist=edges, # Specify edge order
        # connectionstyle=connectionstyle,
        arrowsize=20 * scale,
        # edge_color = edge_props,
        # edge_cmap = plt.cm.tab10,
        # edge_vmax = 9,
        width=2 * scale,
        # alpha=0.5,
        node_size=600 * scale,
    )
    if org is not None:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            ax=ax,
            # connectionstyle=connectionstyle,
            edge_labels = {edges[i]: org[i] for i,link in enumerate(org)},
            # edge_labels=tuple(int(i) for i in org),
            alpha=0.5,
        )
    plt.title(title)
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    if show:
        plt.show()











def contour(org, ax=None, scale=1, title='Network', save=True, show=False, **kwargs):

    x = []
    y = []
    z = []

    for i,link in enumerate(kwargs['links']):
        y0, x0 = kwargs['nodes'][link[0]]
        y1, x1 = kwargs['nodes'][link[1]]
        x.append((x0 + x1) / 2)
        y.append((y0 + y1) / 2)
        z.append(org[i])

        # w = 1
        # x.append((w*x0 + 1*x1) / (w+1))
        # y.append((w*y0 + 1*y1) / (w+1))
        # z.append(org[i])
        # x.append((1*x0 + w*x1) / (w+1))
        # y.append((1*y0 + w*y1) / (w+1))
        # z.append(org[i])

    fig, ax = plt.subplots()

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    # xi = np.linspace(0, kwargs['network_shape'][1], kwargs['network_shape'][1]*100)
    # yi = np.linspace(0, kwargs['network_shape'][0], kwargs['network_shape'][0]*100)
    # triang = tri.Triangulation(x, y)
    # interpolator = tri.LinearTriInterpolator(triang, z)
    # Xi, Yi = np.meshgrid(xi, yi)
    # zi = interpolator(Xi, Yi)
    # ax.contour(xi, yi, zi, levels=11, linewidths=0.5, colors='k')
    # cntr = ax.contourf(xi, yi, zi, levels=11, cmap="RdBu_r")

    ax.contour(x, y, z, levels=11, linewidths=0.5, colors='k')
    cntr = ax.contourf(x, y, z, levels=11, cmap="RdBu_r")

    # Use tricolor to interpolate
    # ax.tricontour(x, y, z, levels=12, linewidths=0.5, colors='k')
    # cntr = ax.tricontourf(x, y, z, levels=12, cmap="RdBu_r")

    # ax.plot(x, y, 'ko', ms=3)
    fig.colorbar(cntr, ax=ax)
    gap = 0.25
    ax.set(xlim=(0-gap, kwargs['network_shape'][1]-1+gap), ylim=(0-gap, kwargs['network_shape'][0]-1+gap))
    plot_network(org, ax=ax, scale=scale, title=title, save=save, show=show, **kwargs)

    # plt.subplots_adjust(hspace=0.5)
    # plt.show()





def contour(org, ax=None, scale=1, title='Network', save=True, show=False, **kwargs):

    x = []
    y = []
    z = []

    for i,link in enumerate(kwargs['links']):
        y0, x0 = kwargs['nodes'][link[0]]
        y1, x1 = kwargs['nodes'][link[1]]
        x.append((x0 + x1) / 2)
        y.append((y0 + y1) / 2)
        z.append(org[i])

        # w = 1
        # x.append((w*x0 + 1*x1) / (w+1))
        # y.append((w*y0 + 1*y1) / (w+1))
        # z.append(org[i])
        # x.append((1*x0 + w*x1) / (w+1))
        # y.append((1*y0 + w*y1) / (w+1))
        # z.append(org[i])

    fig, ax = plt.subplots()

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    xi = np.linspace(0, kwargs['network_shape'][1], kwargs['network_shape'][1]*100)
    yi = np.linspace(0, kwargs['network_shape'][0], kwargs['network_shape'][0]*100)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    ax.contour(xi, yi, zi, levels=10, linewidths=0.5, colors='k')
    cntr = ax.contourf(xi, yi, zi, levels=10, cmap="RdBu_r")

    # ax.contour(x, y, z, levels=11, linewidths=0.5, colors='k')
    # cntr = ax.contourf(x, y, z, levels=11, cmap="RdBu_r")

    # Use tricolor to interpolate
    # ax.tricontour(x, y, z, levels=13, linewidths=0.5, colors='k')
    # cntr = ax.tricontourf(x, y, z, levels=13, cmap="RdBu_r")

    # ax.plot(x, y, 'ko', ms=3)
    fig.colorbar(cntr, ax=ax)
    gap = 0.25
    ax.set(xlim=(0-gap, kwargs['network_shape'][1]-1+gap), ylim=(0-gap, kwargs['network_shape'][0]-1+gap))
    plot_network(org, ax=ax, scale=scale, title=title, save=save, show=show, **kwargs)

    # plt.subplots_adjust(hspace=0.5)
    # plt.show()