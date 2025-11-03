def plot_tm_graph(trans, ax=None, scale=1, title=None, save=True, show=True, **kwargs):
    """Plot a TM as a graph"""

    if ax is None:
        fig, ax = plt.subplots()

    # Convert trans array into a list of transitions
    shape = trans.shape
    states = range(shape[0])
    symbols = range(shape[1])
    trans = [[state, symbol, *trans[(state, symbol)]] for state in states for symbol in symbols]

    # Convert a list of transitions into a list of vertices and edges
    verts = []
    edges = []
    edge_labels = {}
    for transition in trans:
        state0, symbol0, state1, symbol1, *move = transition
        # Add vertices and edges if they are not already present
        if state0 not in verts: verts.append(state0)
        if state1 not in verts: verts.append(state1)
        edge = (verts.index(state0), verts.index(state1))
        if edge not in edges:
            edges.append(edge)
        # Either create a label or append to an existing label
        edge_label = f'{symbol0}→{symbol1} (' + ','.join(map(str,move)) + ')'
        if edge in edge_labels:
            edge_labels[edge] += '\n' + edge_label
        else:
            edge_labels[edge] = edge_label

    # Create networkxs graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(len(verts)))
    G.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(G)
    connectionstyle = [f"arc3,rad={r}" for r in [.25, .75]]
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=range(len(verts)),
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
        # edge_labels = {edges[key]: label for key,label in enumerate(edge_labels)},
        edge_labels=edge_labels,
        alpha=0.5,
        # label_pos=0.0,
        # node_size=24000 * scale,
        bbox=None,
    )
    plt.title(title)
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    if show:
        plt.show()


def plot_tm_maze(trans, ax=None, title=None, save=True, show=True, **kwargs):
    """Plot the resulting tape after running the TM"""

    if ax is None:
        fig, ax = plt.subplots()

    tm = _run_maze_tm(trans, **kwargs)
    fit = maze_fitness([trans], **kwargs)[0]
    tape = tm.get_tape_as_array()

    colors = tm.state_history
    xy = list(zip(*tm.head_pos_history))[::-1]

    ax.set_title(f'{title} ({fit})')
    ax.plot(*xy)
    ax.scatter(*xy, c=colors)
    ax.imshow(tape)
    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    if show:
        plt.show()


def plot_tm_trans_array(trans, ax=None, title=None, save=True, show=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    # States and symbols can be inferred instead of using kwargs
    shape = trans.shape
    states = list(range(shape[0]))
    symbols = list(range(shape[1]))
    im = trans[:,:,1]
    # Append states
    im = np.concat(([list(symbols)], im), axis=0)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('States')
    ax.set_xlabel('Symbols')
    # ax.set_axis_off()
    ax.invert_yaxis()
    ax.set_xlim((-1.5, shape[1]-.5))
    # ax.set_ylim((-1.5, shape[0]+.5))
    for state in states:
        ax.text(-1, state+1, state, ha='center', va='center', color='k')
    for symbol in symbols:
        ax.text(symbol, 0, symbol, ha='center', va='center', color='k')
    # Scatter plot points
    # Initial values are used for the states axis
    xs = [-1] * shape[0]
    ys = [y+1 for y in states]
    c = states.copy()
    # Loop over data dimensions and create text annotations.
    for state in states:
        for symbol in symbols:
            xs.append(symbol)
            ys.append(state+1)
            c.append(trans[state, symbol, 0])
    # marker = ((-1,-1),(1,-1),(-1,1))
    marker = 'o'
    ax.scatter(xs, ys, marker='o', s=1000, c=c, edgecolors='black')
    ax.plot((-.5,-.5,shape[1]-.5),(shape[0]+.5,.5,.5), color='black')
    if save:
        plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    if show:
        plt.show()
