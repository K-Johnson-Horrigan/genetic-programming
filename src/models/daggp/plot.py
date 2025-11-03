def plot_nodes(nodes, result_fitness_func=None, labels=None, title=None, legend_title=None, **kwargs):
    """Plot all given nodes and the fitness function"""

    # Only plot the first domain
    xs = np.linspace(*kwargs['domains'][0])

    # Plot target function if given
    if 'target_func' in kwargs and result_fitness_func is not None:
        label = 'Target Function'
        target_ys = [kwargs['target_func'](x) for x in xs]
        plt.scatter(xs, target_ys, label=label)
        plt.plot(xs, target_ys, lw=5)

    # Plot nodes
    for i, node in enumerate(nodes):
        # Determine label based on what info is known
        if labels is not None:
            label = labels[i]
        elif 'test_kwargs' in kwargs:
            label = kwargs['test_kwargs'][i + 1][0]
        else:
            label = ''
        if 'target_func' in kwargs and result_fitness_func is not None:
            label += f' Fitness = {result_fitness_func([node], **kwargs)[0]:f}'

        # Evaluate and plot real part and imaginary part if applicable
        node_ys = [node(i, eval_method=kwargs['eval_method']) for i in xs]
        plt.scatter(xs, np.real(node_ys), label=label)
        plt.plot(xs, np.real(node_ys))
        if np.iscomplex(node_ys).any():
            label = label.split('Fitness')[0] + 'Imaginary Part'
            plt.scatter(xs, np.imag(node_ys), label=label)
            plt.plot(xs, np.imag(node_ys), ':')

    plt.title(title)
    plt.legend(title=kwargs['test_kwargs'][0][0])
    plt.savefig(f'{kwargs["saves_path"]}{kwargs["name"]}/plots/{title}.png')
    plt.show()
