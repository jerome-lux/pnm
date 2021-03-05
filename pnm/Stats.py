# coding=utf-8


import numpy as np


def stats(pn, bins=15, mode='vol', xscale='linear', elements='all', unit="nm", filename=None, show=True):
    """mode = 'count' or 'vol'
    """

    from warnings import warn
    import numpy as np
    import matplotlib as mpl
    import networkx as nx
    import matplotlib.pyplot as plt

    # p_radii = np.zeros(pn.graph.number_of_nodes())
    # t_radii = np.zeros(pn.graph.number_of_edges())
    # p_vol = np.zeros(pn.graph.number_of_nodes())
    # t_vol = np.zeros(pn.graph.number_of_edges())

    # for i,n in enumerate(pn.graph.nodes):
    # p_radii[i] = pn.graph.nodes[n]['radius']
    # p_vol[i] = pn.graph.nodes[n]['volume']

    # i = 0
    # for n1,n2 in pn.graph.edges:
    # t_radii[i] = pn.graph[n1][n2]['radius']
    # t_vol[i] = pn.graph[n1][n2]['volume']
    # i += 1

    p_radii = np.fromiter(nx.get_node_attributes(
        pn.graph, 'radius').values(), dtype=np.float)
    t_radii = np.fromiter(nx.get_edge_attributes(
        pn.graph, 'radius').values(), dtype=np.float)
    p_vol = np.fromiter(nx.get_node_attributes(
        pn.graph, 'volume').values(), dtype=np.float)
    t_vol = np.fromiter(nx.get_edge_attributes(
        pn.graph, 'volume').values(), dtype=np.float)

    minr = min(p_radii.min(), t_radii.min())
    maxr = max(p_radii.max(), t_radii.max())

    if xscale == 'log':

        try:
            _ = iter(bins)
        except TypeError:
            #exp_min = int(np.floor(np.log10(np.abs(minr))))
            #exp_max = int(np.floor(np.log10(np.abs(maxr))))
            exp_min = np.log10(np.abs(minr))
            exp_max = np.log10(np.abs(maxr))
            bins = list(np.logspace(exp_min, exp_max, bins))

        plt.xscale('log')

    elif xscale == 'linear':
        try:
            _ = iter(bins)
        except TypeError:
            bins = list(np.linspace(0, maxr, bins))

    else:
        warn("xscale should be either 'log', or 'linear'")

    data = None

    if mode == 'vol':
        data = np.zeros(len(bins)-1)
        # data2 = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            if elements == 'all':
                if i == 0:
                    data[i] = np.where((p_radii >= bins[i]) & (
                        p_radii <= bins[i+1]), p_vol, 0).sum()
                    data[i] += np.where((t_radii >= bins[i])
                                        & (t_radii <= bins[i+1]), t_vol, 0).sum()
                else:
                    data[i] = np.where((p_radii > bins[i]) & (
                        p_radii <= bins[i+1]), p_vol, 0).sum()
                    data[i] += np.where((t_radii > bins[i]) &
                                        (t_radii <= bins[i+1]), t_vol, 0).sum()

            elif elements == 'pores':
                # for n in pn.graph.nodes:
                # if pn.graph.nodes[n]['radius']>bins[i] and pn.graph.nodes[n]['radius']<=bins[i+1]:
                # data2[i] += pn.graph.nodes[n]['volume']
                if i == 0:
                    data[i] = np.where((p_radii >= bins[i]) & (
                        p_radii <= bins[i+1]), p_vol, 0).sum()
                else:
                    data[i] = np.where((p_radii > bins[i]) & (
                        p_radii <= bins[i+1]), p_vol, 0).sum()

            elif elements == 'throats':
                # for n1,n2 in pn.graph.edges:
                # if pn.graph[n1][n2]['radius']>bins[i] and pn.graph[n1][n2]['radius']<=bins[i+1]:
                # data2[i] += pn.graph[n1][n2]['volume']
                if i == 0:
                    data[i] = np.where((t_radii >= bins[i]) & (
                        t_radii <= bins[i+1]), t_vol, 0).sum()
                else:
                    data[i] = np.where((t_radii > bins[i]) & (
                        t_radii <= bins[i+1]), t_vol, 0).sum()

        data /= data.sum()

    elif mode == 'count':
        if elements == 'all':
            data, _ = np.histogram(
                np.hstack((p_radii, t_radii)), bins=bins, density=False)
        elif elements == 'pores':
            data, _ = np.histogram(p_radii, bins=bins, density=False)
        elif elements == 'throats':
            data, _ = np.histogram(t_radii, bins=bins, density=False)

        data = np.array(data)
        # print(data)

        if elements == 'all':
            data = data / (p_radii.size + t_radii.size)
        elif elements == 'pores':
            data = data / p_radii.size
        elif elements == 'throats':
            data = data / t_radii.size
    else:
        warn("mode should be either 'vol' or 'number'")
        return

    plt.plot(bins[1:], data)

    plt.grid(True)

    if mode == "count":
        title = "probability"
    else:
        title = "volume fraction"

    if elements == 'all':
        el = 'Pores and throats'
    elif elements == 'pores':
        el = 'Pores'
    elif elements == 'throats':
        el = 'Throats'

    plt.ylabel("{} {}".format(el, title))

    plt.xlabel('Pore radius ({})'.format(unit))

    if show:
        plt.show()

    if filename is not None:
        plt.savefig(filename)

    plt.clf()

    return (bins, data)


def distribution(n_samples, name='normal', high=np.inf, low=0, **kwargs):

    names = {'normal': np.random.normal,
             'uniform': np.random.uniform, 'weibull': np.random.weibull}

    return np.clip(names[name](**kwargs, size=n_samples), low, high)


def inverse_transform_sampling_from_raw_data(data, bins, n_samples=1000, low=0, high=np.inf, **kwargs):
    """data is the raw data distribution (may be not normalized) and bins a list of categories (in increasing order !!)"""
    import scipy.interpolate as interpolate
    data = np.array(data, dtype=np.float64)
    prob = data / data.sum()
    cum_prob = np.cumsum(prob, dtype=np.float64)
    inv_cdf = interpolate.interp1d(
        cum_prob, bins, bounds_error=None, fill_value='extrapolate')
    # return (bins[np.where(cum_prob > r)[0].min()] for r in np.random.rand(n_samples))
    return np.clip(inv_cdf(np.random.random(n_samples)), a_min=low, a_max=high)


def inverse_transform_sampling_from_cdf(cdf, bins, n_samples=1000, low=0, high=np.inf, **kwargs):
    """cdf is the cumulative density function bin_edges and cdf must have the same length"""
    import scipy.interpolate as interpolate
    inv_cdf = interpolate.interp1d(
        cum_prob, bins, bounds_error=None, fill_value='extrapolate')
    return np.clip(inv_cdf(np.random.random(n_samples)), a_min=low, a_max=high)
