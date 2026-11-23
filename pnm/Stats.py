# coding=utf-8

from warnings import warn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.special as special


def stats(pn, bins=15, mode="vol", xscale="linear", elements="all", unit="nm", filename=None, show=True):
    """Compute various network statistics either weighted by nulber or by the volume of each element
    pn: pore network model
    mode = 'count' or 'vol'
    xscale: 'linear' or 'log'
    elements: 'all', 'pores' or 'throats'
    unit: length unit
    filename: if not None, save the graphs in filename
    show: plot the graphs if True [useful in a jupyter notebook]
    """

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

    p_radii = np.fromiter(nx.get_node_attributes(pn.graph, "radius").values(), dtype=float)
    t_radii = np.fromiter(nx.get_edge_attributes(pn.graph, "radius").values(), dtype=float)
    p_vol = np.fromiter(nx.get_node_attributes(pn.graph, "volume").values(), dtype=float)
    t_vol = np.fromiter(nx.get_edge_attributes(pn.graph, "volume").values(), dtype=float)

    minr = min(p_radii.min(), t_radii.min())
    maxr = max(p_radii.max(), t_radii.max())

    if xscale == "log":
        try:
            _ = iter(bins)
        except TypeError:
            # exp_min = int(np.floor(np.log10(np.abs(minr))))
            # exp_max = int(np.floor(np.log10(np.abs(maxr))))
            exp_min = np.log10(np.abs(minr))
            exp_max = np.log10(np.abs(maxr))
            bins = list(np.logspace(exp_min, exp_max, bins))

        plt.xscale("log")

    elif xscale == "linear":
        try:
            _ = iter(bins)
        except TypeError:
            bins = list(np.linspace(0, maxr, bins))

    else:
        warn("xscale should be either 'log', or 'linear'")

    data = None

    if mode == "vol":
        data = np.zeros(len(bins) - 1)
        # data2 = np.zeros(len(bins)-1)
        for i in range(len(bins) - 1):
            if elements == "all":
                if i == 0:
                    data[i] = np.where((p_radii >= bins[i]) & (p_radii <= bins[i + 1]), p_vol, 0).sum()
                    data[i] += np.where((t_radii >= bins[i]) & (t_radii <= bins[i + 1]), t_vol, 0).sum()
                else:
                    data[i] = np.where((p_radii > bins[i]) & (p_radii <= bins[i + 1]), p_vol, 0).sum()
                    data[i] += np.where((t_radii > bins[i]) & (t_radii <= bins[i + 1]), t_vol, 0).sum()

            elif elements == "pores":
                # for n in pn.graph.nodes:
                # if pn.graph.nodes[n]['radius']>bins[i] and pn.graph.nodes[n]['radius']<=bins[i+1]:
                # data2[i] += pn.graph.nodes[n]['volume']
                if i == 0:
                    data[i] = np.where((p_radii >= bins[i]) & (p_radii <= bins[i + 1]), p_vol, 0).sum()
                else:
                    data[i] = np.where((p_radii > bins[i]) & (p_radii <= bins[i + 1]), p_vol, 0).sum()

            elif elements == "throats":
                # for n1,n2 in pn.graph.edges:
                # if pn.graph[n1][n2]['radius']>bins[i] and pn.graph[n1][n2]['radius']<=bins[i+1]:
                # data2[i] += pn.graph[n1][n2]['volume']
                if i == 0:
                    data[i] = np.where((t_radii >= bins[i]) & (t_radii <= bins[i + 1]), t_vol, 0).sum()
                else:
                    data[i] = np.where((t_radii > bins[i]) & (t_radii <= bins[i + 1]), t_vol, 0).sum()

        data /= data.sum()

    elif mode == "count":
        if elements == "all":
            data, _ = np.histogram(np.hstack((p_radii, t_radii)), bins=bins, density=False)
        elif elements == "pores":
            data, _ = np.histogram(p_radii, bins=bins, density=False)
        elif elements == "throats":
            data, _ = np.histogram(t_radii, bins=bins, density=False)

        data = np.array(data)
        # print(data)

        if elements == "all":
            data = data / (p_radii.size + t_radii.size)
        elif elements == "pores":
            data = data / p_radii.size
        elif elements == "throats":
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

    if elements == "all":
        el = "Pores and throats"
    elif elements == "pores":
        el = "Pores"
    elif elements == "throats":
        el = "Throats"

    plt.ylabel("{} {}".format(el, title))

    plt.xlabel("Pore radius ({})".format(unit))

    if show:
        plt.show()

    if filename is not None:
        plt.savefig(filename)

    plt.clf()

    return (bins, data)


def _normalized_volume_histogram(radii, volumes, bins):
    """Normalized volume histogram of ``radii`` weighted by ``volumes`` on ``bins``.

    Uses the same bin convention as :func:`stats` (mode="vol"): bin i spans
    ``(bins[i], bins[i+1]]``, with the first bin closed at its lower edge.
    Values outside ``[bins[0], bins[-1]]`` are excluded.

    Parameters
    ----------
    radii : ndarray of float
        Element radii.
    volumes : ndarray of float
        Volume of each element (same order as ``radii``).
    bins : ndarray of float
        Bin edges (n_bins + 1 values, strictly increasing).

    Returns
    -------
    ndarray of float
        Normalized volume fraction per bin (sums to 1 if any value falls in range,
        zeros otherwise).
    """
    n = len(bins) - 1
    # searchsorted(side="left") - 1 implements the (a, b] convention; values exactly
    # equal to bins[0] must go into the first bin (closed at its lower edge).
    idx = np.searchsorted(bins, radii, side="left") - 1
    idx[radii == bins[0]] = 0
    mask = (idx >= 0) & (idx < n)
    hist = np.bincount(idx[mask], weights=volumes[mask], minlength=n)[:n]
    total = hist.sum()
    return hist / total if total > 0 else hist


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence KL(P||Q) between two discrete distributions.

    Uses the standard convention 0 * log(0 / q_i) = 0 for zero-mass bins of P. Returns +inf
    when P has mass in a bin where Q is exactly zero (disjoint supports).

    Parameters
    ----------
    p, q : np.ndarray
        Non-negative arrays summing to 1 (same length).

    Returns
    -------
    float
        KL(P||Q) in nats.
    """
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two discrete distributions (nats).

    JSD(P||Q) = 1/2 KL(P||M) + 1/2 KL(Q||M) with M = (P+Q)/2. Symmetric and bounded in
    [0, ln 2]; unlike raw KL it stays finite when the two distributions do not share a common
    support (e.g. one of them has no throats at all in its largest bins), which is exactly what
    happens with heavy-tailed MIP data.

    Parameters
    ----------
    p, q : np.ndarray
        Non-negative arrays summing to 1 (same length).

    Returns
    -------
    float
        JSD(P||Q) in [0, ln 2].
    """
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def save_distributions(pn, bins, out_base, pore_bins=None):
    """Save the pore and throat volume distributions of a network as two CSV files.

    The distributions are binned on ``bins`` (throats) and ``pore_bins`` (pores)
    with the same convention as :func:`stats` (mode="vol"), so the results match
    what the notebook plots for comparison against the experimental data. Each
    file contains one row per bin, using a semicolon separator to stay consistent
    with the experimental MIP CSV files in ``examples/data/``.

    Parameters
    ----------
    pn : pore_network
        The network whose distributions are saved (nodes and edges must have a
        ``radius`` attribute; missing ``volume`` attributes are filled from the
        geometry, 4/3 pi r^3 for pores and pi r^2 L for throats).
    bins : array-like of float
        Throat bin edges (n_bins + 1 values), typically the experimental throat
        radius bin edges used for comparison.
    out_base : str or Path
        Base file name without extension, e.g. the path passed to
        :meth:`pore_network.write`. The files ``{out_base}_pores.csv`` and
        ``{out_base}_throats.csv`` are written using this base as-is (including
        any directory part).
    pore_bins : array-like of float or None, optional
        Bin edges for the pore distribution. If None, pores are binned on
        ``bins`` like throats. Pore radii are typically ~PRTRR times larger than
        throat radii, so passing the throat bin edges scaled by PRTRR (e.g.
        ``throat_radii * PRTRR``) is recommended: otherwise the pore CSV only
        covers pores smaller than the largest throat bin edge and its fractions
        are normalized over that truncated subset.

    Returns
    -------
    dict
        Mapping ``{"pores": path, "throats": path}`` of the written CSV files.

    Notes
    -----
    Columns: ``radius_min; radius_max; mid_radius; volume_fraction; cumulative_volume_fraction``.
    ``volume_fraction`` is the normalized fraction of the total pore (or throat) volume in each
    bin; ``cumulative_volume_fraction`` accumulates from small to large radii.
    """
    bins = np.asarray(bins, dtype=float)
    pore_bins = bins if pore_bins is None else np.asarray(pore_bins, dtype=float)

    # Pores: radius + volume (missing volumes filled with 4/3 pi r^3 so the function
    # works even before compute_geometry() has run).
    p_radius = nx.get_node_attributes(pn.graph, "radius")
    p_volume = nx.get_node_attributes(pn.graph, "volume")
    p_labels = list(p_radius)
    p_radii = np.array([p_radius[n] for n in p_labels], dtype=float)
    p_vol = np.array([p_volume.get(n, 4.0 / 3.0 * np.pi * p_radius[n] ** 3) for n in p_labels], dtype=float)

    # Throats: radius + volume (missing volumes filled with pi r^2 L).
    t_radius = nx.get_edge_attributes(pn.graph, "radius")
    t_volume = nx.get_edge_attributes(pn.graph, "volume")
    t_length = nx.get_edge_attributes(pn.graph, "length")
    t_labels = list(t_radius)
    t_radii = np.array([t_radius[e] for e in t_labels], dtype=float)
    t_vol = np.array(
        [t_volume.get(e, np.pi * t_radius[e] ** 2 * t_length.get(e, 0.0)) for e in t_labels],
        dtype=float,
    )

    paths = {}
    for name, radii, vol, b in (
        ("pores", p_radii, p_vol, pore_bins),
        ("throats", t_radii, t_vol, bins),
    ):
        frac = _normalized_volume_histogram(radii, vol, b)
        cumfrac = np.cumsum(frac)
        path = f"{out_base}_{name}.csv"
        with open(path, "w") as f:
            f.write("radius_min;radius_max;mid_radius;volume_fraction;cumulative_volume_fraction\n")
            for i in range(len(b) - 1):
                mid = 0.5 * (b[i] + b[i + 1])
                f.write(f"{b[i]:.6g};{b[i + 1]:.6g};{mid:.6g};{frac[i]:.6g};{cumfrac[i]:.6g}\n")
        paths[name] = path

    return paths


def distribution(n_samples, name="normal", high=np.inf, low=0, **kwargs):
    names = {"normal": np.random.normal, "uniform": np.random.uniform, "weibull": np.random.weibull}

    return np.clip(names[name](**kwargs, size=n_samples), low, high)


def inverse_transform_sampling_from_raw_data(data, bins, n_samples=1000, low=0, high=np.inf, **kwargs):
    """Draw samples from a raw per-bin distribution using the inverse CDF method.

    ``data`` holds one (unnormalized) weight per bin and ``bins`` must hold the full bin
    edges: len(bins) == len(data) + 1 values in strictly increasing order, so that bin i
    spans [bins[i], bins[i+1]]. The inverse CDF is built by linearly interpolating the
    cumulative distribution at the bin edges — points (0, bins[0]) and
    (cumsum(data)[i], bins[i+1]) for each bin — which spreads each bin's mass uniformly
    over its own span [bins[i], bins[i+1]] with no extrapolation. This is the "inverse
    sampling" construction of the HDR thesis: the inverse of the piecewise-linear CDF of a
    piecewise-uniform distribution.

    Parameters
    ----------
    data : array-like of float
        Per-bin weights (may be unnormalized), length n_bins. Zero-weight bins are simply
        never sampled.
    bins : array-like of float
        Full bin edges in strictly increasing order, length n_bins + 1. Passing only the
        left (or right) edges is a common mistake: it shifts every bin's mass one bin down
        and extrapolates below the first edge.
    n_samples : int, optional
        Number of samples to draw (default 1000).
    low, high : float, optional
        Samples are clipped to [low, high] after the inverse transform (defaults 0 and inf).

    Returns
    -------
    ndarray of float
        n_samples values drawn from the piecewise-uniform distribution defined by data/bins.
    """
    data = np.asarray(data, dtype=float)
    bins = np.asarray(bins, dtype=float)
    if len(bins) != len(data) + 1:
        raise ValueError(
            f"bins must hold the full bin edges (len {len(data) + 1} for {len(data)} data values), "
            f"got len {len(bins)} — pass all n_bins + 1 edges, not one value per bin."
        )

    prob = data / data.sum()
    # Cumulative distribution at the bin edges: cum[i] = P(value <= bins[i+1]). Normalized so
    # the last value is exactly 1 (guards against float rounding of the final sum).
    cum = np.cumsum(prob, dtype=float)
    cum /= cum[-1]
    seg_start = np.concatenate(([0.0], cum))  # CDF values at every edge: [F(bins[0]), ..., F(bins[-1])]

    u = np.random.random(n_samples)
    # Segment i (spanning bins[i]) is selected for u in [seg_start[i], seg_start[i+1]).
    # side="right" makes zero-width segments (zero-weight bins) unselectable.
    i = np.clip(np.searchsorted(seg_start, u, side="right") - 1, 0, len(data) - 1)
    width = seg_start[1:] - seg_start[:-1]
    t = (u - seg_start[i]) / width[i]
    samples = bins[i] + t * (bins[i + 1] - bins[i])

    return np.clip(samples, a_min=low, a_max=high)


def _corrected_throat_counts(volumes, midradii, deltaR, e):
    """Step 1 of the pore-distribution inversions: throat volume -> count distribution.

    Each bin's experimental volume is divided by a characteristic cross-section raised to the
    empirical corrective power ``e`` and the result is normalized to sum to 1::

        cdist[i] = volumes[i] / (pi * (midradii[i]**2 + deltaR[i]**2/12))**e
        cdist /= cdist.sum()

    The term ``midradii[i]**2 + deltaR[i]**2/12`` is exactly the mean squared radius ``E[r^2]``
    of a radius drawn uniformly in bin i, so each bin's volume is divided by
    ``(pi * E[r^2])**e`` (``e = 1`` means no correction).

    Parameters
    ----------
    volumes : array-like of float
        Normalized experimental per-bin throat volumes (one value per bin, should sum to 1).
    midradii : array-like of float
        Throat bin centers (same length as ``volumes``).
    deltaR : array-like of float
        Throat bin widths (same length as ``volumes``); must equal the differences between
        consecutive bin edges.
    e : float
        Empirical corrective factor ("alpha"; 1 = no correction).

    Returns
    -------
    ndarray
        The corrected normalized throat COUNT distribution: one probability per bin, summing to 1.

    Raises
    ------
    ValueError
        If the three input arrays do not have the same length, or if the corrected throat
        distribution has no positive mass.
    """
    volumes = np.asarray(volumes, dtype=float).ravel()
    midradii = np.asarray(midradii, dtype=float).ravel()
    deltaR = np.asarray(deltaR, dtype=float).ravel()
    if not (volumes.size == midradii.size == deltaR.size):
        raise ValueError(
            "volumes, midradii and deltaR must have the same length, got "
            f"{volumes.size}, {midradii.size}, {deltaR.size}"
        )

    cdist = np.array(
        [volumes[i] / (np.pi * (midradii[i] ** 2 + (1.0 / 12) * deltaR[i] ** 2)) ** e for i in range(len(volumes))]
    )
    csum = float(cdist.sum())
    if not np.isfinite(csum) or csum <= 0:
        raise ValueError("corrected throat distribution has no positive mass")
    return cdist / csum


def _independent_recurrence(cdist):
    """Solve the independent-pairing model by backward substitution (largest bin first).

    Solves ``cdist[k] = x_k**2 + 2*x_k*sum(x[k+1:])`` for the pore COUNT distribution ``x``:
    ``x[-1] = sqrt(cdist[-1])``, then ``x[i] = -S_i + sqrt(S_i**2 + cdist[i])`` with
    ``S_i = sum(x[i+1:])`` (the positive root of a quadratic). Also used as the fixed-point
    initialization of :func:`invert_pore_distribution_kernel`.

    Parameters
    ----------
    cdist : ndarray
        Corrected normalized throat COUNT distribution (see :func:`_corrected_throat_counts`).

    Returns
    -------
    ndarray
        Pore COUNT distribution ``x``: one probability per bin, summing to 1.
    """
    x = np.zeros_like(cdist)
    for i in range(x.size - 1, -1, -1):
        s = float(x[i + 1 :].sum())
        x[i] = -s + np.sqrt(s * s + cdist[i])
    return x


def invert_pore_distribution_independent(volumes, midradii, deltaR, e):
    """Predict the pore size distribution from an experimental throat (MIP) distribution,
    assuming **independent pairing** of the two pores connected by a throat.

    Step 1 (:func:`_corrected_throat_counts`) converts the per-bin throat volumes into a throat
    COUNT distribution ``cdist`` with the empirical corrective factor ``e``:

        cdist[i] = volumes[i] / (pi * (midradii[i]**2 + deltaR[i]**2/12))**e,  then normalize.

    Step 2 inverts the pairing model under which the two pores of a throat are drawn
    independently from the pore distribution ``x`` and the throat class is set by the smaller
    endpoint (throat radius = min(r1, r2)/PRTRR)::

        cdist[k] = x_k**2 + 2*x_k*sum(x[k+1:]),   solved backward (largest bin first).

    Parameters
    ----------
    volumes : array-like of float
        Normalized experimental per-bin throat volumes (one value per bin, should sum to 1).
    midradii : array-like of float
        Throat bin centers (same length as ``volumes``).
    deltaR : array-like of float
        Throat bin widths (same length as ``volumes``); must equal the differences between
        consecutive bin edges.
    e : float
        Empirical corrective factor ("alpha"; 1 = no correction).

    Returns
    -------
    tuple(ndarray, ndarray)
        ``(cdist, pdist)``: the corrected normalized throat distribution and the predicted pore
        COUNT distribution (one probability per bin, summing to 1), both of length ``len(volumes)``.
        Feed ``pdist`` together with the full pore bin edges (``throat_radii * PRTRR``, i.e.
        n_bins + 1 values) to :func:`inverse_transform_sampling_from_raw_data`.

    Notes
    -----
    This is the classical model of ``examples/theory.typ`` and the routine historically used by
    ``examples/PNM-generation.ipynb`` and ``examples/param_study.py``. It assumes random (i.i.d.)
    pairing, but the actual generator pairs pores spatially under per-pore surface budgets: a
    large pore spends its budget on cheap nearby small throats before distant large partners are
    reached, so large-large pairs are rarer than independence predicts and this inversion
    systematically mis-estimates the pore distribution (large-throat deficit). For the corrected
    model see :func:`invert_pore_distribution_kernel` and ``doc/pore-distribution-inversion.md``.
    """
    cdist = _corrected_throat_counts(volumes, midradii, deltaR, e)
    return cdist, _independent_recurrence(cdist)


def invert_pore_distribution_kernel(
    volumes,
    midradii,
    deltaR,
    e,
    PRTRR=2.0,
    porosity=0.35,
    sratio=0.8,
    max_iter=500,
    tol=1e-12,
    verbose=False,
):
    """Predict the pore size distribution from an experimental throat (MIP) distribution with a
    spatial **exclusion kernel** that models the pairing actually performed by the generator.

    Step 1 is shared with :func:`invert_pore_distribution_independent`
    (:func:`_corrected_throat_counts`): each bin's volume is divided by
    ``(pi * (midradii**2 + deltaR**2/12))**e`` and normalized, yielding the throat COUNT
    distribution ``cdist`` (one probability per throat bin, summing to 1).

    Step 2 inverts the pairing model with an **exclusion kernel** that models the spatial
    pairing actually performed by :func:`pnm.Transformation.add_throats_by_surf` (see Notes for
    a summary of the derivation; full derivation in ``doc/pore-distribution-inversion.md``). The
    partner of a focal pore in class i is not drawn from ``x`` but from the size-conditioned
    distribution

      .. math::

          K(i \\to j) = \\frac{x_j\\, W_{ij}}{\\sum_k x_k\\, W_{ik}},
          \\qquad W_{ij} = \\exp\\!\\left(-\\left(\\frac{r_i + r_j}{D_i}\\right)^3\\right),

    where ``r_i`` is the mid-radius of pore bin i and ``D_i`` is the effective connection reach
    of a class-i pore, derived from its surface budget (no free parameter). The throat
    distribution identity becomes

      .. math::

          cdist[k] = x_k\\left[K(k{\\to}k) + \\sum_{j>k} K(k{\\to}j)\\right]
                    + \\sum_{j>k} x_j\\, K(j{\\to}k),

    which is solved by a fixed-point iteration with backward substitution (largest class first);
    the kernel is frozen at the previous iterate except for the diagonal self-term, which keeps
    each equation quadratic in ``x[k]``. The iteration is renormalized after every sweep and
    stops when the L-infinity change drops below ``tol``.

    Parameters
    ----------
    volumes : array-like of float
        Normalized experimental per-bin throat volumes (one value per bin, should sum to 1).
    midradii : array-like of float
        Throat bin centers (same length as ``volumes``).
    deltaR : array-like of float
        Throat bin widths (same length as ``volumes``); must equal the differences between
        consecutive bin edges.
    e : float
        Empirical corrective factor ("alpha"; 1 = no correction).
    PRTRR : float, optional
        Pore radius to throat radius ratio (throat radius = min(r1, r2) / PRTRR). Used to map
        throat bins onto pore bins in the kernel. Default is 2.0.
    porosity : float, optional
        Target pore-body porosity of the packing that will be generated. Enters the number
        density ``rho = 9*porosity/(4*pi*<r^3>)`` used by the kernel. Must match the ``porosity``
        argument passed to :func:`pnm.Generators.random_PNM`. Default is 0.35.
    sratio : float, optional
        Surface coverage ratio (fraction of a pore's surface covered by throats). Enters the
        per-class budget ``B_i = sratio * 4*pi*r_i**2`` used to derive the reach ``D_i``. Must
        match the ``sratio`` argument passed to :func:`pnm.Generators.random_PNM`. Default is 0.8.
    max_iter : int, optional
        Maximum number of fixed-point iterations. Default is 500.
    tol : float, optional
        Convergence threshold on the L-infinity change between successive iterates. Default is 1e-12.
    verbose : bool, optional
        If True, print one line per 25 iterations with the current residual. Default is False.

    Returns
    -------
    tuple(ndarray, ndarray)
        ``(cdist, pdist)``: the corrected normalized throat distribution and the predicted pore
        COUNT distribution (one probability per bin, summing to 1), both of length
        ``len(volumes)``. Feed ``pdist`` together with the full pore bin edges
        (``throat_radii * PRTRR``, i.e. n_bins + 1 values) to
        :func:`inverse_transform_sampling_from_raw_data` — exactly like the output of
        :func:`invert_pore_distribution_independent`.

    Raises
    ------
    ValueError
        If the three input arrays do not have the same length, or if the corrected throat
        distribution has no positive mass.

    Notes
    -----
    **Why the independent model fails.** The recurrence of theory.typ assumes that the two pores
    connected by a throat are drawn independently from the pore distribution ``x``. In the actual
    generator, pairing is spatial: pass 1 of :func:`add_throats_by_surf` scans each pore's
    candidates in increasing surface-to-surface distance and accepts a pair only while both
    pores' surface budgets (``sratio * 4*pi*r**2``) still fit the throat cross-section. A large
    focal pore therefore spends its budget on cheap nearby small throats long before a distant
    large partner is reached, so large-large pairs are far rarer than independence predicts and
    the generated network systematically under-produces large throats (the ">0.5 um band"
    deficit). The independent inversion compensates for nothing: it predicts an ``x`` that would
    reproduce the experiment only if pairing were random.

    **Kernel derivation.** Let a focal pore of class i have radius ``r_i`` and surface budget
    ``B_i = sratio * 4*pi*r_i**2``. A partner at center distance s is connected with probability
    ~ exp(-(s/D_i)^3): the number of neighbors within s grows as (4/3)*pi*s^3*rho, so the
    cumulative throat cost paid by pore i up to distance s scales like s^3 and its budget is
    exhausted when that cost reaches B_i. With mean throat cross-section to a partner drawn from
    ``x``,

        <S_t>_i = pi * E[min(r_i, R)^2] / PRTRR**2  (R ~ x),

    the reach is  D_i^3 = B_i / (rho * (4/3)*pi * <S_t>_i).
    The number of class-j partners in the shell [s, s+ds] around pore i is rho_j*4*pi*s^2 ds with
    hard-core exclusion s >= r_i + r_j. Integrating the survival probability over allowed
    distances gives (the integral has the closed form (4/3)*pi*D_i^3 * exp(-(a)^3) with
    a = (r_i+r_j)/D_i):

        K(i -> j)  ~  rho_j * exp(-((r_i + r_j)/D_i)^3),   rho_j = rho * x_j,

    which is the kernel above after row normalization. The number density follows from the
    packing that realizes ``x`` at porosity ``porosity`` (box scaled so total pore volume / box
    = porosity, as done by the kdtree packers): V_box = N*(4/3)*pi*<r^3>/porosity with
    <r^3> = sum_k x_k r_k**3, hence rho = 9*porosity/(4*pi*<r^3>).

    **Throat identity.** Counting undirected throats from directed connections (each pore has the
    same mean connectivity c; a throat's class is set by its smaller endpoint) gives
    ``cdist[k] = x_k*[K(k->k) + sum_{j>k} K(k->j)] + sum_{j>k} x_j*K(j->k)``, which sums to 1
    identically and reduces to the theory.typ expression when K(i->j) = x_j.

    **Fixed point.** The kernel depends on ``x`` (through its rows, their normalization and
    rho), so the inversion is a fixed-point iteration: freeze K at iterate l, solve each class's
    equation by backward substitution from the largest bin down — keeping the diagonal self-term
    ``K(k->k) = x_k * W[k,k]/rowsum_k`` coupled makes each equation quadratic in ``x[k]`` (the
    positive root is taken), which avoids the period-2 oscillation a fully frozen top class would
    exhibit. Negative right-hand sides are clamped to zero; the iterate is renormalized after
    every sweep so it stays on the simplex.

    **No new calibration parameter.** The kernel uses only quantities already fixed by the model:
    ``porosity``, ``sratio`` and ``PRTRR`` (plus the experimental data). In particular it replaces
    the ad-hoc ``pdist_shift`` gauge, which existed to absorb exactly this pairing bias.

    **Finite-size caveat.** The kernel is an infinite-system (Poisson-gas) idealization: it
    assumes every drawn pore can be placed. At finite N the largest bins of a wide window may be
    unplaceable in the box sized for the target porosity; those pores are dropped by the packer
    and the corresponding throat tail stays unreachable regardless of the inversion (a shared
    artifact, not a kernel defect).

    Full derivation with all intermediate steps: ``doc/pore-distribution-inversion.md``.
    """
    # ------------------------------------------------------------------
    # Step 1: volume -> count distribution with empirical corrective factor e.
    # (Shared with invert_pore_distribution_independent.)
    # ------------------------------------------------------------------
    cdist = _corrected_throat_counts(volumes, midradii, deltaR, e)
    n_bins = cdist.size

    # Pore bin mid-radii (pores are PRTRR times larger than throats).
    r = PRTRR * np.asarray(midradii, dtype=float).ravel()
    r3 = r**3

    # ------------------------------------------------------------------
    # Step 2 (kernel mode): exclusion-kernel fixed point.
    # ------------------------------------------------------------------
    tiny = np.finfo(float).tiny

    def _kernel_step(x):
        """One fixed-point sweep: build the kernel from ``x``, solve for the new distribution."""
        r3mean = float(np.dot(x, r3))
        if not (r3mean > 0.0) or not np.isfinite(r3mean):
            return None
        # Number density of the packing that realizes x at the target porosity.
        rho = 9.0 * porosity / (4.0 * np.pi * r3mean)
        # Mean squared throat radius to a partner drawn from x, per focal class i.
        m = np.minimum.outer(r, r) ** 2 @ x
        budget = sratio * 4.0 * np.pi * r**2
        mean_section = np.pi * m / PRTRR**2
        D3 = budget / (rho * (4.0 / 3.0) * np.pi * mean_section)
        D = np.cbrt(np.maximum(D3, tiny))

        # Pair weight W_ij = exp(-((r_i + r_j)/D_i)^3); the exponent is clipped so that
        # strongly suppressed pairs underflow to zero instead of raising a warning.
        s = r[:, None] + r[None, :]
        expo = -np.minimum((s / D[:, None]) ** 3, 745.0)
        W = np.exp(expo)

        # Row normalization is x-weighted: K[i, j] = x_j*W_ij / sum_k(x_k*W_ik), so that each
        # row of K sums to 1 (K is the partner-size distribution of a focal pore in class i).
        rowsum = W @ x
        K = x[None, :] * W / np.maximum(rowsum[:, None], tiny)
        K[rowsum <= 0.0] = 0.0

        xn = np.zeros(n_bins)
        for k in range(n_bins - 1, -1, -1):
            # cdist[k] = a_k*x_k^2 + b_k*x_k + C_k with the diagonal self-term kept coupled:
            #   a_k = W[k,k]/rowsum_k   (so K(k->k) = x_k * a_k exactly),
            #   b_k = sum_{j>k} K[k,j]  (frozen off-diagonal row entries),
            #   C_k = sum_{j>k} x_j*K[j,k] (already-solved larger classes, frozen column).
            a_k = W[k, k] / rowsum[k] if rowsum[k] > 0.0 else 0.0
            b_k = float(K[k, k + 1 :].sum())
            C_k = float(np.dot(xn[k + 1 :], K[k + 1 :, k])) if k < n_bins - 1 else 0.0
            rhs = cdist[k] - C_k
            if rhs <= 0.0:
                xn[k] = 0.0
            elif a_k > 0.0:
                # Positive root of a_k*x^2 + b_k*x - rhs = 0 (discriminant >= b_k^2 > 0).
                xn[k] = (-b_k + np.sqrt(b_k * b_k + 4.0 * a_k * rhs)) / (2.0 * a_k)
            elif b_k > 0.0:
                xn[k] = rhs / b_k
            else:
                xn[k] = 0.0

        total = float(xn.sum())
        return xn / total if total > 0.0 else None

    def _forward_residual(x):
        """Max |P_c(x) - cdist| with the kernel built from ``x`` (the true fixed-point residual)."""
        r3mean = float(np.dot(x, r3))
        if not (r3mean > 0.0):
            return np.inf
        rho = 9.0 * porosity / (4.0 * np.pi * r3mean)
        m = np.minimum.outer(r, r) ** 2 @ x
        D = np.cbrt(np.maximum(sratio * 4.0 * np.pi * r**2 / (rho * (4.0 / 3.0) * np.pi * (np.pi * m / PRTRR**2)), tiny))
        s = r[:, None] + r[None, :]
        W = np.exp(-np.minimum((s / D[:, None]) ** 3, 745.0))
        rowsum = W @ x
        K = x[None, :] * W / np.maximum(rowsum[:, None], tiny)
        pc = np.empty(n_bins)
        for k in range(n_bins):
            pc[k] = (
                x[k] * (K[k, k] + float(K[k, k + 1 :].sum()))
                + (float(np.dot(x[k + 1 :], K[k + 1 :, k])) if k < n_bins - 1 else 0.0)
            )
        return float(np.max(np.abs(pc - cdist)))

    x = _independent_recurrence(cdist).copy()
    err = np.inf
    for it in range(max_iter):
        xn = _kernel_step(x)
        if xn is None:
            break
        err = float(np.max(np.abs(xn - x)))
        x = xn
        if verbose and (it % 25 == 0 or err < tol):
            print(f"  invert_pore_distribution_kernel: iteration {it}: max |dx| = {err:.3e}")
        if err < tol:
            break

    # The iterate may stop moving while a few near-empty classes stay pinned at zero (their
    # right-hand side is a difference of ~1e-9 numbers), so judge success on the forward-model
    # residual instead of |dx|: it measures how well x actually reproduces cdist through K.
    resid = _forward_residual(x)
    if resid > 1e-3:
        warn(
            "invert_pore_distribution_kernel: fixed point not reached within {} iterations "
            "(forward-model residual {:.3e} > 1e-3); returning the last iterate.".format(max_iter, resid)
        )

    return cdist, x


def invert_pore_distribution_kernel_experimental(
    volumes,
    midradii,
    deltaR,
    e,
    PRTRR=2.0,
    porosity=0.35,
    sratio=0.8,
    zcap=None,
    max_iter=500,
    tol=1e-12,
    verbose=False,
):
    """EXPERIMENTAL — class-dependent coordination number in the exclusion-kernel inversion.

    Research variant of :func:`invert_pore_distribution_kernel` that drops the equal-connectivity
    assumption: instead of every pore sharing one mean degree (which cancels out of the throat
    identity), each class i carries its own mean-field degree

        z_i = B_i / <S_t>_i = 4*sratio*PRTRR^2 * r_i**2 / m_i,   m_i = sum_j x_j min(r_i, r_j)**2,

    which equals the floor ``4*sratio*PRTRR^2`` for small pores (budget saturated) and rises as
    ``r_i**2/<r^2>`` for large ones. The throat identity is then normalized by the mean degree
    ``zbar = sum_i x_i z_i`` (frozen per sweep). Full discussion: doc §5.5.

    **Why this is experimental and not used in production.** In generated networks the size-
    dependence of connectivity is a *rare-giant heavy-tail* effect: ~90 % of pores sit at a nearly
    flat degree (~10-12) and only a handful of giant pores reach high degrees (hundreds). The
    mean-field ``z_i`` above overshoots that tail by orders of magnitude, which makes the fixed
    point stiff — left unclamped it oscillates and does not converge. Pass ``zcap`` to clamp
    ``z_i`` from above to explore stabilized behaviour; even then the inverted distribution tends
    to push mass away from the large classes (a high-degree large pore "explains" many small
    throats, so fewer are needed), which works against the >0.5 um band goal. The production model
    therefore keeps equal connectivity (:func:`invert_pore_distribution_kernel`).

    Parameters
    ----------
    volumes : array-like of float
        Normalized experimental per-bin throat volumes (one value per bin, should sum to 1).
    midradii : array-like of float
        Throat bin centers (same length as ``volumes``).
    deltaR : array-like of float
        Throat bin widths (same length as ``volumes``); must equal the differences between
        consecutive bin edges.
    e : float
        Empirical corrective factor ("alpha"; 1 = no correction).
    PRTRR : float, optional
        Pore radius to throat radius ratio (throat radius = min(r1, r2) / PRTRR). Default is 2.0.
    porosity : float, optional
        Target pore-body porosity of the packing that will be generated. Must match the ``porosity``
        argument passed to :func:`pnm.Generators.random_PNM`. Default is 0.35.
    sratio : float, optional
        Surface coverage ratio (fraction of a pore's surface covered by throats). Must match the
        ``sratio`` argument passed to :func:`pnm.Generators.random_PNM`. Default is 0.8.
    zcap : float or None, optional
        Optional upper clamp on the per-class degree: ``z_i = min(z_i, zcap)``. ``None`` (default)
        leaves ``z_i`` unclamped — the raw mean-field value, which typically does not converge. A
        finite cap stabilizes the iteration for exploration but is an ad-hoc parameter.
    max_iter : int, optional
        Maximum number of fixed-point iterations. Default is 500.
    tol : float, optional
        Convergence threshold on the L-infinity change between successive iterates. Default is 1e-12.
    verbose : bool, optional
        If True, print one line per 25 iterations with the current residual. Default is False.

    Returns
    -------
    tuple(ndarray, ndarray)
        ``(cdist, pdist)``: same contract as :func:`invert_pore_distribution_kernel`.

    Raises
    ------
    ValueError
        If the three input arrays do not have the same length, or if the corrected throat
        distribution has no positive mass.
    """
    cdist = _corrected_throat_counts(volumes, midradii, deltaR, e)
    n_bins = cdist.size

    # Pore bin mid-radii (pores are PRTRR times larger than throats).
    r = PRTRR * np.asarray(midradii, dtype=float).ravel()
    r3 = r**3
    tiny = np.finfo(float).tiny

    def _degree(x):
        """Per-class mean-field degree ``z_i``, optionally clamped at ``zcap``."""
        m = np.minimum.outer(r, r) ** 2 @ x
        z = 4.0 * sratio * PRTRR**2 * r**2 / np.maximum(m, tiny)
        if zcap is not None:
            z = np.minimum(z, float(zcap))
        return z

    def _kernel_step(x):
        """One fixed-point sweep with class-dependent degree (zbar frozen at the current iterate)."""
        r3mean = float(np.dot(x, r3))
        if not (r3mean > 0.0) or not np.isfinite(r3mean):
            return None
        rho = 9.0 * porosity / (4.0 * np.pi * r3mean)
        m = np.minimum.outer(r, r) ** 2 @ x
        budget = sratio * 4.0 * np.pi * r**2
        mean_section = np.pi * m / PRTRR**2
        D = np.cbrt(np.maximum(budget / (rho * (4.0 / 3.0) * np.pi * mean_section), tiny))

        s = r[:, None] + r[None, :]
        W = np.exp(-np.minimum((s / D[:, None]) ** 3, 745.0))
        rowsum = W @ x
        K = x[None, :] * W / np.maximum(rowsum[:, None], tiny)
        K[rowsum <= 0.0] = 0.0

        z = _degree(x)
        zbar = float(np.dot(x, z))

        xn = np.zeros(n_bins)
        for k in range(n_bins - 1, -1, -1):
            # cdist[k]*zbar = a_k*x_k^2 + b_k*x_k + C_k with the diagonal self-term kept coupled:
            #   a_k = z_k*W[k,k]/rowsum_k, b_k = z_k*sum_{j>k} K[k,j],
            #   C_k = sum_{j>k} x_j*z_j*K[j,k] (already-solved larger classes).
            a_k = (W[k, k] / rowsum[k]) * z[k] if rowsum[k] > 0.0 else 0.0
            b_k = float(K[k, k + 1 :].sum()) * z[k]
            C_k = float(np.dot(xn[k + 1 :] * z[k + 1 :], K[k + 1 :, k])) if k < n_bins - 1 else 0.0
            rhs = cdist[k] * zbar - C_k
            if rhs <= 0.0:
                xn[k] = 0.0
            elif a_k > 0.0:
                # Positive root of a_k*x^2 + b_k*x - rhs = 0 (discriminant >= b_k^2 > 0).
                xn[k] = (-b_k + np.sqrt(b_k * b_k + 4.0 * a_k * rhs)) / (2.0 * a_k)
            elif b_k > 0.0:
                xn[k] = rhs / b_k
            else:
                xn[k] = 0.0

        total = float(xn.sum())
        return xn / total if total > 0.0 else None

    def _forward_residual(x):
        """Max |P_c(x) - cdist| with the class-dependent kernel built from ``x``."""
        r3mean = float(np.dot(x, r3))
        if not (r3mean > 0.0):
            return np.inf
        rho = 9.0 * porosity / (4.0 * np.pi * r3mean)
        m = np.minimum.outer(r, r) ** 2 @ x
        D = np.cbrt(np.maximum(sratio * 4.0 * np.pi * r**2 / (rho * (4.0 / 3.0) * np.pi * (np.pi * m / PRTRR**2)), tiny))
        s = r[:, None] + r[None, :]
        W = np.exp(-np.minimum((s / D[:, None]) ** 3, 745.0))
        rowsum = W @ x
        K = x[None, :] * W / np.maximum(rowsum[:, None], tiny)
        z = _degree(x)
        zbar = float(np.dot(x, z))
        pc = np.empty(n_bins)
        for k in range(n_bins):
            pc[k] = (
                x[k] * z[k] * (K[k, k] + float(K[k, k + 1 :].sum()))
                + (float(np.dot(x[k + 1 :] * z[k + 1 :], K[k + 1 :, k])) if k < n_bins - 1 else 0.0)
            ) / zbar
        return float(np.max(np.abs(pc - cdist)))

    x = _independent_recurrence(cdist).copy()
    err = np.inf
    for it in range(max_iter):
        xn = _kernel_step(x)
        if xn is None:
            break
        err = float(np.max(np.abs(xn - x)))
        x = xn
        if verbose and (it % 25 == 0 or err < tol):
            print(f"  invert_pore_distribution_kernel_experimental: iteration {it}: max |dx| = {err:.3e}")
        if err < tol:
            break

    resid = _forward_residual(x)
    if resid > 1e-3:
        warn(
            "invert_pore_distribution_kernel_experimental: fixed point not reached within {} iterations "
            "(forward-model residual {:.3e} > 1e-3); returning the last iterate.".format(max_iter, resid)
        )

    return cdist, x


def invert_pore_distribution_kernel_waterfill(
    volumes,
    midradii,
    deltaR,
    e,
    PRTRR=2.0,
    porosity=0.35,
    sratio=0.8,
    zcap=None,
    max_iter=500,
    tol=1e-12,
    verbose=False,
):
    """Predict the pore size distribution with a **water-filling** model of the generator's
    sequential capillary formation — the multi-capillary correction to the exclusion kernel.

    The production kernel (:func:`invert_pore_distribution_kernel`) treats each pore's connections
    as independent draws from one partner probability ``K(i -> j)`` with a class-independent mean
    degree, so the *number* of capillaries and their *size composition* are separable. In reality a
    pore creates several capillaries that share ONE surface budget: it scans candidates in
    increasing surface-to-surface distance and accepts each only while its remaining budget still
    fits the throat cross-section, so an expensive (large-partner) throat consumes budget that could
    have gone to another. This function models that greedy process directly and derives a JOINT
    (count, composition) per focal class instead of two independent factors.

    **Per-class connected-partner count.** For a focal pore in class i with radius ``r_i`` and
    surface budget ``B_i = sratio * 4*pi*r_i**2``, let ``S_t[i, j] = pi*min(r_i, r_j)**2/PRTRR**2``
    be the cross-section of a throat to a class-j partner and ``<S_t>_i = sum_j x_j*S_t[i, j]`` the
    mean cross-section to a partner drawn from ``x``. Sweeping candidates in distance order, the
    cumulative cost grows as ``(4/3)*pi*rho*<S_t>_i*s**3`` and runs out at the reach of the kernel;
    a class-j candidate at that boundary is accepted only if it still fits in the *remaining* budget.
    Integrating over allowed distances gives the closed-form per-class count

      .. math::

          n_{ij} = x_j\\, \\frac{B_i - S_t[i, j]}{\\langle S_t\\rangle_i},
          \\qquad z_i = \\sum_j n_{ij} = B_i/\\langle S_t\\rangle_i - 1.

    Two consequences distinguish this from the production kernel and from the class-dependent-degree
    variant (:func:`invert_pore_distribution_kernel_experimental`):

    - **The degree is size-dependent by construction** (``z_i = B_i/<S_t>_i - 1``), not an assumed
      constant: a large pore spends its budget on cheap small throats and forms more of them. It is
      derived from the budget, so it does not introduce a free parameter.
    - **The composition is corrected per class**, not just rescaled: expensive classes (large
      partners, ``S_t[i, j]`` large) are suppressed by ``(B_i - S_t[i, j])`` because they must fit in
      the budget left after cheaper closer throats are paid for. The experimental variant only scales
      every class by one common degree factor and leaves the composition untouched; here count and
      composition come from a single object.

    **Throat identity (budget-coupled).** Counting undirected throats from directed connections with
    per-class counts ``n_ij`` (no separate degree factor) gives

      .. math::

          cdist[k] = \\frac{x_k\\sum_{j\\ge k} n_{kj} + \\sum_{j>k} x_j\\, n_{jk}}{\\bar z},
          \\qquad \\bar z = \\sum_i x_i z_i.

    which is solved by the same backward-substitution fixed point as the production kernel (largest
    class first, diagonal self-term kept coupled so each equation stays quadratic in ``x[k]``). The
    iteration is renormalized after every sweep and stops when the L-infinity change drops below
    ``tol``; success is judged on the forward-model residual.

    Parameters
    ----------
    volumes : array-like of float
        Normalized experimental per-bin throat volumes (one value per bin, should sum to 1).
    midradii : array-like of float
        Throat bin centers (same length as ``volumes``).
    deltaR : array-like of float
        Throat bin widths (same length as ``volumes``); must equal the differences between
        consecutive bin edges.
    e : float
        Empirical corrective factor ("alpha"; 1 = no correction) used by step 1.
    PRTRR : float, optional
        Pore radius to throat radius ratio (throat radius = min(r1, r2) / PRTRR). Default is 2.0.
    porosity : float, optional
        Target pore-body porosity of the packing that will be generated. Enters the number density
        ``rho`` used by the kernel reach. Must match the ``porosity`` argument passed to
        :func:`pnm.Generators.random_PNM`. Default is 0.35.
    sratio : float, optional
        Surface coverage ratio (fraction of a pore's surface covered by throats). Enters the per-class
        budget ``B_i = sratio * 4*pi*r_i**2``. Must match the ``sratio`` argument passed to
        :func:`pnm.Generators.random_PNM`. Default is 0.8.
    zcap : float or None, optional
        Optional upper clamp on the per-class degree ``z_i`` (rows of the count matrix are scaled so
        their sum equals ``zcap``). ``None`` (default) leaves ``z_i`` unclamped. On wide windows with
        extreme giant pores the mean-field degree grows as ``r_i**2/<r^2>`` and can overshoot the
        finite-N generator; a finite cap stabilizes the fixed point at the cost of an ad-hoc parameter.
    max_iter : int, optional
        Maximum number of fixed-point iterations. Default is 500.
    tol : float, optional
        Convergence threshold on the L-infinity change between successive iterates. Default is 1e-12.
    verbose : bool, optional
        If True, print one line per 25 iterations with the current residual. Default is False.

    Returns
    -------
    tuple(ndarray, ndarray)
        ``(cdist, pdist)``: same contract as :func:`invert_pore_distribution_kernel`. Feed ``pdist``
        together with the full pore bin edges (``throat_radii * PRTRR``, i.e. n_bins + 1 values) to
        :func:`inverse_transform_sampling_from_raw_data`.

    Raises
    ------
    ValueError
        If the three input arrays do not have the same length, or if the corrected throat
        distribution has no positive mass.

    Notes
    -----
    **No new calibration parameter.** Like the production kernel, this model uses only ``porosity``,
    ``sratio`` and ``PRTRR`` (plus the empirical step-1 factor ``e``). The size-dependent degree is a
    consequence of the budget constraint, not an added lever.

    **Stability.** On truncated windows (e.g. keep75) the fixed point converges cleanly without a cap.
    On wide windows with extreme giants the unclamped degree can overshoot the finite-N generator and
    stiffen the iteration; pass ``zcap`` in that case. This is the same caveat as
    :func:`invert_pore_distribution_kernel_experimental`, but here the composition is also corrected,
    so the two should be compared on the target window rather than assumed equivalent.

    Full context: this addresses the multi-capillary limitation of the single-link derivation — see
    ``doc/pore-distribution-inversion.md`` §5.4–5.5 for the equal-degree idealization this replaces.
    """
    cdist = _corrected_throat_counts(volumes, midradii, deltaR, e)
    n_bins = cdist.size

    # Pore bin mid-radii (pores are PRTRR times larger than throats).
    r = PRTRR * np.asarray(midradii, dtype=float).ravel()
    tiny = np.finfo(float).tiny

    # Throat cross-section matrix S_t[i, j] and per-class surface budget B_i.
    S = np.pi * (np.minimum.outer(r, r) ** 2) / PRTRR**2
    budget = sratio * 4.0 * np.pi * r**2

    def _counts(x):
        """Water-filling per-class count matrix ``T`` with ``n_ij = x_j*T[i, j]``, plus degrees."""
        ms = np.maximum(S @ x, tiny)  # <S_t>_i under the pore distribution x
        T = np.maximum((budget[:, None] - S) / ms[:, None], 0.0)
        z = (T * x[None, :]).sum(axis=1)  # z_i = B_i/<S_t>_i - 1
        if zcap is not None:
            over = z > float(zcap)
            if over.any():
                scale = np.where(over, float(zcap) / np.maximum(z, tiny), 1.0)
                T = T * scale[:, None]
                z = z * scale
        return T, z

    def _wf_step(x):
        """One fixed-point sweep: build the water-filling counts from ``x``, solve for the new x."""
        T, z = _counts(x)
        zbar = float(np.dot(x, z))
        if not (zbar > 0.0) or not np.isfinite(zbar):
            return None

        xn = np.zeros(n_bins)
        for k in range(n_bins - 1, -1, -1):
            # cdist[k]*zbar = a_k*x_k^2 + b_k*x_k with the diagonal self-term kept coupled:
            #   a_k = T[k,k] (so n_kk = x_k*T[k,k]),
            #   b_k = sum_{j>k} x_j*T[k,j]  (focal k -> larger partner j)
            #       + sum_{j>k} x_j*T[j,k]  (larger focal j pointing down at class k).
            a_k = T[k, k]
            if k < n_bins - 1:
                b_k = float((xn[k + 1 :] * T[k, k + 1 :]).sum() + (xn[k + 1 :] * T[k + 1 :, k]).sum())
            else:
                b_k = 0.0
            rhs = cdist[k] * zbar
            if rhs <= 0.0:
                xn[k] = 0.0
            elif a_k > 0.0:
                # Positive root of a_k*x^2 + b_k*x - rhs = 0 (discriminant >= b_k^2 > 0).
                xn[k] = (-b_k + np.sqrt(b_k * b_k + 4.0 * a_k * rhs)) / (2.0 * a_k)
            elif b_k > 0.0:
                xn[k] = rhs / b_k
            else:
                xn[k] = 0.0

        total = float(xn.sum())
        return xn / total if total > 0.0 else None

    def _forward_residual(x):
        """Max |P_c(x) - cdist| with the water-filling counts built from ``x``."""
        T, z = _counts(x)
        zbar = float(np.dot(x, z))
        if not (zbar > 0.0):
            return np.inf
        pc = np.empty(n_bins)
        for k in range(n_bins):
            up = float((x[k:] * T[k, k:]).sum())          # sum_{j>=k} x_j*T[k,j]
            dn = float((x[k + 1 :] * T[k + 1 :, k]).sum()) if k < n_bins - 1 else 0.0
            pc[k] = x[k] * (up + dn) / zbar
        return float(np.max(np.abs(pc - cdist)))

    x = _independent_recurrence(cdist).copy()
    err = np.inf
    for it in range(max_iter):
        xn = _wf_step(x)
        if xn is None:
            break
        err = float(np.max(np.abs(xn - x)))
        x = xn
        if verbose and (it % 25 == 0 or err < tol):
            print(f"  invert_pore_distribution_kernel_waterfill: iteration {it}: max |dx| = {err:.3e}")
        if err < tol:
            break

    resid = _forward_residual(x)
    if resid > 1e-3:
        warn(
            "invert_pore_distribution_kernel_waterfill: fixed point not reached within {} iterations "
            "(forward-model residual {:.3e} > 1e-3); returning the last iterate.".format(max_iter, resid)
        )

    return cdist, x


def invert_pore_distribution_kernel_budget(
    volumes,
    midradii,
    deltaR,
    e,
    PRTRR=2.0,
    porosity=0.35,
    sratio=0.8,
    zcap=None,
    max_iter=500,
    tol=1e-12,
    verbose=False,
):
    """Predict the pore size distribution with a **budget-coupled throat identity** applied to the
    production exclusion kernel — the conservative multi-capillary correction.

    This is the second of two responses to the single-link limitation of the derivation (see
    :func:`invert_pore_distribution_kernel_waterfill` for the full sequential model). It keeps the
    *validated* partner composition ``K(i -> j)`` of the production kernel exactly unchanged and fixes
    only the count/composition coupling in the throat identity: instead of assuming a class-
    independent mean degree that cancels out, each focal class i carries the number of capillaries its
    budget can actually afford. It is the **degree-only** counterpart to water-filling — it shares the
    same size-dependent degree but leaves the per-class composition untouched, so the two form a clean
    A/B pair (does fixing only the count help, or is the per-class composition correction also needed?).

    **Budget-saturated degree.** For a focal pore in class i with surface budget ``B_i = sratio *
    4*pi*r_i**2`` and mean cross-section to a partner drawn from the pore distribution
    ``<S_t>_i = sum_j x_j*S_t[i, j]``, the number of capillaries that fit in the budget is

      .. math::

          z_i = B_i / \\langle S_t\\rangle_i - 1.

    The degree rises for large pores (their partners are mostly small and cheap, so more capillaries
    fit) — exactly the emergent degree of :func:`invert_pore_distribution_kernel_waterfill`, which is
    why the two variants differ only in composition. The throat identity is then normalized by the mean
    degree ``zbar = sum_i x_i z_i``:

      .. math::

          cdist[k] = \\frac{x_k z_k\\left[K(k{\\to}k) + \\sum_{j>k} K(k{\\to}j)\\right]
                      + \\sum_{j>k} x_j z_j K(j{\\to}k)}{zbar},

    solved by the same backward-substitution fixed point as the production kernel (diagonal self-term
    kept coupled so each equation stays quadratic in ``x[k]``), renormalized after every sweep. The
    update is **adaptively damped** (``lam = 1 -> halved to a 0.05 floor on stagnation``, exactly as in
    :func:`invert_pore_distribution_kernel_gamma`): the class-dependent degree makes the undamped map
    exhibit a period-2 cycle, so damping is required for convergence rather than optional.

    **Why the pore-distribution cost, not the kernel mix.** Computing the degree from the *kernel*
    partner mix (``sum_j K(i -> j)*S_t[i, j]``) double-counts the budget — the reach ``D_i`` already
    encodes it — and collapses to ~0 for large pores whose kernel mix concentrates on tiny cheap
    partners, blowing up ``z_i`` (~1e3–1e4) and destabilizing the fixed point. The pore-distribution
    cost is the stable choice (it is what water-filling uses).

    Parameters
    ----------
    volumes : array-like of float
        Normalized experimental per-bin throat volumes (one value per bin, should sum to 1).
    midradii : array-like of float
        Throat bin centers (same length as ``volumes``).
    deltaR : array-like of float
        Throat bin widths (same length as ``volumes``); must equal the differences between
        consecutive bin edges.
    e : float
        Empirical corrective factor ("alpha"; 1 = no correction) used by step 1.
    PRTRR : float, optional
        Pore radius to throat radius ratio (throat radius = min(r1, r2) / PRTRR). Default is 2.0.
    porosity : float, optional
        Target pore-body porosity of the packing that will be generated. Must match the ``porosity``
        argument passed to :func:`pnm.Generators.random_PNM`. Default is 0.35.
    sratio : float, optional
        Surface coverage ratio (fraction of a pore's surface covered by throats). Enters the per-class
        budget ``B_i = sratio * 4*pi*r_i**2``. Must match the ``sratio`` argument passed to
        :func:`pnm.Generators.random_PNM`. Default is 0.8.
    zcap : float or None, optional
        Optional upper clamp on the per-class degree ``z_i = min(z_i, zcap)``. ``None`` (default)
        leaves it unclamped; a finite cap stabilizes wide windows with extreme giants at the cost of an
        ad-hoc parameter.
    max_iter : int, optional
        Maximum number of fixed-point iterations. Default is 500.
    tol : float, optional
        Convergence threshold on the L-infinity change between successive iterates. Default is 1e-12.
    verbose : bool, optional
        If True, print one line per 25 iterations with the current residual. Default is False.

    Returns
    -------
    tuple(ndarray, ndarray)
        ``(cdist, pdist)``: same contract as :func:`invert_pore_distribution_kernel`. Feed ``pdist``
        together with the full pore bin edges (``throat_radii * PRTRR``, i.e. n_bins + 1 values) to
        :func:`inverse_transform_sampling_from_raw_data`.

    Raises
    ------
    ValueError
        If the three input arrays do not have the same length, or if the corrected throat
        distribution has no positive mass.

    Notes
    -----
    **Relation to the other variants.** Compared with :func:`invert_pore_distribution_kernel` (equal
    degree) this couples count and composition through a size-dependent ``z_i``. It uses the same
    pore-distribution mean cost as :func:`invert_pore_distribution_kernel_experimental` but adds the
    water-filling boundary correction (the ``- 1``), so it is the stable, production-oriented member of
    that family rather than an experimental throwaway. Compared with
    :func:`invert_pore_distribution_kernel_waterfill` it leaves the per-class composition untouched
    (only the overall count per class changes): if the two agree on a window, the degree was the whole
    story; if they differ, water-filling's per-class composition correction is what matters.

    **No new calibration parameter.** Uses only ``porosity``, ``sratio`` and ``PRTRR`` (plus step-1's
    ``e``); the size-dependent degree follows from the budget constraint.
    """
    cdist = _corrected_throat_counts(volumes, midradii, deltaR, e)
    n_bins = cdist.size

    # Pore bin mid-radii (pores are PRTRR times larger than throats).
    r = PRTRR * np.asarray(midradii, dtype=float).ravel()
    r3 = r**3
    tiny = np.finfo(float).tiny

    # Throat cross-section matrix S_t[i, j] and per-class surface budget B_i.
    S = np.pi * (np.minimum.outer(r, r) ** 2) / PRTRR**2
    budget = sratio * 4.0 * np.pi * r**2

    def _kernel_and_degree(x):
        """Production kernel ``K(i -> j)`` plus the budget-saturated degree under the pore mix.

        The degree uses the PORE-distribution mean cost ``ms_i = sum_j x_j*S_t[i, j]`` (the same
        stable quantity as :func:`invert_pore_distribution_kernel_waterfill`), NOT the kernel-weighted
        cost: the latter double-counts the budget (the reach ``D_i`` already encodes it) and collapses
        to ~0 for large pores whose kernel mix concentrates on tiny cheap partners, blowing up ``z_i``
        and destabilizing the fixed point.
        """
        r3mean = float(np.dot(x, r3))
        if not (r3mean > 0.0) or not np.isfinite(r3mean):
            return None
        rho = 9.0 * porosity / (4.0 * np.pi * r3mean)
        m = np.minimum.outer(r, r) ** 2 @ x
        mean_section = np.pi * m / PRTRR**2
        D = np.cbrt(np.maximum(budget / (rho * (4.0 / 3.0) * np.pi * mean_section), tiny))

        s = r[:, None] + r[None, :]
        W = np.exp(-np.minimum((s / D[:, None]) ** 3, 745.0))
        rowsum = W @ x
        K = x[None, :] * W / np.maximum(rowsum[:, None], tiny)
        K[rowsum <= 0.0] = 0.0

        # Budget-saturated degree under the pore-distribution cost: z_i = B_i/<S_t>_i - 1 (the same
        # emergent degree as water-filling, so this variant differs from it only in composition).
        ms = np.maximum(S @ x, tiny)
        z = np.maximum(budget / ms - 1.0, 0.0)
        if zcap is not None:
            z = np.minimum(z, float(zcap))
        return W, rowsum, K, z

    def _budget_step(x):
        """One fixed-point sweep with the budget-coupled throat identity."""
        geom = _kernel_and_degree(x)
        if geom is None:
            return None
        W, rowsum, K, z = geom
        zbar = float(np.dot(x, z))
        if not (zbar > 0.0):
            return None

        xn = np.zeros(n_bins)
        for k in range(n_bins - 1, -1, -1):
            # cdist[k]*zbar = a_k*x_k^2 + b_k*x_k + C_k with the diagonal self-term kept coupled:
            #   a_k = z_k*W[k,k]/rowsum_k, b_k = z_k*sum_{j>k} K[k,j],
            #   C_k = sum_{j>k} x_j*z_j*K[j,k] (already-solved larger classes).
            a_k = (W[k, k] / rowsum[k]) * z[k] if rowsum[k] > 0.0 else 0.0
            b_k = float(K[k, k + 1 :].sum()) * z[k]
            C_k = float(np.dot(xn[k + 1 :] * z[k + 1 :], K[k + 1 :, k])) if k < n_bins - 1 else 0.0
            rhs = cdist[k] * zbar - C_k
            if rhs <= 0.0:
                xn[k] = 0.0
            elif a_k > 0.0:
                # Positive root of a_k*x^2 + b_k*x - rhs = 0 (discriminant >= b_k^2 > 0).
                xn[k] = (-b_k + np.sqrt(b_k * b_k + 4.0 * a_k * rhs)) / (2.0 * a_k)
            elif b_k > 0.0:
                xn[k] = rhs / b_k
            else:
                xn[k] = 0.0

        total = float(xn.sum())
        return xn / total if total > 0.0 else None

    def _forward_residual(x):
        """Max |P_c(x) - cdist| with the budget-coupled identity built from ``x``."""
        geom = _kernel_and_degree(x)
        if geom is None:
            return np.inf
        W, rowsum, K, z = geom
        zbar = float(np.dot(x, z))
        if not (zbar > 0.0):
            return np.inf
        pc = np.empty(n_bins)
        for k in range(n_bins):
            pc[k] = (
                x[k] * z[k] * (K[k, k] + float(K[k, k + 1 :].sum()))
                + (float(np.dot(x[k + 1 :] * z[k + 1 :], K[k + 1 :, k])) if k < n_bins - 1 else 0.0)
            ) / zbar
        return float(np.max(np.abs(pc - cdist)))

    x = _independent_recurrence(cdist).copy()
    err = np.inf
    lam = 1.0  # damping factor of the fixed-point update (adaptive: halved on stagnation)
    best_err = np.inf
    stall = 0
    for it in range(max_iter):
        xn = _budget_step(x)
        if xn is None:
            break
        # Damped update. The undamped map exhibits a period-2 cycle (the class-dependent degree makes
        # the top classes oscillate), so lam starts at 1 (fast path) and is halved below whenever
        # progress stalls — the same scheme as invert_pore_distribution_kernel_gamma.
        xd = (1.0 - lam) * x + lam * xn
        total = float(xd.sum())
        if not (total > 0.0):
            break
        xd /= total
        err = float(np.max(np.abs(xd - x)))
        x = xd
        # Adaptive damping: halve lam after 25 consecutive iterations without a 5% improvement of the
        # step size (floor 0.05; below it further halving is numerically pointless).
        if err < 0.95 * best_err:
            best_err = err
            stall = 0
        else:
            stall += 1
            if stall >= 25 and lam > 0.05:
                lam *= 0.5
                stall = 0
        if verbose and (it % 25 == 0 or err < tol):
            print(f"  invert_pore_distribution_kernel_budget: iteration {it}: max |dx| = {err:.3e} (lam={lam:g})")
        if err < tol:
            break

    resid = _forward_residual(x)
    if resid > 1e-3:
        warn(
            "invert_pore_distribution_kernel_budget: fixed point not reached within {} iterations "
            "(forward-model residual {:.3e} > 1e-3); returning the last iterate.".format(max_iter, resid)
        )

    return cdist, x


def _exp_upper_incomplete_gamma(s: float, x):
    """Stable evaluation of ``exp(x) * Gamma(s, x)``, the upper incomplete gamma times its exponential.

    The product appears in the kernel length moment of ``doc/pore-distribution-inversion.md`` §5.6.2::

        <s>_ij = D_i * exp(x_ij) * Gamma(4/3, x_ij),   x_ij = ((r_i + r_j)/D_i)**3 > 0.

    It is computed directly through the regularized form ``gammaincc`` for moderate ``x`` and
    switches to the asymptotic expansion::

        exp(x) * Gamma(s, x) ~ x**(s-1) * (1 + (s-1)/x + (s-1)(s-2)/x**2 + ...),   x -> inf,

    for large ``x``, where ``exp(x)`` would overflow and ``gammaincc`` underflows — the product
    itself stays finite (it tends to ``x**(s-1)``), so a threshold on ``x`` keeps both branches
    safe. The three-term expansion has a relative error of O(x**-4).

    Parameters
    ----------
    s : float
        Shape parameter of the gamma function (4/3 in this project).
    x : array-like of float
        Lower-limit values (>= 0), e.g. ``x_ij = ((r_i + r_j)/D_i)**3``.

    Returns
    -------
    ndarray
        ``exp(x) * Gamma(s, x)``, same shape as ``x``; finite for all ``x >= 0``.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    # exp(700) ~ 1e304 is still safe in float64 (overflow starts at ln(maxfloat) ~ 709.8); beyond
    # it the asymptotic branch takes over before either factor can overflow/underflow separately.
    direct = x < 700.0
    if np.any(direct):
        xd = x[direct]
        out[direct] = np.exp(xd + special.gammaln(s)) * special.gammaincc(s, xd)
    if np.any(~direct):
        xa = x[~direct]
        out[~direct] = xa ** (s - 1.0) * (
            1.0
            + (s - 1.0) / xa
            + (s - 1.0) * (s - 2.0) / xa**2
            + (s - 1.0) * (s - 2.0) * (s - 3.0) / xa**3
        )
    return out


def sratio_profile_values(radii, profile):
    """Evaluate a diameter-dependent surface-coverage ratio at the given pore radii.

    The profile is piecewise linear in log10(radius): constant ``sr_small`` for radius <=
    r_lo_um, varying linearly (in log10) to ``sr_large`` at r_hi_um, then constant ``sr_large``
    beyond. It provides a recalibration lever for the
    structural small-pore deficit / mid-band excess of the budget-driven throat generation:
    raising the ratio of the smallest pores lets them host more (smaller) throats before
    saturating, while lowering it on the large pores makes them saturate earlier. The SAME
    profile must be used by both the inversion (:func:`invert_pore_distribution_kernel_gamma`,
    which evaluates it at the pore bin mid-radii) and the generator
    (:func:`pnm.Transformation.add_throats_by_surf_v3`, which evaluates it per pore).

    Parameters
    ----------
    radii : array-like of float
        Pore radii in µm (same units as ``r_lo_um`` / ``r_hi_um``).
    profile : dict
        Keys: "sr_small" (ratio for r <= r_lo), "sr_large" (floor ratio for r >= r_hi — keep it
        >= 0.3 so the large-pore budget stays meaningful), "r_lo_um" and "r_hi_um" (transition
        interval in pore radius, µm; requires r_lo < r_hi).

    Returns
    -------
    ndarray of float
        Per-radius ratio values, same shape as ``radii``, between ``sr_small`` and ``sr_large``.

    Raises
    ------
    ValueError
        If the profile is not a dict with four positive keys and r_lo_um < r_hi_um.

    Notes
    -----
    A surface-coverage ratio above 1 assigns more throat cross-section to a pore than its whole
    surface area (overlapping contact circles) — physically impossible. Values > 1 act only as an
    effective over-connectivity lever and must not be used for production recipes (study of
    2026-11-17, see AGENTS.md). A warning is emitted when either endpoint exceeds 1.
    """
    try:
        sr_small = float(profile["sr_small"])
        sr_large = float(profile["sr_large"])
        r_lo = float(profile["r_lo_um"])
        r_hi = float(profile["r_hi_um"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "sratio profile must be a dict with keys 'sr_small', 'sr_large', 'r_lo_um' and "
            f"'r_hi_um' (positive floats), got {profile!r}"
        ) from exc
    if not all(v > 0.0 for v in (sr_small, sr_large, r_lo, r_hi)):
        raise ValueError(f"all sratio profile values must be positive, got {profile!r}")
    if not r_lo < r_hi:
        raise ValueError(
            f"profile transition interval requires r_lo_um < r_hi_um, got {r_lo:g} >= {r_hi:g}"
        )
    if max(sr_small, sr_large) > 1.0:
        warn(
            "sratio profile exceeds 1 (max of sr_small/sr_large = "
            f"{max(sr_small, sr_large):g}): a surface-coverage ratio above 1 assigns more throat "
            "cross-section to a pore than its whole surface area — unphysical; values > 1 act as an "
            "effective over-connectivity lever only and must not be used for production recipes.",
            stacklevel=2,
        )

    r = np.asarray(radii, dtype=float)
    with np.errstate(divide="ignore"):  # non-positive radii (defensive) map to the small-pore value
        t = (np.log10(r) - np.log10(r_lo)) / (np.log10(r_hi) - np.log10(r_lo))
    t = np.clip(t, 0.0, 1.0)
    return sr_small + (sr_large - sr_small) * t


def sratio_profile_tag(profile):
    """Compact single-line identifier of an sratio profile (cache file names, CSV columns).

    Example: ``{"sr_small": 0.8, "sr_large": 0.3, "r_lo_um": 0.02, "r_hi_um": 0.5}`` maps to
    ``"p0.8-0.3@0.02-0.5"`` (no path separators — safe in file names).

    Parameters
    ----------
    profile : dict
        Profile as accepted by :func:`sratio_profile_values`.

    Returns
    -------
    str
        Compact tag, e.g. ``"p0.8-0.3@0.02-0.5"``.
    """
    return (
        f"p{float(profile['sr_small']):g}-{float(profile['sr_large']):g}"
        f"@{float(profile['r_lo_um']):g}-{float(profile['r_hi_um']):g}"
    )


def invert_pore_distribution_kernel_gamma(
    volumes,
    midradii,
    deltaR,
    PRTRR=2.0,
    porosity=0.35,
    sratio=0.8,
    length_stat="mean",
    max_iter=1000,
    tol=1e-12,
    verbose=False,
    sratio_profile=None,
):
    """Predict the pore size distribution from an experimental throat (MIP) distribution with a
    **parameter-free capillary-length correction** derived from the exclusion-kernel geometry.

    This is the parameter-free "Phase 2" inversion derived in
    ``doc/pore-distribution-inversion.md`` §5.6: it replaces the empirical corrective exponent
    ``e`` of step 1 by a per-class mean capillary length computed from the *same* Poisson-gas model
    that produces the exclusion kernel (no free parameter —
    only ``porosity``, ``sratio`` and ``PRTRR``, exactly as in
    :func:`invert_pore_distribution_kernel`).

    **Step 1 (parameter-free).** A throat of class k has volume ≈ cross-section × length, so its
    count fraction is the experimental volume divided by the mean product ``<S L>_k``. With the
    bin-mean cross-section ``S_k = pi * (midradii[k]**2 + deltaR[k]**2/12)`` and the model mean
    gap ``Lbar_k`` of §5 below::

        cdist[k] = volumes[k] / (S_k * Lbar_k),   then normalize.

    **Per-pair length moments (§5.1–5.2).** For a focal pore in class i (reach ``D_i``, derived
    from its surface budget exactly as in :func:`invert_pore_distribution_kernel`) and partner
    class j, with hard-core exclusion distance ``a_ij = r_i + r_j`` and
    ``x_ij = (a_ij/D_i)**3 > 0``, the partner-distance density is
    ``f(s) ~ s**2 * exp(-(s/D_i)**3)`` on ``[a_ij, inf)``, giving

      .. math::

          \\langle s\\rangle_{ij} = D_i\\, e^{x_{ij}}\\, \\Gamma\\!\\left(\\tfrac{4}{3}, x_{ij}\\right),
          \\qquad s_{{\\mathrm{med}},ij} = \\left(a_{ij}^3 + D_i^3 \\ln 2\\right)^{1/3},

    where ``Gamma(4/3, x)`` is the **upper incomplete gamma function** (the length moment carries
    one extra power of s than the count integral, which is why the incomplete form appears — see
    doc §4). The mean gap is ``<s>_ij - a_ij`` and the median gap ``s_med,ij - a_ij``; both are
    non-negative by construction.

    **Per-class aggregation (§5.3).** A class-k throat can be produced by a focal pore in class k
    pointing at an equal-or-larger partner (j >= k) or by a larger focal pore (i > k) pointing
    down; weighting each directed connection by its production rate ``x_i K(i->j)``::

        Lbar_k = [ x_k * sum_{j>=k} K(k->j) Lbar_kj + sum_{i>k} x_i K(i->k) Lbar_ik ] / cdist[k],

    where the denominator is identically ``cdist[k]`` at a fixed point (throat identity — verified
    to ~1e-13 on the keep75 fixed point). Classes with zero mass at an intermediate iterate have
    an undefined 0/0 average; they fall back to the x-independent geometric mean of the same
    pairs weighted by the spatial kernel ``W_ij`` alone, which keeps ``Lbar_k`` finite and O(D_i)
    so step 1 stays well-scaled.

    **Coupled fixed point.** Because step 1 now depends on the pore distribution (through ``D_i``,
    ``K`` and hence ``Lbar_k``), the inversion is a coupled fixed-point iteration: at each sweep,
    build the kernel geometry from the current iterate, recompute ``Lbar_k`` and the step-1
    ``cdist``, then solve the throat identity by backward substitution (largest class first) with
    the kernel frozen except for the diagonal self-term — exactly as in
    :func:`invert_pore_distribution_kernel`. The update is damped, ``x <- (1-lam)*x + lam*xn``:
    the undamped map exhibits a period-2 cycle at low sratio (a kink induced by zero-mass classes,
    whose production-weighted ``Lbar_k`` and the W-based fallback disagree), so ``lam`` starts at 1
    (fast path) and is halved automatically after 25 stalled iterations, down to a floor of 0.05.
    The iteration is renormalized after every sweep and stops when the L-infinity change drops
    below ``tol``; success is judged on the forward-model residual of the *coupled* map (how well x
    reproduces its own step-1 cdist through K).

    Parameters
    ----------
    volumes : array-like of float
        Normalized experimental per-bin throat volumes (one value per bin, should sum to 1).
    midradii : array-like of float
        Throat bin centers (same length as ``volumes``).
    deltaR : array-like of float
        Throat bin widths (same length as ``volumes``); must equal the differences between
        consecutive bin edges.
    PRTRR : float, optional
        Pore radius to throat radius ratio (throat radius = min(r1, r2) / PRTRR). Used to map
        throat bins onto pore bins in the kernel. Default is 2.0.
    porosity : float, optional
        Target pore-body porosity of the packing that will be generated. Enters the number density
        ``rho = 9*porosity/(4*pi*<r^3>)`` used by the kernel and must match the ``porosity``
        argument passed to :func:`pnm.Generators.random_PNM`. Default is 0.35.
    sratio : float, optional
        Surface coverage ratio (fraction of a pore's surface covered by throats). Enters the
        per-class budget ``B_i = sratio * 4*pi*r_i**2`` used to derive the reach ``D_i`` and must
        match the ``sratio`` argument passed to :func:`pnm.Generators.random_PNM`. Default is 0.8.
    sratio_profile : dict or None, optional
        Diameter-dependent surface coverage ratio (see :func:`sratio_profile_values`): a dict with
        keys "sr_small", "sr_large", "r_lo_um", "r_hi_um". When set it REPLACES the scalar
        ``sratio`` — evaluated at each pore bin mid-radius to form the per-class budget
        ``B_i = sr(r_i) * 4*pi*r_i**2`` (the scalar is then ignored). The SAME profile must be
        passed as ``ratio`` to :func:`pnm.Transformation.add_throats_by_surf_v3` so that the
        generator's per-pore budgets match the model. Default is None (scalar sratio).
    length_stat : str, optional
        Which per-pair gap statistic feeds step 1: "mean" (default) uses the incomplete-gamma
        moment ``<s>_ij - a_ij`` — the physically correct choice for a volume-to-count conversion;
        "median" uses the closed form ``(a_ij**3 + D_i**3 ln 2)**(1/3) - a_ij``, which is more
        robust in the large classes where the gap distribution piles up at contact; "packing"
        uses the same incomplete-gamma moment as "mean" but with a single GLOBAL reach
        ``D = (<r^3>/porosity)**(1/3)`` derived from the physical number density of the packing
        (the radius of a sphere holding on average one pore), instead of the per-class budget
        reach ``D_i`` — the kernel geometry (W, K) keeps the budget reach in all cases. Default
        is "mean".
    max_iter : int, optional
        Maximum number of fixed-point iterations. Default is 1000 (the coupled map moves step 1 as
        well and can need a few more sweeps than the plain kernel inversion).
    tol : float, optional
        Convergence threshold on the L-infinity change between successive iterates. Default is 1e-12.
    verbose : bool, optional
        If True, print one line per 25 iterations with the current residual. Default is False.

    Returns
    -------
    tuple(ndarray, ndarray)
        ``(cdist, pdist)``: the final step-1 throat COUNT distribution (one probability per bin,
        summing to 1) and the predicted pore COUNT distribution (one probability per bin, summing
        to 1), both of length ``len(volumes)``. Feed ``pdist`` together with the full pore bin
        edges (``throat_radii * PRTRR``, i.e. n_bins + 1 values) to
        :func:`inverse_transform_sampling_from_raw_data` — exactly like the output of
        :func:`invert_pore_distribution_kernel`.

    Raises
    ------
    ValueError
        If the three input arrays do not have the same length, if ``length_stat`` is not one of
        "mean"/"median"/"packing", or if step 1 has no positive mass.

    Notes
    -----
    **Why this exists.** Step 1 of :func:`invert_pore_distribution_kernel` divides each bin's
    volume by ``(pi * E[r^2])**e``, which silently assumes a power law ``Lbar_k ~ m_k**(2(e-1))``
    for the mean capillary length with a POSITIVE exponent. Measured per-class gaps on generated
    networks (doc §3) have the opposite sign: nearly flat / slightly increasing for
    ``m_k < 0.03 um`` and DECREASING as ``m_k**(-0.8...-1.1)`` above — no value of ``e`` can fix
    it, which is why no (e, sratio) combination matches both ends of the distribution at once.

    **Validation status.** The derived mean gap reproduces the measurement within <=3 % for
    ``m_k < 0.03 um`` (~95 % of throat count; doc §6), with a +10...+21 % transition region at
    0.03–0.05 um, but OVERPREDICTS large-class gaps by a factor of ~2 up to ~12–16: the generator
    connects large pores at near-contact distances (nearest-neighbor selection in pass 1/2), which
    the mean-field Poisson-gas survival weight washes out. A production use should therefore treat
    the large-class regime as a known bias (doc §7 lists the correction options); the small-class
    agreement — where ~95 % of the throat count lives — is parameter-free and exact to within a few
    percent.

    **The "packing" variant (tested 2026-11-16, rejected).** ``length_stat="packing"`` replaces
    the per-class budget reach by a single global reach from the physical number density of the
    packing, ``D = (<r^3>_x/porosity)**(1/3)`` (recomputed at every iterate through ``<r^3>_x``):
    it is the radius of a sphere holding on average one pore, i.e. the literal Poisson-gas length
    scale of the packing with no sratio or class dependence. Only step 1's gap matrix changes —
    the kernel geometry (W, K) always uses the budget reach, so the fixed-point structure and the
    sratio dependence of the pore set are preserved. On the keep75 window (v3 pass 1, cutoff
    0.5 um, N=1e5, sratio 0.6-0.8) it converges but shifts the pore target strongly toward LARGER
    pores: the global D (~0.56 um at the "mean" fixed point) is far below the budget reach of the
    largest classes (~3.8 um), so their step-1 lengths collapse, their count fractions balloon,
    and the fixed-point ``<r^3>`` grows ~5x (0.06 -> 0.32 um**3). The generated networks empty the
    smallest band to ~0.7-0.9 % of throat volume (target 25.7 %) and overshoot the >0.5 um band by
    +30...+34 pt; JSD degrades from 0.023-0.039 ("mean") to 0.185-0.202 (~6-7x worse). The class
    dependence of the budget reach is therefore load-bearing — it acts as an implicit large-throat
    penalty that keeps small-pore mass in the target and cannot be replaced by a global physical
    length scale. Kept for research only; production uses "mean".

    **Finite-size caveat.** Inherited from :func:`invert_pore_distribution_kernel`: the kernel is an
    infinite-system idealization; at finite N the largest bins of a wide window may be unplaceable
    in the box sized for the target porosity and their throat tail stays unreachable regardless of
    the inversion.

    **Diameter-dependent sratio (2026-11-17).** ``sratio_profile`` replaces the scalar budget ratio
    with a per-class profile decreasing in radius (:func:`sratio_profile_values`); it is a
    recalibration lever for the structural small-pore deficit / mid-band excess of the generated
    throat distribution (study: ``examples/study_sratio_profile_150k.py``, results in AGENTS.md).

    Full derivation + validation data: ``doc/pore-distribution-inversion.md`` §5.6.
    """
    volumes = np.asarray(volumes, dtype=float).ravel()
    midradii = np.asarray(midradii, dtype=float).ravel()
    deltaR = np.asarray(deltaR, dtype=float).ravel()
    if not (volumes.size == midradii.size == deltaR.size):
        raise ValueError(
            "volumes, midradii and deltaR must have the same length, got "
            f"{volumes.size}, {midradii.size}, {deltaR.size}"
        )
    if length_stat not in ("mean", "median", "packing"):
        raise ValueError(
            f"length_stat must be 'mean', 'median' or 'packing', got {length_stat!r}"
        )

    n_bins = volumes.size
    # Pore bin mid-radii (pores are PRTRR times larger than throats).
    r = PRTRR * midradii
    r3 = r**3
    # Mean throat cross-section per class: pi * E[r^2] of a radius drawn uniformly in the bin.
    S_k = np.pi * (midradii**2 + deltaR**2 / 12.0)
    tiny = np.finfo(float).tiny

    # Per-class surface ratio: scalar, or the diameter-dependent profile evaluated at the pore bin
    # mid-radii (constant across iterations — it depends only on class radius, not on x).
    sr_arr = sratio_profile_values(r, sratio_profile) if sratio_profile is not None else float(sratio)

    def _geometry(x):
        """Kernel geometry at iterate ``x``: reach D_i, pair weights W_ij and kernel K(i->j)."""
        r3mean = float(np.dot(x, r3))
        if not (r3mean > 0.0) or not np.isfinite(r3mean):
            return None
        # Number density of the packing that realizes x at the target porosity.
        rho = 9.0 * porosity / (4.0 * np.pi * r3mean)
        # Mean squared throat radius to a partner drawn from x, per focal class i.
        m = np.minimum.outer(r, r) ** 2 @ x
        budget = sr_arr * 4.0 * np.pi * r**2
        mean_section = np.pi * m / PRTRR**2
        D = np.cbrt(np.maximum(budget / (rho * (4.0 / 3.0) * np.pi * mean_section), tiny))

        # Pair weight W_ij = exp(-((r_i + r_j)/D_i)^3); the exponent is clipped so that strongly
        # suppressed pairs underflow to zero instead of raising a warning.
        s = r[:, None] + r[None, :]
        X = np.minimum((s / D[:, None]) ** 3, 745.0)
        W = np.exp(-X)

        # Row normalization is x-weighted: K[i, j] = x_j*W_ij / sum_k(x_k*W_ik), so that each row
        # of K sums to 1 (K is the partner-size distribution of a focal pore in class i).
        rowsum = W @ x
        K = x[None, :] * W / np.maximum(rowsum[:, None], tiny)
        K[rowsum <= 0.0] = 0.0
        return D, s, X, W, rowsum, K

    def _packing_D(x):
        """Global reach from the physical number density of the packing that realizes ``x``.

        The physical pore number density is ``rho_phys = 3*porosity/(4*pi*<r^3>)``; a Poisson gas
        at that density has partner-distance weight
        ``exp(-rho_phys*(4/3)*pi*s**3) = exp(-(s/D)**3)`` with ``D**3 = <r^3>/porosity`` — the
        radius of a sphere holding on average one pore. Unlike the per-class budget reach it has no
        sratio or class dependence, so step 1 sees a single global D for every pair (§5.1 with
        ``D_i -> D``). Returns None when ``x`` carries no positive mass (same guard as _geometry).
        """
        r3mean = float(np.dot(x, r3))
        if not (r3mean > 0.0) or not np.isfinite(r3mean):
            return None
        return np.cbrt(r3mean / porosity)

    def _pair_gaps(D, s, X, D_pack=None):
        """Per-pair gap matrix (surface-to-surface), mean/median (§5.1–5.2) or packing variant."""
        if length_stat == "packing":
            # Global physical-density reach for every pair: same incomplete-gamma moment as "mean"
            # with D_i replaced by the scalar packing D (doc §5.1, D -> D_pack).
            Xp = np.minimum((s / D_pack) ** 3, 745.0)
            s_mean = D_pack * _exp_upper_incomplete_gamma(4.0 / 3.0, Xp)
            return np.maximum(s_mean - s, 0.0)
        if length_stat == "mean":
            # <s>_ij = D_i * e^{x_ij} Gamma(4/3, x_ij); the gap is non-negative by construction of
            # the integral (clamped against round-off in the contact limit).
            s_mean = D[:, None] * _exp_upper_incomplete_gamma(4.0 / 3.0, X)
            return np.maximum(s_mean - s, 0.0)
        # Median: closed form, no special functions (doc §5.2).
        return (s**3 + D[:, None] ** 3 * np.log(2.0)) ** (1.0 / 3.0) - s

    def _class_lengths(gaps, K, x, W):
        """Aggregate per-pair gaps to the per-throat-class mean gap Lbar_k (§5.3).

        Returns ``(Lbar_k, prod_k)`` where ``prod_k`` is the forward-model production rate of
        class-k throats at iterate ``x`` — identically equal to step-1's ``cdist[k]`` at a fixed
        point (throat identity), so it doubles as the aggregation denominator.
        """
        Lbar = np.zeros(n_bins)
        prod = np.zeros(n_bins)
        for k in range(n_bins):
            up_w = x[k] * K[k, :]  # focal k -> partner j; j >= k keeps the smaller endpoint in k
            up_w[:k] = 0.0
            dn_w = x * K[:, k]  # larger focal i > k pointing down at class k
            dn_w[: k + 1] = 0.0
            wk = float(up_w.sum() + dn_w.sum())
            prod[k] = wk
            if wk > 0.0:
                Lbar[k] = (float((up_w * gaps[k, :]).sum()) + float((dn_w * gaps[:, k]).sum())) / wk
            else:
                # Zero-mass class: the production-weighted average is a 0/0 indeterminate form. Fall
                # back to the x-independent geometric mean of the same pairs weighted by the spatial
                # kernel W alone, so Lbar_k stays finite and O(D_i) (keeps step 1 well-scaled for
                # classes that are pinned at zero during iteration). Copies: W is shared with
                # _solve (which reads W[k, k]) and must not be modified here.
                up_g = W[k, :].copy()
                up_g[:k] = 0.0
                dn_g = W[:, k].copy()
                dn_g[: k + 1] = 0.0
                wg = float(up_g.sum() + dn_g.sum())
                if wg > 0.0:
                    Lbar[k] = (float((up_g * gaps[k, :]).sum()) + float((dn_g * gaps[:, k]).sum())) / wg
        return Lbar, prod

    def _step1_cdist(Lbar_k):
        """Step 1 (parameter-free): cdist_k = v_k / (S_k * Lbar_k), normalized to sum to 1."""
        raw = volumes / (S_k * np.maximum(Lbar_k, tiny))
        raw[~np.isfinite(raw)] = 0.0
        total = float(raw.sum())
        if not (total > 0.0):
            raise ValueError("step-1 throat distribution has no positive mass")
        return raw / total

    def _solve(cdist_l, W, rowsum, K):
        """Backward substitution of the throat identity with the kernel frozen at one iterate."""
        xn = np.zeros(n_bins)
        for k in range(n_bins - 1, -1, -1):
            # cdist[k] = a_k*x_k^2 + b_k*x_k + C_k with the diagonal self-term kept coupled:
            #   a_k = W[k,k]/rowsum_k   (so K(k->k) = x_k * a_k exactly),
            #   b_k = sum_{j>k} K[k,j]  (frozen off-diagonal row entries),
            #   C_k = sum_{j>k} x_j*K[j,k] (already-solved larger classes, frozen column).
            a_k = W[k, k] / rowsum[k] if rowsum[k] > 0.0 else 0.0
            b_k = float(K[k, k + 1 :].sum())
            C_k = float(np.dot(xn[k + 1 :], K[k + 1 :, k])) if k < n_bins - 1 else 0.0
            rhs = cdist_l[k] - C_k
            if rhs <= 0.0:
                xn[k] = 0.0
            elif a_k > 0.0:
                # Positive root of a_k*x^2 + b_k*x - rhs = 0 (discriminant >= b_k^2 > 0).
                xn[k] = (-b_k + np.sqrt(b_k * b_k + 4.0 * a_k * rhs)) / (2.0 * a_k)
            elif b_k > 0.0:
                xn[k] = rhs / b_k
            else:
                xn[k] = 0.0
        total = float(xn.sum())
        return xn / total if total > 0.0 else None

    # Initial iterate: independent-pairing recurrence on the e=1 (uncorrected) count distribution —
    # a neutral starting point, since this model carries no empirical exponent at all.
    x = _independent_recurrence(_corrected_throat_counts(volumes, midradii, deltaR, 1.0)).copy()

    err = np.inf
    cdist_final = None
    lam = 1.0  # damping factor of the fixed-point update (adaptive: halved on stagnation)
    best_err = np.inf
    stall = 0
    for it in range(max_iter):
        geom = _geometry(x)
        if geom is None:
            break
        D, s, X, W, rowsum, K = geom
        # "packing" step 1 needs the global physical-density reach (valid whenever _geometry was).
        D_pack = _packing_D(x) if length_stat == "packing" else None
        gaps = _pair_gaps(D, s, X, D_pack)
        Lbar_k, _prod_k = _class_lengths(gaps, K, x, W)
        cdist_l = _step1_cdist(Lbar_k)
        xn = _solve(cdist_l, W, rowsum, K)
        if xn is None:
            break
        # Damped update: the convex combination is exact for all lam, including lam = 1 (full step
        # to xn). The undamped map exhibits a period-2 cycle at low sratio (a kink induced by
        # zero-mass classes: their production-weighted Lbar_k and the W-based fallback disagree,
        # and C_k ~ x_k vanishes when a class is pinned at zero); damping with lam < 1 removes it.
        # lam starts at 1 (fast path) and is halved below whenever progress stalls.
        xd = (1.0 - lam) * x + lam * xn
        total = float(xd.sum())
        if not (total > 0.0):
            break
        xd /= total
        err = float(np.max(np.abs(xd - x)))
        x = xd
        cdist_final = cdist_l
        # Adaptive damping: halve lam after 25 consecutive iterations without a 5% improvement of
        # the step size (floor 0.05; below it further halving is numerically pointless).
        if err < 0.95 * best_err:
            best_err = err
            stall = 0
        else:
            stall += 1
            if stall >= 25 and lam > 0.05:
                lam *= 0.5
                stall = 0
        if verbose and (it % 25 == 0 or err < tol):
            print(
                f"  invert_pore_distribution_kernel_gamma: iteration {it}: max |dx| = {err:.3e} "
                f"(lam={lam:g})"
            )
        if err < tol:
            break

    # Judge success on the forward-model residual of the COUPLED map: how well x reproduces its own
    # step-1 cdist through K (at a fixed point, prod_k == cdist_model,k identically).
    resid = np.inf
    geom = _geometry(x)
    if geom is not None:
        D, s, X, W, rowsum, K = geom
        gaps = _pair_gaps(D, s, X, _packing_D(x) if length_stat == "packing" else None)
        Lbar_k, prod_k = _class_lengths(gaps, K, x, W)
        cdist_final = _step1_cdist(Lbar_k)
        resid = float(np.max(np.abs(prod_k - cdist_final)))
    if resid > 1e-3:
        warn(
            "invert_pore_distribution_kernel_gamma: fixed point not reached within {} iterations "
            "(forward-model residual {:.3e} > 1e-3); returning the last iterate.".format(max_iter, resid)
        )

    return cdist_final, x


def invert_pore_distribution_kernel_gamma_hybrid(
    volumes,
    midradii,
    deltaR,
    PRTRR=2.0,
    porosity=0.35,
    sratio=0.8,
    m_switch=0.03,
    tail_exponent=0.5,
    max_iter=1000,
    tol=1e-12,
    verbose=False,
):
    """Phase-2 kernel inversion with a **hybrid step-1 length**: the derived incomplete-gamma
    profile where it is validated, and a power-law continuation beyond the validation boundary.

    This is :func:`invert_pore_distribution_kernel_gamma` with one change — the per-class mean
    capillary length fed to step 1. The pure derivation has a **turnover**: it peaks near
    ``m_k ~ 0.1 um`` and then declines toward contact in the largest classes (the generator's
    nearest-neighbor selection among sparse giants, doc/pore-distribution-inversion.md §5.6.5). A flat or
    declining step-1 length demands too many large throats per unit volume; through the coupled
    fixed point this shifts pore mass into the largest classes (measured on keep75: 93.8 % vs
    90.2 % of pore volume in pores >= 1 um) and the generated networks then overshoot the >0.5 um
    band (+3.8 pt at sratio=0.8, N=1e5; +8.5 pt at sratio=0.7, N=2e5) while starving the smallest
    band (~-14…-16 pt), JSD ~0.023–0.030 vs 0.019 for ``invert_pore_distribution_kernel`` with e=1.25.

    **The correction.** Keep the derived length (validated to <=3 % against measurement) in the
    small-class regime ``m_k <= m_switch``, and beyond it replace the turnover by a power-law
    continuation anchored at the boundary::

        Lbar_k = max(Lbar_derived_k,  Lbar_anchor * (m_k / m_anchor)**tail_exponent),   k > anchor,

    where ``anchor`` is the last class with ``midradii <= m_switch``. The profile is continuous
    at the anchor and monotone beyond it: the derived values are flat/declining there while the
    tail grows, so the tail takes over for every larger class. Step 1 then reads
    ``cdist_k = v_k / (S_k * Lbar_k)`` exactly as in :func:`invert_pore_distribution_kernel_gamma`,
    and the rest of the machinery (budget-reach kernel geometry, throat identity, adaptive-damping
    coupled fixed point) is unchanged.

    Parameters
    ----------
    volumes, midradii, deltaR, PRTRR, porosity, sratio : array-like / float
        Same meaning as in :func:`invert_pore_distribution_kernel_gamma`.
    m_switch : float, optional
        Throat-radius boundary (um) of the validated small-class regime; the derived profile is
        used up to and including this radius. Default is 0.03 um (the validation limit of doc §6).
    tail_exponent : float, optional
        Exponent ``beta`` of the power-law continuation beyond ``m_switch``. Default is 0.5 — it
        reproduces the effective steepness of the JSD-optimal kernel recipe e=1.25 (whose step-1
        length grows as ``m**(2*(e-1)) = m**0.5``). This is an explicit parameter: the hybrid is a
        two-regime model, not fully parameter-free; beta absorbs the class dependence that the pure
        derivation misses in the transition/large regime.
    max_iter, tol, verbose : int / float / bool, optional
        Same as in :func:`invert_pore_distribution_kernel_gamma`.

    Returns
    -------
    tuple(ndarray, ndarray)
        ``(cdist, pdist)`` — same contract as :func:`invert_pore_distribution_kernel_gamma`.

    Raises
    ------
    ValueError
        If the three input arrays do not have the same length, if ``m_switch`` is below the
        smallest bin center (no anchor class), or if step 1 has no positive mass.

    Notes
    -----
    **Status.** Tested 2026-11-16 on keep75 (v3 pass 1, cutoff 0.5 um, N=1e5): best JSD 0.0234 at
    beta=0.3 — does not beat ``invert_pore_distribution_kernel`` with e=1.25 (JSD 0.0195) and only
    ties the plain gamma inversion; the structural b1-deficit/b3-excess residual persists for every
    tested beta, so this variant is rejected as a reference recipe and kept for research only. Full
    analysis: ``doc/pore-distribution-inversion.md`` §5.6.6.
    """
    volumes = np.asarray(volumes, dtype=float).ravel()
    midradii = np.asarray(midradii, dtype=float).ravel()
    deltaR = np.asarray(deltaR, dtype=float).ravel()
    if not (volumes.size == midradii.size == deltaR.size):
        raise ValueError(
            "volumes, midradii and deltaR must have the same length, got "
            f"{volumes.size}, {midradii.size}, {deltaR.size}"
        )
    if midradii[0] > m_switch:
        raise ValueError(
            f"m_switch={m_switch:g} is below the smallest bin center ({midradii[0]:g}); "
            "there is no anchor class for the power-law tail"
        )

    n_bins = volumes.size
    r = PRTRR * midradii
    r3 = r**3
    S_k = np.pi * (midradii**2 + deltaR**2 / 12.0)
    tiny = np.finfo(float).tiny

    def _geometry(x):
        """Kernel geometry at iterate ``x``: reach D_i, pair weights W_ij and kernel K(i->j)."""
        r3mean = float(np.dot(x, r3))
        if not (r3mean > 0.0) or not np.isfinite(r3mean):
            return None
        rho = 9.0 * porosity / (4.0 * np.pi * r3mean)
        m = np.minimum.outer(r, r) ** 2 @ x
        budget = sratio * 4.0 * np.pi * r**2
        mean_section = np.pi * m / PRTRR**2
        D = np.cbrt(np.maximum(budget / (rho * (4.0 / 3.0) * np.pi * mean_section), tiny))

        s = r[:, None] + r[None, :]
        X = np.minimum((s / D[:, None]) ** 3, 745.0)
        W = np.exp(-X)

        rowsum = W @ x
        K = x[None, :] * W / np.maximum(rowsum[:, None], tiny)
        K[rowsum <= 0.0] = 0.0
        return D, s, X, W, rowsum, K

    def _pair_gaps(D, s, X):
        """Per-pair mean gap matrix (surface-to-surface), incomplete-gamma moment (§5.1)."""
        s_mean = D[:, None] * _exp_upper_incomplete_gamma(4.0 / 3.0, X)
        return np.maximum(s_mean - s, 0.0)

    def _class_lengths(gaps, K, x, W):
        """Aggregate per-pair gaps to the per-throat-class mean gap Lbar_k (§5.3)."""
        Lbar = np.zeros(n_bins)
        prod = np.zeros(n_bins)
        for k in range(n_bins):
            up_w = x[k] * K[k, :]
            up_w[:k] = 0.0
            dn_w = x * K[:, k]
            dn_w[: k + 1] = 0.0
            wk = float(up_w.sum() + dn_w.sum())
            prod[k] = wk
            if wk > 0.0:
                Lbar[k] = (float((up_w * gaps[k, :]).sum()) + float((dn_w * gaps[:, k]).sum())) / wk
            else:
                # Copies: W is shared with _solve (W[k, k]) and must not be modified here.
                up_g = W[k, :].copy()
                up_g[:k] = 0.0
                dn_g = W[:, k].copy()
                dn_g[: k + 1] = 0.0
                wg = float(up_g.sum() + dn_g.sum())
                if wg > 0.0:
                    Lbar[k] = (float((up_g * gaps[k, :]).sum()) + float((dn_g * gaps[:, k]).sum())) / wg
        return Lbar, prod

    def _hybrid_lengths(Lbar_k):
        """Hybrid step-1 length: derived profile up to m_switch, power-law tail beyond it."""
        L = np.array(Lbar_k, dtype=float)
        anchor = int(np.max(np.nonzero(midradii <= m_switch)[0]))
        L_anchor = max(float(L[anchor]), tiny)
        for k in range(anchor + 1, n_bins):
            tail = L_anchor * (midradii[k] / midradii[anchor]) ** tail_exponent
            if tail > L[k]:
                L[k] = tail
        return L

    def _step1_cdist(Lbar_k):
        """Step 1: cdist_k = v_k / (S_k * Lbar_hybrid_k), normalized to sum to 1."""
        raw = volumes / (S_k * np.maximum(_hybrid_lengths(Lbar_k), tiny))
        raw[~np.isfinite(raw)] = 0.0
        total = float(raw.sum())
        if not (total > 0.0):
            raise ValueError("step-1 throat distribution has no positive mass")
        return raw / total

    def _solve(cdist_l, W, rowsum, K):
        """Backward substitution of the throat identity with the kernel frozen at one iterate."""
        xn = np.zeros(n_bins)
        for k in range(n_bins - 1, -1, -1):
            a_k = W[k, k] / rowsum[k] if rowsum[k] > 0.0 else 0.0
            b_k = float(K[k, k + 1 :].sum())
            C_k = float(np.dot(xn[k + 1 :], K[k + 1 :, k])) if k < n_bins - 1 else 0.0
            rhs = cdist_l[k] - C_k
            if rhs <= 0.0:
                xn[k] = 0.0
            elif a_k > 0.0:
                xn[k] = (-b_k + np.sqrt(b_k * b_k + 4.0 * a_k * rhs)) / (2.0 * a_k)
            elif b_k > 0.0:
                xn[k] = rhs / b_k
            else:
                xn[k] = 0.0
        total = float(xn.sum())
        return xn / total if total > 0.0 else None

    # Initial iterate: same neutral starting point as the plain gamma inversion (e=1 counts).
    x = _independent_recurrence(_corrected_throat_counts(volumes, midradii, deltaR, 1.0)).copy()

    err = np.inf
    cdist_final = None
    lam = 1.0
    best_err = np.inf
    stall = 0
    for it in range(max_iter):
        geom = _geometry(x)
        if geom is None:
            break
        D, s, X, W, rowsum, K = geom
        gaps = _pair_gaps(D, s, X)
        Lbar_k, _prod_k = _class_lengths(gaps, K, x, W)
        cdist_l = _step1_cdist(Lbar_k)
        xn = _solve(cdist_l, W, rowsum, K)
        if xn is None:
            break
        xd = (1.0 - lam) * x + lam * xn
        total = float(xd.sum())
        if not (total > 0.0):
            break
        xd /= total
        err = float(np.max(np.abs(xd - x)))
        x = xd
        cdist_final = cdist_l
        if err < 0.95 * best_err:
            best_err = err
            stall = 0
        else:
            stall += 1
            if stall >= 25 and lam > 0.05:
                lam *= 0.5
                stall = 0
        if verbose and (it % 25 == 0 or err < tol):
            print(
                f"  invert_pore_distribution_kernel_gamma_hybrid: iteration {it}: "
                f"max |dx| = {err:.3e} (lam={lam:g})"
            )
        if err < tol:
            break

    resid = np.inf
    geom = _geometry(x)
    if geom is not None:
        D, s, X, W, rowsum, K = geom
        gaps = _pair_gaps(D, s, X)
        Lbar_k, prod_k = _class_lengths(gaps, K, x, W)
        cdist_final = _step1_cdist(Lbar_k)
        resid = float(np.max(np.abs(prod_k - cdist_final)))
    if resid > 1e-3:
        warn(
            "invert_pore_distribution_kernel_gamma_hybrid: fixed point not reached within {} "
            "iterations (forward-model residual {:.3e} > 1e-3); returning the last iterate.".format(
                max_iter, resid
            )
        )

    return cdist_final, x


def shift_pore_distribution(pdist, bins, shift=0.0):
    """Return the ``(data, edges)`` pair that feeds :func:`inverse_transform_sampling_from_raw_data`
    with a **continuous size shift** of the pore distribution toward smaller radii.

    This is the continuous generalization of the "one-bin" off-by-one shift (the pre-fix v1
    behaviour): instead of moving every bin's mass exactly one bin down, each bin edge is moved
    down by a *fraction* ``shift`` of one bin-width. Concretely, with original edges
    ``e_0 < e_1 < ... < e_n`` (n_bins + 1 values) the shifted edges are

        e'_j = (1 - shift) * e_j + shift * e_{j-1}      for j = 1 .. n
        e'_0 = (1 - shift) * e_0 + shift * (2*e_0 - e_1)   (linear extrapolation below e_0)

    and the per-bin masses ``pdist`` are left unchanged. The result is still a valid
    piecewise-uniform distribution on n_bins bins, so it can be passed straight to the sampler.

    Interpretation: ``shift`` is measured in *bins* toward smaller radii (a global size gauge).
    A positive shift makes every drawn pore smaller; a negative shift makes them larger. This is
    useful to absorb a systematic bias between the predicted pore distribution and the pore sizes
    that actually reproduce the experimental throat distribution, without re-deriving ``pdist``.

    Parameters
    ----------
    pdist : array-like of float
        Per-bin pore COUNT distribution (one probability per bin), length n_bins. It is returned
        unchanged (only the edges are shifted).
    bins : array-like of float
        Full pore bin edges in strictly increasing order, length n_bins + 1 (same convention as
        :func:`inverse_transform_sampling_from_raw_data`).
    shift : float, optional
        Continuous shift in units of bins toward smaller radii. ``0`` (default) is the identity:
        the original ``(pdist, bins)`` are returned unchanged, so existing behaviour is preserved
        exactly. For |shift| <= 1 each edge moves by a fraction of its own bin-width (``shift = 1``
        is exactly one full bin down — the v1 effect; values in between interpolate smoothly). For
        |shift| > 1 the shift is one full bin plus a log-multiplicative scale for the excess, so any
        magnitude is supported without folding (positive shifts shrink every pore, negative grow it).
        Default is 0.

    Returns
    -------
    tuple(ndarray, ndarray)
        ``(data, edges)`` to pass as the ``data``/``bins`` arguments of
        :func:`inverse_transform_sampling_from_raw_data`. For ``shift == 0`` this is exactly
        ``(pdist, bins)``.

    Raises
    ------
    ValueError
        If ``len(bins) != len(pdist) + 1``, if the edges are not strictly increasing and positive,
        or if the shifted edges would no longer be strictly increasing (extreme shift values).

    Notes
    -----
    - The lowest edge is extrapolated below ``e_0`` (linearly for |shift| <= 1, and further scaled
      down multiplicatively for larger positive shifts). If it becomes non-positive it is clamped to
      a small positive floor with a warning; this can trigger for large positive shifts.
    - The per-bin masses are left untouched (only the edges move), so this is a pure change of
      variable on the radius axis: each sampled pore keeps its bin and its within-bin relative
      position, but every bin's span is rescaled. This makes ``shift`` a smooth continuous knob
      with an exact identity at 0, suitable for absorbing a systematic size bias between the
      predicted pore distribution and the pore sizes that actually reproduce the experimental
      throat distribution (see the v3 parametric study in AGENTS.md).
    """
    pdist = np.asarray(pdist, dtype=float).ravel()
    edges = np.asarray(bins, dtype=float).ravel()

    n_bins = edges.size - 1
    if n_bins < 1:
        raise ValueError("bins must hold at least 2 edges (n_bins + 1)")
    if pdist.shape != (n_bins,):
        raise ValueError("pdist must have one value per bin ({}), got {}".format(n_bins, pdist.size))
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin edges must be strictly increasing")
    if edges[0] <= 0:
        raise ValueError("bin edges must be positive")

    s = float(shift)
    if s == 0.0:
        # Identity: return the inputs unchanged (no copy, no numerical drift).
        return pdist, edges

    def _linear(t):
        """Linear edge interpolation for a shift t in [-1, 1]: each interior/top edge moves down by
        a fraction t of its own bin-width (t=+1 is exactly one full bin toward smaller radii), and
        the lowest edge is extrapolated linearly below e_0 (e_{-1} = 2*e_0 - e_1)."""
        ne = np.empty_like(edges)
        ne[1:] = (1.0 - t) * edges[1:] + t * edges[:-1]
        e_minus_1 = 2.0 * edges[0] - edges[1]
        ne[0] = (1.0 - t) * edges[0] + t * e_minus_1
        return ne

    if abs(s) <= 1.0:
        # Exact behaviour for |s| <= 1 (identity at 0, one full bin down/up at +/-1).
        new_edges = _linear(s)
    else:
        # Beyond one bin the linear extrapolation can fold onto itself on irregularly-spaced bins
        # (e.g. the wide top MIP bins), so shift exactly one full bin in the sign direction and then
        # apply a log-multiplicative scale for the remaining |s|-1. Multiplying by a positive
        # constant preserves strict ordering, so this is valid for ANY magnitude: positive s shrinks
        # every pore (scale < 1), negative grows them. It is continuous with the linear branch at
        # |s| = 1 and matches "k bins smaller" exactly on a geometric distribution.
        sign = 1.0 if s > 0 else -1.0
        w_mean = float(np.mean(np.diff(np.log(edges))))
        new_edges = _linear(sign) * np.exp(-sign * (abs(s) - 1.0) * w_mean)

    # Reject shifts so large that the distribution folds onto itself (adjacent bins cross). This is
    # checked on the raw edges first, before any clamping, so an invalid shift fails cleanly without
    # a spurious clamp warning. The |s| > 1 path above is guaranteed increasing; this still guards
    # the |s| <= 1 path (which can fold near |s| = 1 on very irregular bins).
    if not np.all(np.diff(new_edges) > 0):
        raise ValueError(
            "shifted bin edges are no longer strictly increasing for shift={:.3g}; "
            "reduce the magnitude of the shift.".format(s)
        )

    # Guard: the extrapolated lowest edge can go non-positive for moderately large positive shifts.
    if new_edges[0] <= 0:
        warn(
            "shift_pore_distribution: shift={:.3g} pushes the lowest edge to {:.3g} (<= 0); "
            "clamping to a small positive floor.".format(s, float(new_edges[0]))
        )
        new_edges[0] = np.nextafter(0.0, 1.0)
        # Clamping raises e'_0; re-verify the first gap is still positive (defensive — for the
        # intended small gauge adjustments |shift| of order 1 this never triggers).
        if new_edges[1] - new_edges[0] <= 0:
            raise ValueError(
                "shifted bin edges are no longer strictly increasing after clamping for shift={:.3g}; "
                "reduce the magnitude of the shift.".format(s)
            )

    return pdist, new_edges


def inverse_transform_sampling_from_cdf(cdf, bins, n_samples=1000, low=0, high=np.inf, **kwargs):
    """cdf is the cumulative density function. bins and cdf must have the same length"""

    inv_cdf = interpolate.interp1d(cdf, bins, bounds_error=None, fill_value="extrapolate")
    return np.clip(inv_cdf(np.random.random(n_samples)), a_min=low, a_max=high)
