# coding=utf-8
import contextlib
import io
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import networkx as nx
from pnm import pore_network

import pnm
from .Stats import distribution
from .utils import *

from scipy.spatial import cKDTree
from functools import partial
from warnings import warn

# TODO: clean the code, add support for periodicity


def cubic_PNM(extent, shape, distributions_props, lattice="c", mode="pore", check_geometry=True, set_radius=True):
    """Generate a regular cubic/bcc pore network with full connectivity. Radius distribution of pores or throats can be fixed to one or several given distributions
    \nextent=(lx,ly,lz) : inner extent (from pore centers)
        \nshape = (nx,ny,nz) nb of pores along each axis [primary grid]
        \npore_distributions_props list of tuples (props,frac,low,high) where 'props' is a dict containing distribution properties
        \n and frac ]0;1] is its occurence probability
        \n lattice is either 'c' or 'bcc' (cubic or body-centered cubic).
        \n Note that bcc is only implemented for 3D networks [i.e. shape[i] >=2]. n
        \n in bcc mode, nx,ny,nz give the number of pores on the cubic grid. centered pore are added after.
        \n mode : either 'pore' or 'throat': fix the radius distribution of either pores or throats
        \nNote: if the sum of fracs is > 1, the radius list is truncated
    """

    net = pore_network()
    net.graph.clear()
    shape = np.array(shape)
    net.graph.graph["extent"] = extent
    net.graph.graph["graph_type"] = "cubic_regular"
    spacing = np.array(extent) / shape
    net.graph.graph["spacing"] = spacing

    # Computing pores center
    npores = np.prod(shape)
    labels = np.arange(npores)

    x = np.linspace(0.5 * spacing[0], extent[0] - 0.5 * spacing[0], shape[0])
    y = np.linspace(0.5 * spacing[1], extent[1] - 0.5 * spacing[1], shape[1])
    z = np.linspace(0.5 * spacing[2], extent[2] - 0.5 * spacing[2], shape[2])
    x, y, z = np.meshgrid(x, y, z, indexing="ij")
    c = np.swapaxes(np.vstack([x.ravel(), y.ravel(), z.ravel()]), 0, 1)

    # Adding np nodes
    net.graph.add_nodes_from(labels)
    # Set centers
    nx.set_node_attributes(net.graph, dict(zip(labels, c)), "center")

    catlist = [set() for _ in labels]

    nx.set_node_attributes(net.graph, dict(zip(labels, catlist)), name="category")  # creating attribute
    labels = labels.reshape(tuple(shape), order="C")

    ind = labels[0, :, :].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("x-"), ind))
    ind = labels[-1, :, :].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("x+"), ind))
    ind = labels[:, 0, :].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("y-"), ind))
    ind = labels[:, -1, :].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("y+"), ind))
    ind = labels[:, :, 0].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("z-"), ind))
    ind = labels[:, :, -1].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("z+"), ind))

    inner_nodes = labels[1:-1, 1:-1, 1:-1].ravel()
    cat = [set(["inner"])] * len(inner_nodes)
    nx.set_node_attributes(net.graph, dict(zip(inner_nodes, cat)), name="category")

    # Adding throats
    pairs = list(zip(labels[:-1, ...].ravel(), labels[1:, ...].ravel()))
    pairs.extend(list(zip(labels[:, :-1, :].ravel(), labels[:, 1:, :].ravel())))
    pairs.extend(list(zip(labels[:, :, :-1].ravel(), labels[:, :, 1:].ravel())))

    net.graph.add_edges_from(pairs)

    if lattice == "bcc":  # Creating the second lattice
        bcc_shape = np.clip(shape - 1, 1, None)
        bcc_npores = np.prod(bcc_shape)
        bcc_labels = np.arange(bcc_npores) + npores

        x = np.linspace(spacing[0], extent[0] - spacing[0], bcc_shape[0])
        y = np.linspace(spacing[1], extent[1] - spacing[1], bcc_shape[1])
        z = np.linspace(spacing[2], extent[2] - spacing[2], bcc_shape[2])
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        c = np.swapaxes(np.vstack([x.ravel(), y.ravel(), z.ravel()]), 0, 1)

        # Adding nodes
        net.graph.add_nodes_from(bcc_labels)
        # Set centers
        nx.set_node_attributes(net.graph, dict(zip(bcc_labels, c)), "center")

        cat = [set(["inner"])] * bcc_npores
        nx.set_node_attributes(net.graph, dict(zip(bcc_labels, cat)), name="category")

        bcc_labels = bcc_labels.reshape(tuple(bcc_shape), order="C")

        # Adding throats
        pairs = list(zip(bcc_labels.ravel(), labels[:-1, :-1, :-1].ravel()))
        pairs.extend(list(zip(bcc_labels.ravel(), labels[1:, :-1, :-1].ravel())))
        pairs.extend(list(zip(bcc_labels.ravel(), labels[:-1, 1:, :-1].ravel())))
        pairs.extend(list(zip(bcc_labels.ravel(), labels[:-1, :-1, 1:].ravel())))

        pairs.extend(list(zip(bcc_labels.ravel(), labels[:-1, 1:, 1:].ravel())))
        pairs.extend(list(zip(bcc_labels.ravel(), labels[1:, 1:, 1:].ravel())))
        pairs.extend(list(zip(bcc_labels.ravel(), labels[1:, :-1, 1:].ravel())))
        pairs.extend(list(zip(bcc_labels.ravel(), labels[1:, 1:, :-1].ravel())))

        net.graph.add_edges_from(pairs)

    if set_radius:
        net.set_radius_distribution(distributions_props, mode)
        net.compute_extent()

    if check_geometry:  # Maybe you want to check geom after doing something else...
        net.check_overlapping(fit_radius=False, merge=True, mindist="auto", update_geometry=True)
        net.compute_geometry()

    return net


def cubic_PNM_no_throats(extent, shape, distributions_props, lattice="c", mindist="auto"):
    """Generate a regular cubic/bcc pore network *without throats*. Several pore radius distribution can be provided
    BEWARE: ***As throats are not created, geometry is not complete.***
    \nextent=(lx,ly,lz)
        \nshape = (nx,ny,nz) nb of pores along each axis
        \npore_distributions_props list of tuples (props,frac,low,high) where 'props' is a dict containing distribution properties
        \n and frac ]0;1] is its occurence probability
        \n lattice is either 'c' or 'bcc' (cubic or body-centered cubic)
        \n mindist is the min distance between two pore surface

        \nNote: if the sum of fracs is > 1, the radius list is truncated
    """

    net = pore_network()
    net.graph.clear()
    shape = np.array(shape)
    net.graph.graph["extent"] = extent
    net.graph.graph["graph_type"] = "cubic"
    spacing = np.array(extent) / shape
    net.graph.graph["spacing"] = spacing

    # Computing possible pores center
    npores = np.prod(shape)
    labels = np.arange(npores)

    x = np.linspace(0.5 * spacing[0], extent[0] - 0.5 * spacing[0], shape[0])
    y = np.linspace(0.5 * spacing[1], extent[1] - 0.5 * spacing[1], shape[1])
    z = np.linspace(0.5 * spacing[2], extent[2] - 0.5 * spacing[2], shape[2])
    x, y, z = np.meshgrid(x, y, z, indexing="ij")
    c = np.swapaxes(np.vstack([x.ravel(), y.ravel(), z.ravel()]), 0, 1)

    # Adding np nodes
    net.graph.add_nodes_from(labels)
    # Set centers
    nx.set_node_attributes(net.graph, dict(zip(labels, c)), "center")

    catlist = [set() for _ in labels]

    nx.set_node_attributes(net.graph, dict(zip(labels, catlist)), name="category")  # creating attribute
    labels = labels.reshape(tuple(shape), order="C")

    ind = labels[0, :, :].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("x-"), ind))
    ind = labels[-1, :, :].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("x+"), ind))
    ind = labels[:, 0, :].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("y-"), ind))
    ind = labels[:, -1, :].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("y+"), ind))
    ind = labels[:, :, 0].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("z-"), ind))
    ind = labels[:, :, -1].ravel()
    _ = list(map(lambda i: net.graph.nodes[i]["category"].add("z+"), ind))

    inner_nodes = labels[1:-1, 1:-1, 1:-1].ravel()
    cat = [set(["inner"])] * len(inner_nodes)
    nx.set_node_attributes(net.graph, dict(zip(inner_nodes, cat)), name="category")

    if lattice == "bcc":  # Creating the second lattice
        bcc_shape = np.clip(shape - 1, 1, None)
        bcc_npores = np.prod(bcc_shape)
        bcc_labels = np.arange(bcc_npores) + npores

        x = np.linspace(spacing[0], extent[0] - spacing[0], bcc_shape[0])
        y = np.linspace(spacing[1], extent[1] - spacing[1], bcc_shape[1])
        z = np.linspace(spacing[2], extent[2] - spacing[2], bcc_shape[2])
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        c = np.swapaxes(np.vstack([x.ravel(), y.ravel(), z.ravel()]), 0, 1)

        # Adding nodes
        net.graph.add_nodes_from(bcc_labels)
        # Set centers
        nx.set_node_attributes(net.graph, dict(zip(bcc_labels, c)), "center")

        cat = [set(["inner"])] * bcc_npores
        nx.set_node_attributes(net.graph, dict(zip(bcc_labels, cat)), name="category")

        bcc_labels = bcc_labels.reshape(tuple(bcc_shape), order="C")

    net.set_radius_distribution(distributions_props, mode="pore")
    net.check_overlapping(fit_radius=False, merge=True, mindist=mindist)
    net.compute_extent()

    return net


def random_PNM(
    npores,
    extent,
    porosity,
    rdist,
    sratio=1,
    PRTRR=1,
    minc=3,
    maxc=100,
    BCgridsize=100,
    fit_box=False,
    add_periodic=True,
    periodic_faces=("x", "y"),
    search_radius_factor=2,
    min_neighbors_factor=2,
    sort_by_radius=False,
    minc_search_factor=None,
    large_pair_radius_frac=None,
    large_pair_radius=None,
    large_pair_min_connections=None,
    large_pair_max_gap=2.5,
    large_radius_cutoff=None,
    large_search_factor=2.0,
    search_radius_factor2=3.0,
    minc2=0,
    throat_method="surf",
    phase1_order="decreasing",
    order="increasing",
    phase1_radius_factor=2,
    phase1_min_neighbors=32,
    phase1_z=1,
    phase1_mode="distance",
    phase1_neighbours=32,
    phase2_radius_factor=3,
    phase2_min_neighbors=32,
    phase2_zmin=2,
    packing_method="kdtree",
    nb_trials=200,
    max_unallocated_frac=0.05,
    packing_verbose=False,
    pool_size=4096,
    tree_rebuild_interval=1024,
    voxel_resolution=96,
    seed=None,
    pores=None,
):
    """Generate a random packing of pores following the given radius distribution rdist with a target porosity "porosity"
    Initial box extent are given, but can be modified dynamically to get the correct porosity
    npores: number of pore
    extent: initial extent
    porosity: target pore-body porosity [NOTE: the total porosity will be higher because of throats]
    rdist: radius distribution: list of radius distributions
        (list of 2-tuple containing a dict with the dist function and probability a pore follow this distribution)
        ({"func":distribution_function},volume_fraction)
    PRTRR: pore radius to to throat radius ratio
    minc minimal pore connectivity (default 1)
    maxc maximal pore connectivity (default 100)
    sort_by_radius: if True, pass-0 connections favor the largest neighbors of each pore
        (favors throats between large pores). Default is False.
    minc_search_factor: only used with sort_by_radius=True; widens the pass-0 candidate pool
        to all pores within center distance minc_search_factor * r1 so that large partners
        beyond the immediate neighborhood can be found. Default is None (k nearest neighbors).
    large_pair_radius_frac: only used by throat_method="surf"; if set, interconnects the
        largest pores in a third pass — every pair of pores with radius >=
        large_pair_radius_frac * max_pore_radius and surface-to-surface gap <=
        large_pair_max_gap gets a throat (recovers the very large throats that random
        placement scatters apart). Default is None (disabled).
    large_pair_radius: only used by throat_method="surf_v2"; phase-2 threshold in µm — every
        pore with radius >= this value gets at least large_pair_min_connections connections to
        other such pores, ignoring the surface budget (bounded by n_large * floor / 2 new
        throats). Set it to PRTRR x the target large-throat radius so forced throats land in
        that band. Default is None (disabled).
    large_pair_min_connections: only used by throat_method="surf_v2"; phase-2 floor count per
        large pore (default None = minc; raise it above minc to add large throats beyond what
        phases 0-1 already provide at large N). Default is None.
    large_pair_max_gap: maximum surface-to-surface gap (length, µm) for the large-pair pass
        of either method ("surf": all pairs within the cap; "surf_v2" phase 2: eligibility
        cap for the floor candidates — use None to disable the cap entirely). Default is 2.5.
    large_radius_cutoff: only used by throat_method="surf_v3"; radius threshold (µm, compared
        against PORE radii) that enables pass 1 of add_throats_by_surf_v3 — an independent
        sub-network of the pores with radius >= this value, each guaranteed at least ``minc``
        connections within the large class before the budget-driven pass 2 runs. Default is
        None (pass 1 disabled; single-pass v3).
    large_search_factor: only used by throat_method="surf_v4"; pass-1 candidate ball has
        center-distance radius ``large_search_factor * r_pore + r_pore`` — the wide ball whose
        candidates are sorted by radius so each pore connects to its ``minc`` LARGEST partners.
        Default is 2.0 (ball = 3 * r_pore).
    search_radius_factor2: only used by throat_method="surf_v4"; pass-2 candidate ball has
        center-distance radius ``search_radius_factor2 * r_pore + r_pore`` — the initial search
        region for the nearest-by-surface-distance fill. The minc2 floor phase widens beyond
        this if needed. Default is 3.0 (ball = 4 * r_pore).
    minc2: only used by throat_method="surf_v4"; minimum total connectivity enforced per pore
        in pass 2. If a pore ends its initial fill with fewer than ``minc2`` connections, the
        search ball is widened iteratively until the floor is met or no candidates remain.
        Default is 0 (disabled).
    throat_method: which throat-generation routine to use. "surf" (default) uses the
        two-phase add_throats_by_surf (see order / phase1_* / phase2_* below; maxc maps to its
        zmax); "surf_v2" uses the redesigned two-phase
        add_throats_by_surf_v2 with its optional large-pair floor (large_pair_radius). For
        "surf_v2", search_radius_factor maps to its phase-1 surface-gap limit and
        minc_search_factor to its phase-0 widening. "surf_v3" uses the two-pass
        add_throats_by_surf_v3: an optional pass 1 (independent large-pore sub-network, see
        large_radius_cutoff; ``minc`` is its per-pore connectivity floor) followed by a
        budget-driven pass 2 with random pore selection. "surf_v4" uses the two-pass
        add_throats_by_surf_v4 (intended v1 replacement): a pass 1 that connects each pore to
        its ``minc`` LARGEST partners within a wide ball (see large_search_factor), followed by
        a pass 2 that fills each pore's budget with nearest-by-surface-distance partners
        (see search_radius_factor2) and an optional minc2 connectivity floor.
        Default is "surf".
    phase1_order: only used by throat_method="surf_v2"; order in which pores are processed
        in phase 1 ("decreasing", "increasing" or "random"). Default is "decreasing".
    order: only used by throat_method="surf"; order in which pores are processed in its phase 2,
        by radius ("increasing" = smallest first (default), "decreasing" = largest first,
        "random" = shuffled).
    phase1_radius_factor: only used by throat_method="surf"; phase-1 candidate ball radius as a
        multiple of the active pore radius — center-distance radius (factor + 1) * r_i. Default is 2.
    phase1_min_neighbors: only used by throat_method="surf"; minimum candidates in the phase-1
        ball; if fewer lie within it, the search widens to this many nearest pores. Default is 32.
    phase1_z: only used by throat_method="surf"; number of LARGEST-radius neighbors each pore
        connects to in phase 1 (subject to budget and zmax). Phase 1 runs only when > 0. Default is 1.
    phase1_mode: only used by throat_method="surf"; phase-1 candidate-pool selection — "distance"
        (default) queries a ball of center-distance radius (phase1_radius_factor + 1) * r_i widened to
        at least phase1_min_neighbors pores; "neighbours" uses the fixed-size pool of the
        phase1_neighbours nearest pores by center distance (the old single-pass v1 pass-0 pool).
    phase1_neighbours: only used by throat_method="surf" with phase1_mode="neighbours"; number of
        nearest pores (by center distance) from which the phase1_z largest partners are drawn.
        Default is 32; ignored in "distance" mode.
    phase2_radius_factor: only used by throat_method="surf"; phase-2 candidate ball radius —
        center-distance radius factor * r_i + R_max (R_max = largest pore radius). Default is 3.
    phase2_min_neighbors: only used by throat_method="surf"; minimum candidates in the phase-2
        ball; if fewer lie within it, the search widens to this many nearest pores. Default is 32.
    phase2_zmin: only used by throat_method="surf"; phase-2 connectivity floor — a pore with fewer
        than this many throats after the budget fill is force-connected to its nearest neighbors,
        overriding the surface budget but keeping covered fraction < 1. Default is 2.
    BCgrid size: In order to assign the boundary conditions, the volume is discretised using a grid containing  BCgridsize cells per side
    sratio: maximal pore surface area over total intersect area of connected throats ratio (default 1, which is not optimal). sratio can also be a tuple
    with (c, l, h, sigma) c center o fa normal distribution, l and h lowest and highest values and sigma the stdev. With throat_method="surf_v3" it can
    additionally be a diameter-dependent profile dict (keys "sr_small", "sr_large", "r_lo_um", "r_hi_um"; see pnm.Stats.sratio_profile_values) evaluated per
    pore from its radius — pass the SAME dict as sratio_profile to pnm.Stats.invert_pore_distribution_kernel_gamma so model and generator budgets match.
    packing_method: packing algorithm to use. One of:
        - "kdtree" (default): candidate pools + cKDTree vectorized batch rejection (fastest at low/mid porosity).
        - "kdtree_importance": as "kdtree" plus occupancy-map importance sampling for the
          candidate pools (best in dense packing, i.e. high target porosity or very large N).
        - "opt": brute-force O(N^2) overlap test (slowest, kept for reference).
    max_unallocated_frac: max fraction of unallocated pore volume before restart (used by "kdtree" and "kdtree_importance").
    nb_trials: number of random placement attempts per pore before giving up.
        Used by the "opt" method only; ignored by "kdtree"/"kdtree_importance"
        (their give-up control is max_empty_pools, not exposed here). Default is 100.
    packing_verbose: print packing progress.
    pool_size: number of candidate points per pool (used by "kdtree" and "kdtree_importance").
        Default is 4096.
    tree_rebuild_interval: rebuild the cKDTree every M placements (used by "kdtree" and
        "kdtree_importance"). Default is 1024.
    voxel_resolution: number of voxels per side of the occupancy map (used by
        "kdtree_importance" only). Default is 96.
    seed: random seed for reproducible generation. If given, the global numpy random state
        is seeded (radius draws) and, for the "kdtree"/"kdtree_importance" methods, the
        internal pool-sampling generator is also seeded; packing restarts then use
        deterministic child streams derived from the same seed, so a seeded run is fully
        reproducible even when it restarts. Default is None (non-deterministic).
    pores: optional pre-built pore-only network (a ``pore_network`` with no throats). If given,
        packing is skipped entirely and this network is used as-is — the throat-generation
        parameters still apply to it. Useful for parametric studies where several grid points
        share one pore placement (npores/porosity/rdist/packing_* are then ignored).
    TODO: use a sratio distribution instead of a constant value (partially done for surf_v3: per-pore
        truncated-normal draw and diameter-dependent profile dict, see above)
    """

    if pores is not None:
        pn = pores  # caller-provided pore set: skip packing, go straight to throat generation
    else:
        if seed is not None:
            np.random.seed(seed)  # deterministic radius draws for all packing methods

        # Build the distributions_props list for the packing routines.
        # rdist may be passed either as a dict of parameters (the default
        # distribution() generator is then used) or as a dict containing a "func"
        # key, or directly as a callable distribution function.
        if isinstance(rdist, dict):
            dist_props = (rdist, 1)
        else:
            dist_props = ({"func": rdist}, 1)

        if packing_method == "opt":
            pn = pnm.random_packing_opt(
                extent=extent,
                nbnodes=npores,
                distributions_props=[dist_props],
                target_porosity=porosity,
                mindist="auto",
                nb_trials=nb_trials,
                BCstep=BCgridsize,
                restart=True,
                fit_box=fit_box,
            )
        elif packing_method == "kdtree":
            pn = pnm.random_packing_kdtree(
                extent=extent,
                nbnodes=npores,
                distributions_props=[dist_props],
                target_porosity=porosity,
                mindist="auto",
                BCstep=BCgridsize,
                restart=True,
                fit_box=fit_box,
                max_unallocated_frac=max_unallocated_frac,
                pool_size=pool_size,
                tree_rebuild_interval=tree_rebuild_interval,
                verbose=packing_verbose,
                seed=seed,
            )
        elif packing_method == "kdtree_importance":
            pn = pnm.random_packing_kdtree_importance(
                extent=extent,
                nbnodes=npores,
                distributions_props=[dist_props],
                target_porosity=porosity,
                mindist="auto",
                BCstep=BCgridsize,
                restart=True,
                fit_box=fit_box,
                max_unallocated_frac=max_unallocated_frac,
                pool_size=pool_size,
                tree_rebuild_interval=tree_rebuild_interval,
                verbose=packing_verbose,
                voxel_resolution=voxel_resolution,
                seed=seed,
            )
        else:
            raise ValueError(
                "Unknown packing_method '{}'. Expected 'opt', 'kdtree' or 'kdtree_importance'.".format(packing_method)
            )

    if throat_method == "surf":
        if sort_by_radius or minc_search_factor is not None or large_pair_radius_frac is not None:
            warn(
                "throat_method='surf' now uses the two-phase add_throats_by_surf; "
                "sort_by_radius, minc_search_factor and large_pair_radius_frac are ignored."
            )
        pnm.add_throats_by_surf(
            pn,
            ratio=sratio,
            PRTRR=PRTRR,
            zmax=maxc,
            order=order,
            border_correction=True,
            phase1_radius_factor=phase1_radius_factor,
            phase1_min_neighbors=phase1_min_neighbors,
            phase1_z=phase1_z,
            phase1_mode=phase1_mode,
            phase1_neighbours=phase1_neighbours,
            phase2_radius_factor=phase2_radius_factor,
            phase2_min_neighbors=phase2_min_neighbors,
            phase2_zmin=phase2_zmin,
        )
    elif throat_method == "surf_v2":
        if large_pair_radius_frac is not None:
            warn("throat_method='surf_v2' ignores large_pair_radius_frac (use the absolute large_pair_radius instead).")
        pnm.add_throats_by_surf_v2(
            pn,
            ratio=sratio,
            PRTRR=PRTRR,
            minc=minc,
            maxc=maxc,
            sort_by_radius=sort_by_radius,
            border_correction=True,
            search_radius_factor1=minc_search_factor,
            search_radius_factor2=search_radius_factor,
            min_neighbors_factor=min_neighbors_factor,
            phase1_order=phase1_order,
            seed=seed,
            large_pair_radius=large_pair_radius,
            large_pair_min_connections=large_pair_min_connections,
            large_pair_max_gap=large_pair_max_gap,
        )
    elif throat_method == "surf_v3":
        if large_pair_radius_frac is not None or minc_search_factor is not None:
            warn("throat_method='surf_v3' ignores large_pair_radius_frac and minc_search_factor (pure budget + random selection).")
        pnm.add_throats_by_surf_v3(
            pn,
            ratio=sratio,
            PRTRR=PRTRR,
            minc=minc,
            large_radius_cutoff=large_radius_cutoff,
            border_correction=True,
            seed=seed,
        )
    elif throat_method == "surf_v4":
        if large_pair_radius_frac is not None or minc_search_factor is not None:
            warn("throat_method='surf_v4' ignores large_pair_radius_frac and minc_search_factor (pass 1 uses its own wide ball + radius sort).")
        pnm.add_throats_by_surf_v4(
            pn,
            ratio=sratio,
            PRTRR=PRTRR,
            minc1=minc,
            maxc=maxc,
            search_radius_factor1=large_search_factor,
            search_radius_factor2=search_radius_factor2,
            minc2=minc2,
            border_correction=True,
        )
    else:
        raise ValueError("Unknown throat_method '{}'. Expected 'surf', 'surf_v2', 'surf_v3' or 'surf_v4'.".format(throat_method))

    if add_periodic:
        pnm.add_periodic_throats(
            pn,
            delta=0.01,
            faces=periodic_faces,
            PRTRR=PRTRR,
            minc=minc,
            maxc=maxc,
            search_radius_factor=search_radius_factor,
            min_neighbors_factor=min_neighbors_factor,
        )

    pn.compute_geometry(autothroats=False)

    return pn


def _parallel_random_PNM_worker(task):
    """Generate one network in this process (worker for :func:`parallel_random_PNM`).

    ``task`` is a ``(kwargs, out_path)`` pair: ``kwargs`` are the keyword arguments of
    :func:`random_PNM` and ``out_path`` is None when the caller wants the graph object back
    instead of a saved file.
    """
    kwargs = dict(task[0])
    out_path = task[1]

    seed = kwargs.get("seed")
    if seed is not None:
        # Each worker has its own process (and RNG state); seeding here makes throat generation
        # reproducible even when packing is skipped (pores= given), where random_PNM itself does
        # not touch the global numpy state. When packing runs, random_PNM re-seeds with the same
        # value, so this is a no-op there.
        np.random.seed(seed)

    pores = kwargs.get("pores")
    if isinstance(pores, (str, os.PathLike)):
        # Read inside the worker: large pore-only graphs never cross the process boundary.
        kwargs["pores"] = pore_network.read(str(pores), method="pickle")

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # keep the worker's progress output out of the log
        pn = random_PNM(**kwargs)
    elapsed = time.perf_counter() - t0
    n_pores = pn.graph.number_of_nodes()
    n_throats = pn.graph.number_of_edges()

    if out_path is not None:
        base = str(Path(out_path))
        if base.endswith(".gml"):
            base = base[: -len(".gml")]  # write() appends the extension itself
        pn.write(base, method="gml")
        return {"path": base + ".gml", "n_pores": n_pores, "n_throats": n_throats, "time_s": elapsed}
    return pn


def parallel_random_PNM(specs, n_jobs=None):
    """Generate several pore networks in parallel forked workers.

    Each entry of ``specs`` is a dict of keyword arguments for :func:`random_PNM`; the networks
    are generated independently (every worker gets its own process and RNG state). Two keys are
    intercepted by this function before the call:

    - ``"pores"`` may be a path (str or Path) to a pickled pore-only network instead of a
      ``pore_network`` object — it is read inside the worker, so large graphs never cross the
      process boundary.
    - ``"out_path"``: when present, the generated network is saved there as GML and a summary
      dict ``{"path", "n_pores", "n_throats", "time_s"}`` is returned for that entry instead of
      the graph object.

    Parameters
    ----------
    specs : list of dict
        Per-network keyword arguments for random_PNM (see above). The first four positional
        parameters of random_PNM (npores, extent, porosity, rdist) must be present in every spec;
        pass ``rdist=None`` when a pre-built pore set is supplied via ``pores``. Every value must
        be picklable (the specs are sent to the forked workers), so use module-level callables —
        not lambdas or closures — for rdist.
    n_jobs : int, optional
        Number of parallel workers; default ``min(len(specs), os.cpu_count())``. 1 runs the
        specs sequentially in the calling process (useful for debugging).

    Returns
    -------
    list
        Aligned with ``specs``: pore_network objects, or summary dicts where out_path was given.
    """
    if not specs:
        return []
    n_jobs = min(n_jobs, len(specs)) if n_jobs else min(len(specs), os.cpu_count() or 1)

    tasks = [({k: v for k, v in s.items() if k != "out_path"}, s.get("out_path")) for s in specs]
    if n_jobs <= 1:
        return [_parallel_random_PNM_worker(t) for t in tasks]

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=n_jobs) as pool:
        results = pool.map(_parallel_random_PNM_worker, tasks)
    return list(results)


def random_packing_opt(
    extent,
    nbnodes,
    distributions_props,
    target_porosity=0.5,
    mindist="auto",
    nb_trials=1000,
    BCstep=100,
    restart=True,
    fit_box=False,
    _restart_count=0,
):
    """
    Spherical pores packing following an arbitrary distribution using brute force approach (-> slow for large number of pores or compact packing...)
    extent: initial extent [Lx, Ly, Lz]
    nbnodes: number of pores
    distributions_props: radius distribution list with probability
    target_porosity: tageted pore-body porosity [not the total porosity]
    mindist: minimal distance between two pore surface. If auto mindist = min (R1, R2) * 0.5
    nb_trials: number of trials per pore
    BCStep:  To detect boundary pores, we use a discrete grid at each face with BCSteps to dicretize the surface (BCsteps * BCSteps)
    Fro each grid cell, we find one BC pore which its nearest neighbor. Of course, several grid cell can mark the same pore. So there are at most BCsteps * BCSteps BC pores
    restart : if True, the algorithm restart if a pore cannot be placed
    fit_box : if True, the number of pores is modified in order to fit into the provided extent (default False)

    \nNote that this function returns only pores. Throats have to be added later"""

    length_OK = False
    extent = np.array(extent)
    original_extent = extent.copy()  # saved before scaling, for correct restart behavior

    if fit_box:
        length = 1
        radii = generate_radii(distributions_props, nbnodes)
        void_volume = (np.pi * (4 / 3) * radii**3).sum()
        temp_porosity = void_volume / np.prod(extent)
        counter = 0
        # Si l'écart est supérieur à 0.05%
        while abs(temp_porosity - target_porosity) > 0.0005 and counter < 5000:
            counter += 1
            nbnodes = int(np.around(nbnodes * target_porosity / temp_porosity))
            radii = generate_radii(distributions_props, nbnodes)[0:nbnodes]
            void_volume = (np.pi * (4 / 3) * radii**3).sum()
            temp_porosity = void_volume / np.prod(extent)

        print(
            "{} pores will be generated to fit the target porosity of {}% in the given box".format(
                nbnodes, 100 * temp_porosity
            )
        )
        max_radius = max(radii)

    else:
        # Provided extent is adapted in order to fit the target porosity
        while not length_OK:
            radii = generate_radii(distributions_props, nbnodes)

            radii = np.array(radii[0:nbnodes])
            max_radius = max(radii)
            # Le volume est déterminé à partir de l'objectif de  porosité et du volume des pores
            void_volume = (np.pi * (4 / 3) * radii**3).sum()
            length = (void_volume / (target_porosity * np.prod(extent))) ** (1 / 3)
            # En fonction de l'objectif de porosité, il est possible que la taille calculée soit
            # inférieure aux dimensions du plus gros pore -> on augmente le nb de pores

            if radii.size > 1:
                lim = 2.1 * radii[:2].sum()
            else:
                lim = 2 * max_radius
            if extent.min() * length < lim:
                print(
                    "Computed min side length {} is smaller than the sum of the 2 largest pores diameter {}".format(
                        extent.min() * length, max_radius
                    )
                )
                print("Increasing number of pores to :", nbnodes * 1.5)
                nbnodes *= 1.5
                length_OK = False
            else:
                break

    min_radius = min(radii)
    # avg_radius = radii.sum() / radii.size

    extent = extent * length

    print("Max radius = {:.2e} and Min radius = {:.2e}".format(max_radius, min_radius))
    print("Extent = ", extent)
    print("Extent/(2*Rmax) = ", end="")
    print(["{:.2f}".format(e / (2 * max_radius)) for e in extent])

    net = pore_network()
    net.graph.clear()
    net.graph.graph["extent"] = list(extent)
    net.graph.graph["graph_type"] = "random_packing"

    radii.sort()
    radii = radii[::-1]  # On place les plus gros pores d'abord

    occupied_centers = []
    occupied_radii = []
    # occupied_categories = []
    unallocated_radii = []

    total_volume = np.prod(extent)
    void_volume = 0

    # maxit = nbnodes * 10

    if mindist == "auto":
        gap = min_radius * 0.25
    else:
        gap = mindist

    pore_added = True

    # it = 0
    trials = 0

    print("Allocated pores | current radius | current trials | current porosity | unaloccated pores")

    unallocated_volume = 0

    for i, r in enumerate(radii):
        pore_added = False

        for j in range(nb_trials):
            trials += 1

            # coordonnée aléatoire
            if i == 0:  # First (and biggest) pore near the boundary
                coords = r + np.random.rand(3) * np.minimum(1.2 * np.array([r, r, r]), extent - 2 * r)

            else:
                coords = r + np.random.rand(3) * (extent - 2 * r)

            if len(occupied_centers) > 0:
                distance = query_mindist_loops(coords, np.array(occupied_centers), np.array(occupied_radii))

            else:
                distance = np.inf

            if distance > r + gap:
                occupied_centers.append(list(coords))  # Convert to list to be serializable when saving the network
                occupied_radii.append(r)
                # occupied_categories.append(category)
                void_volume += np.pi * (4 / 3) * r**3
                pore_added = True
                print(
                    " {:14} {:16.2e} {:16} {:16.2f} % {:19}   ".format(
                        len(occupied_centers), r, trials, 100 * void_volume / total_volume, len(unallocated_radii)
                    ),
                    end="\r",
                    flush=True,
                )
                # del(radii[j])
                break
            # else:
            # todelete.append(j)

        # for n in todelete:
        # del(coords_dict[n])

        if not pore_added:
            unallocated_radii.append(i)
            unallocated_volume += (4 / 3) * np.pi * r**3
            # Si le volume non alloué dépasse 5% du volume poreux cible
            if restart and unallocated_volume > 0.05 * target_porosity * total_volume:
                if _restart_count >= 5:
                    print("\n!!!--- Max restart count reached, stopping ---!!!\n")
                else:
                    print("\n!!!--- Restarting ---!!!\n")
                    return random_packing_opt(
                        extent=original_extent,
                        nbnodes=nbnodes,
                        distributions_props=distributions_props,
                        target_porosity=target_porosity,
                        mindist=mindist,
                        nb_trials=nb_trials,
                        BCstep=BCstep,
                        restart=restart,
                        fit_box=fit_box,
                        _restart_count=_restart_count + 1,
                    )

    print("\n{} pores generated".format(len(occupied_centers)))

    labels = np.arange(len(occupied_centers))
    net.graph.add_nodes_from(labels)
    nx.set_node_attributes(net.graph, dict(zip(labels, occupied_centers)), "center")
    nx.set_node_attributes(net.graph, dict(zip(labels, occupied_radii)), "radius")
    nx.set_node_attributes(net.graph, False, "periodic")
    # nx.set_node_attributes(net.graph,dict(zip(labels,occupied_categories)),'category')

    pnm.SetFaceBCnodes(net, BCstep)

    print("Porosity:", void_volume / total_volume)
    # unallocated_volume = (np.pi*(4/3)*radii[np.array(unallocated_radii,dtype=np.int)]**3).sum()
    print(
        "Unallocated volume: {:.2e}, corresponding void_fraction: {:.2f} %".format(
            unallocated_volume, 100 * unallocated_volume / total_volume
        )
    )

    return net


# ----------------------------------------------------------------------
# Fast packing with candidate pools + cKDTree (options A and B)
# ----------------------------------------------------------------------


def _prepare_packing_radii(extent, nbnodes, distributions_props, target_porosity, fit_box, verbose, mindist="auto"):
    """Generate pore radii and adapt the box size or pore count to reach the target porosity.

    Shared pre-processing for the kdtree packing routines; same adaptation logic as
    :func:`random_packing_opt`, but with a corrected "box too small" guard (see Notes). Two strategies are available:

    - ``fit_box=False`` (default): the number of pores is fixed and the box side length is
      scaled so that the total pore volume matches ``target_porosity``. If in the resulting
      box the two largest pores cannot both be placed, the pore count is increased by 50%
      until they fit.
    - ``fit_box=True``: the box is kept as given and the number of pores is adjusted so that
      the target porosity is met inside it.

    Parameters
    ----------
    extent : array-like of float
        Initial box size [Lx, Ly, Lz].
    nbnodes : int
        Target number of pores.
    distributions_props : list of tuple
        List of (properties_dict, fraction) tuples describing the radius distributions
        (see ``generate_radii`` / ``distribution`` in Stats).
    target_porosity : float
        Target pore-body porosity (void volume / box volume).
    fit_box : bool
        Selects the adaptation strategy (see above).
    verbose : bool
        Print progress information.
    mindist : str or float, optional
        Minimum surface-to-surface distance between pores ("auto" = 0.25 * min_radius), used
        in the "box too small" guard. Default is "auto".

    Returns
    -------
    tuple
        ``(radii, scaled_extent, original_extent)`` where ``radii`` is a numpy array of the
        generated radii sorted from largest to smallest, ``scaled_extent = extent * length``
        and ``original_extent`` is the input extent before scaling.

    Notes
    -----
    The "box too small" guard checks that the two largest pores can both be placed in the box.
    :func:`random_packing_opt` has two defects in this guard (left unfixed there):

    - it tested ``2.1 * radii[:2].sum() <= L_min``, i.e. the first two *generated* (unsorted)
      radii instead of the two largest, with a conservative "side by side along one axis"
      condition; for heavy-tailed distributions this makes the guard a near no-op that passes
      on almost every draw even when the actual two largest pores cannot both fit;
    - when it does trigger, ``nbnodes *= 1.5`` produces a float and the next iteration crashes
      in ``radii[0:nbnodes]`` (TypeError).

    Here the nearly necessary-and-sufficient opposite-corner condition is used on the actual
    two largest radii:

        sqrt(3) * (L_min - r0 - r1) >= r0 + r1 + gap

    i.e. the two pores placed at opposite corners of the box are far enough apart, and the pore
    count stays an integer when it is increased.
    """
    extent = np.array(extent, dtype=float)

    if fit_box:
        length = 1.0
        radii = generate_radii(distributions_props, nbnodes)
        void_volume = (np.pi * (4.0 / 3.0) * radii**3).sum()
        temp_porosity = void_volume / np.prod(extent)
        counter = 0
        while abs(temp_porosity - target_porosity) > 0.0005 and counter < 5000:
            counter += 1
            nbnodes = int(np.around(nbnodes * target_porosity / temp_porosity))
            radii = generate_radii(distributions_props, nbnodes)[0:nbnodes]
            void_volume = (np.pi * (4.0 / 3.0) * radii**3).sum()
            temp_porosity = void_volume / np.prod(extent)

        if verbose:
            print(
                "{} pores will be generated to fit the target porosity of {}% in the given box".format(
                    nbnodes, 100 * temp_porosity
                )
            )
    else:
        length_OK = False
        while not length_OK:
            radii = generate_radii(distributions_props, nbnodes)
            radii = np.array(radii[0:nbnodes])
            void_volume = (np.pi * (4.0 / 3.0) * radii**3).sum()
            length = (void_volume / (target_porosity * np.prod(extent))) ** (1.0 / 3.0)

            # The box must be able to contain the two largest pores (placed at opposite
            # corners, which maximizes their center-to-center distance). Note: use the actual
            # two largest radii (random_packing_opt uses the first two generated ones, which
            # are not necessarily the largest).
            rs = np.sort(radii)[::-1]
            r0 = float(rs[0])
            r1 = float(rs[1]) if radii.size > 1 else 0.0
            gap_check = 0.25 * float(rs[-1]) if mindist == "auto" else float(mindist)
            lim = (r0 + r1) * (1.0 + 1.0 / np.sqrt(3.0)) + gap_check / np.sqrt(3.0)
            if extent.min() * length < lim:
                if verbose:
                    print(
                        "Computed min side length {} is smaller than the sum of the 2 largest pores diameter {}".format(
                            extent.min() * length, radii.max()
                        )
                    )
                    print("Increasing number of pores to :", int(nbnodes * 1.5))
                nbnodes = int(nbnodes * 1.5)  # keep an integer (original code produced a float)
            else:
                break

    radii = np.sort(radii)[::-1]  # place largest pores first
    original_extent = extent.copy()
    scaled_extent = extent * length
    return radii, scaled_extent, original_extent


def _candidate_clearances(
    pool,
    tree,
    static_centers,
    static_radii,
    recent_centers,
    recent_radii,
    large_centers,
    large_radii,
    query_dist,
    extent,
):
    """Compute for each candidate point the largest sphere radius that can be centered there.

    The "clearance" of a candidate is::

        clearance(c) = min( min_j (||c - c_j|| - r_j),  min_k distance from c to face k )

    where j runs over all already-placed pores and k over the six box faces. A candidate can
    host a pore of radius ``r`` iff ``clearance(c) >= r + gap``; the minimum surface distance
    ``gap`` is therefore NOT included here, it is applied by the caller at comparison time.

    Placed pores are handled in three groups:

    - "static": stored in the cKDTree, checked with a ball query of radius ``query_dist``;
    - "recent": added since the last tree rebuild, checked through a small temporary tree;
    - "large": pores whose radius exceeds the large-pore threshold. They are few (bounded by
      construction) but can constrain candidates at any distance, so they are checked
      exhaustively against every candidate.

    The ball-query radius must satisfy ``query_dist >= r_i + gap + max_small_radius`` for the
    current pore radius ``r_i``: pairs of "small" pores farther apart than that cannot
    influence the clearance of any remaining pore, while all "large" pores are covered by the
    exhaustive check regardless of distance.

    Parameters
    ----------
    pool : ndarray, shape (B, 3)
        Candidate center positions.
    tree : cKDTree or None
        Tree over ``static_centers`` (None when no pore has been placed yet).
    static_centers : ndarray, shape (K, 3) or None
        Centers of the pores stored in ``tree``.
    static_radii : ndarray, shape (K,) or None
        Radii of the pores stored in ``tree``.
    recent_centers : list of array-like
        Centers of the pores placed since the last tree rebuild.
    recent_radii : list of float
        Radii of those pores.
    large_centers : list of array-like
        Centers of the "large" placed pores (checked exhaustively).
    large_radii : list of float
        Radii of those pores.
    query_dist : float
        Ball-query radius for the static/recent trees (see Notes above).
    extent : ndarray, shape (3,)
        Box size [Lx, Ly, Lz].

    Returns
    -------
    ndarray, shape (B,)
        Clearance of each candidate point.
    """
    B = pool.shape[0]
    fit = np.full(B, np.inf)

    def _merge_neighbors(nbrs, centers, radii):
        nonempty = [(k, n) for k, n in enumerate(nbrs) if len(n)]
        if not nonempty:
            return
        cand_idx = np.repeat([k for k, _ in nonempty], [len(n) for _, n in nonempty])
        pore_idx = np.concatenate([np.asarray(n, dtype=np.int64) for _, n in nonempty])
        diff = pool[cand_idx] - centers[pore_idx]
        d = np.sqrt((diff * diff).sum(axis=1))
        val = d - radii[pore_idx]
        np.minimum.at(fit, cand_idx, val)

    if tree is not None:
        _merge_neighbors(tree.query_ball_point(pool, query_dist), static_centers, static_radii)

    if recent_centers:
        rtree = cKDTree(np.asarray(recent_centers))
        _merge_neighbors(rtree.query_ball_point(pool, query_dist), np.asarray(recent_centers), np.asarray(recent_radii))

    # Exhaustive check against the (few) large pores, chunked to bound memory.
    if large_centers:
        lc = np.asarray(large_centers)
        lr = np.asarray(large_radii)
        m = lc.shape[0]
        chunk = max(1, int(1_000_000 // m))  # keep B_chunk * m <= ~1e6 distance computations
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            diff = pool[s:e, None, :] - lc[None, :, :]
            d = np.sqrt((diff * diff).sum(axis=2))
            fit[s:e] = np.minimum(fit[s:e], (d - lr[None, :]).min(axis=1))

    # Box faces: a sphere of radius r centered at c needs c_k >= r and c_k <= L_k - r.
    face_clearance = np.minimum(pool, extent - pool).min(axis=1)
    fit = np.minimum(fit, face_clearance)
    return fit


def _kdtree_packing_engine(
    extent,
    nbnodes,
    distributions_props,
    target_porosity=0.5,
    mindist="auto",
    BCstep=100,
    restart=True,
    fit_box=False,
    max_unallocated_frac=0.05,
    pool_size=4096,
    tree_rebuild_interval=1024,
    max_empty_pools=20,
    verbose=True,
    seed=None,
    use_importance_sampling=False,
    voxel_resolution=96,
    graph_type="random_packing_kdtree",
    _restart_count=0,
    restart_fn=None,
):
    """Core placement loop shared by ``random_packing_kdtree`` and ``random_packing_kdtree_importance``.

    Pores are placed from largest to smallest using a pool of B random candidate points. For
    every candidate the "clearance" (largest sphere radius that fits at that point, see
    :func:`_candidate_clearances`) is computed once with vectorized cKDTree ball queries; each
    pore then picks a uniform random candidate among those whose clearance is large enough, and
    all clearances are updated in O(B) after each placement. A pool therefore serves many
    consecutive pores before being discarded, which amortizes the tree query cost over hundreds
    of placements instead of one.

    The cKDTree over placed pores is rebuilt every ``tree_rebuild_interval`` placements; pores
    added since the last rebuild ("recent") are checked through a small temporary tree, so the
    expensive O(N log N) rebuild stays amortized to ~O(log N) per placement.

    When ``use_importance_sampling`` is True (option B), candidate pools are drawn from an
    occupancy map: a voxel grid whose sampling weights are proportional to the remaining free
    volume of each voxel. This concentrates samples in free space and reduces wasted candidates
    once the box is nearly full. The map is updated by adding each placed sphere's volume to its
    center voxel (a coarse approximation; exact validity is always re-checked afterwards).

    Parameters
    ----------
    extent : array-like of float
        Initial box size [Lx, Ly, Lz].
    nbnodes : int
        Target number of pores.
    distributions_props : list of tuple
        List of (properties_dict, fraction) tuples describing the radius distributions.
    target_porosity : float, optional
        Target pore-body porosity (void volume / box volume). Default is 0.5.
    mindist : str or float, optional
        Minimum surface-to-surface distance between pores. If "auto", uses min_radius * 0.25.
    BCstep : int, optional
        Grid resolution for boundary face detection (passed to SetFaceBCnodes). Default is 100.
    restart : bool, optional
        If True, restart the whole packing when the unallocated volume fraction exceeds
        max_unallocated_frac. Default is True.
    fit_box : bool, optional
        If True, adjust the number of pores so that the target porosity is met in the provided
        extent; otherwise adapt the box size. Default is False.
    max_unallocated_frac : float, optional
        Fraction of the target pore volume above which a restart is triggered (only if
        restart=True). Default is 0.05 (5%).
    pool_size : int, optional
        Number B of candidate points per pool. Default is 4096.
    tree_rebuild_interval : int, optional
        Rebuild the main cKDTree every M placements (pores placed in between are kept in a
        small "recent" list). Default is 1024.
    max_empty_pools : int, optional
        Number of fresh pools tried for a pore before it is declared unallocated. Default is 20.
    verbose : bool, optional
        If True, print progress information. Default is True.
    seed : int or None, optional
        Seed for the pool-sampling random number generator. If given, restarts use
        deterministic child streams derived from it, so the run is fully reproducible;
        None = non-deterministic (fresh entropy per attempt). Note that radius draws always
        follow the global numpy random state (seed it with ``np.random.seed`` for full
        reproducibility).
    use_importance_sampling : bool, optional
        If True, sample candidate pools from an occupancy map instead of uniformly in the box.
    voxel_resolution : int, optional
        Number of voxels per side of the occupancy map (capped so that voxels are not smaller
        than 2 * min_radius + gap, with a floor of 16 per side). Only used when
        use_importance_sampling=True. Default is 96.
    graph_type : str, optional
        Value stored in ``graph.graph["graph_type"]``.
    _restart_count : int, optional
        Internal recursion counter for restarts (max 5). Do not set manually.
    restart_fn : callable or None, optional
        Public function used to perform a full restart (one of the two public wrappers).

    Returns
    -------
    pore_network
        A pore_network instance containing only the placed pores (no throats).
    """
    # ------------------------------------------------------------------
    # 1. Radii + box adaptation (shared pre-processing)
    # ------------------------------------------------------------------
    radii, extent_scaled, original_extent = _prepare_packing_radii(
        extent, nbnodes, distributions_props, target_porosity, fit_box, verbose, mindist=mindist
    )
    min_radius = float(radii.min())
    max_radius = float(radii.max())
    if mindist == "auto":
        gap = 0.25 * min_radius
    else:
        gap = float(mindist)

    # Seed handling: a given seed is expanded into one child stream per attempt (initial run +
    # up to 5 restarts), so that seeded runs are fully reproducible even when they restart.
    # Unseeded runs keep fresh entropy for every attempt. ``seed`` may be an int/array of ints
    # or a SeedSequence instance passed down by a parent run's restart logic.
    if seed is not None:
        ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        _child_seeds = ss.spawn(7)
        rng = np.random.default_rng(_child_seeds[0])
    else:
        _child_seeds = None
        rng = np.random.default_rng(None)

    nbnodes_in = nbnodes
    nbnodes = radii.size
    total_volume = float(np.prod(extent_scaled))

    # "Large" pores: the top ~1% of the radius distribution (bounded count). They are checked
    # exhaustively against every candidate; all other ("small") pores are covered by the ball
    # query, whose radius only needs to reach r_i + gap + large_thresh. This keeps the number
    # of tested pairs small even when a few very large pores dominate the box size.
    n_large_cap = max(16, nbnodes // 100)
    large_thresh = float(radii[n_large_cap - 1])

    if verbose:
        print("Max radius = {:.2e} and Min radius = {:.2e}".format(max_radius, min_radius))
        print("Extent = ", extent_scaled)
        print("Extent/(2*Rmax) = ", end="")
        print(["{:.2f}".format(e / (2.0 * max_radius)) for e in extent_scaled])

    # ------------------------------------------------------------------
    # 2. Optional occupancy map (importance sampling, option B)
    # ------------------------------------------------------------------
    if use_importance_sampling:
        # Cap the grid so voxels are not smaller than 2*min_radius+gap; skip the cap when the
        # minimum radius is (numerically) zero, where it would be meaningless.
        min_voxel_size = 2.0 * min_radius + gap
        if min_voxel_size > 0:
            n_vox = int(min(voxel_resolution, max(16.0, extent_scaled.min() / min_voxel_size)))
        else:
            n_vox = int(voxel_resolution)
        h = extent_scaled / n_vox  # voxel size per axis
        n_vox_total = n_vox**3
        occ = np.zeros(n_vox_total)  # occupied volume per voxel
        voxel_vol = total_volume / n_vox_total

    def _uniform_pool(B):
        """Sample B candidate points uniformly in [min_radius, L - min_radius]^3."""
        lo = min_radius
        hi = extent_scaled - min_radius
        return rng.random((B, 3)) * (hi - lo) + lo

    def _sample_pool(B):
        """Sample a pool of B candidate points (uniform or from the occupancy map)."""
        if not use_importance_sampling:
            return _uniform_pool(B)
        w = np.clip(voxel_vol - occ, 0.0, None)
        s = float(w.sum())
        if s <= voxel_vol * 1e-9:  # degenerate (should not happen): fall back to uniform
            return _uniform_pool(B)
        idx = rng.choice(n_vox_total, size=B, replace=True, p=w / s)
        gx, gy, gz = np.unravel_index(idx, (n_vox, n_vox, n_vox))
        pos = np.empty((B, 3))
        pos[:, 0] = (gx + rng.random(B)) * h[0]
        pos[:, 1] = (gy + rng.random(B)) * h[1]
        pos[:, 2] = (gz + rng.random(B)) * h[2]
        return np.clip(pos, min_radius, extent_scaled - min_radius)

    # ------------------------------------------------------------------
    # 3. Placement state
    # ------------------------------------------------------------------
    net = pore_network()
    net.graph.clear()
    net.graph.graph["extent"] = list(extent_scaled)
    net.graph.graph["graph_type"] = graph_type

    static_centers = None  # (K, 3), frozen between rebuilds
    static_radii = None  # (K,)
    tree = None  # cKDTree over static_centers
    recent_centers = []  # placed since last rebuild
    recent_radii = []
    large_centers = []  # placed pores with r > large_thresh (checked exhaustively)
    large_radii = []

    placed_centers = []
    placed_radii = []
    void_volume = 0.0
    unallocated_volume = 0.0
    n_unallocated = 0
    pools_used = 0

    def _place(pos, r):
        """Record a placed pore and update the occupancy structures."""
        nonlocal tree, static_centers, static_radii, void_volume
        pos = np.array(pos)  # independent copy (pos may be a view into the pool array)
        placed_centers.append(pos)
        placed_radii.append(r)
        recent_centers.append(pos)
        recent_radii.append(r)
        if r > large_thresh:
            large_centers.append(pos)
            large_radii.append(r)
        void_volume += (4.0 / 3.0) * np.pi * r**3

        if use_importance_sampling:
            v = np.floor(np.asarray(pos) / h).astype(int)
            np.clip(v, 0, n_vox - 1, out=v)
            occ[v[0] * n_vox * n_vox + v[1] * n_vox + v[2]] += (4.0 / 3.0) * np.pi * r**3

        if len(recent_centers) >= tree_rebuild_interval:
            if static_centers is None:
                static_centers = np.asarray(recent_centers)
                static_radii = np.asarray(recent_radii)
            else:
                static_centers = np.vstack([static_centers, np.asarray(recent_centers)])
                static_radii = np.concatenate([static_radii, np.asarray(recent_radii)])
            tree = cKDTree(static_centers)
            recent_centers.clear()
            recent_radii.clear()

    def _maybe_restart():
        """Restart the whole packing if unallocated volume exceeds the threshold.

        Returns the restarted pore_network when a restart was performed, otherwise None.
        """
        if not (restart and unallocated_volume > max_unallocated_frac * target_porosity * total_volume):
            return None
        if _restart_count >= 5:
            if verbose:
                print("\n!!!--- Max restart count reached, stopping ---!!!\n")
            return None
        if verbose:
            print("\n!!!--- Restarting ---!!!\n")
        return restart_fn(
            extent=original_extent,
            nbnodes=nbnodes_in,
            distributions_props=distributions_props,
            target_porosity=target_porosity,
            mindist=mindist,
            BCstep=BCstep,
            restart=restart,
            fit_box=fit_box,
            max_unallocated_frac=max_unallocated_frac,
            pool_size=pool_size,
            tree_rebuild_interval=tree_rebuild_interval,
            max_empty_pools=max_empty_pools,
            verbose=verbose,
            # A restart must not reproduce the same failure: unseeded runs get fresh entropy;
            # seeded runs use the next deterministic child stream (see _child_seeds above), so
            # a seeded run stays fully reproducible even when it restarts.
            seed=None if _child_seeds is None else _child_seeds[_restart_count + 1],
            _restart_count=_restart_count + 1,
        )

    def _clearances_of(pool):
        """Clearance of every candidate in ``pool`` w.r.t. current occupancy and box faces."""
        # Small pores (r_j <= large_thresh) only matter if d < r_i + gap + r_j for some
        # remaining radius r_f <= r_i, which is covered by the bound below; large pores are
        # checked exhaustively inside _candidate_clearances regardless of distance.
        query_dist = radii[i] + gap + large_thresh
        return _candidate_clearances(
            pool,
            tree,
            static_centers,
            static_radii,
            recent_centers,
            recent_radii,
            large_centers,
            large_radii,
            query_dist,
            extent_scaled,
        )

    # ------------------------------------------------------------------
    # 4. Main placement loop (largest first)
    # ------------------------------------------------------------------
    i = 0
    pool = None
    fit = None

    if verbose:
        print("Placed pores | current radius | pools drawn | current porosity | unallocated pores")
    progress_step = max(1, nbnodes // 100)  # ~100 real-time updates per run, whatever N

    while i < nbnodes:
        if pool is None:
            B = min(pool_size, nbnodes - i)
            pool = _sample_pool(B)
            fit = _clearances_of(pool)
            pools_used += 1

        r = radii[i]
        elig = np.flatnonzero(fit >= r + gap)

        if elig.size == 0:
            # No candidate in the current pool fits pore i. Try up to max_empty_pools fresh
            # pools; each failed one is recycled for smaller pores afterwards (never wasted).
            room_found = False
            for _ in range(max_empty_pools):
                B2 = min(pool_size, nbnodes - i)
                p2 = _sample_pool(B2)
                f2 = _clearances_of(p2)
                pools_used += 1
                if (f2 >= r + gap).any():
                    pool, fit = p2, f2
                    room_found = True
                    break
                pool, fit = p2, f2  # recycle: may still serve smaller pores

            if not room_found:
                unallocated_volume += (4.0 / 3.0) * np.pi * r**3
                n_unallocated += 1
                i += 1
                restarted = _maybe_restart()
                if restarted is not None:
                    return restarted
                continue

            elig = np.flatnonzero(fit >= r + gap)

        pick = int(elig[rng.integers(elig.size)])
        pos = pool[pick]
        _place(pos, r)
        i += 1

        # Update clearances of all remaining candidates w.r.t. the newly placed pore.
        d = np.sqrt(((pool - pos) ** 2).sum(axis=1))
        fit = np.minimum(fit, d - r)
        # (the used candidate gets d = 0 -> clearance -r < 0 and is never eligible again)

        if verbose and i % progress_step == 0:
            print(
                " {:14} {:16.2e} {:16} {:16.2f} % {:19}   ".format(
                    len(placed_centers), r, pools_used, 100.0 * void_volume / total_volume, n_unallocated
                ),
                end="\r",
                flush=True,
            )

        # Drop the pool when it can no longer serve any remaining pore.
        if i < nbnodes and fit.max() < radii[i] + gap:
            pool = None

    # ------------------------------------------------------------------
    # 5. Build the pore_network (pores only, no throats)
    # ------------------------------------------------------------------
    if verbose:
        print("\n{} pores generated".format(len(placed_centers)))

    labels = np.arange(len(placed_centers))
    net.graph.add_nodes_from(labels)
    nx.set_node_attributes(net.graph, dict(zip(labels, placed_centers)), "center")
    nx.set_node_attributes(net.graph, dict(zip(labels, placed_radii)), "radius")
    nx.set_node_attributes(net.graph, False, "periodic")

    pnm.SetFaceBCnodes(net, BCstep)

    if verbose:
        print("Porosity:", void_volume / total_volume)
        print(
            "Unallocated volume: {:.2e}, corresponding void_fraction: {:.2f} %".format(
                unallocated_volume, 100.0 * unallocated_volume / total_volume
            )
        )

    return net


def random_packing_kdtree(
    extent,
    nbnodes,
    distributions_props,
    target_porosity=0.5,
    mindist="auto",
    BCstep=100,
    restart=True,
    fit_box=False,
    max_unallocated_frac=0.05,
    pool_size=4096,
    tree_rebuild_interval=1024,
    max_empty_pools=20,
    verbose=True,
    seed=None,
    _restart_count=0,
):
    """Spherical pores packing using candidate pools + cKDTree (vectorized batch rejection).

    Place spherical pores following an arbitrary radius distribution, ordered from largest to
    smallest. Instead of testing one random candidate at a time in pure Python, this routine
    works with **pools** of B random candidate points:

    1. A pool of B candidates is drawn uniformly in the box and the "clearance" of every
       candidate (largest sphere radius that fits there) is computed at once with vectorized
       cKDTree ball queries plus an O(B) boundary check.
    2. Pores are then placed one by one: each pore picks a uniform random candidate among those
       whose clearance is large enough, and all clearances are updated in O(B). A pool serves
       many consecutive pores before being discarded, so the tree query cost is amortized over
       hundreds of placements.

    The cKDTree over placed pores is rebuilt every ``tree_rebuild_interval`` placements; pores
    added since the last rebuild are checked through a small temporary tree, keeping the
    amortized per-placement cost close to O(log N).

    Parameters
    ----------
    extent : array-like of float
        Initial box size [Lx, Ly, Lz].
    nbnodes : int
        Target number of pores.
    distributions_props : list of tuple
        List of (properties_dict, fraction) tuples describing the radius distributions
        (see generate_radii / distribution in Stats).
    target_porosity : float, optional
        Target pore-body porosity (void volume / box volume). Default is 0.5.
    mindist : str or float, optional
        Minimum surface-to-surface distance between pores. If "auto", uses min_radius * 0.25.
        Default is "auto".
    BCstep : int, optional
        Grid resolution for boundary face detection (passed to SetFaceBCnodes). Default is 100.
    restart : bool, optional
        If True, restart the whole packing when the unallocated volume fraction exceeds
        max_unallocated_frac. Default is True.
    fit_box : bool, optional
        If True, adjust the number of pores so that the target porosity is met in the provided
        extent; otherwise adapt the box size. Default is False.
    max_unallocated_frac : float, optional
        Fraction of the target pore volume above which a restart is triggered (only if
        restart=True). Default is 0.05 (5%).
    pool_size : int, optional
        Number B of candidate points per pool. Larger pools amortize the tree query cost over
        more placements but use more memory. Default is 4096.
    tree_rebuild_interval : int, optional
        Rebuild the main cKDTree every M placements (pores placed in between are kept in a small
        "recent" list). Default is 1024.
    max_empty_pools : int, optional
        Number of fresh pools tried for a pore before it is declared unallocated. Default is 20.
    verbose : bool, optional
        If True, print progress information. Default is True.
    seed : int or None, optional
        Seed for the pool-sampling random number generator. If given, restarts use
        deterministic child streams derived from it, so the run is fully reproducible;
        None = non-deterministic (fresh entropy per attempt). Note that radius draws always
        follow the global numpy random state (seed it with ``np.random.seed`` for full
        reproducibility).

    Returns
    -------
    pore_network
        A pore_network instance containing only the placed pores (no throats).
        Graph attribute "graph_type" is set to "random_packing_kdtree".

    Notes
    -----
    The placement rule (uniform random candidate among all valid positions, largest first) is
    the standard random-sequential one, so the resulting ensemble has the same character as a
    pore-by-pore rejection-sampling packing. Typical speedup over brute-force or cell-list
    approaches: 10x or more for networks with tens of thousands of pores.
    """
    return _kdtree_packing_engine(
        extent=extent,
        nbnodes=nbnodes,
        distributions_props=distributions_props,
        target_porosity=target_porosity,
        mindist=mindist,
        BCstep=BCstep,
        restart=restart,
        fit_box=fit_box,
        max_unallocated_frac=max_unallocated_frac,
        pool_size=pool_size,
        tree_rebuild_interval=tree_rebuild_interval,
        max_empty_pools=max_empty_pools,
        verbose=verbose,
        seed=seed,
        use_importance_sampling=False,
        voxel_resolution=None,
        graph_type="random_packing_kdtree",
        _restart_count=_restart_count,
        restart_fn=random_packing_kdtree,
    )


def random_packing_kdtree_importance(
    extent,
    nbnodes,
    distributions_props,
    target_porosity=0.5,
    mindist="auto",
    BCstep=100,
    restart=True,
    fit_box=False,
    max_unallocated_frac=0.05,
    pool_size=4096,
    tree_rebuild_interval=1024,
    max_empty_pools=20,
    verbose=True,
    seed=None,
    voxel_resolution=96,
    _restart_count=0,
):
    """Spherical pores packing using candidate pools + cKDTree + occupancy-map importance sampling.

    Same algorithm as :func:`random_packing_kdtree`, except that candidate pools are not drawn
    uniformly in the box: an **occupancy map** (voxel grid) tracks the occupied volume of each
    voxel, and new candidates are sampled with a probability proportional to the remaining free
    volume of their voxel. This concentrates samples in free space and drastically reduces
    wasted candidates once the box is nearly full, which is the regime where uniform rejection
    sampling slows down the most (dense packing, small pores placed last).

    The map is updated by adding each placed sphere's volume to its center voxel: a coarse
    approximation that only biases the *sampling* weights. Exact validity of every placement is
    always re-checked against all placed pores and the box faces, so the output is guaranteed
    non-overlapping regardless of the map resolution.

    Parameters
    ----------
    extent : array-like of float
        Initial box size [Lx, Ly, Lz].
    nbnodes : int
        Target number of pores.
    distributions_props : list of tuple
        List of (properties_dict, fraction) tuples describing the radius distributions
        (see generate_radii / distribution in Stats).
    target_porosity : float, optional
        Target pore-body porosity (void volume / box volume). Default is 0.5.
    mindist : str or float, optional
        Minimum surface-to-surface distance between pores. If "auto", uses min_radius * 0.25.
        Default is "auto".
    BCstep : int, optional
        Grid resolution for boundary face detection (passed to SetFaceBCnodes). Default is 100.
    restart : bool, optional
        If True, restart the whole packing when the unallocated volume fraction exceeds
        max_unallocated_frac. Default is True.
    fit_box : bool, optional
        If True, adjust the number of pores so that the target porosity is met in the provided
        extent; otherwise adapt the box size. Default is False.
    max_unallocated_frac : float, optional
        Fraction of the target pore volume above which a restart is triggered (only if
        restart=True). Default is 0.05 (5%).
    pool_size : int, optional
        Number B of candidate points per pool. Default is 4096.
    tree_rebuild_interval : int, optional
        Rebuild the main cKDTree every M placements. Default is 1024.
    max_empty_pools : int, optional
        Number of fresh pools tried for a pore before it is declared unallocated. Default is 20.
    verbose : bool, optional
        If True, print progress information. Default is True.
    seed : int or None, optional
        Seed for the pool-sampling random number generator. If given, restarts use
        deterministic child streams derived from it, so the run is fully reproducible;
        None = non-deterministic (fresh entropy per attempt). Note that radius draws always
        follow the global numpy random state (seed it with ``np.random.seed`` for full
        reproducibility).
    voxel_resolution : int, optional
        Number of voxels per side of the occupancy map. It is capped so that voxels are not
        smaller than 2 * min_radius + gap (finer would be useless) and floored at 16.
        Default is 96.

    Returns
    -------
    pore_network
        A pore_network instance containing only the placed pores (no throats).
        Graph attribute "graph_type" is set to "random_packing_kdtree_importance".

    Notes
    -----
    Compared to :func:`random_packing_kdtree`, this variant trades a small amount of extra work
    per pool (voxel sampling + map update) for fewer wasted candidates. The gain grows with the
    target porosity and with the number of pores; at low porosity both variants perform
    similarly.
    """
    return _kdtree_packing_engine(
        extent=extent,
        nbnodes=nbnodes,
        distributions_props=distributions_props,
        target_porosity=target_porosity,
        mindist=mindist,
        BCstep=BCstep,
        restart=restart,
        fit_box=fit_box,
        max_unallocated_frac=max_unallocated_frac,
        pool_size=pool_size,
        tree_rebuild_interval=tree_rebuild_interval,
        max_empty_pools=max_empty_pools,
        verbose=verbose,
        seed=seed,
        use_importance_sampling=True,
        voxel_resolution=voxel_resolution,
        graph_type="random_packing_kdtree_importance",
        _restart_count=_restart_count,
        restart_fn=random_packing_kdtree_importance,
    )
