# coding=utf-8
import numpy as np
import networkx as nx
from pnm import pore_network

import pnm
from .Stats import distribution
from .utils import *

# from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree  # A priori bcp plus rapide avec sklearn que le cKDTree de scipy
import multiprocessing
from functools import partial


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
    min_neighbors_factor=2
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
    BCgrid size: In order to assign the boundary conditions, the volume is discretised using a grid containing  BCgridsize cells per side
    sratio: maximal pore surface area over total intersect area of connected throats ratio (default 1, which is not optimal)
    TODO: use a sratio distribution instead of a constant value
    """

    pn = pnm.random_packing_opt(
        extent=extent,
        nbnodes=npores,
        distributions_props=[({"func": rdist}, 1)],
        target_porosity=porosity,
        mindist="auto",
        nb_trials=2000,
        BCstep=BCgridsize,
        fit_box=fit_box,
    )

    pnm.add_throats_by_surf(
        pn,
        ratio=sratio,
        PRTRR=PRTRR,
        minc=minc,
        maxc=maxc,
        sort_by_radius=False,
        border_correction=True,
        search_radius_factor=search_radius_factor,
        min_neighbors_factor=min_neighbors_factor
    )

    if add_periodic:
        pnm.add_periodic_throats(
            pn,
            delta=0.01,
            faces=periodic_faces,
            ratio=sratio,
            PRTRR=PRTRR,
            minc=minc,
            maxc=maxc,
            search_radius_factor=search_radius_factor,
            min_neighbors_factor=min_neighbors_factor
        )

    pn.compute_geometry(autothroats=False)

    return pn


def deprec_random_packing(
    extent,
    nbnodes,
    distributions_props,
    target_porosity=0.5,
    mindist="auto",
    boundary_thickness="auto",
    nb_trials=1000,
):
    """
    Sphere packing following an arbitrary distribution using brute force approach (-> slow for large number of pores...)
    Use random_packing_opt instead
    """

    # On génère les rayons

    length_OK = False
    extent = np.array(extent)

    while not length_OK:
        radii = []

        for props, probs in distributions_props:
            dist = props.get("func", distribution)
            r = list(dist(n_samples=int(np.ceil(probs * nbnodes)), **props))
            radii.extend(r)

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
    avg_radius = radii.sum() / radii.size

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
    occupied_categories = []
    unallocated_radii = []

    total_volume = np.prod(extent)
    void_volume = 0

    maxit = nbnodes * 10

    if mindist == "auto":
        gap = min_radius * 0.5
    else:
        gap = mindist

    pore_added = True
    it = 0

    if boundary_thickness == "auto":
        epsilon = avg_radius
    else:
        epsilon = boundary_thickness

    net.graph.graph["inner_extent"] = list(np.array(extent - 2 * epsilon))

    for i, _ in enumerate(radii):
        if len(occupied_centers) > 0:
            # Remise à jour du kdtree seuleument si le nombre de pores a changé
            if pore_added:
                kdtree = KDTree(occupied_centers, leaf_size=40)
        r = radii[i]

        pore_added = False

        for j in range(nb_trials):
            # coordonnée aléatoire
            if i == 0:  # First pore near the boundary
                coords = r + np.random.rand(3) * np.minimum(1.2 * np.array([r, r, r]), extent - r)
            # elif i == 1:    # Second pore
            # coords = coords + radii[i-1] + np.random.rand(3) * (extent - r)
            # else:       #Test d'overlapping sur les premiers gros pores afin de d'avoir une coord qui a plus de probablité d'être dans une zone libre. Limite le recours au kdtree
            # c = r + np.random.rand(3) * (extent - r)
            # r2 = (r+radii[0:min(i,20)])**2      #On teste les 20 premiers pores...
            # while np.any(((c - coords)**2).sum() < r2):
            # c = r + np.random.rand(3) * (extent - r)
            # coords = c
            else:
                coords = r + np.random.rand(3) * (extent - r)

            if len(occupied_centers) > 0:
                # labels = kdtree.query_ball_point(coords,(max_radius + r)*1.05, n_jobs= 1)
                labels, distances = kdtree.query_radius([coords], (max_radius + r) * 1.05 + gap, return_distance=True)

                if len(labels[0]) == 0:
                    distances = np.array([np.inf])
                    # _, labels =  kdtree.query([coords],k=1,n_jobs=1)
                    # distances, labels = kdtree.query([coords],k = 1,return_distance = True)
                    # try:
                    # _ = iter(labels)
                    # except TypeError:
                    # labels = [labels]
                # labels = np.array(labels)
                # distances = pairwise_distances(np.array(occupied_centers)[labels],np.array([coords]),n_jobs=4)[:,0]*0.998 - np.array(occupied_radii)[labels]
                # distances = cdist(np.array(occupied_centers)[labels],[coords])[:,0]*0.998 - np.array(occupied_radii)[labels]
                # if len(labels) > 5000: #Multiprocessing slows down the computation...
                # with multiprocessing.Pool(processes=4) as pool:
                # distances = pool.map(partial(eucl_dist,np.array([coords])), np.array(occupied_centers)[labels]) - np.array(occupied_radii)[labels]
                # else:
                # distances = 0.998 * np.einsum('ij,ij->i', [coords]-np.array(occupied_centers)[labels], [coords]-np.array(occupied_centers)[labels])**0.5 - np.array(occupied_radii)[labels]
                else:
                    distances = distances[0] * 0.998 - np.array(occupied_radii)[labels[0]]
            else:
                distances = np.array([np.inf])

            if distances.min() > r + gap:
                category = isinside(coords, r, extent, epsilon)

                if category is not None:
                    occupied_centers.append(coords)
                    occupied_radii.append(r)
                    occupied_categories.append(category)
                    void_volume += np.pi * (4 / 3) * r**3
                    pore_added = True
                    # if i*100 % len(radii) == 0:
                    # print ("#")
                    print(
                        "Pore {} with radius {:.2e} added after {} trials. Current porosity: {:.2f} % - unaloccated pores: {}   ".format(
                            len(occupied_centers), r, j + 1, 100 * void_volume / total_volume, len(unallocated_radii)
                        ),
                        end="\r",
                    )
                    # del(radii[j])
                    break
            # else:
            # todelete.append(j)

        # for n in todelete:
        # del(coords_dict[n])

        if not pore_added:
            unallocated_radii.append(i)

    print("\n{} pores generated".format(len(occupied_centers)))

    labels = np.arange(len(occupied_centers))
    net.graph.add_nodes_from(labels)
    nx.set_node_attributes(net.graph, dict(zip(labels, occupied_centers)), "center")
    nx.set_node_attributes(net.graph, dict(zip(labels, occupied_radii)), "radius")
    nx.set_node_attributes(net.graph, dict(zip(labels, occupied_categories)), "category")

    print("Porosity:", void_volume / total_volume)
    unallocated_volume = (np.pi * (4 / 3) * radii[np.array(unallocated_radii, dtype=np.int)] ** 3).sum()
    print(
        "Unallocated volume: {:.2e}, corresponding void_fraction: {:.2f} %".format(
            unallocated_volume, 100 * unallocated_volume / total_volume
        )
    )

    return net


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
):
    """
    Spherical pores packing following an arbitrary distribution using brute force approach (-> slow for large number of pores or compact packing...)
    extent: initial extent [Lx, Ly, Lz]
    nbnodes: number of pores
    distributions_props: radius distribution list with probability
    target_porosity: tageted pore-body porosity [not the total porosity]
    mindist: minimal distance between two pore surface. If auto mindist = min (R1, R2) * 0.5
    nb_trials: number of trials per pore
    BCStep:  To detect boundary pores, we use a discrete grid with BCSteps to dicretize the volume
    restart : if True, the algorithm restart if a pore cannot be placed
    fit_box : if True, the number of pores is modified in order to fit into the provided extent (default False)

    \nNote that this function returns only pores. Throats have to be added later"""

    length_OK = False
    extent = np.array(extent)

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
        gap = min_radius * 0.5
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
                print("\n!!!--- Restarting ---!!!\n")
                return random_packing_opt(
                    extent, nbnodes, distributions_props, target_porosity, mindist, nb_trials, restart
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


def deprec_random_packing_scipy(
    extent,
    nbnodes,
    distributions_props,
    target_porosity=0.5,
    mindist="auto",
    boundary_thickness="auto",
    nb_trials=1000,
    mp_nbpoints=10000,
):
    """Sphere packing following an arbitrary distribution using brute force approach (-> slow for large number of pores...)
    Use random_packing_opt instead"""

    # On génère les rayons

    length_OK = False
    extent = np.array(extent)

    while not length_OK:
        radii = []

        for props, probs in distributions_props:
            dist = props.get("func", distribution)
            r = list(dist(n_samples=int(np.ceil(probs * nbnodes)), **props))
            radii.extend(r)

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
    avg_radius = radii.sum() / radii.size

    extent = extent * length

    print("Max radius = {:.2e} and Min radius = {:.2e}".format(max_radius, min_radius))
    print("Extent = ", extent)
    print("Extent/(2*Rmax) = ", end="")
    print(["{:.2f}".format(e / (2 * max_radius)) for e in extent])

    net = pore_network()
    net.graph.clear()
    net.graph.graph["extent"] = list(extent)
    net.graph.graph["inner_extent"] = list(extent)
    net.graph.graph["graph_type"] = "random_packing"

    radii.sort()
    radii = radii[::-1]  # On place les plus gros pores d'abord

    occupied_centers = []
    occupied_radii = []
    occupied_categories = []
    unallocated_radii = []

    total_volume = np.prod(extent)
    void_volume = 0

    maxit = nbnodes * 10

    if mindist == "auto":
        gap = min_radius * 0.5
    else:
        gap = mindist

    pore_added = True
    it = 0

    if boundary_thickness == "auto":
        epsilon = avg_radius
    else:
        epsilon = boundary_thickness

    for i, _ in enumerate(radii):
        if len(occupied_centers) > 0:
            # Remise à jour du kdtree seuleument si le nombre de pores a changé
            if pore_added:
                kdtree = cKDTree(occupied_centers, leafsize=40, compact_nodes=True, balanced_tree=False)
        r = radii[i]

        pore_added = False

        for j in range(nb_trials):
            # coordonnée aléatoire
            if i == 0:  # First pore near the boundary
                coords = r + np.random.rand(3) * np.minimum(1.2 * np.array([r, r, r]), extent - r)
            else:
                coords = r + np.random.rand(3) * (extent - r)

            if len(occupied_centers) > 0:
                labels = kdtree.query_ball_point(coords, (max_radius + r) * 1.05 + gap, n_jobs=1)
                # labels, distances = kdtree.query_radius([coords], (max_radius + r)*1.05,return_distance =True)

                if len(labels) == 0:
                    distances = np.array([np.inf])
                    # distances, labels = kdtree.query([coords],k = 1,return_distance = True)
                    # try:
                    # _ = iter(labels)
                    # except TypeError:
                    # labels = [labels]
                # labels = np.array(labels)
                # distances = cdist(np.array(occupied_centers)[labels],[coords])[:,0]*0.998 - np.array(occupied_radii)[labels]

                # Multiprocessing slows down the computation...
                elif len(labels) > mp_nbpoints:
                    with multiprocessing.Pool(processes=4) as pool:
                        distances = (
                            0.998
                            * pool.map(partial(eucl_dist, np.array([coords])), np.array(occupied_centers)[labels])
                            - np.array(occupied_radii)[labels]
                        )
                else:
                    distances = (
                        0.998 * eucl_dist([coords], np.array(occupied_centers)[labels])
                        - np.array(occupied_radii)[labels]
                    )

            else:
                distances = np.array([np.inf])

            minimal_dist = distances.min()

            if minimal_dist > r + gap:
                category = isinside(coords, r, extent, epsilon)

                if category is not None:
                    occupied_centers.append(coords)
                    occupied_radii.append(r)
                    occupied_categories.append(category)
                    void_volume += np.pi * (4 / 3) * r**3
                    pore_added = True
                    print(
                        "Pore {} with radius {:.2e} added after {} trials. Current porosity: {:.2f} % - unaloccated pores: {}        ".format(
                            len(occupied_centers), r, j + 1, 100 * void_volume / total_volume, len(unallocated_radii)
                        ),
                        end="\r",
                    )
                    # del(radii[j])
                    break
            # else:
            # todelete.append(j)

        # for n in todelete:
        # del(coords_dict[n])

        if not pore_added:
            unallocated_radii.append(i)

    print("\n{} pores generated".format(len(occupied_centers)))

    labels = np.arange(len(occupied_centers))
    net.graph.add_nodes_from(labels)
    nx.set_node_attributes(net.graph, dict(zip(labels, occupied_centers)), "center")
    nx.set_node_attributes(net.graph, dict(zip(labels, occupied_radii)), "radius")
    nx.set_node_attributes(net.graph, dict(zip(labels, occupied_categories)), "category")

    print("Porosity:", void_volume / total_volume)
    unallocated_volume = (np.pi * (4 / 3) * radii[np.array(unallocated_radii, dtype=np.int)] ** 3).sum()
    print(
        "Unallocated volume: {:.2e}, corresponding void_fraction: {:.2f} %".format(
            unallocated_volume, 100 * unallocated_volume / total_volume
        )
    )

    return net
