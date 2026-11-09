# coding=utf-8

import networkx as nx
import numpy as np
from warnings import warn
from heapq import heappush, heappop, heapify
from sklearn.neighbors import KDTree
from pnm import pore_network
from .utils import nearest_neighbor, isinside, isinside_surface
from .Stats import distribution

EPS = 1.00000001


def invasion_percolation(pn, target_porosity, face="x-", box=None, verbose=False, keep_connected=True,
                         articulation_recompute_interval=100):
    """Delete pores using an invasion percolation process.

    Pores are removed starting from a given face, ordered by radius (smallest
    first), until the porosity reaches the target value. Articulation points
    can be preserved to keep the network connected.

    Parameters
    ----------
    pn : pore_network
        The pore network model instance.
    target_porosity : float
        The invasion stops when the porosity reaches this value.
    face : str, optional
        Boundary face where the invasion begins (e.g. "x-", "y+", "z-").
        Default is "x-".
    box : list of float or None, optional
        Subvolume [xmin, xmax, ymin, ymax, zmin, zmax] to restrict the invasion.
        If None, the whole network is used. Default is None.
    verbose : bool, optional
        If True, print progress information. Default is False.
    keep_connected : bool, optional
        If True, articulation points are not removed, keeping the network
        connected. Default is True.
    articulation_recompute_interval : int, optional
        Recompute articulation points every N removed pores (instead of every
        pore). Setting this to a large value (or 0 to disable recomputation)
        greatly speeds up the algorithm for large networks, at the cost of
        occasionally removing a pore that has become an articulation point
        since the last recomputation. Default is 100.

    Notes
    -----
    Requires node attributes ``radius``, ``center`` and ``volume``, and graph
    attributes ``bbox`` and ``extent`` to be set (e.g. after compute_geometry).
    The input ``box`` is not modified (a copy is made internally).

    Performance notes
    -----------------
    - Articulation points are recomputed every ``articulation_recompute_interval``
      removed pores instead of every pore, reducing complexity from O(V^2) to
      O(V^2 / interval) for that step.
    - Pores are marked for removal (numpy boolean array) instead of being
      deleted from the graph one by one; all removed pores are deleted in a
      single batch at the end.
    """

    pn.graph = nx.convert_node_labels_to_integers(pn.graph)

    # centers
    c = np.array(list(zip(*nx.get_node_attributes(pn.graph, "center").values()))).T
    # radii
    radii = np.array(list(nx.get_node_attributes(pn.graph, "radius").values()))
    # volumes
    volumes = np.array(list(nx.get_node_attributes(pn.graph, "volume").values()))

    node_list = list(pn.graph.nodes())
    n_nodes = len(node_list)
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    if box is not None:
        # Work on a copy to avoid mutating the caller's list
        box = list(box)
        bbox = pn.graph.graph.get("bbox")
        if bbox is None:
            warn("bbox is not set. Cannot restrict invasion to a box.")
            return
        box[0] = max(box[0], bbox[0])
        box[1] = max(box[1], bbox[1])
        box[2] = max(box[2], bbox[2])
        box[3] = min(box[3], bbox[3])
        box[4] = min(box[4], bbox[4])
        box[5] = min(box[5], bbox[5])
        active_pores = np.where(
            (c[..., 0] >= box[0])
            & (c[..., 0] <= box[1])
            & (c[..., 1] >= box[2])
            & (c[..., 1] <= box[3])
            & (c[..., 2] >= box[4])
            & (c[..., 2] <= box[5])
        )[0]

        subgraph = pn.graph.subgraph(active_pores)

        vv_ini = volumes[active_pores].sum()
        for n1, n2 in subgraph.edges():
            vv_ini += subgraph[n1][n2]["volume"]
        vbox = (box[1] - box[0]) * (box[3] - box[2]) * (box[5] - box[4])
        pini = vv_ini / vbox

    else:
        active_pores = np.array(pn.graph.nodes())

        subgraph = pn.graph
        vv_ini = volumes.sum()
        for n1, n2 in subgraph.edges():
            vv_ini += subgraph[n1][n2]["volume"]

        extent = pn.graph.graph.get("extent")
        if extent is None:
            warn("Extent is not set. Cannot compute initial porosity.")
            return
        vbox = np.prod(np.array(extent))
        pini = vv_ini / vbox

    print("Box volume = {}, volume = {}".format(vbox, vv_ini))
    print("Number of pores in box:", len(active_pores))

    if pini <= target_porosity:
        print("Initial porosity = {} <= target porosity. Nothing to do".format(pini))
        return

    # nodes on the starting face
    starting_face_nodes = pn.get_pores_by_category(category=face, mode="match_one", nodes=active_pores)
    if len(starting_face_nodes) <= 0:
        print("Pores on face {} are not inside the provided box. Quit".format(face))
        return

    # Store potential pores to invade. Ordered by radius [(r,ind)]
    n0 = starting_face_nodes[np.argmin(radii[starting_face_nodes])]

    # Use numpy arrays for marking (faster than dict lookups)
    removed = np.zeros(n_nodes, dtype=bool)  # pores removed from the network
    marked = np.zeros(n_nodes, dtype=bool)   # pores discovered (added to heap)

    stored_nodes = [(radii[n0], n0)]
    marked[node_to_idx[n0]] = True
    invaded = []
    heapify(stored_nodes)
    pf = pini
    vv = vv_ini

    art = set()
    pores_removed_count = 0
    recompute = articulation_recompute_interval > 0

    while pf > target_porosity and stored_nodes:
        # Recompute articulation points periodically (not every iteration).
        # This is the main speedup: nx.articulation_points is O(V + E) per call.
        if keep_connected and recompute and pores_removed_count % articulation_recompute_interval == 0:
            art = set(nx.articulation_points(pn.graph))

        current_node = heappop(stored_nodes)[1]
        current_idx = node_to_idx[current_node]

        # Skip if already removed (duplicate heap entry)
        if removed[current_idx]:
            continue

        if keep_connected and current_node in art:
            continue

        # Mark as removed (physical deletion from graph happens in batch at the end)
        removed[current_idx] = True
        invaded.append(current_node)

        # Update volume and heap queue
        vol = volumes[current_idx]
        for neighbor in subgraph.neighbors(current_node):
            neighbor_idx = node_to_idx[neighbor]
            if not removed[neighbor_idx]:
                vol += pn.graph[current_node][neighbor]["volume"]
                if not marked[neighbor_idx]:
                    marked[neighbor_idx] = True
                    heappush(stored_nodes, (subgraph[current_node][neighbor]["radius"], neighbor))
        vv -= vol
        pf = vv / vbox

        pores_removed_count += 1

        if verbose:
            print("Pore {} deleted (total {}), new box porosity = {} ".format(
                current_node, len(invaded), pf))

    # Remove all invaded pores from the graph in one batch
    if invaded:
        pn.graph.remove_nodes_from(invaded)

    pn.compute_geometry()
    print(
        "{} pores removed."
        "\n {} remaining."
        "\n final porosity of the whole network: {} "
        "\nPorosity of the box: {}".format(len(invaded), pn.graph.number_of_nodes(), pn.graph.graph["porosity"], pf)
    )


def assemble_pnm(pn1, pn2, axis="x", ratio=1, minc=3, maxc=np.inf, BCgridsize=100, PRTRR=1):
    """Merge two pore networks into a single pore_network instance.

    The coordinates of pn2 are shifted along the given axis so that pn2 starts
    where pn1 ends (offset = pn1 extent / bbox in that direction). Boundary
    nodes are recomputed and throats are added between the two networks at the
    interface.

    Parameters
    ----------
    pn1 : pore_network
        First pore network (upstream).
    pn2 : pore_network
        Second pore network (downstream), whose coordinates are shifted.
    axis : str, optional
        Assembly axis: "x", "y" or "z". Default is "x".
    ratio : float, optional
        Pore surface coverage ratio passed to add_throats_by_surf.
        Default is 1.
    minc : int, optional
        Minimum connectivity for interface throats. Default is 3.
    maxc : int or float, optional
        Maximum connectivity for interface throats. Default is np.inf.
    BCgridsize : int, optional
        Grid resolution for boundary face detection in SetFaceBCnodes.
        Default is 100.
    PRTRR : float, optional
        Pore radius to throat radius ratio for interface throats.
        Default is 1.

    Returns
    -------
    pore_network
        The assembled network.

    Notes
    -----
    Requires graph attributes ``graph_type``, ``bbox`` and ``spacing`` on pn1,
    and ``bbox`` on pn2. Falls back gracefully if some are missing.
    """

    i = {"x": 0, "y": 1, "z": 2}[axis]

    graph_type1 = pn1.graph.graph.get("graph_type", "generic")
    if graph_type1 == "cubic_regular":
        centers = np.array(list(zip(*nx.get_node_attributes(pn1.graph, "center").values()))).T
        spacing = pn1.graph.graph.get("spacing")
        if spacing is not None:
            offset = centers[:, i].max() + spacing[i] * 0.5
        else:
            offset = centers[:, i].max()
    else:
        bbox1 = pn1.graph.graph.get("bbox")
        if bbox1 is None:
            warn("pn1 bbox is not set. Cannot compute offset.")
            return pn1
        offset = bbox1[i + 3]

    c = np.array(list(zip(*nx.get_node_attributes(pn2.graph, "center").values()))).T
    bbox2 = pn2.graph.graph.get("bbox")
    if bbox2 is None:
        warn("pn2 bbox is not set. Cannot assemble networks.")
        return pn1
    c[:, i] += offset - bbox2[i]
    nx.set_node_attributes(pn2.graph, dict(zip(list(pn2.graph.nodes), list(c))), name="center")

    newgraph = nx.disjoint_union(pn1.graph, pn2.graph)

    newpnm = pore_network(newgraph)

    # Get nodes to connect
    cat1 = {"x": "x+", "y": "y+", "z": "z+"}
    cat2 = {"x": "x-", "y": "y-", "z": "z-"}

    nodes_to_connect1 = pn1.get_pores_by_category(cat1[axis])
    nodes_to_connect2 = np.array(pn2.get_pores_by_category(cat2[axis])) + pn1.graph.number_of_nodes()

    # Recompute BC nodes
    newpnm.clear_category()
    SetFaceBCnodes(newpnm, step=BCgridsize)

    if pn1.graph.graph["graph_type"] == "cubic_regular" or pn2.graph.graph["graph_type"] == "cubic_regular":
        connect_pores(newpnm, nodes_to_connect1, nodes_to_connect2)

    else:
        nodes_to_connect1.extend(nodes_to_connect2)
        # Removing existing throats from old boundary nodes
        for n in nodes_to_connect1:
            ebunch = list(newpnm.graph.edges(n))
            newpnm.graph.remove_edges_from(ebunch)

        # Adding new throats to connect the 2 networks
        add_throats_by_surf(
            newpnm, nodes=nodes_to_connect1, ratio=ratio, minc=minc, maxc=maxc, sort_by_radius=False, PRTRR=PRTRR
        )

    # Update geometry
    newpnm.compute_geometry(autothroats=True)

    return newpnm


def connect_pores(pn, nbunch1, nbunch2):
    """Connect each pore in nbunch1 to its nearest pore in nbunch2.

    Uses a nearest-neighbor search (via numba-accelerated euclidean distance)
    to find, for each pore in nbunch1, the closest pore in nbunch2, and adds
    a throat between them.

    Parameters
    ----------
    pn : pore_network
        The pore network model instance.
    nbunch1 : iterable of int
        Source pore node indices.
    nbunch2 : iterable of int
        Target pore node indices.

    Notes
    -----
    Edge attribute ``radius`` is set on each new throat (auto-computed as
    min(r(n1), r(n2))). Throat length is NOT set here.
    Requires ``center`` attribute on all nodes in nbunch1 and nbunch2.
    """

    centers = nx.get_node_attributes(pn.graph, "center")
    centers1 = np.array([np.array(centers[i]) for i in nbunch1])
    centers2 = np.array([np.array(centers[i]) for i in nbunch2])

    indices = nearest_neighbor(centers1, centers2)

    for i in range(indices.shape[0]):
        pn.add_throat(nbunch1[i], nbunch2[indices[i]])


def add_throats(net, lowc=0, highc=np.inf, minc=0, maxc=np.inf, mode="auto", sort_by_radius=True, spacing="auto"):
    """Set throats between n-th nearest pore neighbors.

    Connects pores to their nearest neighbors up to a target connectivity.
    Pores are processed from largest to smallest radius. In "auto" mode,
    lowc and highc are scaled by the pore radius relative to spacing.

    Parameters
    ----------
    net : pore_network
        The pore network model instance (must have pores with radius and center set).
        It is assumed that there are no existing throats in the network.
    lowc : int, optional
        Minimum target connectivity per pore. Default is 0.
    highc : int or float, optional
        Maximum connectivity per pore. Default is np.inf.
    minc : int, optional
        Minimum enforced connectivity per pore. Default is 0.
    maxc : int or float, optional
        Maximum enforced connectivity per pore. Default is np.inf.
    mode : str, optional
        Either "auto" (scale lowc/highc by pore radius/spacing) or any other
        value (use fixed lowc/highc). Default is "auto".
    sort_by_radius : bool, optional
        If True, neighbors are sorted by radius instead of distance, favoring
        connections between large pores. Default is True.
    spacing : str or float, optional
        Characteristic spacing for "auto" mode scaling. If "auto", uses
        net.graph.graph["spacing"].min() if available, otherwise the mean pore
        radius. Default is "auto".

    Notes
    -----
    Edge attribute ``radius`` is set to min(r(n1), r(n2)) on each new throat.
    Throat length is NOT set here (use pore_network.set_auto_throats_length).
    """

    if highc > maxc:
        highc = maxc

    if lowc > highc:
        lowc = highc

    if minc is None:
        minc = lowc

    elif minc > lowc:
        minc = lowc

    nodelist = list(net.graph.nodes)
    pores_radii = np.array(list(nx.get_node_attributes(net.graph, "radius").values()))

    max_pore_radius = pores_radii.max()

    if spacing == "auto":
        try:
            spacing = net.graph.graph["spacing"].min()
        except KeyError:
            # Arithmetic mean of pore radii
            spacing = pores_radii.sum() / pores_radii.size

    if len(nodelist) <= 0:
        return
    centers = nx.get_node_attributes(net.graph, "center")
    centers = np.array([np.array(centers[i]) for i in nodelist])
    kdtree = KDTree(centers, leaf_size=40)

    # Pores with degree < minc need additional connections
    candidates = [(n, net.graph.nodes[n]["radius"]) for n in nodelist if net.graph.degree(n) < minc]
    # Process larger pores first
    candidates.sort(key=lambda t: t[1], reverse=True)
    candidates = [t[0] for t in candidates]

    for n1 in candidates:
        if mode == "auto":
            m = net.graph.nodes[n1]["radius"] / spacing
            high2 = min(int(np.around(max(highc, highc * m))), maxc)
            low2 = min(int(np.around(max(lowc, lowc * m))), maxc)
        else:
            high2 = highc
            low2 = lowc

        # Search for neighbors within (max_pore_radius + r1) * 1.5
        indices, distances = kdtree.query_radius(
            [net.graph.nodes[n1]["center"]],
            (max_pore_radius + net.graph.nodes[n1]["radius"]) * 1.5,
            return_distance=True,
        )
        if len(indices[0]) < low2 * 2:
            # Otherwise, take the high2*2 nearest neighbors
            distances, indices = kdtree.query(
                [net.graph.nodes[n1]["center"]], k=min(high2 * 2, len(nodelist)), return_distance=True
            )

        indices = indices[0][: len(centers)]
        degree = np.random.randint(low2, high2 + 1) - net.graph.degree(n1)

        if degree <= 0 or len(indices) <= 0:
            continue

        # Convert center-to-center distance to surface-to-surface distance
        distances = distances[0] - pores_radii[indices] - net.graph.nodes[n1]["radius"]
        dist_and_labels = [(distances[j], k) for j, k in enumerate(indices)]
        dist_and_labels.sort(key=lambda t: t[0])
        # Keep only the high2*2 nearest pores
        dist_and_labels = dist_and_labels[: min(high2 * 2, len(dist_and_labels))]
        dist_and_labels = [(dist, pores_radii[k], k) for dist, k in dist_and_labels]

        if sort_by_radius:  # Sort by pore size to favor connections between large pores
            dist_and_labels.sort(key=lambda t: t[1], reverse=True)
        nthroats = 0

        for dist, r, j in dist_and_labels:
            n2 = nodelist[j]

            if n2 != n1:
                if mode == "auto" and net.graph.graph.get("spacing", None) is not None:
                    m = net.graph.nodes[n2]["radius"] / spacing
                    high3 = min(int(np.around(max(highc, highc * m))), maxc)
                else:
                    high3 = highc

                if dist < 0:
                    warn(
                        "Overlapping detected ! Pore {}, R = {}  and {}, R = {}, distance = {}".format(
                            n1, net.graph.nodes[n1]["radius"], n2, net.graph.nodes[n2]["radius"], dist
                        )
                    )
                if nthroats >= degree:
                    break
                if net.graph.degree(n2) < high3:
                    nthroats += 1
                    net.graph.add_edge(
                        n1, n2, radius=min(net.graph.nodes[n1]["radius"], net.graph.nodes[n2]["radius"])
                    )

        # If connectivity is still < minc, add a capillary even if it exceeds
        # the max connectivity of the neighbor
        if net.graph.degree(n1) < minc:
            # Consider only close neighbors
            indices = kdtree.query(
                [net.graph.nodes[n1]["center"]], k=min(high2 * 2, len(nodelist)), return_distance=False
            )

            neighbors = [
                (net.graph.nodes[nodelist[n]]["radius"], n)
                for n in indices[0]
                if nodelist[n] != n1
                if net.graph.has_node(nodelist[n])
            ]
            if len(neighbors) == 0:
                break
            neighbors.sort(key=lambda t: t[0], reverse=True)
            for i in range(minc - net.graph.degree(n1)):
                n2 = nodelist[neighbors[i][1]]
                net.graph.add_edge(n1, n2, radius=min(net.graph.nodes[n1]["radius"], net.graph.nodes[n2]["radius"]))


def add_periodic_throats(
    net,
    delta=0.01,
    faces=("x", "y", "z"),
    PRTRR=2,
    minc=0,
    maxc=np.inf,
    search_radius_factor=2,
    min_neighbors_factor=3,
):
    """Add throats between opposed BC pores - must be called AFTER the function add_throats_by_surf

    Connects pores on opposite boundary faces (e.g. "x" connects "x-" and "x+")
    with periodic throats, enforcing surface coverage and connectivity constraints.

    Parameters
    ----------
    net : pore_network
        The pore network model instance. Must have been processed by
        add_throats_by_surf (so that exchange_area and surface_ratio exist)
        and by SetFaceBCnodes/compute_geometry (so that extent and categories exist).
    delta : float, optional
        Minimum throat length factor: min_length = min(r1, r2) * delta.
        Default is 0.01.
    faces : tuple of str, optional
        Faces to connect. Each entry is an axis name ("x", "y", or "z");
        the corresponding "-"/"+" faces are connected.
        Default is ("x", "y", "z").
    PRTRR : float, optional
        Pore radius to throat radius ratio. Throat radius = min(r1, r2) / PRTRR.
        Default is 2.
    minc : int, optional
        Minimum enforced connectivity per pore. Default is 0.
    maxc : int or float, optional
        Maximum connectivity per pore. Default is np.inf.
    search_radius_factor : float, optional
        Neighbors are searched within R * search_radius_factor in pass 1.
        Default is 2.
    min_neighbors_factor : int, optional
        Factor applied to minc when searching for neighbors.
        Default is 3.

    Notes
    -----
    Edge attributes ``radius``, ``length``, ``periodic`` and ``pdir`` are set
    on each new throat. Node attribute ``exchange_area`` is updated.
    Requires node attribute ``surface_ratio`` (set by add_throats_by_surf).
    """

    if maxc < minc:
        maxc = minc + 1

    extent = net.graph.graph.get("extent")
    if extent is None:
        warn("Extent is not set. Cannot add periodic throats.")
        return

    def min_length(r1, r2):
        return min(r1, r2) * delta

    # TODO: revoir coorected_ratio
    def corrected_ratio(n, face_id):
        innercat = set(["inner", "none"])
        k = len(set(net[n]["category"]) - innercat) # x+, y+, z+ etc. : attention pour les réseaux 2D on peut avoir un pore z+ ET z- ce qui est 1 pb pour la suite...
        if len(innercat.intersection(net[n]["category"])) > 0:
            return 1
        else:
            return (face_id + 1) * (1 - 0.5**k) / k + 0.5**k

    for face_id, face in enumerate(faces):
        delta_c = np.zeros(3)
        delta_c[face_id] = extent[face_id]

        poreset1 = net.get_pores_by_category(face + "-", mode="match_one")
        poreset2 = net.get_pores_by_category(face + "+", mode="match_one")

        if len(poreset1) == 0:
            print(f"No nodes found on face {face}-")
            continue
        if len(poreset2) == 0:
            print(f"No nodes found on face {face}+")
            continue

        candidates1 = [(n, net.graph.nodes[n]["radius"]) for n in poreset1 if net.graph.degree(n) < maxc]
        candidates2 = [(n, net.graph.nodes[n]["radius"]) for n in poreset2 if net.graph.degree(n) < maxc]

        # On traite les plus gros pores en premier
        candidates1.sort(key=lambda t: t[1], reverse=True)
        candidates2.sort(key=lambda t: t[1], reverse=True)
        candidates = [[t[0] for t in candidates1], [t[0] for t in candidates2]]

        centers = [np.array([np.array(net.graph.nodes[n]["center"]) + delta_c for n in candidates[0]])]
        centers.append(np.array([np.array(net.graph.nodes[n]["center"]) for n in candidates[1]]))

        radii = [np.array([net.graph.nodes[n]["radius"] for n in candidates[0]])]
        radii.append(np.array([net.graph.nodes[n]["radius"] for n in candidates[1]]))

        max_radius = np.array([rad.max() for rad in radii]).max()

        kdtree = [KDTree(centers[0], leaf_size=40)]
        kdtree.append(KDTree(centers[1], leaf_size=40))

        # print(f"Connecting {len(candidates[0])} nodes on face {face}- to {len(candidates[1])} nodes on face {face}+")

        PP_counter = 0  # Periodic Pore counter

        for i, face_candidates in enumerate(candidates):
            opposed_face_index = (i + 1) % 2

            for counter, n1 in enumerate(face_candidates):
                # print(f'node {n1} degree {net.graph.degree(n1)},'
                #       f'exchange surface {net.graph.nodes[n1]["exchange_area"]}, total surface {net.graph.nodes[n1]["radius"] * 4 * np.pi}')
                print("{:5.2f}% completed  ".format(100 * (counter + 1) / len(face_candidates)), end="\r")

                # Contrainte de connectivité max
                if net.graph.degree(n1) >= maxc:
                    continue

                r1 = net.graph.nodes[n1]["radius"]
                S1max = 4 * np.pi * r1**2

                # Contrainte de surface d'échange max
                if net.graph.nodes[n1]["exchange_area"] >= S1max:
                    continue

                # On recherche d'abord les premiers voisins dans une sphere de raton Ri*search_radius_factor
                indices, distances = kdtree[opposed_face_index].query_radius(
                    [net.graph.nodes[n1]["center"]],
                    net.graph.nodes[n1]["radius"] * 2 + max_radius,
                    return_distance=True,
                )
                # Dans le cas où il y a peu de voisins proches (cas des petits pores) on cherche au moins min_neighbors_factor*minc pores voisins
                # A noter que les petits pores ont déjà de bonnes chances d'être déjà connectés aux plus gros (on itère par ordre de rayon croissant)
                if len(indices[0]) < min_neighbors_factor * minc:
                    distances, indices = kdtree[opposed_face_index].query(
                        [net.graph.nodes[n1]["center"]],
                        k=min(minc * min_neighbors_factor, len(candidates[opposed_face_index])),
                        return_distance=True,
                    )

                indices = indices[0]
                # On cherche les pores les plus proches (i.e. la distance entre les deux surfaces la plus faible !)
                distances = distances[0] - radii[opposed_face_index][indices] - net.graph.nodes[n1]["radius"]
                dist_and_labels = [(distances[j], k) for j, k in enumerate(indices)]
                dist_and_labels.sort(key=lambda t: t[0])
                dist_and_labels = [(dist, radii[opposed_face_index][k], k) for dist, k in dist_and_labels]
                temp_dist_and_labels = [
                    (dist, r, k) for dist, r, k in dist_and_labels if dist < r1 * search_radius_factor
                ]
                if len(temp_dist_and_labels) < minc:
                    dist_and_labels = dist_and_labels[: min(min_neighbors_factor * minc, len(dist_and_labels))]
                else:
                    dist_and_labels = temp_dist_and_labels

                for dist, _, j in dist_and_labels:
                    n2 = candidates[opposed_face_index][j]

                    if n2 != n1:
                        r2 = net.graph.nodes[n2]["radius"]
                        S2max = 4 * np.pi * r2**2

                        if dist < min_length(r1, r2):
                            warn(
                                "Overlapping detected ! Pore {}, R = {}  and {}, R = {}, distance = {}".format(
                                    n1, net.graph.nodes[n1]["radius"], n2, net.graph.nodes[n2]["radius"], dist
                                )
                            )
                            continue

                        if net.graph.degree(n2) > maxc or net.graph.degree(n1) > maxc:
                            continue

                        r12 = min(r1, r2) / PRTRR
                        S12 = np.pi * r12**2
                        S1 = net.graph.nodes[n1]["exchange_area"] + S12
                        S2 = net.graph.nodes[n2]["exchange_area"] + S12
                        sr1 = net.graph.nodes[n1].get("surface_ratio", 1.0)
                        sr2 = net.graph.nodes[n2].get("surface_ratio", 1.0)

                        if (
                            S1 <= corrected_ratio(n1, face_id) * S1max * sr1
                            and S2 <= corrected_ratio(n2, face_id) * S2max * sr2
                        ):
                            vtemp = centers[i][counter] - centers[opposed_face_index][j]
                            net.add_throat(
                                n1, n2, radius=r12, length=np.sqrt(np.dot(vtemp, vtemp)), periodic=True, pdir=face
                            )
                            net.graph.nodes[n1]["exchange_area"] += S12
                            net.graph.nodes[n2]["exchange_area"] += S12
                            PP_counter += 1

                        # Force a min connectivity even if covered area > max area ???
                        # elif S1 > corrected_ratio(n1) * S1max and net.graph.degree(n1) < minc:
                        # if S2 <= corrected_ratio(n2) * S2max:
                        #     vtemp = centers[i][counter] - centers[opposed_face_index][j]
                        #     net.add_throat(n1, n2, radius=r12, length=np.sqrt(np.dot(vtemp, vtemp)), periodic=True)
                        #     net.graph.nodes[n1]["exchange_area"] += S12
                        #     net.graph.nodes[n2]["exchange_area"] += S12
                        # else:
                        #     continue

                        elif S1 > corrected_ratio(n1, face_id) * sr1 * S1max and net.graph.degree(n1) >= minc:
                            break

        print(f"Creating {PP_counter} periodic links between face {face}- and face {face}+")


def add_throats_by_surf(
    net,
    nodes=None,
    ratio=0.75,
    PRTRR=2,
    minc=0,
    maxc=100,
    sort_by_radius=False,
    border_correction=True,
    search_radius_factor=3,
    min_neighbors_factor=3,

):
    # TODO: allow PRTRR distribution
    """Set throats between nearest pore neighbors such that pore surface coverage stays below a ratio.

    Throats are added iteratively between pores, prioritizing larger pores first.
    The pore surface covered by connected throats is limited so that
    sum(throat_sections) / pore_surface <= ratio * corrected_ratio.

    Two passes are performed:
      - Pass 0: connect up to minc nearest neighbors per pore.
      - Pass 1: connect additional neighbors within search_radius_factor * pore radius.

    Parameters
    ----------
    net : pore_network
        The pore network model instance (must have pores with radius and center set).
    nodes : iterable of int or None, optional
        List of node indices to process. If None, all nodes are processed.
    ratio : float or tuple of float, optional
        Maximum fraction of pore surface covered by throats (default 0.75).
        If a 4-tuple (loc, low, high, sigma), the ratio is drawn from a truncated
        normal distribution with these parameters. Default is 0.75.
    PRTRR : float, optional
        Pore radius to throat radius ratio. Throat radius = min(r1, r2) / PRTRR.
        Default is 2.
    minc : int, optional
        Minimum enforced connectivity per pore. Default is 0.
    maxc : int, optional
        Maximum connectivity per pore. Default is 100.
    sort_by_radius : bool, optional
        If True, neighbors are sorted by radius instead of distance (pass 0 only).
        Default is False.
    border_correction : bool, optional
        If True, reduce the allowed surface coverage for boundary pores
        (50% per adjacent face). Default is True.
    search_radius_factor : float, optional
        In pass 1, only neighbors within R * search_radius_factor are considered.
        Default is 3.
    min_neighbors_factor : int, optional
        Factor applied to minc when searching for neighbors (pass 0).
        Default is 3.

    Notes
    -----
    Node attributes ``exchange_area`` and ``surface_ratio`` are set/updated by this function.
    Edge attribute ``radius`` is set on each new throat.
    """

    if maxc < minc:
        maxc = minc + 1

    nodelist = list(net.graph.nodes)

    if len(nodelist) <= 0:
        return

    if nodes is None:
        nodes = nodelist

    pores_radii = np.array(list(nx.get_node_attributes(net.graph, "radius").values()))

    nx.set_node_attributes(net.graph, 0, name="exchange_area")
    # mean_pore_radius = pores_radii.mean()
    max_pore_radius = pores_radii.max()

    centers = nx.get_node_attributes(net.graph, "center")
    centers = np.array([np.array(centers[i]) for i in nodelist])

    kdtree = KDTree(centers, leaf_size=40)

    candidates = [(n, net.graph.nodes[n]["radius"]) for n in nodes if net.graph.degree(n) < maxc]
    # On traite les plus gros pores en premier
    candidates.sort(key=lambda t: t[1], reverse=True)
    candidates = [t[0] for t in candidates]

    # If ratio is a list, then the max pore surface covered by throat is defined as a random distribution
    try:
        loc, low, high, sigma = ratio
        surface_ratio = distribution(
            net.graph.number_of_nodes(),
            name="normal",
            high=high,
            low=low,
            loc=loc,
            scale=sigma,
        )
    except (TypeError, ValueError):
        surface_ratio = np.zeros(net.graph.number_of_nodes()) + ratio

    # Map surface_ratio values to actual node labels (nodes may not be 0..N-1)
    surface_ratio_dict = dict(zip(nodelist, surface_ratio))
    nx.set_node_attributes(net.graph, surface_ratio_dict, name="surface_ratio")

    if border_correction:
        # Reduce the number of capillaries at boundary pores
        def corrected_ratio(n):
            innercat = set(["inner", "none"])
            if len(innercat.intersection(net[n]["category"])) > 0:
                return 1.0
            else:
                # Reduce by 50% per adjacent face (50% for one face, 25% for an edge, 12.5% for a corner)
                return 0.5 ** len(set(net[n]["category"]) - innercat)

    else:
        def corrected_ratio(n):
            return 1.0

    print(len(candidates), "nodes to process")

    # nombre de passes. Lors de la première on ne connecte que les minc premiers voisins
    niter = 2
    climit = [minc, maxc]
    counter = -1
    for n in range(niter):
        for counter, n1 in enumerate(candidates):
            print(f"Iteration {n:d}:  {100 * (counter + 1) / len(candidates):5.2f}% completed  ", end="\r")

            if net.graph.degree(n1) >= max(1, climit[n] * corrected_ratio(n1)):
                continue

            r1 = net.graph.nodes[n1]["radius"]
            S1max = 4 * np.pi * r1**2

            if net.graph.nodes[n1]["exchange_area"] >= S1max:
                continue

            # query_radius() returns (indices, distances), query() returns (distances, indices).
            # Pass 0: try to connect minc capillaries. Iterate over min_neighbors_factor * minc nearest neighbors.
            # Pass 1: connect other pores within R * search_radius_factor.
            # Note: KDTree returns center-to-center distance, not surface-to-surface distance.
            # We must cast wide and then sort/filter results afterwards.
            if n == 0:
                distances, indices = kdtree.query(
                    [net.graph.nodes[n1]["center"]],
                    k=min(minc * min_neighbors_factor, len(nodelist)),
                    return_distance=True,
                )
            else:
                indices, distances = kdtree.query_radius(
                    [net.graph.nodes[n1]["center"]],
                    2*net.graph.nodes[n1]["radius"] + max_pore_radius,
                    return_distance=True,
                )

            indices = indices[0]

            # Search for nearest pores (i.e. smallest surface-to-surface distance)
            distances = distances[0] - pores_radii[indices] - net.graph.nodes[n1]["radius"]
            dist_and_labels = [(distances[j], k) for j, k in enumerate(indices)]
            dist_and_labels.sort(key=lambda t: t[0])
            dist_and_labels = [(dist, pores_radii[k], k) for dist, k in dist_and_labels]

            if n > 0:
                # Distance constraint based on pore radius
                temp_dist_and_labels = [
                    (dist, r, k) for dist, r, k in dist_and_labels if dist < r1 * search_radius_factor
                ]
                if len(temp_dist_and_labels) < min_neighbors_factor * minc:
                    dist_and_labels = dist_and_labels[: min(min_neighbors_factor * minc, len(dist_and_labels))]


            if sort_by_radius and n == 0:  # Sort by pore size to favor connections between large pores (pass 0 only)
                dist_and_labels.sort(key=lambda t: t[1], reverse=True)

            for dist, r2, j in dist_and_labels:
                n2 = nodelist[j]

                if n2 != n1:
                    if dist < 0:
                        warn(
                            "Overlapping detected ! Pore {}, R = {}  and {}, R = {}, distance = {}".format(
                                n1, net.graph.nodes[n1]["radius"], n2, net.graph.nodes[n2]["radius"], dist
                            )
                        )

                    # Stop if connectivity limit is exceeded
                    if (
                        net.graph.degree(n1) >= max(1, climit[n] * corrected_ratio(n1))
                        or net.graph.degree(n2)  >= max(1, climit[n] * corrected_ratio(n2))
                    ):
                        continue

                    r2 = net.graph.nodes[n2]["radius"]
                    r12 = min(r1, r2) / PRTRR
                    S12 = np.pi * r12**2
                    S1 = net.graph.nodes[n1]["exchange_area"] + S12
                    S2 = net.graph.nodes[n2]["exchange_area"] + S12
                    S2max = 4 * np.pi * r2**2

                    if S1 <= corrected_ratio(n1) * surface_ratio_dict[n1] * S1max and S2 <= corrected_ratio(n2) * surface_ratio_dict[n2] * S2max:
                        net.add_throat(n1, n2, radius=r12)
                        net.graph.nodes[n1]["exchange_area"] += S12
                        net.graph.nodes[n2]["exchange_area"] += S12

                    # Force a min connectivity even if covered area > max area
                    # elif S1 > corrected_ratio(n1) * 4 * np.pi * r1**2 and net.graph.degree(n1) < minc:
                    #     if S2 <= corrected_ratio(n2) * 4 * np.pi * r2**2:
                    #         net.add_throat(n1, n2, radius=r12)
                    #         net.graph.nodes[n1]["exchange_area"] += S12
                    #         net.graph.nodes[n2]["exchange_area"] += S12
                    #     else:
                    #         continue

                    elif S1 > corrected_ratio(n1) * surface_ratio_dict[n1] * 4 * np.pi * r1**2 and net.graph.degree(n1) >= minc:
                        break

            # If some pores still have connectivity < minc, force a connection
            if net.graph.degree(n1) < minc:
                for dist, r, j in dist_and_labels:
                    n2 = nodelist[j]

                    if n2 != n1 and not net.graph.has_edge(n1, n2):
                        r1 = net.graph.nodes[n1]["radius"]
                        r2 = net.graph.nodes[n2]["radius"]
                        r12 = min(r1, r2) / PRTRR
                        S12 = np.pi * r12**2
                        net.add_throat(n1, n2, radius=r12)
                        net.graph.nodes[n1]["exchange_area"] += S12
                        net.graph.nodes[n2]["exchange_area"] += S12
                        break
        print(f"Iteration {n:d}:  {100 * (counter + 1) / len(candidates):5.2f}% completed  ", end="\r")


def scale(net, factor):
    """Scale the network dimensions, centers, radii, throat lengths, etc. by a factor.

    All geometric attributes (pore centers and radii, throat radii and lengths)
    are multiplied by ``factor``. Graph attributes ``extent`` and ``inner_extent``
    are also scaled. Geometry is recomputed afterwards.

    Parameters
    ----------
    net : pore_network
        The pore network model instance to scale.
    factor : float
        Scaling factor (must be > 0).

    Raises
    ------
    ValueError
        If ``factor`` is not positive.
    KeyError
        If ``extent`` or ``inner_extent`` graph attributes are not set.

    Notes
    -----
    Edge attributes ``radius`` and ``length`` are scaled. Node attributes
    ``center`` and ``radius`` are scaled.
    """

    if factor <= 0:
        raise ValueError("Scaling factor must be positive, got {}".format(factor))

    indices = list(net.graph.nodes)

    c = np.array(list(zip(*nx.get_node_attributes(net.graph, "center").values()))).T
    c *= factor
    nx.set_node_attributes(net.graph, dict(zip(indices, list(c))), name="center")

    pores_radii = np.array(list(nx.get_node_attributes(net.graph, "radius").values()))
    pores_radii *= factor
    nx.set_node_attributes(net.graph, dict(zip(indices, list(pores_radii))), name="radius")

    throats_radii_dict = nx.get_edge_attributes(net.graph, "radius")
    throats_radii = np.array(list(throats_radii_dict.values()))
    edges_keys = list(throats_radii_dict.keys())
    throats_radii *= factor
    nx.set_edge_attributes(net.graph, dict(zip(edges_keys, list(throats_radii))), name="radius")

    throats_length_dict = nx.get_edge_attributes(net.graph, "length")
    throats_length = np.array(list(throats_length_dict.values()))
    edges_keys = list(throats_length_dict.keys())
    throats_length *= factor
    nx.set_edge_attributes(net.graph, dict(zip(edges_keys, list(throats_length))), name="length")

    extent = net.graph.graph.get("extent")
    if extent is not None:
        net.graph.graph["extent"] = list(np.array(extent) * factor)
    inner_extent = net.graph.graph.get("inner_extent")
    if inner_extent is not None:
        net.graph.graph["inner_extent"] = list(np.array(inner_extent) * factor)
    net.compute_geometry()


def aggregates_nodes_if_too_big(pn, nodes=None, only_smaller=True, maxradius=None):
    """Merge overlapping pores with the larger one.

    For each pore in ``nodes`` (or all pores if None), checks whether any
    neighboring pore overlaps (throat length <= 0 and pore distance <= maxradius).
    Overlapping pores are merged into the larger one.

    Parameters
    ----------
    pn : pore_network
        The pore network model instance.
    nodes : iterable of int or None, optional
        Pore node indices to process. If None, all pores are processed,
        starting from the largest. Default is None.
    only_smaller : bool, optional
        If True, only merge with smaller or equal-radius neighbors.
        Default is True.
    maxradius : float or None, optional
        Maximum pore radius. Pores larger than this are capped, and only
        neighbors within this distance are considered for merging.
        If None, no radius limit is applied. Default is None.

    Notes
    -----
    Requires node attributes ``radius`` and ``center``. Uses
    ``pore_network.merge_pores`` to merge overlapping pores.
    """

    iterate = True

    if maxradius is None:
        maxradius = float("inf")

    while iterate:
        iterate = False

        if nodes is None:
            rdict = list(nx.get_node_attributes(pn.graph, "radius").items())
            # Process larger pores first
            rdict.sort(key=lambda tup: tup[1], reverse=True)
            l, _ = zip(*rdict)

        else:
            l = nodes[:]

        for n1 in l:
            if pn.graph.has_node(n1):
                if pn.graph.nodes[n1]["radius"] > maxradius:
                    pn.graph.nodes[n1]["radius"] = maxradius

                stop = False

                while not stop:
                    stop = True
                    neighbors = [n for n in pn.graph[n1]]

                    for n2 in neighbors:
                        if (only_smaller and pn.graph.nodes[n1]["radius"] >= pn.graph.nodes[n2]["radius"]) or (
                            not only_smaller
                        ):
                            if (
                                pn._compute_auto_throat_length(n1, n2) <= 0
                                and pn.get_pore_distance(n1, n2) <= maxradius
                            ):
                                pn.merge_pores(n1, n2)
                                stop = False
                                iterate = True


def SetFaceBCnodes(net, step=100):
    """Automatically detect boundary pores and mark them with category labels.

    Pores closest to the boundary faces are tagged with side labels ('x+', 'x-',
    'y+', 'y-', 'z+', 'z-'). A node can have several categories. Distance is
    computed from the face to the pore center (not the pore surface), so large
    pores are less likely to be marked as boundary pores.

    Each face is divided into (step * step) cells. For each cell, the nearest
    node is marked as a BC node. There will be at most (step * step) BC nodes
    per face.

    Parameters
    ----------
    net : pore_network
        The pore network model instance.
    step : int, optional
        Number of grid cells per side of each face. Default is 100.

    Notes
    -----
    Requires graph attribute ``extent`` to be set (e.g. after compute_geometry).
    Sets graph attribute ``inner_extent`` to the extent value.
    Node attribute ``category`` is updated for all nodes.
    """

    nodes_indices = list(net.graph.nodes)
    centers = nx.get_node_attributes(net.graph, "center")
    centers = np.array([np.array(centers[i]) for i in nodes_indices])
    kdtree = KDTree(centers, leaf_size=40)

    net.clear_category()

    extent = net.graph.graph.get("extent")
    if extent is None:
        warn("Extent is not set. Cannot set boundary nodes.")
        return
    # TODO: consider using the average distance of each face instead
    net.graph.graph["inner_extent"] = extent

    x = np.linspace(0, extent[0], step)
    y = np.linspace(0, extent[1], step)
    z = np.linspace(0, extent[2], step)

    X, Y = np.meshgrid(x, y)
    face_Z0 = zip(X.ravel(), Y.ravel(), np.zeros(step * step))
    face_Z1 = zip(X.ravel(), Y.ravel(), np.zeros(step * step) + extent[2])

    X, Z = np.meshgrid(x, z)
    face_Y0 = zip(X.ravel(), np.zeros(step * step), Z.ravel())
    face_Y1 = zip(X.ravel(), np.zeros(step * step) + extent[1], Z.ravel())

    Y, Z = np.meshgrid(y, z)
    face_X0 = zip(np.zeros(step * step), Y.ravel(), Z.ravel())
    face_X1 = zip(np.zeros(step * step) + extent[0], Y.ravel(), Z.ravel())

    for coords in face_Z0:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]["category"].add("z-")
    for coords in face_Z1:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]["category"].add("z+")
    for coords in face_Y0:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]["category"].add("y-")
    for coords in face_Y1:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]["category"].add("y+")
    for coords in face_X0:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]["category"].add("x-")
    for coords in face_X1:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]["category"].add("x+")

    for n in net.graph.nodes:
        if len(net.graph.nodes[n]["category"]) == 0:
            net.graph.nodes[n]["category"].add("inner")


def SetBCnodes(net, thickness, mode="center"):
    """Set node categories based on distance from the boundary.

    Pores within a layer of given thickness from the boundary are tagged with
    side labels ('x+', 'x-', 'y+', 'y-', 'z+', 'z-') or 'inner'. Two methods
    are available: checking the pore center or the pore surface against the
    boundary layer.

    Parameters
    ----------
    net : pore_network
        The pore network model instance.
    thickness : float
        Thickness of the boundary layer.
    mode : str, optional
        Either "center" (pore center in the layer) or "surface" (pore surface
        touches the layer). Default is "center".

    Notes
    -----
    Requires graph attribute ``extent`` to be set.
    Sets graph attribute ``inner_extent`` to extent - 2 * thickness.
    Node attribute ``category`` is replaced for all nodes.
    Uses ``isinside`` (center mode) or ``isinside_surface`` (surface mode)
    from pnm.utils.
    """

    extent = net.graph.graph.get("extent")
    if extent is None:
        warn("Extent is not set. Cannot set boundary nodes.")
        return
    net.graph.graph["inner_extent"] = list(np.array(extent) - 2 * thickness)

    for n in net.graph.nodes:
        if mode == "center":
            net.graph.nodes[n]["category"] = isinside(
                net.graph.nodes[n]["center"], net.graph.nodes[n]["radius"], extent, thickness
            )
        elif mode == "surface":
            net.graph.nodes[n]["category"] = isinside_surface(
                net.graph.nodes[n]["center"], net.graph.nodes[n]["radius"], extent, thickness
            )
