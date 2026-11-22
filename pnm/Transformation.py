# coding=utf-8

import networkx as nx
import numpy as np
from warnings import warn
from heapq import heappush, heappop, heapify
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
from pnm import pore_network
from .utils import nearest_neighbor, isinside, isinside_surface
from .Stats import distribution, sratio_profile_values

EPS = 1.00000001


def invasion_percolation(
    pn, target_porosity, face="x-", box=None, verbose=False, keep_connected=True, articulation_recompute_interval=100
):
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
    marked = np.zeros(n_nodes, dtype=bool)  # pores discovered (added to heap)

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
            print("Pore {} deleted (total {}), new box porosity = {} ".format(current_node, len(invaded), pf))

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

        # Adding new throats to connect the 2 networks. The two-phase method bridges the seam by
        # nearest surface distance (phase 1 disabled); minc -> phase2_zmin floor, maxc -> zmax cap.
        add_throats_by_surf(
            newpnm, nodes=nodes_to_connect1, ratio=ratio, PRTRR=PRTRR, zmax=maxc, phase1_z=0, phase2_zmin=minc
        )

    # Update geometry: complete only the throats missing radius/length (defensive),
    # without overwriting the PRTRR-based radii of pre-existing throats.
    newpnm.complete_missing_throat_geometry()
    newpnm.compute_geometry(autothroats=False)

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
    Each new throat is created via :meth:`pore_network.add_throat` without explicit
    geometry, so both ``radius`` (auto-computed as min(r(n1), r(n2))) and ``length``
    are set automatically.
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
                    net.graph.add_edge(n1, n2, radius=min(net.graph.nodes[n1]["radius"], net.graph.nodes[n2]["radius"]))

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

    innercat = frozenset(["inner", "none"])

    def corrected_ratio(n, face_id):
        """Fraction of the full budget (sratio * 4 pi r^2) pore n may cover after this face.

        The in-box fill (add_throats_by_surf, border_correction=True) gives a pore with k side
        labels the budget 0.5**k (the other part of the pore lies "outside" the box). Each
        periodic face restores the share (1 - 0.5**k)/k of one of its side labels, so after the
        periodic faces processed so far the allowed fraction is

            0.5**k + j * (1 - 0.5**k) / k,

        j = number of the pore's side labels whose axis is among faces[:face_id + 1]. It reaches
        1 (never more) once all of the pore's faces are periodic; labels on non-periodic faces
        keep their reduced share. (The previous formula used face_id + 1 instead of j, which let
        a pore on a single y face reach 1.5 * sratio, i.e. > 100 % coverage for sratio > 2/3.)
        """
        cat = set(net.graph.nodes[n]["category"])
        if innercat & cat:
            return 1.0
        sides = cat - innercat
        k = len(sides)
        if k == 0:
            return 1.0
        processed_axes = {f[0] for f in faces[: face_id + 1]}
        j = sum(1 for s in sides if s[0] in processed_axes)
        return min(1.0, 0.5**k + j * (1.0 - 0.5**k) / k)

    for face_id, face in enumerate(faces):
        axis = "xyz".index(face)
        delta_c = np.zeros(3)
        delta_c[axis] = extent[axis]

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
                        # Never re-count an existing throat (phantom surface): both opposed
                        # faces are processed as primaries, so a pair connected from one side
                        # is re-encountered from the other; add_throat would be a no-op but
                        # the exchange_area increments below would still fire.
                        if net.graph.has_edge(n1, n2):
                            continue

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
    zmax=1024,
    order="increasing",
    border_correction=True,
    phase1_radius_factor=2,
    phase1_min_neighbors=32,
    phase1_z=1,
    phase1_mode="distance",
    phase1_neighbours=32,
    phase2_radius_factor=3,
    phase2_min_neighbors=32,
    phase2_zmin=2,
):
    """Connect pores by a surface-coverage budget in two phases.

    Phase 1 (optional) seeds each pore with its largest partners; phase 2 fills the
    remaining surface budget with nearest-by-surface-distance neighbors and enforces a
    minimum connectivity floor. Throat radius is ``min(r1, r2) / PRTRR`` and every new
    throat charges both endpoints' covered-surface budgets by its cross-section.

    Parameters
    ----------
    net : pore_network
        The pore network model instance (pores must have ``radius`` and ``center`` set;
        ``category`` is used for border correction).
    nodes : iterable of int or None, optional
        Pores processed as the "active" side. If None, all pores are processed. Neighbors may
        be any pore in the network (looked up from the full spatial tree).
    ratio : float or 4-tuple (loc, low, high, sigma), optional
        Maximum fraction of pore surface covered by throats (default 0.75). A 4-tuple draws each
        pore's ratio from a truncated normal distribution with these parameters.
    PRTRR : float, optional
        Pore radius to throat radius ratio: throat radius = min(r1, r2) / PRTRR. Default is 2.
    zmax : int, optional
        Maximum connectivity per pore (default 1024). A hard cap enforced in both phases; the
        phase-2 forced floor may push a pore's covered surface toward (but strictly below) full
        coverage without exceeding ``zmax`` on either side.
    order : {"increasing", "decreasing", "random"}, optional
        Order in which pores are processed in phase 2, by radius: "increasing" = smallest first
        (default), "decreasing" = largest first, "random" = shuffled (uses the global numpy RNG).
    border_correction : bool, optional
        If True, reduce the allowed surface coverage for boundary pores (50% per adjacent face).
        Default is True.
    phase1_radius_factor : float, optional
        Phase-1 candidate ball radius as a multiple of the active pore radius (only used with
        ``phase1_mode="distance"``): the ball has center-distance radius
        ``phase1_radius_factor * r_i + r_i`` = ``(factor + 1) * r_i``. Default is 2 (ball = 3 * r_i).
    phase1_min_neighbors : int, optional
        Minimum number of candidates in the phase-1 ball (only used with
        ``phase1_mode="distance"``); if fewer lie within it, the search is widened to the nearest
        ``phase1_min_neighbors`` pores. Default is 32.
    phase1_z : int, optional
        Number of LARGEST-radius neighbors each pore connects to in phase 1 (subject to the
        surface budget and ``zmax``). Phase 1 runs only when this is > 0. Default is 1.
    phase1_mode : {"distance", "neighbours"}, optional
        Phase-1 candidate-pool selection (only used when ``phase1_z > 0``). "distance" (default)
        queries a ball of center-distance radius ``(phase1_radius_factor + 1) * r_i``, widened to at
        least ``phase1_min_neighbors`` pores when sparse. "neighbours" uses a fixed-size pool: the
        ``phase1_neighbours`` nearest pores by center distance (the old single-pass v1 pass-0 pool,
        from which the largest partners are then drawn). Default is "distance".
    phase1_neighbours : int, optional
        Phase-1 candidate-pool size in "neighbours" mode: number of nearest pores (by center
        distance) from which the ``phase1_z`` largest partners are drawn. Default is 32. Ignored in
        "distance" mode.
    phase2_radius_factor : float, optional
        Phase-2 candidate ball radius: center-distance radius
        ``phase2_radius_factor * r_i + R_max`` where ``R_max`` is the largest pore radius in the
        network (so even a far large partner can be reached). Default is 3.
    phase2_min_neighbors : int, optional
        Minimum number of candidates in the phase-2 ball; if fewer lie within it, the search is
        widened to the nearest ``phase2_min_neighbors`` pores. Default is 32.
    phase2_zmin : int, optional
        Phase-2 connectivity floor: after the budget-driven fill, a pore with fewer than this many
        throats is force-connected to its nearest (surface-to-surface) neighbors, overriding the
        surface budget but keeping the covered fraction strictly below 1. Default is 2.

    Notes
    -----
    Node attributes ``exchange_area`` and ``surface_ratio`` are set/updated by this function.
    Each new throat gets its ``radius`` attribute; length is computed automatically, with
    periodic=False and pdir="none" (see :meth:`pore_network.add_throat`).

    Pairs that are already connected are skipped (``graph.has_edge``), so re-encountering an
    existing throat can no longer inflate the local degree / exchange mirrors ("phantom"
    increments) and wrongly saturate a pore's surface budget.
    """

    nodelist = list(net.graph.nodes)
    N = len(nodelist)
    if N <= 0:
        return

    if nodes is None:
        nodes = nodelist

    # ------------------------------------------------------------------
    # Vectorized state (arrays indexed by position in ``nodelist``).
    # ------------------------------------------------------------------
    pos_of = {n: i for i, n in enumerate(nodelist)}
    radii_all = np.array([net.graph.nodes[n]["radius"] for n in nodelist], dtype=float)
    centers = np.array([net.graph.nodes[n]["center"] for n in nodelist], dtype=float)

    nx.set_node_attributes(net.graph, 0, name="exchange_area")
    exchange = np.zeros(N)  # covered surface area per pore (local mirror of the node attribute)
    deg = np.fromiter((net.graph.degree(n) for n in nodelist), dtype=np.int64, count=N)

    max_pore_radius = float(radii_all.max())

    # Per-pore surface ratio: constant, or drawn from a truncated normal distribution.
    try:
        loc, low, high, sigma = ratio
        sr_vals = distribution(N, name="normal", high=high, low=low, loc=loc, scale=sigma)
    except (TypeError, ValueError):
        sr_vals = np.full(N, float(ratio))

    surface_ratio_dict = dict(zip(nodelist, sr_vals))
    nx.set_node_attributes(net.graph, surface_ratio_dict, name="surface_ratio")

    # Per-pore corrected ratio and the resulting covered-surface budget.
    if border_correction:
        innercat = frozenset(["inner", "none"])
        corr = np.ones(N)
        for i, n in enumerate(nodelist):
            cat = net.graph.nodes[n].get("category") or set()
            if not (innercat & cat):
                # Reduce by 50% per adjacent face (50% one face, 25% an edge, 12.5% a corner).
                corr[i] = 0.5 ** len(set(cat) - innercat)
    else:
        corr = np.ones(N)

    budget = corr * sr_vals * (4.0 * np.pi * radii_all**2)  # covered-surface budget per pore
    full_surf = 4.0 * np.pi * radii_all**2                  # total surface area per pore

    tree = cKDTree(centers)

    active = list(nodes)
    print(len(active), "nodes to process")

    def _cap(i):
        return max(1, int(zmax * corr[i]))

    # ------------------------------------------------------------------
    # Phase 1 (optional): seed each pore with its ``phase1_z`` LARGEST partners.
    # Pores are processed largest-first; the candidate pool is either a ball of radius
    # (factor + 1) * r_i widened to at least phase1_min_neighbors when sparse ("distance"), or
    # the fixed-size set of the phase1_neighbours nearest pores by center distance
    # ("neighbours"). Connections respect both the surface budget and zmax on either side.
    # ------------------------------------------------------------------
    if phase1_z > 0:
        if phase1_mode not in ("distance", "neighbours"):
            raise ValueError(f"phase1_mode must be 'distance' or 'neighbours', got {phase1_mode!r}")
        order_desc = sorted(active, key=lambda n: net.graph.nodes[n]["radius"], reverse=True)
        progress_step = max(1, len(order_desc) // 100)
        for counter, n1 in enumerate(order_desc):
            if (counter + 1) % progress_step == 0:
                print(f"Phase 1: {100 * (counter + 1) / len(order_desc):5.2f}% completed   ", end="\r", flush=True)

            i1 = pos_of[n1]
            r1 = radii_all[i1]
            if deg[i1] >= _cap(i1):
                continue

            if phase1_mode == "neighbours":
                # Fixed-size pool: the ``phase1_neighbours`` nearest pores by center distance.
                _, idx_k = tree.query(centers[i1], k=min(int(phase1_neighbours), N))
                idx = np.atleast_1d(idx_k)
            else:
                idx = np.asarray(tree.query_ball_point(centers[i1], (float(phase1_radius_factor) + 1.0) * r1))
                if len(idx) < phase1_min_neighbors:
                    _, idx_k = tree.query(centers[i1], k=min(int(phase1_min_neighbors), N))
                    idx = np.unique(np.concatenate((idx, np.atleast_1d(idx_k))))
            keep = idx != i1
            idx = idx[keep]
            if len(idx) == 0:
                continue

            # Largest partners first.
            seq = idx[np.argsort(-radii_all[idx], kind="stable")]
            added = 0
            for j in range(len(seq)):
                if added >= phase1_z or deg[i1] >= _cap(i1):
                    break
                p = int(seq[j])
                n2 = nodelist[p]
                if net.graph.has_edge(n1, n2):
                    continue
                if deg[p] >= _cap(p):
                    continue
                r2 = radii_all[p]
                r12 = min(r1, r2) / PRTRR
                S12 = np.pi * r12**2
                if exchange[i1] + S12 <= budget[i1] and exchange[p] + S12 <= budget[p]:
                    net.add_throat(n1, n2, radius=r12)
                    exchange[i1] += S12
                    exchange[p] += S12
                    deg[i1] += 1
                    deg[p] += 1
                    added += 1

        print(f"Phase 1: {100.0:5.2f}% completed", end="\r", flush=True)

    # ------------------------------------------------------------------
    # Phase 2: fill the remaining budget with nearest-by-surface-distance neighbors, then
    # enforce a ``phase2_zmin`` connectivity floor (budget override, covered fraction < 1).
    # ------------------------------------------------------------------
    if order == "decreasing":
        order2 = sorted(active, key=lambda n: net.graph.nodes[n]["radius"], reverse=True)
    elif order == "increasing":
        order2 = sorted(active, key=lambda n: net.graph.nodes[n]["radius"])
    elif order == "random":
        order2 = list(active)
        np.random.shuffle(order2)
    else:
        raise ValueError(f"order must be 'increasing', 'decreasing' or 'random', got {order!r}")

    progress_step = max(1, len(order2) // 100)
    for counter, n1 in enumerate(order2):
        if (counter + 1) % progress_step == 0:
            print(f"Phase 2: {100 * (counter + 1) / len(order2):5.2f}% completed   ", end="\r", flush=True)

        i1 = pos_of[n1]
        r1 = radii_all[i1]
        if deg[i1] >= _cap(i1):
            continue

        idx = np.asarray(tree.query_ball_point(centers[i1], float(phase2_radius_factor) * r1 + max_pore_radius))
        if len(idx) < phase2_min_neighbors:
            _, idx_k = tree.query(centers[i1], k=min(int(phase2_min_neighbors), N))
            idx = np.unique(np.concatenate((idx, np.atleast_1d(idx_k))))
        keep = idx != i1
        idx = idx[keep]
        if len(idx) == 0:
            continue

        # Nearest by surface-to-surface distance first.
        d_surf = np.linalg.norm(centers[idx] - centers[i1], axis=1) - radii_all[idx] - r1
        o = np.argsort(d_surf, kind="stable")
        seq = idx[o]

        # Budget-driven fill: stop when i1's budget is exhausted or zmax is reached.
        for j in range(len(seq)):
            if deg[i1] >= _cap(i1):
                break
            p = int(seq[j])
            n2 = nodelist[p]
            if net.graph.has_edge(n1, n2):
                continue
            r2 = radii_all[p]
            r12 = min(r1, r2) / PRTRR
            S12 = np.pi * r12**2
            # i1's own budget exhausted -> stop (nearest-first: further candidates are no closer).
            if exchange[i1] + S12 > budget[i1]:
                break
            if deg[p] >= _cap(p):
                continue
            if exchange[p] + S12 > budget[p]:
                continue
            net.add_throat(n1, n2, radius=r12)
            exchange[i1] += S12
            exchange[p] += S12
            deg[i1] += 1
            deg[p] += 1

        # Connectivity floor: force-connect nearest neighbors until phase2_zmin is reached,
        # overriding the surface budget but keeping, on BOTH sides, the degree within zmax and
        # the covered fraction strictly below 1.
        if deg[i1] < phase2_zmin:
            for j in range(len(seq)):
                if deg[i1] >= phase2_zmin:
                    break
                p = int(seq[j])
                n2 = nodelist[p]
                if net.graph.has_edge(n1, n2):
                    continue
                if deg[p] >= _cap(p):
                    continue
                r2 = radii_all[p]
                r12 = min(r1, r2) / PRTRR
                S12 = np.pi * r12**2
                # Keep the covered surface strictly below full coverage (fraction < 1).
                if exchange[i1] + S12 >= full_surf[i1] or exchange[p] + S12 >= full_surf[p]:
                    continue
                net.add_throat(n1, n2, radius=r12)
                exchange[i1] += S12
                exchange[p] += S12
                deg[i1] += 1
                deg[p] += 1

    print(f"Phase 2: {100.0:5.2f}% completed", end="\r", flush=True)

    # Write back the final covered surface areas (the graph attribute is the contract).
    nx.set_node_attributes(net.graph, dict(zip(nodelist, exchange)), name="exchange_area")


def add_throats_by_surf_v3(
    net,
    nodes=None,
    ratio=0.75,
    PRTRR=2,
    minc=2,
    large_radius_cutoff=None,
    border_correction=True,
    seed=None,
):
    """Connect pores by a surface-budget algorithm in two passes (v3).

    Pass 2 is the original minimal baseline: no search-radius factors, no radius
    sorting, no maximum connectivity. The only controls are the per-pore surface
    budget ``ratio`` and the random choice of which pore acts next. This is intended
    as a clean baseline to test whether the i.i.d.-based pore-distribution inversion
    (see :func:`pnm.Stats.invert_pore_distribution_independent`) can reproduce the
    experimental throat distribution without any ad-hoc correction pass.

    Pass 1 (optional, enabled when ``large_radius_cutoff`` is set) builds an
    *independent* sub-network of the large pores first: every pore with radius >=
    ``large_radius_cutoff`` is connected to other such pores until it has at least
    ``minc`` connections within the large class. This targets wide-range (two-peak)
    distributions, where the largest throats require BOTH endpoints to be large and
    are systematically missed by pass 2 alone: a large pore saturates on nearby small
    pores before any distant large-large pair is considered.

    Algorithm
    ---------
    Pass 1 (only when ``large_radius_cutoff`` is not None and ``minc > 0``):
      - restrict to the large class (radius >= cutoff) and build a dedicated cKDTree
        over its centers, so neighbor queries return only large partners;
      - process the large pores in DECREASING radius order (like v1/v2 phase 0);
      - for each pore below ``minc`` connections, connect up to ``minc`` nearest large
        neighbors by surface-to-surface distance subject to both endpoints' surface
        budgets; if still below ``minc``, force-connect the next unconnected large
        neighbor IGNORING the budget (the connectivity floor is guaranteed even if
        ``ratio`` is exceeded, matching v2 phase 0).

    Pass 2 (always):
       1. Every pore starts "active" with a surface budget
          ``budget_i = corr_i * ratio_i * 4*pi*r_i^2`` (``corr_i`` is the boundary
          correction, see ``border_correction``; ``ratio_i`` is drawn from a truncated
          normal when ``ratio`` is a 4-tuple, evaluated per pore radius from a profile
          dict when ``ratio`` is one, constant otherwise). Pores already charged by
          pass 1 start with their residual budget.
      2. Repeat until no active pores remain:

         - pick one active pore i uniformly at random;
         - among its k nearest neighbors by center distance (k = max(32, deg(i)+32),
           capped at N-1), find the first unconnected neighbor j in surface-to-surface
           distance order whose own budget can also absorb the throat cross-section;
         - add a throat of radius ``min(r_i, r_j)/PRTRR`` and charge both budgets with
           its cross-section ``pi * (min(r_i, r_j)/PRTRR)^2``;
         - remove from the active list every pore whose budget is now full.

         If no valid neighbor exists among the k candidates, i is removed as "stuck"
         (this guarantees termination: each iteration either adds a throat or removes
         an active pore).

    Parameters
    ----------
    net : pore_network
        The pore network model instance (must have pores with radius and center set).
    nodes : iterable of int or None, optional
        List of node indices to process. If None, all nodes are processed.
    ratio : float or tuple of float or dict, optional
        Maximum fraction of pore surface covered by throats (default 0.75).
        If a 4-tuple (loc, low, high, sigma), the ratio is drawn from a truncated
        normal distribution with these parameters (same convention as v1/v2).
        If a dict, it is a diameter-dependent profile (keys "sr_small", "sr_large",
        "r_lo_um", "r_hi_um"; see :func:`pnm.Stats.sratio_profile_values`) evaluated
        per pore from its radius — the same profile must be passed to the inversion
        (:func:`pnm.Stats.invert_pore_distribution_kernel_gamma`, ``sratio_profile``)
        so that model and generator budgets match.
    PRTRR : float, optional
        Pore radius to throat radius ratio. Throat radius = min(r1, r2) / PRTRR.
        Default is 2.
    minc : int, optional
        Minimum connections enforced per LARGE pore in pass 1 (default 2). Only used
        when ``large_radius_cutoff`` is set; ignored otherwise.
    large_radius_cutoff : float or None, optional
        Radius threshold that enables pass 1 (the independent large-pore sub-network):
        pores with radius >= this value are interconnected to reach ``minc`` connections
        within the large class before pass 2 runs. Can be chosen from the MIP pore
        distribution (e.g. at the valley between its two peaks). Default is None, which
        disables pass 1 and reproduces the original single-pass v3 behavior exactly.
    border_correction : bool, optional
        If True, reduce the allowed surface coverage for boundary pores (50% per
        adjacent face). Default is True.
    seed : int or None, optional
        Seed for the random pore selection in pass 2 (reproducibility). Default is None.

    Returns
    -------
    None
        Throats are added in place to ``net.graph``. Node attributes
        ``exchange_area`` and ``surface_ratio`` are set/updated by this function.

    Notes
    -----
    Each new throat gets its ``radius`` attribute; length is computed automatically,
    with periodic=False and pdir="none" (see :meth:`pore_network.add_throat`).
    Pairs that are already connected are skipped (``graph.has_edge``).

    Pass 1 counts a pore's TOTAL degree against ``minc`` (matching v1/v2 phase 0), so on
    a fresh pore-only network this is exactly the number of large-large connections; any
    pre-existing edges also count toward the floor. Periodic images are not considered in
    either pass.
    """

    nodelist = list(net.graph.nodes) if nodes is None else list(nodes)
    N = len(nodelist)

    if N <= 0:
        return

    # ------------------------------------------------------------------
    # Vectorized state (arrays indexed by position in ``nodelist``)
    # ------------------------------------------------------------------
    radii_all = np.array([net.graph.nodes[n]["radius"] for n in nodelist], dtype=float)
    centers = np.array([net.graph.nodes[n]["center"] for n in nodelist], dtype=float)

    # Per-pore ratio: constant, drawn from a truncated normal distribution, or evaluated from a
    # diameter-dependent profile dict (see pnm.Stats.sratio_profile_values — the same profile must
    # be passed to the inversion so model and generator budgets match).
    if isinstance(ratio, dict):
        sr_vals = sratio_profile_values(radii_all, ratio)
    else:
        try:
            loc, low, high, sigma = ratio
            sr_vals = distribution(N, name="normal", high=high, low=low, loc=loc, scale=sigma)
        except (TypeError, ValueError):
            sr_vals = np.full(N, float(ratio))

    surface_ratio_dict = dict(zip(nodelist, sr_vals))
    nx.set_node_attributes(net.graph, surface_ratio_dict, name="surface_ratio")

    # Per-pore corrected ratio, precomputed once.
    if border_correction:
        innercat = frozenset(["inner", "none"])
        corr = np.ones(N)
        for i, n in enumerate(nodelist):
            cat = net.graph.nodes[n].get("category") or set()
            if not (innercat & cat):
                # Reduce by 50% per adjacent face (50% for one face, 25% for an edge, 12.5% for a corner)
                corr[i] = 0.5 ** len(set(cat) - innercat)
    else:
        corr = np.ones(N)

    budget = corr * sr_vals * 4.0 * np.pi * radii_all**2
    exchange = np.zeros(N)
    deg = np.fromiter((net.graph.degree(n) for n in nodelist), dtype=np.int64, count=N)

    # Smallest throat cross-section this pore could ever form (partner = smallest pore of
    # the network). A pore whose residual budget cannot absorb even that is saturated:
    # budgets are never filled *exactly* by the acceptance test, so "exchange >= budget"
    # would almost never trigger and every pore would end up classified as "stuck".
    s12_min = np.pi * (float(radii_all.min()) / PRTRR) ** 2

    tree = cKDTree(centers)
    rng = np.random.default_rng(seed)

    # Active list: positions of pores that can still accept throats. ``pos_in_active``
    # maps position -> index into ``active`` (-1 once removed); swap-remove keeps both O(1).
    active = list(range(N))
    pos_in_active = np.arange(N, dtype=np.int64)

    def _remove_at(pos):
        """Remove the element at index ``pos`` of ``active`` (swap with last + pop).

        The removed pore's entry in ``pos_in_active`` is set to -1 so that it can no
        longer be accepted as a partner (a stale index would corrupt the active list).
        """
        last = len(active) - 1
        removed = active[pos]
        if pos != last:
            moved = active[last]
            active[pos] = moved
            pos_in_active[moved] = pos
        pos_in_active[removed] = -1
        active.pop()

    n_added = 0
    n_stuck = 0
    step = 0
    progress_step = max(1, N // 100)

    # ------------------------------------------------------------------
    # Pass 1: independent large-pore sub-network (only when a cutoff radius is set).
    # Interconnects the pores with radius >= ``large_radius_cutoff`` so that each gets
    # at least ``minc`` connections WITHIN the large class, before pass 2 fills in the
    # rest. This builds an isolated network of large throats (throat radius =
    # min(r1, r2)/PRTRR requires BOTH endpoints to be large) that pass 2 alone would
    # miss: a large pore saturates on nearby small pores before any distant large-large
    # pair is considered. Mirrors the phase-0 connectivity floor of v1/v2, restricted
    # to the large class and run first so it is independent of the small pores.
    # ------------------------------------------------------------------
    n_pass1 = 0
    if large_radius_cutoff is not None and minc > 0:
        cutoff = float(large_radius_cutoff)
        large_mask = radii_all >= cutoff
        n_large = int(large_mask.sum())
        if n_large > 1:
            # Global positions of the large pores; a dedicated tree over their centers
            # so neighbor queries return only large partners.
            large_pos = np.flatnonzero(large_mask)
            large_tree = cKDTree(centers[large_pos])
            # Query one extra slot beyond the pool: the tree contains the query pore's own
            # center, which is always its nearest hit and would otherwise consume a slot
            # (leaving zero real candidates when n_large is small). The loop skips p == i.
            k_pool = min(n_large, max(32, minc * 8) + 1)
            # Largest first (like v1/v2 phase 0): the biggest pores claim their floor
            # before smaller large-pores can saturate nearby partners.
            for i in large_pos[np.argsort(-radii_all[large_pos], kind="stable")]:
                n1 = nodelist[i]
                r_i = radii_all[i]
                if deg[i] >= minc:
                    continue  # already has its floor (edges added earlier this pass)

                dists_l, idx_l = large_tree.query(centers[i], k=k_pool)
                dists_l = np.atleast_1d(dists_l).astype(float)
                idx_l = np.atleast_1d(idx_l).astype(np.int64)
                # Surface-to-surface distance order (same metric as pass 2 / v1/v2).
                d_surf = dists_l - radii_all[large_pos[idx_l]] - r_i
                o = np.argsort(d_surf, kind="stable")

                # Greedy acceptance within both endpoints' surface budgets.
                for t in range(len(idx_l)):
                    if deg[i] >= minc:
                        break
                    p = int(large_pos[int(idx_l[o[t]])])
                    if p == i or net.graph.has_edge(n1, nodelist[p]):
                        continue
                    r_t = min(r_i, radii_all[p]) / PRTRR
                    s12 = np.pi * r_t**2
                    if exchange[i] + s12 <= budget[i] and exchange[p] + s12 <= budget[p]:
                        net.add_throat(n1, nodelist[p], radius=r_t)
                        exchange[i] += s12
                        exchange[p] += s12
                        deg[i] += 1
                        deg[p] += 1
                        n_added += 1
                        n_pass1 += 1

                # Connectivity floor: force-connect the nearest unconnected large
                # neighbors until minc is reached, IGNORING the surface budget (the
                # floor is guaranteed even if ratio is exceeded, matching v2 phase 0).
                for t in range(len(idx_l)):
                    if deg[i] >= minc:
                        break
                    p = int(large_pos[int(idx_l[o[t]])])
                    if p == i or net.graph.has_edge(n1, nodelist[p]):
                        continue
                    r_t = min(r_i, radii_all[p]) / PRTRR
                    s12 = np.pi * r_t**2
                    net.add_throat(n1, nodelist[p], radius=r_t)
                    exchange[i] += s12
                    exchange[p] += s12
                    deg[i] += 1
                    deg[p] += 1
                    n_added += 1
                    n_pass1 += 1

            print(f"v3 pass 1: {n_pass1} large-large throats " f"(pores with r >= {cutoff:.4g}, minc={minc})")

    while len(active) > 0:
        step += 1
        if (step - 1) % progress_step == 0:
            print(f"v3: {n_added} throats | active pores {len(active)}", end="\r", flush=True)

        a = int(rng.integers(len(active)))
        i = int(active[a])
        n1 = nodelist[i]
        r_i = radii_all[i]

        # Candidate pool: k nearest by center distance, large enough to contain at
        # least 32 pores not yet connected to i (its current degree is a lower bound
        # on how many of the nearest candidates are already its partners).
        k = min(N - 1, max(32, int(deg[i]) + 32))
        dists, idx = tree.query(centers[i], k=k)
        dists = np.atleast_1d(dists).astype(float)
        idx = np.atleast_1d(idx).astype(np.int64)

        # Surface-to-surface distance order (same metric as v1/v2).
        d_surf = dists - radii_all[idx] - r_i
        order = np.argsort(d_surf, kind="stable")

        chosen = -1
        S12 = 0.0
        for t in range(k):
            p = int(idx[order[t]])
            if p == i:
                continue
            # Only still-active pores can be partners (a removed pore's budget is frozen).
            if pos_in_active[p] < 0:
                continue
            n2 = nodelist[p]
            if net.graph.has_edge(n1, n2):
                continue
            r_t = min(r_i, radii_all[p]) / PRTRR
            s12 = np.pi * r_t**2
            # Both endpoints must be able to absorb the throat cross-section.
            if exchange[i] + s12 <= budget[i] and exchange[p] + s12 <= budget[p]:
                chosen = p
                S12 = float(s12)
                break

        if chosen < 0:
            # No valid partner among the k nearest: remove i (guarantees termination).
            n_stuck += 1
            _remove_at(a)
            continue

        net.add_throat(n1, nodelist[chosen], radius=min(r_i, radii_all[chosen]) / PRTRR)
        exchange[i] += S12
        exchange[chosen] += S12
        deg[i] += 1
        deg[chosen] += 1
        n_added += 1

        # Deactivate every pore whose residual budget can no longer absorb even the
        # smallest possible throat (higher index first so the other position stays valid).
        to_remove = []
        if exchange[i] + s12_min > budget[i]:
            to_remove.append(a)
        if exchange[chosen] + s12_min > budget[chosen]:
            to_remove.append(int(pos_in_active[chosen]))
        for pos in sorted(set(to_remove), reverse=True):
            _remove_at(pos)

    # Every pore is removed exactly once in pass 2: either saturated (residual budget
    # too small) or stuck (picked with no valid partner among its k nearest). ``n_added``
    # counts both passes; the pass-1 large-large throats are broken out separately.
    print(
        f"v3: {n_added} throats added ({n_pass1} large-large in pass 1) | "
        f"{N - n_stuck} pores saturated, {n_stuck} stuck (no valid partner among k nearest)"
    )

    # Write back the final covered surface areas (the graph attribute is the contract).
    nx.set_node_attributes(net.graph, dict(zip(nodelist, exchange)), name="exchange_area")


def add_throats_by_surf_v4(
    net,
    nodes=None,
    ratio=0.75,
    PRTRR=2,
    minc1=2,
    maxc=np.inf,
    search_radius_factor1=2.0,
    search_radius_factor2=3.0,
    minc2=0,
    pass2_order="increasing",
    border_correction=True,
    seed=None,
):
    """Connect pores by a two-pass surface-budget algorithm (v4).

    v4 is the intended replacement for :func:`add_throats_by_surf` (v1). It fixes the
    systematic miss of large-large throats that limits v1: in v1 the radius-sorted pass
    operates on a tiny k-nearest-by-center pool (which, for a giant pore, contains only
    small surface-hugging pores), while the wide candidate ball is sorted by surface
    distance and fills up with small-gap pores before any distant large partner is reached.

    Pass 1 (large-partner floor):
      - process pores in DECREASING radius order;
      - for each pore i, cast a ball of center-distance radius
        ``search_radius_factor1 * r_i + r_i`` and sort the candidates by RADIUS descending;
      - connect up to ``minc1`` of them (the largest partners) subject to both endpoints'
        surface budgets and the ``maxc`` degree cap.

    Pass 2 (budget fill):
      - process pores in the order given by ``pass2_order`` ("increasing" radius by default);
      - for each pore i, cast a ball of center-distance radius
        ``search_radius_factor2 * r_i + r_i``, sort candidates by SURFACE-TO-SURFACE
        distance ascending, and connect to them one by one until the pore's budget is
        exhausted, its degree reaches ``maxc``, or no valid candidate remains in the ball;
      - if after this fill the pore has fewer than ``minc2`` total connections, widen the
        search ball iteratively (doubling the radius factor) and continue connecting
        nearest-by-surface-distance partners until ``minc2`` is reached or no candidates
        remain anywhere in the network.

    The processing order matters: with "increasing", small pores form their local network
    before large pores can pre-consume their budget; with "decreasing", large pores claim
    their (wide-ball) partners first; "random" averages over both effects (use ``seed`` for
    reproducibility).

    Parameters
    ----------
    net : pore_network
        The pore network model instance (must have pores with radius and center set).
    nodes : iterable of int or None, optional
        List of node indices to process. If None, all nodes are processed.
    ratio : float or tuple of float or dict, optional
        Maximum fraction of pore surface covered by throats (default 0.75). If a 4-tuple
        (loc, low, high, sigma), the ratio is drawn from a truncated normal distribution
        with these parameters (same convention as v1/v2/v3). If a dict, it is a
        diameter-dependent profile (see :func:`pnm.Stats.sratio_profile_values`) evaluated
        per pore from its radius. Default is 0.75.
    PRTRR : float, optional
        Pore radius to throat radius ratio. Throat radius = min(r1, r2) / PRTRR.
        Default is 2.
    minc1 : int, optional
        Number of LARGEST partners each pore connects to in pass 1 (default 2). Subject to
        both endpoints' surface budgets and the ``maxc`` cap; no force-connect beyond budget.
    maxc : float, optional
        Maximum total connectivity per pore across both passes (default np.inf = no cap).
        A finite value tames the heavy degree tail of large pores. Default is np.inf.
    search_radius_factor1 : float, optional
        Pass-1 candidate ball has center-distance radius ``search_radius_factor1 * r_i + r_i``
        (default 2.0 -> 3 * r_i). Larger values let each pore reach more distant partners so
        that sort-by-radius can pick larger ones. Default is 2.0.
    search_radius_factor2 : float, optional
        Pass-2 candidate ball has center-distance radius ``search_radius_factor2 * r_i + r_i``
        (default 3.0 -> 4 * r_i). This defines the initial search region for the nearest-by-
        surface-distance fill. The minc2 floor phase widens beyond this if needed. Default is 3.0.
    minc2 : int, optional
        Minimum total connectivity enforced per pore in pass 2 (default 0 = disabled). If a
        pore ends its initial fill with fewer than ``minc2`` connections, the search ball is
        widened iteratively until the floor is met or no candidates remain. Default is 0.
    pass2_order : str, optional
        Order in which pores are processed in pass 2: "increasing" (smallest radius first,
        default), "decreasing" (largest first) or "random" (uniform random permutation; use
        ``seed`` for reproducibility). Default is "increasing".
    border_correction : bool, optional
        If True, reduce the allowed surface coverage for boundary pores (50% per adjacent
        face). Default is True.
    seed : int or None, optional
        Seed for the random pore ordering when ``pass2_order="random"`` (reproducibility).
        Ignored otherwise. Default is None.

    Returns
    -------
    None
        Throats are added in place to ``net.graph``. Node attributes ``exchange_area`` and
        ``surface_ratio`` are set/updated by this function.

    Notes
    -----
    Each new throat gets its ``radius`` attribute; length is computed automatically, with
    periodic=False and pdir="none" (see :meth:`pore_network.add_throat`). Pairs that are
    already connected are skipped (``graph.has_edge``). Periodic images are not considered.
    """

    nodelist = list(net.graph.nodes) if nodes is None else list(nodes)
    N = len(nodelist)

    if N <= 0:
        return

    # ------------------------------------------------------------------
    # Vectorized state (arrays indexed by position in ``nodelist``).
    # ------------------------------------------------------------------
    radii_all = np.array([net.graph.nodes[n]["radius"] for n in nodelist], dtype=float)
    centers = np.array([net.graph.nodes[n]["center"] for n in nodelist], dtype=float)

    # Per-pore ratio: constant, truncated-normal draw, or diameter-dependent profile dict.
    if isinstance(ratio, dict):
        sr_vals = sratio_profile_values(radii_all, ratio)
    else:
        try:
            loc, low, high, sigma = ratio
            sr_vals = distribution(N, name="normal", high=high, low=low, loc=loc, scale=sigma)
        except (TypeError, ValueError):
            sr_vals = np.full(N, float(ratio))

    surface_ratio_dict = dict(zip(nodelist, sr_vals))
    nx.set_node_attributes(net.graph, surface_ratio_dict, name="surface_ratio")

    # Per-pore corrected ratio, precomputed once.
    if border_correction:
        innercat = frozenset(["inner", "none"])
        corr = np.ones(N)
        for i, n in enumerate(nodelist):
            cat = net.graph.nodes[n].get("category") or set()
            if not (innercat & cat):
                # Reduce by 50% per adjacent face (50% for one face, 25% for an edge, 12.5% for a corner).
                corr[i] = 0.5 ** len(set(cat) - innercat)
    else:
        corr = np.ones(N)

    budget = corr * sr_vals * 4.0 * np.pi * radii_all**2
    exchange = np.zeros(N)
    deg = np.fromiter((net.graph.degree(n) for n in nodelist), dtype=np.int64, count=N)

    tree = cKDTree(centers)
    maxc_finite = bool(np.isfinite(maxc))

    # Precompute the box diagonal as an upper bound for the widening loop.
    extent = np.ptp(centers, axis=0)
    box_diag = float(np.linalg.norm(extent))

    # ------------------------------------------------------------------
    # Pass 1: large-partner floor. For each pore (largest first), cast a ball of radius
    # search_radius_factor1 * r_i + r_i, sort candidates by RADIUS descending and connect up
    # to minc1 of them within both endpoints' surface budgets and the maxc cap.
    # ------------------------------------------------------------------
    n_pass1 = 0
    if minc1 > 0:
        order_desc = np.argsort(-radii_all, kind="stable")
        progress_step = max(1, len(order_desc) // 100)
        for counter, i in enumerate(order_desc):
            if (counter + 1) % progress_step == 0:
                print(
                    f"v4 pass 1: {n_pass1} throats | {100 * (counter + 1) / len(order_desc):5.2f}% ",
                    end="\r",
                    flush=True,
                )

            i = int(i)
            n1 = nodelist[i]
            r_i = radii_all[i]
            if deg[i] >= minc1 or (maxc_finite and deg[i] >= maxc):
                continue  # already has its floor / cap

            # Ball: center distance < search_radius_factor1 * r_i + r_i.
            ball_r = float(search_radius_factor1) * r_i + r_i
            idx = np.asarray(tree.query_ball_point(centers[i], ball_r))
            if len(idx) <= 1:
                continue

            # Sort by RADIUS descending (largest partners first); ties broken by surface distance.
            d_surf = np.linalg.norm(centers[idx] - centers[i], axis=1) - radii_all[idx] - r_i
            o = np.lexsort((d_surf, -radii_all[idx]))
            seq = idx[o]

            # Greedy acceptance within both endpoints' budgets and the maxc cap.
            for t in range(len(seq)):
                if deg[i] >= minc1 or (maxc_finite and deg[i] >= maxc):
                    break
                p = int(seq[t])
                if p == i or net.graph.has_edge(n1, nodelist[p]):
                    continue
                if maxc_finite and deg[p] >= maxc:
                    continue
                r_t = min(r_i, radii_all[p]) / PRTRR
                s12 = np.pi * r_t**2
                if exchange[i] + s12 <= budget[i] and exchange[p] + s12 <= budget[p]:
                    net.add_throat(n1, nodelist[p], radius=r_t)
                    exchange[i] += s12
                    exchange[p] += s12
                    deg[i] += 1
                    deg[p] += 1
                    n_pass1 += 1

        print(
            f"v4 pass 1: {n_pass1} large-partner throats (minc1={minc1}, ball={(float(search_radius_factor1) + 1):g}*r_i)"
        )

    # ------------------------------------------------------------------
    # Pass 2: budget fill. For each pore (in the order given by ``pass2_order``), cast a ball
    # of radius search_radius_factor2 * r_i + r_i, sort candidates by SURFACE-TO-SURFACE
    # distance and connect to them one by one until the pore's budget is exhausted, its degree
    # reaches maxc, or no valid candidate remains. If the resulting degree is below minc2,
    # widen the ball iteratively (doubling) and continue connecting nearest-by-surface-
    # distance partners. "increasing" lets small pores form their local network before large
    # pores can pre-consume their budget; "decreasing" does the opposite; "random" averages.
    # ------------------------------------------------------------------
    n_pass2 = 0
    if pass2_order == "decreasing":
        order2 = np.argsort(-radii_all, kind="stable")
    elif pass2_order == "increasing":
        order2 = np.argsort(radii_all, kind="stable")
    elif pass2_order == "random":
        rng2 = np.random.default_rng(seed)
        order2 = rng2.permutation(N)
    else:
        raise ValueError(f"Unknown pass2_order {pass2_order!r} (expected 'increasing', 'decreasing' or 'random')")
    progress_step = max(1, len(order2) // 100)

    for counter, i in enumerate(order2):
        if (counter + 1) % progress_step == 0:
            print(
                f"v4 pass 2 ({pass2_order}): {n_pass2} throats | {100 * (counter + 1) / len(order2):5.2f}% ",
                end="\r",
                flush=True,
            )

        i = int(i)
        n1 = nodelist[i]
        r_i = radii_all[i]

        # Skip pores that are already saturated (budget full or degree capped).
        if maxc_finite and deg[i] >= maxc:
            continue
        s12_min_possible = np.pi * (float(radii_all.min()) / PRTRR) ** 2
        if exchange[i] + s12_min_possible > budget[i]:
            continue

        # Initial ball: center distance < search_radius_factor2 * r_i + r_i.
        factor = float(search_radius_factor2)
        ball_r = factor * r_i + r_i
        idx = np.asarray(tree.query_ball_point(centers[i], ball_r))
        if len(idx) > 1:
            # Sort by SURFACE-TO-SURFACE distance ascending (nearest surface partners first).
            d_surf = np.linalg.norm(centers[idx] - centers[i], axis=1) - radii_all[idx] - r_i
            o = np.argsort(d_surf, kind="stable")
            seq = idx[o]

            # Connect to candidates in surface-distance order until budget/maxc/no valid partner.
            for t in range(len(seq)):
                if maxc_finite and deg[i] >= maxc:
                    break
                p = int(seq[t])
                if p == i or net.graph.has_edge(n1, nodelist[p]):
                    continue
                if maxc_finite and deg[p] >= maxc:
                    continue
                r_t = min(r_i, radii_all[p]) / PRTRR
                s12 = np.pi * r_t**2
                if exchange[i] + s12 <= budget[i] and exchange[p] + s12 <= budget[p]:
                    net.add_throat(n1, nodelist[p], radius=r_t)
                    exchange[i] += s12
                    exchange[p] += s12
                    deg[i] += 1
                    deg[p] += 1
                    n_pass2 += 1

        # minc2 floor: if the pore has fewer than minc2 total connections, widen the search
        # ball iteratively (doubling the radius factor) and continue connecting nearest-by-
        # surface-distance partners until the floor is met or no candidates remain.
        if minc2 > 0 and deg[i] < minc2:
            # Track which candidates we've already tried to avoid re-checking them.
            tried = set(int(x) for x in idx) if len(idx) > 1 else set()
            while deg[i] < minc2 and not (maxc_finite and deg[i] >= maxc):
                # Widen: double the factor, capped at the box diagonal.
                ball_r = ball_r * 2.0
                if ball_r > box_diag:
                    break  # no further widening possible
                idx_wide = np.asarray(tree.query_ball_point(centers[i], ball_r))
                if len(idx_wide) <= 1:
                    break
                d_surf_w = np.linalg.norm(centers[idx_wide] - centers[i], axis=1) - radii_all[idx_wide] - r_i
                o_w = np.argsort(d_surf_w, kind="stable")
                seq_w = idx_wide[o_w]

                found_any = False
                for t in range(len(seq_w)):
                    if deg[i] >= minc2 or (maxc_finite and deg[i] >= maxc):
                        break
                    p = int(seq_w[t])
                    if p == i or p in tried:
                        continue
                    tried.add(p)
                    if net.graph.has_edge(n1, nodelist[p]):
                        continue
                    if maxc_finite and deg[p] >= maxc:
                        continue
                    r_t = min(r_i, radii_all[p]) / PRTRR
                    s12 = np.pi * r_t**2
                    if exchange[i] + s12 <= budget[i] and exchange[p] + s12 <= budget[p]:
                        net.add_throat(n1, nodelist[p], radius=r_t)
                        exchange[i] += s12
                        exchange[p] += s12
                        deg[i] += 1
                        deg[p] += 1
                        n_pass2 += 1
                        found_any = True

                if not found_any:
                    break  # no new valid partner in the widened ball; stop widening

    print(f"v4: {n_pass1 + n_pass2} throats added ({n_pass1} pass 1, {n_pass2} pass 2)")

    # Write back the final covered surface areas (the graph attribute is the contract).
    nx.set_node_attributes(net.graph, dict(zip(nodelist, exchange)), name="exchange_area")


def add_throats_by_surf_v2(
    net,
    nodes=None,
    ratio=0.75,
    PRTRR=2,
    minc=2,
    maxc=100,
    sort_by_radius=False,
    border_correction=True,
    search_radius_factor1=2.0,
    search_radius_factor2=3.0,
    min_neighbors_factor=3,
    phase1_order="decreasing",
    seed=None,
    large_pair_radius=None,
    large_pair_min_connections=None,
    large_pair_max_gap=4.0,
):
    """Connect pores by a surface-coverage budget in two explicit phases (plus an optional large-pair floor).

    Redesigned replacement for :func:`add_throats_by_surf` with a cleaner
    two-phase structure and an optional targeted large-pair pass:

      - **Phase 0 (connectivity floor)** — every pore is guaranteed at least
        ``minc`` connections, iterating pores in DECREASING radius order. The
        candidate pool is the ``minc * min_neighbors_factor`` nearest neighbors,
        optionally widened by all pores whose surface-to-surface gap to the active
        pore is <= ``search_radius_factor1 * r1`` when that factor is set (so large
        partners beyond the immediate neighborhood can be found). Candidates are sorted by
        radius (if ``sort_by_radius``) or surface distance and accepted greedily
        up to ``minc`` subject to the surface budget. If the budget leaves the
        pore below ``minc``, the nearest available neighbors are force-connected
        until ``minc`` is reached, IGNORING the surface budget (connectivity is
        guaranteed even if ``ratio`` is exceeded).

      - **Phase 1 (fill)** — additional connections up to ``maxc`` / the surface
        budget, only with neighbors whose surface-to-surface gap <=
        ``search_radius_factor2 * r1``. Nearest neighbors are tried first; there
        is no fallback when a pore has no neighbor in range. The order in which
        pores are processed is chosen by ``phase1_order`` ("decreasing",
        "increasing" or "random").

      - **Phase 2 (optional large-pair floor)** — only when ``large_pair_radius``
        is set: every pore with radius >= ``large_pair_radius`` is guaranteed at
        least ``large_pair_min_connections`` connections to OTHER such pores
        (default: ``minc``), IGNORING the surface budget. Candidates are the other
        large pores sorted by surface-to-surface gap; when ``large_pair_max_gap``
        is not None, only partners with gap <= cap are eligible (best effort
        otherwise). The number of new throats is bounded by n_large * floor / 2, so
        this adds a few dozen large throats (radius >= large_pair_radius / PRTRR)
        without disturbing the rest of the distribution. This targets the
        finite-size deficit of the largest MIP band: random placement scatters the
        few very large pores, and phases 0-1 only give each of them a couple of
        large partners — use ``large_pair_min_connections`` > minc to raise the
        floor above what phases 0-1 already provide.

    Unlike :func:`add_throats_by_surf`, the acceptance loop skips pairs that are
    already connected (``graph.has_edge``), so re-encountering an existing throat
    can no longer inflate the local degree / exchange mirrors ("phantom"
    increments) and wrongly trigger early breaks or suppress the phase-0 floor.

    Parameters
    ----------
    net : pore_network
        The pore network model instance (pores must have ``radius`` and
        ``center`` set; ``category`` for border correction).
    nodes : iterable of int or None, optional
        Pores processed as the "active" side. If None, all pores. Neighbors may
        be any pore in the network.
    ratio : float or 4-tuple (loc, low, high, sigma), optional
        Maximum fraction of pore surface covered by throats (default 0.75). A
        4-tuple draws each pore's ratio from a truncated normal distribution.
    PRTRR : float, optional
        Pore radius to throat radius ratio: throat radius = min(r1, r2) / PRTRR.
        Default is 2.
    minc : int, optional
        Minimum connectivity enforced per pore in phase 0 (default 2).
    maxc : int, optional
        Maximum connectivity per pore (default 100). Raised to ``minc + 1`` if
        smaller than ``minc``.
    sort_by_radius : bool, optional
        If True, phase-0 candidates are sorted by radius (largest first) so each
        pore connects to its largest available neighbors; otherwise by surface
        distance. Default is False.
    border_correction : bool, optional
        If True, reduce the allowed surface coverage for boundary pores (50% per
        adjacent face). Default is True.
    search_radius_factor1 : float or None, optional
        Phase-0 pool widening as a multiple of the active pore radius, measured as a
        SURFACE-to-surface gap: when set, every pore whose surface gap to the active
        pore is <= factor * r1 is added to the nearest-neighbor pool (so large partners
        beyond the immediate neighborhood can be found). If None, only the
        ``minc * min_neighbors_factor`` nearest neighbors are used. Default is None.
    search_radius_factor2 : float, optional
        Phase-1 maximum surface-to-surface gap as a multiple of the active pore
        radius: only neighbors with gap <= factor * r1 are connected. Default is 3.0.
    min_neighbors_factor : int, optional
        Phase-0 nearest-neighbor pool size is ``minc * min_neighbors_factor``.
        Default is 3.
    phase1_order : {"decreasing", "increasing", "random"}, optional
        Order in which pores are processed in phase 1 (default "decreasing").
        "decreasing" = largest first (favors large throats / hubs), "increasing"
        = smallest first (more uniform degree), "random" = shuffled.
    seed : int or None, optional
        Seed for the ``phase1_order="random"`` shuffle (reproducibility). Ignored
        otherwise. Default is None.
    large_pair_radius : float or None, optional
        Phase-2 threshold in µm: pores with radius >= this value are "large" and
        each gets at least ``large_pair_min_connections`` connections to other
        large pores (see Phase 2 above). Set it to PRTRR x the target large-throat
        radius (e.g. 1.1 for a >0.5 µm band with PRTRR=2) so every forced throat
        lands in that band. If None, phase 2 is skipped entirely. Default is None.
    large_pair_min_connections : int or None, optional
        Phase-2 floor: minimum number of connections each large pore must have to
        OTHER large pores (existing ones from phases 0-1 count). If None, ``minc``
        is used — note that at large N with a generous phase-1 gap limit this may
        already be satisfied by phases 0-1 alone, in which case raise it explicitly
        to actually add large throats. Default is None (use minc).
    large_pair_max_gap : float or None, optional
        Phase-2 maximum surface-to-surface gap (µm) between the two large pores;
        partners farther apart are not eligible (the floor then becomes best
        effort). If None, any other large pore is a candidate (still nearest
        first). Default is 4.0.

    Notes
    -----
    Node attributes ``exchange_area`` and ``surface_ratio`` are set/updated by this
    function. Each new throat gets its ``radius``; length is computed automatically,
    with periodic=False and pdir="none" (see :meth:`pore_network.add_throat`).

    Phase 2 deliberately ignores the surface budget AND the degree caps on both sides
    (it is an explicit override): at large N the phase-1 fill saturates every large pore
    at maxc, so respecting either constraint would make the floor a no-op exactly where it
    is needed. The excess per pore is bounded by the floor count. Forced throats still
    update the exchange/degree mirrors so downstream routines (e.g. ``add_periodic_throats``)
    see a consistent state, but a large pore's covered fraction may exceed ``ratio`` and its
    degree may exceed maxc as a result.
    """

    if maxc < minc:
        maxc = minc + 1

    nodelist = list(net.graph.nodes)
    N = len(nodelist)
    if N <= 0:
        return

    if nodes is None:
        nodes = nodelist

    # ------------------------------------------------------------------
    # Vectorized state (arrays indexed by position in ``nodelist``).
    # ------------------------------------------------------------------
    pos_of = {n: i for i, n in enumerate(nodelist)}
    radii_all = np.array([net.graph.nodes[n]["radius"] for n in nodelist], dtype=float)
    centers = np.array([net.graph.nodes[n]["center"] for n in nodelist], dtype=float)

    nx.set_node_attributes(net.graph, 0, name="exchange_area")
    exchange = np.zeros(N)
    deg = np.fromiter((net.graph.degree(n) for n in nodelist), dtype=np.int64, count=N)

    max_pore_radius = float(radii_all.max())

    # Per-pore surface ratio: constant or drawn from a truncated normal.
    try:
        loc, low, high, sigma = ratio
        sr_vals = distribution(N, name="normal", high=high, low=low, loc=loc, scale=sigma)
    except (TypeError, ValueError):
        sr_vals = np.full(N, float(ratio))

    surface_ratio_dict = dict(zip(nodelist, sr_vals))
    nx.set_node_attributes(net.graph, surface_ratio_dict, name="surface_ratio")

    # Per-pore corrected ratio and the resulting covered-surface budget.
    if border_correction:
        innercat = frozenset(["inner", "none"])
        corr = np.ones(N)
        for i, n in enumerate(nodelist):
            cat = net.graph.nodes[n].get("category") or set()
            if not (innercat & cat):
                corr[i] = 0.5 ** len(set(cat) - innercat)
    else:
        corr = np.ones(N)

    budget = corr * sr_vals * (4.0 * np.pi * radii_all**2)

    tree = cKDTree(centers)

    candidates = [n for n in nodes if net.graph.degree(n) < maxc]
    order_desc = sorted(candidates, key=lambda n: net.graph.nodes[n]["radius"], reverse=True)

    print(len(candidates), "nodes to process")

    f1 = None if search_radius_factor1 is None else float(search_radius_factor1)
    f2 = float(search_radius_factor2)
    k_pool = min_neighbors_factor * minc

    # ------------------------------------------------------------------
    # Phase 0: guarantee at least ``minc`` connections per pore (largest first).
    # ------------------------------------------------------------------
    # progress_step = max(1, len(order_desc) // 25)
    progress_step = 1
    for counter, n1 in enumerate(order_desc):
        if (counter + 1) % progress_step == 0:
            print(f"Phase 0: {100 * (counter + 1) / len(order_desc):5.2f}% completed   ", end="\r", flush=True)

        i1 = pos_of[n1]
        if deg[i1] >= minc:
            continue
        r1 = radii_all[i1]

        # Candidate pool: k nearest neighbors (local connectivity floor), optionally
        # widened by every pore whose SURFACE-to-surface gap to n1 is <= f1*r1.
        k = max(1, min(k_pool, N))
        _, idx_k = tree.query(centers[i1], k=k)
        idx = np.atleast_1d(idx_k)
        if f1 is not None:
            # Cast wide in center distance so the ball provably contains every pore with
            # gap <= f1*r1, then keep only those that actually satisfy the surface-gap cap.
            ib = np.asarray(tree.query_ball_point(centers[i1], (1.0 + f1) * r1 + max_pore_radius))
            db = np.linalg.norm(centers[ib] - centers[i1], axis=1) - radii_all[ib] - r1
            ib = ib[db <= f1 * r1]
            if len(ib):
                idx = np.unique(np.concatenate((idx, ib)))

        d_surf = np.linalg.norm(centers[idx] - centers[i1], axis=1) - radii_all[idx] - r1
        keep = np.asarray(idx) != i1  # drop the self query hit
        idx, d_surf = idx[keep], d_surf[keep]
        if len(idx) == 0:
            continue

        o_greedy = np.argsort(-radii_all[idx]) if sort_by_radius else np.argsort(d_surf, kind="stable")
        seq_greedy, ds_greedy = idx[o_greedy], d_surf[o_greedy]
        o_force = np.argsort(d_surf, kind="stable")  # nearest first, always
        seq_force = idx[o_force]

        for j in range(len(seq_greedy)):
            if deg[i1] >= minc:
                break
            p = int(seq_greedy[j])
            n2 = nodelist[p]
            if net.graph.has_edge(n1, n2):  # never re-count an existing throat
                continue
            if deg[p] >= max(1.0, maxc * corr[p]):
                continue
            r2 = radii_all[p]
            r12 = min(r1, r2) / PRTRR
            S12 = np.pi * r12**2
            if exchange[i1] + S12 <= budget[i1] and exchange[p] + S12 <= budget[p]:
                net.add_throat(n1, n2, radius=r12)
                exchange[i1] += S12
                exchange[p] += S12
                deg[i1] += 1
                deg[p] += 1

        # Force-connect the nearest available neighbors until minc is reached,
        # ignoring the surface budget (connectivity floor).
        if deg[i1] < minc:
            for j in range(len(seq_force)):
                if deg[i1] >= minc:
                    break
                p = int(seq_force[j])
                n2 = nodelist[p]
                if net.graph.has_edge(n1, n2):
                    continue
                if deg[p] >= max(1.0, maxc * corr[p]):
                    continue
                r2 = radii_all[p]
                r12 = min(r1, r2) / PRTRR
                S12 = np.pi * r12**2
                net.add_throat(n1, n2, radius=r12)
                exchange[i1] += S12
                exchange[p] += S12
                deg[i1] += 1
                deg[p] += 1

    print(f"Phase 0: {100.0:5.2f}% completed", end="\r", flush=True)

    # ------------------------------------------------------------------
    # Phase 1: fill up to maxc / budget within the phase-1 gap limit.
    # ------------------------------------------------------------------
    if phase1_order == "decreasing":
        order1 = order_desc
    elif phase1_order == "increasing":
        order1 = sorted(candidates, key=lambda n: net.graph.nodes[n]["radius"])
    elif phase1_order == "random":
        rng = np.random.default_rng(seed)
        order1 = list(candidates)
        rng.shuffle(order1)
    else:
        raise ValueError(f"phase1_order must be 'decreasing', 'increasing' or 'random', got {phase1_order!r}")

    progress_step = max(1, len(order1) // 25)
    for counter, n1 in enumerate(order1):
        if (counter + 1) % progress_step == 0:
            print(f"Phase 1: {100 * (counter + 1) / len(order1):5.2f}% completed", end="\r", flush=True)

        i1 = pos_of[n1]
        my_cap = max(1.0, maxc * corr[i1])
        if deg[i1] >= my_cap:
            continue
        r1 = radii_all[i1]

        # Cast wide enough to include every pore with gap <= f2*r1, then filter.
        idx = np.asarray(tree.query_ball_point(centers[i1], (1.0 + f2) * r1 + max_pore_radius))
        d_surf = np.linalg.norm(centers[idx] - centers[i1], axis=1) - radii_all[idx] - r1
        keep = (d_surf <= f2 * r1) & (np.asarray(idx) != i1)
        idx, d_surf = idx[keep], d_surf[keep]
        if len(idx) == 0:
            continue

        o = np.argsort(d_surf, kind="stable")
        seq = idx[o]

        for j in range(len(seq)):
            if deg[i1] >= my_cap:
                break
            p = int(seq[j])
            n2 = nodelist[p]
            if net.graph.has_edge(n1, n2):  # never re-count an existing throat
                continue
            if deg[p] >= max(1.0, maxc * corr[p]):
                continue
            r2 = radii_all[p]
            r12 = min(r1, r2) / PRTRR
            S12 = np.pi * r12**2
            if exchange[i1] + S12 <= budget[i1] and exchange[p] + S12 <= budget[p]:
                net.add_throat(n1, n2, radius=r12)
                exchange[i1] += S12
                exchange[p] += S12
                deg[i1] += 1
                deg[p] += 1
            elif exchange[i1] + S12 > budget[i1] and deg[i1] >= minc:
                break

    print(f"Phase 1: {100.0:5.2f}% completed", end="\r", flush=True)

    # ------------------------------------------------------------------
    # Phase 2 (optional): large-pair connectivity floor.
    # Every pore with radius >= ``large_pair_radius`` gets at least ``minc``
    # connections to other such pores, ignoring the surface budget (bounded by
    # n_large * minc / 2 new throats). See the docstring for details.
    # ------------------------------------------------------------------
    if large_pair_radius is not None:
        R_lp = float(large_pair_radius)
        lp_floor = minc if large_pair_min_connections is None else int(large_pair_min_connections)
        large = np.where(radii_all >= R_lp)[0]
        n_large = len(large)
        print(f"Phase 2 (large-pair floor): {n_large} pores with r >= {R_lp:g}, floor={lp_floor}")

        if n_large > 1:
            # Pairwise surface gaps among the large pores only (n_large is small, so a full
            # matrix is cheap and exact).
            cL, rL = centers[large], radii_all[large]
            gap_ll = np.linalg.norm(cL[:, None, :] - cL[None, :, :], axis=2) - rL[:, None] - rL[None, :]
            np.fill_diagonal(gap_ll, np.inf)
            if large_pair_max_gap is not None:
                gap_ll[gap_ll > float(large_pair_max_gap)] = np.inf

            active_set = set(nodes)  # scope the floor like phases 0-1 (nodes defaults to all pores)
            for a in np.argsort(-rL, kind="stable"):  # largest first: hubs get their floor first
                i1 = int(large[a])
                if nodelist[i1] not in active_set:
                    continue

                # Count existing connections to other large pores (phases 0-1 may have made some).
                n_ll = sum(net.graph.has_edge(nodelist[i1], nodelist[int(large[b])]) for b in range(n_large) if b != a)
                need = lp_floor - n_ll
                if need <= 0:
                    continue

                partners = np.where(np.isfinite(gap_ll[a]))[0]
                partners = partners[np.argsort(gap_ll[a, partners], kind="stable")]  # nearest gap first
                added = 0
                for b in partners:
                    if added >= need:
                        break
                    i2 = int(large[int(b)])
                    n2 = nodelist[i2]
                    if net.graph.has_edge(nodelist[i1], n2):
                        continue
                    # Degree caps are intentionally NOT checked on either side: at large N the
                    # phase-1 fill saturates every large pore at maxc, so respecting the cap
                    # would make this floor a no-op exactly where it is needed. The excess per
                    # pore is bounded by ``lp_floor`` (a few edges over an already-saturated hub).
                    r12 = min(radii_all[i1], radii_all[i2]) / PRTRR
                    S12 = np.pi * r12**2
                    # Budget is intentionally NOT checked here (explicit override); the mirrors
                    # are still updated so downstream accounting stays consistent.
                    net.add_throat(nodelist[i1], n2, radius=r12)
                    exchange[i1] += S12
                    exchange[i2] += S12
                    deg[i1] += 1
                    deg[i2] += 1
                    added += 1

    # Write back the final covered surface areas (the graph attribute is the contract).
    nx.set_node_attributes(net.graph, dict(zip(nodelist, exchange)), name="exchange_area")


def add_throats_by_distribution(
    net,
    bins,
    volfrac,
    n_throats=None,
    PRTRR=2.0,
    ratio=0.75,
    border_correction=True,
    max_gap_factor=5.0,
    gap_radius_floor=None,
    force_pair_max_eligible=64,
    maxc=1024,
    minc=1,
    ensure_connected=True,
    max_retries=5,
    exhaust_streak=16,
    n_probe=32,
    n_iterations=2,
    seed=None,
    verbose=False,
):
    """Constrain the volumetric throat distribution at pairing time.

    Instead of letting the throat size distribution emerge from surface budgets
    (as in :func:`add_throats_by_surf`), the throat radii are *prescribed* by a
    target volumetric size distribution and placed via direct pair assignment:

      1. All gap-valid pore pairs are precomputed exactly once (a pair {i, j} is
         "gap-valid" when its surface-to-surface gap <= ``max_gap_factor * max(r_i, r_j)``
         AND both endpoints can host even the smallest prescribed throat). Each pair's
         maximum hostable radius is ``min(r_i, r_j) / PRTRR``, which determines the set of
         bins it can serve.
      2. Pairs are processed in ASCENDING max-hostable-radius order (small-capacity pairs
         first), so that small throats are placed while all surface budgets are still fresh.
         Each pair is assigned to its HIGHEST bin that still has demand and whose constraints
         (endpoint eligibility, remaining surface budget, degree cap) are satisfied for the
         drawn throat radius. This eliminates the "parking problem" of random-primary search:
         each pair is considered exactly once, with no wasted retries.
      3. The throat is created with a radius ``Rc`` drawn uniformly inside the assigned bin
         (the PRTRR relation is used only as an eligibility constraint, not to compute Rc).

    The per-pore surface budget has the same convention as
    :func:`add_throats_by_surf`: each pore may have at most
    ``ratio * corrected_ratio * 4*pi*r^2`` of its surface covered by throat
    cross-sections (pi*Rc^2 consumed at each endpoint).

    Notes on throughput: the number of placeable throats is limited by the precomputed pair
    count, which depends on ``max_gap_factor``, ``gap_radius_floor`` and the packing geometry. At
    f=5 with no floor, small pores typically have only ~2 gap-valid partners each; in sparse
    packings this can limit total placement to ~10-20k throats at N=20k. The preferred way to give
    small pores more partners is ``gap_radius_floor`` (default auto = median pore radius): it raises
    the allowed absolute gap ONLY for below-floor pores, leaving large-pore pairing untouched and
    adding no extra query cost when the floor is <= r_cut. Increasing ``max_gap_factor`` globally
    (e.g. 10-20) also works but lengthens throats of ALL sizes. The demand is capped at 25% of the
    pair count to maintain >80% placement success.

    Parameters
    ----------
    net : pore_network
        The pore network model instance. Pores must have ``radius``, ``center``
        and (for border correction) ``category`` attributes set, e.g. after a
        kdtree packing + SetFaceBCnodes. Existing throats are kept; their
        surface consumption is NOT accounted for in the budgets.
    bins : array-like of float
        Throat radius bin edges (n_bins + 1 strictly increasing values), same
        convention as ``Stats.stats`` mode="vol" and the experimental MIP files.
    volfrac : array-like of float
        Target volumetric fraction per bin (one value per bin, n_bins values).
        It is normalized internally, so it does not need to sum exactly to 1.
    n_throats : int or None, optional
        Total number of prescribed throats to place (a BUDGET: per-bin counts are derived from
        it and then capped — see the Notes). If None, defaults to a mean connectivity of ~9
        (``n_throats = int(4.5 * N)``), comparable to ``add_throats_by_surf`` at ratio ~0.75.
        Note that each placed throat consumes one DISTINCT pore pair: the total number of pairs
        satisfying the gap cap is finite, so very large values do not place more throats — the
        demand is capped globally at 25% of the exact gap-valid pair count (to maintain >80%
        placement success with the greedy assignment) and per bin at its structural capacity
        (no redistribution). Default is None.
    PRTRR : float, optional
        Pore radius to throat radius ratio used as the eligibility constraint:
        both endpoints of a throat of radius Rc must have pore radius >=
        PRTRR * Rc. Default is 2.0.
    ratio : float or tuple of float, optional
        Maximum fraction of pore surface covered by throats (same meaning and
        truncated-normal tuple form as in ``add_throats_by_surf``).
        Default is 0.75.
    border_correction : bool, optional
        If True, reduce the allowed surface coverage for boundary pores
        (50% per adjacent face), like ``add_throats_by_surf``. Default is True.
    max_gap_factor : float, optional
        Maximum surface-to-surface gap between connected pores, expressed as a
        factor of the larger pore radius: gap <= f * max(r1, r2). This keeps the
        pairing local and avoids long thin throats. The value also sets the total
        number of placeable pairs (each throat consumes one distinct pair): for
        GM1-like data at N=20k it is ~4e4 for f = 2 but ~1e5 for f = 5, while the
        volume target demands tens of thousands of small throats — so f must be
        large enough that the capacity exceeds n_throats. f = 5 keeps gaps below
        ~0.5 um for typical small pores (still local) and is a good default.
        Default is 5.0.
    gap_radius_floor : float or None, optional
        Absolute floor on the reference radius used by the gap cap, so that SMALL pores are not
        starved of partners. The per-pair surface-to-surface gap cap becomes
        ``gap <= max_gap_factor * max(max(r1, r2), r_floor)`` instead of
        ``max_gap_factor * max(r1, r2)``: for large pores (radius >= r_floor) this is unchanged,
        but for tiny pores — whose relative cap ``f * r`` is far below the inter-pore spacing in a
        dense packing and therefore yields only ~2 gap-valid partners each — the allowed absolute
        gap no longer shrinks to zero. This directly increases the number of gap-valid pairs
        involving small pores, which are exactly the ones that host the (numerous) small throats.
        The cost is slightly longer/thinner throats for those small-pore connections.

        - ``None`` (default): AUTO — the floor is set so that a below-floor pore may bridge up to
          ~3 inter-pore spacings: ``r_floor = 3 * s_mean / max_gap_factor``, where ``s_mean`` is the
          mean nearest-neighbour centre distance. This is data-driven and scale-invariant (it tracks
          the packing density, not the network size), so it gives sensible values for any porosity or
          N. It raises small-pore partner counts several-fold without touching large-pore pairing,
          which is what lets the intended throat count be reached instead of being capped by pair
          scarcity. This is the recommended setting and what the parametric study uses.
        - A positive float: an explicit absolute floor (in the same units as the radii). Use a value
          around ``s_mean / max_gap_factor`` to bridge one spacing, or larger for longer small throats.
        - ``0`` or negative: DISABLED — the cap reverts to the plain ``f * max(r1, r2)`` (the
          pre-floor behaviour), useful for baseline comparisons.

        Default is None (auto = 3 inter-pore spacings).
    force_pair_max_eligible : int, optional
        Large-throat "forced pairing" threshold. When fewer than this many pores are large
        enough to host the current throat (radius >= PRTRR * Rc), the pairing switches to a
        FORCED mode: the primary is drawn among all radius-eligible pores with degree headroom
        and its partner is simply the NEAREST such pore that is not already connected — ignoring
        both the ``max_gap_factor`` gap cap and the per-pore surface budget (only the "not yet
        connected" test and ``deg < maxc`` are enforced). This mirrors the forced min-connectivity
        connection of :func:`add_throats_by_surf` pass 0: large throats are rare and carry most of
        the volume, so they must be placed by connecting the available (few) large pores even if
        the resulting throat is long, rather than being left empty. Normal (budget- and
        gap-constrained) pairing is used whenever at least this many eligible pores exist. Set to
        0 to disable forced mode entirely. Default is 64.
    maxc : int or float, optional
        Maximum connectivity per pore enforced during the constrained loop
        (safety cap; the surface budget usually binds first). A low value makes hub pores
        saturate early and block every remaining pair that involves them, which costs placement
        success and distorts the distribution: at N=80k (GM1) maxc=256 gives JSD 0.217 while
        maxc=1024 gives JSD 0.177 with ~7k more throats placed. Default is 1024.
    minc : int, optional
        Minimum connectivity enforced by a final repair pass: pores with fewer
        than ``minc`` throats are connected to their nearest unconnected pore
        (throat radius = min(r1, r2) / PRTRR; these repair throats are NOT part
        of the prescribed distribution). Default is 1. Set to 0 to disable.
    ensure_connected : bool, optional
        If True, a final pass adds throats between different connected
        components (radius = min(r1, r2) / PRTRR) until the network is
        connected. These repair throats are NOT part of the prescribed
        distribution. Default is True.
    max_retries : int, optional
        [UNUSED in the current pair-assignment implementation — kept for API compatibility.]
        Was: number of additional primary-pore draws per prescribed throat in the old
        random-search algorithm. Default is 5.
    exhaust_streak : int, optional
        [UNUSED in the current pair-assignment implementation — kept for API compatibility.]
        Was: per-bin early-skip threshold in the old retry-based algorithm. Default is 16.
    n_probe : int, optional
        Number of PROBE throats placed per placeable bin in the first
        (calibration) pass. The probe pass measures the actual mean throat
        length and placement success rate in every bin — information that cannot
        be known a priori for heavy-tailed targets (in dense packings throat
        lengths are set by inter-pore spacing, not proportional to radius) — so
        that the following passes can convert volume fractions into per-bin
        counts accurately. Default is 32.
    n_iterations : int, optional
        Total number of placement passes; each pass removes the previous pass's
        throats before placing its own. Pass 1 is always the probe pass (``n_probe``
        throats per placeable bin); every following pass places the full placeable budget
        (``n_throats`` scaled to the placeable volume fraction — unplaceable bins are excluded,
        not redistributed) with per-bin counts re-weighted from the MEASURED mean
        length and success rate of the previous pass, so that the expected placed
        volume matches ``volfrac``. Default is 2 (probe + one calibrated pass);
        use 1 for a probe-only run (fast but coarse).
    seed : int or None, optional
        Seed for the numpy Generator used to draw the prescribed radii and the
        primary pores. Seeded runs are fully reproducible. Default is None.
    verbose : bool, optional
        If True, print a real-time progress line during the constrained loop.
        Default is False.

    Returns
    -------
    dict or None
        Statistics of the run: ``n_placed`` (prescribed throats placed),
        ``placed_per_bin`` and ``misses_per_bin`` (lists, one entry per bin),
        ``n_insufficient_large_pores`` (misses where fewer than 2 pores were
        large enough — a finite-size limit, not fixable by retries),
        ``unplaceable_bins`` (list of bin indices that are structurally impossible at this
        network size: fewer than 2 pores with radius >= PRTRR times the bin's LOWER edge, so no
        throat in the bin can ever be hosted. They get zero prescribed throats and their target
        volume fraction is NOT redistributed — ``achieved_volfrac`` simply stays 0 there, an
        honest gap rather than a distortion of the other bins),
        ``n_pair_capacity`` (EXACT total number of gap-valid pore pairs — the global upper bound
        on placeable prescribed throats; the calibrated demand is capped at 90% of it),
        ``n_connectivity_added``, ``n_minc_added``, ``achieved_volfrac``
        (volumetric fraction per bin of the placed prescribed throats only,
        using their actual lengths) and ``mean_length_per_bin`` (actual mean
        throat length per bin; inf where nothing was placed — a diagnostic for
        the "length proportional to radius" assumption used in the count
        conversion). Returns None if there is nothing to place.

    Notes
    -----
    - Volume fractions are converted into per-bin COUNTS from a length model:
      N_i ∝ volfrac_i / (Rmid_i^2 * Lbar_i), where Rmid_i is the bin mid-radius
      and Lbar_i the assumed mean throat length in that bin. The probe pass
      measures Lbar_i and the placement success rate s_i per bin; every
      calibrated pass replaces them by the MEASURED values of the previous pass,
      so that the expected placed volume per bin matches ``volfrac`` (the
      self-similar model Lbar_i = Rmid_i is only a fallback for bins whose probe
      placed nothing). The achieved volumetric distribution therefore follows
      the target up to within-bin length variability and to missed throats;
      ``achieved_volfrac`` reports how close it got (actual lengths are used).
    - Prescribed radii are drawn uniformly inside each bin, consistent with the
      piecewise-uniform convention of ``inverse_transform_sampling_from_raw_data``.
    - No volume is ever redistributed between bins: a bin that cannot be filled simply reports a
      lower ``achieved_volfrac`` (0 for structurally unplaceable bins), so the per-bin comparison
      against ``volfrac`` stays honest instead of over-filling the small bins. The same principle
      applies to PRESCRIPTION: each bin's prescribed count is capped at its structural capacity
      C(n, 2) where n is the number of pores large enough to host even the smallest throat of the
      bin (both endpoints must be eligible and a pore pair hosts at most one throat). Pairs shared
      between bins (a large-large pair can also host a throat of any smaller bin) are consumed in
      ascending capacity order, so the caps are per-bin upper bounds, not exact quotas.
    - Gap-valid partner lists are PRECOMPUTED exactly once up front: each of the K largest pores
      (K = min(256, N//80)) is tested against all pores, and every other hostable pore ball-queries
      radius r_i + (1+f_gap)*max(r_cut, r_floor) (r_cut = smallest top-pore radius), which provably
      contains all of its non-top gap-valid partners under the floored cap. The exact pair count that
      falls out of this precomputation caps the TOTAL demand at 25% of capacity: each placed throat
      consumes one distinct gap-valid pair, and in sparse boxes (large extent relative to N) that
      number can be well below ``n_throats`` — the default mean-connectivity-9 budget over-prescribes
      there. Capping scales all bins' counts down proportionally, so the achieved distribution keeps
      its shape (``achieved_volfrac`` is normalized over placed volume). The 25% fraction (not 90%)
      accounts for the greedy assignment's inability to use all pairs efficiently when small pores
      have only a handful of gap-valid partners each: at higher utilization, hub pores saturate their
      degree cap and block remaining pairs involving them. ``gap_radius_floor`` raises that partner
      count (and hence capacity) specifically for small pores without touching large-pore pairing.
    - Large throats (bins with very few eligible pores) may not be fully filled because there are
      simply too few gap-valid pairs between the available large pores — a finite-size limitation
      of the packing geometry, not an algorithmic failure. The ``force_pair_max_eligible`` parameter
      is accepted for API compatibility but does NOT trigger forced-mode pairing in the current
      implementation (the pair-assignment approach uses only precomputed gap-valid pairs). To fill
      the large-throat band, increase ``max_gap_factor`` or use ``add_throats_by_surf``'s
      ``large_pair_radius_frac`` option on a separate pass.
    - Periodic images are not considered (same as ``add_throats_by_surf``); call
      ``add_periodic_throats`` afterwards if needed.
    """

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    bins = np.asarray(bins, dtype=float)
    volfrac = np.asarray(volfrac, dtype=float).ravel()
    if bins.ndim != 1 or bins.size < 2:
        raise ValueError("bins must be an array of n_bins+1 (>= 2) bin edges")
    if not np.all(np.diff(bins) > 0):
        raise ValueError("bin edges must be strictly increasing")
    if bins[0] <= 0:
        raise ValueError("bin edges must be positive (throat radii)")
    n_bins = bins.size - 1
    if volfrac.shape != (n_bins,):
        raise ValueError("volfrac must have one entry per bin ({})".format(n_bins))
    if np.any(volfrac < 0):
        raise ValueError("volfrac entries must be >= 0")
    vsum = float(volfrac.sum())
    if vsum <= 0:
        raise ValueError("volfrac must sum to a positive value")
    volfrac = volfrac / vsum

    nodelist = list(net.graph.nodes)
    N = len(nodelist)
    if N < 2:
        warn("add_throats_by_distribution needs at least 2 pores, got {}".format(N))
        return None

    pos_of = {n: i for i, n in enumerate(nodelist)}
    radii_all = np.array([net.graph.nodes[n]["radius"] for n in nodelist], dtype=float)
    centers = np.array([np.asarray(net.graph.nodes[n]["center"], dtype=float) for n in nodelist])

    # ------------------------------------------------------------------
    # Per-pore surface budgets (same convention as add_throats_by_surf)
    # ------------------------------------------------------------------
    try:
        loc, low, high, sigma = ratio
        sr_vals = distribution(N, name="normal", high=high, low=low, loc=loc, scale=sigma)
    except (TypeError, ValueError):
        sr_vals = np.full(N, float(ratio))

    if border_correction:
        innercat = frozenset(["inner", "none"])
        corr = np.ones(N)
        for i, n in enumerate(nodelist):
            cat = net.graph.nodes[n].get("category") or set()
            if not (innercat & cat):
                # Reduce by 50% per adjacent face (same as add_throats_by_surf).
                corr[i] = 0.5 ** len(set(cat) - innercat)
    else:
        corr = np.ones(N)

    budget_max = sr_vals * corr * 4.0 * np.pi * radii_all**2
    budget_rem = budget_max.copy()

    deg = np.fromiter((net.graph.degree(n) for n in nodelist), dtype=np.int64, count=N)
    # Neighbor sets indexed by POSITION (not label): O(1) "already connected" tests.
    neighbors_pos = [set() for _ in range(N)]
    for i, n in enumerate(nodelist):
        for nb in net.graph[n]:
            neighbors_pos[i].add(pos_of[nb])

    rng_base = np.random.default_rng(seed)
    n_iterations = max(1, int(n_iterations))
    # One independent child stream per pass: seeded runs stay fully reproducible.
    pass_rngs = [np.random.default_rng(ss) for ss in rng_base.spawn(n_iterations)]

    tree = cKDTree(centers)
    f_gap = float(max_gap_factor)

    # Absolute floor on the gap-cap reference radius (see ``gap_radius_floor`` docstring). It lets
    # small pores — whose relative cap f*r is far below the inter-pore spacing in a dense packing —
    # reach more partners. None -> auto; <= 0 -> disabled; > 0 -> explicit value.
    if gap_radius_floor is None:
        # AUTO: let a below-floor pore bridge up to ~3 inter-pore spacings. s_mean is the mean
        # nearest-neighbour centre distance, so the floored cap f_gap*r_floor = 3*s_mean — i.e. a
        # small pore may connect over at most three typical gaps. Tying the floor to the spacing
        # (not to pore radii, which are tiny relative to it) makes it scale-invariant: the same
        # rule gives sensible values for any packing density or network size. Measured on GM1
        # N=20k this raises small-pore partner counts ~5x and total pair capacity ~3x, enough that
        # the 25%-of-capacity demand cap stops binding (the intended throat count is reachable).
        nn, _ = tree.query(centers, k=2)
        s_mean = float(nn[:, 1].mean()) if N >= 2 else 0.0
        r_floor = 3.0 * s_mean / f_gap
    else:
        r_floor = float(gap_radius_floor)
    floor_on = r_floor > 0.0

    if n_throats is None:
        # Mean connectivity ~9, comparable to add_throats_by_surf at ratio ~0.75; see the
        # n_throats docstring for the capacity rationale (fits under max_gap_factor=5).
        n_throats = int(4.5 * N)
    n_throats = max(int(n_throats), 0)

    # ------------------------------------------------------------------
    # Precomputed gap-valid partner lists (EXACT). A pair {i, j} is "gap-valid" when its
    # surface-to-surface gap is <= f_gap * max(max(r_i, r_j), r_floor) AND both endpoints can host
    # even the smallest prescribed throat (radius >= PRTRR * bins[0]). The optional floor r_floor
    # (see ``gap_radius_floor``) stops the cap from shrinking to zero for tiny pores. Every
    # normal-mode placed throat consumes one such pair, so:
    #   - partner search reduces to scanning the primary's list (a handful of candidates) instead
    #     of kNN + full-box ball queries — in sparse boxes a small primary's valid partners are
    #     rare large pores that sit far down its kNN ordering, which made calibrated passes take
    #     minutes;
    #   - the total number of placeable throats is EXACTLY half the sum of the list lengths, so
    #     the calibrated demand can be capped at 90% of it (prescribing more only creates misses).
    #
    # The lists are built exactly in two parts:
    #   1. each of the K largest pores (K = min(256, N//80)) is tested against ALL pores — their
    #      huge gap caps would otherwise require whole-box queries; both list directions are set.
    #      (The floor never binds here: a top pore's radius already exceeds r_floor.)
    #   2. every other hostable pore i ball-queries radius r_i + (1+f_gap)*max(r_cut, r_floor)
    #      (r_cut = radius of the smallest top pore): any non-top partner j of a non-top pore i
    #      satisfies d(i,j) <= r_i + f_gap*max(max(r_i,r_j), r_floor) + r_j <= r_i + (1+f_gap)*
    #      max(r_cut, r_floor) (both r_i, r_j <= r_cut and the floor term is bounded by
    #      f_gap*r_floor + r_cut <= (1+f_gap)*max(r_cut, r_floor)), so no valid pair is missed;
    #      candidates are then filtered by the exact gap condition. Top partners of i were already
    #      added in part 1 and are skipped here. When r_floor <= r_cut this radius equals the
    #      original r_i + (1+f_gap)*r_cut, so the floor adds no extra query cost in that case.
    # ------------------------------------------------------------------
    r_min_pair = PRTRR * bins[0]

    k_top = min(256, max(8, N // 80)) if N >= 16 else N
    top_idx = np.argsort(radii_all)[-k_top:]  # K largest pores (ascending argsort -> suffix)
    is_top = np.zeros(N, dtype=bool)
    is_top[top_idx] = True
    r_cut = float(radii_all[top_idx[0]]) if k_top < N else 0.0

    partner_sets = [set() for _ in range(N)]

    def _gap_cap(ra, rb):
        """Per-pair surface-to-surface gap cap: f_gap * max(max(ra, rb), r_floor). With the floor
        disabled (r_floor <= 0) this reduces to the plain f_gap * max(ra, rb)."""
        if not floor_on:
            return f_gap * np.maximum(ra, rb)
        return f_gap * np.maximum(np.maximum(ra, rb), r_floor)

    def _valid_mask(rows):
        """Boolean (len(rows) x N) matrix of gap-valid partners: gap <= _gap_cap(r1, r2), both
        endpoints hostable (radius >= r_min_pair), self excluded."""
        d = np.linalg.norm(centers[rows][:, None, :] - centers[None, :, :], axis=2)
        r1 = radii_all[rows][:, None]
        rq = radii_all[None, :]
        ok = (d - r1 - rq <= _gap_cap(r1, rq)) & (rq >= r_min_pair) & (r1 >= r_min_pair)
        for a in range(len(rows)):  # exclude the pore itself
            ok[a, rows[a]] = False
        return ok

    # Part 1: top pores vs everything (chunked to bound memory); both directions.
    for s0 in range(0, k_top, 64):
        rows = top_idx[s0 : s0 + 64]
        ok = _valid_mask(rows)
        for a, i in enumerate(rows.tolist()):
            js = np.flatnonzero(ok[a])
            partner_sets[i].update(js.tolist())
            for j in js.tolist():
                partner_sets[j].add(int(i))

    # Part 2: non-top hostable pores vs their (exact) ball of radius r_i + (1+f_gap)*max(r_cut, r_floor).
    if k_top < N:
        rest_idx = np.flatnonzero(~is_top)
        q_extra = (1.0 + f_gap) * max(r_cut, r_floor)
        for i in rest_idx.tolist():
            ri = radii_all[i]
            if ri < r_min_pair:  # cannot host any prescribed throat; its list stays empty
                continue
            cand = np.asarray(tree.query_ball_point(centers[i], ri + q_extra), dtype=np.int64)
            if len(cand):
                d = np.linalg.norm(centers[cand] - centers[i], axis=1)
                rq = radii_all[cand]
                # cand includes i itself (distance 0 passes every filter) — exclude it.
                ok = (cand != i) & (d - ri - rq <= _gap_cap(ri, rq)) & (rq >= r_min_pair) & ~is_top[cand]
                partner_sets[i].update(cand[ok].tolist())

    # Sorted: canonical order so equal-gap tie-breaking is deterministic (sets are not sequences
    # to np.asarray, and their iteration order must not influence results).
    partner_lists = [np.sort(np.asarray(list(s), dtype=np.int64)) for s in partner_sets]
    n_pair_capacity = sum(len(s) for s in partner_sets) // 2
    # Per-pore partner list length (used for diagnostics and the pair-assignment loop).
    list_len = np.fromiter((len(pl) for pl in partner_lists), dtype=np.int64, count=N)

    rmid = 0.5 * (bins[:-1] + bins[1:])
    # Per-bin weights for the count conversion: w_i ∝ volfrac_i / (Rmid_i^2 * Lbar_i).
    # The initial value uses the self-similar model Lbar_i = Rmid_i (i.e. N_i ∝ f_i/Rmid_i^3)
    # as a fallback only; calibrated passes replace Lbar_i by the MEASURED mean length of
    # the previous pass for every bin that had at least one placement there.
    w = volfrac / rmid**3

    initial_edges = set(net.graph.edges())

    def _round_counts(weights, total):
        """Largest-remainder rounding so the total is exactly ``total`` (the placeable share)."""
        wsum = float(weights.sum())
        if wsum <= 0 or total <= 0:
            return np.zeros(n_bins, dtype=np.int64)
        exact = total * weights / wsum
        counts = np.floor(exact).astype(np.int64)
        diff = int(total - counts.sum())
        if diff > 0:
            for i in np.argsort(-exact + counts)[:diff]:
                counts[i] += 1
        return counts

    def _reset_pass_state():
        """Remove the edges added by previous passes and rebuild the mutable state."""
        added = [e for e in net.graph.edges() if e not in initial_edges]
        if added:
            net.graph.remove_edges_from(added)
        budget_rem[:] = budget_max
        deg[:] = np.fromiter((net.graph.degree(n) for n in nodelist), dtype=np.int64, count=N)
        for i in range(N):
            neighbors_pos[i].clear()
        for i, n in enumerate(nodelist):
            for nb in net.graph[n]:
                neighbors_pos[i].add(pos_of[nb])

    # Pores sorted by ascending radius: the set of pores with radius >= thr is the
    # suffix from searchsorted(thr); it only GROWS as thr decreases (we process
    # prescribed radii in descending order).
    order_asc = np.argsort(radii_all, kind="stable")
    radii_asc = radii_all[order_asc]

    # Structurally unplaceable bins: fewer than 2 pores with radius >= PRTRR * (bin LOWER edge),
    # i.e. not even the smallest throat in the bin can be hosted at this network size. They get
    # zero prescribed throats; their target volume is NOT redistributed (achieved stays 0 there).
    n_large_lower = N - np.searchsorted(radii_asc, PRTRR * bins[:-1], side="left")
    unplaceable_bins = [int(i) for i in range(n_bins) if n_large_lower[i] < 2]

    # Per-bin structural capacity: a throat of bin i needs BOTH endpoints among the pores with
    # radius >= PRTRR * bins[i] (the LOWER edge gives the largest eligible set within the bin),
    # and each pore pair can host at most one throat. So at most C(n_large_lower[i], 2) throats
    # of bin i can ever be placed, whatever the gap cap or surface budgets allow. Prescribing
    # more than that only creates guaranteed misses (each burning up to max_retries + 1 partner
    # searches). The caps are per-bin upper bounds: pairs shared between bins (a large-large pair
    # also belongs to every smaller bin's eligible set) are consumed in descending radius order.
    bin_capacity = np.where(n_large_lower >= 2, (n_large_lower * (n_large_lower - 1)) // 2, 0).astype(np.int64)

    def _run_pass(counts, pass_rng):
        """Place one full pass of prescribed throats via direct pair assignment.

        Instead of the old "for each throat, search for a pair" approach (which suffers from a
        parking-problem at high utilization when small pores have only ~2 gap-valid partners),
        this iterates over ALL precomputed gap-valid pairs and assigns each to the highest bin
        that still has demand and whose constraints (eligibility, budget, degree) are satisfied.
        Each pair is considered exactly once — no wasted retries, no parking problem.

        Returns (placed, placed_per_bin, misses_per_bin, n_insufficient_large_pores,
        vol_placed, len_sum) for the throats added by THIS pass only.
        """
        # Build the flat list of all gap-valid pairs (i < j to avoid duplicates), with each pair's
        # maximum hostable throat radius: min(r_i, r_j) / PRTRR. A pair can serve any bin b where
        # bins[b+1] <= max_hostable (both endpoints are large enough for the LARGEST throat in b).
        pair_i = []
        pair_j = []
        pair_max_rc = []
        for i in range(N):
            pl = partner_lists[i]
            if len(pl) == 0:
                continue
            ri = radii_all[i]
            for j in pl.tolist():
                if j > i:  # each undirected pair appears once
                    rj = radii_all[j]
                    pair_i.append(i)
                    pair_j.append(j)
                    pair_max_rc.append(min(ri, rj) / PRTRR)

        n_pairs = len(pair_i)
        if n_pairs == 0:
            return (
                0,
                np.zeros(n_bins, dtype=np.int64),
                counts.astype(np.int64).copy(),
                0,
                np.zeros(n_bins),
                np.zeros(n_bins),
            )

        pair_i = np.asarray(pair_i, dtype=np.int64)
        pair_j = np.asarray(pair_j, dtype=np.int64)
        pair_max_rc = np.asarray(pair_max_rc, dtype=float)

        # Process pairs in ASCENDING max-hostable-radius order: small-capacity pairs first, so
        # that small throats are placed while all surface budgets are still fresh. Without this,
        # medium/large throats (processed earlier in descending or random order) consume the
        # limited budget of their small-pore endpoints before small throats can use them — a pore
        # with r=0.01 has budget for only ~8-12 small throats, and 3 medium throats already
        # exhaust it. Large-capacity pairs are processed later; they get assigned to large bins
        # (which need their specific high-capacity pairs) or fall through to smaller bins if those
        # are still unsatisfied. Each pair is always assigned to its HIGHEST available bin, so
        # large bins retain priority for the few pairs that can serve them.
        pair_order = np.argsort(pair_max_rc)

        # Per-bin remaining demand (how many more throats each bin still needs).
        remaining = counts.astype(np.int64).copy()
        n_total_demand = int(remaining.sum())

        placed_p = 0
        placed_pb = np.zeros(n_bins, dtype=np.int64)
        vol_p = np.zeros(n_bins)
        lensum_p = np.zeros(n_bins)
        n_insuff = 0
        progress_step = max(1, n_pairs // 100)

        # For each pair (in ascending capacity order), assign it to the highest bin that:
        #   (a) still has demand (remaining[b] > 0),
        #   (b) can be served by this pair (bins[b+1] <= pair_max_rc, i.e. both endpoints large enough),
        #   (c) passes budget and degree checks for the drawn throat radius.
        for pi in range(n_pairs):
            if verbose and (pi + 1) % progress_step == 0:
                print(
                    "Constrained pairing: {:5.2f}% completed  ".format(100.0 * (pi + 1) / n_pairs), end="\r", flush=True
                )

            idx = int(pair_order[pi])
            i = int(pair_i[idx])
            j = int(pair_j[idx])
            max_rc = float(pair_max_rc[idx])

            # Highest bin this pair can serve: bins[b+1] <= max_rc  =>  b <= searchsorted(bins, max_rc) - 2
            b_hi = int(np.searchsorted(bins, max_rc, side="right")) - 2
            if b_hi < 0:
                continue  # pair cannot serve any bin (both endpoints too small for even the smallest throat)
            b_hi = min(b_hi, n_bins - 1)

            assigned = False
            for b in range(b_hi, -1, -1):
                if remaining[b] <= 0:
                    continue
                # Draw a random radius within this bin.
                Rc = float(pass_rng.uniform(bins[b], bins[b + 1]))
                sec = np.pi * Rc**2
                thr = PRTRR * Rc

                # Eligibility is guaranteed by construction (max_rc >= bins[b+1] > Rc), but verify
                # the actual drawn threshold against both endpoints (handles edge cases at bin boundaries).
                if radii_all[i] < thr or radii_all[j] < thr:
                    continue  # try next smaller bin for this pair

                # Budget and degree checks.
                if budget_rem[i] < sec or budget_rem[j] < sec:
                    continue
                if deg[i] >= maxc or deg[j] >= maxc:
                    continue

                # Place the throat.
                L = float(np.linalg.norm(centers[i] - centers[j]) - radii_all[i] - radii_all[j])
                net.add_throat(nodelist[i], nodelist[j], radius=Rc)
                budget_rem[i] -= sec
                budget_rem[j] -= sec
                deg[i] += 1
                deg[j] += 1
                neighbors_pos[i].add(j)
                neighbors_pos[j].add(i)

                Lmax = max(L, 0.0)
                vol_p[b] += np.pi * Rc**2 * Lmax
                lensum_p[b] += Lmax
                placed_pb[b] += 1
                remaining[b] -= 1
                placed_p += 1
                assigned = True
                break

            if not assigned:
                # This pair could not be used by any bin (budget/degree exhausted for all eligible bins).
                pass

        # Unmet demand becomes misses.
        misses_pb = remaining.copy()
        # Count "insufficient large pores" misses: bins where n_large < 2 at the drawn radius.
        # In this approach, those are simply bins with very few eligible pairs — already reflected
        # in their miss count. We report 0 here since the distinction is less meaningful now.

        return placed_p, placed_pb, misses_pb, n_insuff, vol_p, lensum_p

    # ------------------------------------------------------------------
    # Passes. Pass 1 is a PROBE pass (n_probe throats per placeable bin) that measures
    # the actual mean throat length and placement success rate in every bin; each
    # following pass re-weights the per-bin counts from the measurements of the previous
    # pass:
    #     w_i <- volfrac_i / (Rmid_i^2 * Lbar_i) / s_i
    # so that the expected placed volume per bin matches volfrac. Bins with no placement
    # in the previous pass keep their previous weight (no measurement available).
    # ------------------------------------------------------------------
    placeable = np.ones(n_bins, dtype=bool)
    if unplaceable_bins:
        placeable[np.asarray(unplaceable_bins)] = False
    # No redistribution: only the placeable share of the target volume is prescribed. Unplaceable
    # bins get zero throats and their volume is NOT absorbed by the others (which would over-fill
    # the small bins); achieved_volfrac simply stays 0 there — an honest gap, not a distortion.
    w[~placeable] = 0.0
    placeable_frac = float(volfrac[placeable].sum())
    n_effective_uncapped = int(round(n_throats * placeable_frac))
    # Cap the demand at a fraction of the EXACT gap-valid pair count: prescribing more throats
    # than there are such pairs only creates misses (each burning up to max_retries + 1 partner
    # searches). The uncapped remainder is simply NOT prescribed — no redistribution; since
    # achieved_volfrac is normalized over the placed volume, capping scales all bins down
    # proportionally without distorting their relative shares.
    # Cap the demand at a fraction of the EXACT gap-valid pair count. The fraction is 25% (not 90%)
    # because the greedy pair-assignment algorithm cannot efficiently use all available pairs: small
    # pores have only ~2 gap-valid partners each, so once those are consumed the pore is "dead" for
    # further assignments. At 90% utilization the success rate drops to ~20%; at 25% it stays >80%.
    # The achieved distribution shape is preserved (all bins scale down proportionally). Users who
    # need more total throats can increase max_gap_factor (more candidate pairs) or n_throats.
    n_effective = min(n_effective_uncapped, int(0.25 * n_pair_capacity))
    if n_effective < n_effective_uncapped:
        print(
            "Pair capacity {:d} gap-valid pairs -> prescribed demand capped to {:d} (was {:d})".format(
                n_pair_capacity, n_effective, n_effective_uncapped
            )
        )

    nonzero = volfrac > 0
    # Probe counts are capped by the per-bin structural capacity (a bin with few eligible pores
    # cannot place more than C(n_elig, 2) throats; probing beyond that only wastes retries).
    counts = np.where(placeable & nonzero, int(max(1, n_probe)), 0).astype(np.int64)
    counts = np.minimum(counts, bin_capacity)
    if int(counts.sum()) == 0:
        warn("add_throats_by_distribution: nothing to place (n_throats = {}).".format(n_throats))
        return None

    placed, placed_per_bin, misses_per_bin, n_insufficient_large_pores, vol_placed, len_sum = (None,) * 6
    for k in range(n_iterations):
        if k > 0:
            # Re-weight from the measurements of the previous pass.
            with np.errstate(divide="ignore", invalid="ignore"):
                lbar_prev = len_sum / np.maximum(placed_per_bin, 1)
                success_prev = placed_per_bin / np.maximum(placed_per_bin + misses_per_bin, 1)
            mask = (placed_per_bin > 0) & placeable
            w_new = w.copy()
            w_new[mask] = volfrac[mask] / (rmid[mask] ** 2 * lbar_prev[mask]) / success_prev[mask]
            # Keep unplaceable bins at zero weight so they never receive throats (no redistribution).
            w_new[~placeable] = 0.0
            w = w_new
            counts = _round_counts(w, n_effective)
            # Cap by structural capacity: prescribing beyond C(n_elig, 2) per bin would only create
            # guaranteed misses (each burning up to max_retries + 1 partner searches). The uncapped
            # remainder is simply NOT prescribed — no redistribution onto the other bins.
            counts = np.minimum(counts, bin_capacity)
            if int(counts.sum()) == 0:
                warn(
                    "add_throats_by_distribution: calibrated pass empty after capacity capping; "
                    "keeping the probe-pass throats."
                )
                break
            _reset_pass_state()

        if verbose and n_iterations > 1:
            print(
                "Constrained pairing, pass {}/{} ({}):".format(k + 1, n_iterations, "probe" if k == 0 else "calibrated")
            )

        placed, placed_per_bin, misses_per_bin, n_insufficient_large_pores, vol_placed, len_sum = _run_pass(
            counts, pass_rngs[k]
        )

    if verbose:
        print("\rConstrained pairing: {0:5.2f}% completed   ".format(100.0), end="\r", flush=True)

    # ------------------------------------------------------------------
    # Repair pass 1: connect the different connected components (if any).
    # Throats added here use radius = min(r1, r2)/PRTRR and are NOT part of the
    # prescribed distribution; budgets/maxc are not enforced in this pass.
    # ------------------------------------------------------------------
    n_conn_added = 0
    if ensure_connected:

        def _connect_components():
            added_total = 0
            for _round in range(100):  # safety cap (converges in a few rounds)
                comps = list(nx.connected_components(net.graph))
                if len(comps) <= 1:
                    break
                comp_id_of_label = {}
                for ci, comp in enumerate(comps):
                    for nlab in comp:
                        comp_id_of_label[nlab] = ci
                comp_id = np.fromiter((comp_id_of_label[n] for n in nodelist), dtype=np.int32, count=N)

                added_round = 0
                for k in (5, 20, 100):
                    kk = min(k + 1, N)
                    _, idxm = tree.query(centers, k=kk)  # (N, kk) positions, includes self
                    diff_mask = comp_id[idxm] != comp_id[:, None]
                    rows = np.flatnonzero(diff_mask.any(axis=1))
                    if len(rows):
                        col = np.argmax(diff_mask[rows], axis=1)  # first cross-component neighbor
                        for i, jj in zip(rows.tolist(), col.tolist()):
                            j = int(idxm[i, jj])
                            if not net.graph.has_edge(nodelist[i], nodelist[j]):
                                r12 = min(radii_all[i], radii_all[j]) / PRTRR
                                net.add_throat(nodelist[i], nodelist[j], radius=r12)
                                deg[i] += 1
                                deg[j] += 1
                                neighbors_pos[i].add(j)
                                neighbors_pos[j].add(i)
                                added_round += 1
                    if added_round:
                        break

                added_total += added_round
                if not added_round:
                    warn("Could not connect all components ({} remain).".format(len(comps)))
                    break
            return added_total

        n_conn_added = _connect_components()

    # ------------------------------------------------------------------
    # Repair pass 2: enforce a minimum connectivity.
    # ------------------------------------------------------------------
    n_minc_added = 0
    if minc > 0:
        for i in range(N):
            while deg[i] < minc:
                found = False
                for k in (10, 50, 200):
                    kk = min(k + 1, N)
                    _, idxm = tree.query(centers[i], k=kk)
                    for jj in idxm.tolist():
                        j = int(jj)
                        if j != i and j not in neighbors_pos[i]:
                            r12 = min(radii_all[i], radii_all[j]) / PRTRR
                            net.add_throat(nodelist[i], nodelist[j], radius=r12)
                            deg[i] += 1
                            deg[j] += 1
                            neighbors_pos[i].add(j)
                            neighbors_pos[j].add(i)
                            n_minc_added += 1
                            found = True
                            break
                    if found:
                        break
                if not found:
                    warn("Pore {} could not reach minc={} (no unconnected pore).".format(nodelist[i], minc))
                    break

    # ------------------------------------------------------------------
    # Write back the covered surface areas and report.
    # ------------------------------------------------------------------
    nx.set_node_attributes(net.graph, dict(zip(nodelist, budget_max - budget_rem)), name="exchange_area")
    nx.set_node_attributes(net.graph, dict(zip(nodelist, sr_vals)), name="surface_ratio")

    vol_tot = float(vol_placed.sum())
    achieved_frac = vol_placed / vol_tot if vol_tot > 0 else np.zeros(n_bins)
    max_diff = float(np.max(np.abs(achieved_frac - volfrac)))

    # Mean actual length per bin (inf where nothing was placed).
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_len = np.where(placed_per_bin > 0, len_sum / np.maximum(placed_per_bin, 1), np.inf)

    print(
        "Constrained pairing: {} prescribed throats placed, {} misses ({} with <2 eligible pores)".format(
            placed, int(misses_per_bin.sum()), n_insufficient_large_pores
        )
    )
    if unplaceable_bins:
        print(
            "Unplaceable bins (<2 pores can host even the smallest throat): {} — left empty, NOT redistributed".format(
                len(unplaceable_bins)
            )
        )
    if ensure_connected or minc > 0:
        print("Repair passes: {} connectivity throats, {} minc throats added".format(n_conn_added, n_minc_added))
    print("Achieved vs target volumetric distribution (prescribed throats): max |diff| = {:.4f}".format(max_diff))

    return {
        "n_placed": placed,
        "placed_per_bin": placed_per_bin.tolist(),
        "n_misses": int(misses_per_bin.sum()),
        "misses_per_bin": misses_per_bin.tolist(),
        "n_insufficient_large_pores": n_insufficient_large_pores,
        "unplaceable_bins": unplaceable_bins,
        "n_pair_capacity": n_pair_capacity,
        "n_connectivity_added": n_conn_added,
        "n_minc_added": n_minc_added,
        "achieved_volfrac": achieved_frac.tolist(),
        "mean_length_per_bin": mean_len.tolist(),
    }


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
