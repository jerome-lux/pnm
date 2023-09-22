# coding=utf-8

import networkx as nx
import numpy as np
from warnings import warn
from pnm import pore_network, nearest_neighbor, isinside, isinside_surface
from heapq import heappush, heappop, heapify
from sklearn.neighbors import KDTree

RATIO_REDUC = 0.5
EPS = 1.00000001


def invasion_percolation(pn, target_porosity, face='x-', box=None, verbose=False, keep_connected=True):
    """Delete pores using an invasion percolation process.
    pn: pore network instance
    face: where to begin the invasion
    target_porosity: the invasion stops when the porosity of the computing volume reach the target
    Note that the porosity is only approximate as t
    box: subvolume (6 ints)
    keep_connected: if True, all remaining pores are connected to the network
    """


    pn.graph = nx.convert_node_labels_to_integers(pn.graph)

    # centers
    c = np.array(
        list(zip(*nx.get_node_attributes(pn.graph, 'center').values()))).T
    # radii
    radii = np.array(list(nx.get_node_attributes(pn.graph, 'radius').values()))

    if box is not None:
        box[0] = max(box[0], pn.graph.graph['bbox'][0])
        box[1] = max(box[1], pn.graph.graph['bbox'][1])
        box[2] = max(box[2], pn.graph.graph['bbox'][2])
        box[3] = min(box[3], pn.graph.graph['bbox'][3])
        box[4] = min(box[4], pn.graph.graph['bbox'][4])
        box[5] = min(box[5], pn.graph.graph['bbox'][5])
        active_pores = np.where((c[..., 0] >= box[0]) & (c[..., 0] <= box[1]) &
                                (c[..., 1] >= box[2]) & (c[..., 1] <= box[3]) &
                                (c[..., 2] >= box[4]) & (c[..., 2] <= box[5]))[0]

        subgraph = pn.graph.subgraph(active_pores)

        vv_ini = np.array(list(nx.get_node_attributes(pn.graph, 'volume').values()))[
            active_pores].sum()
        for n1, n2 in subgraph.edges():
            vv_ini += subgraph[n1][n2]['volume']
        vbox = (box[1]-box[0]) * (box[3]-box[2]) * (box[5]-box[4])
        pini = vv_ini / vbox

    else:
        active_pores = np.array(pn.graph.nodes())

        subgraph = pn.graph
        vv_ini = np.array(list(nx.get_node_attributes(
            pn.graph, 'volume').values())).sum()
        for n1, n2 in subgraph.edges():
            vv_ini += subgraph[n1][n2]['volume']

        vbox = np.prod(pn.graph.graph['extent'])
        pini = vv_ini / vbox

    print('Box volume = {}, volume = {}'.format(vbox, vv_ini))
    print('Number of pores in box:', len(active_pores))

    if pini <= target_porosity:
        print('Initial porosity = {} <= target porosity. Nothing to do'.format(pini))
        return

    # nodes on the starting face
    starting_face_nodes = pn.get_pores_by_category(
        category=face, mode='match_one', nodes=active_pores)
    if len(starting_face_nodes) <= 0:
        print("Pores on face {} are not inside the provided box. Quit".format(face))
        return

    # Store potential pores to invade. Ordered by radius [(r,ind)]
    n0 = starting_face_nodes[np.argmin(radii[starting_face_nodes])]
    stored_nodes = [(radii[n0], n0)]
    marked = {i: 0 for i in subgraph.nodes()}
    marked[n0] = 1
    invaded = []
    heapify(stored_nodes)
    pf = pini
    vv = vv_ini
    art = []
    while pf > target_porosity and stored_nodes:

        if keep_connected:
            art = list(nx.articulation_points(pn.graph))

        current_node = heappop(stored_nodes)[1]
        if (current_node not in art) or not keep_connected:
            invaded.append(current_node)
            # Update volume and heap queue
            vol = pn[current_node]['volume']
            for node in [n for n in subgraph.neighbors(current_node) if marked[n] == 0]:
                vol += pn.graph[current_node][node]['volume']
                marked[node] = 1
                heappush(stored_nodes,
                         (subgraph[current_node][node]['radius'], node))
            vv -= vol
            pf = vv / vbox
            pn.remove_pore(current_node, verbose=verbose)
            if verbose:
                print("Pore {} deleted (total {}), new box porosity = {} ".format(
                    current_node, len(invaded), pf))

    pn.compute_geometry()
    print("{} pores removed."
          "\n {} remaining."
          "\n final porosity of the whole network: {} "
          "\nPorosity of the box: {}".format(len(invaded), pn.graph.number_of_nodes(), pn.graph.graph['porosity'], pf))


def assemble_pnm(pn1, pn2, axis='x', ratio=1, minc=3, maxc=np.inf, BCgridsize=100, TRR=1):
    """Merge two pore networks. Return a single pnm instance.
    The 2 pnm are assembled in the following manneer:
    the pn2 coords are increased by the pn1 extent in the direction given by the axis argument
    TODO: change extent to bbox
    """

    i = {"x": 0, "y": 1, "z": 2}[axis]

    if pn1.graph.graph['graph_type'] == 'cubic_regular':
        centers = np.array(
            list(zip(*nx.get_node_attributes(pn1.graph, 'center').values()))).T
        offset = centers[:, i].max() + pn1.graph.graph['spacing'][i]*0.5
    else:
        offset = pn1.graph.graph['bbox'][i+3]

    c = np.array(
        list(zip(*nx.get_node_attributes(pn2.graph, 'center').values()))).T
    c[:, i] += (offset - pn2.graph.graph['bbox'][i])
    nx.set_node_attributes(pn2.graph, dict(
        zip(list(pn2.graph.nodes), list(c))), name='center')

    newgraph = nx.disjoint_union(pn1.graph, pn2.graph)

    newpnm = pore_network(newgraph)

    # Get nodes to connect
    cat1 = {"x": "x+", "y": "y+", "z": "z+"}
    cat2 = {"x": "x-", "y": "y-", "z": "z-"}

    nodes_to_connect1 = pn1.get_pores_by_category(cat1[axis])
    nodes_to_connect2 = np.array(pn2.get_pores_by_category(
        cat2[axis])) + pn1.graph.number_of_nodes()

    # Recompute BC nodes
    newpnm.clear_category()
    SetFaceBCnodes(newpnm, step=BCgridsize)

    if pn1.graph.graph['graph_type'] == 'cubic_regular' or pn2.graph.graph['graph_type'] == 'cubic_regular':
        connect_pores(newpnm, nodes_to_connect1, nodes_to_connect2)

    else:
        nodes_to_connect1.extend(nodes_to_connect2)
        # Removing existing throats from old boundary nodes
        for n in nodes_to_connect1:
            ebunch = list(newpnm.graph.edges(n))
            newpnm.graph.remove_edges_from(ebunch)

        # Adding new throats to connect the 2 networks
        add_throats_by_surf(newpnm, nodes=nodes_to_connect1, ratio=ratio,
                            minc=minc, maxc=maxc, sort_by_radius=False, TRR=TRR)

    # Update geometry
    newpnm.compute_geometry(autothroats=True)

    return newpnm


def connect_pores(pn, nbunch1, nbunch2):
    """Connect pores in nbunch1 to nearest pore in nbunch2
    """

    centers = nx.get_node_attributes(pn.graph, 'center')
    centers1 = np.array([np.array(centers[i]) for i in nbunch1])
    centers2 = np.array([np.array(centers[i]) for i in nbunch2])

    indices = nearest_neighbor(centers1, centers2)

    for i in range(indices.shape[0]):
        pn.add_throat(nbunch1[i], nbunch2[indices[i]])


def add_throats(net, lowc=0, highc=np.inf, minc=0, maxc=np.inf, mode='auto', sort_by_radius=True, spacing='auto'):
    """Set throats between nth-nearest pore neighbors
    \n lowc: min target connectivity
    \n highc: max connectivity
    \n minc: min enforced connectivity
    \n if mode='auto', lowc and highc are scaled by the pore radius
    \n Note that it is assumed that there are no throats in the provided network"""

    #from scipy.spatial import cKDTree
    #from scipy.spatial.distance import cdist


    if highc > maxc:
        highc = maxc

    if lowc > highc:
        lowc = highc

    if minc is None:
        minc = lowc

    elif minc > lowc:
        minc = lowc

    nodelist = list(net.graph.nodes)
    pores_radii = np.array(
        list(nx.get_node_attributes(net.graph, 'radius').values()))

    max_pore_radius = pores_radii.max()

    # Pour ne traiter que les pores de rayon > rlimit
    #i = np.argmax(pores_radii<rmax)
    #j = np.argmin(pores_radii>rmin)
    # if i>0:
    #pores_radii = pores_radii[i:j]
    #nodelist = nodelist[i:j]

    if spacing == 'auto':
        try:
            spacing = net.graph.graph['spacing'].min()
        except KeyError:
            # Moyenne arithmétique des rayons de pore
            spacing = pores_radii.sum() / pores_radii.size

    if len(nodelist) <= 0:
        return
    centers = nx.get_node_attributes(net.graph, 'center')
    centers = np.array([np.array(centers[i]) for i in nodelist])
    #kdtree = cKDTree(centers)
    kdtree = KDTree(centers, leaf_size=40)

    #candidates = [n for n in nodelist if net.graph.degree(n)<minc]
    candidates = [(n, net.graph.nodes[n]['radius'])
                  for n in nodelist if net.graph.degree(n) < minc]
    # On traite les plus gros pores en premier
    candidates.sort(key=lambda t: t[1], reverse=True)
    candidates = [t[0] for t in candidates]

    for n1 in candidates:

        if mode == 'auto':
            m = (net.graph.nodes[n1]['radius'] / spacing)
            high2 = min(int(np.around(max(highc, highc * m))), maxc)
            low2 = min(int(np.around(max(lowc, lowc * m))), maxc)
        else:
            high2 = highc
            low2 = lowc

        #indices = kdtree.query_ball_point(net.graph.nodes[n1]['center'],(max_pore_radius + net.graph.nodes[n1]['radius']) * 1.5)
        indices, distances = kdtree.query_radius([net.graph.nodes[n1]['center']], (
            max_pore_radius + net.graph.nodes[n1]['radius']) * 1.5, return_distance=True)
        if len(indices[0]) < low2*2:
            # _, indices =  kdtree.query(x=net.graph.nodes[n1]['center'],k=high2*2,n_jobs=1)  #sinon, on cherche les n premiers voisins
            distances, indices = kdtree.query([net.graph.nodes[n1]['center']], k=min(
                high2*2, len(nodelist)), return_distance=True)

        indices = indices[0][:len(centers)]
        degree = np.random.randint(low2, high2+1) - net.graph.degree(n1)

        if degree <= 0 or len(indices) <= 0:
            continue

        #indices = indices[:min(len(indices),len(nodelist))]
        # On tri par taille de pore pour favoriser les connections entre "gros" pores
        # indices = [(n,net.graph.nodes[nodelist[n]]['radius']) for n in indices]
        # indices.sort(key = lambda t:t[1],reverse=True)
        # indices = [t[0] for t in indices]

        # On cherche les pores les plus proches (i.e. la distance entre les deux surfaces la plus faible !)

        #distances = cdist(centers[indices],[net.graph.nodes[n1]['center']])[:,0] - pores_radii[indices]
        distances = distances[0] - pores_radii[indices] - \
            net.graph.nodes[n1]['radius']
        #distances -= net.graph.nodes[n1]['radius']
        dist_and_labels = [(distances[j], k) for j, k in enumerate(indices)]
        dist_and_labels.sort(key=lambda t: t[0])
        # On ne garde que les high2*2 premiers pores
        dist_and_labels = dist_and_labels[:min(high2*2, len(dist_and_labels))]
        dist_and_labels = [(dist, pores_radii[k], k)
                           for dist, k in dist_and_labels]

        if sort_by_radius:  # On tri par taille de pore pour favoriser les connections entre "gros" pores
            dist_and_labels.sort(key=lambda t: t[1], reverse=True)
        nthroats = 0

        for (dist, r, j) in dist_and_labels:
            n2 = nodelist[j]

            if n2 != n1:
                if mode == 'auto' and net.graph.graph.get('spacing', None) is not None:
                    m = (net.graph.nodes[n2]['radius'] / spacing)
                    high3 = min(int(np.around(max(highc, highc * m))), maxc)
                else:
                    high3 = highc

                if dist < 0:
                    warn("Overlapping detected ! Pore {}, R = {}  and {}, R = {}, distance = {}".format(
                        n1, net.graph.nodes[n1]['radius'], n2, net.graph.nodes[n2]['radius'], dist))
                if nthroats >= degree:
                    break
                if net.graph.degree(n2) < high3:
                    nthroats += 1
                    net.graph.add_edge(n1, n2, radius=min(
                        net.graph.nodes[n1]['radius'], net.graph.nodes[n2]['radius']))

        # Si la connectivité est toujours < minc, on ajoute un capillaire même si cela implique un dépassement de la connectivité max pour le pore voisin
        if net.graph.degree(n1) < minc:
            # On traite ici uniquement les voisins proches
            indices = kdtree.query([net.graph.nodes[n1]['center']], k=min(
                high2*2, len(nodelist)), return_distance=False)
            #_,indices = kdtree.query(x=net.graph.nodes[n1]['center'],k=high2*2,n_jobs=1)
            # neighbors = [(net.graph.degree(nodelist[n]),n) for n in indices if nodelist[n] != n1]
            # neighbors.sort(key=lambda t:t[0])

            neighbors = [(net.graph.nodes[nodelist[n]]['radius'], n)
                         for n in indices[0] if nodelist[n] != n1 if net.graph.has_node(nodelist[n])]
            if (len(neighbors) == 0):
                break
            neighbors.sort(key=lambda t: t[0], reverse=True)
            for i in range(minc-net.graph.degree(n1)):
                n2 = nodelist[neighbors[i][1]]
                net.graph.add_edge(n1, n2, radius=min(
                    net.graph.nodes[n1]['radius'], net.graph.nodes[n2]['radius']))



def add_periodic_throats(net, delta, faces=["x", "y", "z"], ratio=0.75, TRR=1, minc=0, maxc=np.inf):

    """Add throats between opposed BC pores.
    net: pore network model instance
    delta: min length of a throat [should be fixed according to the network's extent]
    faces: faces to be connected ["x" connects faces "x+" and "x-"]
    """

    pass


def add_throats_by_surf(net, nodes=None, ratio=0.75, TRR=1, minc=0, maxc=1000, sort_by_radius=False, border_correction=True):

    """Set throats between nearest pore neighbors such that the  pore surface / connected throat section surface <= ratio
    nodes: list of nodes to process, if None, all nodes will be processed
    ratio: pore area over total area of the connected throats
    TRR: throat radius ratio: radius between the throat radius and the min of the adjacent pores radii
    minc: min connectiviy
    maxc: max connectivity
    sort_by_radius: bool (default False). If True, neighbors are sorted by radius and not by distance [not recommended]
    border_correction: bool (default True). Reduce connecivity (i.e. ratio) of pores at the boundaries """


    if maxc < minc:
        maxc = minc+1

    nodelist = list(net.graph.nodes)

    if len(nodelist) <= 0:
        return

    if nodes is None:
        nodes = nodelist

    pores_radii = np.array(
        list(nx.get_node_attributes(net.graph, 'radius').values()))

    nx.set_node_attributes(net.graph, 0, name='exchange_area')
    max_pore_radius = pores_radii.max()

    centers = nx.get_node_attributes(net.graph, 'center')
    centers = np.array([np.array(centers[i]) for i in nodelist])

    kdtree = KDTree(centers, leaf_size=40)

    candidates = [(n, net.graph.nodes[n]['radius'])
                  for n in nodes if net.graph.degree(n) < minc]
    # On traite les plus gros pores en premier
    candidates.sort(key=lambda t: t[1], reverse=True)
    candidates = [t[0] for t in candidates]

    if border_correction:
        def corrected_ratio(n):
            innercat = set(["inner", "none"])
            if len(innercat.intersection(net[n]['category'])) > 0:
                return ratio
            else:
                return ratio*RATIO_REDUC
    else:
        def corrected_ratio(n):
            return ratio

    print(len(candidates), "nodes to process")

    for counter, n1 in enumerate(candidates):

        print("{:5.2f}% completed  ".format(
            100*(counter+1)/len(candidates)), end="\r")
        # the query_radius() and query() methods return (ind, dist) and (dist, ind) respectively...
        indices, distances = kdtree.query_radius([net.graph.nodes[n1]['center']], (
            max_pore_radius + net.graph.nodes[n1]['radius']) * 1.5, return_distance=True)
        if len(indices[0]) < maxc*2:
            # sinon, on cherche les n premiers voisins
            distances, indices = kdtree.query([net.graph.nodes[n1]['center']], k=min(
                maxc*2, len(nodelist)), return_distance=True)
        indices = indices[0]
        # On cherche les pores les plus proches (i.e. la distance entre les deux surfaces la plus faible !)

        distances = distances[0] - pores_radii[indices] - \
            net.graph.nodes[n1]['radius']
        dist_and_labels = [(distances[j], k) for j, k in enumerate(indices)]
        dist_and_labels.sort(key=lambda t: t[0])
        dist_and_labels = [(dist, pores_radii[k], k)
                           for dist, k in dist_and_labels]

        if sort_by_radius:  # On tri par taille de pore pour favoriser les connections entre "gros" pores
            dist_and_labels.sort(key=lambda t: t[1], reverse=True)

        for (dist, r, j) in dist_and_labels:

            n2 = nodelist[j]

            if n2 != n1:

                if dist < 0:
                    warn("Overlapping detected ! Pore {}, R = {}  and {}, R = {}, distance = {}".format(
                        n1, net.graph.nodes[n1]['radius'], n2, net.graph.nodes[n2]['radius'], dist))

                if net.graph.degree(n2) > maxc or net.graph.degree(n1) > maxc:
                    continue

                r1 = net.graph.nodes[n1]['radius']
                r2 = net.graph.nodes[n2]['radius']
                r12 = TRR*min(r1, r2)
                S12 = np.pi*r12**2
                S1 = net.graph.nodes[n1]['exchange_area'] + S12
                S2 = net.graph.nodes[n2]['exchange_area'] + S12

                if S1 <= corrected_ratio(n1) * 4 * np.pi * r1**2 and S2 <= corrected_ratio(n2) * 4 * np.pi * r2**2:
                    net.add_throat(n1, n2, radius=r12)
                    net.graph.nodes[n1]['exchange_area'] += S12
                    net.graph.nodes[n2]['exchange_area'] += S12

                # Force a min connectivity even if covered area > max area
                elif S1 > corrected_ratio(n1) * 4 * np.pi * r1**2 and net.graph.degree(n1) < minc:
                    if S2 <= corrected_ratio(n2) * 4 * np.pi * r2**2:
                        net.add_throat(n1, n2, radius=r12)
                        net.graph.nodes[n1]['exchange_area'] += S12
                        net.graph.nodes[n2]['exchange_area'] += S12
                    else:
                        continue

                elif S1 > corrected_ratio(n1) * 4 * np.pi * r1**2 and net.graph.degree(n1) >= minc:
                    break

        # S'il reste des pores avec une connectivité < minc malgré tout, on force
        if net.graph.degree(n1) < minc:

            for (dist, r, j) in dist_and_labels:

                n2 = nodelist[j]

                if n2 != n1 and not net.graph.has_edge(n1, n2):
                    r1 = net.graph.nodes[n1]['radius']
                    r2 = net.graph.nodes[n2]['radius']
                    r12 = TRR * min(r1, r2)
                    S12 = np.pi * r12**2
                    net.add_throat(n1, n2, radius=r12)
                    net.graph.nodes[n1]['exchange_area'] += S12
                    net.graph.nodes[n2]['exchange_area'] += S12
                    break


def scale(net, factor):
    """Scales the network dimensions, centers, radii, etc. by 'factor' """

    indices = net.graph.nodes

    c = np.array(
        list(zip(*nx.get_node_attributes(net.graph, 'center').values()))).T
    c *= factor
    nx.set_node_attributes(net.graph, dict(
        zip(indices, list(c))), name='center')

    pores_radii = np.array(
        list(nx.get_node_attributes(net.graph, 'radius').values()))
    pores_radii *= factor
    nx.set_node_attributes(net.graph, dict(
        zip(indices, list(pores_radii))), name='radius')

    throats_radii_dict = nx.get_edge_attributes(net.graph, 'radius')
    throats_radii = np.array(list(throats_radii_dict.values()))
    edges_keys = list(throats_radii_dict.keys())
    throats_radii *= factor
    nx.set_edge_attributes(net.graph, dict(
        zip(edges_keys, list(throats_radii))), name='radius')

    net.graph.graph['extent'] *= factor
    net.graph.graph['inner_extent'] *= factor
    net.compute_geometry()


def aggregates_nodes_if_too_big(pn, nodes=None, only_smaller=True, maxradius=None):
    """For each pore in nodes (iterable), the function checks if other pore overlaps.
    It merges overlapping pores with the larger one.
    """

    iterate = True

    while iterate:

        iterate = False

        if nodes is None:
            rdict = list(nx.get_node_attributes(pn.graph, 'radius').items())
            # we begin by the larger pores
            rdict.sort(key=lambda tup: tup[1], reverse=True)
            l, _ = zip(*rdict)

        else:
            l = nodes[:]

        for n1 in l:
            if pn.graph.has_node(n1):

                if pn.graph.nodes[n1]['radius'] > maxradius:
                    pn.graph.nodes[n1]['radius'] = maxradius

                stop = False

                while not stop:
                    stop = True
                    neighbors = [n for n in pn.graph[n1]]

                    for n2 in neighbors:
                        if (only_smaller and pn.graph.nodes[n1]['radius'] >= pn.graph.nodes[n2]['radius']) or (not only_smaller):
                            if pn._compute_auto_throat_length(n1, n2) <= 0 and pn.get_pore_distance(n1, n2) <= maxradius:
                                #center += pn.graph.nodes[n2]['center']
                                #counter += 1
                                pn.merge_pores(n1, n2)
                                # print('merging',n1,n2,pn.graph.nodes[n1]['radius'])
                                stop = False
                                iterate = True


def SetFaceBCnodes(net, step=100):

    """Auto detect pores that are close to the boundary and mark them as boundary nodes
    Adds category  'x+', 'x-', 'y+', 'y-', 'z+', 'z-' to boundary nodes (a node can have several caterogies)
    Distance is computed from face to pore center and NOT pore surface
    it means that big pores are not likely to be marked as boundary pore """

    nodes_indices = list(net.graph.nodes)
    centers = nx.get_node_attributes(net.graph, 'center')
    centers = np.array([np.array(centers[i]) for i in nodes_indices])
    kdtree = KDTree(centers, leaf_size=40)

    net.clear_category()

    extent = extent = net.graph.graph['extent']
    # A modifier en prenant la valeur moyenne des distances de chaque face ?
    net.graph.graph['inner_extent'] = extent

    x = np.linspace(0, extent[0], step)
    y = np.linspace(0, extent[1], step)
    z = np.linspace(0, extent[2], step)

    X, Y = np.meshgrid(x, y)
    face_Z0 = zip(X.ravel(), Y.ravel(), np.zeros(step*step))
    face_Z1 = zip(X.ravel(), Y.ravel(), np.zeros(step*step)+extent[2])

    X, Z = np.meshgrid(x, z)
    face_Y0 = zip(X.ravel(), np.zeros(step*step), Z.ravel())
    face_Y1 = zip(X.ravel(), np.zeros(step*step)+extent[1], Z.ravel())

    Y, Z = np.meshgrid(y, z)
    face_X0 = zip(np.zeros(step*step), Y.ravel(), Z.ravel())
    face_X1 = zip(np.zeros(step*step)+extent[0], Y.ravel(), Z.ravel())

    for coords in face_Z0:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]['category'].add('z-')
    for coords in face_Z1:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]['category'].add('z+')
    for coords in face_Y0:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]['category'].add('y-')
    for coords in face_Y1:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]['category'].add('y+')
    for coords in face_X0:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]['category'].add('x-')
    for coords in face_X1:
        n = kdtree.query([coords], k=1, return_distance=False)[0][0]
        net.graph.nodes[nodes_indices[n]]['category'].add('x+')

    for n in net.graph.nodes:

        if len(net.graph.nodes[n]['category']) == 0:
            net.graph.nodes[n]['category'].add('inner')


def SetBCnodes(net, thickness, mode='center'):
    """Set the nodes category. If nodes are in a layer of given thickness from the boundary, they are tagged as boundary nodes
    Two methods: either the center or the surface of the pore in the layer"""


    extent = net.graph.graph['extent']
    net.graph.graph['inner_extent'] = np.array(extent) - 2*thickness

    for n in net.graph.nodes:

        if mode == 'center':
            net.graph.nodes[n]['category'] = isinside(
                net.graph.nodes[n]['center'], net.graph.nodes[n]['radius'], extent, thickness)
        elif mode == 'surface':
            net.graph.nodes[n]['category'] = isinside_surface(
                net.graph.nodes[n]['center'], net.graph.nodes[n]['radius'], extent, thickness)
