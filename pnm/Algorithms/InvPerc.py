import networkx as nx
import numpy as np
from pnm import pore_network
import collections


def mercury_intrusion(pn, faces=("x+", "x-", "y+", "y-", "z+", "z-"), D_ratio=1.2, Dmax=500, Dmin=5e-3, verbose=False):
    """Mercury intrusion
    D_ratio is the diameter ratio between two iterations
    Dmax and Dmin -> depending on the network units. Usually in micrometers.
    On suppose qu'il n'y a pas de piègeage possible (par ex un capillaire entre deux pores saturés peut être saturé également)
    """

    pn.graph = nx.convert_node_labels_to_integers(pn.graph)

    # radii
    # radii = np.array(list(nx.get_node_attributes(pn.graph, "radius").values()))

    entrance_pores = pn.get_pores_by_category(category=faces, mode="match_one")
    # les pores "sources", qui peuvent être remplis ou non
    # Si source_nodes(i] == True, alors le fluide peut remplir les capillaires adjacents)
    source_nodes = {i: False for i in entrance_pores}
    filled_throats = collections.defaultdict(dict)

    if len(source_nodes) <= 0:
        print(f"No Pores belonging to category {faces}. Quit")
        return

    vint = []
    D = Dmax
    Dlist = []

    total_void_volume = pn.pores_volume + pn.throats_volume

    while D > Dmin and len(source_nodes) > 0:

        # Hg volume & current diameter
        vol = 0
        Dlist.append(D)
        new_addition = True
        counter = 0

        print(f"Filling pores with entrance diameter >= {D:.2e}.")

        while new_addition:

            counter += 1
            new_nodes = []
            nodes_to_delete = []
            new_addition = False
            new_filled_throats_counter = 0
            fill_current_pore = False

            # Iterating source nodes
            for active_node, filled in source_nodes.items():

                # if Di> D, an empty active pore can be filled by Hg - this is useful for BC pores
                if not filled and 2 * pn[active_node]["radius"] >= D:
                    vol += pn[active_node]["volume"]
                    new_addition = True
                    new_nodes.append(active_node)
                    fill_current_pore = True

                # Active filled pore - Hg can enter neighbors ##and not pn.graph[active_node][n]['periodic']
                elif filled:
                    # iterate empty throats - if 2*rt >= D, the throats is filled (as well as the pore, since rt <= rp)
                    for node in [
                        n
                        for n in pn.graph.neighbors(active_node)
                        if pn.graph[active_node][n]["radius"] * 2 >= D and not filled_throats.get(active_node, {}).get(n, False)
                    ]:
                        vol += pn.graph[active_node][node]["volume"]
                        new_filled_throats_counter += 1
                        # Need to update [i][j] and [j][i] dict
                        filled_throats[active_node][node] = True
                        filled_throats[node][active_node] = True
                        new_addition = True
                        # Si le pore voisin n'est pas actif ou pas rempli alors on l'ajoute à la liste des pores actifs et on le rempli
                        if not source_nodes.get(node, False) and node not in new_nodes:
                            new_addition = True
                            new_nodes.append(node)
                            vol += pn[node]["volume"]

                # Si tous les capillaires sont remplis on désactive le pore
                if filled or fill_current_pore:
                    n_filled_throats = len([n for n in pn.graph.neighbors(active_node) if filled_throats.get(active_node, {}).get(n, False)])

                    if n_filled_throats == pn.graph.degree(active_node):
                        nodes_to_delete.append(active_node)

            if new_addition:
                print(f"-Iteration {counter:<5d}, Volume: {vol:.2e}, pores filled: {len(new_nodes):<5d}, throats filled {new_filled_throats_counter:<5d} ")

            # Update source _nodes
            for index in nodes_to_delete:
                source_nodes.pop(index, None)
            for index in new_nodes:
                source_nodes[index] = True

        vint.append(vol)
        D /= D_ratio
        if verbose:
            print((f"-Hg volume={vol:.2e}, {100*vol/total_void_volume:.2f}% of total void volume"))

    return Dlist, vint
