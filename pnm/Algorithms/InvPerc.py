import networkx as nx
import numpy as np
from pnm import pore_network


def mercury_intrusion(pn, faces=("x+", "x-", "y+", "y-", "z+", "z-"), D_ratio=1.2, Dmax=500, Dmin=5e-3, verbose=False):
    """Mercury intrusion simulation.

    Simulates mercury intrusion into a pore network by progressively decreasing
    the intrusion diameter D. Pores and throats are filled when their diameter
    (2*radius) is >= D, starting from boundary (entrance) pores and propagating
    through connected throats.

    Parameters
    ----------
    pn : pore_network
        The pore network model instance. Must have ``radius`` and ``volume``
        node attributes and ``radius`` and ``volume`` edge attributes
        (e.g. after compute_geometry()).
    faces : tuple of str, optional
        Boundary faces used as mercury entry points
        (e.g. ("x+", "x-") connects x+ and x- faces). Default is all 6 faces.
    D_ratio : float, optional
        Diameter reduction ratio between iterations: D[i+1] = D[i] / D_ratio.
        Default is 1.2.
    Dmax : float, optional
        Maximum intrusion diameter. Default is 500.
    Dmin : float, optional
        Minimum intrusion diameter. The loop stops when D <= Dmin.
        Default is 5e-3.
    verbose : bool, optional
        If True, print progress information. Default is False.

    Returns
    -------
    Dlist : list of float
        Sequence of intrusion diameters.
    vint : list of float
        Intruded mercury volume at each diameter step.

    Notes
    -----
    Performance optimizations vs the original implementation:
    - Uses a stack-based BFS instead of iterating over all source nodes at
      every iteration. Only pores with newly fillable throats are processed.
    - Uses sets (O(1) lookup) for filled pores and filled throats instead of
      nested dict lookups.
    - Precomputes node radii and volumes as numpy arrays for fast access.

    The algorithm assumes no trapping (a capillary between two saturated pores
    is also saturated).
    """

    pn.graph = nx.convert_node_labels_to_integers(pn.graph)

    entrance_pores = pn.get_pores_by_category(category=faces, mode="match_one")

    if len(entrance_pores) <= 0:
        print(f"No Pores belonging to category {faces}. Quit")
        return

    # Precompute node attributes as numpy arrays for fast indexed access
    node_list = list(pn.graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    radii = np.array(list(nx.get_node_attributes(pn.graph, "radius").values()))
    volumes = np.array(list(nx.get_node_attributes(pn.graph, "volume").values()))

    # Filled state tracking using sets for O(1) membership tests
    filled_pores = set()          # pore indices that are filled with Hg
    filled_throats_set = set()    # (min_node, max_node) tuples for filled throats

    vint = []
    D = Dmax
    Dlist = []

    total_void_volume = pn.pores_volume + pn.throats_volume

    while D > Dmin:

        # Hg volume & current diameter
        vol = 0
        Dlist.append(D)
        counter = 0

        print(f"Filling pores with entrance diameter >= {D:.2e}.")

        # Stack of filled pores whose throats still need testing at current D.
        # At each new D, all filled pores with open throats are re-added so
        # their capillaries can be re-tested at the smaller diameter.
        throat_test_stack = []
        # List of pores discovered but not yet tested for self-filling (2*r >= D)
        candidate_pores = []

        # Add entrance pores that can self-fill at current D
        for n in entrance_pores:
            if n not in filled_pores and 2 * radii[node_to_idx[n]] >= D:
                candidate_pores.append(n)

        # Re-add all filled pores that still have unfilled throats, so their
        # capillaries are re-tested at the new (smaller) diameter.
        for n in filled_pores:
            for neighbor in pn.graph.neighbors(n):
                edge_key = (min(n, neighbor), max(n, neighbor))
                if edge_key not in filled_throats_set:
                    throat_test_stack.append(n)
                    break

        new_addition = True

        while new_addition:

            counter += 1
            new_addition = False
            new_filled_throats_counter = 0
            newly_filled = []

            # Phase 1: Fill candidate pores that can self-fill (2*r >= D)
            still_candidates = []
            for n in candidate_pores:
                if n not in filled_pores and 2 * radii[node_to_idx[n]] >= D:
                    vol += volumes[node_to_idx[n]]
                    filled_pores.add(n)
                    newly_filled.append(n)
                    throat_test_stack.append(n)
                    new_addition = True
                else:
                    still_candidates.append(n)
            candidate_pores = still_candidates

            # Phase 2: Test throats of filled pores (BFS propagation)
            while throat_test_stack:
                active_node = throat_test_stack.pop()

                for neighbor in pn.graph.neighbors(active_node):
                    edge_key = (min(active_node, neighbor), max(active_node, neighbor))
                    if edge_key in filled_throats_set:
                        continue

                    throat_radius = pn.graph[active_node][neighbor]["radius"]
                    if 2 * throat_radius >= D:
                        # Fill the throat
                        vol += pn.graph[active_node][neighbor]["volume"]
                        filled_throats_set.add(edge_key)
                        new_filled_throats_counter += 1
                        new_addition = True

                        # Fill the neighbor pore if not already filled
                        if neighbor not in filled_pores:
                            vol += volumes[node_to_idx[neighbor]]
                            filled_pores.add(neighbor)
                            newly_filled.append(neighbor)
                            # The neighbor can now try to fill its own throats
                            throat_test_stack.append(neighbor)
                            new_addition = True

            if new_addition:
                print(f"-Iteration {counter:<5d}, Volume: {vol:.2e}, pores filled: {len(newly_filled):<5d}, throats filled {new_filled_throats_counter:<5d} ")

        vint.append(vol)
        D /= D_ratio
        if verbose:
            print((f"-Hg volume={vol:.2e}, {100*vol/total_void_volume:.2f}% of total void volume"))

    return Dlist, vint
