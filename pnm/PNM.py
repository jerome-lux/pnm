# coding=utf-8

import numpy as np
import networkx as nx
from warnings import warn
import pickle

# TODO:  clean the code. Remove unused functions.
#       Add support for periodicity -> Modify get_throats_length -> throats must have a fixed length attributes, get_throats_length should only return the attribute
# A new function should check the length

# Make sure that methods for adding or removing elements do not recompute geom or update porosity, extent or other parameters

side_to_int_dict = {"x-": 256, "x+": 512, "y-": 1024, "y+": 2048, "z-": 4096, "z+": 8192, "none": 0, "inner": 0}
int_to_side_dict = {v: k for k, v in side_to_int_dict.items()}


class pore_network:
    """Ball and stick pore network model.

    The graph is stored in a networkx.Graph. Geometry and other properties are
    stored in node and edge attributes. The class provides methods to manipulate
    the network (add/remove pores and throats, compute geometry, merge pores,
    check overlapping, etc.).

    Graph attributes:
        graph_type : str
            Type of network (e.g. "generic", "cubic_regular", "random_packing").
        extent : list of float or None
            Box size [lx, ly, lz] defined by pore surfaces.
        bbox : list of float or None
            Bounding box [xmin, ymin, zmin, xmax, ymax, zmax].
        porosity : float or None
            Porosity (void volume / box volume).
        inner_extent : list of float or None
            Inner extent (box size from pore centers), set by boundary node routines.
        spacing : list of float or None
            Grid spacing for regular networks.

    Node attributes:
        radius : float
            Pore radius.
        center : tuple of float
            (x, y, z) coordinates of the pore center.
        volume : float
            Pore volume.
        category : set of str
            Set of side labels (x-, x+, y-, y+, z-, z+, inner, none).
        periodic : bool
            Whether the pore is periodic.
        exchange_area : float
            Cumulated exchange area at pore boundaries.
        surface_ratio : float
            Surface ratio for boundary correction.
        boundary : str
            Boundary label ("inlet", "outlet").

    Edge attributes:
        radius : float
            Throat radius.
        length : float
            Throat length.
        volume : float
            Throat volume.
        periodic : bool
            Whether the throat is periodic.
        pdir : str
            Periodic direction.
        sw : float
            Wetting fluid saturation.
        wf_side : int
            Wetting fluid side (-1, 0, 1).
    """

    def __init__(self, network=None):
        """Initialize a pore network.

        Parameters
        ----------
        network : networkx.Graph or None, optional
            An existing networkx graph to wrap. If None, a new empty graph is
            created. When a graph is provided, compute_geometry() is called.
        """

        if network is None:
            self.graph = nx.Graph()
            self.graph.graph["graph_type"] = "generic"
            # extent is the extent defined by pores surfaces
            self.graph.graph["extent"] = None
            self.graph.graph["bbox"] = None
            self.geom_complete = False
            self.pores_volume = 0
            self.throats_volume = 0
        else:
            self.graph = network
            self.compute_geometry()

    def __getitem__(self, n):
        """Return the node attribute dictionary for a given node index.

        Parameters
        ----------
        n : int
            Node index.

        Returns
        -------
        dict
            The node attribute dictionary.
        """
        return self.graph.nodes[n]

    @classmethod
    def read(cls, inputfilename, method="default"):
        """Read an existing network from a file.

        Parameters
        ----------
        inputfilename : str
            Path to the input file.
        method : str, optional
            File format: "default"/"gml" (GML), "gexf", or "pickle".
            Default is "default".

        Returns
        -------
        pore_network
            A new pore_network instance containing the read graph.
        """
        method = method.lower()

        # import json
        # with open(inputfilename, 'w') as infile:
        # data = json.load(infile)
        # g = nx.readwrite.json_graph.node_link_graph(data)
        # return cls(network = g)
        if method == "gml" or method == "default":
            return cls(network=nx.read_gml(inputfilename))
        elif method == "gexf":
            return cls(network=nx.read_gexf(inputfilename))
        elif method == "pickle":
            with open(inputfilename, 'rb') as infile:
                return cls(network=pickle.load(infile))

    @classmethod
    def read_ascii(cls, inputfilename, data=True, legacy=False):
        """Read an ascii network file used in imbibition simulations.

        File format:
            Header line: lx ly lz npores nthroats
            Pore lines: x y z r sw pw pnw cc side
            Throat lines: pi pj r l sw cc w_ind

        Parameters
        ----------
        inputfilename : str
            Path to the ascii input file.
        data : bool, optional
            If True, import saturation, pressure and ganglia label values.
            Default is True.
        legacy : bool, optional
            If True, read the old network format (pore indices are 1-based).
            Default is False.

        Returns
        -------
        pore_network
            A new pore_network instance populated from the file.
        """

        network = cls.__new__(cls)
        network.__init__()

        with open(inputfilename, "r") as netfile:
            header = netfile.readline().split()
            network.graph.graph["extent"] = (float(header[0]), float(header[1]), float(header[2]))

            # Read and allocates pores: x,y,x,r,sw,pw,pnw,cc,side
            for i in range(int(header[3])):
                p = netfile.readline().split()
                side = int(p[-1])
                glabel = int(p[-2])
                p = list(map(float, p[:-2]))
                if side == 0:
                    cat = set(["inner"])
                else:
                    cat = set([int_to_side_dict[side & k] for k in int_to_side_dict.keys() if side & k > 0])

                if data:
                    network.add_pore(p[3], (p[0], p[1], p[2]), category=cat, sw=p[4], pw=p[5], pnw=p[6], glabel=glabel)
                else:
                    network.add_pore(p[3], (p[0], p[1], p[2]), category=cat)

            # Read and allocates throats: pi,pj,r,l,sw,cc,w_ind
            for i in range(int(header[4])):
                t = netfile.readline().split()
                pi = int(t[0])
                pj = int(t[1])

                if legacy:
                    pi -= 1
                    pj -= 1

                if data:
                    network.add_throat(
                        pi, pj, radius=float(t[2]), length=float(t[3]), sw=float(t[4]), glabel=int(t[5])
                    )
                else:
                    network.add_throat(pi, pj, radius=float(t[2]), length=float(t[3]))

        network.compute_geometry()

        return network

    @staticmethod
    def _to_serializable(value, array_as_list=True):
        """Recursively convert numpy containers/scalars to plain Python types.

        NetworkX text formats (GML, GEXF) only accept str/int/float attribute values:
        numpy arrays and integer scalars raise a ``NetworkXError`` / ``TypeError``.

        Parameters
        ----------
        value : object
            Attribute value to convert.
        array_as_list : bool, optional
            If True (GML), numpy arrays become nested Python lists — the GML writer emits
            them as repeated key/value lines and the reader parses them back into lists.
            If False (GEXF), arrays and lists are stringified because GEXF interprets list
            values as "dynamic" time-series data. Default is True.

        Returns
        -------
        object
            Equivalent value using only plain Python types (list, int, float, bool, str).
        """
        if isinstance(value, np.ndarray):
            return value.tolist() if array_as_list else str(value.tolist())
        if isinstance(value, (list, tuple)):
            # GEXF interprets any list as "dynamic" time-series data ((val, start, end) tuples),
            # so lists must be stringified there; GML writes them natively as repeated values.
            converted = [pore_network._to_serializable(v, array_as_list) for v in value]
            return converted if array_as_list else str(converted)
        if isinstance(value, dict):
            return {k: pore_network._to_serializable(v, array_as_list) for k, v in value.items()}
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    def _sanitized_copy(self, array_as_list=True):
        """Return a shallow copy of the graph with attributes converted to plain Python types.

        The live graph is not modified: ``Graph.copy()`` provides independent node/edge/graph
        attribute dicts. Node ids that are not int/float/str (e.g., numpy integers from
        ``np.arange`` before ``compute_geometry`` relabels) are converted, since the GML writer
        rejects them; attributes with a None value ("unset") are dropped because neither GML nor
        GEXF has a null representation.

        Parameters
        ----------
        array_as_list : bool, optional
            Forwarded to ``_to_serializable`` (True for GML, False for GEXF). Default is True.

        Returns
        -------
        networkx.Graph
            A copy of the graph whose node ids and attributes use only plain Python types.
        """
        G = self.graph.copy()

        # The GML writer requires node ids to be int/float/str (GEXF stringifies them itself).
        if not all(isinstance(n, (int, float, str)) for n in G.nodes):
            mapping = {n: (int(n) if isinstance(n, np.integer) else str(n)) for n in G.nodes}
            G = nx.relabel_nodes(G, mapping)

        def conv(value):
            return self._to_serializable(value, array_as_list=array_as_list)

        for _, datadict in G.nodes(data=True):
            for key, val in list(datadict.items()):
                if val is None:
                    del datadict[key]
                else:
                    datadict[key] = conv(val)
        for _, _, datadict in G.edges(data=True):
            for key, val in list(datadict.items()):
                if val is None:
                    del datadict[key]
                else:
                    datadict[key] = conv(val)
        for key, val in list(G.graph.items()):
            if val is None:
                del G.graph[key]
            else:
                G.graph[key] = conv(val)
        return G

    def write(self, outfilename, method="default"):
        """Write the network to a file.

        Node category sets are converted to lists before saving for serialization. For the text
        formats (GML, GEXF), a sanitized copy of the graph is written: numpy arrays/scalars in
        node/edge/graph attributes are converted to plain Python types (arrays become lists in
        GML and strings in GEXF) so that ``nx.write_gml`` / ``nx.write_gexf`` do not fail on them.

        Parameters
        ----------
        outfilename : str
            Output file path (without extension; extension is added by the format).
        method : str, optional
            Output format: "default"/"gml" (GML), "gexf", or "pickle".
            Default is "default".
        """

        method = method.lower()

        # Ensure that some graph attributes are converted to list before saving
        extent = self.graph.graph.get("extent")
        if extent is not None:
            self.graph.graph["extent"] = list(extent)
        inner_extent = self.graph.graph.get("inner_extent")
        if inner_extent is not None:
            self.graph.graph["inner_extent"] = list(inner_extent)

        # transform category attributes [set] to list
        bcattr = {k:list(v) for k, v in nx.get_node_attributes(self.graph, "category").items()}
        nx.set_node_attributes(self.graph, bcattr,"category")
        
        if method == "pickle":
            with open(outfilename + ".pnm", 'wb') as f:
                pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
                return

        if method == "gml" or method == "default":
            nx.write_gml(self._sanitized_copy(array_as_list=True), outfilename + ".gml")
        elif method == "gexf":
            nx.write_gexf(self._sanitized_copy(array_as_list=False), outfilename + ".gexf")

    def save_as_ascii(self, outfilename):
        """Write the network in a text file using the following format:

        Header line: lx ly lz npores nthroats
        Pore lines: x y z r sw pw pnw cc side
        Throat lines: pi pj r l sw cc w_ind

        Pore fields:
            x, y, z : pore center coordinates
            r : pore radius
            sw : wetting fluid saturation
            pw : wetting fluid pressure
            pnw : non-wetting fluid pressure
            cc : connected component label
            side : face to which the pore belongs (bit-coded side labels:
                256 x-, 512 x+, 1024 y-, 2048 y+, 4096 z-, 8192 z+)

        Throat fields:
            pi, pj : adjacent pore indices (starting at 0)
            r : throat radius
            l : throat length
            sw : wetting fluid saturation
            cc : connected component label
            w_ind : index of the pore on the wetting-fluid side

        Parameters
        ----------
        outfilename : str
            Path to the output ascii file.
        """

        with open(outfilename, "w", encoding='utf-8') as netfile:
            extent = self.graph.graph.get("extent")
            if extent is None:
                warn("Extent is not set. Cannot save as ascii.")
                return
            netfile.write(
                "{} {} {} {} {}\n".format(
                    self.graph.graph["extent"][0],
                    self.graph.graph["extent"][1],
                    self.graph.graph["extent"][2],
                    self.graph.number_of_nodes(),
                    self.graph.number_of_edges(),
                )
            )

            for node in self.graph.nodes:
                n = self.graph.nodes[node]
                side = sum([side_to_int_dict[c] for c in n["category"]])
                netfile.write(
                    "{} {} {} {} {} {} {} {} {}\n".format(
                        n["center"][0], n["center"][1], n["center"][2], n["radius"], 0.0, 0.0, 0.0, 0, side
                    )
                )

            for n1, n2 in self.graph.edges:
                netfile.write(
                    "{} {} {} {} {} {} {}\n".format(
                        n1, n2, self.graph[n1][n2]["radius"], self.graph[n1][n2]["length"], 0.0, 0, 0
                    )
                )

    def check_overlapping(self, fit_radius=True, merge=True, mindist="auto", update_geometry=False):
        """Check for overlapping pores and resolve them.

        Pores are processed from largest to smallest. Overlapping pairs are
        handled as follows:
        - If fit_radius=True and the overlap can be removed by shrinking the
          smaller pore, its radius is reduced.
        - Otherwise, if merge=True the two pores are merged (the smaller is
          removed into the larger); otherwise the smaller pore is deleted.

        Parameters
        ----------
        fit_radius : bool, optional
            If True, attempt to resolve overlaps by shrinking pore radii.
            Default is True.
        merge : bool, optional
            If True, merge overlapping pores instead of deleting them.
            Default is True.
        mindist : str or float, optional
            Minimum surface-to-surface distance. If "auto", uses 2% of the
            pore radius. Default is "auto".
        update_geometry : bool, optional
            If True, recompute throat lengths and radii after resolving.
            Default is False.
        """

        from scipy.spatial.distance import cdist
        from scipy.spatial import cKDTree

        # index = list(self.graph)[:]
        # centers = np.array(list(zip(*nx.get_node_attributes(self.graph,'center').values()))).T
        # pores_radii = np.fromiter(nx.get_node_attributes(self.graph,'radius').values(),dtype=float)

        pores_radii = list(nx.get_node_attributes(self.graph, "radius").items())
        # we begin by the bigger pores
        pores_radii.sort(key=lambda tup: tup[1], reverse=True)
        index, pores_radii = zip(*pores_radii)
        pores_radii = np.array(pores_radii)

        centers = nx.get_node_attributes(self.graph, "center")
        centers = [np.array(centers[i]) for i in index]
        centers = np.array(centers)
        # distances = cdist(centers,centers)
        kdtree = cKDTree(centers)

        stop = False

        while not stop:
            stop = True

            for i, n1 in enumerate(index):
                # distances = cdist(centers,[self.graph.nodes[n1]['center']])[:,0]

                if self.graph.has_node(n1):
                    if mindist == "auto":
                        gap = self.graph.nodes[n1]["radius"] * 0.02
                    else:
                        gap = mindist

                    labels = kdtree.query_ball_point(
                        self.graph.nodes[n1]["center"], 2.5 * self.graph.nodes[n1]["radius"]
                    )
                    # Remove the query point itself safely (it should always be present, but guard anyway)
                    labels = [idx for idx in labels if idx != i]
                    # distances,labels = kdtree.query(x=net.graph.nodes[n1]['center'],2*self.graph.nodes[n1]['radius'],n_jobs=1)
                    # labels.remove(i)
                    # distance *= 0.998
                    distances = cdist(centers[labels], [self.graph.nodes[n1]["center"]])[:, 0] * 0.998
                    d = distances - pores_radii[labels]
                    d -= self.graph.nodes[n1]["radius"]
                    # On commence par la distance la plus faible
                    d_and_labels = [(d[j], k) for j, k in enumerate(labels)]
                    d_and_labels.sort(key=lambda t: t[0])

                    for dist, ind in d_and_labels:
                        n2 = index[ind]
                        if self.graph.has_node(n2) and self.graph.has_node(n1):
                            # Le centre du pore né est dans la sphère du pore n1 OU il y a overlapping et fit_radius == False
                            # -> Merging ou suppression du pore de plus petit rayon
                            if (dist + self.graph.nodes[n2]["radius"] <= gap) or (
                                dist < gap and dist + self.graph.nodes[n2]["radius"] > gap and not fit_radius
                            ):
                                if self.graph.nodes[n1]["radius"] >= self.graph.nodes[n2]["radius"]:
                                    if merge:
                                        self.merge_pores(n1, n2)
                                        print("pore", n1, "and", n2, "overlap: merging (deleting", n2, ")")
                                    else:
                                        self.remove_pore(n2)
                                        print("pore", n1, "and", n2, "overlap: deleting", n2)

                                else:
                                    if merge:
                                        self.merge_pores(n2, n1)
                                        print("pore", n1, "and", n2, "overlap: merging (deleting", n1, ")")
                                    else:
                                        self.remove_pore(n1)
                                        print("pore", n1, "and", n2, "overlap: deleting", n2)
                                    # On termine l'itération car le pore n1 n'existe plus...
                                    break

                            # Overlapping et fit_radius == True
                            # 3 options:
                            # -Le rayon du pore le plus petit est modifié
                            # -Merging
                            # -Suppression
                            elif dist < gap and dist + self.graph.nodes[n2]["radius"] > gap and fit_radius:
                                if self.graph.nodes[n1]["radius"] >= self.graph.nodes[n2]["radius"]:
                                    r = dist + self.graph.nodes[n2]["radius"] - self.graph.nodes[n1]["radius"] - gap
                                    if self.graph.nodes[n2]["radius"] >= r and r > 0:
                                        self.graph.nodes[n2]["radius"] = r
                                        pores_radii[ind] = r
                                        print("pore", n1, "and", n2, "overlap: changin radius of", n2, "to", r)
                                    else:
                                        if merge:
                                            self.merge_pores(n1, n2)
                                            print("pore", n1, "and", n2, "overlap: merging (deleting", n2, ")")
                                        else:
                                            self.remove_pore(n2)
                                            print("pore", n1, "and", n2, "overlap: deleting", n2)
                                else:
                                    if self.graph.nodes[n1]["radius"] >= dist:
                                        self.graph.nodes[n1]["radius"] = dist
                                        pores_radii[i] = dist
                                        print("pore", n1, "and", n2, "overlap: changin radius of", n1, "to", dist)
                                    else:
                                        if merge:
                                            self.merge_pores(n2, n1)
                                            print("pore", n1, "and", n2, "overlap: merging (deleting", n1, ")")
                                        else:
                                            self.remove_pore(n1)
                                            print("pore", n1, "and", n2, "overlap: deleting", n1)
                                        # On termine l'itération car le pore n1 n'existe plus...
                                        break

        if update_geometry:
            self.set_auto_throats_length()
            self.set_auto_throats_radius()

    def compute_extent(self):
        """Compute the bounding box and extent of the network.

        The extent is derived from pore centers and radii (pore surfaces).
        Results are stored in graph attributes "bbox" and "extent".
        """
        pores_radii = np.fromiter(nx.get_node_attributes(self.graph, "radius").values(), dtype=float)
        minbox = np.array(list(zip(*nx.get_node_attributes(self.graph, "center").values()))).T
        maxbox = minbox.copy()

        maxbox[:, 0] += pores_radii
        maxbox[:, 1] += pores_radii
        maxbox[:, 2] += pores_radii
        minbox[:, 0] -= pores_radii
        minbox[:, 1] -= pores_radii
        minbox[:, 2] -= pores_radii

        xmin = minbox[:, 0].min()
        ymin = minbox[:, 1].min()
        zmin = minbox[:, 2].min()
        xmax = maxbox[:, 0].max()
        ymax = maxbox[:, 1].max()
        zmax = maxbox[:, 2].max()

        self.graph.graph["bbox"] = [xmin, ymin, zmin, xmax, ymax, zmax]

        self.graph.graph["extent"] = [0., 0., 0.]
        self.graph.graph["extent"][0] = abs(xmax - xmin)
        self.graph.graph["extent"][1] = abs(ymax - ymin)
        self.graph.graph["extent"][2] = abs(zmax - zmin)

    def compute_porosity(self):
        """Compute the network porosity.

        Porosity is the total void volume (pores + throats) divided by the box
        volume (product of extent). The result is stored in graph attribute
        "porosity". Instance attributes pores_volume and throats_volume are
        also updated.

        Requires geometry to be complete (compute_geometry() called).
        """
        if self.geom_complete:
            self.graph.graph["porosity"] = 0

            pore_volumes = nx.get_node_attributes(self.graph, "volume")
            self.pores_volume = np.fromiter(pore_volumes.values(), dtype=float).sum()

            throat_volumes = nx.get_edge_attributes(self.graph, "volume")
            self.throats_volume = np.fromiter(throat_volumes.values(), dtype=float).sum()

            # Compute the extent from pores center and radii
            if self.graph.graph.get("extent", None) is None:
                self.compute_extent()

            self.graph.graph["porosity"] = (self.pores_volume + self.throats_volume) / np.prod(
                np.array(self.graph.graph["extent"])
            )

        else:
            warn("Geometry not complete. Cannot compute porosity")

    def _compute_pore_vol(self, n):
        """Compute the volume of a single pore (sphere).

        Parameters
        ----------
        n : int
            Node index.

        Returns
        -------
        float
            Pore volume = 4/3 * pi * r^3.
        """
        return (4 / 3) * np.pi * self.graph.nodes[n]["radius"] ** 3

    def _compute_throat_vol(self, n1, n2):
        """Compute the volume of a single throat (cylinder).

        Parameters
        ----------
        n1, n2 : int
            Node indices of the throat endpoints.

        Returns
        -------
        float
            Throat volume = length * pi * r^2.
        """
        return self.graph[n1][n2]["length"] * np.pi * self.graph[n1][n2]["radius"] ** 2

    def compute_throats_volume(self):
        """Compute the volume of all throats and store it as edge attribute "volume".

        Requires geometry to be complete. Updates the instance attribute
        throats_volume.
        """
        if self.geom_complete:
            throats_radii = np.fromiter(nx.get_edge_attributes(self.graph, "radius").values(), dtype=float)
            throats_length = np.fromiter(nx.get_edge_attributes(self.graph, "length").values(), dtype=float)
            v = throats_length * np.pi * throats_radii**2

            volumes = dict(zip(self.graph.edges, v))

            nx.set_edge_attributes(self.graph, volumes, "volume")

            self.throats_volume = v.sum()

        else:
            warn("Geometry not complete. Cannot compute throat volumes")

    def compute_pores_volume(self):
        """Compute the volume of all pores and store it as node attribute "volume".

        Requires geometry to be complete. Updates the instance attribute
        pores_volume.
        """
        if self.geom_complete:
            pores_radii = np.fromiter(nx.get_node_attributes(self.graph, "radius").values(), dtype=float)
            v = (pores_radii**3) * np.pi * 4 / 3
            volumes = dict(zip(self.graph.nodes, v))

            nx.set_node_attributes(self.graph, volumes, "volume")

            self.pores_volume = v.sum()

        else:
            warn("Geometry not complete. Cannot compute pore volumes")

    def remove_isolated_pores(self, only_inner_pores=False):
        """Remove pores that are not connected to any other pore (degree 0).

        Parameters
        ----------
        only_inner_pores : bool, optional
            If True, only consider pores with category exactly {"inner"}.
            Default is False (all pores).
        """
        if only_inner_pores:
            nodes = self.get_pores_by_category("inner", mode="equal")
        else:
            nodes = list(self.graph.nodes.keys())

        for n in nodes:
            if self.graph.degree(n) == 0:
                print("Removing isolated pore", n)
                self.remove_pore(n)

    def automatic_geometry(self):
        """Set throat lengths and radii automatically from pore geometry.

        Erases any custom throat length/radius values. Not compatible with
        periodic networks (periodic throats keep their length).
        """
        self.set_auto_throats_length()
        self.set_auto_throats_radius()

    def complete_missing_throat_geometry(self):
        """Set radius and/or length only on throats that are missing them.

        Unlike :meth:`automatic_geometry`, existing values are never overwritten, so
        PRTRR-based radii set by ``add_throats_by_surf`` (or any custom geometry) are
        preserved. Periodic throats without a length are left untouched, mirroring
        :meth:`set_auto_throats_length`.

        Returns
        -------
        int
            Number of throats that were missing at least one attribute and got completed.
        """
        n_completed = 0
        for n1, n2 in self.graph.edges:
            d = self.graph[n1][n2]
            if "radius" not in d or "length" not in d:
                if "radius" not in d:
                    d["radius"] = self._compute_auto_throat_radius(n1, n2)
                if "length" not in d and not d.get("periodic", False):
                    d["length"] = self._compute_auto_throat_length(n1, n2)
                n_completed += 1
        return n_completed

    def compute_geometry(self, autothroats=False):
        """Compute all geometry properties and check completeness.

        Steps:
        1. Reset volume accumulators.
        2. Remove isolated pores.
        3. Relabel nodes to consecutive integers (note: this breaks external
           references to old node labels).
        4. Check that all pores have radius and center attributes.
        5. Check that all throats have radius and length attributes
           (unless autothroats=True).
        6. If complete, compute extent, pore volumes, throat volumes, and porosity.

        Parameters
        ----------
        autothroats : bool, optional
            If True, automatically compute throat lengths and radii instead of
            requiring them as attributes. Default is False.

        Returns
        -------
        bool
            True if the geometry is complete, False otherwise.
        """

        self.pores_volume = 0
        self.throats_volume = 0

        self.geom_complete = True

        self.remove_isolated_pores()

        self.graph = nx.convert_node_labels_to_integers(self.graph)

        if len(nx.get_node_attributes(self.graph, "radius")) != self.graph.number_of_nodes():
            self.geom_complete = False
            warn("Cannot compute geometry properties. Some pores do not have a radius !")

        if len(nx.get_node_attributes(self.graph, "center")) != self.graph.number_of_nodes():
            self.geom_complete = False
            warn("Cannot compute geometry properties. Some pores do not have a center !")

        if self.graph.number_of_nodes() == 0 or self.graph.number_of_edges() == 0:
            self.geom_complete = False
            warn("Cannot compute geometry properties. Number of nodes and/or throats is 0 !")

        if autothroats:
            self.automatic_geometry()
        else:
            if len(nx.get_edge_attributes(self.graph, "radius")) != self.graph.number_of_edges():
                self.geom_complete = False
                warn("Cannot compute geometry properties. Some throats do not have a radius !")
            if len(nx.get_edge_attributes(self.graph, "length")) != self.graph.number_of_edges():
                self.geom_complete = False
                warn("Cannot compute geometry properties. Some throats do not have a length !")

        if self.geom_complete:
            self.compute_extent()
            self.compute_pores_volume()
            self.compute_throats_volume()
            self.compute_porosity()

        return self.geom_complete

    def merge_pores(
        self,
        n1,
        n2,
        setcategory="union",
        radius=None,
        center=None,
        check_throats=True,
        inner_category="inner",
        verbose=False,
    ):
        """Merge two pores n1 and n2 into a single pore (n1).

        Pore n2 is removed. All throats connected to n2 are rewired to n1.
        Throat lengths and radii connected to n1 are recomputed if
        check_throats is True.

        Parameters
        ----------
        n1, n2 : int
            Node indices of the two pores to merge. n1 is kept, n2 is removed.
        setcategory : str, optional
            If "union", the resulting category is the union of both pores'
            categories (with "inner" removed if multiple categories remain).
            Default is "union".
        radius : float or None, optional
            Radius of the merged pore. If None, keeps n1's radius.
        center : tuple of float or None, optional
            Center of the merged pore. If None, keeps n1's center.
        check_throats : bool, optional
            If True, recompute throat length and radius for throats connected
            to the merged pore. Default is True.
        inner_category : str, optional
            Unused parameter kept for compatibility. Default is "inner".
        verbose : bool, optional
            If True, print merge information. Default is False.
        """

        if not self.graph.has_node(n1) or not self.graph.has_node(n2):
            warn("Nodes {} or {} does not exist. Cannot merge them".format(n1, n2))
            return
        elif verbose:
            print("Merging pore {} and {}".format(n1, n2))

        if center is not None:
            self.graph.nodes[n1]["center"] = center

        if radius is not None:
            self.graph.nodes[n1]["radius"] = radius

        category = self.graph.nodes[n2]["category"]

        if setcategory == "union":
            self.graph.nodes[n1]["category"] = self.graph.nodes[n1]["category"].union(category)
            if len(self.graph.nodes[n1]["category"]) > 1 and "inner" in self.graph.nodes[n1]["category"]:
                self.graph.nodes[n1]["category"] = self.graph.nodes[n1]["category"].difference(set(["inner"]))

        # if not G.has_edge(u,v):
        # warn("Nodes {} and {} will be merged but they are not adjacent".format(u,v))

        # Warning : here we just copy the old edge attributes to the new one, so that the attributes are already defined.
        # The values must however be checked !
        new_edges = [(n1, n3, d) for _, n3, d in self.graph.edges(n2, data=True) if (n3 != n1 and n3 != n2)]
        try:
            self.graph.add_edges_from(new_edges)
        except:
            warn("Error trying to create new edges when merging pores {} and {}".format(n1, n2))
            warn("Edges list {}".format(new_edges))

        self.graph.remove_node(n2)

        if check_throats:
            for n3 in self.graph[n1]:
                self._compute_auto_throat_length(n1, n3)
                self._compute_auto_throat_radius(n1, n3)

    def _compute_auto_throat_radius(self, n1, n2):
        """Compute the automatic throat radius as min(r(n1), r(n2)).

        Parameters
        ----------
        n1, n2 : int
            Node indices of the throat endpoints.

        Returns
        -------
        float or None
            The computed radius, or None if node attributes are missing.
        """
        try:
            return min(self.graph.nodes[n1]["radius"], self.graph.nodes[n2]["radius"])
        except KeyError:
            warn("Cannot compute auto radius of throat between {} and {}".format(n1, n2))
            return None

    def _compute_auto_throat_length(self, n1, n2):
        """Compute the automatic throat length as center distance minus radii.

        The length is clamped to a minimum of 0 (overlapping pores give 0).

        Parameters
        ----------
        n1, n2 : int
            Node indices of the throat endpoints.

        Returns
        -------
        float or None
            The computed length, or None if node attributes are missing.
        """
        try:
            delta = np.array(self.graph.nodes[n1]["center"]) - np.array(self.graph.nodes[n2]["center"])
            return max(
                0, np.sqrt(np.dot(delta, delta)) - self.graph.nodes[n1]["radius"] - self.graph.nodes[n2]["radius"]
            )
        except KeyError:
            warn("Cannot compute auto length of throat between {} and {}".format(n1, n2))
            return None

    def set_auto_throats_radius(self):
        """Set the radius of every throat to min(r(n1), r(n2)).

        Erases any custom throat radius values.
        """

        for n1, n2 in self.graph.edges:
            self.graph[n1][n2]["radius"] = self._compute_auto_throat_radius(n1, n2)

    def set_auto_throats_length(self):
        """Compute and set the length of every non-periodic throat.

        The length is the center-to-center distance minus the two pore radii,
        clamped to a minimum of 0. Periodic throats (flagged by the "periodic"
        edge attribute) are skipped.
        """

        for n1, n2 in self.graph.edges:
            # Periodic throats are flagged via the 'periodic' edge attribute,
            # not via a 'category' attribute.
            if not self.graph[n1][n2].get("periodic", False):
                self.graph[n1][n2]["length"] = self._compute_auto_throat_length(n1, n2)

    def get_pore_distance(self, n1, n2):
        """Compute the Euclidean distance between two pore centers.

        Parameters
        ----------
        n1, n2 : int
            Node indices.

        Returns
        -------
        float
            Distance between pore centers, or 0 if attributes are missing.
        """
        # TODO: take into account periodic boundary conditions

        try:
            delta = np.array(self.graph.nodes[n1]["center"]) - np.array(self.graph.nodes[n2]["center"])
            return np.sqrt(np.dot(delta, delta))

        except KeyError:
            warn("Cannot compute distance between pores {} and {}".format(n1, n2))
            return 0

    def get_throat_section(self, n1, n2):
        """Compute the cross-sectional area of a throat (pi * r^2).

        Parameters
        ----------
        n1, n2 : int
            Node indices of the throat endpoints.

        Returns
        -------
        float
            Throat section area, or 0 if attributes are missing.
        """
        try:
            return np.pi * self.graph[n1][n2]["radius"] ** 2

        except KeyError:
            warn("Cannot compute section of throat between pores {} and {}".format(n1, n2))
            return 0

    def update_porosity(self):
        """Update the porosity from stored pore and throat volumes.

        Porosity = (pores_volume + throats_volume) / box_volume.
        Does not recompute volumes or iterate through elements.
        Requires geometry to be complete.
        """

        if self.geom_complete:
            self.graph.graph["porosity"] = (self.pores_volume + self.throats_volume) / np.prod(
                np.array(self.graph.graph["extent"])
            )

    def add_pore(self, radius, center, category=None, periodic=False, **kwargs):
        """Add a pore to the network.

        Parameters
        ----------
        radius : float
            Pore radius.
        center : tuple of float
            (x, y, z) coordinates of the pore center.
        category : set of str or None, optional
            Set of side labels (e.g. {"x-", "y+"}) or {"inner"}. Defaults to an empty set.
        periodic : bool, optional
            Whether the pore is periodic. Default is False.
        **kwargs
            Additional node attributes.
        """
        if category is None:
            category = set()

        n = self.graph.number_of_nodes()

        while self.graph.has_node(n):
            n += 1

        self.graph.add_node(
            n, radius=radius, center=center, volume=4 * np.pi * radius**3 / 3, category=category, periodic=periodic, **kwargs
        )

    def add_throat(self, n1, n2, radius=None, length=None, periodic=False, pdir="none", **kwargs):
        """Add a throat between two existing pores.

        If radius or length is not provided, it is computed automatically from
        the connected pore geometries. The "periodic" node attribute is set on
        both endpoint pores.

        Parameters
        ----------
        n1, n2 : int
            Node indices of the two pores to connect.
        radius : float or None, optional
            Throat radius. If None, computed as min(r(n1), r(n2)).
        length : float or None, optional
            Throat length. If None, computed from pore centers and radii.
        periodic : bool, optional
            Whether the throat is periodic. Default is False.
        pdir : str, optional
            Periodic direction label. Default is "none".
        **kwargs
            Additional edge attributes.
        """

        if self.graph.has_node(n1) and self.graph.has_node(n2):
            if radius is None:
                radius = self._compute_auto_throat_radius(n1, n2)
            if length is None:
                length = self._compute_auto_throat_length(n1, n2)

            self.graph.add_edge(n1, n2, radius=radius, length=length, periodic=periodic, pdir=pdir, **kwargs)
            nx.set_node_attributes(self.graph, dict(zip([n1, n2], [periodic, periodic])), "periodic")

        else:
            warn("Cannot create throat between {} and {}: nodes do not exist".format(n1, n2))

    def remove_throat(self, n1, n2):
        """Remove the throat between two pores. Silently ignores missing edges."""

        try:
            self.graph.remove_edge(n1, n2)
        except:
            pass

    def remove_pore(self, n, verbose=False):
        """Remove a pore from the network.

        Parameters
        ----------
        n : int
            Node index of the pore to remove.
        verbose : bool, optional
            If True, print a confirmation message. Default is False.
        """

        try:
            self.graph.remove_node(n)
            if verbose:
                print("removing pore", n)
        except:
            pass

    def clear_category(self, nodes=None):
        """Clear the category attribute of pores (set to empty set).

        Parameters
        ----------
        nodes : iterable of int or None, optional
            Node indices to clear. If None, all nodes are cleared.
        """

        if nodes is None:
            nodes = self.graph.nodes

        for n in nodes:
            self.graph.nodes[n]["category"] = set()

    def clear_throats(self):
        """Remove all throats from the network."""

        print("removing all throats")
        self.graph.remove_edges_from(list(self.graph.edges))

    def add_attributes(self, pore_dict, throat_dict):
        """Add attributes properties to an existing network.

        Parameters
        ----------
        pore_dict : dict
            Dictionary mapping attribute name to a dict of {node: value} for node attributes.
        throat_dict : dict
            Dictionary mapping attribute name to a dict of {edge: value} for edge attributes.
        """
        for attr_name, values in pore_dict.items():
            nx.set_node_attributes(self.graph, values, attr_name)

        for attr_name, values in throat_dict.items():
            nx.set_edge_attributes(self.graph, values, attr_name)

        self.compute_geometry()

    def set_radius_distribution(self, distributions_props, mode="pore", labels=None):
        """Assign random radii to pores or throats from one or more distributions.

        Generates 2x more radii than needed, shuffles them, and keeps the first n.
        For pore mode, throat radii are set to min(r(n1), r(n2)).
        For throat mode, pore radii are set to max of adjacent throat radii.

        Parameters
        ----------
        distributions_props : list of tuple
            List of (properties_dict, fraction) tuples. properties_dict must
            contain a "func" key with a distribution generator function (defaults
            to Stats.distribution) and any keyword arguments for that function.
            fraction is the occurrence probability of this distribution.
        mode : str, optional
            Either "pore" (assign to pores) or "throat" (assign to throats).
            Default is "pore".
        labels : list of int or None, optional
            Subset of node/edge indices to assign radii to. If None, all
            nodes (pore mode) or all edges (throat mode) are used.
        """

        from .Stats import distribution

        radii = []

        if mode not in ["pore", "throat"]:
            mode = "pore"

        if mode == "pore":
            n = self.graph.number_of_nodes()
            if labels is not None:
                indices = labels
            else:
                indices = self.graph.nodes

        elif mode == "throat":
            n = self.graph.number_of_edges()

        # On génère 2 x plus de rayons que de pores, puis on mélange et on prend les n premiers
        for properties, frac in distributions_props:
            dist = properties.get("func", distribution)
            r = list(dist(n_samples=int(np.ceil(frac * n * 2)), **properties))
            radii.extend(r)

        np.random.shuffle(radii)
        radii = radii[0:n]

        if mode == "pore":
            nx.set_node_attributes(self.graph, dict(zip(indices, radii)), "radius")
            for n1, n2 in self.graph.edges:
                try:
                    self.graph[n1][n2]["radius"] = min(self.graph.nodes[n1]["radius"], self.graph.nodes[n2]["radius"])
                except Exception as e:
                    warn("Cannot assign radius value to throat between nodes", n1, " and ", n2)

        elif mode == "throat":
            nx.set_edge_attributes(self.graph, dict(zip(self.graph.edges, radii)), "radius")
            for node in self.graph.nodes:
                try:
                    if self.graph.degree(node) > 0:
                        self.graph.nodes[node]["radius"] = np.array(
                            [self.graph[node][neighbor]["radius"] for neighbor in self.graph[node]]
                        ).max()
                except Exception as e:
                    warn("Cannot assign radius value to node", node)

    def get_pores_by_category(self, category, mode="match_one", nodes=None):
        """Filter pores by their category attribute.

        Parameters
        ----------
        category : str or list of str or tuple of str
            Side label(s) to match (e.g. "x-", ["x+", "y-"], "inner").
        mode : str, optional
            Matching mode:
            - "match_one": pore category intersects the query (at least one common label).
            - "match_all": query category is a subset of the pore category.
            - "not_in": pore category has no intersection with the query.
            - "equal": pore category is exactly equal to the query.
            Default is "match_one".
        nodes : iterable of int or None, optional
            Restrict the search to these node indices. If None, all nodes are
            considered.

        Returns
        -------
        list of int
            List of matching pore node indices.
        """

        if not (type(category) is list or type(category) is tuple):
            category = [category]

        category = set(category)

        if nodes is not None:
            pores = [(n, self.graph.nodes[n]["category"]) for n in nodes]
        else:
            pores = list(nx.get_node_attributes(self.graph, "category").items())

        if mode == "match_one":
            return [k for k, v in pores if len(set(v).intersection(category)) > 0]
        elif mode == "match_all":
            return [k for k, v in pores if category.issubset(v)]
        elif mode == "not_in":
            return [k for k, v in pores if len(set(v).intersection(category)) == 0]
        elif mode == "equal":
            return [k for k, v in pores if category == set(v)]

        # val = nx.get_node_attributes(self.graph,'category')
        # return filter(lambda n: category in val[n], self.graph.nodes)

    def write_props(self, filename=None):
        """Write network statistics to a JSON file (and print them).

        Computes and stores a dictionary of network properties in
        self.pnmprops: boundary pore counts per direction, number of pores and
        throats, porosity, pore-body and throat porosity, average connectivity,
        and extent.

        Parameters
        ----------
        filename : str or None, optional
            Path to the output JSON file. If None, stats are only printed.
        """
        import json

        self.pnmprops = {}
        self.pnmprops["Boundary pores"] = {}
        n_nodes = self.graph.number_of_nodes()
        if n_nodes > 0:
            avgc = sum([self.graph.degree(n) for n in self.graph.nodes]) / n_nodes
        else:
            avgc = 0.0
        for direction in ["x", "y", "z"]:
            self.pnmprops["Boundary pores"][direction] = len(self.get_pores_by_category([direction + "+"])) + len(
                self.get_pores_by_category([direction + "-"])
            )

        extent = self.graph.graph.get("extent")
        if extent is None:
            extent_prod = 1.0
        else:
            extent_prod = float(np.prod(np.array(extent, dtype=float)))
        self.pnmprops["Number of pores"] = n_nodes
        self.pnmprops["Number of throats"] = self.graph.number_of_edges()
        self.pnmprops["Porosity"] = self.graph.graph.get("porosity")
        self.pnmprops["Pore body Porosity"] = float(self.pores_volume) / extent_prod
        self.pnmprops["Throat Porosity"] = float(self.throats_volume) / extent_prod
        self.pnmprops["Average Connectivity"] = avgc
        extent_val = self.graph.graph.get("extent")
        self.pnmprops["Extent"] = list(extent_val) if extent_val is not None else None

        for key, val in self.pnmprops.items():
            print(key, val)

        if filename is not None:
            with open(filename, "w") as pnminfofile:
                json.dump(self.pnmprops, pnminfofile)
