# coding=utf-8

import numpy as np
import networkx as nx
from warnings import warn

# TODO:  clean the code. Remove unused functions.
#       Add support for periodicity -> Modify get_throats_length -> throats must have a fixed length attributes, get_throats_length should only return the attribute
# A new function should check the length

# Make sure that methods for adding or removing elements do not recompute geom or update porosity, extent or other parameters

side_to_int_dict = {"x-": 256, "x+": 512, "y-": 1024,
                    "y+": 2048, "z-": 4096, "z+": 8192, "none": 0, "inner": 0}
int_to_side_dict = {v: k for k, v in side_to_int_dict.items()}


class pore_network():

    """Ball and stick pore network model
    \n The graph is stored in a networkX graph
    \n geometry and other properties are stored in node and edge attributes"""

    def __init__(self, network=None):
        """network is a valid unidirectional networkX graph
        \n extent is a tuple (lx,ly,z) giving the extent of the PN """

        if network is None:
            self.graph = nx.Graph()
            self.graph.graph['graph_type'] = 'generic'
            # extent is the extent defined by pores surfaces
            self.graph.graph['extent'] = None
            self.graph.graph['bbox'] = None
            self.geom_complete = False
            self.pores_volume = 0
            self.throats_volume = 0
        else:
            self.graph = network
            self.compute_geometry()

    def __getitem__(self, n):

        return self.graph.nodes[n]

    @classmethod
    def read(cls, inputfilename):
        """Read an existing network in recommended pickle format"""

        # import json
        # with open(inputfilename, 'w') as infile:
        # data = json.load(infile)
        # g = nx.readwrite.json_graph.node_link_graph(data)
        # return cls(network = g)
        return cls(network=nx.read_gpickle(inputfilename))

    @classmethod
    def read_ascii(cls, inputfilename, data=True, legacy=False):
        # Read ascii network file used in imbibition simulations
        # if data == True, saturation, pressure values and ganglia labels are also imported
        # if legacy==True: reads old network format

        network = cls.__new__(cls)
        network.__init__()

        with open(inputfilename, 'r') as netfile:
            header = netfile.readline().split()
            network.graph.graph['extent'] = (
                float(header[0]), float(header[1]), float(header[2]))

            # Read and allocates pores: x,y,x,r,sw,pw,pnw,cc,side
            for i in range(int(header[3])):
                p = netfile.readline().split()
                side = int(p[-1])
                glabel = int(p[-2])
                p = list(map(float, p[:-2]))
                if side == 0:
                    cat = set(["inner"])
                else:
                    cat = set([int_to_side_dict[side & k]
                               for k in int_to_side_dict.keys() if side & k > 0])

                if data:
                    network.add_pore(
                        p[3], (p[0], p[1], p[2]), category=cat, sw=p[4], pw=p[5], pnw=p[6], glabel=glabel)
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
                    network.add_throat(pi, pj, radius=float(t[2]), length=float(
                        t[3]), sw=float(t[4]), glabel=int(t[5]))
                else:
                    network.add_throat(pi, pj, radius=float(
                        t[2]), length=float(t[3]))

        network.compute_geometry()

        return network

    def write(self, outfilename):
        """Write the network to a file using gpickle"""

        nx.write_gpickle(self.graph, outfilename)

    def save_as_ascii(self, outfilename):
        """Write the network in a text file using following format
        * lx,ly,lz,npores,nthroats  #Header
        * x,y,x,r,sw,pw,pnw,cc,side #npores lines
        * .
        * .
        * .
        * pi,pj,r,l,sw,cc,w_ind   #nthroats lines
        *
        * où
        * pour les pores:
        * x,y,z: coordonnées du centre du pore
        * r: rayon
        * sw: saturation fluide mouillant
        * pw: pression fluide mouillant
        * pnw pression non mouillant
        * cc: label de la composante connectée
        * side: face à laquelle le pore appartient
            * #sides are coded as follows :
              #256  x-
              #512  x+
              #1024 y-
              #2048 y+
              #4096 z-
              #8192 z+

        * pour les capillaires:
        * pi, pj: indexes des pores adjacents [commence à 0]
        * r: rayon,
        * l: longueur du capillaire
        * sw: saturation fluide mouillant
        * cc, label de la composante connectée
        * w_ind: index du pore du côté duquel se trouve le fluide mouillant
        """

        with open(outfilename, "w") as netfile:

            netfile.write("{} {} {} {} {}\n".format(
                self.graph.graph['extent'][0], self.graph.graph['extent'][1], self.graph.graph['extent'][2],
                self.graph.number_of_nodes(), self.graph.number_of_edges()))

            for node in self.graph.nodes:
                n = self.graph.nodes[node]
                side = sum([side_to_int_dict[c] for c in n['category']])
                netfile.write("{} {} {} {} {} {} {} {} {}\n".format(
                    n['center'][0], n['center'][1], n['center'][2], n['radius'], 0.0, 0.0, 0.0, 0, side))

            for n1, n2 in self.graph.edges:
                netfile.write("{} {} {} {} {} {} {}\n".format(
                    n1, n2, self.graph[n1][n2]['radius'], self.graph[n1][n2]['length'], 0.0, 0, 0))

    def check_overlapping(self, fit_radius=True, merge=True, mindist='auto', update_geometry=False):
        """Check for overlapping pores.  if fit_radius=True, radius are modified
        if pores cannot be fited or fit_radius = False, they are merged if merge=True or deleted if merge = False
        if update_geometry is True, throats length and radii are automatically recomputed"""

        from scipy.spatial.distance import cdist
        from scipy.spatial import cKDTree
        # index = list(self.graph)[:]
        # centers = np.array(list(zip(*nx.get_node_attributes(self.graph,'center').values()))).T
        # pores_radii = np.fromiter(nx.get_node_attributes(self.graph,'radius').values(),dtype=np.float)

        pores_radii = list(nx.get_node_attributes(
            self.graph, 'radius').items())
        # we begin by the bigger pores
        pores_radii.sort(key=lambda tup: tup[1], reverse=True)
        index, pores_radii = zip(*pores_radii)
        pores_radii = np.array(pores_radii)

        centers = nx.get_node_attributes(self.graph, 'center')
        centers = [np.array(centers[i]) for i in index]
        centers = np.array(centers)
        # distances = cdist(centers,centers)
        kdtree = cKDTree(centers)

        stop = False

        while not stop:

            stop = True

            for i, n1 in enumerate(index):

                #distances = cdist(centers,[self.graph.nodes[n1]['center']])[:,0]

                if self.graph.has_node(n1):

                    if mindist == 'auto':
                        gap = self.graph.nodes[n1]['radius']*0.02
                    else:
                        gap = mindist

                    labels = kdtree.query_ball_point(
                        self.graph.nodes[n1]['center'], 2.5*self.graph.nodes[n1]['radius'])
                    labels.remove(i)
                    # distances,labels = kdtree.query(x=net.graph.nodes[n1]['center'],2*self.graph.nodes[n1]['radius'],n_jobs=1)
                    # labels.remove(i)
                    #distance *= 0.998
                    distances = cdist(centers[labels], [self.graph.nodes[n1]['center']])[
                        :, 0]*0.998
                    d = distances - pores_radii[labels]
                    d -= self.graph.nodes[n1]['radius']
                    # On commence par la distance la plus faible
                    d_and_labels = [(d[j], k) for j, k in enumerate(labels)]
                    d_and_labels.sort(key=lambda t: t[0])

                    for (dist, ind) in d_and_labels:

                        n2 = index[ind]
                        if self.graph.has_node(n2) and self.graph.has_node(n1):

                            # Le centre du pore né est dans la sphère du pore n1 OU il y a overlapping et fit_radius == False
                            # -> Merging ou suppression du pore de plus petit rayon
                            if (dist + self.graph.nodes[n2]['radius'] <= gap) or (dist < gap and dist + self.graph.nodes[n2]['radius'] > gap and not fit_radius):

                                if (self.graph.nodes[n1]['radius'] >= self.graph.nodes[n2]['radius']):
                                    if merge:
                                        self.merge_pores(n1, n2)
                                        print("pore", n1, "and", n2,
                                              "overlap: merging (deleting", n2, ")")
                                    else:
                                        self.remove_pore(n2)
                                        print("pore", n1, "and", n2,
                                              "overlap: deleting", n2)

                                else:
                                    if merge:
                                        self.merge_pores(n2, n1)
                                        print("pore", n1, "and", n2,
                                              "overlap: merging (deleting", n1, ")")
                                    else:
                                        self.remove_pore(n1)
                                        print("pore", n1, "and", n2,
                                              "overlap: deleting", n2)
                                    # On termine l'itération car le pore n1 n'existe plus...
                                    break

                            # Overlapping et fit_radius == True
                            # 3 options:
                            # -Le rayon du pore le plus petit est modifié
                            # -Merging
                            # -Suppression
                            elif dist < gap and dist + self.graph.nodes[n2]['radius'] > gap and fit_radius:
                                if (self.graph.nodes[n1]['radius'] >= self.graph.nodes[n2]['radius']):
                                    r = dist + \
                                        self.graph.nodes[n2]['radius'] - \
                                        self.graph.nodes[n1]['radius'] - gap
                                    if self.graph.nodes[n2]['radius'] >= r and r > 0:
                                        self.graph.nodes[n2]['radius'] = r
                                        pores_radii[ind] = r
                                        print(
                                            "pore", n1, "and", n2, "overlap: changin radius of", n2, "to", r)
                                    else:
                                        if merge:
                                            self.merge_pores(n1, n2)
                                            print(
                                                "pore", n1, "and", n2, "overlap: merging (deleting", n2, ")")
                                        else:
                                            self.remove_pore(n2)
                                            print("pore", n1, "and", n2,
                                                  "overlap: deleting", n2)
                                else:
                                    if self.graph.nodes[n1]['radius'] >= dist:
                                        self.graph.nodes[n1]['radius'] = dist
                                        pores_radii[i] = dist
                                        print(
                                            "pore", n1, "and", n2, "overlap: changin radius of", n1, "to", dist)
                                    else:
                                        if merge:
                                            self.merge_pores(n2, n1)
                                            print(
                                                "pore", n1, "and", n2, "overlap: merging (deleting", n1, ")")
                                        else:
                                            self.remove_pore(n1)
                                            print("pore", n1, "and", n2,
                                                  "overlap: deleting", n1)
                                        # On termine l'itération car le pore n1 n'existe plus...
                                        break

        if update_geometry:
            self.set_auto_throats_length()
            self.set_auto_throats_radius()

    def compute_extent(self):

        pores_radii = np.fromiter(nx.get_node_attributes(
            self.graph, 'radius').values(), dtype=np.float)
        minbox = np.array(
            list(zip(*nx.get_node_attributes(self.graph, 'center').values()))).T
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

        self.graph.graph['bbox'] = [xmin, ymin, zmin, xmax, ymax, zmax]

        self.graph.graph['extent'] = np.zeros(3)
        self.graph.graph['extent'][0] = abs(xmax - xmin)
        self.graph.graph['extent'][1] = abs(ymax - ymin)
        self.graph.graph['extent'][2] = abs(zmax - zmin)

    def compute_porosity(self):

        if self.geom_complete:

            self.graph.graph['porosity'] = 0

            pore_volumes = nx.get_node_attributes(self.graph, 'volume')
            self.pores_volume = np.fromiter(
                pore_volumes.values(), dtype=np.float).sum()

            throat_volumes = nx.get_edge_attributes(self.graph, 'volume')
            self.throats_volume = np.fromiter(
                throat_volumes.values(), dtype=np.float).sum()

            # Compute the extent from pores center and radii
            if self.graph.graph.get('extent', None) is None:

                self.compute_extent()

            self.graph.graph['porosity'] = (
                self.pores_volume + self.throats_volume) / np.prod(self.graph.graph['extent'])

        else:
            warn("Geometry not complete. Cannot compute porosity")

    def _compute_pore_vol(self, n):

        return (4/3)*np.pi*self.graph.nodes[n]['radius']**3

    def _compute_throat_vol(self, n1, n2):

        return self.graph[n1][n2]['length']*np.pi*self.graph[n1][n2]['radius']**2

    def compute_throats_volume(self):

        if self.geom_complete:

            throats_radii = np.fromiter(nx.get_edge_attributes(
                self.graph, 'radius').values(), dtype=np.float)
            throats_length = np.fromiter(nx.get_edge_attributes(
                self.graph, 'length').values(), dtype=np.float)
            v = throats_length*np.pi*throats_radii**2

            volumes = dict(zip(self.graph.edges, v))

            nx.set_edge_attributes(self.graph, volumes, 'volume')

            self.throats_volume = v.sum()

        else:
            warn("Geometry not complete. Cannot compute throat volumes")

    def compute_pores_volume(self):

        if self.geom_complete:

            pores_radii = np.fromiter(nx.get_node_attributes(
                self.graph, 'radius').values(), dtype=np.float)
            v = (pores_radii**3)*np.pi*4/3
            volumes = dict(zip(self.graph.nodes, v))

            nx.set_node_attributes(self.graph, volumes, 'volume')

            self.pores_volume = v.sum()

        else:
            warn("Geometry not complete. Cannot compute pore volumes")

    def remove_isolated_pores(self, only_inner_pores=False):

        if only_inner_pores:
            nodes = self.get_pores_by_category("inner", mode="equal")
        else:
            nodes = list(self.graph.nodes.keys())

        for n in nodes:
            if self.graph.degree(n) == 0:
                print("Removing isolated pore", n)
                self.remove_pore(n)

    def automatic_geometry(self):
        """Set throats length and radii automatically. Erase custom length/radius !"""
        self.set_auto_throats_length()
        self.set_auto_throats_radius()

    def compute_geometry(self, autothroats=False):
        """Compute geometry properties.
        Check if geometry is complete"""

        self.pores_volume = 0
        self.throats_volume = 0

        self.geom_complete = True

        self.remove_isolated_pores()

        self.graph = nx.convert_node_labels_to_integers(self.graph)

        if len(nx.get_node_attributes(self.graph, 'radius')) != self.graph.number_of_nodes():
            self.geom_complete = False
            warn("Cannot compute geometry properties. Some pores do not have a radius !")

        if len(nx.get_node_attributes(self.graph, 'center')) != self.graph.number_of_nodes():
            self.geom_complete = False
            warn("Cannot compute geometry properties. Some pores do not have a center !")

        if self.graph.number_of_nodes() == 0 or self.graph.number_of_edges() == 0:
            self.geom_complete = False
            warn(
                "Cannot compute geometry properties. Number of nodes and/or throats is 0 !")

        if autothroats:
            self.automatic_geometry()
        else:
            if len(nx.get_edge_attributes(self.graph, 'radius')) != self.graph.number_of_edges():
                self.geom_complete = False
                warn(
                    "Cannot compute geometry properties. Some throats do not have a radius !")
            if len(nx.get_edge_attributes(self.graph, 'length')) != self.graph.number_of_edges():
                self.geom_complete = False
                warn(
                    "Cannot compute geometry properties. Some throats do not have a length !")

        if self.geom_complete:
            self.compute_extent()
            self.compute_pores_volume()
            self.compute_throats_volume()
            self.compute_porosity()

        return self.geom_complete

    def merge_pores(self, n1, n2, setcategory='union', radius=None, center=None, check_throats=True, inner_category='inner', verbose=False):
        """Merge two pores n1 and n2
        setcategory: if "union", the category of the pore is the union of n1 and n2
        radius: merged pore radius. if not provided, it keeps the n1 radius
        center: if not provided, it keeps the n1 center
        check_throats: check throats radius so that their radius is the min of two connected pore
        """

        if not self.graph.has_node(n1) or not self.graph.has_node(n2):
            warn("Nodes {} or {} does not exist. Cannot merge them".format(u, v))
            return
        elif verbose:
            print("Merging pore {} and {}".format(n1, n2))

        if center is not None:
            self.graph.nodes[n1]['center'] = center

        if radius is not None:
            self.graph.nodes[n1]['radius'] = radius

        category = self.graph.nodes[n2]['category']

        if setcategory == 'union':
            self.graph.nodes[n1]['category'] = self.graph.nodes[n1]['category'].union(
                category)
            if len(self.graph.nodes[n1]['category']) > 1 and 'inner' in self.graph.nodes[n1]['category']:
                self.graph.nodes[n1]['category'] = self.graph.nodes[n1]['category'].difference(
                    set(['inner']))

        # if not G.has_edge(u,v):
            # warn("Nodes {} and {} will be merged but they are not adjacent".format(u,v))

        # Warning : here we just copy the old edge attributes to the new one, so that the attributes are already defined.
        # The values must however be checked !
        new_edges = [(n1, n3, d)
                     for _, n3, d in self.graph.edges(n2, data=True)
                     if (n3 != n1 and n3 != n2)]
        try:
            self.graph.add_edges_from(new_edges)
        except:
            warn(
                'Error trying to create new edges when merging pores {} and {}'.format(n1, n2))
            warn('Edges list {}'.format(new_edges))

        self.graph.remove_node(n2)

        if check_throats:
            for n3 in self.graph[n1]:
                self._compute_auto_throat_length(n1, n3)
                self._compute_auto_throat_radius(n1, n3)

    def _compute_auto_throat_radius(self, n1, n2):

        try:
            return min(self.graph.nodes[n1]['radius'], self.graph.nodes[n2]['radius'])
        except KeyError:
            warn("Cannot compute auto radius of throat between {} and {}".format(n1, n2))
            return None

    def _compute_auto_throat_length(self, n1, n2):

        try:
            delta = np.array(
                self.graph.nodes[n1]['center']) - np.array(self.graph.nodes[n2]['center'])
            return max(0, np.sqrt(np.dot(delta, delta)) - self.graph.nodes[n1]['radius'] - self.graph.nodes[n2]['radius'])
        except KeyError:
            warn("Cannot compute auto length of throat between {} and {}".format(n1, n2))
            return None

    def set_auto_throats_radius(self):
        """Compute throat radius as min(r(n1),r(n2)) """

        for n1, n2 in self.graph.edges:
            self.graph[n1][n2]['radius'] = self._compute_auto_throat_radius(
                n1, n2)

    def set_auto_throats_length(self):
        """Compute throats length [not for throats used to enforce periodicity]. If l<0 (i.e. pores overlap), set it to 0.
        \nAdd 'length' attribute to edges """

        for n1, n2 in self.graph.edges:
            if self.graph[n1][n2]['category'] != "periodic":
                self.graph[n1][n2]['length'] = self._compute_auto_throat_length(
                    n1, n2)


    def get_pore_distance(self, n1, n2):
        # TODO: take into account periodic BCs !s

        try:
            delta = np.array(
                self.graph.nodes[n1]['center']) - np.array(self.graph.nodes[n2]['center'])
            return np.sqrt(np.dot(delta, delta))

        except KeyError:
            warn("Cannot Compute distance between pores {} and {}".format(n1, n2))
            return 0

    def get_throat_section(self, n1, n2):

        try:
            return np.pi*self.graph[n1][n2]['radius']**2

        except KeyError:
            warn("Cannot Compute section of throat between pores {} and {}".format(n1, n2))
            return 0

    def update_porosity(self):
        """just update the porosity using throats and pores volumes. Do not iterate through elements """

        if self.geom_complete:
            self.graph.graph['porosity'] = (
                self.pores_volume + self.throats_volume) / np.prod(self.graph.graph['extent'])

    def add_pore(self, radius, center, category=set(), **kwargs):

        n = self.graph.number_of_nodes()

        while self.graph.has_node(n):
            n += 1

        self.graph.add_node(n, radius=radius, center=center,
                            volume=4*np.pi*radius**3/3, category=category, **kwargs)

    def add_throat(self, n1, n2, radius=None, length=None, **kwargs):
        # TODO either remove update porosity or compute throat volume

        if self.graph.has_node(n1) and self.graph.has_node(n2):

            if radius is None:
                radius = self._compute_auto_throat_radius(n1, n2)
            if length is None:
                length = self._compute_auto_throat_length(n1, n2)

            self.graph.add_edge(n1, n2, radius=radius, length=length, **kwargs)

        else:
            warn(
                "Cannot create throats between {} and {}: nodes does not exist".format(n1, n2))

    def remove_throat(self, n1, n2):

        try:
            self.graph.remove_edge(n1, n2)
        except:
            pass

    def remove_pore(self, n, verbose=False):

        try:
            self.graph.remove_node(n)
            if verbose:
                print("removing pore", n)
        except:
            pass

    def clear_category(self, nodes=None):

        if nodes is None:
            nodes = self.graph.nodes

        for n in nodes:
            self.graph.nodes[n]['category'] = set()

    def clear_throats(self):
        """Remove all throats"""

        print("removing all throats")
        self.graph.remove_edges_from(list(self.graph.edges))

    def add_attributes(self, pore_dict, throat_dict):
        """Add attributes properties to an existing network"""

        self.graph.add_node_attributes(self.graph, pore_dict)
        self.graph.add_edge_attributes(self.graph, throat_dict)

        self.compute_geometry()

    def set_radius_distribution(self, distributions_props, mode='pore', labels=None):
        """Properties is a list of tuple (dictionary, probability) containing properties of each distribution
        dict contains name and kwargs passed to the generator. Generator function is given by the key "func"
        \n and probability of occurence"""

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
            r = list(dist(n_samples=int(np.ceil(frac*n*2)), **properties))
            radii.extend(r)

        np.random.shuffle(radii)
        radii = radii[0:n]

        if mode == "pore":
            nx.set_node_attributes(self.graph, dict(
                zip(indices, radii)), 'radius')
            for n1, n2 in self.graph.edges:
                try:
                    self.graph[n1][n2]['radius'] = min(
                        self.graph.nodes[n1]['radius'], self.graph.nodes[n2]['radius'])
                except:
                    warn(
                        "Cannot assign radius value to throat between nodes", n1, " and ", n2)

        elif mode == "throat":
            nx.set_edge_attributes(self.graph, dict(
                zip(self.graph.edges, radii)), 'radius')
            for node in self.graph.nodes:
                try:
                    if self.graph.degree(node) > 0:
                        self.graph.nodes[node]['radius'] = np.array(
                            [self.graph[node][neighbor]['radius'] for neighbor in self.graph[node]]).max()
                except:
                    warn("Cannot assign radius value to node", node)

    def get_pores_by_category(self, category, mode='match_one', nodes=None):
        """Return pore labels if either:
        \n intersection between pores['category'] and provided category is non empty (mode='match_one')
        \n provided category is a subset of pores['category'] (mode='match_all')
        \n provided category not in pores['category'] (mode='not_in')
        \n provided category is the only category in pores['category'] (mode='equal')
        \n nodes: a list of nodes to limit the search to
        Category: list of pores labels """

        if not isinstance(category, list):
            category = [category]

        category = set(category)

        if nodes is not None:
            pores = [(n, self.graph.nodes[n]['category']) for n in nodes]
        else:
            pores = list(nx.get_node_attributes(
                self.graph, 'category').items())

        if mode == 'match_one':
            return [k for k, v in pores if len(v.intersection(category)) > 0]
        elif mode == 'match_all':
            return [k for k, v in pores if category.issubset(v) > 0]
        elif mode == 'not_in':
            return [k for k, v in pores if len(v.intersection(category)) == 0]
        if mode == 'equal':
            return [k for k, v in pores if category == v]

        # val = nx.get_node_attributes(self.graph,'category')
        # return filter(lambda n: category in val[n], self.graph.nodes)

    def write_props(self, filename=None):
        """Write the network stats to a json file
        """
        import json

        self.pnmprops = {}
        self.pnmprops["Boundary pores"] = {}
        avgc = sum([self.graph.degree(n)
                    for n in self.graph.nodes])/self.graph.number_of_nodes()
        for direction in ["x", "y", "z"]:
            self.pnmprops["Boundary pores"][direction] = (len(self.get_pores_by_category([direction+"+"])) +
                                                          len(self.get_pores_by_category([direction+"-"])))

        self.pnmprops["Number of pores"] = self.graph.number_of_nodes()
        self.pnmprops["Number of throats"] = self.graph.number_of_edges()
        self.pnmprops['Porosity'] = self.graph.graph["porosity"]
        self.pnmprops['Pore body Porosity'] = self.pores_volume / \
            np.prod(self.graph.graph['extent'])
        self.pnmprops['Throat Porosity'] = self.throats_volume / \
            np.prod(self.graph.graph['extent'])
        self.pnmprops["Average Connectivity"] = avgc
        self.pnmprops['Extent'] = list(self.graph.graph.get('extent'))

        for key, val in self.pnmprops.items():
            print(key, val)

        if filename is not None:
            with open(filename, 'w') as pnminfofile:
                json.dump(self.pnmprops, pnminfofile)
