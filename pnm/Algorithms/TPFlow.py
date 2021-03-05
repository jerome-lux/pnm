# coding=utf-8

import numpy as np
import networkx as nx
from warnings import warn
import scipy.sparse as sp
from copy import deepcopy
from scipy.sparse.linalg import bicg, bicgstab, cg, cgs, gmres, lgmres, lgmres, minres, qmr, gcrotmk, spilu, LinearOperator
import math
from collections import deque

# Some CSTS
PATM = 1e5
WATER_VISC = 0.001005
AIR_VISC = 18.5e-6
WET_ANGLE = 0
DEF_SIGMA = 0.0728

implemented_bc = ['Dirichlet', 'Neumann']
implemented_solvers = {'cg': cg,
                       'bicgstab': bicgstab,
                       'bicg': bicg,
                       'cgs': cgs,
                       'gmres': gmres,
                       'lgmres': lgmres,
                       'minres': minres,
                       'qmr': qmr,
                       'gcrotmk': gcrotmk}


# TODO check neumann conditions [pc are put in b for connection between free and neumann node]

class TPFlow():

    """Simple Two-phase flow in pore network"""

    def __init__(self, pn):

        self.pn = deepcopy(pn)
        # IMPORTANT: the graph nodes MUST be labeled using consecutive integers from 0 to n-1 where n is the number of nodes !
        self.pn.graph = nx.convert_node_labels_to_integers(pn.graph)
        self.bc_dict = {}
        self.throat_model = None

        # Check if necessary steps are OK
        self.bc_OK = False
        self.throat_model_OK = False
        self.params_OK = False
        self.physic_OK = False
        self.setup_OK = False

    def set_physics(self,
                    wf_visc=WATER_VISC,
                    nwf_visc=AIR_VISC,
                    sigma=DEF_SIGMA,
                    wetting_angle=WET_ANGLE,
                    p0=PATM):

        self.wf_viscosity = wf_viscosity  # Pa.s
        self.nwf_viscosity = nwf_viscosity  # Pa.s
        self.sigma = sigma  # N/m
        self.wetting_angle = wetting_angle  # in radian
        self.p0 = p0  # Pa, initiale pressure value for pressure field
        self.physic_OK = True

    def set_params(self, **kwargs):
        """
        max_iter: number of time steps
        stop_if_all_trapped: stops simulation if all ganglia are trapped
        avgsw_stopping_criterion: stops simulation if average sw does not vary more than the criterion (absolute variation of the saturation
        averaged on i-200 to i-100 and i to i-100)
        max_delta_swp: maximum varation of saturation in a pore during a time step
        max_delta_swt: maximum varation of saturation in a throat during a time step
        dt_min: minimal dt, even if computed dt using max_delta_swp or max_delta_swt are smaller
        rounding_val: if sw <rounding_val or 1-sw <rounding_val, then round the saturation to 0 and 1 respectively
        """
        self.max_iter = kwargs.get('max_iter', 10000)
        self.stop_if_all_trapped = kwargs.get('stop_if_all_trapped', True)
        self.avgsw_stopping_criterion = kwargs.get(
            'avgsw_stopping_criterion', 1e-3)
        self.max_delta_swp = kwargs.get('max_delta_swp', 0.1)
        self.max_delta_swt = kwargs.get('max_delta_swc', 0.334)
        self.dt_min = kwargs.get('dt_min', 1e-8)
        self.rounding_val = kwargs.get('rounding_val', 1e-10)

        self.params_OK = True

    def psolve(self, solver='cg', pfield0=None, tol=1e-10, maxiter=None):

        # if 'solver' not in implemented solver, 'solver' is assumed to be a user-provided solver
        # with input prameters: A sparse scipy matrix size NxN, b numpy array of size N, x0 initial value for the solution, maxiter
        # Must return solution and iterations
        solver = implemented_solvers.get(solver, solver)

        try:
            self.pfield, self.iter = solver(
                self.A, self.b, x0=pfield0, tol=tol, maxiter=maxiter)  # ,M=M
        except:
            warn('Solving linear system failed. You should check matrix and RHS')

        return self.iter

    def _get_coef(self, i, j):

        if self.get_config(i, j) > 0:
            return self.get_conductance(i, j)
        else:
            return 0

    def get_conductance(self, i, j):

        # Two-phase conductance between pore i and j

        swi = max(1, 2*self.swp[i])
        swj = max(1, 2*self.swp[j])

        return (np.pi*self.radius(i, j)**4 /
                (8*(
                    self.pn.graph[i][j]['length']*(
                        self.pn.graph[i][j]['sw']*self.wf_viscosity +
                        (1-self.pn.graph[i][j]['sw'])*self.nwf_viscosity) +
                    self.pn.graph.nodes[i]['radius']*(swi*self.wf_viscosity + (1-swi)*self.nwf_viscosity) +
                    self.pn.graph.nodes[j]['radius'] *
                    (swj*self.wf_viscosity + (1-swj)*self.nwf_viscosity)
                ))
                )

    def get_pc(self, i, j):
        """Get capillary pressure between node i and j. either negative or positive depedending on the position of the wf/nwf
        If the config is not valid, return 0
        """

        if self.get_config(i, j) == 2:
            return 2 * self.sigma * np.cos(self.wetting_angle) / self.radius(i, j)
        elif self.get_config(i, j) == 3:
            return - 2 * self.sigma * np.cos(self.wetting_angle) / self.radius(i, j)
        else:
            return 0

    def get_config(self, i, j):
        """Return fluid config number between to pores i and j. If a config is not valid return 0 """

        if self.swp[i] >= 1 and self.swp[j] >= 1 and self.pn.graph[i][j]['sw'] >= 1:
            # tout est saturé de fluide mouillant.
            return 1
        elif (self.swp[i] >= 1 and self.swp[j] < 1 and
              ((self.pn.graph[i][j]['wf_side'] == i and self.pn.graph[i][j]['sw'] > 0) or self.pn.graph[i][j]['sw'] <= 0)):
            # pore i saturé, pore j non saturé. Swc quelconque, mais fluide mouillant du côté i si capillaire non vide
            return 2
        elif (self.swp[i] < 1 and self.swp[j] >= 1 and
              ((self.pn.graph[i][j]['wf_side'] == j and self.pn.graph[i][j]['sw'] > 0) or self.pn.graph[i][j]['sw'] <= 0)):
            # pore i non saturé, pore j saturé. Swc quelconque symétrique de 2
            return 3
        elif self.swp[i] <= 0 and self.swp[j] <= 0 and self.pn.graph[i][j]['sw'] <= 0:
            # tout est saturé de fluide non mouillant.
            return 4

        return 0

    def setup_simulation(self):

        # Adjacency matrix
        # convert to lil for fast slicing ops
        self.adjm = nx.adj_matrix(self.pn.graph).tolil()

        self.inlet_pores = list(
            filter(lambda x: self.bc_dict[x]['sw'] > 0, self.bc_dict.keys()))
        self.outlet_pores = list(
            filter(lambda x: self.bc_dict[x]['sw'] <= 0, self.bc_dict.keys()))

        nx.set_node_attributes(self.pn.graph, dict(
            zip(self.inlet_pores, 'inlet')), 'boundary')
        nx.set_node_attributes(self.pn.graph, dict(
            zip(self.outlet_pores, 'outlet')), 'boundary')

        # Pore Saturation is stored in array, while throat sw is stored directly in graph
        self.swp = np.zeros(self.pn.graph.number_of_nodes(), dtype=np.float64)
        # Set pore saturation to 1 in inlet
        self.swp[self.inlet_pores] = 1

        # Set throats saturation to 1 between two inlet pores and to 0 otherwise
        nx.set_edge_attributes(self.pn.graph, values=0.0, name='sw')
        nx.set_edge_attributes(G, values={(u, v): 1 for u, v in G.edges self.swp[u] == 1 and self.swp[v] == 1}, name='sw')

        # pressure in pore (either pw if sw=1 or pnw otherwise)
        self.p = np.zeros(self.pn.graph.number_of_nodes(),
                          dtype=np.float64) + self.p0

        # De quel côté est le fluide mouillant entre deux pores i et j: -1 = pas d'interface
        nx.set_edge_attributes(self.pn.graph, -1, 'wf_side')

        # Labels of the connected components of unsatured pores
        self.CClabels = np.zeros(
            self.pn.graph.number_of_nodes(), dtype=np.int32)
        # status of pores: 0=trapped, 1=connected to the outlet
        self.node_status = np.zeros(
            self.pn.graph.number_of_nodes(), dtype=np.int8)

        self.setup_OK = True

    def run(self):
        if not self.bc_OK:
            print("You must set at least on boundary condition")
            return -1
        if not self.throat_model_OK:
            print("You must choose a throat model. Using default")
            self.setup_throat_radius_model()
        if not self.params_OK:
            print("Simulation's parameters are not defined. Using default")
            self.set_params()
        if not self.physic_OK:
            print("Physical parameters not defined. Using default")
            self.set_physics()

        self.setup_simulation()

    def update_log(self):
        pass

    def update_connected_components(self):
        """Find all connected components of unsaturated pores.
        A connected component not connected to the outlet is trapped
        """

        self.CC_number = 0
        # Reset CC labels
        self.CClabels[:] = 0
        # node status
        self.node_status[:] = 0

        component_found = True

        # index of unsaturated pores
        unsaturated_pores = np.where(self.swp < 1)

        while component_found:

            component_found = False

            for n in unsaturated_pores:
                if self.CClabels == 0:
                    self.CC_number += 1
                    self.CClabels[n] = self.CC_number
                    self.label_connected_pores(n, self.CC_number)
                    component_found = True

    def label_connected_pores(self, n1, CClabel):
        """Label all pores connected to pore n1 by empty throats"""
        plist = deque([n1])
        ccnodes = [n1]
        status = 0
        while plist:
            i = str(plist.pop())
            if(self.pn.graph.nodes[i]['boundary'] == 'outlet'):
                status = 1
            nlist = filter(
                lambda x: self.pn.graph[i][x]['sw'] <= 0 and self.CClabels[x] == 0, self.pn.graph.neighbors(i))
            plist.extend(nlist)
            ccnodes.extend(nlist)

        self.CClabels[ccnodes] = CClabel
        self.node_status[ccnodes] = status  # Conected or not to outlet

    def update_sw(self):
        pass

    def compute_dt(self):
        pass

    def compute_flow(self):
        pass

    def setup_coef_matrix(self):
        """ Create coefficient matrix and RHS vector """

        if self.model is None or len(self.bc_dict) == 0:
            warn("You must set up the Boundary conditions and diffusivity model before building the matrix")
            return

        n = self.pn.graph.number_of_nodes()
        adjm = deepcopy(self.adjm)

        # put capillary pressures in b
        self.b = np.zeros(n)
        rows, cols = adjm.nonzero()
        # pc(i,j)=0 if one phase or no valid config.
        pcij = np.array([-self._get_coef(i, j)*self.get_pc(i, j)
                         for i, j in zip(rows, cols)])
        PCIJ = sp.coo_matrix((pcij, (rows, cols)), shape=(
            n, n), dtype=np.float64).tocsr()

        for i in self.pn.graph.nodes:
            a = PCIJ.getrow(i).toarray()
            self.b += a[0, :]

        # Boundary conditions
        dirichlet_nodes = {}
        neumann_nodes = {}

        dirichlet_nodes = list(
            filter(lambda x: self.bc_dict[x]['bctype'] == 'Dirichlet', self.bc_dict.keys()))
        neumann_nodes = list(
            filter(lambda x: self.bc_dict[x]['bctype'] == 'Neumann', self.bc_dict.keys()))

        # ajout des CL neumann au second membre
        if (len(neumann_nodes)) > 0:
            values = [self.bc_dict[i]['value']*np.pi *
                      self.pn.graph.nodes[i]['radius']**2 for i in neumann_nodes]
            # values est directement la densité de flux imposée*section du pore externe
            self.b[neumann_nodes] = - np.array(values)

        active_nodes = list(
            set(self.pn.graph.nodes).difference(dirichlet_nodes))

        # ajout des CL Dirichlet au second membre
        if len(dirichlet_nodes) > 0:
            adjm1 = deepcopy(adjm)
            # Composantes extra diagonales des pores aux limites.
            # On ne conserve que les coefficients qui correspondent à un flux entre noeud libre-noeud aux limites
            # On met à zéro les lignes de la matrice d'adjacence qui corresondent aux noeuds libres
            adjm1[active_nodes] = 0
            # On met à zéro les termes qui correspondent au flux entre deux noeuds aux limites sur les colonnes
            adjm1[:, dirichlet_nodes] = 0
            brows, bcols = adjm1.nonzero()
            bcij = np.array([self._get_coef(i, j)*self.bc_dict[i]['value']
                             for (i, j) in zip(brows, bcols)])
            T = sp.coo_matrix((bcij, (brows, bcols)), shape=(
                n, n), dtype=np.float64).tocsr()

            for i in dirichlet_nodes:
                a = T.getrow(i).toarray()
                self.b += a[0, :]

            # Put Boundary values in b for boundary nodes
            values = [self.bc_dict[i]['value'] for i in dirichlet_nodes]
            self.b[dirichlet_nodes] = np.array(values)

        # Génération des sum(aij) ou i est un noeud libre pour le calcul des termes diagonaux
        adjm[dirichlet_nodes] = 0
        diag, all_cols = adjm.nonzero()
        aii = np.array([self._get_coef(i, j)
                        for (i, j) in zip(diag, all_cols)])

        # Génération des termes extra diagonaux uniquement entre les noeuds libres
        adjm[:, dirichlet_nodes] = 0
        rows, cols = adjm.nonzero()
        aij = np.array([self._get_coef(i, j) for (i, j) in zip(rows, cols)])
        # Termes diagonaux pour les noeuds fixés
        bcdiagvalues = np.ones(len(dirichlet_nodes))

        data = np.concatenate((aii, -aij, bcdiagvalues))
        all_rows = np.concatenate((diag, cols, dirichlet_nodes))
        all_cols = np.concatenate((diag, rows, dirichlet_nodes))

        self.A = sp.coo_matrix((data, (all_rows, all_cols)), shape=(
            n, n), dtype=np.float64).tocsr()

    def setup_throat_radius_model(self, model='throat', alpha=1):
        """model is either throat or effective
        throat is just the throat radius
        Effective model is based on https://doi.org/10.1016/j.micromeso.2013.09.038"""

        if model not in ["throat", "effective"]:
            warn(
                "Radius Model should be either 'throat' or 'effective'. Set it to default (throat)")
            model = 'throat'

        print("Using ", model, " radius model")

        self.throat_model = model

        if model == 'throat':
            def radius(ni, nj):
                return self.pn.graph[ni][nj]['radius']

        elif model == 'effective':
            def radius(ni, nj):
                ri = self.pn.graph.nodes[ni]['radius']
                rj = self.pn.graph.nodes[nj]['radius']
                lij = self.pn.graph[ni][nj]['length']
                #lij = self.pn.get_pore_distance(ni,nj)
                return 0.5*(ri*(rj/np.sqrt(rj**2+lij**2))**alpha + rj*(ri/np.sqrt(ri**2+lij**2))**alpha)

        self.radius = radius

    def add_bc(self, bc_val, sw, nodes, bc_type='Dirichlet'):
        """ Set up boundary conditions for a given set of nodes
        \nif mode='Dirichlet', value is the node concentration
        \nif mode ='Neumann', value is the input flux density
        \n sw is the wetting fluid saturation.
        Boundary pores are assumed to be connected to the outside world.
        Boundary pores with sw = 0 are implicitly connected to the outlet, while boundary pores  with sw > 0 are connected to the ioutlet
        if positive, the value corresponds to a positive flux in boundary pores (from the outside)"""

        if bc_type not in implemented_bc:
            warn('Only Dirichlet and Neumann BC are implemented. Using Dirichlet BC')

        # make sure 'nodes' is iterable
        try:
            _ = iter(nodes)
        except TypeError:
            nodes = [nodes]

        #self.bc_list.append({'bctype':bc_type,'values':{i:bc_val for i in nodes}})
        self.bc_dict.update(
            {i: {'value': bc_val, 'bctype': bc_type, 'phase': sw} for i in nodes})
