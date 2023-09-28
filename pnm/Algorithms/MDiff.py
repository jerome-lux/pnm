# coding=utf-8

import numpy as np
import networkx as nx
from warnings import warn
import scipy.sparse as sp
from copy import deepcopy
from scipy.sparse.linalg import bicg, bicgstab, cg, cgs, gmres, lgmres, minres, qmr, gcrotmk, spilu, LinearOperator
import math
from collections import deque

implemented_bc = ["Dirichlet", "Neumann"]
implemented_solvers = {
    "cg": cg,
    "bicgstab": bicgstab,
    "bicg": bicg,
    "cgs": cgs,
    "gmres": gmres,
    "lgmres": lgmres,
    "minres": minres,
    "qmr": qmr,
    "gcrotmk": gcrotmk,
}

DEFAULT_DKNUDSEN = (1 / 6) * np.sqrt((8 * 8.3144 * 293) / (np.pi * 44 * 1e-3))
# Diff CO2 dans l'air à 20° 0.16 cm2/s


class molecular_diffusion:

    """Fix Neumann BC conditions to set flux density and not flux->OK ?"""

    def __init__(self, pn):
        self.pn = pn
        # IMPORTANT: the graph nodes MUST be labeled using consecutive integers from 0 to n-1 where n is the number of nodes !
        self.pn.graph = nx.convert_node_labels_to_integers(pn.graph)
        self.diffusivity = None
        self.bc_dict = {}
        self.model = None
        self.throat_model = None

    def solve(self, solver="cg", x0=None, tol=1e-6, maxiter=None):
        # if 'solver' not in implemented solver, 'solver' is assumed to be a user-provided solver
        solver = implemented_solvers.get(solver, solver)

        # Lorsque la matrice contient tous les noeuds (dont noeuds aux frontières) le préconditionnement spilu ralenti la convergence
        # ilu = spilu(self.A.tocsc())
        # Mx = lambda x: ilu.solve(x)
        # M = LinearOperator(self.A.shape, Mx)

        try:
            self.x, self.iter = solver(self.A, self.b, x0=x0, tol=tol, maxiter=maxiter)  # ,M=M
        except Exception as e:
            warn("Solving linear system failed. You should check matrix and RHS")

        return self.iter

    def get_concentration(self, n1):
        # bctype = self.bctype_dict.get(n1,None)

        # if bctype is None:
        # return self.x[self.nodes_to_indices[n1]]
        # elif bctype == "Dirichlet":
        # return self.bcvalues_dict[n1]
        # elif bctype == "Neumann":
        # for n2 in [ n for n in self.pn.graph[n1] if self.bctype_dict.get(n2,None) is None]: #Adjacent inner pores
        # return (self.pn.get_throat_length(n1,n2)*self.bcvalues_dict[n1] /
        # (self.pn.get_throat_section(n1,n2)*self.diffusivity(n1,n2)) + self.x[self.nodes_to_indices[n2]])
        try:
            return self.x[n1]
        except:
            warn("Concentration field must be computed !")
            return None

    def compute_boundary_flow(self, bc_nodes):
        """Compute sum of molar flow in provided  boundary nodes"""

        bc_nodes = set(bc_nodes)  # make sure that there are not any duplicates

        active_nodes = list(set(self.pn.graph.nodes).difference(bc_nodes))

        adjm = nx.adjacency_matrix(self.pn.graph).tolil()
        # Preparing adjacency matrix
        # delete coefficients between two boundary nodes [cols]
        adjm[:, list(bc_nodes)] = 0
        # delete coefficients between two free nodes [rows]
        adjm[active_nodes] = 0
        rows, cols = adjm.nonzero()
        try:
            qij_1 = np.array(
                [
                    self._flow_coef(i, j) * (self.x[j] - self.bc_dict[i]["value"])
                    for (i, j) in zip(rows, cols)
                    if self.bc_dict[i]["bctype"] == "Dirichlet"
                ]
            ).sum()
        except KeyError:
            warn("bc_nodes must be a list of boundary nodes only")

        try:
            qij_2 = np.array(
                [
                    self.bc_dict[i]["value"] * np.pi * self.pn.graph.nodes[i]["radius"] ** 2
                    for (i, j) in zip(rows, cols)
                    if self.bc_dict[i]["bctype"] == "Neumann"
                ]
            ).sum()
        except KeyError:
            warn("bc_nodes must be a list of boundary nodes only")

        return qij_1 + qij_2

    def setup_coef_matrix(self):
        """Create coefficient matrix and RHS vector"""

        if self.model is None or len(self.bc_dict) == 0:
            warn("You must set up the Boundary conditions and diffusivity model before building the matrix")
            return

        n = self.pn.graph.number_of_nodes()
        adjm = nx.adjacency_matrix(self.pn.graph).tolil()  # for fast slicing ops

        self.b = np.zeros(n)

        # Boundary conditions
        dirichlet_nodes = {}
        neumann_nodes = {}

        dirichlet_nodes = list(filter(lambda x: self.bc_dict[x]["bctype"] == "Dirichlet", self.bc_dict.keys()))
        neumann_nodes = list(filter(lambda x: self.bc_dict[x]["bctype"] == "Neumann", self.bc_dict.keys()))

        # ajout des CL neumann au second membre
        if (len(neumann_nodes)) > 0:
            values = [self.bc_dict[i]["value"] * np.pi * self.pn.graph.nodes[i]["radius"] ** 2 for i in neumann_nodes]
            self.b[neumann_nodes] = -np.array(values)

        active_nodes = list(set(self.pn.graph.nodes).difference(dirichlet_nodes))

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
            bcij = np.array([self._flow_coef(i, j) * self.bc_dict[i]["value"] for (i, j) in zip(brows, bcols)])
            T = sp.coo_matrix((bcij, (brows, bcols)), shape=(n, n), dtype=np.float64).tocsr()

            for i in dirichlet_nodes:
                a = T.getrow(i).toarray()
                self.b += a[0, :]

            values = [self.bc_dict[i]["value"] for i in dirichlet_nodes]
            self.b[dirichlet_nodes] = np.array(values)

        # Génération des sum(aij) ou i est un noeud libre pour le calcul des termes diagonaux
        adjm[dirichlet_nodes] = 0
        diag, all_cols = adjm.nonzero()
        aii = np.array([self._flow_coef(i, j) for (i, j) in zip(diag, all_cols)])

        # Génération des termes extra diagonaux uniquement entre les noeuds libres
        adjm[:, dirichlet_nodes] = 0
        rows, cols = adjm.nonzero()
        aij = np.array([self._flow_coef(i, j) for (i, j) in zip(rows, cols)])
        # Termes diagonaux pour les noeuds fixés
        bcdiagvalues = np.ones(len(dirichlet_nodes))

        data = np.concatenate((aii, -aij, bcdiagvalues))
        all_rows = np.concatenate((diag, cols, dirichlet_nodes))
        all_cols = np.concatenate((diag, rows, dirichlet_nodes))

        self.A = sp.coo_matrix((data, (all_rows, all_cols)), shape=(n, n), dtype=np.float64).tocsr()

    def _flow_coef(self, n1, n2):
        return self.diffusivity(n1, n2) * np.pi * self.radius(n1, n2) ** 2 / self.pn.get_pore_distance(n1, n2)

    def setup_diffusivity_model(self, model="Fick", scale=1, **kwargs):
        """Setup the diffusivity values. scale is used to rescale the diffusivity if e.g. the network dimans are in micrometers, let scale = 1e6
        model is either 'Fick', 'Knudsen' or 'All'"""

        if model not in ["Fick", "Knudsen", "All"]:
            warn(
                "setup_diffusivity: model must be in {} \nSet it to default (Fick)".format(["Fick", "Knudsen", "All"])
            )
            self.model = "Fick"
        else:
            self.model = model

        if self.throat_model is None:
            self.setup_throat_radius_model()

        print("Using ", model, " diffusivity model(s)")

        # Knudsen diffusivity without the radius term
        D_Knudsen = kwargs.get("d_knudsen", DEFAULT_DKNUDSEN)
        # Bulk diffusivity is a constant given by the user
        D_Fick = kwargs.get("d_fick", 1)

        if self.model == "Fick":

            def diffusivity(n1, n2):
                return D_Fick * scale**2

        elif self.model == "Knudsen":

            def diffusivity(n1, n2):
                return self.radius(n1, n2) * D_Knudsen * scale

        elif self.model == "All":

            def diffusivity(n1, n2):
                return 1 / (1 / (D_Fick * scale**2) + 1 / (self.radius(n1, n2) * D_Knudsen * scale))

        self.diffusivity = diffusivity

    def setup_throat_radius_model(self, model="throat", alpha=1):
        """model is either throat or effective
        throat is just the throat radius
        Effective model is based on https://doi.org/10.1016/j.micromeso.2013.09.038"""

        if model not in ["throat", "effective", "series"]:
            warn("Radius Model should be either 'throat' or 'effective'. Set it to default (throat)")
            model = "throat"

        print("Using ", model, " radius model")

        self.throat_model = model

        if model == "throat":

            def radius(ni, nj):
                return self.pn.graph[ni][nj]["radius"]

        elif model == "effective":

            def radius(ni, nj):
                ri = self.pn.graph.nodes[ni]["radius"]
                rj = self.pn.graph.nodes[nj]["radius"]
                lij = self.pn.graph[ni][nj]["length"]
                # lij = self.pn.get_pore_distance(ni,nj)
                return 0.5 * (
                    ri * (rj / np.sqrt(rj**2 + lij**2)) ** alpha + rj * (ri / np.sqrt(ri**2 + lij**2)) ** alpha
                )

        elif model == "series":

            def radius(ni, nj):
                # minr = min( self.pn.graph.nodes[ni]['radius'], self.pn.graph.nodes[nj]['radius'])
                maxr = max(self.pn.graph.nodes[ni]["radius"], self.pn.graph.nodes[nj]["radius"])
                L = self.pn.get_pore_distance(ni, nj)
                rij = self.pn.graph[ni][nj]["radius"]
                # longeur du pore du centre jusqu'à l'intersection du capillaire
                H = math.sqrt(maxr**2 - rij**2)
                # rayon "moyen" de 0 à H
                Reff = maxr * math.sin(math.acos(rij / maxr))
                return (Reff * H + rij * (L - H)) / L

        self.radius = radius

    def add_bc(self, bc_val, nodes, bc_type="Dirichlet"):
        """Set up boundary conditions for a given set of nodes
        \nif mode='Dirichlet', value is the node concentration
        \nif mode ='Neumann', value is the molar flux per area (to be multiplied by the pore section)
        if positive, the value corresponds to a positive flux in boundary pores (from the outside)"""

        if bc_type not in implemented_bc:
            warn("Only Dirichlet and Neumann BC are implemented. Using Dirichlet BC")

        # make sure 'nodes' is iterable
        try:
            _ = iter(nodes)
        except TypeError:
            nodes = [nodes]

        # self.bc_list.append({'bctype':bc_type,'values':{i:bc_val for i in nodes}})
        self.bc_dict.update({i: {"value": bc_val, "bctype": bc_type} for i in nodes})


class molecular_diffusion_deprecated:
    def __init__(self, pn):
        self.pn = pn
        self.diffusivity = None
        self.bc_dict = {}
        self.bcvalues_dict = {}
        self.bctype_dict = {}
        self.model = None
        self.throat_model = None

    def solve(self, solver="cg", x0=None, tol=1e-6, maxiter=None, precond=False):
        # if 'solver' not in implemented solver, 'solver' is assumed to be a user-provided solver
        solver = implemented_solvers.get(solver, solver)

        if precond:
            ilu = spilu(self.A.tocsc())

            def Mx(x):
                return ilu.solve(x)

            M = LinearOperator(self.A.shape, Mx)
            try:
                self.x, self.iter = solver(self.A, self.b, x0=x0, tol=tol, maxiter=maxiter, M=M)
            except Exception as e:
                warn("Solving linear system failed. You should check matrix and RHS")
        else:
            try:
                self.x, self.iter = solver(self.A, self.b, x0=x0, tol=tol, maxiter=maxiter)
            except Exception as e:
                warn("Solving linear system failed. You should check matrix and RHS")

        return self.iter

    def get_pores_concentration_dict(self):
        return {n: self.get_concentration(n) for n in self.pn.graph.nodes}

    def get_concentration(self, n1):
        bctype = self.bctype_dict.get(n1, None)

        if bctype is None:
            return self.x[self.nodes_to_indices[n1]]
        elif bctype == "Dirichlet":
            return self.bcvalues_dict[n1]
        elif bctype == "Neumann":
            # Adjacent inner pores
            for n2 in [n for n in self.pn.graph[n1] if self.bctype_dict.get(n2, None) is None]:
                return (
                    self.pn.graph[n1][n2]["length"]
                    * self.bcvalues_dict[n1]
                    / (self.pn.get_throat_section(n1, n2) * self.diffusivity(n1, n2))
                    + self.x[self.nodes_to_indices[n2]]
                )

    def compute_flow(self, bc_nodes):
        """Compute sum of molar flow in provided  bc_nodes"""

        bc_nodes_set = set(bc_nodes)

        q = 0

        for n1 in bc_nodes:
            for n2 in self.pn.graph[n1]:
                if n2 not in bc_nodes_set:
                    f1, f2 = self.bc_values(n1, n2)
                    try:
                        val = self.x[self.nodes_to_indices[n2]]
                    except KeyError:
                        val = self.bcvalues_dict[n2]
                    q += -f1 + f2 * val

        return q

    def setup_coef_matrix(self):
        # TODO: make this faster ! use of adjacency matrix and csgraph routines in scipy ?
        """Create coefficient matrix and RHS vector for given list of nodes"""

        if self.model is None or len(self.bc_dict) == 0:
            warn("You must set up the Boundary conditions and diffusivity model before building the matrix")
            return

        bc_nodes = set(self.bc_dict)

        active_nodes = tuple(set(self.pn.graph.nodes).difference(bc_nodes))
        nnodes = len(active_nodes)

        data = []
        rows = []
        cols = []
        self.b = np.zeros(nnodes)
        self.nodes_to_indices = {n: i for i, n in enumerate(active_nodes)}

        for i, n1 in enumerate(active_nodes):
            diag = 0
            btemp = 0
            for n2 in self.pn.graph[n1]:
                if n2 not in bc_nodes:
                    rows.append(i)
                    cols.append(self.nodes_to_indices[n2])
                    val = self._flow_coef(n1, n2)
                    diag += val  # Adding to diagonal term - save for later
                    data.append(-val)  # non diagonal term
                else:
                    bc1, bc2 = self.bc_values(n1, n2)
                    btemp += bc1
                    diag += bc2

            rows.append(i)
            cols.append(i)
            data.append(diag)
            self.b[i] = btemp

        self.A = sp.coo_matrix((data, (rows, cols)), shape=(nnodes, nnodes), dtype=np.float64).tocsr()

    def _flow_coef(self, n1, n2):
        return self.diffusivity(n1, n2) * np.pi * self.radius(n1, n2) ** 2 / self.pn.get_pore_distance(n1, n2)

    def setup_diffusivity_model(self, model="Fick", scale=1, **kwargs):
        """model is either 'Fick', 'Knudsen' or 'All'"""

        if model not in ["Fick", "Knudsen", "All"]:
            warn(
                "setup_diffusivity: model must be in {} \nSet it to default (Fick)".format(["Fick", "Knudsen", "All"])
            )
            self.model = "Fick"
        else:
            self.model = model

        if self.throat_model is None:
            self.setup_throat_radius_model()

        print("Using ", model, " diffusivity model(s)")

        # Knudsen diffusivity without the radius term
        D_Knudsen = kwargs.get("d_knudsen", DEFAULT_DKNUDSEN)
        # Bulk diffusivity is a constant given by the user
        D_Fick = kwargs.get("d_fick", 1)

        if self.model == "Fick":

            def diffusivity(n1, n2):
                return D_Fick * scale**2

        elif self.model == "Knudsen":

            def diffusivity(n1, n2):
                return self.radius(n1, n2) * D_Knudsen * scale

        elif self.model == "All":

            def diffusivity(n1, n2):
                return 1 / (1 / (D_Fick * scale**2) + 1 / (self.radius(n1, n2) * D_Knudsen * scale))

        self.diffusivity = diffusivity

    def setup_throat_radius_model(self, model="throat", alpha=1):
        """model is either throat or effective
        Effective model is based on https://doi.org/10.1016/j.micromeso.2013.09.038"""

        if model not in ["throat", "effective", "series"]:
            warn("Radius Model should be either 'throat' or 'effective'. Set it to default")
            model = "throat"

        print("Using ", model, " radius model")

        self.throat_model = model

        if model == "throat":

            def radius(ni, nj):
                return self.pn.graph[ni][nj]["radius"]

        elif model == "effective":

            def radius(ni, nj):
                ri = self.pn.graph.nodes[ni]["radius"]
                rj = self.pn.graph.nodes[nj]["radius"]
                lij = self.pn.graph[ni][nj]["length"]
                # lij = self.pn.get_pore_distance(ni,nj)
                return 0.5 * (
                    ri * (rj / np.sqrt(rj**2 + lij**2)) ** alpha + rj * (ri / np.sqrt(ri**2 + lij**2)) ** alpha
                )

        elif model == "series":

            def radius(ni, nj):
                # minr = min( self.pn.graph.nodes[ni]['radius'], self.pn.graph.nodes[nj]['radius'])
                maxr = max(self.pn.graph.nodes[ni]["radius"], self.pn.graph.nodes[nj]["radius"])
                L = self.pn.get_pore_distance(ni, nj)
                rij = self.pn.graph[ni][nj]["radius"]
                # longeur du pore du centre jusqu'à l'intersection du capillaire
                H = math.sqrt(maxr**2 - rij**2)
                # rayon "moyen" de 0 à H
                Reff = maxr * math.sin(math.acos(rij / maxr))
                return (Reff * H + rij * (L - H)) / L

        self.radius = radius

    def bc_values(self, n1, n2):
        for n in [n1, n2]:
            try:
                return self.bc_dict[n](n1, n2)
            except KeyError:
                pass

        warn("Trying to aply BC to non boundary pores {} and {}".format(n1, n2))
        return 0, 0

    def add_bc(self, bc_val, nodes, bc_type="Dirichlet"):
        """Set up boundary conditions for given set of nodes
        \nif mode='Dirichlet', value is the node concentration
        \nif mode ='Neumann', value is the molar flux **density** if positive value correspond to a positive flux in adj pores
        """

        if self.diffusivity is None:
            warn("You must set the diffusivity model before the boundary conditions")

        if not isinstance(nodes, list):
            nodes = [nodes]

        if bc_type not in implemented_bc:
            warn("Only Dirichlet and Neumann BC are implemented. Using Dirichlet BC")

        if bc_type == "Dirichlet":

            def get_bc(n1, n2):
                coef = self._flow_coef(n1, n2)
                # returns contribution for the RHS and the matrix diagonal coeff
                return bc_val * coef, coef

        elif bc_type == "Neumann":

            def get_bc(n1, n2):
                # returns contribution for the RHS and the matrix diagonal coeff
                return -bc_val * np.pi * self.radius(n1, n2) ** 2, 0

        # Saving coef, values and type
        for n in nodes:
            self.bc_dict[n] = get_bc
            self.bcvalues_dict[n] = bc_val
            self.bctype_dict[n] = bc_type

    # def set_source(self,labels,value):
    # """ Apply source term to nodes in labels. Matrix and RHS vector must be already computed"""
