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
    """Molecular diffusion simulation on a pore network.

    Solves the steady-state diffusion equation on a graph-based pore network
    using finite-volume discretization. Boundary conditions (Dirichlet or
    Neumann) can be applied to individual pores. The linear system is solved
    using iterative solvers from scipy.sparse.linalg.

    The coefficient matrix is built in a vectorized manner: all throat
    conductances are pre-computed in one pass, and the sparse matrix is
    assembled directly in COO format (avoiding slow LIL slicing and deepcopy).

    Parameters
    ----------
    pn : pore_network
        The pore network model instance. The graph nodes must be labeled with
        consecutive integers from 0 to n-1 (enforced by
        ``nx.convert_node_labels_to_integers`` in ``__init__``).
        Must have node attributes ``radius`` and ``volume``, and edge
        attributes ``radius`` and ``length`` (e.g. after
        ``pore_network.compute_geometry()``).

    Attributes
    ----------
    pn : pore_network
        The pore network instance.
    diffusivity : callable or None
        Function ``diffusivity(n1, n2)`` returning the diffusivity between two
        pores. Set by ``setup_diffusivity_model``.
    bc_dict : dict
        Dictionary mapping node index to ``{"value": float, "bctype": str}``.
    model : str or None
        Diffusivity model name ("Fick", "Knudsen", or "All").
    throat_model : str or None
        Throat conductance model name ("throat", "effective", "path",
        "conduit" or "conduit_shared"; see ``setup_throat_model``).
    A : scipy.sparse.csr_matrix or None
        Coefficient matrix (built by ``setup_coef_matrix``).
    b : numpy.ndarray or None
        Right-hand side vector (built by ``setup_coef_matrix``).
    x : numpy.ndarray or None
        Solution vector (concentration field, set by ``solve``).
    """

    def __init__(self, pn):
        self.pn = pn
        # IMPORTANT: graph nodes MUST be labeled using consecutive integers
        # from 0 to n-1 where n is the number of nodes.
        self.pn.graph = nx.convert_node_labels_to_integers(pn.graph)
        self.diffusivity = None
        self.bc_dict = {}
        self.model = None
        self.throat_model = None
        self.A = None
        self.b = None
        self.x = None

    def solve(self, solver="cg", x0=None, tol=1e-6, maxiter=None):
        """Solve the linear system for the concentration field.

        Parameters
        ----------
        solver : str or callable, optional
            Solver name (one of ``implemented_solvers`` keys) or a user-provided
            solver callable with signature ``solver(A, b, x0=..., tol=..., maxiter=...)``.
            Default is "cg".
        x0 : array-like or None, optional
            Initial guess for the iterative solver. Default is None.
        tol : float, optional
            Convergence tolerance. Default is 1e-6.
        maxiter : int or None, optional
            Maximum number of iterations. Default is None (solver-dependent).

        Returns
        -------
        int
            Number of iterations performed (or 0 on failure).

        Notes
        -----
        If ``setup_coef_matrix`` has not been called, ``self.A`` and ``self.b``
        will be None and the solver will fail.
        """
        if self.A is None or self.b is None:
            warn("Coefficient matrix and RHS must be built with setup_coef_matrix before solving.")
            return 0

        # if 'solver' not in implemented_solvers, 'solver' is assumed to be
        # a user-provided solver callable
        solver_fn = implemented_solvers.get(solver, solver)

        try:
            # scipy >= 1.0 uses 'rtol'; older versions may use 'tol'
            try:
                self.x, self.iter = solver_fn(
                    self.A, self.b, x0=x0, rtol=tol, maxiter=maxiter
                )
            except TypeError:
                self.x, self.iter = solver_fn(
                    self.A, self.b, x0=x0, tol=tol, maxiter=maxiter
                )
        except Exception as e:
            print("Exception occurred when solving linear system:", e)
            warn("Solving linear system failed. Check matrix and RHS.")
            self.iter = 0

        return self.iter

    def get_concentration(self, n1):
        """Return the concentration at a given node.

        Parameters
        ----------
        n1 : int
            Node index.

        Returns
        -------
        float or None
            Concentration at node ``n1``, or None if the system has not been
            solved yet.
        """
        try:
            return self.x[n1]
        except (AttributeError, TypeError, IndexError):
            warn("Concentration field must be computed before calling get_concentration.")
            return None

    def compute_boundary_flow(self, bc_nodes):
        """Compute the sum of molar flow through the given boundary nodes.

        For Dirichlet nodes, the flow is computed from the concentration
        gradient across adjacent throats. For Neumann nodes, the flow is
        the imposed flux density multiplied by the pore cross-section.

        Parameters
        ----------
        bc_nodes : iterable of int
            List of boundary node indices.

        Returns
        -------
        float
            Total molar flow through the boundary nodes.
        """
        if self.x is None:
            warn("Concentration field must be computed before computing boundary flow.")
            return 0.0

        bc_nodes = set(bc_nodes)  # remove possible duplicates
        active_nodes = set(self.pn.graph.nodes).difference(bc_nodes)

        total_flow = 0.0

        for i in bc_nodes:
            if i not in self.bc_dict:
                continue
            bctype = self.bc_dict[i]["bctype"]
            bc_value = self.bc_dict[i]["value"]

            if bctype == "Dirichlet":
                # Sum of flows from node i to adjacent active (free) pores
                for j in self.pn.graph[i]:
                    if j in active_nodes:
                        coef = self._flow_coef(i, j)
                        total_flow += coef * (self.x[j] - bc_value)
            elif bctype == "Neumann":
                # Imposed flux density multiplied by pore cross-section
                pore_radius = self.pn.graph.nodes[i]["radius"]
                total_flow += bc_value * np.pi * pore_radius**2

        return total_flow

    def setup_coef_matrix(self):
        """Build the coefficient matrix A and RHS vector b for the diffusion problem.

        The matrix is assembled in a single vectorized pass:
        1. All throat conductances are pre-computed for every edge.
        2. Diagonal terms (sum of conductances for free nodes) are accumulated
           with ``np.add.at``.
        3. Off-diagonal terms (-conductance between two free nodes) are added
           in both directions.
        4. Dirichlet boundary contributions are added to the RHS of adjacent
           free nodes.
        5. Dirichlet nodes get a unit diagonal entry (A[i,i] = 1).

        The resulting system is stored in ``self.A`` (CSR format) and
        ``self.b``.

        Notes
        -----
        Requires ``self.model`` and ``self.bc_dict`` to be set (via
        ``setup_diffusivity_model`` and ``add_bc``). Node attributes ``radius``
        and edge attributes ``radius`` and ``length`` must exist on the graph.

        Performance
        -----------
        Unlike the previous implementation which used ``nx.adjacency_matrix``
        with slow LIL slicing and ``deepcopy``, this version builds the matrix
        directly in COO format from pre-computed arrays. The dominant cost is
        the single pass over all edges to compute conductances.
        """
        if self.model is None or len(self.bc_dict) == 0:
            warn("Boundary conditions and diffusivity model must be set before building the matrix.")
            return

        n = self.pn.graph.number_of_nodes()
        if n == 0:
            warn("Network has no nodes.")
            return

        edges = list(self.pn.graph.edges())
        if len(edges) == 0:
            warn("Network has no edges.")
            return

        # ------------------------------------------------------------------
        # 1. Pre-compute edge attributes and conductances in one pass
        # ------------------------------------------------------------------
        n1_list = np.array([e[0] for e in edges], dtype=int)
        n2_list = np.array([e[1] for e in edges], dtype=int)

        # Vectorized conductance computation (single pass over all edges)
        flow_coefs = np.array(
            [self._flow_coef(int(i), int(j)) for i, j in zip(n1_list, n2_list)],
            dtype=float,
        )

        # ------------------------------------------------------------------
        # 2. Classify nodes by BC type
        # ------------------------------------------------------------------
        is_dirichlet = np.zeros(n, dtype=bool)
        is_neumann = np.zeros(n, dtype=bool)
        bc_values = np.zeros(n)

        for node, bc in self.bc_dict.items():
            bctype = bc["bctype"]
            bval = bc["value"]
            if bctype == "Dirichlet":
                is_dirichlet[node] = True
                bc_values[node] = bval
            elif bctype == "Neumann":
                is_neumann[node] = True
                bc_values[node] = bval

        active_mask = ~is_dirichlet  # free (unknown) nodes

        # ------------------------------------------------------------------
        # 3. Initialize RHS vector
        # ------------------------------------------------------------------
        self.b = np.zeros(n, dtype=float)

        # Neumann BC: RHS = -value * pore_section
        neumann_nodes = np.where(is_neumann)[0]
        if neumann_nodes.size > 0:
            pore_radii = np.array(
                [self.pn.graph.nodes[int(nn)]["radius"] for nn in neumann_nodes]
            )
            neumann_values = bc_values[neumann_nodes] * np.pi * pore_radii**2
            self.b[neumann_nodes] = -neumann_values

        # ------------------------------------------------------------------
        # 4. Diagonal terms: sum of conductances for each free node
        # ------------------------------------------------------------------
        diag_contrib = np.zeros(n, dtype=float)

        # For each edge (i, j) with conductance c:
        #   if i is free: diag[i] += c
        #   if j is free: diag[j] += c
        free_n1 = active_mask[n1_list]
        free_n2 = active_mask[n2_list]

        np.add.at(diag_contrib, n1_list[free_n1], flow_coefs[free_n1])
        np.add.at(diag_contrib, n2_list[free_n2], flow_coefs[free_n2])

        # ------------------------------------------------------------------
        # 5. Off-diagonal terms between free nodes (-conductance, both dirs)
        # ------------------------------------------------------------------
        rows_list = []
        cols_list = []
        data_list = []

        both_free = free_n1 & free_n2
        if np.any(both_free):
            # (i, j) and (j, i) for symmetry
            rows_list.append(n1_list[both_free])
            cols_list.append(n2_list[both_free])
            data_list.append(-flow_coefs[both_free])
            rows_list.append(n2_list[both_free])
            cols_list.append(n1_list[both_free])
            data_list.append(-flow_coefs[both_free])

        # ------------------------------------------------------------------
        # 6. Dirichlet contributions to RHS of adjacent free nodes
        # ------------------------------------------------------------------
        # Edge (i, j) with i Dirichlet, j free: b[j] += c * value_Dirichlet(i)
        dir_n1 = is_dirichlet[n1_list] & active_mask[n2_list]
        if np.any(dir_n1):
            contrib = flow_coefs[dir_n1] * bc_values[n1_list[dir_n1]]
            np.add.at(self.b, n2_list[dir_n1], contrib)

        # Edge (i, j) with j Dirichlet, i free: b[i] += c * value_Dirichlet(j)
        dir_n2 = is_dirichlet[n2_list] & active_mask[n1_list]
        if np.any(dir_n2):
            contrib = flow_coefs[dir_n2] * bc_values[n2_list[dir_n2]]
            np.add.at(self.b, n1_list[dir_n2], contrib)

        # ------------------------------------------------------------------
        # 7. Assemble diagonal entries
        # ------------------------------------------------------------------
        free_nodes = np.where(active_mask)[0]
        rows_list.append(free_nodes)
        cols_list.append(free_nodes)
        data_list.append(diag_contrib[free_nodes])

        # Dirichlet nodes: unit diagonal (A[i,i] = 1)
        dirichlet_nodes = np.where(is_dirichlet)[0]
        if dirichlet_nodes.size > 0:
            rows_list.append(dirichlet_nodes)
            cols_list.append(dirichlet_nodes)
            data_list.append(np.ones(dirichlet_nodes.size, dtype=float))
            self.b[dirichlet_nodes] = bc_values[dirichlet_nodes]

        # ------------------------------------------------------------------
        # 8. Build sparse matrix
        # ------------------------------------------------------------------
        if rows_list:
            all_rows = np.concatenate(rows_list)
            all_cols = np.concatenate(cols_list)
            all_data = np.concatenate(data_list)
            self.A = sp.coo_matrix(
                (all_data, (all_rows, all_cols)), shape=(n, n), dtype=float
            ).tocsr()
        else:
            self.A = sp.csr_matrix((n, n), dtype=float)

    def _flow_coef(self, n1, n2):
        """Compute the throat conductance between two pores.

        Conductance = diffusivity * throat_factor, where throat_factor
        encodes the geometry (e.g. pi * r^2 / L for the "throat" model).

        Parameters
        ----------
        n1, n2 : int
            Node indices of the two pores.

        Returns
        -------
        float
            Throat conductance.

        Notes
        -----
        For the conduit models ("conduit", "conduit_shared") the conductance is
        that of three segments in series (pore i half, throat, pore j half),
        each with its own diffusivity evaluated at the segment's channel radius
        (per-segment Knudsen): G = 1 / sum_s F_s / D(rho_s), where F_s is the
        geometric resistance factor of segment s (integral of dx/A(x)).
        """
        if self.throat_model in ("conduit", "conduit_shared"):
            resistance = 0.0
            for factor, rho in self._conduit_segments(n1, n2):
                if factor > 0.0:
                    resistance += factor / self.segment_diffusivity(rho)
            return 1.0 / resistance if resistance > 0.0 else 0.0
        return self.diffusivity(n1, n2) * self.throat_factor(n1, n2)

    def setup_diffusivity_model(self, model="Fick", scale=1, **kwargs):
        """Set up the diffusivity model.

        Parameters
        ----------
        model : str, optional
            Diffusivity model: "Fick" (constant bulk), "Knudsen"
            (radius-dependent), or "All" (combined resistance).
            Default is "Fick".
        scale : float, optional
            Scaling factor applied to the diffusivity (e.g. 1e-6 if network
            dimensions are in micrometers). Default is 1.
        **kwargs
            Additional model parameters:
            - ``d_knudsen`` : Knudsen diffusivity constant (default from
              ``DEFAULT_DKNUDSEN``).
            - ``d_fick`` : Fickian bulk diffusivity (default 1).

        Notes
        -----
        Sets ``self.diffusivity`` to a callable ``diffusivity(n1, n2)``.
        If ``self.throat_model`` is not yet set, ``setup_throat_model`` is
        called with default parameters.
        """
        if model not in ["Fick", "Knudsen", "All"]:
            warn(
                "setup_diffusivity: model must be in ['Fick', 'Knudsen', 'All']. "
                "Using default (Fick)."
            )
            self.model = "Fick"
        else:
            self.model = model

        if self.throat_model is None:
            self.setup_throat_model()

        print("Using", model, "diffusivity model(s)")

        # Knudsen diffusivity without the radius term
        D_Knudsen = kwargs.get("d_knudsen", DEFAULT_DKNUDSEN)
        # Bulk diffusivity (constant given by the user)
        D_Fick = kwargs.get("d_fick", 1)

        # Diffusivity as a function of a local channel radius r. Used directly
        # by the conduit models (per-segment Knudsen) and, through
        # self.radius(n1, n2), by the per-edge models.
        if self.model == "Fick":

            def segment_diffusivity(r):
                return D_Fick * scale**2

        elif self.model == "Knudsen":

            def segment_diffusivity(r):
                return r * D_Knudsen * scale

        else:  # "All": Fick and Knudsen resistances in series

            def segment_diffusivity(r):
                return 1 / (1 / (D_Fick * scale**2) + 1 / (r * D_Knudsen * scale))

        def diffusivity(n1, n2):
            return segment_diffusivity(self.radius(n1, n2))

        self.segment_diffusivity = segment_diffusivity
        self.diffusivity = diffusivity

    def setup_throat_model(self, model="throat", alpha=1, share="area"):
        """Set up the throat conductance (radius) model.

        Parameters
        ----------
        model : str, optional
            Throat model: "throat" (simple throat radius), "effective"
            (effective radius based on
            https://doi.org/10.1016/j.micromeso.2013.09.038), "path"
            (path-length conductance), "conduit" (pore-throat-pore conduit,
            spheres and cylinders) or "conduit_shared" (conduit with the pore
            volume partitioned among its throats). See Notes. Default is
            "throat".
        alpha : float, optional
            Exponent for the "effective" model. Default is 1.
        share : str, optional
            Pore-volume partition rule for "conduit_shared": "area" (share of
            pore i given to throat t is w = r_t^2 / sum_k r_k^2 over the throats
            of i) or "uniform" (w = 1/degree(i)). Ignored by the other models.
            Default is "area".

        Notes
        -----
        Sets ``self.throat_factor`` to a callable ``throat_factor(n1, n2)``
        and ``self.radius`` to a callable ``radius(n1, n2)``.

        The "conduit" model treats each edge as three resistances in series
        between the two pore centers (the lumped nodes):

            1/G = F_i/D(rho_i) + F_t/D(r_t) + F_j/D(rho_j),

        * pore half i: the sphere's cross-section pi*(r_i^2 - x^2) integrated
          from the center (x=0) to the junction with the throat cylinder,
          x_i = sqrt(r_i^2 - r_t^2):
              F_i = ln((r_i + x_i)/r_t) / (pi*r_i)
          (stable closed form of (1/(2 pi r_i)) ln((r_i+x_i)/(r_i-x_i)));
        * throat: cylinder of radius r_t between the two junctions,
              F_t = L'_t/(pi*r_t^2),  L'_t = max(|c_i - c_j| - x_i - x_j, 0)
          (L'_t >= stored gap length L_t; the pore caps inside the cylinder
          radius belong to the throat);
        * rho_i = sqrt(x_i/(pi*F_i)) is the equivalent radius of the pore half
          (uniform cylinder with the same length and resistance); it is used
          for the per-segment Knudsen diffusivity, so the Knudsen penalty of
          narrow throats is not applied to the wide pore bodies.

        This is the standard "spheres and cylinders" conduit model (OpenPNM,
        Gostick et al. 2016; pore-throat-pore series as in Valvatne & Blunt
        2004). It uses the three radii r_i, r_j, r_t, stays finite when the gap
        vanishes (G -> 1/(F_i/D + F_j/D)), and gives the pores the finite
        diffusivity of the fluid they contain instead of treating them as
        perfect conductors. It does not, however, guarantee Deff/Dinf <= 1:
        every conduit of a pore uses the pore's full cross-section, so the pore
        volume is counted ~degree/2 times.

        The "conduit_shared" model is identical except that each pore half's
        resistance is divided by the share w of the pore assigned to that
        throat (F_i -> F_i/w_i), with sum_t w_i,t = 1 over the throats of pore
        i. The conduit volumes then partition the pore space, and a rigorous
        bound follows (Cauchy-Schwarz: R_c >= l_c^2/(D V_c) for any series
        conduit of length l_c and volume V_c; Dirichlet principle with the
        linear test potential; l_c = |c_i - c_j| >= its projection on the flow
        axis):

            Deff/Dinf <= sum_c V_c / V_box <= (V_pore + V_throat)/V_box,

        up to the small offset of the boundary pores from the box faces. The
        model is therefore incapable of aberrant (Deff > phi_total*Dinf)
        results by construction; it is conservative for the small throats of
        highly connected pores. The Knudsen radius rho_i stays the physical
        (unshared) one: the partition is a volume-accounting device, not a
        narrowing of the channel.

        In both conduit models periodic edges use the unwrapped center
        distance, like "path", so they are nearly inert in open-box Dirichlet
        simulations.

        The "path" model uses the conductance

            G = D * pi * r_t^2 / L_eff,   L_eff = max(L_t + pi*r_t/2, |c_i - c_j|),

        where L_t is the stored throat (gap) length and |c_i - c_j| the
        center-to-center distance. Rationale: a lumped node represents the BULK
        of its pore, so any flow path from bulk i to bulk j traverses an arc of
        length r_i + L_t + r_j = |c_i - c_j| whose minimum cross-section is the
        throat area pi*r_t^2; using that full length (instead of the bare gap)
        bounds the conductance by D*pi*r_t^2/|c_i - c_j|. For PRTRR >= 1 the
        max() always selects |c_i - c_j|, since pi*r_t/2 <= (pi/4)*min(r_i, r_j)
        < r_i + r_j. This removes the unphysical divergence of A_t/L_t for
        near-touching pore pairs (L_t -> 0), which made Deff/Dinf exceed 1 on
        dense small-pore networks; with it, simulated values fall below the
        rigorous real-medium bound Deff/Dinf <= phi_total (total void fraction)
        and in the physically expected range for rocks. For periodic edges the
        unwrapped center distance is their true separation inside the box, so
        they no longer act as shortcuts in open-box Dirichlet simulations
        (a finite sample has no throats crossing its surface). The model is
        conservative: it underestimates transport relative to the well-mixed-
        reservoir limit D*pi*r_t^2/(L_t + pi*r_t/2) (disk access resistance,
        J = 4*D*a*C per mouth).
        """
        valid_models = ["throat", "effective", "path", "conduit", "conduit_shared"]
        if model not in valid_models:
            warn(f"Radius model should be one of {valid_models}. Using 'throat'.")
            model = "throat"
        if share not in ("area", "uniform"):
            warn("share must be 'area' or 'uniform'. Using 'area'.")
            share = "area"

        print("Using", model, "radius model")

        self.throat_model = model
        self.share = share
        if model == "conduit_shared":
            self._setup_pore_shares(share)

        if model == "throat":
            # Neglecting pore resistance (concentration is constant in pore)
            def throat_factor(ni, nj):
                return (
                    self.pn.graph[ni][nj]["radius"] ** 2 * np.pi / self.pn.graph[ni][nj]["length"]
                )

            def radius(ni, nj):
                return self.pn.graph[ni][nj]["radius"]

        elif model == "effective":

            def throat_factor(ni, nj):
                ri = self.pn.graph.nodes[ni]["radius"]
                rj = self.pn.graph.nodes[nj]["radius"]
                lij = self.pn.graph[ni][nj]["length"]
                return (
                    (
                        0.5
                        * (
                            ri * (rj / np.sqrt(rj**2 + lij**2)) ** alpha
                            + rj * (ri / np.sqrt(ri**2 + lij**2)) ** alpha
                        )
                    )
                    ** 2
                    * np.pi
                    / lij
                )

            def radius(ni, nj):
                ri = self.pn.graph.nodes[ni]["radius"]
                rj = self.pn.graph.nodes[nj]["radius"]
                lij = self.pn.graph[ni][nj]["length"]
                return 0.5 * (
                    ri * (rj / np.sqrt(rj**2 + lij**2)) ** alpha
                    + rj * (ri / np.sqrt(ri**2 + lij**2)) ** alpha
                )

        elif model == "path":
            # Path-length conductance (see Notes): G = D*pi*r_t^2/L_eff with
            # L_eff = max(L_t + pi*r_t/2, |c_i - c_j|). For PRTRR >= 1 the
            # center distance always wins, so near-touching pairs (L_t -> 0) no
            # longer diverge; periodic edges get their true in-box separation
            # and stop acting as shortcuts in open-box Dirichlet simulations.

            def throat_factor(ni, nj):
                edge = self.pn.graph[ni][nj]
                ci = np.asarray(self.pn.graph.nodes[ni]["center"], dtype=float)
                cj = np.asarray(self.pn.graph.nodes[nj]["center"], dtype=float)
                d = float(np.linalg.norm(ci - cj))
                l_eff = max(edge["length"] + np.pi * edge["radius"] / 2.0, d)
                return (edge["radius"] ** 2 * np.pi) / l_eff

            def radius(ni, nj):
                return self.pn.graph[ni][nj]["radius"]

        elif model in ("conduit", "conduit_shared"):
            # The conduit conductance is not diffusivity * geometric factor: each
            # segment has its own diffusivity (per-segment Knudsen), so
            # _flow_coef computes it directly from _conduit_segments. Refuse the
            # product form so no caller silently gets a different conductance.

            def throat_factor(ni, nj):
                raise NotImplementedError(
                    "Conduit models have no separable throat factor; use _flow_coef "
                    "(series of per-segment resistances, see _conduit_segments)."
                )

            def radius(ni, nj):
                return self.pn.graph[ni][nj]["radius"]

        self.throat_factor = throat_factor
        self.radius = radius

    def _setup_pore_shares(self, share):
        """Pre-compute the per-pore normalization of the conduit volume shares.

        Parameters
        ----------
        share : str
            "area": the share of pore i given to throat t is
            r_t^2 / sum_k r_k^2 over the throats k of pore i, so wide throats
            drain a larger part of the pore. "uniform": 1/degree(i).

        Notes
        -----
        Stores ``self._pore_share_norm`` (numpy array indexed by node): the sum
        of r_k^2 over incident throats ("area") or the node degree
        ("uniform"). Nodes are consecutive integers (enforced in __init__).
        """
        g = self.pn.graph
        norm = np.zeros(g.number_of_nodes(), dtype=float)
        for u, v, data in g.edges(data=True):
            weight = data["radius"] ** 2 if share == "area" else 1.0
            norm[u] += weight
            norm[v] += weight
        self._pore_share_norm = norm

    def _conduit_segments(self, ni, nj):
        """Return the series segments of the conduit between pores ni and nj.

        Parameters
        ----------
        ni, nj : int
            Node indices of the two pores.

        Returns
        -------
        list of tuple(float, float)
            ``[(F_i, rho_i), (F_t, r_t), (F_j, rho_j)]`` where F is the
            geometric resistance factor (integral of dx/A(x), in 1/length) and
            rho the channel radius used for the per-segment diffusivity. See
            ``setup_throat_model`` for the formulas. For "conduit_shared" the
            pore factors are divided by the pore volume share of this throat.
        """
        g = self.pn.graph
        rt = float(g[ni][nj]["radius"])
        ci = np.asarray(g.nodes[ni]["center"], dtype=float)
        cj = np.asarray(g.nodes[nj]["center"], dtype=float)
        dist = float(np.linalg.norm(ci - cj))

        pore_segments = []
        junctions = 0.0
        for node in (ni, nj):
            rp = float(g.nodes[node]["radius"])
            if rt >= rp:
                # Throat as wide as the pore: no pore taper, the throat starts
                # at the center (defensive; impossible for PRTRR >= 1).
                pore_segments.append((0.0, rp))
                continue
            x = np.sqrt(rp * rp - rt * rt)  # center -> junction distance
            factor = np.log((rp + x) / rt) / (np.pi * rp)
            rho = np.sqrt(x / (np.pi * factor))  # equivalent uniform radius
            if self.throat_model == "conduit_shared":
                norm = self._pore_share_norm[node]
                w = (rt * rt if self.share == "area" else 1.0) / norm if norm > 0.0 else 1.0
                factor /= w
            pore_segments.append((factor, rho))
            junctions += x

        throat_length = max(dist - junctions, 0.0)
        throat_factor = throat_length / (np.pi * rt * rt)
        return [pore_segments[0], (throat_factor, rt), pore_segments[1]]

    def add_bc(self, bc_val, nodes, bc_type="Dirichlet"):
        """Set up boundary conditions for a given set of nodes.

        Parameters
        ----------
        bc_val : float
            Boundary value. For Dirichlet, this is the pore concentration.
            For Neumann, this is the molar flux density (multiplied by the
            pore cross-section to get the molar flow). Positive values
            correspond to flux entering the network from the outside.
        nodes : int or iterable of int
            Node index or list of node indices to apply the BC to.
        bc_type : str, optional
            Boundary condition type: "Dirichlet" or "Neumann".
            Default is "Dirichlet".

        Notes
        -----
        Updates ``self.bc_dict`` with entries mapping node index to
        ``{"value": bc_val, "bctype": bc_type}``.
        """
        if bc_type not in implemented_bc:
            warn("Only Dirichlet and Neumann BC are implemented. Using Dirichlet BC.")
            bc_type = "Dirichlet"

        # Make sure 'nodes' is iterable
        try:
            _ = iter(nodes)
        except TypeError:
            nodes = [nodes]

        self.bc_dict.update(
            {i: {"value": bc_val, "bctype": bc_type} for i in nodes}
        )


class molecular_diffusion_deprecated:
    """Deprecated molecular diffusion class.

    Kept for backward compatibility. Prefer ``molecular_diffusion`` which uses
    a vectorized matrix assembly for better performance on large networks.

    Parameters
    ----------
    pn : pore_network
        The pore network model instance.
    """

    def __init__(self, pn):
        self.pn = pn
        self.diffusivity = None
        self.bc_dict = {}
        self.bcvalues_dict = {}
        self.bctype_dict = {}
        self.model = None
        self.throat_model = None

    def solve(self, solver="cg", x0=None, tol=1e-6, maxiter=None, precond=False):
        """Solve the linear system for the concentration field.

        Parameters
        ----------
        solver : str or callable, optional
            Solver name or callable. Default is "cg".
        x0 : array-like or None, optional
            Initial guess. Default is None.
        tol : float, optional
            Convergence tolerance. Default is 1e-6.
        maxiter : int or None, optional
            Maximum iterations. Default is None.
        precond : bool, optional
            If True, use ILU preconditioning. Default is False.

        Returns
        -------
        int
            Number of iterations (or 0 on failure).
        """
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
        """Return a dictionary mapping node index to concentration.

        Returns
        -------
        dict
            {node_index: concentration}.
        """
        return {n: self.get_concentration(n) for n in self.pn.graph.nodes}

    def get_concentration(self, n1):
        """Return the concentration at a given node, accounting for BC type.

        Parameters
        ----------
        n1 : int
            Node index.

        Returns
        -------
        float or None
            Concentration at node ``n1``.
        """
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
        """Compute sum of molar flow in the given boundary nodes.

        Parameters
        ----------
        bc_nodes : iterable of int
            Boundary node indices.

        Returns
        -------
        float
            Total molar flow.
        """
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
        """Create coefficient matrix and RHS vector.

        Uses a node-by-node assembly (slower than the vectorized version in
        ``molecular_diffusion``). Only active (non-BC) nodes are included in
        the system.

        Notes
        -----
        This is the deprecated implementation. Use ``molecular_diffusion``
        for better performance.
        """
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

        self.A = sp.coo_matrix((data, (rows, cols)), shape=(nnodes, nnodes), dtype=float).tocsr()

    def _flow_coef(self, n1, n2):
        """Compute the throat conductance between two pores.

        Parameters
        ----------
        n1, n2 : int
            Node indices.

        Returns
        -------
        float
            Throat conductance.
        """
        return self.diffusivity(n1, n2) * np.pi * self.radius(n1, n2) ** 2 / self.pn.get_pore_distance(n1, n2)

    def setup_diffusivity_model(self, model="Fick", scale=1, **kwargs):
        """Set up the diffusivity model.

        Parameters
        ----------
        model : str, optional
            "Fick", "Knudsen", or "All". Default is "Fick".
        scale : float, optional
            Diffusivity scaling factor. Default is 1.
        **kwargs
            Additional model parameters (``d_knudsen``, ``d_fick``).
        """
        if model not in ["Fick", "Knudsen", "All"]:
            warn("setup_diffusivity: model must be in {} \nSet it to default (Fick)".format(["Fick", "Knudsen", "All"]))
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
        """Set up the throat radius (conductance) model.

        Parameters
        ----------
        model : str, optional
            "throat", "effective", or "series". Default is "throat".
        alpha : float, optional
            Exponent for the "effective" model. Default is 1.
        """
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
        """Return the boundary condition coefficients for two adjacent nodes.

        Parameters
        ----------
        n1, n2 : int
            Node indices.

        Returns
        -------
        tuple of float
            (bc1, bc2) coefficients for the RHS and diagonal contribution.
        """
        for n in [n1, n2]:
            try:
                return self.bc_dict[n](n1, n2)
            except KeyError:
                pass

        warn("Trying to apply BC to non boundary pores {} and {}".format(n1, n2))
        return 0, 0

    def add_bc(self, bc_val, nodes, bc_type="Dirichlet"):
        """Set up boundary conditions for a given set of nodes.

        Parameters
        ----------
        bc_val : float
            Boundary value (concentration for Dirichlet, flux density for Neumann).
        nodes : int or iterable of int
            Node index or list of node indices.
        bc_type : str, optional
            "Dirichlet" or "Neumann". Default is "Dirichlet".

        Notes
        -----
        Requires ``self.diffusivity`` to be set before calling this method.
        Stores boundary condition callables in ``self.bc_dict``.
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
