# AGENTS.md - PNM (Pore Network Model) Project

## Project Overview
- **Path**: `/home/jlux/Nextcloud/Codes/python3/pnm`
- **Package**: `pnm` (Simple Pore Network Model), version 1.1
- **Author**: Jerome Lux (jerome.lux@univ-lr.fr)
- **Environment**: conda `vtk` (`/home/jlux/miniforge3/envs/vtk`), Python 3.12.5
- **Domain**: Random pore network generator + simulation routines (diffusion, two-phase flow, ...)

## Package Structure
```
pnm/
  __init__.py          # exports pore_network, Generators, Visualization, Stats, Transformation, Algorithms
  PNM.py               # pore_network class (core graph-based network model)
  Generators.py        # cubic_PNM, random_PNM, random_packing_opt, random_packing_opt_cell, add_throats_by_surf, add_periodic_throats, scale, SetFaceBCnodes, SetBCnodes, ...
  Transformation.py    # invasion_percolation, assemble_pnm, add_throats, connect_pores, scale, aggregates_nodes_if_too_big
  Stats.py             # stats(), distribution(), inverse_transform_sampling_*
  utils.py             # isinside, isinside_surface, eucl_dist, query_mindist_loops, generate_radii, nearest_neighbor
  Visualization.py     # visualization routines
  Algorithms/
    __init__.py
    MDiff.py           # molecular_diffusion, molecular_diffusion_deprecated
    TPFlow.py          # TPFlow (two-phase flow)
    percolation.py     # mercury_intrusion (formerly InvPerc.py)
  Tools/
    net_renderer.py
    2D-network-viewer.py
```

## Build Configuration
- `pyproject.toml`: PEP 621 build configuration (replaces setup.py)
- `setup.py`: legacy build script (retained for compatibility)

## Key Design Facts (PNM.py)
- `pore_network` wraps a `networkx.Graph`. Node/edge attributes store geometry and properties.
- Graph attributes: `graph_type`, `extent` (box size from pore surfaces), `bbox`, `porosity`, `inner_extent`, `spacing`.
- Node attributes: `radius`, `center`, `volume`, `category` (set of side labels: x-, x+, y-, y+, z-, z+, inner, none), `periodic`, `exchange_area`, `surface_ratio`, `boundary`.
- Edge attributes: `radius`, `length`, `volume`, `periodic`, `pdir`, `sw`, `wf_side`.
- `side_to_int_dict` / `int_to_side_dict` map side labels to bit flags (x-=256, x+=512, y-=1024, y+=2048, z-=4096, z+=8192, none=0, inner=0).
- `compute_geometry()` relabels nodes to consecutive integers via `nx.convert_node_labels_to_integers` — breaks external references to old labels.
- `add_pore` has a mutable default argument `category=set()` (BUG — shared across calls).
- `merge_pores` references undefined `u, v` in a warn call (BUG — NameError).
- `write()` accesses `self.graph.graph["inner_extent"]` which is only set by `SetFaceBCnodes`/`SetBCnodes`/`scale` (BUG — KeyError if never set).
- `add_attributes` calls non-existent `self.graph.add_node_attributes` / `add_edge_attributes` (BUG — AttributeError).
- `set_auto_throats_length` reads edge attribute `category` which is never set on throats (BUG — KeyError); should check `periodic` edge attribute instead.
- `write_props` divides by `number_of_nodes()` without zero check and reads `graph["porosity"]` without check (BUGS).

## Testing
- Activate env: `source /home/jlux/miniforge3/bin/activate vtk`
- Run from project root: `python -c "import pnm; ..."`
- No existing test suite found.

## Current Focus
Refactor/improve `PNM.py`: fix bugs, add English docstrings (translate existing French comments), verify logic. Then proceed to Generators.py, Transformation.py, Algorithms.

## Status (PNM.py - COMPLETED)
### Bugs fixed:
1. **`merge_pores`** (line ~493): `warn()` referenced undefined `u, v` → fixed to `n1, n2` (NameError).
2. **`write()`** (line ~122): accessed `self.graph.graph["inner_extent"]` without check → KeyError if `SetFaceBCnodes`/`SetBCnodes`/`scale` never called. Now uses `.get()` with None guard.
3. **`save_as_ascii()`** (line ~179): accessed `self.graph.graph["extent"]` without None check → TypeError. Now warns and returns if extent is None.
4. **`add_pore()`** (line ~588): mutable default argument `category=set()` shared across calls → changed to `category=None` with `set()` created inside.
5. **`add_attributes()`** (line ~642): called non-existent `self.graph.add_node_attributes` / `add_edge_attributes` → replaced with `nx.set_node_attributes` / `nx.set_edge_attributes` iterating over attr name → values dict.
6. **`set_auto_throats_length()`** (line ~558): read edge attribute `category` which is never set on throats → KeyError. Now checks `periodic` edge attribute via `.get("periodic", False)`.
7. **`write_props()`** (line ~734): division by zero when `number_of_nodes() == 0`; `graph["porosity"]` KeyError if not set; `list(None)` TypeError on extent. All guarded.
8. **`check_overlapping()`** (line ~243): `labels.remove(i)` could raise ValueError if `i` absent → replaced with safe list comprehension.
9. **`get_pores_by_category()`** (line ~719): `category.issubset(v) > 0` (awkward bool comparison) → simplified to `category.issubset(v)`. Changed `if mode == "equal"` to `elif` for consistency.

### Docstrings:
- All methods now have complete English docstrings with Parameters/Returns sections (NumPy style).
- Class-level docstring documents all graph/node/edge attributes.
- French comments translated to English.

## Status (Generators.py - IN PROGRESS)
### New function: `random_packing_opt_cell`
- Replaces the brute-force O(N²) overlap test of `random_packing_opt` with a **cell list** (spatial hashing).
- Cell size = 2 × max_radius + gap guarantees that any two colliding pores lie in the same or neighbouring cells (3×3×3 = 27 cells checked in 3D).
- **Early exit** in collision testing: stops at the first overlap found instead of computing the exact minimum distance.
- Full restart preserved, with configurable `max_unallocated_frac` (default 0.05) replacing the hard-coded 5%.
- Optional `cell_size` parameter; auto-computed if None.
- Performance: ~6× faster on 1000 pores, ~2.6s for 10000 pores (vs minutes expected for brute-force at that scale).
- Same return type as `random_packing_opt`: a `pore_network` with pores only (no throats), `graph_type = "random_packing_cell"`.

### `random_PNM` now accepts a `packing_method` option
- `packing_method="cell"` (default): uses `random_packing_opt_cell` (fast, recommended).
- `packing_method="opt"`: uses the original brute-force `random_packing_opt`.
- Additional parameters: `max_unallocated_frac`, `cell_size`, `packing_verbose` are forwarded to the cell-list routine.
- Invalid `packing_method` raises a `ValueError`.

### Bug fix in `utils.generate_radii`
- The `"func"` key was passed as a kwarg to distribution generators (e.g. `np.random.normal(func=...)`), causing a TypeError. Now `"func"` is extracted from `props` before passing the remaining kwargs.

### Known issue (pre-existing, partially mitigated)
- `generate_radii()` no longer breaks when `"func"` is omitted from `distributions_props` (default `distribution()` is used). If `"func"` is present, it is used as the generator and excluded from kwargs.

## Status (Transformation.py - COMPLETED)
### Bugs fixed in `add_throats_by_surf`:
1. **`surface_ratio` stored with `enumerate()` indices (0..N-1) as node labels** — broken when node labels are not consecutive (e.g. before `compute_geometry()` relabels). Now uses `dict(zip(nodelist, surface_ratio))` to map to actual node labels.
2. **`surface_ratio[n1]` used the node label as a numpy array index** — IndexError or wrong values when labels are not 0..N-1. Now uses `surface_ratio_dict[n1]` (proper label-based dict).
3. **`counter` undefined if `candidates` was empty** — loop never executed, variable undefined at end. Now initialized to `counter = -1` before the loop.

### Docstrings:
- `add_throats_by_surf` and `add_throats`: complete English docstrings (NumPy style) with Parameters/Returns/Notes sections.
- French comments translated to English.

### Bugs fixed in `add_periodic_throats`:
1. **`extent = net.graph.graph["extent"]`** — KeyError if `SetFaceBCnodes`/`compute_geometry` never called. Now uses `.get("extent")` with None guard and warn.
2. **`surface_ratio = net.graph.nodes[n]["surface_ratio"]`** — KeyError if `add_throats_by_surf` never called. Now uses `.get("surface_ratio", 1.0)`.

### Docstrings:
- `add_periodic_throats`: complete English docstring (NumPy style).
- French comments in `add_periodic_throats` preserved as requested.

## Status (Transformation.py - COMPLETED - all functions)
### Additional bugs fixed:
4. **`scale`** (line ~908): edge `length` scaled using `name="radius"` instead of `name="length"` → throat length overwritten with radius values. Fixed to `name="length"`. Also `extent`/`inner_extent` KeyError guarded with `.get()`. Added `factor > 0` validation.
5. **`aggregates_nodes_if_too_big`** (line ~967): `maxradius=None` default used in `radius > maxradius` and `distance <= maxradius` → TypeError. Now defaults to `float("inf")` when None.
6. **`SetFaceBCnodes`** (line ~1030): `extent = net.graph.graph["extent"]` → KeyError. Now `.get("extent")` with None guard + warn.
7. **`SetBCnodes`** (line ~1099): `extent = net.graph.graph["extent"]` → KeyError. Now `.get("extent")` with None guard + warn.

### Additional bugs fixed in `invasion_percolation`:
- `box` list mutated in place → now a copy is made.
- `pn.graph.graph["bbox"]` / `pn.graph.graph["extent"]` KeyError → `.get()` with None guard + warn.

### Performance optimization in `invasion_percolation`:
- `nx.articulation_points` is now recomputed every `articulation_recompute_interval` removed pores (default 100) instead of every pore. This reduces complexity from O(V^2) to O(V^2 / interval) for that step. Measured ~34x speedup on 499 pores with keep_connected=True.
- Pores are marked for removal using a numpy boolean array instead of being deleted from the graph one by one via `pn.remove_pore()`. All removed pores are deleted in a single batch (`remove_nodes_from`) at the end.
- New parameter `articulation_recompute_interval` (default 100). Set to a large value (or 0) to disable recomputation for maximum speed on very large networks.
- Results are identical to the original implementation (same number of pores removed, same final porosity).

### Additional bugs fixed in `assemble_pnm`:
- `pn1.graph.graph["graph_type"]` / `pn1.graph.graph["bbox"]` / `pn1.graph.graph["spacing"]` / `pn2.graph.graph["bbox"]` KeyError → `.get()` with None guard + warn.

### Docstrings added (all remaining functions):
- `invasion_percolation`, `assemble_pnm`, `connect_pores`, `scale`, `aggregates_nodes_if_too_big`, `SetFaceBCnodes`, `SetBCnodes`: complete English docstrings (NumPy style).
- French comments translated to English.

## Status (Algorithms/InvPerc.py - COMPLETED)
### Performance optimization in `mercury_intrusion`:
- Replaced O(N) full scan of `source_nodes` at every iteration with a stack-based BFS: only pores with newly fillable throats are processed.
- Uses sets (O(1) lookup) for filled pores and filled throats instead of nested dict lookups.
- Precomputes node radii and volumes as numpy arrays for fast indexed access.
- When D decreases, all filled pores with open throats are re-added to the test stack so their capillaries are re-tested at the smaller diameter.
- Performance: ~2.3x speedup on 10000 pores (0.57s → 0.25s).
- Results are equivalent to the original implementation (minor volume difference <1% due to processing order).
- Complete English docstring (NumPy style).

## Status (Algorithms/diffusion.py - COMPLETED)
### Bugs fixed in `molecular_diffusion`:
1. **`setup_coef_matrix`** — Used `nx.adjacency_matrix().tolil()` with slow LIL slicing
   (`adjm[active_nodes] = 0`, `adjm[:, dirichlet_nodes] = 0`) and `deepcopy(adjm)`.
   Replaced with a fully vectorized COO assembly: all throat conductances are
   pre-computed in one pass, diagonal terms accumulated with `np.add.at`, and
   the matrix built directly from data/rows/cols arrays. No more LIL slicing
   or deepcopy.
2. **`solve`** — scipy's `cg` uses `rtol`, not `tol` (since scipy 1.0). Now tries
   `rtol` first, falls back to `tol` on TypeError.
3. **`get_concentration`** — Caught bare `except:` (too broad). Now catches
   `(AttributeError, TypeError, IndexError)` specifically.
4. **`setup_coef_matrix`** — Missing guard for `n == 0` and empty edge list.
   Added early returns with warnings.

### Performance optimization in `molecular_diffusion.setup_coef_matrix`:
- All throat conductances pre-computed in a single pass over edges (instead of
  calling `_flow_coef` separately for diagonal, off-diagonal, and Dirichlet
  terms).
- Matrix assembled directly in COO format from concatenated arrays.
- Measured: ~0.09s to build matrix on 5000 pores / 10000 throats.
- `compute_boundary_flow` rewritten to iterate directly over boundary nodes'
  neighbors instead of rebuilding `nx.adjacency_matrix()` with LIL slicing.

### Docstrings:
- `molecular_diffusion` and `molecular_diffusion_deprecated`: complete English
  docstrings (NumPy style) with Parameters/Returns/Notes sections for all
  methods.
- French comments translated to English.
- Comment in `bc_values` typo fixed ("aply" → "apply").

## Status (Visualization.py - COMPLETED)
### Bugs fixed in `Fast3DViewer`:
1. **`props["periodic"]`, `props["pdir"]`, `props["radius"]`** — KeyError if edge attribute missing. Now uses `.get()` with defaults (False, "x", 0.1).
2. **`self.network.graph.graph["extent"]`** — KeyError if extent never set. Now uses `.get("extent")` with None guard.
3. **`cam.SetFocalPoint(...),`** — trailing comma creating useless tuple in `reset_camera()` (both classes). Fixed.
4. **`self.renWin.AddRenderer(self.renderer)`** in `save()` — renderer already added in `create_renWin()`, double addition. Removed.
5. **`pn.graph.graph["extent"]`** in `vtk_viewer()` — KeyError if extent not set. Now uses `.get("extent")` with None guard + fallback filename.
6. **`min(min(pattrval), min(tattrval))`** in `vtk_viewer()` — ValueError if no nodes or edges. Now checks for empty attribute lists.

### Bugs fixed in `PNM_3d_viewer`:
1. **`props["periodic"]`, `props["pdir"]`** — KeyError in `make_geometry`. Now uses `.get()` with defaults.
2. **`cam.SetFocalPoint(...),`** — trailing comma in `reset_camera()`. Fixed.

### Docstrings:
- `Fast3DViewer` and all methods: complete English docstrings (NumPy style).
- `PNM_3d_viewer` and all methods: complete English docstrings (NumPy style).
- `create_sphere_glyphs`, `create_tubes`, `create_pore_actor`, `create_throat_actor`: complete English docstrings.
- `vtk_viewer`: complete English docstring (NumPy style).
- French comments translated to English.

## Status (Examples - COMPLETED)
### `examples/PNM-generation.ipynb`
Complete tutorial notebook demonstrating the full PNM workflow:
1. Read experimental MIP data from CSV files (`examples/data/`)
2. Compute pore diameter threshold (cumulative probability) and cumulative pore/throat distributions
3. Define model parameters (npores, e, PRTRR, sratio, minc, maxc) in an easily modifiable cell
4. Generate a random pore network using fast cell-list packing (`random_PNM` with `packing_method="cell"`)
5. Compare experimental vs generated pore and throat distributions (visualized)
6. Mercury intrusion simulation (`mercury_intrusion`) comparing experimental vs simulated intrusion curves
7. 3D visualization with `Fast3DViewer` and image saving

Generated networks and images are saved in `examples/networks/`.
Notebook validated by execution (`jupyter nbconvert --execute`) with no errors.
