# AGENTS.md - PNM (Pore Network Model) Project

## Project Overview
- **Path**: `/home/jlux/Nextcloud/Codes/python3/pnm`
- **Package**: `pnm` (Simple Pore Network Model), version 1.1
- **Author**: Jerome Lux (jerome.lux@univ-lr.fr)
- **Environment**: conda `vtk` (`/home/jlux/miniforge3/envs/vtk`), Python 3.12.5
- **Installation**: `pip install -e .` (editable mode, from project root)
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
    diffusion.py       # molecular_diffusion, molecular_diffusion_deprecated (formerly MDiff.py)
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

## Status (Generators.py - COMPLETED)
### `random_packing_opt_cell` (removed)
- Cell-list packing routine, superseded by the faster and more robust kdtree routines below; removed from the codebase. Its performance is kept as the baseline in the benchmark further down.

### `random_PNM` now accepts a `packing_method` option
- `packing_method="kdtree"` (default): uses `random_packing_kdtree`.
- `packing_method="kdtree_importance"`: uses `random_packing_kdtree_importance`.
- `packing_method="opt"`: uses the original brute-force `random_packing_opt` (reference only).
- Invalid `packing_method` raises a `ValueError`.

### Bug fix in `utils.generate_radii`
- The `"func"` key was passed as a kwarg to distribution generators (e.g. `np.random.normal(func=...)`), causing a TypeError. Now `"func"` is extracted from `props` before passing the remaining kwargs.

### Known issue (pre-existing, partially mitigated)
- `generate_radii()` no longer breaks when `"func"` is omitted from `distributions_props` (default `distribution()` is used). If `"func"` is present, it is used as the generator and excluded from kwargs.

### New functions: `random_packing_kdtree` and `random_packing_kdtree_importance`
Two new packing routines for large networks (1e4–1e6 pores) with heavy-tailed radius distributions, sharing a common private engine `_kdtree_packing_engine`:

- **Candidate pools + clearance**: instead of testing one random candidate at a time in pure Python, a pool of B candidates (`pool_size`, default 4096) is drawn and the "clearance" of every candidate (largest sphere radius that fits there, w.r.t. all placed pores and the box faces) is computed once with vectorized `scipy.spatial.cKDTree` ball queries. Pores are then placed largest-first: each pore picks a uniform random candidate among those whose clearance is large enough, and all clearances are updated in O(B). A pool serves many consecutive pores before being discarded, amortizing the tree query over hundreds of placements.
- **Incremental cKDTree**: the main tree is rebuilt every `tree_rebuild_interval` placements (default 1024); pores placed since the last rebuild ("recent") are checked through a small temporary tree. Rebuilding N=1e5 points costs ~22 ms, i.e. ~20 µs amortized per placement — negligible.
- **Two-level clearance check** (exact): the top ~1% of the radius distribution (`n_large_cap = max(16, N//100)`) are "large" pores, checked exhaustively against every candidate (bounded count); all other ("small") pores are covered by a ball query whose radius only needs to reach `r_i + gap + large_thresh`. This keeps the number of tested pairs small even when a few very large pores dominate the box size (essential for MIP-derived distributions spanning several orders of magnitude).
- **Option B (`random_packing_kdtree_importance`)**: identical, except candidate pools are sampled from an occupancy map — a voxel grid (`voxel_resolution`, default 96 per side) whose weights are proportional to the remaining free volume per voxel (each placed sphere's volume is added to its center voxel). Exact validity is always re-checked afterwards; the map only biases sampling.
- Output: pores only, `graph_type = "random_packing_kdtree"` / `"random_packing_kdtree_importance"`, `SetFaceBCnodes` called at the end. Restart logic (`max_unallocated_frac`, max 5 restarts) preserved. Real-time progress print (placed pores | current radius | pools drawn | porosity | unallocated), ~100 updates per run.
- Optional `seed` parameter for reproducibility (restarts always use a fresh seed).

### `random_PNM` now accepts `packing_method="kdtree"` and `"kdtree_importance"`
New forwarded parameters: `pool_size` (default 4096), `tree_rebuild_interval` (default 1024), `voxel_resolution` (default 96, importance only). Note that `nb_trials` is **not** forwarded to the kdtree methods (their give-up control is `max_empty_pools`, default 20); it applies to "opt" only.

### Bugs found in the original pre-processing (`random_packing_opt`, NOT modified)
Discovered while writing the new routines; fixed only in the shared helper `_prepare_packing_radii` used by the kdtree functions:
1. **"Box too small" guard used `radii[:2]`** (the first two *generated* radii, not the two largest). With heavy-tailed MIP data this makes the guard a near no-op that passes on almost every draw even when the two largest pores cannot both fit in the box — the baseline then silently drops the largest pore(s) (see benchmark run 2: porosity 0.239 instead of 0.35). The new helper uses the nearly necessary-and-sufficient opposite-corner condition `sqrt(3)*(L - r0 - r1) >= r0 + r1 + gap` on the actual two largest radii (measured: passes 28/30 draws at N=1e4 with GM1 data, vs 0/30 for the conservative `2.1*(r0+r1)` variant).
2. **`nbnodes *= 1.5` produces a float** → next loop iteration crashes in `radii[0:nbnodes]` (TypeError) whenever the guard triggers. New helper keeps an integer (`int(nbnodes * 1.5)`).

### Bug fixes in `random_packing_opt` (restart logic)
- **Unbounded restart recursion**: when unallocated volume exceeded 5%, the function restarted itself with no cap → RecursionError on targets that random sequential addition cannot reach. Now capped at 5 restarts via an internal `_restart_count` parameter, same as the other routines.
- **Restart passed the already-scaled extent** (box size drifted on every restart) and silently dropped `BCstep`/`fit_box`. It now passes `original_extent` (saved before scaling) and all parameters explicitly.

### Imports cleanup
- Removed unused `from sklearn.neighbors import KDTree` (leftover from an earlier attempt; its comment claimed sklearn was faster than scipy, but sklearn 1.6.1's `KDTree.query` does not support ball queries at all) and unused `import multiprocessing`.
- Added `from scipy.spatial import cKDTree` (scipy is already a dependency).

### Benchmark: kdtree vs cell-list baseline (10000 pores, extent=(1,1,1), porosity=0.35, GM1.csv notebook pipeline, e=1.5, PRTRR=2.0)
4 runs per algorithm; cell-list baseline called with `nb_trials=500`.

| Algorithm | Mean time | Std dev | Pores placed (per run) | Porosity (per run) |
|-----------|-----------|---------|------------------------|--------------------|
| cell-list baseline (`random_packing_opt_cell`, since removed) | 18.36 s | 18.98 s | 10000, **9998**, 10000, 10000 | 0.350, **0.239**, 0.350, 0.350 |
| `random_packing_kdtree` (A) | **1.73 s** | 0.03 s | 10000 ×4 | 0.350 ×4 |
| `random_packing_kdtree_importance` (B) | **1.89 s** | 0.08 s | 10000 ×4 | 0.350 ×4 |

- **Speedup: ~10.6× on average, up to ~29× vs the slowest baseline run.**
- Baseline time is highly draw-dependent (4.5–51 s) because the box size varies with the random radii draw (extent 4.75–31.7 in these runs); one run even failed to place the largest pore (porosity 0.239). The new algorithms are nearly constant-time per network (~1.7–2 s) regardless of box size, and always reach exactly the target porosity with all requested pores placed.
- **A vs B at porosity 0.35**: importance sampling is slightly *slower* (voxel sampling + map update overhead outweighs the gain when free space is still abundant). The expected crossover where B wins is higher target porosity / larger N; to be re-benchmarked if needed.
- Correctness verified at N=10000: exhaustive pair check (cKDTree `query_pairs`, ~3e7 pairs) found **zero overlaps** for both new algorithms (worst surface margin > 0), all pores inside the box. End-to-end `random_PNM(packing_method="kdtree"/"kdtree_importance")` validated (throats + periodic links + geometry complete).

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
