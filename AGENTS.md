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
  PNM.py               # pore_network class (core graph-based network model) <- current focus
  Generators.py        # cubic_PNM, random_PNM, random_packing_opt, random_packing_opt_cell, add_throats_by_surf, add_periodic_throats, scale, SetFaceBCnodes, SetBCnodes, ...
  Transformation.py    # invasion_percolation, assemble_pnm, add_throats, connect_pores, scale, aggregates_nodes_if_too_big
  Stats.py             # stats(), distribution(), inverse_transform_sampling_*
  utils.py             # isinside, isinside_surface, eucl_dist, query_mindist_loops, generate_radii, nearest_neighbor
  Visualization.py     # visualization routines
  Algorithms/
    __init__.py
    MDiff.py           # molecular_diffusion, molecular_diffusion_deprecated
    TPFlow.py          # TPFlow (two-phase flow)
  Tools/
    net_renderer.py
    2D-network-viewer.py
```

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