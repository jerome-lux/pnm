#!/usr/bin/env python3
"""Run 3-direction molecular-diffusion simulations on every network in a directory.

This is the batch counterpart of ``examples/Simulation.ipynb``: for each network file found in
the input directory (matching ``--pattern``), it loads the graph, solves the steady-state
diffusion equation in the x/y/z directions with Dirichlet BCs on the boundary pores, and reports
the effective diffusivity Deff (m^2/s) per direction.

The diffusion parameters default to the values used by ``examples/Simulation.ipynb``:
Dinf = 1.25e-5 m^2/s (CO2 in air), Ma = 44e-3 kg/mol, T = 293 K, scale = 1e-6 (networks in um),
BC1/BC2 = 1/0, throat model "throat", diffusivity model "All" (Fick + Knudsen combined).

Outputs (in ``--out-dir``, default ``<input dir>/diffusion_results``), keyed by the run
configuration ``<config> = <throat model>[-<share>]-<diffusivity model>-<directions>``
(e.g. ``conduit-All-x``, ``conduit_shared-area-Fick-xyz``):
    <netname>__<config>.csv   one row per network, same columns as the notebook CSV
    summary__<config>.csv     one row per network with Deff/Dinf ratios and timings; rows of
                              networks run earlier with the same configuration are kept (merged
                              by file name), so the summary always covers every simulated network
    <netname>__<config>_conc.gml   (only with --save-concentrations) copy of the network with
                              per-direction pore concentration node attributes
                              (concentration_x/y/z) for field visualization

Networks whose per-network CSV for the same configuration already exists are skipped unless
``--force`` is given; runs with different throat/diffusivity models or directions never skip or
overwrite each other. Workers run in parallel (forked processes); each loads its own graph, so
peak RAM scales with ``--n-jobs`` (~2-5 GB per 250k-pore network).

Usage:
    python3 run_diffusion_all.py examples/networks --pattern "20261120_asym5095_250k_*"
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pnm  # noqa: E402

# Module-level parameters for the forked workers (set in main() before pool creation).
_PARAMS: dict = {}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", help="Directory containing the network files to simulate.")
    p.add_argument("--pattern", default="*.gml",
                   help='Glob pattern for network filenames (default "*.gml"; .pnm = pickle).')
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <directory>/diffusion_results).")
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel workers (one per network).")
    p.add_argument("--omp-threads", type=int, default=8,
                   help="BLAS threads per worker (avoids CPU oversubscription). Default 8.")
    p.add_argument("--directions", default="x,y,z",
                   help='Comma-separated directions to simulate, subset of "x,y,z" (default all three).')
    p.add_argument("--force", action="store_true", help="Re-run networks that already have a CSV result.")
    # Diffusion parameters — defaults are the values of examples/Simulation.ipynb.
    p.add_argument("--dinf", type=float, default=1.25e-5, help="Bulk (Fick) diffusivity, m^2/s.")
    p.add_argument("--ma", type=float, default=44e-3, help="Molar mass of the diffusing species, kg/mol.")
    p.add_argument("--temperature", type=float, default=293.0, help="Temperature for Knudsen diffusivity, K.")
    p.add_argument("--scale", type=float, default=1e-6,
                   help="Network length unit in meters (networks are generated in um).")
    p.add_argument("--bc1", type=float, default=1.0, help="Dirichlet value on the '+' faces.")
    p.add_argument("--bc2", type=float, default=0.0, help="Dirichlet value on the '-' faces.")
    p.add_argument("--throat-model", default="throat",
                   choices=["throat", "effective", "path", "conduit", "conduit_shared"])
    p.add_argument("--share", default="area", choices=["area", "uniform"],
                   help="Pore-volume partition rule for --throat-model conduit_shared.")
    p.add_argument("--diff-model", default="All", choices=["Fick", "Knudsen", "All"])
    p.add_argument("--solver", default="cg")
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--save-concentrations", action="store_true",
                   help="Also save each network as GML with per-direction pore concentration "
                        "node attributes (concentration_x/y/z), for field visualization.")
    return p.parse_args(argv)


NETWORK_SUFFIXES = {".gml", ".pnm"}


def find_networks(directory: Path, pattern: str) -> list[Path]:
    """Return the sorted network files in ``directory`` matching ``pattern``.

    Only recognized network formats (.gml / .pnm) are kept, so a broad pattern
    (e.g. "20261120_asym5095_250k_*") does not pick up companion files such as
    *_throats.png or *_pores.csv.
    """
    files = [f for f in directory.glob(pattern) if f.is_file() and f.suffix.lower() in NETWORK_SUFFIXES]
    return sorted(files)


def read_method(path: Path) -> str:
    """File extension -> pore_network.read method ('.pnm' is a pickled graph)."""
    return "pickle" if path.suffix == ".pnm" else "gml"


SUMMARY_FIELDS = ["file", "nnodes", "nthroats", "porosity", "avg_connectivity",
                  "Deff-x", "Deff-y", "Deff-z",
                  "Deff_over_Dinf_x", "Deff_over_Dinf_y", "Deff_over_Dinf_z",
                  "status_x", "status_y", "status_z",
                  "time_load_s", "time_sim_s", "time_total_s"]


def config_tag(throat_model: str, share: str, diff_model: str, directions: list[str]) -> str:
    """Run-configuration key used in output file names.

    Parameters
    ----------
    throat_model : str
        Throat conductance model.
    share : str
        Pore-volume partition rule; only part of the key for "conduit_shared".
    diff_model : str
        Diffusivity model ("Fick", "Knudsen" or "All").
    directions : list of str
        Simulated directions, e.g. ["x"] or ["x", "y", "z"].

    Returns
    -------
    str
        Key such as "conduit-All-x" or "conduit_shared-area-Fick-xyz".
    """
    model = f"{throat_model}-{share}" if throat_model == "conduit_shared" else throat_model
    return f"{model}-{diff_model}-{''.join(directions)}"


def result_csv_path(out_dir: Path, net_path: Path, tag: str) -> Path:
    """Per-network result CSV for one run configuration."""
    return out_dir / f"{net_path.stem}__{tag}.csv"


def write_summary(summary_path: Path, results: list[dict]) -> list[dict]:
    """Merge new result rows into the configuration's summary CSV and rewrite it.

    Rows already present for networks that were not re-run are kept; rows of re-run networks
    are replaced. Returns the merged rows sorted by file name.

    Parameters
    ----------
    summary_path : Path
        Summary CSV for one run configuration.
    results : list of dict
        New rows (keys = ``SUMMARY_FIELDS``).

    Returns
    -------
    list of dict
        All rows written to the summary.
    """
    rows: dict[str, dict] = {}
    if summary_path.exists():
        with open(summary_path, newline='') as csvfile:
            for row in csv.DictReader(csvfile, delimiter=';'):
                if row.get("file"):
                    rows[row["file"]] = row
    for r in results:
        rows[r["file"]] = r
    merged = [rows[k] for k in sorted(rows)]
    with open(summary_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SUMMARY_FIELDS, delimiter=';', extrasaction='ignore')
        writer.writeheader()
        for r in merged:
            writer.writerow(r)
    return merged


def simulate_one(net_path: Path, out_dir: Path) -> dict:
    """Load one network and run the diffusion simulation in x/y/z; write its CSV.

    Returns a result row (dict) for the summary table. Runs inside a forked worker.
    """
    name = net_path.stem
    csv_path = result_csv_path(out_dir, net_path, _PARAMS["tag"])
    t_start = time.perf_counter()

    print(f"[{name}] loading {net_path.name} ...", flush=True)
    t0 = time.perf_counter()
    pn = pnm.pore_network.read(str(net_path), method=read_method(net_path))
    t_load = time.perf_counter() - t0
    n_nodes = pn.graph.number_of_nodes()
    n_edges = pn.graph.number_of_edges()
    print(f"[{name}] N={n_nodes} E={n_edges} in {t_load:.1f}s", flush=True)

    # --- simulation parameters (identical to examples/Simulation.ipynb) ---
    scale = _PARAMS["scale"]
    BC1, BC2 = _PARAMS["bc1"], _PARAMS["bc2"]
    Dinf = _PARAMS["dinf"]
    DKNUDSEN = (1 / 6) * np.sqrt((8 * 8.3144 * _PARAMS["temperature"]) / (np.pi * _PARAMS["ma"]))
    S = {'x': np.prod(pn.graph.graph['extent'][1:]),
         'y': np.prod(pn.graph.graph['extent'][0::2]),
         'z': np.prod(pn.graph.graph['extent'][:2])}
    L = {'x': pn.graph.graph['extent'][0],
         'y': pn.graph.graph['extent'][1],
         'z': pn.graph.graph['extent'][2]}

    # Boundary-pore coverage is cheap metadata — report it for all three directions.
    nbcpores = {d: 100 * (len(pn.get_pores_by_category([d + "+"])) +
                          len(pn.get_pores_by_category([d + "-"]))) / n_nodes for d in 'xyz'}

    Deff, status = {}, {}
    t_sim = 0.0
    for direction in _PARAMS["directions"]:
        dir1, dir2 = direction + "+", direction + "-"

        t0 = time.perf_counter()
        solver = pnm.molecular_diffusion(pn)
        solver.setup_throat_model(model=_PARAMS["throat_model"], share=_PARAMS["share"])
        solver.setup_diffusivity_model(model=_PARAMS["diff_model"], d_fick=Dinf,
                                       d_knudsen=DKNUDSEN, scale=1 / scale)
        bc_nodes = pn.get_pores_by_category([dir1], mode='match_one')
        solver.add_bc(BC1, bc_nodes, bc_type='Dirichlet')
        solver.add_bc(BC2, pn.get_pores_by_category([dir2], mode='match_one'), bc_type='Dirichlet')
        solver.setup_coef_matrix()
        solver.solve(solver=_PARAMS["solver"], tol=_PARAMS["tol"])
        t_sim += time.perf_counter() - t0

        # scipy iterative solvers return (x, info): info == 0 means converged.
        status[direction] = int(solver.iter)
        q1 = solver.compute_boundary_flow(bc_nodes)
        Deff[direction] = q1 * L[direction] * scale**2 / ((BC2 - BC1) * S[direction])
        print(f"[{name}] Deff-{direction} = {Deff[direction]:.3e} m^2/s "
              f"(Deff/Dinf = {Deff[direction]/Dinf:.4f}, status {status[direction]}) in "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

        # Keep the solved field for visualization (node labels are consecutive integers).
        if _PARAMS["save_conc"] and solver.x is not None:
            nx.set_node_attributes(
                pn.graph, {i: float(v) for i, v in enumerate(solver.x)}, f"concentration_{direction}"
            )

    # Save a copy of the network carrying every solved concentration field.
    if _PARAMS["save_conc"]:
        conc_path = out_dir / f"{net_path.stem}__{_PARAMS['tag']}_conc.gml"
        t0 = time.perf_counter()
        # write() appends ".gml" to the given path (same convention as the pore cache).
        pn.write(str(conc_path.with_suffix("")), method="gml")  # serializes a copy; live graph untouched
        print(f"[{name}] concentration field saved -> {conc_path.name} "
              f"in {time.perf_counter()-t0:.0f}s", flush=True)

    avgc = sum(pn.graph.degree(n) for n in pn.graph.nodes) / n_nodes
    row_name = _PARAMS["tag"] + "-" + name

    # Per-network CSV — same columns as the notebook.
    with open(csv_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';', dialect='excel')
        csvwriter.writerow(['name', 'nnodes', 'nthroats', 'porosity', 'avg connectivity',
                            'throat model', '%BCP-x', '%BCP-y', '%BCP-z',
                            'Deff-x', 'Deff-y', 'Deff-z'])
        csvwriter.writerow([row_name, n_nodes, n_edges, pn.graph.graph['porosity'], avgc,
                            _PARAMS["throat_model"],
                            nbcpores['x'], nbcpores['y'], nbcpores['z'],
                            Deff.get('x', ''), Deff.get('y', ''), Deff.get('z', '')])

    return {
        "file": net_path.name,
        "nnodes": n_nodes,
        "nthroats": n_edges,
        "porosity": pn.graph.graph['porosity'],
        "avg_connectivity": avgc,
        **{f"Deff-{d}": Deff.get(d) for d in 'xyz'},
        **{f"Deff_over_Dinf_{d}": (Deff[d] / Dinf if d in Deff else None) for d in 'xyz'},
        **{f"status_{d}": status.get(d) for d in 'xyz'},
        "time_load_s": t_load,
        "time_sim_s": t_sim,
        "time_total_s": time.perf_counter() - t_start,
    }


def _worker_init(omp_threads: int) -> None:
    """Limit BLAS threads per forked worker (called once in each child)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(max(1, omp_threads))


def main(argv=None) -> int:
    args = parse_args(argv)

    directory = Path(args.directory).resolve()
    if not directory.is_dir():
        print(f"error: {directory} is not a directory", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir).resolve() if args.out_dir else directory / "diffusion_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_networks(directory, args.pattern)
    if not files:
        print(f"error: no network file matching {args.pattern!r} in {directory}", file=sys.stderr)
        return 1

    directions = [d.strip() for d in args.directions.split(",") if d.strip()]
    bad = [d for d in directions if d not in ("x", "y", "z")]
    if bad or not directions:
        print(f"error: --directions must be a non-empty subset of x,y,z (got {args.directions!r})", file=sys.stderr)
        return 1

    _PARAMS.update({k: getattr(args, k) for k in
                    ("scale", "bc1", "bc2", "dinf", "ma", "temperature",
                     "throat_model", "share", "diff_model", "solver", "tol")})
    _PARAMS["save_conc"] = args.save_concentrations
    _PARAMS["directions"] = directions
    tag = config_tag(args.throat_model, args.share, args.diff_model, directions)
    _PARAMS["tag"] = tag

    todo = [f for f in files if args.force or not result_csv_path(out_dir, f, tag).exists()]
    skipped = [f.name for f in files if f not in todo]
    print(f"Configuration: {tag}")
    print(f"Found {len(files)} network(s) matching {args.pattern!r} in {directory}")
    if skipped:
        print(f"Skipping {len(skipped)} already simulated (use --force to re-run): {', '.join(skipped)}")
    print(f"Simulating {len(todo)} network(s) with {max(1, min(args.n_jobs, len(todo)))} worker(s); "
          f"results -> {out_dir}")

    t0 = time.perf_counter()
    if not todo:
        results = []
    elif len(todo) == 1 or args.n_jobs <= 1:
        _worker_init(args.omp_threads)
        results = [simulate_one(f, out_dir) for f in todo]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(args.n_jobs, len(todo)),
                      initializer=_worker_init, initargs=(args.omp_threads,)) as pool:
            results = pool.starmap(simulate_one, [(f, out_dir) for f in todo])

    wall = time.perf_counter() - t0

    # Summary table for this configuration (merged with earlier runs of the same configuration).
    summary_path = out_dir / f"summary__{tag}.csv"
    merged = write_summary(summary_path, results)

    print(f"\n=== Done (wall {wall:.0f}s) — summary ({len(merged)} network(s)) -> {summary_path} ===")
    for r in results:
        done = [d for d in 'xyz' if r[f"status_{d}"] is not None]
        flag = "" if all(r[f"status_{d}"] == 0 for d in done) else "  [NOT CONVERGED]"
        ratios = "/".join(f"{r['Deff_over_Dinf_' + d]:.4f}" if r["Deff_over_Dinf_" + d] is not None
                          else "-" for d in 'xyz')
        print(f"{r['file']}: Deff/Dinf x/y/z = {ratios}  (sim {r['time_sim_s']:.0f}s){flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
