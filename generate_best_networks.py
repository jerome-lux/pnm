#!/usr/bin/env python3
"""Generate the four best-recipe networks of the asym50_95 study at N=250k pores, in parallel.

Recipes = top-4 JSD runs of the two asym50_95 studies (window [0.068, 26.53] um, see
doc/experiment-summary.typ):

    #   inversion    e     sratio  order       p1z  p2f   JSD @ N=150k
    1   kernel       1.75  0.7     decreasing   0    2     0.0093
    2   kernel_budget 1.25  0.4     random       0    2     0.0099
    3   kernel_budget 1.25  0.3     random       0    3     0.0104
    4   kernel_waterfill 1.25 0.6   random       0    2     0.0129

Each recipe defines its own pore set (inversion + e + sratio), packed once with the kdtree
packer and cached as a pickle under ``examples/networks/pores_250k/``; throat generation then
runs in parallel via :func:`pnm.parallel_random_PNM` (one forked worker per recipe) and each
network is saved as GML in ``examples/networks/``. Finally every saved network is read back
(GML round-trip check) and scored against the experimental MIP throat distribution.

Usage: python3 generate_best_networks.py [--npores 250000] [--n-jobs 4] [--seed 7]
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import io
import sys
import time
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx  # noqa: E402

from pnm import (  # noqa: E402
    inverse_transform_sampling_from_raw_data,
    invert_pore_distribution_kernel,
    invert_pore_distribution_kernel_budget,
    invert_pore_distribution_kernel_waterfill,
    jensen_shannon_divergence,
    load_mip_data,
    parallel_random_PNM,
    pore_network,
    random_packing_kdtree,
)

DATA = PROJECT_ROOT / "examples" / "data" / "GM1.csv"
OUTDIR = PROJECT_ROOT / "examples" / "networks"

# Top-4 JSD recipes of the asym50_95 studies (see module docstring).
RECIPES = [
    {"tag": "kernel_e1.75_sr0.7_dec", "inversion": "kernel", "e": 1.75, "sratio": 0.7,
     "order": "decreasing", "phase2_radius_factor": 2.0},
    {"tag": "budget_e1.25_sr0.4_rand", "inversion": "kernel_budget", "e": 1.25, "sratio": 0.4,
     "order": "random", "phase2_radius_factor": 2.0},
    {"tag": "budget_e1.25_sr0.3_rand", "inversion": "kernel_budget", "e": 1.25, "sratio": 0.3,
     "order": "random", "phase2_radius_factor": 3.0},
    {"tag": "waterfill_e1.25_sr0.6_rand", "inversion": "kernel_waterfill", "e": 1.25, "sratio": 0.6,
     "order": "random", "phase2_radius_factor": 2.0},
]

INVERSION_FUNCS = {
    "kernel": invert_pore_distribution_kernel,
    "kernel_budget": invert_pore_distribution_kernel_budget,
    "kernel_waterfill": invert_pore_distribution_kernel_waterfill,
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npores", type=int, default=250_000, help="Number of pores per network.")
    p.add_argument("--porosity", type=float, default=0.35, help="Target pore-body porosity.")
    p.add_argument("--prtrr", type=float, default=2.0, help="Pore-to-throat radius ratio.")
    p.add_argument("--seed", type=int, default=7, help="Shared seed (packing + throat generation).")
    p.add_argument("--bcgridsize", type=int, default=150, help="Boundary-detection grid resolution.")
    p.add_argument("--asym-frac", default="0.5,0.95",
                   help="Asymmetric cumulative-volume window 'low,high' (study convention).")
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel workers for throat generation.")
    p.add_argument("--force", action="store_true", help="Overwrite existing output networks.")
    return p.parse_args(argv)


# Packing inputs of the cache files written before the key included them (2026-11-20).
_LEGACY_CACHE_PARAMS = (0.5, 0.95, 2.0, 150)


def pore_cache_path(cache_dir: Path, recipe_tag: str, args, seed: int) -> Path:
    """Pore-set cache file keyed by every input that changes the packed pore set.

    The key holds the recipe (inversion, e, sratio via ``recipe_tag``), N, porosity, seed, the
    cumulative-volume window, PRTRR and the boundary grid. Cache files written before these last
    three were part of the key are reused only when the current values equal the ones they were
    built with (window 0.5-0.95, PRTRR 2, BC grid 150).

    Parameters
    ----------
    cache_dir : Path
        Cache directory.
    recipe_tag : str
        Recipe identifier (inversion, e, sratio, ...).
    args : argparse.Namespace
        Parsed arguments (npores, porosity, asym_frac, prtrr, bcgridsize).
    seed : int
        Packing seed.

    Returns
    -------
    Path
        Cache file path (``.pnm``).
    """
    low, high = (float(x) for x in args.asym_frac.split(","))
    base = f"pores_{recipe_tag}_n{args.npores}_phi{args.porosity:g}_seed{seed}"
    path = cache_dir / f"{base}_w{low:g}-{high:g}_prtrr{args.prtrr:g}_bc{args.bcgridsize}.pnm"
    legacy = cache_dir / f"{base}.pnm"
    if not path.exists() and legacy.exists() and (low, high, args.prtrr, args.bcgridsize) == _LEGACY_CACHE_PARAMS:
        return legacy
    return path


def network_name(args, recipe_tag: str) -> str:
    """Output network name: generation date, N, recipe and seed (never a fixed date)."""
    stamp = date.today().strftime("%Y%m%d")
    return f"{stamp}_asym5095_{args.npores // 1000}k_{recipe_tag}_seed{args.seed}"


def load_window(data: Path, low: float, high: float):
    """Asymmetric cumulative-volume window [low, high] -> MIPData (same convention as param_study_v2)."""
    base = load_mip_data(str(data), cumulative_proba=1.0)  # full curve, no filter
    F = np.cumsum(base.volumes)
    i_min = int(np.searchsorted(F, low))
    i_max = min(int(np.searchsorted(F, high)), len(F) - 1)
    minr = float(base.midradii[i_min])
    maxr = float(base.midradii[i_max])
    return load_mip_data(str(data), rmin=minr, rmax=maxr)


def pack_pore_set(mip, recipe: dict, args, cache_dir: Path) -> str:
    """Pack (or load from cache) the pore-only network of one recipe; returns its pickle path."""
    path = pore_cache_path(cache_dir, recipe["tag"], args, args.seed)
    if path.exists():
        print(f"Pore cache hit: {path}", flush=True)
        return str(path)

    inv_func = INVERSION_FUNCS[recipe["inversion"]]
    _, pdist = inv_func(
        mip.volumes, mip.midradii, mip.deltaR,
        e=recipe["e"], PRTRR=args.prtrr, porosity=args.porosity, sratio=recipe["sratio"],
    )
    if not np.all(np.isfinite(pdist)) or abs(float(pdist.sum()) - 1.0) > 1e-6:
        raise RuntimeError(f"Inversion {recipe['inversion']} produced an invalid pore distribution")

    rdist = partial(
        inverse_transform_sampling_from_raw_data, pdist, bins=mip.throat_radii * args.prtrr, high=np.inf
    )
    np.random.seed(args.seed)  # deterministic radius draws (same convention as random_PNM)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        pn = random_packing_kdtree(
            extent=(1.0, 1.0, 1.0),
            nbnodes=args.npores,
            distributions_props=[({"func": rdist}, 1)],
            target_porosity=args.porosity,
            mindist="auto",
            BCstep=args.bcgridsize,
            restart=True,
            fit_box=False,
            max_unallocated_frac=0.05,
            pool_size=4096,
            tree_rebuild_interval=1024,
            verbose=False,
            seed=args.seed,
        )
    # write() converts category sets to lists in place — save a copy so the live object keeps its sets.
    copy.deepcopy(pn).write(str(path.with_suffix("")), method="pickle")  # .pnm appended by write()
    print(
        f"Packed {recipe['tag']}: N={pn.graph.number_of_nodes()} "
        f"in {time.perf_counter() - t0:.1f}s — cached to {path}",
        flush=True,
    )
    return str(path)


def throat_volume_distribution(pn: pore_network, bins: np.ndarray) -> np.ndarray:
    """Normalized volume-weighted throat radius distribution binned on ``bins``.

    Same convention as the study scripts (first bin [b0, b1], others (bi, bi+1]).
    """
    radii = np.fromiter(nx.get_edge_attributes(pn.graph, "radius").values(), dtype=float)
    vol = np.fromiter(nx.get_edge_attributes(pn.graph, "volume").values(), dtype=float)
    valid = (radii >= bins[0]) & (radii <= bins[-1])
    idx = np.clip(np.searchsorted(bins, radii[valid], side="right") - 1, 0, len(bins) - 2)
    data = np.zeros(len(bins) - 1)
    np.add.at(data, idx, vol[valid])
    return data / data.sum() if data.sum() > 0 else data


def main(argv=None) -> int:
    args = parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUTDIR / f"pores_{args.npores}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    low, high = (float(x) for x in args.asym_frac.split(","))

    # Refuse to overwrite existing networks unless --force (checked before any expensive work).
    existing = [OUTDIR / f"{network_name(args, r['tag'])}.gml" for r in RECIPES]
    existing = [p for p in existing if p.exists()]
    if existing and not args.force:
        print("error: output network(s) already exist (use --force to overwrite):\n  "
              + "\n  ".join(str(p) for p in existing), file=sys.stderr)
        return 1

    mip = load_window(DATA, low, high)
    print(f"Window: [{mip.midradii.min():.4f}, {mip.midradii.max():.4f}] um ({len(mip.volumes)} bins)")

    # 1) Pack one pore set per recipe (sequentially; each is cached for re-runs).
    print(f"\n=== Packing {len(RECIPES)} pore sets at N={args.npores} ===")
    pore_paths = [pack_pore_set(mip, r, args, cache_dir) for r in RECIPES]

    # 2) Throat generation in parallel (one forked worker per recipe).
    print(f"\n=== Generating throats in parallel ({args.n_jobs} workers) ===")
    common = dict(
        npores=args.npores,
        extent=(1.0, 1.0, 1.0),
        porosity=args.porosity,
        rdist=None,  # ignored: pores= is supplied (packing skipped)
        PRTRR=args.prtrr,
        BCgridsize=args.bcgridsize,
        seed=args.seed,
        throat_method="surf",
        phase1_radius_factor=2.0,
        phase1_min_neighbors=32,
        phase1_z=0,  # best of every recipe: no dedicated large-pair pass
        phase2_min_neighbors=32,
        phase2_zmin=2,
        maxc=1024,
    )
    specs = []
    for r, pore_path in zip(RECIPES, pore_paths):
        name = network_name(args, r["tag"])
        specs.append({
            **common,
            "sratio": r["sratio"],
            "order": r["order"],
            "phase2_radius_factor": r["phase2_radius_factor"],
            "pores": pore_path,  # read inside the worker (avoids pickling the graph)
            "out_path": OUTDIR / f"{name}.gml",
        })

    t0 = time.perf_counter()
    summaries = parallel_random_PNM(specs, n_jobs=args.n_jobs)
    wall = time.perf_counter() - t0

    print(f"\n=== Generation done (wall {wall:.1f}s) ===")
    for r, s in zip(RECIPES, summaries):
        mean_conn = 2.0 * s["n_throats"] / max(s["n_pores"], 1)
        print(
            f"{r['tag']}: N={s['n_pores']} E={s['n_throats']} (conn {mean_conn:.2f}) "
            f"in {s['time_s']:.0f}s -> {s['path']}",
            flush=True,
        )

    # 3) Read each network back (GML round-trip check) and score it against the experiment.
    print("\n=== Scoring (JSD / L2 vs experimental throat distribution) ===")
    exp = np.asarray(mip.volumes, dtype=float)
    for r, s in zip(RECIPES, summaries):
        pn = pore_network.read(s["path"])  # default method: GML
        gen = throat_volume_distribution(pn, mip.throat_radii)
        jsd = jensen_shannon_divergence(gen, exp)
        l2 = float(np.sqrt(np.sum((gen - exp) ** 2)))
        large_gen = 100.0 * float(gen[mip.throat_radii[:-1] >= 0.5].sum())
        large_exp = 100.0 * float(exp[mip.throat_radii[:-1] >= 0.5].sum())
        print(
            f"{r['tag']}: JSD {jsd:.4f}, L2 {l2:.4f}, "
            f"large band (>=0.5 um) {large_gen:.1f}% vs exp {large_exp:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
