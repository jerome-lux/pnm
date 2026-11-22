#!/usr/bin/env python3
"""Generate four N=250k pore networks from the single best asym50_95 recipe (one seed each).

The best recipe of the two asym50_95 studies (window [0.068, 26.53] um, see
doc/experiment-summary.typ) is:

    inversion kernel, e=1.75, sratio=0.7, v1 two-phase "surf" order="decreasing",
    phase1_z=0, phase2_radius_factor=2, maxc=1024   (JSD 0.0093 @ N=150k)

Each of the four networks is an independent realization of this recipe (different seed for
packing AND throat generation), so the set can be used to estimate Monte-Carlo variability of
derived quantities such as the effective diffusivity. Pore sets are packed once per seed and
cached under ``examples/networks/pores_250k/``; throat generation runs in parallel via
:func:`pnm.parallel_random_PNM`. After generation every network is read back (GML round-trip),
scored against the experimental MIP throat distribution, and a gen-vs-exp comparison plot is
saved next to each GML as ``<name>_throats.png``.

Usage: python3 generate_diffusion_test_networks.py [--seeds 7,8,9,10] [--npores 250000] [--n-jobs 4]
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

from pnm import (  # noqa: E402
    inverse_transform_sampling_from_raw_data,
    invert_pore_distribution_kernel,
    jensen_shannon_divergence,
    load_mip_data,
    parallel_random_PNM,
    pore_network,
    random_packing_kdtree,
)

DATA = PROJECT_ROOT / "examples" / "data" / "GM1.csv"
OUTDIR = PROJECT_ROOT / "examples" / "networks"

# The single best recipe (see module docstring).
RECIPE = {
    "tag": "kernel_e1.75_sr0.7_dec",
    "inversion": "kernel",
    "e": 1.75,
    "sratio": 0.7,
    "order": "decreasing",
    "phase2_radius_factor": 2.0,
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="7,8,9,10", help="Comma-separated seeds (one network each).")
    p.add_argument("--npores", type=int, default=250_000, help="Number of pores per network.")
    p.add_argument("--porosity", type=float, default=0.35, help="Target pore-body porosity.")
    p.add_argument("--prtrr", type=float, default=2.0, help="Pore-to-throat radius ratio.")
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


def load_window(data: Path, low: float, high: float):
    """Asymmetric cumulative-volume window [low, high] -> MIPData (same convention as param_study_v2)."""
    base = load_mip_data(str(data), cumulative_proba=1.0)  # full curve, no filter
    F = np.cumsum(base.volumes)
    i_min = int(np.searchsorted(F, low))
    i_max = min(int(np.searchsorted(F, high)), len(F) - 1)
    minr = float(base.midradii[i_min])
    maxr = float(base.midradii[i_max])
    return load_mip_data(str(data), rmin=minr, rmax=maxr)


def pack_pore_set(mip, args, seed: int, cache_dir: Path) -> str:
    """Pack (or load from cache) the pore-only network for one seed; returns its pickle path."""
    path = pore_cache_path(cache_dir, RECIPE["tag"], args, seed)
    if path.exists():
        print(f"Pore cache hit: {path}", flush=True)
        return str(path)

    _, pdist = invert_pore_distribution_kernel(
        mip.volumes, mip.midradii, mip.deltaR,
        e=RECIPE["e"], PRTRR=args.prtrr, porosity=args.porosity, sratio=RECIPE["sratio"],
    )
    if not np.all(np.isfinite(pdist)) or abs(float(pdist.sum()) - 1.0) > 1e-6:
        raise RuntimeError("Kernel inversion produced an invalid pore distribution")

    rdist = partial(
        inverse_transform_sampling_from_raw_data, pdist, bins=mip.throat_radii * args.prtrr, high=np.inf
    )
    np.random.seed(seed)  # deterministic radius draws (same convention as random_PNM)
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
            seed=seed,
        )
    # write() converts category sets to lists in place — save a copy so the live object keeps its sets.
    copy.deepcopy(pn).write(str(path.with_suffix("")), method="pickle")  # .pnm appended by write()
    print(
        f"Packed seed {seed}: N={pn.graph.number_of_nodes()} "
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


def plot_throats(mip, gen: np.ndarray, jsd: float, l2: float, n_pores: int, n_throats: int, out_path: Path) -> None:
    """Save the single-panel gen-vs-exp throat distribution plot (study convention)."""
    fig = plt.figure(figsize=(16, 5.5))
    axt = fig.add_subplot(111)
    ax2 = axt.twinx()
    width = mip.deltaR * 0.8
    axt.bar(mip.midradii, mip.volumes, width=width, alpha=0.6, color="steelblue", label="Experimental throats")
    axt.bar(mip.midradii, np.asarray(gen), width=width, alpha=0.6, color="darkorange",
            label=f"Generated ({RECIPE['tag']})")
    ax2.plot(mip.midradii, np.cumsum(mip.volumes), "b-", lw=1.5, label="Experimental cumulative")
    ax2.plot(mip.midradii, np.cumsum(np.asarray(gen)), color="darkorange", ls="--", lw=1.5,
             label="Generated cumulative")
    axt.set_xlabel("Throat radius (um)")
    axt.set_ylabel("Normalized volume (non-cum.)")
    ax2.set_ylabel("Cumulative fraction")
    axt.set_xscale("log")
    axt.set_yscale("log")
    ax2.set_yscale("log")
    axt.grid(True, alpha=0.3)
    h1, l1 = axt.get_legend_handles_labels()
    h2, l2h = ax2.get_legend_handles_labels()
    axt.legend(h1 + h2, l1 + l2h, loc="upper left", fontsize=8)
    conn = 2.0 * n_throats / max(n_pores, 1)
    fig.suptitle(f"{RECIPE['tag']} — N={n_pores} E={n_throats} (conn {conn:.2f})\n"
                 f"L2={l2:.4f}, JSD={jsd:.4f}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUTDIR / f"pores_{args.npores}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    low, high = (float(x) for x in args.asym_frac.split(","))

    # Refuse to overwrite existing networks unless --force (checked before any expensive work).
    stamp = date.today().strftime("%Y%m%d")
    names = {s: f"{stamp}_asym5095_{args.npores // 1000}k_{RECIPE['tag']}_seed{s}" for s in seeds}
    existing = [OUTDIR / f"{n}.gml" for n in names.values() if (OUTDIR / f"{n}.gml").exists()]
    if existing and not args.force:
        print("error: output network(s) already exist (use --force to overwrite):\n  "
              + "\n  ".join(str(p) for p in existing), file=sys.stderr)
        return 1

    mip = load_window(DATA, low, high)
    print(f"Window: [{mip.midradii.min():.4f}, {mip.midradii.max():.4f}] um ({len(mip.volumes)} bins)")

    # 1) Pack one pore set per seed (sequentially; each is cached for re-runs).
    print(f"\n=== Packing {len(seeds)} pore sets at N={args.npores} ===")
    pore_paths = [pack_pore_set(mip, args, s, cache_dir) for s in seeds]

    # 2) Throat generation in parallel (one forked worker per seed).
    print(f"\n=== Generating throats in parallel ({args.n_jobs} workers) ===")
    common = dict(
        npores=args.npores,
        extent=(1.0, 1.0, 1.0),
        porosity=args.porosity,
        rdist=None,  # ignored: pores= is supplied (packing skipped)
        PRTRR=args.prtrr,
        BCgridsize=args.bcgridsize,
        throat_method="surf",
        phase1_radius_factor=2.0,
        phase1_min_neighbors=32,
        phase1_z=0,  # best of the recipe: no dedicated large-pair pass
        phase2_min_neighbors=32,
        phase2_zmin=2,
        maxc=1024,
    )
    specs = []
    for seed, pore_path in zip(seeds, pore_paths):
        name = names[seed]
        specs.append({
            **common,
            "sratio": RECIPE["sratio"],
            "order": RECIPE["order"],
            "phase2_radius_factor": RECIPE["phase2_radius_factor"],
            "seed": seed,
            "pores": pore_path,  # read inside the worker (avoids pickling the graph)
            "out_path": OUTDIR / f"{name}.gml",
        })

    t0 = time.perf_counter()
    summaries = parallel_random_PNM(specs, n_jobs=args.n_jobs)
    wall = time.perf_counter() - t0

    print(f"\n=== Generation done (wall {wall:.1f}s) ===")
    for seed, s in zip(seeds, summaries):
        mean_conn = 2.0 * s["n_throats"] / max(s["n_pores"], 1)
        print(
            f"seed {seed}: N={s['n_pores']} E={s['n_throats']} (conn {mean_conn:.2f}) "
            f"in {s['time_s']:.0f}s -> {s['path']}",
            flush=True,
        )

    # 3) Read each network back (GML round-trip check), score it and save the comparison plot.
    print("\n=== Scoring + plots ===")
    exp = np.asarray(mip.volumes, dtype=float)
    for seed, s in zip(seeds, summaries):
        pn = pore_network.read(s["path"])  # default method: GML
        gen = throat_volume_distribution(pn, mip.throat_radii)
        jsd = jensen_shannon_divergence(gen, exp)
        l2 = float(np.sqrt(np.sum((gen - exp) ** 2)))
        large_gen = 100.0 * float(gen[mip.throat_radii[:-1] >= 0.5].sum())
        large_exp = 100.0 * float(exp[mip.throat_radii[:-1] >= 0.5].sum())
        plot_throats(mip, gen, jsd, l2, s["n_pores"], s["n_throats"],
                     Path(s["path"]).with_name(Path(s["path"]).stem + "_throats.png"))
        print(
            f"seed {seed}: JSD {jsd:.4f}, L2 {l2:.4f}, "
            f"large band (>=0.5 um) {large_gen:.1f}% vs exp {large_exp:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
