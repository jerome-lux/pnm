#!/usr/bin/env python3
"""Generate pore networks fitted to an experimental MIP throat distribution (one seed each).

Thin CLI around :func:`pnm.fitted_random_PNM`. The defaults are the best-fit recipe of the
asym50_95 study: plain ``kernel`` inversion with e = 1.5 / sratio = 0.3 and two-phase surf
order="decreasing", phase1_z=0, phase2_radius_factor=3, maxc=1024 (JSD 0.0059 at N = 250k seed 7,
stable across seeds 7–10; see doc/pore-distribution-inversion.pdf §5). Each seed is an independent
realization of the recipe
(different packing AND throat generation), so a set of seeds estimates Monte-Carlo variability of
derived quantities such as the effective diffusivity.

Pore sets are packed once per seed and cached under ``<out-dir>/pores_<N>/`` (reused on re-runs);
throat generation then runs in parallel forked workers, each network is saved as GML in
``--out-dir``, read back (GML round-trip check), scored against the experimental MIP throat
distribution, and a gen-vs-exp comparison plot is saved next to it as ``<name>_throats.png``.

Usage: python3 generate_networks.py [--seeds 7,8,9,10] [--npores 250000] [--out-dir DIR] [--n-jobs 4]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pnm  # noqa: E402

DATA = PROJECT_ROOT / "examples" / "data" / "GM1.csv"


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments (see module docstring for defaults).

    Parameters
    ----------
    argv : list of str or None
        Argument vector; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="7,8,9,10", help="Comma-separated seeds (one network each).")
    p.add_argument("--npores", type=int, default=250_000, help="Number of pores per network.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for the GML networks and pore cache "
                        "(default: examples/networks).")
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel workers (one per seed).")
    p.add_argument("--force", action="store_true", help="Overwrite existing output networks.")
    # Experimental window and recipe parameters (defaults = best-fit kernel recipe).
    p.add_argument("--data", default=str(DATA), help="Experimental MIP CSV file.")
    p.add_argument("--asym-frac", default="0.5,0.95",
                   help="Asymmetric cumulative-volume window 'low,high' on the full MIP curve "
                        "(study convention; 0.5,0.95 = asym50_95).")
    p.add_argument("--porosity", type=float, default=0.35, help="Target pore-body porosity.")
    p.add_argument("--prtrr", type=float, default=2.0, help="Pore-to-throat radius ratio.")
    p.add_argument("--bcgridsize", type=int, default=150, help="Boundary-detection grid resolution.")
    p.add_argument("--inversion", default="kernel",
                   choices=["kernel_budget", "kernel_waterfill", "kernel", "kernel_gamma", "independent"],
                   help="Pore-distribution inversion (default: best-fit kernel recipe).")
    p.add_argument("--e", type=float, default=1.5,
                   help="Step-1 corrective exponent: mean capillary length modelled as r**e "
                        "(0 = pure cross-section correction; ignored by the parameter-free kernel_gamma).")
    p.add_argument("--sratio", type=float, default=0.3, help="Surface coverage ratio (in [0, 1]).")
    p.add_argument("--order", default="decreasing", choices=["increasing", "decreasing", "random"],
                   help='Phase-2 processing order of the two-phase "surf" method.')
    p.add_argument("--phase1-z", type=int, default=0,
                   help="Largest-partner seeding count (0 disables phase 1).")
    p.add_argument("--phase2-radius-factor", type=float, default=3.0,
                   help="Phase-2 candidate-ball factor of the two-phase 'surf' method.")
    p.add_argument("--maxc", type=int, default=1024, help="Maximum pore connectivity (zmax).")
    return p.parse_args(argv)


def recipe_tag(args) -> str:
    """Recipe identifier used in output names and the pore cache key.

    The parameter-free ``kernel_gamma`` inversion ignores ``--e``, so it is left out of the tag
    (and hence of the pore-cache key): two runs differing only in ``--e`` are the same recipe there.
    """
    if args.inversion == "kernel_gamma":
        return f"{args.inversion}_sr{args.sratio:g}_{args.order}"
    return f"{args.inversion}_e{args.e:g}_sr{args.sratio:g}_{args.order}"


def load_window(data: Path, low: float, high: float):
    """Asymmetric cumulative-volume window [low, high] -> MIPData (same convention as the studies).

    Parameters
    ----------
    data : Path
        Experimental MIP CSV file.
    low : float
        Lower cumulative-volume fraction of the window.
    high : float
        Upper cumulative-volume fraction of the window.

    Returns
    -------
    MIPData
        Binned experimental throat distribution restricted to the window.
    """
    base = pnm.load_mip_data(str(data), cumulative_proba=1.0)  # full curve, no filter
    F = np.cumsum(base.volumes)
    i_min = int(np.searchsorted(F, low))
    i_max = min(int(np.searchsorted(F, high)), len(F) - 1)
    minr = float(base.midradii[i_min])
    maxr = float(base.midradii[i_max])
    return pnm.load_mip_data(str(data), rmin=minr, rmax=maxr)


def pore_cache_path(cache_dir: Path, tag: str, args, seed: int) -> Path:
    """Pore-set cache file keyed by every input that changes the packed pore set.

    Parameters
    ----------
    cache_dir : Path
        Cache directory (``<out-dir>/pores_<N>/``).
    tag : str
        Recipe identifier (inversion, e, sratio, order).
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
    base = f"pores_{tag}_n{args.npores}_phi{args.porosity:g}_seed{seed}"
    return cache_dir / f"{base}_w{low:g}-{high:g}_prtrr{args.prtrr:g}_bc{args.bcgridsize}.pnm"


def network_name(args, tag: str, seed: int) -> str:
    """Output network name: generation date, window, N, recipe and seed (never a fixed date)."""
    low, high = (float(x) for x in args.asym_frac.split(","))
    stamp = date.today().strftime("%Y%m%d")
    return f"{stamp}_w{low:g}-{high:g}_{args.npores // 1000}k_{tag}_seed{seed}"


def _worker(spec: dict) -> dict:
    """Generate one network in this process (forked worker); save its GML and return a summary.

    Parameters
    ----------
    spec : dict
        Keyword arguments for :func:`pnm.fitted_random_PNM` plus ``"out_path"`` (GML path).

    Returns
    -------
    dict
        Summary ``{"path", "n_pores", "n_throats", "time_s"}``.
    """
    out_path = Path(spec.pop("out_path"))
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # keep the worker's progress output out of the log
        pn = pnm.fitted_random_PNM(**spec)
    elapsed = time.perf_counter() - t0

    base = str(out_path)
    if base.endswith(".gml"):
        base = base[: -len(".gml")]  # write() appends the extension itself
    pn.write(base, method="gml")
    return {
        "path": base + ".gml",
        "n_pores": pn.graph.number_of_nodes(),
        "n_throats": pn.graph.number_of_edges(),
        "time_s": elapsed,
    }


def throat_volume_distribution(pn: pnm.pore_network, bins: np.ndarray) -> np.ndarray:
    """Normalized volume-weighted throat radius distribution binned on ``bins``.

    Same convention as the study scripts (first bin [b0, b1], others (bi, bi+1]).

    Parameters
    ----------
    pn : pore_network
        Generated network.
    bins : np.ndarray
        Throat-radius bin edges (n_bins + 1).

    Returns
    -------
    numpy.ndarray
        Normalized per-bin throat volumes (length n_bins).
    """
    radii = np.fromiter(nx.get_edge_attributes(pn.graph, "radius").values(), dtype=float)
    vol = np.fromiter(nx.get_edge_attributes(pn.graph, "volume").values(), dtype=float)
    valid = (radii >= bins[0]) & (radii <= bins[-1])
    idx = np.clip(np.searchsorted(bins, radii[valid], side="right") - 1, 0, len(bins) - 2)
    data = np.zeros(len(bins) - 1)
    np.add.at(data, idx, vol[valid])
    return data / data.sum() if data.sum() > 0 else data


def plot_throats(mip, gen: np.ndarray, jsd: float, l2: float, n_pores: int, n_throats: int, tag: str, out_path: Path) -> None:
    """Save the single-panel gen-vs-exp throat distribution plot (study convention).

    Parameters
    ----------
    mip : MIPData
        Windowed experimental throat distribution.
    gen : np.ndarray
        Generated normalized per-bin throat volumes.
    jsd, l2 : float
        JSD and L2 distance of ``gen`` against the experiment.
    n_pores, n_throats : int
        Network size (for the title).
    tag : str
        Recipe identifier (for the legend/title).
    out_path : Path
        Output PNG path.
    """
    fig = plt.figure(figsize=(16, 5.5))
    axt = fig.add_subplot(111)
    ax2 = axt.twinx()
    width = mip.deltaR * 0.8
    axt.bar(mip.midradii, mip.volumes, width=width, alpha=0.6, color="steelblue", label="Experimental throats")
    axt.bar(mip.midradii, np.asarray(gen), width=width, alpha=0.6, color="darkorange",
            label=f"Generated ({tag})")
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
    fig.suptitle(f"{tag} — N={n_pores} E={n_throats} (conn {conn:.2f})\n"
                 f"L2={l2:.4f}, JSD={jsd:.4f}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    """Pack the pore sets, generate the networks in parallel, score and plot them.

    Parameters
    ----------
    argv : list of str or None
        Argument vector; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else PROJECT_ROOT / "examples" / "networks"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / f"pores_{args.npores}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    low, high = (float(x) for x in args.asym_frac.split(","))
    tag = recipe_tag(args)

    # Refuse to overwrite existing networks unless --force (checked before any expensive work).
    names = {s: network_name(args, tag, s) for s in seeds}
    existing = [out_dir / f"{n}.gml" for n in names.values() if (out_dir / f"{n}.gml").exists()]
    if existing and not args.force:
        print("error: output network(s) already exist (use --force to overwrite):\n  "
              + "\n  ".join(str(p) for p in existing), file=sys.stderr)
        return 1

    mip = load_window(Path(args.data), low, high)
    print(f"Window: [{mip.midradii.min():.4f}, {mip.midradii.max():.4f}] um ({len(mip.volumes)} bins)")
    print(f"Recipe: {tag} (porosity={args.porosity:g}, PRTRR={args.prtrr:g})")

    # 1) Generate the networks in parallel (one forked worker per seed; pore cache inside).
    specs = []
    for seed in seeds:
        specs.append({
            "mip": mip,
            "npores": args.npores,
            "seed": seed,
            "porosity": args.porosity,
            "PRTRR": args.prtrr,
            "inversion": args.inversion,
            "e": args.e,
            "sratio": args.sratio,
            "BCgridsize": args.bcgridsize,
            "order": args.order,
            "phase1_z": args.phase1_z,
            "phase2_radius_factor": args.phase2_radius_factor,
            "maxc": args.maxc,
            "pore_cache_path": pore_cache_path(cache_dir, tag, args, seed),
            "out_path": out_dir / f"{names[seed]}.gml",
        })

    print(f"\n=== Generating {len(seeds)} network(s) at N={args.npores} ({max(1, min(args.n_jobs, len(specs)))} worker(s)) ===")
    t0 = time.perf_counter()
    if len(specs) == 1 or args.n_jobs <= 1:
        summaries = [_worker(dict(s)) for s in specs]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(args.n_jobs, len(specs))) as pool:
            summaries = pool.map(_worker, specs)
    wall = time.perf_counter() - t0

    print(f"\n=== Generation done (wall {wall:.1f}s) ===")
    for seed, s in zip(seeds, summaries):
        mean_conn = 2.0 * s["n_throats"] / max(s["n_pores"], 1)
        print(
            f"seed {seed}: N={s['n_pores']} E={s['n_throats']} (conn {mean_conn:.2f}) "
            f"in {s['time_s']:.0f}s -> {s['path']}",
            flush=True,
        )

    # 2) Read each network back (GML round-trip check), score it and save the comparison plot.
    print("\n=== Scoring + plots ===")
    exp = np.asarray(mip.volumes, dtype=float)
    for seed, s in zip(seeds, summaries):
        pn = pnm.pore_network.read(s["path"])  # default method: GML
        gen = throat_volume_distribution(pn, mip.throat_radii)
        jsd = pnm.jensen_shannon_divergence(gen, exp)
        l2 = float(np.sqrt(np.sum((gen - exp) ** 2)))
        large_gen = 100.0 * float(gen[mip.throat_radii[:-1] >= 0.5].sum())
        large_exp = 100.0 * float(exp[mip.throat_radii[:-1] >= 0.5].sum())
        plot_throats(mip, gen, jsd, l2, s["n_pores"], s["n_throats"], tag,
                     Path(s["path"]).with_name(Path(s["path"]).stem + "_throats.png"))
        print(
            f"seed {seed}: JSD {jsd:.4f}, L2 {l2:.4f}, "
            f"large band (>=0.5 um) {large_gen:.1f}% vs exp {large_exp:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
