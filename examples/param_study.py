#!/usr/bin/env python3
"""Parametric study: find (e, sratio) that best reproduces the experimental throat size distribution.

The generation pipeline is identical to ``examples/PNM-generation.ipynb``:

1. Load an experimental MIP CSV (cumulative intruded volume vs diameter).
2. Derive the per-bin throat volume distribution and apply a cumulative-probability threshold.
3. Predict the pore distribution from the throat distribution using the corrective factor
   ``e`` (the "alpha" of the study) and the iterative algorithm of the notebook.
4. Generate a random pore network with ``random_PNM(packing_method="kdtree")`` for each
   ``(e, sratio)`` grid point, using the **same seed** for every run so that only the
   studied parameters vary between runs.
5. Compute the generated throat volume distribution on the experimental bins and compare it
   to the experimental one with two metrics:

   - ``l2``: Euclidean distance between the normalized non-cumulative distributions;
   - ``ks``: Kolmogorov-Smirnov statistic between the cumulative distributions.

Outputs (in ``--outdir``, default ``examples/param_study_results``):

- ``results.csv``: one row per grid point (e, sratio, n_pores, n_throats, l2, ks, time_s);
- ``heatmap_l2.png``: L2 distance over the (sratio, e) grid with the best point marked;
- ``best_match.png``: experimental vs generated throat distribution for the best parameters.

Usage example::

    python examples/param_study.py --npores 10000 \
        --e-grid 1.0,1.25,1.5,1.75,2.0 --sratio-grid 0.3,0.4,0.5,0.6,0.7

Keep only experimental throats in a radius interval (drops both smallest and largest) and
report the lost volume fraction::

    python examples/param_study.py --rmin 0.01 --rmax 2.0

Notes
-----
- All other generation parameters (PRTRR, porosity, minc, maxc, BCgridsize) are fixed to the
  notebook defaults so that only ``e`` and ``sratio`` vary. PRTRR is forwarded to
  ``random_PNM`` so throat radii are ``min(r1, r2)/PRTRR``, consistent with the pore bins
  built as ``throat_radii * PRTRR``.
- By default the experimental data are filtered with the notebook's 95% cumulative threshold
  (largest throats dropped); ``--rmin``/``--rmax`` select an explicit radius interval instead,
  and the excluded volume fraction ("porosité perdue") is printed for information.
- The default network size is 10000 pores: with ~9 throats per pore this yields ~45k throats,
  enough samples for a stable distribution comparison while keeping each run under ~10 s.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: plots are saved to files only

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pnm.Generators import random_PNM  # noqa: E402
from pnm.Stats import inverse_transform_sampling_from_raw_data  # noqa: E402


# ----------------------------------------------------------------------
# Experimental data (identical to notebook cells 10 + 13)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MIPData:
    """Experimental MIP throat distribution and its binning."""

    name: str
    throat_radii: np.ndarray  # bin edges, ascending (len = n_bins + 1)
    volumes: np.ndarray       # normalized intruded volume per kept bin (len = n_bins)
    deltaR: np.ndarray        # bin widths (len = n_bins)
    midradii: np.ndarray      # bin centers (len = n_bins)
    lost_volume_frac: float   # fraction of total intruded volume excluded by the filter


def load_mip_data(
    filepath: Path, cumulative_proba: float = 0.95, rmin: float | None = None, rmax: float | None = None
) -> MIPData:
    """Load an experimental MIP CSV and derive the per-bin throat volume distribution.

    Replicates notebook cells 10 and 13 exactly: cumulative volume is read from a
    ``;``-separated CSV (diameter, cumulative intruded volume), sorted by ascending
    diameter, converted to per-bin volumes with ``-np.diff`` (radii kept as bin edges:
    n_bins + 1 values for n_bins bins), normalized, then filtered.

    Two filtering modes are available:

    - default (``rmin is None and rmax is None``): keep bins from the smallest radius up to
      the largest bin whose cumulative volume stays below ``cumulative_proba`` (the notebook
      behavior; drops the largest throats);
    - radius interval: if ``rmin``/``rmax`` are given, keep only the bins whose center lies
      in ``[rmin, rmax]`` (drops both the smallest and the largest throats outside the
      interval).

    The fraction of total intruded volume excluded by the filter is reported as
    ``lost_volume_frac`` ("porosité perdue" for information).

    Parameters
    ----------
    filepath : Path or str
        Path to the MIP CSV file (e.g. ``examples/data/GM1.csv``).
    cumulative_proba : float, optional
        Cumulative volume fraction used as threshold in default mode (default 0.95).
    rmin : float or None, optional
        Minimum throat radius to keep (None = no lower bound).
    rmax : float or None, optional
        Maximum throat radius to keep (None = no upper bound).

    Returns
    -------
    MIPData
        Bin edges, normalized per-bin volumes, bin widths, bin centers and lost volume fraction.
    """
    filepath = Path(filepath)
    diameters: list[float] = []
    volumes_cum: list[float] = []
    with open(filepath, "r", newline="") as infile:
        reader = csv.reader(infile, delimiter=";")
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                diameters.append(float(row[0].replace(",", ".")))
                volumes_cum.append(float(row[1].replace(",", ".")))

    diameters = np.array(diameters)
    volumes_cum = np.array(volumes_cum)

    sort_idx = np.argsort(diameters)
    diameters = diameters[sort_idx]
    volumes_cum = volumes_cum[sort_idx]

    # Throat radii are bin edges: bin i spans [throat_radii[i], throat_radii[i+1]].
    throat_radii = diameters / 2.0
    # Per-bin intruded volume (cumulative decreases when read from large to small pores).
    volumes = -np.diff(volumes_cum)
    volumes = np.array(volumes, dtype=float)
    volumes = volumes / volumes.sum()

    n = len(volumes)                              # number of bins
    dR_full = np.diff(throat_radii)               # length n (n_bins + 1 edges)
    mid_full = throat_radii[:-1] + 0.5 * dR_full  # bin centers, length n

    if rmin is not None or rmax is not None:
        # Keep bins whose center lies in [rmin, rmax].
        lo = 0 if rmin is None else int(np.searchsorted(mid_full, rmin, side="left"))
        hi = n - 1 if rmax is None else int(np.searchsorted(mid_full, rmax, side="right") - 1)
        lo = max(0, min(lo, n - 1))
        hi = max(lo, min(hi, n - 1))
    else:
        # Notebook behavior: threshold on cumulative probability (drops the largest throats).
        # Small tolerance so that cumulative_proba=1 keeps all bins despite float rounding.
        maxindex = int(np.argwhere(np.cumsum(volumes) <= cumulative_proba + 1e-9).max())
        lo, hi = 0, maxindex

    kept_frac = float(volumes[lo : hi + 1].sum())
    throat_radii = throat_radii[lo : hi + 2]
    volumes = volumes[lo : hi + 1]
    volumes = volumes / volumes.sum()

    # Recompute bin geometry after all slicing.
    deltaR = np.diff(throat_radii)
    midradii = throat_radii[:-1] + deltaR * 0.5

    return MIPData(
        name=filepath.stem,
        throat_radii=throat_radii,
        volumes=volumes,
        deltaR=deltaR,
        midradii=midradii,
        lost_volume_frac=1.0 - kept_frac,
    )


# ----------------------------------------------------------------------
# Pore distribution prediction (identical to notebook cell 15)
# ----------------------------------------------------------------------
def predict_pore_distribution(
    volumes: np.ndarray, midradii: np.ndarray, deltaR: np.ndarray, e: float
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the pore size distribution from the experimental throat distribution.

    Applies the corrective factor ``e`` to the per-bin throat volumes (notebook cell 15):
    ``cdist[i] = volumes[i] / (pi * (midradii[i]^2 + deltaR[i]^2/3)^e)``, then runs the
    iterative algorithm that yields a pore distribution which, when pores are connected by
    throats of radius ``min(r1, r2)/PRTRR``, recovers the measured throat distribution.

    Parameters
    ----------
    volumes : np.ndarray
        Normalized experimental per-bin throat volumes (len = n_bins).
    midradii : np.ndarray
        Throat bin centers (len = n_bins).
    deltaR : np.ndarray
        Throat bin widths (len = n_bins).
    e : float
        Empirical corrective factor ("alpha"; 1 = no correction).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(cdist, pdist)``: the corrected normalized throat distribution and the predicted
        pore distribution, both of length ``n_bins``.
    """
    cdist = np.array(
        [volumes[i] / (np.pi * (midradii[i] ** 2 + (1.0 / 3) * deltaR[i] ** 2) ** e) for i in range(len(volumes))]
    )
    cdist /= cdist.sum()

    pdist = np.zeros_like(cdist)
    pdist[-1] = np.sqrt(cdist[-1])
    for i in range(pdist.size - 1, -1, -1):
        s = pdist[i + 1 :].sum()
        pdist[i] = -s + np.sqrt(s**2 + cdist[i])

    return cdist, pdist


# ----------------------------------------------------------------------
# Generated throat distribution (identical to stats(mode='vol', elements='throats'))
# ----------------------------------------------------------------------
def throat_volume_distribution(pn, bins: np.ndarray) -> np.ndarray:
    """Volume-weighted throat radius distribution binned on ``bins``.

    Replicates the binning convention of ``pnm.Stats.stats(mode="vol", elements="throats")``
    without any matplotlib dependency: first bin is ``[bins[0], bins[1]]``, all other bins
    are ``(bins[i], bins[i+1]]``, result normalized by total throat volume.

    Parameters
    ----------
    pn : pore_network
        Generated network (edges must have ``radius`` and ``volume`` attributes).
    bins : np.ndarray
        Bin edges, ascending (same as the experimental ``throat_radii``).

    Returns
    -------
    np.ndarray
        Normalized throat volume per bin (len = len(bins) - 1).
    """
    t_radii = np.fromiter(nx.get_edge_attributes(pn.graph, "radius").values(), dtype=float)
    t_vol = np.fromiter(nx.get_edge_attributes(pn.graph, "volume").values(), dtype=float)

    idx = np.searchsorted(bins, t_radii, side="right") - 1
    valid = (t_radii >= bins[0]) & (t_radii <= bins[-1])
    idx = np.clip(idx, 0, len(bins) - 2)

    data = np.zeros(len(bins) - 1)
    np.add.at(data, idx[valid], t_vol[valid])
    return data / data.sum() if data.sum() > 0 else data


# ----------------------------------------------------------------------
# One grid point
# ----------------------------------------------------------------------
def run_one(
    mip: MIPData,
    e: float,
    sratio: float,
    npores: int,
    porosity: float,
    PRTRR: float,
    minc: int,
    maxc: int,
    BCgridsize: int,
    seed: int | None,
) -> dict:
    """Generate one network for a (e, sratio) pair and score it against the experiment.

    Parameters
    ----------
    mip : MIPData
        Experimental data (binning + target distribution).
    e : float
        Corrective factor ("alpha").
    sratio : float
        Maximum pore surface fraction covered by throats.
    npores, porosity, PRTRR, minc, maxc, BCgridsize : generation parameters (fixed across the grid)
    seed : int or None
        Random seed; identical for every grid point so only (e, sratio) vary.

    Returns
    -------
    dict
        Row of results: e, sratio, n_pores, n_throats, l2, ks, time_s.
    """
    t0 = time.perf_counter()

    _, pdist = predict_pore_distribution(mip.volumes, mip.midradii, mip.deltaR, e)
    pore_radii_bins = mip.throat_radii * PRTRR
    rdist = partial(
        inverse_transform_sampling_from_raw_data,
        pdist,
        bins=pore_radii_bins[:-1],
        high=np.inf,
    )

    # Suppress the per-pass progress prints of the generation routines (useless in batch mode;
    # warnings and tracebacks still go to stderr). The caller prints one summary line per run.
    with contextlib.redirect_stdout(io.StringIO()):
        pn = random_PNM(
            npores=npores,
            extent=(1.0, 1.0, 1.0),
            porosity=porosity,
            rdist=rdist,
            sratio=sratio,
            PRTRR=PRTRR,
            minc=minc,
            maxc=maxc,
            BCgridsize=BCgridsize,
            packing_method="kdtree",
            packing_verbose=False,
            seed=seed,
        )
        pn.compute_geometry(autothroats=True)

    gen = throat_volume_distribution(pn, mip.throat_radii)
    exp = mip.volumes

    l2 = float(np.sqrt(np.sum((gen - exp) ** 2)))
    # KS-style distance between the two discrete CDFs on the common bin grid.
    ks = float(np.max(np.abs(np.cumsum(gen) - np.cumsum(exp))))

    return {
        "e": e,
        "sratio": sratio,
        "n_pores": pn.graph.number_of_nodes(),
        "n_throats": pn.graph.number_of_edges(),
        "l2": l2,
        "ks": ks,
        "time_s": time.perf_counter() - t0,
        # Kept in memory (small: one value per experimental bin) so the best-match plot
        # does not require regenerating the network. Excluded from the CSV by extrasaction.
        "gen_dist": gen,
    }


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_heatmap(rows: list[dict], outdir: Path) -> None:
    """Save the L2-distance heatmap over the (sratio, e) grid with the best point marked."""
    es = sorted({r["e"] for r in rows})
    sratios = sorted({r["sratio"] for r in rows})
    grid = np.full((len(es), len(sratios)), np.nan)
    for r in rows:
        grid[es.index(r["e"]), sratios.index(r["sratio"])] = r["l2"]

    best = min(rows, key=lambda r: r["l2"])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.pcolormesh(sratios, es, grid.T, shading="auto", cmap="viridis")
    for i, e in enumerate(es):
        for j, s in enumerate(sratios):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8)
    bi, bj = es.index(best["e"]), sratios.index(best["sratio"])
    ax.plot(sratios[bj], es[bi], "r*", markersize=18, markeredgecolor="white")
    ax.set_xlabel("sratio (surface coverage)")
    ax.set_ylabel("e (corrective factor)")
    ax.set_title(f"L2 distance to experimental throat distribution — best: e={best['e']}, sratio={best['sratio']}")
    fig.colorbar(im, ax=ax, label="L2")
    fig.tight_layout()
    fig.savefig(outdir / "heatmap_l2.png", dpi=150)
    plt.close(fig)


def plot_best_match(mip: MIPData, best: dict, outdir: Path) -> None:
    """Save the experimental vs generated throat distribution for the best parameters.

    ``best['gen_dist']`` must hold the generated normalized volume distribution computed by
    :func:`run_one` (kept in memory so no network regeneration is needed).
    """
    gen = np.asarray(best["gen_dist"])

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()
    width = mip.deltaR * 0.8
    ax1.bar(mip.midradii, mip.volumes, width=width, alpha=0.6, color="steelblue", label="Experimental throats")
    ax1.bar(mip.midradii, gen, width=width, alpha=0.6, color="darkorange", label=f"Generated (e={best['e']}, sratio={best['sratio']})")
    ax2.plot(mip.midradii, np.cumsum(mip.volumes), "b-", lw=1.5, label="Experimental cumulative")
    ax2.plot(mip.midradii, np.cumsum(gen), color="darkorange", ls="--", lw=1.5, label="Generated cumulative")

    ax1.set_xlabel("Throat radius")
    ax1.set_ylabel("Normalized volume (non-cum.)")
    ax2.set_ylabel("Cumulative fraction")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax1.set_title(f"Best match: e={best['e']}, sratio={best['sratio']} (L2={best['l2']:.4f}, KS={best['ks']:.4f})")
    fig.tight_layout()
    fig.savefig(outdir / "best_match.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the parametric study."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", default=str(PROJECT_ROOT / "examples" / "data" / "GM1.csv"), help="MIP CSV file")
    p.add_argument("--npores", type=int, default=10000, help="target number of pores (default 10000)")
    p.add_argument("--porosity", type=float, default=0.35, help="target pore-body porosity (default 0.35)")
    p.add_argument("--PRTRR", type=float, default=2.0, help="pore-to-throat radius ratio (default 2.0)")
    p.add_argument("--minc", type=int, default=2, help="minimum pore connectivity (default 2)")
    p.add_argument("--maxc", type=int, default=512, help="maximum pore connectivity (default 512)")
    p.add_argument("--BCgridsize", type=int, default=100, help="boundary grid resolution (default 100)")
    p.add_argument("--seed", type=int, default=7, help="random seed shared by all runs (default 7; use -1 for none)")
    p.add_argument("--e-grid", default="1.0,1.25,1.5,1.75,2.0", help="comma-separated e values")
    p.add_argument("--sratio-grid", default="0.3,0.4,0.5,0.6,0.7", help="comma-separated sratio values")
    p.add_argument(
        "--rmin", type=float, default=None,
        help="minimum throat radius to keep from the experimental data (None = notebook 95%% cumulative threshold)",
    )
    p.add_argument(
        "--rmax", type=float, default=None,
        help="maximum throat radius to keep from the experimental data (None = no upper bound)",
    )
    p.add_argument(
        "--outdir",
        default=str(PROJECT_ROOT / "examples" / "param_study_results"),
        help="output directory for results and plots",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the (e, sratio) grid study and save results + plots. Returns a process exit code."""
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mip = load_mip_data(Path(args.data_file), rmin=args.rmin, rmax=args.rmax)
    filter_desc = (
        f"interval [{args.rmin}, {args.rmax}]"
        if (args.rmin is not None or args.rmax is not None)
        else "95% cumulative threshold (notebook default)"
    )
    print(f"Data: {mip.name} — {len(mip.volumes)} throat bins, "
          f"radius range {mip.throat_radii.min():.4f}-{mip.throat_radii.max():.4f}")
    print(f"Filter: {filter_desc} — lost volume fraction (porosité perdue): {100 * mip.lost_volume_frac:.2f}%")

    e_grid = [float(x) for x in args.e_grid.split(",")]
    sratio_grid = [float(x) for x in args.sratio_grid.split(",")]
    seed = None if args.seed < 0 else args.seed

    rows: list[dict] = []
    total = len(e_grid) * len(sratio_grid)
    n_done = 0
    csv_path = outdir / "results.csv"
    fieldnames = ["e", "sratio", "n_pores", "n_throats", "l2", "ks", "time_s"]

    # Single handle, truncated at start: results are written incrementally (crash-resilient)
    # and a rerun never appends to stale rows.
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in e_grid:
            for sratio in sratio_grid:
                n_done += 1
                row = run_one(
                    mip, e, sratio, args.npores, args.porosity, args.PRTRR,
                    args.minc, args.maxc, args.BCgridsize, seed,
                )
                rows.append(row)
                writer.writerow(row)
                f.flush()

                print(
                    f"[{n_done}/{total}] e={e:<5} sratio={sratio:<4} "
                    f"throats={row['n_throats']:<7} L2={row['l2']:.4f} KS={row['ks']:.4f} "
                    f"({row['time_s']:5.1f}s)",
                    flush=True,
                )

    best = min(rows, key=lambda r: r["l2"])
    print(f"\nBest parameters (min L2): e={best['e']}, sratio={best['sratio']} "
          f"(L2={best['l2']:.4f}, KS={best['ks']:.4f})")

    plot_heatmap(rows, outdir)
    plot_best_match(mip, best, outdir)
    print(f"Results saved to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
