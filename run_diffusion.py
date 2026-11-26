#!/usr/bin/env python3
"""Run 3-direction molecular-diffusion simulations on the four best-param kernel N=250k networks.

Thin driver over :mod:`run_diffusion_all` (same machinery, same defaults as
``examples/Simulation.ipynb``): simulates ``20261126_asym5095_250k_kernel_e1.5_sr0.3_dec_p1z0_seed*``
in ``examples/networks/`` with the throat model "conduit", diffusivity model "All" (Fick + Knudsen)
and directions x/y/z, writing everything to ``examples/diffusivity/``:

    <netname>__conduit-All-xyz.csv        one row per network (Deff per direction, BC coverage)
    summary__conduit-All-xyz.csv          Deff/Dinf ratios + timings for every simulated network
    <netname>__conduit-All-xyz_conc.gml   copy of the network with per-direction pore concentration
                                          node attributes (concentration_x/y/z)

Usage:  python3 run_diffusion.py [--n-jobs N] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import run_diffusion_all as rda  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel workers (one per network).")
    p.add_argument("--omp-threads", type=int, default=8, help="BLAS threads per worker.")
    p.add_argument("--force", action="store_true", help="Re-run networks that already have a CSV result.")
    args = p.parse_args()

    argv = [
        str(PROJECT_ROOT / "examples/networks"),
        "--pattern", "20261126_asym5095_250k_kernel_e1.5_sr0.3_dec_p1z0_seed*.gml",
        "--out-dir", str(PROJECT_ROOT / "examples/diffusivity"),
        "--throat-model", "conduit",
        "--diff-model", "All",
        "--directions", "x,y,z",
        "--save-concentrations",
        "--n-jobs", str(args.n_jobs),
        "--omp-threads", str(args.omp_threads),
    ]
    if args.force:
        argv.append("--force")
    return rda.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
