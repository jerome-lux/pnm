# coding=utf-8
"""Experimental mercury intrusion porosimetry (MIP) data handling.

Loads the ``;``-separated MIP CSV files used by the examples (diameter vs
cumulative intruded volume), converts them into a per-bin throat VOLUME
distribution with explicit bin edges, and applies either a radius window or a
cumulative-volume threshold filter. This is the single source of truth for the
experimental data: both ``examples/PNM-generation.ipynb`` and
``examples/param_study.py`` use :func:`load_mip_data` so that every generated
network is scored against exactly the same experimental distribution.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MIPData:
    """Experimental MIP throat distribution and its binning.

    Attributes
    ----------
    name : str
        Base name of the source CSV file (without extension).
    throat_radii : numpy.ndarray
        Throat radius BIN EDGES, ascending (len = n_bins + 1): bin i spans
        ``[throat_radii[i], throat_radii[i+1]]``.
    volumes : numpy.ndarray
        Normalized intruded volume per kept bin (len = n_bins); sums to 1 over
        the kept bins only.
    deltaR : numpy.ndarray
        Bin widths (len = n_bins).
    midradii : numpy.ndarray
        Bin centers (len = n_bins).
    lost_volume_frac : float
        Fraction of total intruded volume excluded by the filter ("porosité
        perdue", for information only).
    diameters : numpy.ndarray
        Raw measured capillary DIAMETERS in µm, sorted ascending (len =
        n_full_bins + 1); unfiltered, i.e. before any radius window or
        cumulative threshold is applied.
    volumes_cum : numpy.ndarray
        Raw cumulative intruded volume at each diameter of ``diameters``
        (unsorted order preserved after sorting by diameter; decreases when
        read from large to small pores). Unfiltered.
    """

    name: str
    throat_radii: np.ndarray  # bin edges, ascending (len = n_bins + 1)
    volumes: np.ndarray  # normalized intruded volume per kept bin (len = n_bins)
    deltaR: np.ndarray  # bin widths (len = n_bins)
    midradii: np.ndarray  # bin centers (len = n_bins)
    lost_volume_frac: float  # fraction of total intruded volume excluded by the filter
    diameters: np.ndarray  # raw diameters (µm), sorted ascending, unfiltered
    volumes_cum: np.ndarray  # raw cumulative intruded volume per diameter, unfiltered


def load_mip_data(
    filepath: Path | str, cumulative_proba: float = 0.95, rmin: float | None = None, rmax: float | None = None
) -> MIPData:
    """Load an experimental MIP CSV and derive the per-bin throat volume distribution.

    The CSV is ``;``-separated with a header row and two columns (capillary
    diameter in µm, cumulative intruded volume). Diameters are sorted
    ascending, converted to throat radii (diameter / 2) which serve as BIN
    EDGES (n_bins + 1 values for n_bins bins), and the per-bin intruded volume
    is computed with ``-np.diff`` of the cumulative column (the cumulative
    decreases when read from large to small pores). Volumes are normalized so
    that they sum to 1 over the kept bins.

    Two filtering modes are available:

    - default (``rmin is None and rmax is None``): keep bins from the smallest
      radius up to the largest bin whose cumulative volume stays below
      ``cumulative_proba`` (drops the largest throats);
    - radius interval: if ``rmin``/``rmax`` are given, keep only the bins whose
      center lies in ``[rmin, rmax]`` (drops both the smallest and the largest
      throats outside the interval).

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
        Bin edges, normalized per-bin volumes, bin widths, bin centers, lost
        volume fraction and the raw unfiltered diameter/cumulative-volume data.
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
    total_vol = float(volumes.sum())
    volumes = volumes / total_vol

    n = len(volumes)  # number of bins
    dR_full = np.diff(throat_radii)  # length n (n_bins + 1 edges)
    mid_full = throat_radii[:-1] + 0.5 * dR_full  # bin centers, length n

    if rmin is not None or rmax is not None:
        # Keep bins whose center lies in [rmin, rmax].
        lo = 0 if rmin is None else int(np.searchsorted(mid_full, rmin, side="left"))
        hi = n - 1 if rmax is None else int(np.searchsorted(mid_full, rmax, side="right") - 1)
        lo = max(0, min(lo, n - 1))
        hi = max(lo, min(hi, n - 1))
    else:
        # Default mode: threshold on cumulative probability (drops the largest throats).
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
        diameters=diameters,
        volumes_cum=volumes_cum,
    )
