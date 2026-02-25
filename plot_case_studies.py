#!/usr/bin/env python3
"""
Thesis-ready plots for all case studies.
Generates PDF figures for DepFiN and Eyeriss architecture sweeps
and layer-fusion comparisons.

Usage:
    python plot_case_studies.py          # generate all plots
    python plot_case_studies.py --show   # also display interactively
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import OrderedDict

# ────────────────────────────────────────────────────────────────────
# Style configuration
# ────────────────────────────────────────────────────────────────────
plt.style.use("classic")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
WORKLOADS = ["FSRCNN", "MC-CNN", "VGG16", "ResNet18"]
COLORS = {"FSRCNN": "#1f77b4", "MC-CNN": "#ff7f0e",
          "VGG16": "#2ca02c", "ResNet18": "#d62728"}
MARKERS = {"FSRCNN": "o", "MC-CNN": "s", "VGG16": "^", "ResNet18": "D"}

# Fusion-specific colors
FUSION_COLORS = {"Full": "#d62728", "Partial": "#ff7f0e", "Single": "#2ca02c"}


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    fig.savefig(path)
    print(f"  saved {path}")


def _sci_fmt(ax, axis="y"):
    """Add scientific-notation formatter."""
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)


# ====================================================================
#  DATA  –  DepFiN Case Studies
# ====================================================================

# CS1 — Tile Size Sensitivity Sweep (fixed PE grid = 16×128)
DEPFIN_CS1 = {
    "FSRCNN": {
        "tiles": [120, 64, 32, 16, 8],
        "energy": [8.356e3, 8.364e3, 8.381e3, 8.433e3, 8.524e3],
        "latency": [6.156e6, 1.154e7, 2.308e7, 4.622e7, 9.244e7],
        "edp": [6.01e4, 1.13e5, 2.26e5, 4.55e5, 9.18e5],
    },
    "MC-CNN": {
        "tiles": [69, 46, 27, 18, 9, 3],
        "energy": [1.127e4, 1.133e4, 1.138e4, 1.142e4, 1.150e4, 1.201e4],
        "latency": [1.218e7, 1.827e7, 3.114e7, 4.671e7, 9.343e7, 2.803e8],
        "edp": [1.64e5, 2.46e5, 4.18e5, 6.31e5, 1.27e6, 3.95e6],
    },
    "VGG16": {
        "tiles": [14, 7, 2, 1],
        "energy": [7.607e4, 7.724e4, 8.326e4, 9.396e4],
        "latency": [2.728e7, 5.024e7, 1.705e8, 3.516e8],
        "edp": [3.82e6, 7.09e6, 2.51e7, 5.55e7],
    },
    "ResNet18": {
        "tiles": [7, 1],
        "energy": [1.083e4, 1.359e4],
        "latency": [8.998e6, 6.299e7],
        "edp": [1.74e5, 1.39e6],
    },
}

# CS2 — PE Row Sweep (fixed cols = 128)
DEPFIN_CS2 = {
    "FSRCNN": {
        "rows": [8, 16, 32],
        "total_pes": [1024, 2048, 4096],
        "energy": [8.67e3, 8.35e3, 8.31e3],
        "latency": [1.148e7, 6.17e6, 5.909e7],
        "edp": [1.16e5, 5.98e4, 5.74e4],
    },
    "MC-CNN": {
        "rows": [8, 16, 32],
        "total_pes": [1024, 2048, 4096],
        "energy": [1.170e4, 1.132e4, 1.111e4],
        "latency": [2.37e7, 1.20e7, 1.19e7],
        "edp": [3.28e5, 1.61e5, 1.58e5],
    },
    "VGG16": {
        "rows": [16, 32, 64, 128],
        "total_pes": [2048, 4096, 8192, 16384],
        "energy": [7.61e4, 7.59e4, 7.57e4, 7.57e4],
        "latency": [2.43e7, 2.43e7, 2.43e7, 2.43e7],
        "edp": [3.41e6, 3.40e6, 3.39e6, 3.39e6],
    },
    "ResNet18": {
        "rows": [16, 32, 64, 128],
        "total_pes": [2048, 4096, 8192, 16384],
        "energy": [1.083e4, 1.081e4, 1.079e4, 1.079e4],
        "latency": [7.812e6, 7.807e6, 7.807e6, 7.807e6],
        "edp": [1.514e5, 1.511e5, 1.510e5, 1.510e5],
    },
}

# CS3 — PE Col Sweep (fixed rows = 16)
DEPFIN_CS3 = {
    "FSRCNN": {
        "cols": [64, 128, 256],
        "total_pes": [1024, 2048, 4096],
        "energy": [8.355e4, 8.352e4, 8.352e4],
        "latency": [1.110e7, 6.120e6, 3.349e6],
        "edp": [1.08e5, 5.98e4, 3.27e4],
    },
    "MC-CNN": {
        "cols": [64, 128, 256],
        "total_pes": [1024, 2048, 4096],
        "energy": [1.13e4, 1.13e4, 1.13e4],
        "latency": [1.52e7, 1.20e7, 4.15e6],
        "edp": [2.05e5, 1.61e5, 5.57e4],
    },
    "VGG16": {
        "cols": [128, 256, 512, 1024],
        "total_pes": [2048, 4096, 8192, 16384],
        "energy": [7.61e4, 7.61e4, 7.61e4, 7.61e4],
        "latency": [2.43e7, 2.38e7, 2.38e7, 2.38e7],
        "edp": [3.41e6, 3.33e6, 3.33e6, 3.33e6],
    },
    "ResNet18": {
        "cols": [128, 256, 512, 1024],
        "total_pes": [2048, 4096, 8192, 16384],
        "energy": [1.08e4, 1.08e4, 1.08e4, 1.08e4],
        "latency": [7.81e6, 7.81e6, 7.81e6, 7.81e6],
        "edp": [1.51e5, 1.51e5, 1.51e5, 1.51e5],
    },
}

# CS4 — PE Aspect Ratio Sweep (fixed 2048 total PEs)
DEPFIN_CS4 = {
    "FSRCNN": {
        "configs": ["2x1024", "4x512", "8x256", "16x128",
                     "32x64", "64x32", "128x16", "256x8", "512x4"],
        "energy": [1.04e4, 9.19e3, 8.66e3, 8.35e3,
                   8.32e3, 8.32e3, 8.36e3, 8.45e3, 8.63e3],
        "latency": [8.89e6, 5.68e6, 5.98e6, 6.12e6,
                    1.07e7, 2.11e7, 4.22e7, 8.43e7, 1.69e8],
        "edp": [1.05e5, 6.02e4, 6.03e4, 5.98e4,
                1.04e5, 2.05e5, 4.12e5, 8.32e5, 1.69e6],
        "best": "16x128",
    },
    "MC-CNN": {
        "configs": ["2x1024", "4x512", "8x256", "16x128", "32x64",
                     "64x32", "128x16", "256x8", "512x4", "1024x2"],
        "energy": [1.39e4, 1.24e4, 1.17e4, 1.13e4, 1.11e4,
                   1.12e4, 1.13e4, 1.14e4, 1.18e4, 1.22e4],
        "latency": [1.08e7, 8.10e6, 8.07e6, 1.20e7, 1.52e7,
                    3.02e7, 9.06e7, 1.36e8, 2.72e8, 4.08e8],
        "edp": [1.73e5, 1.18e5, 1.11e5, 1.61e5, 2.01e5,
                4.01e5, 1.22e6, 1.84e6, 3.78e6, 5.82e6],
        "best": "8x256",
    },
    "VGG16": {
        "configs": ["2x1024", "4x512", "8x256", "16x128", "32x64",
                     "64x32", "128x16", "256x8", "512x4", "1024x2"],
        "energy": [7.93e4, 7.74e4, 7.65e4, 7.61e4, 7.60e4,
                   7.66e4, 7.85e4, 8.25e4, 9.15e4, 1.06e5],
        "latency": [1.90e8, 9.51e7, 4.76e7, 2.43e7, 2.69e7,
                    3.68e7, 6.59e7, 1.27e8, 2.61e8, 4.80e8],
        "edp": [2.72e7, 1.34e7, 6.68e6, 3.41e6, 3.77e6,
                5.17e6, 9.38e6, 1.85e7, 4.06e7, 8.15e7],
        "best": "16x128",
    },
    "ResNet18": {
        "configs": ["2x1024", "4x512", "8x256", "16x128", "32x64",
                     "64x32", "128x16", "256x8", "512x4", "1024x2"],
        "energy": [1.12e4, 1.10e4, 1.09e4, 1.08e4, 1.08e4,
                   1.09e4, 1.10e4, 1.15e4, 1.39e4, 1.52e4],
        "latency": [6.25e7, 3.12e7, 1.56e7, 7.81e6, 7.87e6,
                    8.78e6, 1.18e7, 1.98e7, 6.15e7, 8.32e7],
        "edp": [1.23e6, 6.10e5, 3.04e5, 1.51e5, 1.52e5,
                1.70e5, 2.31e5, 3.96e5, 1.38e6, 1.98e6],
        "best": "16x128",
    },
}

# ====================================================================
#  DATA  –  Eyeriss Case Studies
# ====================================================================

# CS1 — WReg Sensitivity Sweep (Summary A: minimum PEs)
EYERISS_CS1 = {
    "FSRCNN": {
        "wreg": [200, 300, 400],
        "min_pe_config": ["128x4", "128x4", "128x4"],
        "min_pe_total": [512, 512, 512],
        "energy": [5.052e4, 6.116e4, 7.181e4],
        "latency": [2.825e7, 2.825e7, 2.825e7],
        "edp": [1.46e6, 1.76e6, 2.06e6],
        "n_feasible": [12, 12, 12],
    },
    "MC-CNN": {
        "wreg": [100, 200, 300, 600],
        "min_pe_config": ["128x16", "64x16", "64x16", "64x16"],
        "min_pe_total": [2048, 1024, 1024, 1024],
        "energy": [4.905e4, 6.099e4, 7.525e4, 1.180e5],
        "latency": [1.223e7, 2.344e7, 2.344e7, 2.344e7],
        "edp": [6.05e5, 1.44e6, 1.77e6, 2.78e6],
        "n_feasible": [6, 7, 7, 7],
    },
    "VGG16": {
        "wreg": [1200, 2400, 4800],
        "min_pe_config": ["256x64", "128x64", "128x32"],
        "min_pe_total": [16384, 8192, 4096],
        "energy": [3.469e5, 5.416e5, 9.328e5],
        "latency": [5.455e6, 5.681e6, 6.715e6],
        "edp": [1.94e6, 3.12e6, 6.29e6],
        "n_feasible": [2, 6, 8],
    },
    "ResNet18": {
        "wreg": [902, 1800, 3600],
        "min_pe_config": ["256x64", "128x64", "128x32"],
        "min_pe_total": [16384, 8192, 4096],
        "energy": [4.709e4, 6.850e4, 1.119e5],
        "latency": [3.042e6, 3.042e6, 3.059e6],
        "edp": [1.48e5, 2.13e5, 3.46e5],
        "n_feasible": [2, 6, 8],
    },
}

# CS1 — WReg Sensitivity Sweep — Summary B (minimum EDP)
EYERISS_CS1_B = {
    "VGG16": {
        "wreg": [1200, 2400, 4800],
        "minedp_pe_config": ["256x64", "128x128", "128x128"],
        "minedp_pe_total": [16384, 16384, 16384],
        "energy": [3.469e5, 5.561e5, 9.587e5],
        "latency": [5.455e6, 5.455e6, 5.455e6],
        "edp": [1.94e6, 3.10e6, 5.30e6],
    },
    "ResNet18": {
        "wreg": [902, 1800, 3600],
        "minedp_pe_config": ["256x64", "128x64", "128x64"],
        "minedp_pe_total": [16384, 8192, 8192],
        "energy": [4.709e4, 6.850e4, 1.140e5],
        "latency": [3.042e6, 3.042e6, 3.042e6],
        "edp": [1.48e5, 2.13e5, 3.53e5],
    },
}

# CS2 — IntReg Sensitivity Sweep (Summary A: minimum PEs)
EYERISS_CS2 = {
    "FSRCNN": {
        "intreg": [200, 300, 400],
        "energy": [1.025e5, 1.191e5, 1.357e5],
        "latency": [2.760e7, 2.760e7, 2.760e7],
        "edp": [2.84e6, 3.30e6, 3.75e6],
        "note": "Non-binding: all identical PE config 128x4",
    },
    "MC-CNN": {
        "intreg": [15, 30, 100],
        "energy": [1.212e5, 1.180e5, 1.253e5],
        "latency": [1.223e7, 2.344e7, 2.344e7],
        "edp": [1.49e6, 2.78e6, 2.78e6],
        "note": "IntReg=15 needs 4096 PEs; saturates at >=30",
    },
    "VGG16": {
        "intreg": [50, 100, 230, 500],
        "min_pe_config": ["512x32", "256x32", "256x16", "256x16"],
        "min_pe_total": [16384, 8192, 4096, 4096],
        "energy": [9.102e5, 9.082e5, 9.420e5, 1.036e6],
        "latency": [5.455e6, 5.681e6, 6.715e6, 6.715e6],
        "edp": [5.04e6, 5.21e6, 6.36e6, 6.99e6],
        "note": "Saturates at IntReg>=230",
    },
    "ResNet18": {
        "intreg": [50, 200, 400, 1000],
        "min_pe_config": ["128x128", "256x32", "256x16", "256x16"],
        "min_pe_total": [16384, 8192, 4096, 4096],
        "energy": [1.139e5, 1.168e5, 1.273e5, 1.589e5],
        "latency": [3.042e6, 3.059e6, 3.059e6, 3.059e6],
        "edp": [3.55e5, 3.61e5, 3.93e5, 4.90e5],
        "note": "Saturates at IntReg>=200",
    },
}

# CS3 — OutReg Sensitivity Sweep (Summary A: minimum PEs)
EYERISS_CS3 = {
    "FSRCNN": {
        "outreg": [200, 300, 400],
        "energy": [9.320e4, 1.023e5, 1.115e5],
        "latency": [1.944e7, 1.944e7, 1.944e7],
        "edp": [1.82e6, 2.00e6, 2.18e6],
        "note": "Scales linearly with OutReg size",
    },
    "MC-CNN": {
        "outreg": [8, 16, 32, 64],
        "energy": [1.180e5, 1.180e5, 1.180e5, 1.180e5],
        "latency": [2.344e7, 2.344e7, 2.344e7, 2.344e7],
        "edp": [2.78e6, 2.78e6, 2.78e6, 2.78e6],
        "note": "Never binding for min PEs",
    },
    "VGG16": {
        "outreg": [4, 8, 16, 32, 64],
        "min_pe_config": ["128x128", "256x32", "128x32", "128x32", "128x32"],
        "min_pe_total": [16384, 8192, 4096, 4096, 4096],
        "energy": [9.922e5, 9.777e5, 9.663e5, 9.663e5, 9.663e5],
        "latency": [5.455e6, 5.681e6, 6.715e6, 6.715e6, 6.715e6],
        "edp": [5.49e6, 5.60e6, 6.52e6, 6.52e6, 6.52e6],
        "note": "Saturates at OutReg>=16",
    },
    "ResNet18": {
        "outreg": [8, 16, 32, 64],
        "min_pe_config": ["256x64", "256x32", "256x16", "256x16"],
        "min_pe_total": [16384, 8192, 4096, 4096],
        "energy": [1.242e5, 1.220e5, 1.220e5, 1.220e5],
        "latency": [3.042e6, 3.059e6, 3.059e6, 3.059e6],
        "edp": [3.84e5, 3.77e5, 3.77e5, 3.77e5],
        "note": "Saturates at OutReg>=16",
    },
}

# CS5 — Eyeriss PE Aspect Ratio Sweep (fixed total PEs)
# FSRCNN/MC-CNN: InReg=234, WReg=500, IntReg=200, OutReg=64
# VGG16: InReg=400, WReg=1200, IntReg=350, OutReg=64
# ResNet18: InReg=400, WReg=902, IntReg=350, OutReg=64
EYERISS_CS5 = {
    "FSRCNN": {
        "total_pes": 2048,
        "configs": ["8x256", "16x128", "32x64", "64x32",
                     "128x16", "256x8", "512x4", "1024x2", "2048x1"],
        "energy": [1.184e5, 1.214e5, 1.225e5, 1.253e5,
                   1.253e5, 1.253e5, 1.253e5, 1.253e5, 1.253e5],
        "latency": [1.889e7, 1.889e7, 1.918e7, 1.918e7,
                    1.918e7, 1.918e7, 1.918e7, 1.918e7, 1.918e7],
        "edp": [2.39e6, 2.44e6, 2.50e6, 2.56e6,
                2.56e6, 2.52e6, 2.52e6, 2.52e6, 2.52e6],
        "best": "8x256",
    },
    "MC-CNN": {
        "total_pes": 2048,
        "configs": ["8x256", "16x128", "32x64", "64x32",
                     "128x16", "256x8", "512x4", "1024x2", "2048x1"],
        "energy": [1.588e5, 1.596e5, 1.613e5, 1.646e5,
                   1.597e5, 1.597e5, 1.597e5, 1.597e5, 1.597e5],
        "latency": [2.344e7, 1.223e7, 1.821e7, 1.261e7,
                    1.261e7, 1.261e7, 1.261e7, 1.261e7, 1.261e7],
        "edp": [3.99e6, 2.09e6, 3.15e6, 2.22e6,
                2.09e6, 2.09e6, 2.09e6, 2.09e6, 2.09e6],
        "best": "16x128",
    },
    "VGG16": {
        "total_pes": 16384,
        "configs": ["256x64", "512x32", "1024x16", "2048x8",
                     "4096x4", "8192x2", "16384x1"],
        "energy": [3.654e5, 3.674e5, 3.746e5, 3.804e5,
                   3.804e5, 3.804e5, 3.804e5],
        "latency": [5.455e6, 5.455e6, 5.455e6, 5.455e6,
                    5.455e6, 5.455e6, 5.455e6],
        "edp": [2.04e6, 2.04e6, 2.07e6, 2.10e6,
                2.10e6, 2.10e6, 2.10e6],
        "best": "512x32",
    },
    "ResNet18": {
        "total_pes": 16384,
        "configs": ["256x64", "512x32", "1024x16", "2048x8",
                     "4096x4", "8192x2", "16384x1"],
        "energy": [4.993e4, 5.030e4, 5.105e4, 5.134e4,
                   5.134e4, 5.134e4, 5.134e4],
        "latency": [3.042e6, 3.042e6, 3.042e6, 3.042e6,
                    3.042e6, 3.042e6, 3.042e6],
        "edp": [1.57e5, 1.57e5, 1.59e5, 1.60e5,
                1.60e5, 1.60e5, 1.60e5],
        "best": "256x64",
    },
}

# ====================================================================
#  DATA  –  Fusion Comparisons (Auto-Sized, Full-Fusion Architecture)
#  All three modes run on the SAME architecture sized for full fusion.
# ====================================================================

FUSION_AUTO = {
    # ── Source: experiment_run_fcomp.py  (Scenario A: full-sized arch) ────
    "DepFiN": {
        "FSRCNN": {
            "pe": "16x128", "fmem": "576KB", "wmem": "19KB", "tile": 120,
            "full":    {"energy": 8.352e3, "latency": 6.156e6, "edp": 5.14e4,
                        "dram_rd": 1_573_992, "dram_wr": 8_294_400},
            "single":  {"energy": 6.536e3, "latency": 1.175e7, "edp": 7.68e4,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
            "partial": {"energy": 9.150e3, "latency": 6.313e6, "edp": 5.81e4,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
        },
        "MC-CNN": {
            "pe": "8x256", "fmem": "522KB", "wmem": "32KB", "tile": 207,
            "full":    {"energy": 1.177e4, "latency": 7.975e6, "edp": 9.39e4,
                        "dram_rd": 494_928, "dram_wr": 14_943_744},
            "single":  {"energy": 4.224e3, "latency": 1.381e7, "edp": 5.83e4,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
            "partial": {"energy": 1.273e4, "latency": 7.975e6, "edp": 1.02e5,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
        },
        "VGG16": {
            "pe": "16x128", "fmem": "568KB", "wmem": "14366KB", "tile": 14,
            "full":    {"energy": 7.609e4, "latency": 2.728e7, "edp": 2.08e6,
                        "dram_rd": 14_860_992, "dram_wr": 100_352},
            "single":  {"energy": 3.413e3, "latency": 2.252e7, "edp": 8.37e4,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
            "partial": {"energy": 5.652e4, "latency": 2.432e7, "edp": 1.37e6,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
        },
        "ResNet18": {
            "pe": "16x128", "fmem": "266KB", "wmem": "10738KB", "tile": 7,
            "full":    {"energy": 1.083e4, "latency": 8.998e6, "edp": 9.75e4,
                        "dram_rd": 11_032_512, "dram_wr": 25_088},
            "single":  {"energy": 9.499e2, "latency": 6.905e6, "edp": 6.56e3,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
            "partial": {"energy": 1.088e4, "latency": 7.814e6, "edp": 8.50e4,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
        },
    },
    "Eyeriss": {
        "FSRCNN": {
            "pe": "128x16", "gb": "128KB", "wreg": 384, "intreg": 36,
            "full":    {"energy": 7.150e4, "latency": 1.918e7, "edp": 1.37e6,
                        "dram_rd": 1_573_992, "dram_wr": 8_294_400},
            "single":  {"energy": 1.117e4, "latency": 3.525e7, "edp": 3.94e5,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
            "partial": {"energy": 7.203e4, "latency": 1.918e7, "edp": 1.38e6,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
        },
        "MC-CNN": {
            "pe": "256x8", "gb": "128KB", "wreg": 384, "intreg": 65,
            "full":    {"energy": 9.217e4, "latency": 1.261e7, "edp": 1.16e6,
                        "dram_rd": 494_928, "dram_wr": 14_943_744},
            "single":  {"energy": 8.784e3, "latency": 1.495e7, "edp": 1.31e5,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
            "partial": {"energy": 9.313e4, "latency": 1.261e7, "edp": 1.17e6,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
        },
        "VGG16": {
            "pe": "512x32", "gb": "128KB", "wreg": 1200, "intreg": 350,
            "full":    {"energy": 3.478e5, "latency": 5.455e6, "edp": 1.90e6,
                        "dram_rd": 14_860_992, "dram_wr": 100_352},
            "single":  {"energy": 9.425e3, "latency": 6.922e6, "edp": 6.52e4,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
            "partial": {"energy": 2.325e5, "latency": 5.852e6, "edp": 1.36e6,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
        },
        "ResNet18": {
            "pe": "512x32", "gb": "128KB", "wreg": 902, "intreg": 300,
            "full":    {"energy": 4.729e4, "latency": 3.042e6, "edp": 1.44e5,
                        "dram_rd": 11_032_512, "dram_wr": 25_088},
            "single":  {"energy": 1.872e3, "latency": 3.301e6, "edp": 6.18e3,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
            "partial": {"energy": 4.444e4, "latency": 3.139e6, "edp": 1.40e5,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
        },
    },
}


# ====================================================================
#  DATA  –  Fusion Comparisons (Auto-Sized, DRAM Energy = 160 pJ/byte)
#  Same architecture configs as FUSION_AUTO but with DRAM access energy
#  hardcoded to 160 pJ/byte (DepFiN/JEDEC value).
# ====================================================================

FUSION_AUTO_DRAM_160 = {
    # ── Source: experiment_runner.py --compare-fusion --dram-energy 160 ───
    "DepFiN": {
        "FSRCNN": {
            "pe": "16x128", "fmem": "576KB", "wmem": "19KB", "tile": 120,
            "full":    {"energy": 7.046e3, "latency": 6.156e6, "edp": 5.62e4,
                        "dram_rd": 1_573_992, "dram_wr": 8_294_400},
            "single":  {"energy": 3.084e4, "latency": 1.175e7, "edp": 3.62e5,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
            "partial": {"energy": 1.034e4, "latency": 6.351e6, "edp": 6.56e4,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
        },
        "MC-CNN": {
            "pe": "8x256", "fmem": "522KB", "wmem": "32KB", "tile": 207,
            "full":    {"energy": 1.043e4, "latency": 7.975e6, "edp": 1.08e5,
                        "dram_rd": 494_928, "dram_wr": 14_943_744},
            "single":  {"energy": 1.806e4, "latency": 1.381e7, "edp": 2.49e5,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
            "partial": {"energy": 1.490e4, "latency": 7.975e6, "edp": 1.19e5,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
        },
        "VGG16": {
            "pe": "16x128", "fmem": "568KB", "wmem": "14366KB", "tile": 14,
            "full":    {"energy": 1.040e5, "latency": 2.728e7, "edp": 5.41e6,
                        "dram_rd": 14_860_992, "dram_wr": 100_352},
            "single":  {"energy": 7.715e3, "latency": 2.452e7, "edp": 1.89e5,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
            "partial": {"energy": 5.073e4, "latency": 2.432e7, "edp": 1.23e6,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
        },
        "ResNet18": {
            "pe": "16x128", "fmem": "266KB", "wmem": "10738KB", "tile": 7,
            "full":    {"energy": 1.573e4, "latency": 8.998e6, "edp": 2.55e5,
                        "dram_rd": 11_032_512, "dram_wr": 25_088},
            "single":  {"energy": 2.715e3, "latency": 6.905e6, "edp": 1.87e4,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
            "partial": {"energy": 1.176e4, "latency": 7.813e6, "edp": 9.19e4,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
        },
    },
    "Eyeriss": {
        "FSRCNN": {
            "pe": "128x16", "gb": "128KB", "wreg": 384, "inreg": 34, "intreg": 32,
            "outreg": 64, "tile": 120,
            "full":    {"energy": 3.294e4, "latency": 1.918e7, "edp": 6.52e5,
                        "dram_rd": 1_573_992, "dram_wr": 8_294_400},
            "single":  {"energy": 3.245e4, "latency": 3.525e7, "edp": 1.14e6,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
            "partial": {"energy": 2.189e4, "latency": 1.918e7, "edp": 4.20e5,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
        },
        "MC-CNN": {
            "pe": "256x8", "gb": "128KB", "wreg": 384, "inreg": 34, "intreg": 32,
            "outreg": 64, "tile": 69,
            "full":    {"energy": 4.281e4, "latency": 1.261e7, "edp": 5.49e5,
                        "dram_rd": 494_928, "dram_wr": 14_943_744},
            "single":  {"energy": 1.925e4, "latency": 1.495e7, "edp": 2.88e5,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
            "partial": {"energy": 2.762e4, "latency": 1.261e7, "edp": 3.48e5,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
        },
        "VGG16": {
            "pe": "512x32", "gb": "128KB", "wreg": 1200, "inreg": 400, "intreg": 350,
            "outreg": 64, "tile": 1,
            "full":    {"energy": 1.632e5, "latency": 5.455e6, "edp": 9.04e5,
                        "dram_rd": 14_860_992, "dram_wr": 100_352},
            "single":  {"energy": 9.594e3, "latency": 6.922e6, "edp": 6.64e4,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
            "partial": {"energy": 6.391e4, "latency": 5.852e6, "edp": 3.74e5,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
        },
        "ResNet18": {
            "pe": "512x32", "gb": "128KB", "wreg": 902, "inreg": 400, "intreg": 350,
            "outreg": 64, "tile": 1,
            "full":    {"energy": 2.366e4, "latency": 3.042e6, "edp": 7.37e4,
                        "dram_rd": 11_032_512, "dram_wr": 25_088},
            "single":  {"energy": 2.977e3, "latency": 3.301e6, "edp": 9.83e3,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
            "partial": {"energy": 1.641e4, "latency": 3.139e6, "edp": 5.15e4,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
        },
    },
}


# ====================================================================
#  DATA  –  Fusion Comparisons (Auto-Sized, DRAM Energy = 200 pJ/byte)
#  Same architecture configs as FUSION_AUTO but with DRAM access energy
#  hardcoded to 200 pJ/byte instead of the Accelergy-derived 32 pJ/byte.
# ====================================================================

FUSION_AUTO_DRAM_200 = {
    # ── Source: experiment_runner.py --compare-fusion --dram-energy 200 ───
    "DepFiN": {
        "FSRCNN": {
            "pe": "16x128", "fmem": "576KB", "wmem": "19KB", "tile": 120,
            "full":    {"energy": 1.001e4, "latency": 6.156e6, "edp": 7.03e4,
                        "dram_rd": 1_573_992, "dram_wr": 8_294_400},
            "single":  {"energy": 3.815e4, "latency": 1.175e7, "edp": 4.48e5,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
            "partial": {"energy": 1.452e4, "latency": 6.351e6, "edp": 9.22e4,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
        },
        "MC-CNN": {
            "pe": "8x256", "fmem": "522KB", "wmem": "32KB", "tile": 207,
            "full":    {"energy": 1.426e4, "latency": 7.975e6, "edp": 1.31e5,
                        "dram_rd": 494_928, "dram_wr": 14_943_744},
            "single":  {"energy": 2.188e4, "latency": 1.381e7, "edp": 3.02e5,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
            "partial": {"energy": 2.002e4, "latency": 7.975e6, "edp": 1.60e5,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
        },
        "VGG16": {
            "pe": "16x128", "fmem": "568KB", "wmem": "14366KB", "tile": 14,
            "full":    {"energy": 7.861e4, "latency": 2.728e7, "edp": 3.89e6,
                        "dram_rd": 14_860_992, "dram_wr": 100_352},
            "single":  {"energy": 8.659e3, "latency": 2.452e7, "edp": 2.12e5,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
            "partial": {"energy": 4.160e4, "latency": 2.432e7, "edp": 1.01e6,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
        },
        "ResNet18": {
            "pe": "16x128", "fmem": "266KB", "wmem": "10738KB", "tile": 7,
            "full":    {"energy": 1.269e4, "latency": 8.998e6, "edp": 1.91e5,
                        "dram_rd": 11_032_512, "dram_wr": 25_088},
            "single":  {"energy": 3.194e3, "latency": 6.905e6, "edp": 2.21e4,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
            "partial": {"energy": 1.013e4, "latency": 7.813e6, "edp": 7.92e4,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
        },
    },
    "Eyeriss": {
        "FSRCNN": {
            "pe": "128x16", "gb": "128KB", "wreg": 384, "inreg": 34,  "intreg":32,
            "outreg": 64, "tile": 120,
            "full":    {"energy": 1.146e5, "latency": 1.918e7, "edp": 2.35e6,
                        "dram_rd": 1_573_992, "dram_wr": 8_294_400},
            "single":  {"energy": 4.913e4, "latency": 3.525e7, "edp": 1.73e6,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
            "partial": {"energy": 8.101e4, "latency": 1.918e7, "edp": 1.55e6,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
        },
        "MC-CNN": {
            "pe": "256x8", "gb": "128KB", "wreg": 384, "inreg": 34, "intreg": 32,
            "outreg": 64, "tile": 69,
            "full":    {"energy": 9.502e4, "latency": 1.261e7, "edp": 1.22e6,
                        "dram_rd": 494_928, "dram_wr": 14_943_744},
            "single":  {"energy": 2.641e4, "latency": 1.495e7, "edp": 3.95e5,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
            "partial": {"energy": 5.524e4, "latency": 1.261e7, "edp": 6.97e5,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
        },
        "VGG16": {
            "pe": "512x32", "gb": "128KB", "wreg": 1200, "inreg": 400, "intreg": 350,
            "outreg": 64, "tile": 1,
            "full":    {"energy": 3.699e5, "latency": 5.455e6, "edp": 2.05e6,
                        "dram_rd": 14_860_992, "dram_wr": 100_352},
            "single":  {"energy": 1.568e4, "latency": 6.922e6, "edp": 1.09e5,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
            "partial": {"energy": 1.415e5, "latency": 5.852e6, "edp": 8.28e5,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
        },
        "ResNet18": {
            "pe": "512x32", "gb": "128KB", "wreg": 902, "inreg": 400, "intreg": 350,
            "outreg": 64, "tile": 1,

            "full":    {"energy": 4.915e4, "latency": 3.042e6, "edp": 1.53e5,
                        "dram_rd": 11_032_512, "dram_wr": 25_088},
            "single":  {"energy": 4.350e3, "latency": 3.301e6, "edp": 1.44e4,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
            "partial": {"energy": 3.344e4, "latency": 3.139e6, "edp": 1.05e5,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
        },
    },
}


# ====================================================================
#  DATA  –  Fusion Comparisons (Auto-Sized, Partial-Fusion Architecture)
#  Partial and Single run on a SMALLER architecture sized for partial
#  fusion only (less memory).  No full-fusion column.
# ====================================================================

FUSION_PARTIAL_SIZED = {
    # ── Source: experiment_run_fcomp.py  (Scenario B: partial-sized arch) ─
    "DepFiN": {
        "FSRCNN": {
            "pe": "16x128", "fmem": "248KB", "wmem": "9KB", "tile": 120,
            "partial": {"energy": 8.519e3, "latency": 6.351e6, "edp": 5.41e4,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
            "single":  {"energy": 6.390e3, "latency": 1.175e7, "edp": 7.51e4,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
        },
        "MC-CNN": {
            "pe": "8x256", "fmem": "396KB", "wmem": "22KB", "tile": 207,
            "partial": {"energy": 1.238e4, "latency": 8.074e6, "edp": 1.00e5,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
            "single":  {"energy": 4.123e3, "latency": 1.381e7, "edp": 5.69e4,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
        },
        "VGG16": {
            "pe": "16x128", "fmem": "112KB", "wmem": "4610KB", "tile": 14,
            "partial": {"energy": 3.705e4, "latency": 2.432e7, "edp": 9.01e5,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
            "single":  {"energy": 2.460e3, "latency": 2.452e7, "edp": 6.03e4,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
        },
        "ResNet18": {
            "pe": "16x128", "fmem": "48KB", "wmem": "4608KB", "tile": 7,
            "partial": {"energy": 8.004e3, "latency": 7.813e6, "edp": 6.25e4,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
            "single":  {"energy": 7.921e2, "latency": 6.905e6, "edp": 5.47e3,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
        },
    },
    "Eyeriss": {
        "FSRCNN": {
            "pe": "128x16", "gb": "128KB", "wreg": 384,
            "partial": {"energy": 7.203e4, "latency": 1.918e7, "edp": 1.38e6,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
            "single":  {"energy": 1.117e4, "latency": 3.525e7, "edp": 3.94e5,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
        },
        "MC-CNN": {
            "pe": "256x8", "gb": "128KB", "wreg": 384,
            "partial": {"energy": 9.313e4, "latency": 1.261e7, "edp": 1.17e6,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
            "single":  {"energy": 8.784e3, "latency": 1.495e7, "edp": 1.31e5,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
        },
        "VGG16": {
            "pe": "512x32", "gb": "128KB", "wreg": 576,
            "partial": {"energy": 1.562e5, "latency": 5.852e6, "edp": 9.14e5,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
            "single":  {"energy": 9.415e3, "latency": 6.922e6, "edp": 6.52e4,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
        },
        "ResNet18": {
            "pe": "512x32", "gb": "128KB", "wreg": 384,
            "partial": {"energy": 3.134e4, "latency": 3.139e6, "edp": 9.84e4,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
            "single":  {"energy": 1.871e3, "latency": 3.301e6, "edp": 6.18e3,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
        },
    },
}


# ====================================================================
#  DATA  –  Fusion Comparisons (Partial-Sized Arch, DRAM Energy = 160 pJ/byte)
#  Same architecture configs as FUSION_PARTIAL_SIZED but with DRAM access
#  energy set to 160 pJ/byte (DepFiN/JEDEC value) instead of Accelergy's.
# ====================================================================

FUSION_PARTIAL_DRAM_160 = {
    # ── Source: experiment_runner.py --compare-partial-vs-single --dram-energy 160
    "DepFiN": {
        "FSRCNN": {
            "pe": "16x128", "fmem": "248KB", "wmem": "9KB", "tile": 120,
            "partial": {"energy": 1.010e4, "latency": 6.351e6, "edp": 6.42e4,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
            "single":  {"energy": 3.063e4, "latency": 1.175e7, "edp": 3.60e5,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
        },
        "MC-CNN": {
            "pe": "8x256", "fmem": "396KB", "wmem": "22KB", "tile": 207,
            "partial": {"energy": 1.486e4, "latency": 7.975e6, "edp": 1.19e5,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
            "single":  {"energy": 1.792e4, "latency": 1.381e7, "edp": 2.48e5,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
        },
        "VGG16": {
            "pe": "16x128", "fmem": "112KB", "wmem": "4610KB", "tile": 14,
            "partial": {"energy": 5.034e4, "latency": 2.432e7, "edp": 1.22e6,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
            "single":  {"energy": 7.829e3, "latency": 2.452e7, "edp": 1.92e5,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
        },
        "ResNet18": {
            "pe": "16x128", "fmem": "48KB", "wmem": "4608KB", "tile": 7,
            "partial": {"energy": 1.021e4, "latency": 7.425e6, "edp": 7.58e4,
                        "dram_rd": 11_448_768, "dram_wr": 551_936},
            "single":  {"energy": 2.827e3, "latency": 6.905e6, "edp": 1.95e4,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
        },
    },
    "Eyeriss": {
        "FSRCNN": {
            "pe": "128x16", "gb": "128KB", "wreg": 384, "inreg": 34, "intreg": 32,
            "outreg": 64, "tile": 120,
            "partial": {"energy": 3.681e4, "latency": 1.918e7, "edp": 7.06e5,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
            "single":  {"energy": 3.247e4, "latency": 3.525e7, "edp": 1.14e6,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
        },
        "MC-CNN": {
            "pe": "256x8", "gb": "128KB", "wreg": 384, "inreg": 34, "intreg": 32,
            "outreg": 64, "tile": 69,
            "partial": {"energy": 4.759e4, "latency": 1.261e7, "edp": 6.00e5,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
            "single":  {"energy": 1.926e4, "latency": 1.495e7, "edp": 2.88e5,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
        },
        "VGG16": {
            "pe": "512x32", "gb": "128KB", "wreg": 576, "inreg": 400, "intreg": 350,
            "outreg": 64, "tile": 1,
            "partial": {"energy": 7.418e4, "latency": 5.852e6, "edp": 4.34e5,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
            "single":  {"energy": 9.597e3, "latency": 6.922e6, "edp": 6.64e4,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
        },
        "ResNet18": {
            "pe": "512x32", "gb": "128KB", "wreg": 384, "inreg": 400, "intreg": 350,
            "outreg": 64, "tile": 1,
            "partial": {"energy": 1.641e4, "latency": 3.139e6, "edp": 5.15e4,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
            "single":  {"energy": 2.977e3, "latency": 3.301e6, "edp": 9.83e3,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
        },
    },
}


# ====================================================================
#  DATA  –  Fusion Comparisons (Partial-Sized Arch, DRAM Energy = 200 pJ/byte)
#  Same architecture configs as FUSION_PARTIAL_SIZED but with DRAM access
#  energy set to 200 pJ/byte instead of the Accelergy-derived 32 pJ/byte.
# ====================================================================

FUSION_PARTIAL_DRAM_200 = {
    # ── Source: experiment_runner.py --compare-partial-vs-single --dram-energy 200
    "DepFiN": {
        "FSRCNN": {
            "pe": "16x128", "fmem": "248KB", "wmem": "9KB", "tile": 120,
            "partial": {"energy": 1.436e4, "latency": 6.351e6, "edp": 9.12e4,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
            "single":  {"energy": 3.801e4, "latency": 1.175e7, "edp": 4.47e5,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
        },
        "MC-CNN": {
            "pe": "8x256", "fmem": "396KB", "wmem": "22KB", "tile": 207,
            "partial": {"energy": 2.000e4, "latency": 7.975e6, "edp": 1.59e5,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
            "single":  {"energy": 2.178e4, "latency": 1.381e7, "edp": 3.01e5,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
        },
        "VGG16": {
            "pe": "16x128", "fmem": "112KB", "wmem": "4610KB", "tile": 14,
            "partial": {"energy": 4.133e4, "latency": 2.432e7, "edp": 1.01e6,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
            "single":  {"energy": 8.736e3, "latency": 2.452e7, "edp": 2.14e5,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
        },
        "ResNet18": {
            "pe": "16x128", "fmem": "48KB", "wmem": "4608KB", "tile": 7,
            "partial": {"energy": 8.883e3, "latency": 7.425e6, "edp": 6.60e4,
                        "dram_rd": 11_448_768, "dram_wr": 551_936},
            "single":  {"energy": 3.270e3, "latency": 6.905e6, "edp": 2.26e4,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
        },
    },
    "Eyeriss": {
        "FSRCNN": {
            "pe": "128x16", "gb": "128KB", "wreg": 384, "inreg": 34,  "intreg":32,
            "outreg": 64, "tile": 120,
            "partial": {"energy": 7.787e4, "latency": 1.918e7, "edp": 1.49e6,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
            "single":  {"energy": 4.279e4, "latency": 3.525e7, "edp": 1.51e6,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
        },
        "MC-CNN": {
            "pe": "256x8", "gb": "128KB", "wreg": 384,"inreg": 34,  "intreg":32,
            "outreg": 64, "tile": 69,
            "partial": {"energy": 1.007e5, "latency": 1.261e7, "edp": 1.27e6,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
            "single":  {"energy": 2.644e4, "latency": 1.495e7, "edp": 3.95e5,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
        },
        "VGG16": {
            "pe": "512x32", "gb": "128KB", "wreg": 576, "inreg": 400, "intreg": 350,
            "outreg": 64, "tile": 1,
            "partial": {"energy": 1.650e5, "latency": 5.852e6, "edp": 9.65e5,
                        "dram_rd": 17_670_848, "dram_wr": 7_426_048},
            "single":  {"energy": 1.569e4, "latency": 6.922e6, "edp": 1.09e5,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
        },
        "ResNet18": {
            "pe": "512x32", "gb": "128KB", "wreg": 384, "inreg": 400, "intreg": 350,
            "outreg": 64, "tile": 1,
            "partial": {"energy": 3.535e4, "latency": 3.139e6, "edp": 1.11e5,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
            "single":  {"energy": 4.350e3, "latency": 3.301e6, "edp": 1.44e4,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
        },
    },
}


# ====================================================================
#  DATA  –  Fusion Comparisons (Auto-Sized, Normalized to Full=1.0)
# ====================================================================
# For each (arch, workload), every metric is divided by the Full value.
FUSION_AUTO_DRAM_160_NORM = {}
for _arch, _wls in FUSION_AUTO_DRAM_160.items():
    FUSION_AUTO_DRAM_160_NORM[_arch] = {}
    for _wl, _modes in _wls.items():
        FUSION_AUTO_DRAM_160_NORM[_arch][_wl] = {}
        for _mode in ("full", "partial", "single"):
            FUSION_AUTO_DRAM_160_NORM[_arch][_wl][_mode] = {
                _metric: _modes[_mode][_metric] / _modes["full"][_metric]
                for _metric in ("energy", "latency", "edp")
            }

# Normalized partial-sized data (Partial = 1.0)
FUSION_PARTIAL_SIZED_DRAM_160_NORM = {}
for _arch, _wls in FUSION_PARTIAL_SIZED.items():
    FUSION_PARTIAL_SIZED_DRAM_160_NORM[_arch] = {}
    for _wl, _modes in _wls.items():
        FUSION_PARTIAL_SIZED_DRAM_160_NORM[_arch][_wl] = {}
        for _mode in ("partial", "single"):
            FUSION_PARTIAL_SIZED_DRAM_160_NORM[_arch][_wl][_mode] = {
                _metric: _modes[_mode][_metric] / _modes["partial"][_metric]
                for _metric in ("energy", "latency", "edp")
            }

# ====================================================================
#  DATA  –  Fusion Comparisons (Fixed-Config, Eyeriss)
#  Scenario A = full-sized architecture (wreg sized for Full fusion)
# ====================================================================

# ResNet18 — Full vs Single energy ratio across PE configs  (Scenario A)
FUSION_FIXED_RESNET18 = {
    "configs": ["512x32", "256x32", "256x16"],
    "total_pes": [16384, 8192, 4096],
    "wreg": [902, 1800, 3600],          # full-sized wreg
    "full_energy": [4.729e4, 6.767e4, 1.121e5],
    "single_energy": [1.872e3, 1.417e3, 1.131e3],
    "partial_energy": [4.444e4, 6.559e4, 1.104e5],
    "full_latency": [3.042e6, 3.042e6, 3.059e6],
    "single_latency": [3.301e6, 3.301e6, 3.301e6],
    "partial_latency": [3.139e6, 3.139e6, 3.147e6],
    "energy_ratio_fs": [19.15, 47.75, 99.10],
    "energy_ratio_fp": [1.06, 1.03, 1.02],  # full/partial ~1
    "latency_ratio": [0.976, 0.976, 0.92],
}

# ResNet18 — Partial-sized architecture  (Scenario B)
# Only configs with distinct ScB data: 512x32, 256x32, 256x16
FUSION_FIXED_RESNET18_PARTIAL = {
    "configs": ["512x32", "256x32", "256x16"],
    "total_pes": [16384, 8192, 4096],
    "wreg": [384, 770, 1600],                       # partial-sized wreg
    "partial_energy": [3.134e4, 3.953e4, 5.978e4],
    "single_energy":  [1.871e3, 1.413e3, 1.118e3],
    "partial_latency": [3.139e6, 3.139e6, 3.147e6],
    "single_latency":  [3.301e6, 3.301e6, 3.301e6],
}

# VGG16 — Full vs Single across PE configs  (Scenario A)
FUSION_FIXED_VGG16 = {
    "configs": ["512x32", "256x32", "256x16"],
    "total_pes": [16384, 8192, 4096],
    "wreg": [1200, 2400, 4800],                     # full-sized wreg
    "full_energy": [3.478e5, 5.368e5, 9.337e5],
    "single_energy": [9.425e3, 6.897e3, 5.190e3],
    "partial_energy": [2.325e5, 3.760e5, 6.678e5],
    "full_latency": [5.455e6, 5.681e6, 6.715e6],
    "single_latency": [6.922e6, 6.922e6, 7.694e6],
    "partial_latency": [5.852e6, 5.952e6, 6.897e6],
    "energy_ratio_fs": [36.90, 77.83, 179.90],
    "energy_ratio_fp": [1.50, 1.43, 1.40],
    "latency_ratio": [0.79, 0.82, 0.87],
}

# VGG16 — Partial-sized architecture  (Scenario B)
FUSION_FIXED_VGG16_PARTIAL = {
    "configs": ["512x32", "256x32", "256x16"],
    "total_pes": [16384, 8192, 4096],
    "wreg": [576, 1200, 2200],                       # partial-sized wreg
    "partial_energy": [1.562e5, 2.292e5, 3.499e5],
    "single_energy":  [9.415e3, 6.862e3, 5.044e3],
    "partial_latency": [5.852e6, 5.952e6, 6.897e6],
    "single_latency":  [6.922e6, 6.922e6, 7.694e6],
}

# MC-CNN — fixed config  (wreg=384 for both scenarios → single scenario)
FUSION_FIXED_MCCNN = {
    "config": "256x8", "total_pes": 2048, "wreg": 384,
    "full":    {"energy": 9.217e4, "latency": 1.261e7},
    "single":  {"energy": 8.784e3, "latency": 1.495e7},
    "partial": {"energy": 9.313e4, "latency": 1.261e7},
    "energy_ratio_fs": 10.49,
    "energy_ratio_fp": 0.99,
}

# FSRCNN — fixed config  (wreg=384 for both scenarios → single scenario)
FUSION_FIXED_FSRCNN = {
    "config": "128x16", "total_pes": 2048, "wreg": 384,
    "full":    {"energy": 7.150e4, "latency": 1.918e7},
    "single":  {"energy": 1.117e4, "latency": 3.525e7},
    "partial": {"energy": 7.203e4, "latency": 1.918e7},
    "energy_ratio_fs": 6.40,
    "energy_ratio_fp": 0.993,
}

# DepFiN fixed-config fusion  (Scenario A: full-sized fmem/wmem)
FUSION_FIXED_DEPFIN = {
    "ResNet18": {
        "config": "16x128", "total_pes": 2048,
        "fmem": "266KB", "wmem": "10738KB",
        "full":    {"energy": 1.083e4, "latency": 8.998e6},
        "single":  {"energy": 9.499e2, "latency": 6.905e6},
        "partial": {"energy": 1.088e4, "latency": 7.813e6},
        "energy_ratio_fs": 11.40,
        "energy_ratio_fp": 0.996,
    },
    "VGG16": {
        "config": "16x128", "total_pes": 2048,
        "fmem": "528KB", "wmem": "14366KB",
        "full":    {"energy": 7.609e4, "latency": 2.728e7},
        "single":  {"energy": 3.413e3, "latency": 2.252e7},
        "partial": {"energy": 5.652e4, "latency": 2.432e7},
        "energy_ratio_fs": 19.37,
        "energy_ratio_fp": 1.34,
    },
    "MC-CNN": {
        "config": "8x256", "total_pes": 2048,
        "fmem": "522KB", "wmem": "32KB",
        "full":    {"energy": 1.177e4, "latency": 7.975e6},
        "single":  {"energy": 4.224e3, "latency": 1.381e7},
        "partial": {"energy": 1.273e4, "latency": 7.975e6},
        "energy_ratio_fs": 2.79,
        "energy_ratio_fp": 0.92,
    },
    "FSRCNN": {
        "config": "16x128", "total_pes": 2048,
        "fmem": "576KB", "wmem": "19KB",
        "full":    {"energy": 8.352e3, "latency": 6.156e6},
        "single":  {"energy": 6.536e3, "latency": 1.175e7},
        "partial": {"energy": 9.150e3, "latency": 6.313e6},
        "energy_ratio_fs": 1.29,
        "energy_ratio_fp": 0.91,
    },
}

# DepFiN fixed-config fusion  (Scenario B: partial-sized fmem/wmem)
FUSION_FIXED_DEPFIN_PARTIAL = {
    "ResNet18": {
        "config": "16x128", "total_pes": 2048,
        "fmem": "48KB", "wmem": "4609KB",
        "partial": {"energy": 8.004e3, "latency": 7.813e6},
        "single":  {"energy": 7.921e2, "latency": 6.905e6},
    },
    "VGG16": {
        "config": "16x128", "total_pes": 2048,
        "fmem": "112KB", "wmem": "4610KB",
        "partial": {"energy": 3.705e4, "latency": 2.432e7},
        "single":  {"energy": 2.460e3, "latency": 2.452e7},
    },
    "MC-CNN": {
        "config": "8x256", "total_pes": 2048,
        "fmem": "396KB", "wmem": "22KB",
        "partial": {"energy": 1.238e4, "latency": 8.074e6},
        "single":  {"energy": 4.123e3, "latency": 1.381e7},
    },
    "FSRCNN": {
        "config": "16x128", "total_pes": 2048,
        "fmem": "248KB", "wmem": "9KB",
        "partial": {"energy": 8.519e3, "latency": 6.313e6},
        "single":  {"energy": 6.390e3, "latency": 1.175e7},
    },
}


# ====================================================================
#  PLOT FUNCTIONS
# ====================================================================

# ────────────────────────────────────────────────────────────────────
#  Figure 1: DepFiN CS4 — PE Aspect Ratio (2×2 grid)
# ────────────────────────────────────────────────────────────────────
def plot_depfin_cs4():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DepFiN — PE Aspect Ratio Sweep  (2048 PEs)", fontsize=16, y=0.98)

    for ax, wl in zip(axes.flat, WORKLOADS):
        d = DEPFIN_CS4[wl]
        x = np.arange(len(d["configs"]))

        # EDP on primary y-axis
        ln1 = ax.semilogy(x, d["edp"], "o-", color=COLORS[wl], label="EDP")
        ax.set_ylabel("EDP  (J·cc)")
        ax.set_xticks(x)
        ax.set_xticklabels(d["configs"], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{wl}  (best: {d['best']})")

        # Energy on secondary axis
        ax2 = ax.twinx()
        ln2 = ax2.plot(x, d["energy"], "s--", color="gray", alpha=0.6, label="Energy")
        ax2.set_ylabel("Energy (μJ)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        # Mark best EDP
        best_idx = d["configs"].index(d["best"])
        ax.plot(best_idx, d["edp"][best_idx], "*", markersize=16,
                color="gold", zorder=5, markeredgecolor="black")

        edp_handle, = ax.plot([], [], "o", color=COLORS[wl], label="EDP")
        lns = [edp_handle] + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc="upper left", fontsize=8, numpoints=1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "depfin_cs4_aspect_ratio")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 2: DepFiN CS1 — Tile Size Sweep
# ────────────────────────────────────────────────────────────────────
def plot_depfin_cs1():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("DepFiN — Tile Size Sensitivity  (16×128 PEs) DepFiN 16×128 PEs, \n FMEM BW scales as: BW_scaled = BW_base × (tile_size / 128)", fontsize=16, y=0.98)

    for ax, wl in zip(axes.flat, WORKLOADS):
        d = DEPFIN_CS1[wl]
        ax.semilogy(d["tiles"], d["edp"], "o-", color=COLORS[wl])
        ax.set_xlabel("Tile size")
        ax.set_ylabel("EDP  (J·cc)")
        ax.set_title(wl)
        ax.set_xticks(d["tiles"])
        # mark best
        best_i = int(np.argmin(d["edp"]))
        ax.plot(d["tiles"][best_i], d["edp"][best_i], "*",
                markersize=14, color="gold", markeredgecolor="black", zorder=5)
        edp_handle, = ax.plot([], [], "o", color=COLORS[wl], label="EDP")
        ax.legend(handles=[edp_handle], fontsize=8, numpoints=1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "depfin_cs1_tile_sweep")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 3: DepFiN CS2+CS3 — Row & Col Sweeps (one figure per workload)
# ────────────────────────────────────────────────────────────────────
def plot_depfin_cs2_cs3():
    """Generate one 1×2 figure per workload: row sweep (left) + col sweep (right)."""
    figs = []
    for wl in WORKLOADS:
        fig, (ax_row, ax_col) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"DepFiN — {wl}: PE Row Sweep vs PE Col Sweep",
                     fontsize=15, y=1.02)

        # CS2: Row sweep (cols fixed = 128)
        d2 = DEPFIN_CS2[wl]
        ax_row.plot(d2["rows"], d2["edp"], "o-", color=COLORS[wl], linewidth=2)
        ax_row.set_xlabel("PE rows  (cols = 128)")
        ax_row.set_ylabel("EDP  (J·cc)")
        ax_row.set_title("Row Sweep (CS2)")
        ax_row.set_xticks(d2["rows"])
        _sci_fmt(ax_row)
        # mark best
        best_i = int(np.argmin(d2["edp"]))
        ax_row.plot(d2["rows"][best_i], d2["edp"][best_i], "*",
                    markersize=14, color="gold", markeredgecolor="black", zorder=5)
        # annotate total PEs
        for i, (r, edp_val) in enumerate(zip(d2["rows"], d2["edp"])):
            ax_row.annotate(f"{d2['total_pes'][i]:,} PEs",
                            (r, edp_val), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec=COLORS[wl], alpha=0.8, lw=0.5))

        # CS3: Col sweep (rows fixed = 16)
        d3 = DEPFIN_CS3[wl]
        ax_col.plot(d3["cols"], d3["edp"], "s-", color=COLORS[wl], linewidth=2)
        ax_col.set_xlabel("PE cols  (rows = 16)")
        ax_col.set_ylabel("EDP  (J·cc)")
        ax_col.set_title("Col Sweep (CS3)")
        ax_col.set_xticks(d3["cols"])
        _sci_fmt(ax_col)
        # mark best
        best_i = int(np.argmin(d3["edp"]))
        ax_col.plot(d3["cols"][best_i], d3["edp"][best_i], "*",
                    markersize=14, color="gold", markeredgecolor="black", zorder=5)
        # annotate total PEs
        for i, (c, edp_val) in enumerate(zip(d3["cols"], d3["edp"])):
            ax_col.annotate(f"{d3['total_pes'][i]:,} PEs",
                            (c, edp_val), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec=COLORS[wl], alpha=0.8, lw=0.5))

        # Add 8% bottom + top margin so data lines don't touch the axes
        for ax in (ax_row, ax_col):
            lo, hi = ax.get_ylim()
            margin = 0.08 * (hi - lo)
            ax.set_ylim(lo - margin, hi + margin)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, f"depfin_cs2cs3_{wl.lower().replace('-', '')}")
        figs.append(fig)
    return figs


# ────────────────────────────────────────────────────────────────────
#  Figure 4: Eyeriss CS1 — WReg Sweep (Summary A + Summary B)
#  Only VGG16 and ResNet18 (FSRCNN/MC-CNN are trivial).
# ────────────────────────────────────────────────────────────────────
def plot_eyeriss_cs1():
    cs1_wls = ["VGG16", "ResNet18"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Eyeriss CS1 — WReg Sensitivity: Summary A (min PEs) vs Summary B (min EDP)",
                 fontsize=14, y=1.02)

    for ax, wl in zip(axes, cs1_wls):
        da = EYERISS_CS1[wl]
        db = EYERISS_CS1_B[wl]

        # Filter None entries (infeasible WReg sizes)
        mask_a = [i for i, e in enumerate(da["edp"]) if e is not None]
        wreg_a = [da["wreg"][i] for i in mask_a]
        edp_a  = [da["edp"][i] for i in mask_a]
        cfg_a  = [da["min_pe_config"][i] for i in mask_a]

        mask_b = [i for i, e in enumerate(db["edp"]) if e is not None]
        wreg_b = [db["wreg"][i] for i in mask_b]
        edp_b  = [db["edp"][i] for i in mask_b]
        cfg_b  = [db.get("minedp_pe_config", db.get("min_pe_config", [""] * len(db["wreg"])))[i] for i in mask_b]

        # Plot both summaries
        ln1 = ax.plot(wreg_a, edp_a, "o-", color=COLORS[wl],
                      label="Summary A (min PEs)", linewidth=2, zorder=2)
        ln2 = ax.plot(wreg_b, edp_b, "s--", color=COLORS[wl], alpha=0.6,
                      label="Summary B (min EDP)", linewidth=2, zorder=2)

        bbox_a = dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS[wl],
                      alpha=0.85, lw=0.6)
        bbox_b = dict(boxstyle="round,pad=0.2", fc="white", ec="gray",
                      alpha=0.85, lw=0.6)

        # Annotate PE config for Summary A (above points)
        for i, (w, edp_val, cfg) in enumerate(zip(wreg_a, edp_a, cfg_a)):
            ax.annotate(f"{cfg}\n{da['min_pe_total'][mask_a[i]]:,} PEs",
                       (w, edp_val),
                       textcoords="offset points", xytext=(0, 18),
                       fontsize=8, ha="center", color=COLORS[wl],
                       bbox=bbox_a, zorder=5,
                       arrowprops=dict(arrowstyle="->", color=COLORS[wl], lw=0.8))

        # Annotate PE config for Summary B (below points, only where different from A)
        for i, (w, edp_val, cfg) in enumerate(zip(wreg_b, edp_b, cfg_b)):
            matching_a = [j for j, wa in enumerate(wreg_a) if wa == w]
            if matching_a and cfg_a[matching_a[0]] == cfg:
                continue  # same config, skip annotation
            ax.annotate(f"{cfg}\n{db['minedp_pe_total'][mask_b[i]]:,} PEs",
                       (w, edp_val),
                       textcoords="offset points", xytext=(0, -28),
                       fontsize=8, ha="center", color="gray",
                       bbox=bbox_b, zorder=5,
                       arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

        ax.set_xlabel("WReg size (entries)")
        ax.set_ylabel("EDP  (J·cc)")
        ax.set_title(wl, fontsize=13)
        ax.set_xticks(sorted(set(wreg_a + wreg_b)))
        ax.legend(fontsize=9, loc="lower right")
        _sci_fmt(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "eyeriss_cs1_wreg_sweep")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 4b: Eyeriss CS1 — Inverted: PE budget → min WReg
# ────────────────────────────────────────────────────────────────────
def plot_eyeriss_cs1_inverted():
    """Designer-oriented view: given a PE budget, what is the minimum WReg?"""
    cs1_wls = ["VGG16", "ResNet18"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Eyeriss CS1 — Minimum WReg for a Given PE Budget",
                 fontsize=14, y=1.02)

    for ax, wl in zip(axes, cs1_wls):
        d = EYERISS_CS1[wl]
        # Invert: x = total PEs (reversed so large budget is on the left),
        #         y = min WReg
        pes  = list(reversed(d["min_pe_total"]))
        wreg = list(reversed(d["wreg"]))
        edp  = list(reversed(d["edp"]))
        cfgs = list(reversed(d["min_pe_config"]))

        # Primary axis: min WReg
        ax.plot(pes, wreg, "o-", color=COLORS[wl], linewidth=2)
        ax.set_xlabel("PE budget (total PEs)")
        ax.set_ylabel("Minimum WReg (entries)")
        ax.set_title(wl, fontsize=13)
        ax.set_xticks(pes)
        ax.set_xticklabels([f"{p:,}" for p in pes], fontsize=9)
        ax.invert_xaxis()  # large budget on the left

        # Annotate configs
        for p, w, cfg in zip(pes, wreg, cfgs):
            ax.annotate(cfg, (p, w), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec=COLORS[wl], alpha=0.85, lw=0.6))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "eyeriss_cs1_wreg_inverted")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 5: Eyeriss CS2+CS3 — IntReg & OutReg Binding Hierarchy
# ────────────────────────────────────────────────────────────────────
def plot_eyeriss_cs2_cs3():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Eyeriss — IntReg Sweep (top) and OutReg Sweep (bottom) — min PEs config",
                 fontsize=15, y=0.98)

    for ax, wl in zip(axes[0], WORKLOADS):
        d = EYERISS_CS2[wl]
        mask = [i for i, e in enumerate(d["energy"]) if e is not None]
        ir = [d["intreg"][i] for i in mask]
        en = [d["energy"][i] for i in mask]
        ax.plot(ir, en, "o-", color=COLORS[wl])
        ax.set_xlabel("IntReg size")
        ax.set_ylabel("Energy (μJ)")
        ax.set_title(f"{wl}\n{d.get('note','')}", fontsize=9)
        _sci_fmt(ax)

    for ax, wl in zip(axes[1], WORKLOADS):
        d = EYERISS_CS3[wl]
        mask = [i for i, e in enumerate(d["energy"]) if e is not None]
        oreg = [d["outreg"][i] for i in mask]
        en = [d["energy"][i] for i in mask]
        ax.plot(oreg, en, "s-", color=COLORS[wl])
        ax.set_xlabel("OutReg size")
        ax.set_ylabel("Energy (μJ)")
        ax.set_title(f"{wl}\n{d.get('note','')}", fontsize=9)
        _sci_fmt(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "eyeriss_cs2cs3_intreg_outreg")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 5b: Eyeriss CS2 — Inverted: PE budget → min IntReg
# ────────────────────────────────────────────────────────────────────
def plot_eyeriss_cs2_inverted():
    """Designer-oriented view: given a PE budget, what is the minimum IntReg?"""
    cs2_wls = ["VGG16", "ResNet18"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Eyeriss CS2 — Minimum IntReg for a Given PE Budget",
                 fontsize=14, y=1.02)

    for ax, wl in zip(axes, cs2_wls):
        d = EYERISS_CS2[wl]
        # Deduplicate saturated points (same min_pe_total)
        seen = set()
        pes, ireg, cfgs = [], [], []
        for pe, ir, cfg in zip(d["min_pe_total"], d["intreg"],
                               d["min_pe_config"]):
            if pe not in seen:
                seen.add(pe)
                pes.append(pe)
                ireg.append(ir)
                cfgs.append(cfg)

        pes  = list(reversed(pes))
        ireg = list(reversed(ireg))
        cfgs = list(reversed(cfgs))

        ax.plot(pes, ireg, "o-", color=COLORS[wl], linewidth=2)
        ax.set_xlabel("PE budget (total PEs)")
        ax.set_ylabel("Minimum IntReg (entries)")
        ax.set_title(wl, fontsize=13)
        ax.set_xticks(pes)
        ax.set_xticklabels([f"{p:,}" for p in pes], fontsize=9)
        ax.invert_xaxis()

        for p, ir_val, cfg in zip(pes, ireg, cfgs):
            ax.annotate(cfg, (p, ir_val), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec=COLORS[wl], alpha=0.85, lw=0.6))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "eyeriss_cs2_intreg_inverted")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 5c: Eyeriss CS3 — Inverted: PE budget → min OutReg
# ────────────────────────────────────────────────────────────────────
def plot_eyeriss_cs3_inverted():
    """Designer-oriented view: given a PE budget, what is the minimum OutReg?"""
    cs3_wls = ["VGG16", "ResNet18"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Eyeriss CS3 — Minimum OutReg for a Given PE Budget",
                 fontsize=14, y=1.02)

    for ax, wl in zip(axes, cs3_wls):
        d = EYERISS_CS3[wl]
        seen = set()
        pes, oreg, cfgs = [], [], []
        for pe, orv, cfg in zip(d["min_pe_total"], d["outreg"],
                                d["min_pe_config"]):
            if pe not in seen:
                seen.add(pe)
                pes.append(pe)
                oreg.append(orv)
                cfgs.append(cfg)

        pes  = list(reversed(pes))
        oreg = list(reversed(oreg))
        cfgs = list(reversed(cfgs))

        ax.plot(pes, oreg, "o-", color=COLORS[wl], linewidth=2)
        ax.set_xlabel("PE budget (total PEs)")
        ax.set_ylabel("Minimum OutReg (entries)")
        ax.set_title(wl, fontsize=13)
        ax.set_xticks(pes)
        ax.set_xticklabels([f"{p:,}" for p in pes], fontsize=9)
        ax.invert_xaxis()

        for p, or_val, cfg in zip(pes, oreg, cfgs):
            ax.annotate(cfg, (p, or_val), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec=COLORS[wl], alpha=0.85, lw=0.6))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "eyeriss_cs3_outreg_inverted")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 6: Eyeriss CS5 — PE Aspect Ratio
# ────────────────────────────────────────────────────────────────────
def plot_eyeriss_cs5():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.subplots_adjust(top=0.90, hspace=0.55, wspace=0.35,
                        bottom=0.12, left=0.05, right=0.97)
    fig.suptitle("Eyeriss — PE Aspect Ratio Sweep  (fixed total PEs)",
                 fontsize=16)

    def _plain_ticks(ax):
        """Use plain float formatting – no offset, no scientific notation."""
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:.3g}"))

    for col, wl in enumerate(WORKLOADS):
        d = EYERISS_CS5[wl]
        x = np.arange(len(d["configs"]))
        best_idx = d["configs"].index(d["best"])
        c = COLORS[wl]

        # ── Top row: Energy ──
        ax_e = axes[0, col]
        ax_e.plot(x, d["energy"], "s-", color=c)
        ax_e.plot(best_idx, d["energy"][best_idx], "*", markersize=14,
                  color="gold", markeredgecolor="black", zorder=5)
        ax_e.set_xticks(x)
        ax_e.set_xticklabels(d["configs"], rotation=45, ha="right", fontsize=7)
        ax_e.set_title(f"{wl}  ({d['total_pes']} PEs)", fontsize=10)
        if col == 0:
            ax_e.set_ylabel("Energy  (μJ)")
        _plain_ticks(ax_e)
        # add 8% top margin so high points aren't flush with the edge
        lo, hi = ax_e.get_ylim()
        ax_e.set_ylim(lo - 0.01 * lo, hi + 0.08 * (hi - lo))

        # ── Bottom row: EDP ──
        ax_d = axes[1, col]
        ax_d.plot(x, d["edp"], "o-", color=c)
        ax_d.plot(best_idx, d["edp"][best_idx], "*", markersize=14,
                  color="gold", markeredgecolor="black", zorder=5)
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(d["configs"], rotation=45, ha="right", fontsize=7)
        ax_d.set_xlabel("PE config  (rows × cols)", fontsize=8)
        if col == 0:
            ax_d.set_ylabel("EDP  (μJ · cc)")
        _plain_ticks(ax_d)
        # add 8% top margin
        lo, hi = ax_d.get_ylim()
        ax_d.set_ylim(lo - 0.01 * lo, hi + 0.08 * (hi - lo))

        # annotate best config
        ax_d.annotate(f"best: {d['best']}",
                      xy=(best_idx, d["edp"][best_idx]),
                      xytext=(0, 18), textcoords="offset points",
                      fontsize=7, ha="center", color=c, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                ec=c, alpha=0.85))

    _save(fig, "eyeriss_cs5_aspect_ratio")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 7: Auto 160 pj/byte-Sized Fusion — Energy Comparison (grouped bars)
# ────────────────────────────────────────────────────────────────────
def plot_fusion_auto_energy():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Full-Sized Arch — Energy  (Full vs Partial vs Single)", fontsize=16, y=1.0)

    for ax, arch in zip(axes, ["DepFiN", "Eyeriss"]):
        data = FUSION_AUTO_DRAM_160[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.25

        full_e = [data[wl]["full"]["energy"] for wl in WORKLOADS]
        part_e = [data[wl]["partial"]["energy"] for wl in WORKLOADS]
        sing_e = [data[wl]["single"]["energy"] for wl in WORKLOADS]

        ax.bar(x - w, full_e, w, label="Full Fusion", color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
        ax.bar(x, part_e, w, label=r"$\Sigma$ Partial Fusion", color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w, sing_e, w, label=r"$\Sigma$ Singles", color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)

        ax.set_yscale("log")
        ax.set_ylabel("Energy (μJ)")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(arch)
        ax.legend(fontsize=9)


    fig.tight_layout()
    _save(fig, "fusion_auto_energy")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 8: Auto-Sized 160 pj/byte Fusion — Latency Comparison
# ────────────────────────────────────────────────────────────────────
def plot_fusion_auto_latency():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Full-Sized Arch — Latency (Full vs Partial vs Single)", fontsize=16, y=1.0)

    for ax, arch in zip(axes, ["DepFiN", "Eyeriss"]):
        data = FUSION_AUTO_DRAM_160[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.25

        full_l = [data[wl]["full"]["latency"] for wl in WORKLOADS]
        part_l = [data[wl]["partial"]["latency"] for wl in WORKLOADS]
        sing_l = [data[wl]["single"]["latency"] for wl in WORKLOADS]

        ax.bar(x - w, full_l, w, label="Full Fusion", color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
        ax.bar(x, part_l, w, label=r"$\Sigma$ Partial Fusion", color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w, sing_l, w, label=r"$\Sigma$ Singles", color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Latency (cc)")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(arch)
        ax.legend(fontsize=9)
        _sci_fmt(ax)

        # Add latency savings annotations
        for i, wl in enumerate(WORKLOADS):
            saving = 1.0 - data[wl]["full"]["latency"] / data[wl]["single"]["latency"]
            if saving > 0:
                ax.annotate(f"-{saving*100:.0f}%", (i + w, sing_l[i]),
                           textcoords="offset points", xytext=(0, 8),
                           fontsize=7, ha="center", color="darkgreen", fontweight="bold")

    fig.tight_layout()
    _save(fig, "fusion_auto_latency")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 9: Auto-Sized 160 pj/byte Fusion — DRAM Traffic
# ────────────────────────────────────────────────────────────────────
def plot_fusion_auto_dram():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Full-Sized Arch — DRAM Traffic Reduction", fontsize=16, y=0.98)

    for col_idx, arch in enumerate(["DepFiN", "Eyeriss"]):
        data = FUSION_AUTO_DRAM_160[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.25

        # DRAM Reads (top row)
        ax = axes[0, col_idx]
        full_rd = [data[wl]["full"]["dram_rd"] for wl in WORKLOADS]
        part_rd = [data[wl]["partial"]["dram_rd"] for wl in WORKLOADS]
        sing_rd = [data[wl]["single"]["dram_rd"] for wl in WORKLOADS]

        ax.bar(x - w, full_rd, w, label="Full", color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
        ax.bar(x, part_rd, w, label="Partial", color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w, sing_rd, w, label="Single", color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_ylabel("DRAM Reads")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(f"{arch} — DRAM Reads")
        ax.legend(fontsize=8)
        ax.set_ylim(top=ax.get_ylim()[1] * 3)

        # DRAM Writes (bottom row)
        ax = axes[1, col_idx]
        full_wr = [data[wl]["full"]["dram_wr"] for wl in WORKLOADS]
        part_wr = [data[wl]["partial"]["dram_wr"] for wl in WORKLOADS]
        sing_wr = [data[wl]["single"]["dram_wr"] for wl in WORKLOADS]

        ax.bar(x - w, full_wr, w, label="Full", color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
        ax.bar(x, part_wr, w, label="Partial", color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w, sing_wr, w, label="Single", color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_ylabel("DRAM Writes")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(f"{arch} — DRAM Writes")
        ax.legend(fontsize=8)
        ax.set_ylim(top=ax.get_ylim()[1] * 3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fusion_auto_dram_traffic")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 9b: Normalized Energy 160 pj/byte (Full Fusion = 1.0)
# ────────────────────────────────────────────────────────────────────
def plot_fusion_auto_energy_norm():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Full-Sized Arch — Normalized Energy  (Full = 1.0)", fontsize=16, y=1.0)

    for ax, arch in zip(axes, ["DepFiN", "Eyeriss"]):
        data = FUSION_AUTO_DRAM_160_NORM[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.25

        full_e = [data[wl]["full"]["energy"] for wl in WORKLOADS]
        part_e = [data[wl]["partial"]["energy"] for wl in WORKLOADS]
        sing_e = [data[wl]["single"]["energy"] for wl in WORKLOADS]

        ax.bar(x - w, full_e, w, label="Full Fusion",
               color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
        ax.bar(x, part_e, w, label=r"$\Sigma$ Partial Fusion",
               color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w, sing_e, w, label=r"$\Sigma$ Singles",
               color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Normalized Energy")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(arch)
        ax.legend(fontsize=9)
        ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", zorder=0)
        ax.set_ylim(top=1.60)

        # Annotate bar values
        for i, (f, p, s) in enumerate(zip(full_e, part_e, sing_e)):
            for val, xpos in [(f, i - w), (p, i), (s, i + w)]:
                ax.annotate(f"{val:.3f}", (xpos, val),
                           textcoords="offset points", xytext=(0, 4),
                           fontsize=7, ha="center", fontweight="bold")

    fig.tight_layout()
    _save(fig, "fusion_auto_energy_norm")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 9c: Normalized Latency (Full Fusion = 1.0)
# ────────────────────────────────────────────────────────────────────
def plot_fusion_auto_latency_norm():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Full-Sized Arch — Normalized Latency  (Full = 1.0)", fontsize=16, y=1.0)

    for ax, arch in zip(axes, ["DepFiN", "Eyeriss"]):
        data = FUSION_AUTO_DRAM_160_NORM[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.25

        full_l = [data[wl]["full"]["latency"] for wl in WORKLOADS]
        part_l = [data[wl]["partial"]["latency"] for wl in WORKLOADS]
        sing_l = [data[wl]["single"]["latency"] for wl in WORKLOADS]

        ax.bar(x - w, full_l, w, label="Full Fusion",
               color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
        ax.bar(x, part_l, w, label=r"$\Sigma$ Partial Fusion",
               color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w, sing_l, w, label=r"$\Sigma$ Singles",
               color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Normalized Latency")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(arch)
        ax.legend(fontsize=9)
        ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", zorder=0)
        ax.set_ylim(top=2.80)

        for i, (f, p, s) in enumerate(zip(full_l, part_l, sing_l)):
            for val, xpos in [(f, i - w), (p, i), (s, i + w)]:
                ax.annotate(f"{val:.3f}", (xpos, val),
                           textcoords="offset points", xytext=(0, 4),
                           fontsize=7, ha="center", fontweight="bold")

    fig.tight_layout()
    _save(fig, "fusion_auto_latency_norm")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 9d: Normalized EDP (Full Fusion = 1.0)
# ────────────────────────────────────────────────────────────────────
def plot_fusion_auto_edp_norm():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Full-Sized Arch — Normalized EDP  (Full = 1.0)", fontsize=16, y=1.0)

    for ax, arch in zip(axes, ["DepFiN", "Eyeriss"]):
        data = FUSION_AUTO_DRAM_160_NORM[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.25

        full_edp = [data[wl]["full"]["edp"] for wl in WORKLOADS]
        part_edp = [data[wl]["partial"]["edp"] for wl in WORKLOADS]
        sing_edp = [data[wl]["single"]["edp"] for wl in WORKLOADS]

        ax.bar(x - w, full_edp, w, label="Full Fusion",
               color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
        ax.bar(x, part_edp, w, label=r"$\Sigma$ Partial Fusion",
               color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w, sing_edp, w, label=r"$\Sigma$ Singles",
               color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Normalized EDP")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(arch)
        ax.legend(fontsize=9)
        ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", zorder=0)
        ax.set_ylim(top=1.70)

        for i, (f, p, s) in enumerate(zip(full_edp, part_edp, sing_edp)):
            for val, xpos in [(f, i - w), (p, i), (s, i + w)]:
                ax.annotate(f"{val:.3f}", (xpos, val),
                           textcoords="offset points", xytext=(0, 4),
                           fontsize=7, ha="center", fontweight="bold")

    fig.tight_layout()
    _save(fig, "fusion_auto_edp_norm")
    return fig


# ── helper: build architecture config label for a data entry ────────
def _arch_cfg_label(arch, entry):
    """Return a short string describing the memory config."""
    pe = entry.get("pe", "")
    if arch == "Eyeriss":
        return f"PE {pe}, WReg={entry.get('wreg','?')}"
    else:  # DepFiN
        return f"PE {pe}, FMEM={entry.get('fmem','?')}, WMEM={entry.get('wmem','?')}"


# ────────────────────────────────────────────────────────────────────
#  Figure 9e: Partial-Sized Arch — Energy (Partial vs Single)
# ────────────────────────────────────────────────────────────────────
def plot_partial_sized_energy():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Partial-Sized Architecture — Energy  (Partial vs Single)",
                 fontsize=16, y=1.0)

    for ax, arch in zip(axes, ["DepFiN", "Eyeriss"]):
        data_ps = FUSION_PARTIAL_SIZED_DRAM_160_NORM[arch]
        data_fs = FUSION_AUTO_DRAM_160[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.3

        part_e = [data_ps[wl]["partial"]["energy"] for wl in WORKLOADS]
        sing_e = [data_ps[wl]["single"]["energy"] for wl in WORKLOADS]

        ax.bar(x - w/2, part_e, w, label="Partial Fusion",
               color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, sing_e, w, label=r"$\Sigma$ Singles",
               color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)

        ax.set_yscale("log")
        ax.set_ylabel("Energy (μJ)")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(arch)
        ax.legend(fontsize=9)

        # Annotate per-workload arch config below x-axis
        for i, wl in enumerate(WORKLOADS):
            cfg = _arch_cfg_label(arch, data_ps[wl])
            ax.annotate(cfg, (i, 0), xycoords=("data", "axes fraction"),
                       xytext=(0, -28), textcoords="offset points",
                       fontsize=5.5, ha="center", color="gray", style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    _save(fig, "fusion_partial_sized_energy")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 9f: Partial-Sized Arch — Latency (Partial vs Single)
# ────────────────────────────────────────────────────────────────────
def plot_partial_sized_latency():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Partial-Sized Architecture — Latency  (Partial vs Single)",
                 fontsize=16, y=1.0)

    for ax, arch in zip(axes, ["DepFiN", "Eyeriss"]):
        data_ps = FUSION_PARTIAL_SIZED_DRAM_160_NORM[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.3

        part_l = [data_ps[wl]["partial"]["latency"] for wl in WORKLOADS]
        sing_l = [data_ps[wl]["single"]["latency"] for wl in WORKLOADS]

        ax.bar(x - w/2, part_l, w, label="Partial Fusion",
               color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, sing_l, w, label=r"$\Sigma$ Singles",
               color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Latency (cc)")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(arch)
        ax.legend(fontsize=9)
        _sci_fmt(ax)

        for i, wl in enumerate(WORKLOADS):
            cfg = _arch_cfg_label(arch, data_ps[wl])
            ax.annotate(cfg, (i, 0), xycoords=("data", "axes fraction"),
                       xytext=(0, -28), textcoords="offset points",
                       fontsize=5.5, ha="center", color="gray", style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    _save(fig, "fusion_partial_sized_latency")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 9g: Partial-Sized Arch — EDP (Partial vs Single)
# ────────────────────────────────────────────────────────────────────
def plot_partial_sized_edp():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Partial-Sized Architecture — EDP  (Partial vs Single)",
                 fontsize=16, y=1.0)

    for ax, arch in zip(axes, ["DepFiN", "Eyeriss"]):
        data_ps = FUSION_PARTIAL_SIZED_DRAM_160_NORM[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.3

        part_edp = [data_ps[wl]["partial"]["edp"] for wl in WORKLOADS]
        sing_edp = [data_ps[wl]["single"]["edp"] for wl in WORKLOADS]

        ax.bar(x - w/2, part_edp, w, label="Partial Fusion",
               color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, sing_edp, w, label=r"$\Sigma$ Singles",
               color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)

        ax.set_yscale("log")
        ax.set_ylabel("EDP (J·cc)")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(arch)
        ax.legend(fontsize=9)

        for i, wl in enumerate(WORKLOADS):
            cfg = _arch_cfg_label(arch, data_ps[wl])
            ax.annotate(cfg, (i, 0), xycoords=("data", "axes fraction"),
                       xytext=(0, -28), textcoords="offset points",
                       fontsize=5.5, ha="center", color="gray", style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    _save(fig, "fusion_partial_sized_edp")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 9h: Partial-Sized Arch — Normalised (Partial = 1.0)
# ────────────────────────────────────────────────────────────────────
def plot_partial_sized_norm():
    metrics = [("energy", "Normalised Energy"), ("latency", "Normalised Latency"),
               ("edp", "Normalised EDP")]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(14, 12))
    fig.suptitle("Partial-Sized Arch — Normalised  (Partial = 1.0)", fontsize=16, y=0.98)

    for col, arch in enumerate(["DepFiN", "Eyeriss"]):
        data = FUSION_PARTIAL_SIZED_DRAM_160_NORM[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.3
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            part_v = [data[wl]["partial"][metric] for wl in WORKLOADS]
            sing_v = [data[wl]["single"][metric] for wl in WORKLOADS]

            ax.bar(x - w/2, part_v, w, label="Partial Fusion",
                   color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
            ax.bar(x + w/2, sing_v, w, label=r"$\Sigma$ Singles",
                   color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)

            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(WORKLOADS)
            if row == 0:
                ax.set_title(arch)
            ax.legend(fontsize=8)
            ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", zorder=0)
            max_val = max(max(part_v), max(sing_v))
            ax.set_ylim(top=max(1.25, max_val * 1.12))

            for i, (p, s) in enumerate(zip(part_v, sing_v)):
                for val, xpos in [(p, i - w/2), (s, i + w/2)]:
                    ax.annotate(f"{val:.3f}", (xpos, val),
                               textcoords="offset points", xytext=(0, 4),
                               fontsize=7, ha="center", fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fusion_partial_sized_norm")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 10: Fixed-Config Fusion — Energy Ratio Scaling (ResNet18)
# ────────────────────────────────────────────────────────────────────
def plot_fusion_fixed_resnet18():
    d = FUSION_FIXED_RESNET18
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("ResNet18 Eyeriss — Fixed-Config Scaling (Full-Sized Arch)", fontsize=15, y=1.0)

    x = np.arange(len(d["configs"]))
    labels = [f"{c}\n({d['total_pes'][i]:,} PEs)\nWReg={d['wreg'][i]}"
              for i, c in enumerate(d["configs"])]

    # Left: Full vs Single energy ratio
    ax1.bar(x, d["energy_ratio_fs"], color="#d62728", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Full / Single Energy Ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_title("Full vs Single — Energy Ratio")
    for i, v in enumerate(d["energy_ratio_fs"]):
        ax1.text(i, v + 2, f"{v:.1f}×", ha="center", fontsize=9, fontweight="bold")

    # Right: Absolute energy comparison
    w = 0.3
    ax2.bar(x - w/2, d["full_energy"], w, label="Full Fusion",
            color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
    ax2.bar(x + w/2, d["single_energy"], w, label=r"$\Sigma$ Singles",
            color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("Energy (μJ)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_title("Absolute Energy")
    ax2.legend()

    fig.tight_layout()
    _save(fig, "fusion_fixed_resnet18_scaling")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 11: Fixed-Config Fusion — Energy Ratio Scaling (VGG16)
# ────────────────────────────────────────────────────────────────────
def plot_fusion_fixed_vgg16():
    d = FUSION_FIXED_VGG16
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("VGG16 Eyeriss — Fixed-Config Scaling (Full-Sized Arch)", fontsize=15, y=1.0)

    x = np.arange(len(d["configs"]))
    labels = [f"{c}\n({d['total_pes'][i]:,} PEs)\nWReg={d['wreg'][i]}"
              for i, c in enumerate(d["configs"])]

    ax1.bar(x, d["energy_ratio_fs"], color="#d62728", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Full / Single Energy Ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("Full vs Single — Energy Ratio")
    for i, v in enumerate(d["energy_ratio_fs"]):
        ax1.text(i, v + 2, f"{v:.1f}×", ha="center", fontsize=9, fontweight="bold")

    w = 0.25
    ax2.bar(x - w, d["full_energy"], w, label="Full",
            color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
    ax2.bar(x, d["partial_energy"], w, label="Partial",
            color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
    ax2.bar(x + w, d["single_energy"], w, label="Single",
            color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("Energy (μJ)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("Full / Partial / Single Energy")
    ax2.legend()

    fig.tight_layout()
    _save(fig, "fusion_fixed_vgg16_scaling")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 12: Fixed-Config — All Workloads at 2048 PEs (both archs)
# ────────────────────────────────────────────────────────────────────
def plot_fusion_fixed_overview():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Fixed-Config (2048 PEs, Full-Sized Arch) — Full vs Partial vs Single",
                 fontsize=15, y=1.0)

    # Eyeriss panel
    ax = axes[0]
    eye_data = {
        "FSRCNN": FUSION_FIXED_FSRCNN,
        "MC-CNN": FUSION_FIXED_MCCNN,
    }
    x = np.arange(len(eye_data))
    w = 0.25
    wls = list(eye_data.keys())

    full_e = [eye_data[wl]["full"]["energy"] for wl in wls]
    part_e = [eye_data[wl]["partial"]["energy"] for wl in wls]
    sing_e = [eye_data[wl]["single"]["energy"] for wl in wls]

    ax.bar(x - w, full_e, w, label="Full", color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
    ax.bar(x, part_e, w, label="Partial", color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
    ax.bar(x + w, sing_e, w, label="Single", color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Energy (μJ)")
    ax.set_xticks(x)
    ax.set_xticklabels(wls)
    ax.set_title("Eyeriss (2048 PEs, WReg=384)")
    ax.legend()
    for i, wl in enumerate(wls):
        r = eye_data[wl]["energy_ratio_fs"]
        ax.annotate(f"{r:.1f}×", (i - w, full_e[i]),
                   textcoords="offset points", xytext=(0, 8),
                   fontsize=8, ha="center", color="darkred", fontweight="bold")

    # DepFiN panel
    ax = axes[1]
    dep_data = FUSION_FIXED_DEPFIN
    x = np.arange(len(WORKLOADS))
    full_e = [dep_data[wl]["full"]["energy"] for wl in WORKLOADS]
    part_e = [dep_data[wl]["partial"]["energy"] for wl in WORKLOADS]
    sing_e = [dep_data[wl]["single"]["energy"] for wl in WORKLOADS]

    ax.bar(x - w, full_e, w, label="Full", color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
    ax.bar(x, part_e, w, label="Partial", color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
    ax.bar(x + w, sing_e, w, label="Single", color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Energy (μJ)")
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS)
    ax.set_title("DepFiN (2048 PEs)")
    ax.legend()
    for i, wl in enumerate(WORKLOADS):
        r = dep_data[wl]["energy_ratio_fs"]
        ax.annotate(f"{r:.1f}×", (i - w, full_e[i]),
                   textcoords="offset points", xytext=(0, 8),
                   fontsize=8, ha="center", color="darkred", fontweight="bold")

    fig.tight_layout()
    _save(fig, "fusion_fixed_overview_2048pe")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Fixed-Config Scaling — Partial-Sized Architecture (Scenario B)
# ────────────────────────────────────────────────────────────────────
def plot_fusion_fixed_resnet18_partial():
    """ResNet18 Eyeriss: Partial vs Single energy scaling on partial-sized architecture."""
    d_full = FUSION_FIXED_RESNET18            # ScA (for reference)
    d_part = FUSION_FIXED_RESNET18_PARTIAL    # ScB

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("ResNet18 Eyeriss — Partial-Sized Arch  (Partial vs Single)",
                 fontsize=15, y=1.0)

    n = len(d_part["configs"])
    x = np.arange(n)
    labels = [f"{d_part['configs'][i]}\n({d_part['total_pes'][i]:,} PEs)"
              f"\nWReg={d_part['wreg'][i]}"
              for i in range(n)]

    # Left: energy comparison
    w = 0.35
    ax1.bar(x - w/2, d_part["partial_energy"], w, label="Partial",
            color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
    ax1.bar(x + w/2, d_part["single_energy"], w, label=r"$\Sigma$ Singles",
            color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Energy (μJ)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_title("Absolute Energy")
    ax1.legend()

    # Right: latency comparison
    ax2.bar(x - w/2, d_part["partial_latency"], w, label="Partial",
            color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
    ax2.bar(x + w/2, d_part["single_latency"], w, label=r"$\Sigma$ Singles",
            color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Latency (cycles)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_title("Absolute Latency")
    ax2.legend()

    fig.tight_layout()
    _save(fig, "fusion_fixed_resnet18_partial_scaling")
    return fig


def plot_fusion_fixed_vgg16_partial():
    """VGG16 Eyeriss: Partial vs Single energy scaling on partial-sized architecture."""
    d_part = FUSION_FIXED_VGG16_PARTIAL

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("VGG16 Eyeriss — Partial-Sized Arch  (Partial vs Single)",
                 fontsize=15, y=1.0)

    n = len(d_part["configs"])
    x = np.arange(n)
    labels = [f"{d_part['configs'][i]}\n({d_part['total_pes'][i]:,} PEs)"
              f"\nWReg={d_part['wreg'][i]}"
              for i in range(n)]

    w = 0.35
    ax1.bar(x - w/2, d_part["partial_energy"], w, label="Partial",
            color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
    ax1.bar(x + w/2, d_part["single_energy"], w, label=r"$\Sigma$ Singles",
            color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Energy (μJ)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_title("Absolute Energy")
    ax1.legend()

    ax2.bar(x - w/2, d_part["partial_latency"], w, label="Partial",
            color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
    ax2.bar(x + w/2, d_part["single_latency"], w, label=r"$\Sigma$ Singles",
            color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Latency (cycles)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("Absolute Latency")
    ax2.legend()

    fig.tight_layout()
    _save(fig, "fusion_fixed_vgg16_partial_scaling")
    return fig


def plot_fusion_fixed_overview_partial():
    """2048-PE overview: Partial vs Single on partial-sized arch (both architectures)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Fixed-Config (2048 PEs, Partial-Sized Arch) — Partial vs Single",
                 fontsize=15, y=0.98)

    # --- Eyeriss panel ---
    # MC-CNN & FSRCNN have same wreg for both scenarios → use ScA data directly
    ax = axes[0]
    eye_wls = ["FSRCNN", "MC-CNN"]
    eye_dicts = {"FSRCNN": FUSION_FIXED_FSRCNN, "MC-CNN": FUSION_FIXED_MCCNN}
    x = np.arange(len(eye_wls))
    w = 0.35

    part_e = [eye_dicts[wl]["partial"]["energy"] for wl in eye_wls]
    sing_e = [eye_dicts[wl]["single"]["energy"] for wl in eye_wls]
    ax.bar(x - w/2, part_e, w, label="Partial",
           color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, sing_e, w, label="Single",
           color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Energy (μJ)")
    ax.set_xticks(x)
    ax.set_xticklabels(eye_wls)
    ax.set_title("Eyeriss (2048 PEs, WReg=384)\n(same wreg for both scenarios)")
    ax.legend()

    # --- DepFiN panel ---
    ax = axes[1]
    dep_data = FUSION_FIXED_DEPFIN_PARTIAL
    x = np.arange(len(WORKLOADS))
    part_e = [dep_data[wl]["partial"]["energy"] for wl in WORKLOADS]
    sing_e = [dep_data[wl]["single"]["energy"] for wl in WORKLOADS]

    ax.bar(x - w/2, part_e, w, label="Partial",
           color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, sing_e, w, label="Single",
           color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Energy (μJ)")
    ax.set_xticks(x)
    sub = [f"{wl}\nFMEM={dep_data[wl]['fmem']}" for wl in WORKLOADS]
    ax.set_xticklabels(sub, fontsize=8)
    ax.set_title("DepFiN (2048 PEs, Partial-Sized Mem)")
    ax.legend()

    fig.tight_layout()
    _save(fig, "fusion_fixed_overview_2048pe_partial")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 13: Cross-Architecture Fusion Energy Ratio Summary
# ────────────────────────────────────────────────────────────────────
def plot_fusion_cross_arch():
    """Bar chart: Full/Single energy ratio per workload for both architectures (auto-sized)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Full-Sized Arch — Full/Single Energy Ratio", fontsize=15, y=1.0)

    x = np.arange(len(WORKLOADS))
    w = 0.35

    depfin_ratios = [FUSION_AUTO["DepFiN"][wl]["full"]["energy"] /
                     FUSION_AUTO["DepFiN"][wl]["single"]["energy"]
                     for wl in WORKLOADS]
    eyeriss_ratios = [FUSION_AUTO["Eyeriss"][wl]["full"]["energy"] /
                      FUSION_AUTO["Eyeriss"][wl]["single"]["energy"]
                      for wl in WORKLOADS]

    ax.bar(x - w/2, depfin_ratios, w, label="DepFiN", color="#1f77b4",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, eyeriss_ratios, w, label="Eyeriss", color="#ff7f0e",
           edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Full / Single Energy Ratio  (higher = full worse)")
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS)
    ax.legend()
    ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.5)

    for i, (dr, er) in enumerate(zip(depfin_ratios, eyeriss_ratios)):
        ax.text(i - w/2, dr + 0.5, f"{dr:.1f}×", ha="center", fontsize=8, fontweight="bold")
        ax.text(i + w/2, er + 0.5, f"{er:.1f}×", ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fusion_cross_arch_energy_ratio")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 14: Eyeriss CS1 — WReg vs Feasibility + Best EDP
# ────────────────────────────────────────────────────────────────────
def plot_eyeriss_cs1_feasibility():
    """Shows how increasing WReg allows more configurations but increases energy."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Eyeriss CS1 — WReg Trade-off: Feasibility vs Energy",
                 fontsize=15, y=0.98)

    for ax, wl in zip(axes.flat, WORKLOADS):
        d = EYERISS_CS1[wl]
        mask = [i for i, e in enumerate(d["energy"]) if e is not None]
        if not mask:
            ax.text(0.5, 0.5, "All infeasible", transform=ax.transAxes,
                   ha="center", va="center", fontsize=14)
            ax.set_title(wl)
            continue

        wreg = [d["wreg"][i] for i in mask]
        n_feas = [d["n_feasible"][i] for i in mask]
        min_pes = [d["min_pe_total"][i] for i in mask]

        # Feasibility count
        ln1 = ax.bar(range(len(wreg)), n_feas, color=COLORS[wl], alpha=0.7, label="# feasible configs")
        ax.set_ylabel("# feasible configs")
        ax.set_xticks(range(len(wreg)))
        ax.set_xticklabels([str(w) for w in wreg])
        ax.set_xlabel("WReg size")
        ax.set_title(wl)

        # Min PEs on twin axis
        ax2 = ax.twinx()
        ln2 = ax2.plot(range(len(wreg)), min_pes, "D-", color="red",
                       markersize=8, label="Min total PEs")
        ax2.set_ylabel("Min total PEs", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "eyeriss_cs1_wreg_feasibility")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 15: Auto-Sized Fusion — Latency Savings Summary
# ────────────────────────────────────────────────────────────────────
def plot_fusion_latency_savings():
    """Bar chart showing latency savings (%) for Full vs Single, both architectures."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Full-Sized Arch — Latency Savings  (Full vs Single)", fontsize=15, y=1.0)

    x = np.arange(len(WORKLOADS))
    w = 0.35

    depfin_savings = []
    eyeriss_savings = []
    for wl in WORKLOADS:
        d_dep = FUSION_AUTO["DepFiN"][wl]
        d_eye = FUSION_AUTO["Eyeriss"][wl]
        depfin_savings.append(
            (1.0 - d_dep["full"]["latency"] / d_dep["single"]["latency"]) * 100)
        eyeriss_savings.append(
            (1.0 - d_eye["full"]["latency"] / d_eye["single"]["latency"]) * 100)

    ax.bar(x - w/2, depfin_savings, w, label="DepFiN", color="#1f77b4",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, eyeriss_savings, w, label="Eyeriss", color="#ff7f0e",
           edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Latency Savings (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(WORKLOADS)
    ax.axhline(0, color="black", ls="-", lw=0.8)
    ax.legend()

    for i, (ds, es) in enumerate(zip(depfin_savings, eyeriss_savings)):
        y_off = 1 if ds >= 0 else -3
        ax.text(i - w/2, ds + y_off, f"{ds:.1f}%", ha="center", fontsize=8)
        y_off = 1 if es >= 0 else -3
        ax.text(i + w/2, es + y_off, f"{es:.1f}%", ha="center", fontsize=8)

    fig.tight_layout()
    _save(fig, "fusion_latency_savings")
    return fig


# ====================================================================
#  MAIN
# ====================================================================

def main():
    show = "--show" in sys.argv
    print("Generating thesis plots...")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}\n")

    figs = []

    # DepFiN sweeps
    print("[Group 1] DepFiN PE sensitivity sweeps")
    figs.append(plot_depfin_cs4())
    figs.append(plot_depfin_cs1())
    figs.extend(plot_depfin_cs2_cs3())

    # Eyeriss sweeps
    print("\n[Group 2] Eyeriss register sensitivity sweeps")
    figs.append(plot_eyeriss_cs1())
    figs.append(plot_eyeriss_cs1_inverted())
    figs.append(plot_eyeriss_cs2_cs3())
    figs.append(plot_eyeriss_cs2_inverted())
    figs.append(plot_eyeriss_cs3_inverted())
    figs.append(plot_eyeriss_cs5())

    # Fusion comparisons
    print("\n[Group 3] Fusion comparisons")
    figs.append(plot_fusion_auto_energy())
    figs.append(plot_fusion_auto_latency())
    figs.append(plot_fusion_auto_dram())
    figs.append(plot_fusion_auto_energy_norm())
    figs.append(plot_fusion_auto_latency_norm())
    figs.append(plot_fusion_auto_edp_norm())
    figs.append(plot_partial_sized_energy())
    figs.append(plot_partial_sized_latency())
    figs.append(plot_partial_sized_edp())
    figs.append(plot_partial_sized_norm())
    figs.append(plot_fusion_cross_arch())
    figs.append(plot_fusion_latency_savings())

    # Fixed-config scaling
    print("\n[Group 4] Fixed-config fusion scaling (full-sized arch)")
    figs.append(plot_fusion_fixed_resnet18())
    figs.append(plot_fusion_fixed_vgg16())
    figs.append(plot_fusion_fixed_overview())

    print("\n[Group 5] Fixed-config fusion scaling (partial-sized arch)")
    figs.append(plot_fusion_fixed_resnet18_partial())
    figs.append(plot_fusion_fixed_vgg16_partial())
    figs.append(plot_fusion_fixed_overview_partial())

    print(f"\nDone — {len(figs)} figures generated.")

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
