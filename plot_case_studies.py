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
        "min_pe_config": ["128x4", "128x4", "80x4"],
        "min_pe_total": [512, 512, 320],
        "energy": [1.203e5, 1.310e5, 1.412e5],
        "latency": [2.760e7, 2.760e7, 4.769e7],
        "edp": [3.33e6, 3.62e6, 6.75e6],
        "n_feasible": [26, 27, 36],
    },
    "MC-CNN": {
        "wreg": [100, 200, 300, 600],
        "min_pe_config": ["128x12", "64x12", "32x12", "12x12"],
        "min_pe_total": [1536, 768, 384, 144],
        "energy": [6.649e4, 7.730e4, 8.983e4, 1.315e5],
        "latency": [1.223e7, 2.344e7, 4.585e7, 9.068e7],
        "edp": [8.28e5, 1.84e6, 4.18e6, 1.20e7],
        "n_feasible": [14, 24, 37, 56],
    },
    "VGG16": {
        "wreg": [500, 1000, 2500, 5000, 10000, 20000],
        "min_pe_config": ["N/A", "N/A", "64x128", "64x64", "64x32", "64x16"],
        "min_pe_total": [0, 0, 8192, 4096, 2048, 1024],
        "energy": [None, None, 6.716e5, 1.064e6, 1.882e6, 3.549e6],
        "latency": [None, None, 5.681e6, 6.715e6, 1.103e7, 2.010e7],
        "edp": [None, None, 3.95e6, 7.25e6, 2.09e7, 7.15e7],
        "n_feasible": [0, 0, 11, 20, 27, 31],
    },
    "ResNet18": {
        "wreg": [400, 902, 1800, 3600, 7200],
        "min_pe_config": ["N/A", "256x64", "128x64", "64x64", "64x32"],
        "min_pe_total": [0, 16384, 8192, 4096, 2048],
        "energy": [None, 5.102e4, 7.245e4, 1.173e5, 2.047e5],
        "latency": [None, 3.042e6, 3.042e6, 3.059e6, 3.422e6],
        "edp": [None, 1.63e5, 2.29e5, 3.68e5, 7.08e5],
        "n_feasible": [0, 2, 9, 18, 21],
    },
}

# CS2 — IntReg Sensitivity Sweep (Summary A: minimum PEs)
EYERISS_CS2 = {
    "FSRCNN": {
        "intreg": [200, 300, 400],
        "energy": [2.880e4, 2.880e4, 2.880e4],
        "latency": [3.525e7, 3.525e7, 3.525e7],
        "edp": [1.02e6, 1.02e6, 1.02e6],
        "note": "Non-binding: all identical",
    },
    "MC-CNN": {
        "intreg": [15, 30, 50, 100],
        "energy": [None, 3.488e4, 3.488e4, 3.488e4],
        "latency": [None, 9.068e7, 9.068e7, 9.068e7],
        "edp": [None, 3.18e6, 3.18e6, 3.18e6],
        "note": "IntReg=15 infeasible; saturates at >=30",
    },
    "VGG16": {
        "intreg": [50, 100, 230, 500],
        "energy": [4.159e4, 4.048e4, 4.114e4, 4.114e4],
        "latency": [5.681e6, 6.715e6, 6.715e6, 6.715e6],
        "edp": [2.42e5, 2.76e5, 2.77e5, 2.77e5],
        "note": "Saturates at IntReg>=230",
    },
    "ResNet18": {
        "intreg": [50, 100, 200, 400, 1000],
        "energy": [7.113e3, 6.944e3, 6.779e3, 6.697e3, 6.697e3],
        "latency": [3.042e6, 3.042e6, 3.042e6, 3.059e6, 3.059e6],
        "edp": [2.22e4, 2.15e4, 2.08e4, 2.06e4, 2.06e4],
        "note": "Saturates at IntReg>=400",
    },
}

# CS3 — OutReg Sensitivity Sweep (Summary A: minimum PEs)
EYERISS_CS3 = {
    "FSRCNN": {
        "outreg": [200, 300, 400],
        "energy": [2.880e4, 2.880e4, 2.880e4],
        "latency": [3.525e7, 3.525e7, 3.525e7],
        "edp": [1.02e6, 1.02e6, 1.02e6],
        "note": "Zero effect: all identical",
    },
    "MC-CNN": {
        "outreg": [8, 16, 32, 64],
        "energy": [3.488e4, 3.488e4, 3.488e4, 3.488e4],
        "latency": [9.068e7, 9.068e7, 9.068e7, 9.068e7],
        "edp": [3.18e6, 3.18e6, 3.18e6, 3.18e6],
        "note": "Never binding for min PEs",
    },
    "VGG16": {
        "outreg": [4, 8, 16, 32, 64],
        "energy": [4.159e4, 4.048e4, 4.026e4, 4.114e4, 4.114e4],
        "latency": [5.681e6, 6.715e6, 6.715e6, 6.715e6, 6.715e6],
        "edp": [2.42e5, 2.76e5, 2.72e5, 2.77e5, 2.77e5],
        "note": "Saturates at OutReg>=32",
    },
    "ResNet18": {
        "outreg": [8, 16, 32, 64],
        "energy": [6.944e3, 6.779e3, 6.697e3, 6.697e3],
        "latency": [3.042e6, 3.042e6, 3.059e6, 3.059e6],
        "edp": [2.15e4, 2.08e4, 2.06e4, 2.06e4],
        "note": "Saturates at OutReg>=32",
    },
}

# CS5 — Eyeriss PE Aspect Ratio Sweep (fixed total PEs)
EYERISS_CS5 = {
    "FSRCNN": {
        "total_pes": 2048,
        "configs": ["64x32", "128x16", "256x8", "512x4", "1024x2", "2048x1"],
        "energy": [1.517e5, 1.523e5, 1.531e5, 1.548e5, 1.562e5, 1.562e5],
        "latency": [1.889e7, 1.889e7, 1.889e7, 1.889e7, 1.889e7, 1.918e7],
        "edp": [2.87e6, 2.88e6, 2.90e6, 2.95e6, 2.98e6, 3.03e6],
        "best": "64x32",
    },
    "MC-CNN": {
        "total_pes": 2048,
        "configs": ["32x64", "64x32", "128x16", "256x8",
                     "512x4", "1024x2", "2048x1"],
        "energy": [1.312e5, 1.323e5, 1.378e5, 1.378e5,
                   1.378e5, 1.452e5, 1.452e5],
        "latency": [1.272e7, 1.272e7, 1.223e7, 1.223e7,
                    1.223e7, 1.261e7, 1.261e7],
        "edp": [1.68e6, 1.69e6, 1.70e6, 1.70e6,
                1.70e6, 1.89e6, 1.89e6],
        "best": "32x64",
    },
    "VGG16": {
        "total_pes": 4096,
        "configs": ["64x64", "128x32", "256x16"],
        "energy": [9.960e5, 9.782e5, 9.743e5],
        "latency": [6.715e6, 6.715e6, 6.715e6],
        "edp": [6.79e6, 6.62e6, 6.57e6],
        "best": "256x16",
    },
    "ResNet18": {
        "total_pes": 16384,
        "configs": ["256x64", "512x32", "1024x16", "2048x8",
                     "4096x4", "8192x2", "16384x1"],
        "energy": [5.103e4, 5.062e4, 5.113e4, 5.136e4,
                   5.136e4, 5.136e4, 5.136e4],
        "latency": [3.042e6, 3.042e6, 3.042e6, 3.042e6,
                    3.042e6, 3.042e6, 3.042e6],
        "edp": [1.63e5, 1.60e5, 1.62e5, 1.62e5,
                1.62e5, 1.62e5, 1.62e5],
        "best": "512x32",
    },
}

# ====================================================================
#  DATA  –  Fusion Comparisons (Auto-Sized)
# ====================================================================

FUSION_AUTO = {
    "DepFiN": {
        "FSRCNN": {
            "pe": "4x512", "fmem": "1056KB",
            "full":    {"energy": 1.003e4, "latency": 5.680e6, "edp": 6.50e4,
                        "dram_rd": 1_573_992, "dram_wr": 8_294_400},
            "single":  {"energy": 8.238e3, "latency": 2.244e7, "edp": 3.19e4,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
            "partial": {"energy": 1.033e4, "latency": 6.224e6, "edp": 2.52e4,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
        },
        "MC-CNN": {
            "pe": "8x256", "fmem": "1056KB",
            "full":    {"energy": 1.227e4, "latency": 8.074e6, "edp": 1.16e5,
                        "dram_rd": 494_928, "dram_wr": 14_943_744},
            "single":  {"energy": 4.833e3, "latency": 1.381e7, "edp": 1.87e4,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
            "partial": {"energy": 1.301e4, "latency": 8.074e6, "edp": 6.58e4,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
        },
        "VGG16": {
            "pe": "16x128", "fmem": "1056KB",
            "full":    {"energy": 7.641e4, "latency": 2.433e7, "edp": 3.41e6,
                        "dram_rd": 14_860_992, "dram_wr": 100_352},
            "single":  {"energy": 2.706e3, "latency": 2.452e7, "edp": 7.49e3,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
            "partial": {"energy": 5.799e4, "latency": 2.432e7, "edp": 5.87e5,
                        "dram_rd": 16_366_272, "dram_wr": 6_121_472},
        },
        "ResNet18": {
            "pe": "16x128", "fmem": "1056KB",
            "full":    {"energy": 1.090e4, "latency": 7.812e6, "edp": 1.52e5,
                        "dram_rd": 11_032_512, "dram_wr": 25_088},
            "single":  {"energy": 4.294e2, "latency": 3.809e6, "edp": 1.78e2,
                        "dram_rd": 5_296_832, "dram_wr": 2_232_832},
            "partial": {"energy": 8.085e3, "latency": 7.813e6, "edp": 1.41e4,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
        },
    },
    "Eyeriss": {
        "FSRCNN": {
            "pe": "84x16", "gb": "128KB",
            "full":    {"energy": 1.519e5, "latency": 3.022e7, "edp": 4.60e6,
                        "dram_rd": 1_573_992, "dram_wr": 8_294_400},
            "single":  {"energy": 8.618e3, "latency": 4.126e7, "edp": 5.50e4,
                        "dram_rd": 90_738_792, "dram_wr": 97_459_200},
            "partial": {"energy": 9.370e4, "latency": 3.022e7, "edp": 9.38e5,
                        "dram_rd": 14_015_592, "dram_wr": 20_736_000},
        },
        "MC-CNN": {
            "pe": "56x64", "gb": "128KB",
            "full":    {"energy": 1.331e5, "latency": 2.302e7, "edp": 3.08e6,
                        "dram_rd": 494_928, "dram_wr": 14_943_744},
            "single":  {"energy": 5.945e3, "latency": 2.483e7, "edp": 3.97e4,
                        "dram_rd": 45_326_160, "dram_wr": 59_774_976},
            "partial": {"energy": 8.024e4, "latency": 2.302e7, "edp": 9.86e5,
                        "dram_rd": 15_438_672, "dram_wr": 29_887_488},
        },
        "VGG16": {
            "pe": "196x64", "gb": "128KB",
            "full":    {"energy": 1.006e6, "latency": 8.159e7, "edp": 8.33e7,
                        "dram_rd": 14_860_992, "dram_wr": 100_352},
            "single":  {"energy": 1.734e4, "latency": 8.159e7, "edp": 2.32e5,
                        "dram_rd": 23_792_320, "dram_wr": 13_547_520},
            "partial": {"energy": 2.532e5, "latency": 8.159e7, "edp": 4.36e6,
                        "dram_rd": 16_366_272, "dram_wr": 6_121_472},
        },
        "ResNet18": {
            "pe": "256x32", "gb": "128KB",
            "full":    {"energy": 1.257e5, "latency": 1.745e7, "edp": 2.22e6,
                        "dram_rd": 11_032_512, "dram_wr": 25_088},
            "single":  {"energy": 1.914e3, "latency": 1.501e7, "edp": 2.21e3,
                        "dram_rd": 12_449_984, "dram_wr": 2_308_096},
            "partial": {"energy": 4.406e4, "latency": 1.746e7, "edp": 1.06e5,
                        "dram_rd": 11_760_064, "dram_wr": 752_640},
        },
    },
}

# ====================================================================
#  DATA  –  Fusion Comparisons (Fixed-Config, Eyeriss)
# ====================================================================

# ResNet18 — Full vs Single energy ratio across PE configs
FUSION_FIXED_RESNET18 = {
    "configs": ["128x128", "512x32", "256x32", "256x16", "128x16"],
    "total_pes": [16384, 16384, 8192, 4096, 2048],
    "wreg": [920, 902, 1800, 3600, 7200],
    "full_energy": [5.400e4, 5.062e4, 7.005e4, 1.137e5, 2.035e5],
    "single_energy": [3.769e3, 2.643e3, 1.920e3, 1.412e3, 1.368e3],
    "partial_energy": [3.928e4, 4.972e4, 7.017e4, 1.144e5, 2.046e5],
    "full_latency": [3.042e6, 3.042e6, 3.042e6, 3.059e6, 3.422e6],
    "single_latency": [3.301e6, 3.301e6, 3.301e6, 3.301e6, 3.412e6],
    "energy_ratio_fs": [14.33, 19.15, 36.49, 80.55, 148.77],
    "energy_ratio_fp": [1.02, 1.02, 1.00, 0.99, 0.99],  # full/partial ~1
    "latency_ratio": [0.92, 0.92, 0.92, 0.93, 1.00],
}

# VGG16 — Full vs Single across PE configs
FUSION_FIXED_VGG16 = {
    "configs": ["512x32", "256x32", "256x16"],
    "total_pes": [16384, 8192, 4096],
    "wreg": [1200, 2400, 4800],
    "full_energy": [3.618e5, 5.471e5, 9.398e5],
    "single_energy": [1.360e4, 9.812e3, 6.707e3],
    "partial_energy": [2.620e5, 4.044e5, 6.952e5],
    "full_latency": [5.455e6, 5.681e6, 6.715e6],
    "single_latency": [6.922e6, 6.922e6, 7.694e6],
    "energy_ratio_fs": [26.60, 55.76, 140.11],
    "energy_ratio_fp": [1.38, 1.35, 1.35],
    "latency_ratio": [0.79, 0.82, 0.87],
}

# MC-CNN — fixed config
FUSION_FIXED_MCCNN = {
    "config": "256x8", "total_pes": 2048, "wreg": 384,
    "full":    {"energy": 8.955e4, "latency": 1.223e7},
    "single":  {"energy": 7.488e3, "latency": 1.495e7},
    "partial": {"energy": 9.138e4, "latency": 1.223e7},
    "energy_ratio_fs": 11.96,
    "energy_ratio_fp": 0.98,
}

# FSRCNN — fixed config
FUSION_FIXED_FSRCNN = {
    "config": "128x16", "total_pes": 2048, "wreg": 384,
    "full":    {"energy": 6.757e4, "latency": 1.889e7},
    "single":  {"energy": 8.747e3, "latency": 3.525e7},
    "partial": {"energy": 6.855e4, "latency": 1.889e7},
    "energy_ratio_fs": 7.73,
    "energy_ratio_fp": 0.99,
}

# DepFiN fixed-config fusion
FUSION_FIXED_DEPFIN = {
    "ResNet18": {
        "config": "16x128", "total_pes": 2048,
        "full":    {"energy": 1.083e4, "latency": 7.812e6},
        "single":  {"energy": 9.499e2, "latency": 6.905e6},
        "partial": {"energy": 1.142e4, "latency": 1.706e7},
        "energy_ratio_fs": 11.40,
        "energy_ratio_fp": 0.95,
    },
    "VGG16": {
        "config": "16x128", "total_pes": 2048,
        "full":    {"energy": 7.607e4, "latency": 2.433e7},
        "single":  {"energy": 3.395e3, "latency": 2.452e7},
        "partial": {"energy": 5.834e4, "latency": 5.204e7},
        "energy_ratio_fs": 22.41,
        "energy_ratio_fp": 1.30,
    },
    "MC-CNN": {
        "config": "8x256", "total_pes": 2048,
        "full":    {"energy": 1.177e4, "latency": 8.074e6},
        "single":  {"energy": 4.224e3, "latency": 1.381e7},
        "partial": {"energy": 1.273e4, "latency": 8.074e6},
        "energy_ratio_fs": 2.79,
        "energy_ratio_fp": 0.92,
    },
    "FSRCNN": {
        "config": "16x128", "total_pes": 2048,
        "full":    {"energy": 8.231e3, "latency": 6.122e6},
        "single":  {"energy": 6.402e3, "latency": 1.175e7},
        "partial": {"energy": 9.027e3, "latency": 6.313e6},
        "energy_ratio_fs": 1.29,
        "energy_ratio_fp": 0.91,
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
        ax.set_ylabel("EDP  (μJ·cc)")
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
    fig.suptitle("DepFiN — Tile Size Sensitivity  (16×128 PEs) DepFiN 16×128 PEs,\n FMEM=1056KB, WMEM=524KB. \n FMEM BW scales as: BW_scaled = BW_base × (tile_size / 128)", fontsize=16, y=0.98)

    for ax, wl in zip(axes.flat, WORKLOADS):
        d = DEPFIN_CS1[wl]
        ax.semilogy(d["tiles"], d["edp"], "o-", color=COLORS[wl])
        ax.set_xlabel("Tile size")
        ax.set_ylabel("EDP  (μJ·cc)")
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
#  Figure 3: DepFiN CS2+CS3 — Row & Col Sweeps
# ────────────────────────────────────────────────────────────────────
def plot_depfin_cs2_cs3():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("DepFiN — PE Row Sweep (top) and PE Col Sweep (bottom)", fontsize=16, y=0.98)

    # Row sweep (top row)
    for ax, wl in zip(axes[0], WORKLOADS):
        d = DEPFIN_CS2[wl]
        ax.plot(d["rows"], d["edp"], "o-", color=COLORS[wl])
        ax.set_xlabel("PE rows  (cols=128)")
        ax.set_ylabel("EDP")
        ax.set_title(f"{wl} rows")
        ax.set_xticks(d["rows"])
        _sci_fmt(ax)

    # Col sweep (bottom row)
    for ax, wl in zip(axes[1], WORKLOADS):
        d = DEPFIN_CS3[wl]
        ax.plot(d["cols"], d["edp"], "s-", color=COLORS[wl])
        ax.set_xlabel("PE cols  (rows=16)")
        ax.set_ylabel("EDP")
        ax.set_title(f"{wl} cols")
        ax.set_xticks(d["cols"])
        _sci_fmt(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "depfin_cs2cs3_row_col_sweep")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 4: Eyeriss CS1 — WReg Sweep (min-PEs view)
# ────────────────────────────────────────────────────────────────────
def plot_eyeriss_cs1():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Eyeriss — WReg Sensitivity Sweep  (min-PEs config)", fontsize=16, y=0.98)

    for ax, wl in zip(axes.flat, WORKLOADS):
        d = EYERISS_CS1[wl]
        # Filter out None entries
        mask = [i for i, e in enumerate(d["energy"]) if e is not None]
        wreg = [d["wreg"][i] for i in mask]
        energy = [d["energy"][i] for i in mask]
        edp = [d["edp"][i] for i in mask]
        n_feas = [d["n_feasible"][i] for i in mask]
        min_pes = [d["min_pe_total"][i] for i in mask]

        # Energy on primary y-axis
        ln1 = ax.semilogy(wreg, energy, "o-", color=COLORS[wl], label="Energy (μJ)")
        ax.set_xlabel("WReg size")
        ax.set_ylabel("Energy (μJ)")
        ax.set_title(wl)

        # Min PEs on secondary y-axis
        ax2 = ax.twinx()
        ln2 = ax2.plot(wreg, min_pes, "D--", color="gray", alpha=0.6, label="Min PEs")
        ax2.set_ylabel("Min total PEs", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        # Feasibility count as text
        for i, (w, n) in enumerate(zip(wreg, n_feas)):
            ax.annotate(f"{n} cfg", (w, energy[i]),
                       textcoords="offset points", xytext=(0, 12),
                       fontsize=7, ha="center", color="navy")

        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc="upper left", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "eyeriss_cs1_wreg_sweep")
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
#  Figure 6: Eyeriss CS5 — PE Aspect Ratio
# ────────────────────────────────────────────────────────────────────
def plot_eyeriss_cs5():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Eyeriss — PE Aspect Ratio Sweep  (fixed total PEs)", fontsize=16, y=0.98)

    for ax, wl in zip(axes.flat, WORKLOADS):
        d = EYERISS_CS5[wl]
        x = np.arange(len(d["configs"]))
        ax.plot(x, d["edp"], "o-", color=COLORS[wl], label="EDP")
        ax.set_ylabel("EDP  (μJ·cc)")
        ax.set_xticks(x)
        ax.set_xticklabels(d["configs"], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{wl}  ({d['total_pes']} PEs, best: {d['best']})")
        _sci_fmt(ax)

        # Mark best
        best_idx = d["configs"].index(d["best"])
        ax.plot(best_idx, d["edp"][best_idx], "*", markersize=14,
                color="gold", markeredgecolor="black", zorder=5)

        # Energy on twin axis
        ax2 = ax.twinx()
        ax2.plot(x, d["energy"], "s--", color="gray", alpha=0.5, label="Energy")
        ax2.set_ylabel("Energy (μJ)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        _sci_fmt(ax2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "eyeriss_cs5_aspect_ratio")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 7: Auto-Sized Fusion — Energy Comparison (grouped bars)
# ────────────────────────────────────────────────────────────────────
def plot_fusion_auto_energy():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Auto-Sized Fusion — Energy Comparison", fontsize=16, y=1.0)

    for ax, arch in zip(axes, ["DepFiN", "Eyeriss"]):
        data = FUSION_AUTO[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.25

        full_e = [data[wl]["full"]["energy"] for wl in WORKLOADS]
        part_e = [data[wl]["partial"]["energy"] for wl in WORKLOADS]
        sing_e = [data[wl]["single"]["energy"] for wl in WORKLOADS]

        ax.bar(x - w, full_e, w, label="Full Fusion", color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
        ax.bar(x, part_e, w, label="Partial Fusion", color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w, sing_e, w, label=r"$\Sigma$ Singles", color=FUSION_COLORS["Single"], edgecolor="black", linewidth=0.5)

        ax.set_yscale("log")
        ax.set_ylabel("Energy (μJ)")
        ax.set_xticks(x)
        ax.set_xticklabels(WORKLOADS)
        ax.set_title(arch)
        ax.legend(fontsize=9)

        # Add ratio annotations
        for i, wl in enumerate(WORKLOADS):
            ratio = data[wl]["full"]["energy"] / data[wl]["single"]["energy"]
            ax.annotate(f"{ratio:.1f}×", (i - w, full_e[i]),
                       textcoords="offset points", xytext=(0, 8),
                       fontsize=7, ha="center", color="darkred", fontweight="bold")

    fig.tight_layout()
    _save(fig, "fusion_auto_energy")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 8: Auto-Sized Fusion — Latency Comparison
# ────────────────────────────────────────────────────────────────────
def plot_fusion_auto_latency():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Auto-Sized Fusion — Latency Comparison", fontsize=16, y=1.0)

    for ax, arch in zip(axes, ["DepFiN", "Eyeriss"]):
        data = FUSION_AUTO[arch]
        x = np.arange(len(WORKLOADS))
        w = 0.25

        full_l = [data[wl]["full"]["latency"] for wl in WORKLOADS]
        part_l = [data[wl]["partial"]["latency"] for wl in WORKLOADS]
        sing_l = [data[wl]["single"]["latency"] for wl in WORKLOADS]

        ax.bar(x - w, full_l, w, label="Full Fusion", color=FUSION_COLORS["Full"], edgecolor="black", linewidth=0.5)
        ax.bar(x, part_l, w, label="Partial Fusion", color=FUSION_COLORS["Partial"], edgecolor="black", linewidth=0.5)
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
#  Figure 9: Auto-Sized Fusion — DRAM Traffic
# ────────────────────────────────────────────────────────────────────
def plot_fusion_auto_dram():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Auto-Sized Fusion — DRAM Traffic Reduction", fontsize=16, y=0.98)

    for col_idx, arch in enumerate(["DepFiN", "Eyeriss"]):
        data = FUSION_AUTO[arch]
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

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fusion_auto_dram_traffic")
    return fig


# ────────────────────────────────────────────────────────────────────
#  Figure 10: Fixed-Config Fusion — Energy Ratio Scaling (ResNet18)
# ────────────────────────────────────────────────────────────────────
def plot_fusion_fixed_resnet18():
    d = FUSION_FIXED_RESNET18
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("ResNet18 Eyeriss — Fixed-Config Fusion Scaling", fontsize=15, y=1.0)

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
    fig.suptitle("VGG16 Eyeriss — Fixed-Config Fusion Scaling", fontsize=15, y=1.0)

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
    fig.suptitle("Fixed-Config Fusion (2048 PEs) — Full vs Partial vs Single",
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
#  Figure 13: Cross-Architecture Fusion Energy Ratio Summary
# ────────────────────────────────────────────────────────────────────
def plot_fusion_cross_arch():
    """Bar chart: Full/Single energy ratio per workload for both architectures (auto-sized)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Auto-Sized Fusion — Full/Single Energy Ratio", fontsize=15, y=1.0)

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
    fig.suptitle("Auto-Sized Fusion — Latency Savings  (Full vs Single)", fontsize=15, y=1.0)

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
    figs.append(plot_depfin_cs2_cs3())

    # Eyeriss sweeps
    print("\n[Group 2] Eyeriss register sensitivity sweeps")
    figs.append(plot_eyeriss_cs1())
    figs.append(plot_eyeriss_cs1_feasibility())
    figs.append(plot_eyeriss_cs2_cs3())
    figs.append(plot_eyeriss_cs5())

    # Fusion comparisons
    print("\n[Group 3] Fusion comparisons")
    figs.append(plot_fusion_auto_energy())
    figs.append(plot_fusion_auto_latency())
    figs.append(plot_fusion_auto_dram())
    figs.append(plot_fusion_cross_arch())
    figs.append(plot_fusion_latency_savings())

    # Fixed-config scaling
    print("\n[Group 4] Fixed-config fusion scaling")
    figs.append(plot_fusion_fixed_resnet18())
    figs.append(plot_fusion_fixed_vgg16())
    figs.append(plot_fusion_fixed_overview())

    print(f"\nDone — {len(figs)} figures generated.")

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
