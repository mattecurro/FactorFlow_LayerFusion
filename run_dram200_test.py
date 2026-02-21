#!/usr/bin/env python3
"""
Quick test: Run Scenario A (fusion comparison) for FSRCNN
on both DepFiN and Eyeriss with DRAM energy = 200 pJ/byte.

Monkey-patches the Accelergy energy functions to override DRAM energy.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Force unbuffered stdout
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# Monkey-patch BEFORE importing experiment_runner
import architectures.thesis_arch as thesis_arch

# Save originals
_orig_depfin_energy = thesis_arch.get_energy_values_from_accelergy
_orig_eyeriss_energy = thesis_arch.get_eyeriss_energy_values

DRAM_OVERRIDE = 200.0  # pJ/byte

def patched_depfin_energy(config):
    """Override DRAM energy to 200 pJ/byte for DepFiN."""
    result = _orig_depfin_energy(config)
    print(f"  [PATCH] DepFiN DRAM energy: {result['dram_energy_per_byte']:.2f} -> {DRAM_OVERRIDE} pJ/byte", flush=True)
    result['dram_energy_per_byte'] = DRAM_OVERRIDE
    if 'raw' in result:
        result['raw']['dram'] = DRAM_OVERRIDE
    return result

def patched_eyeriss_energy(config):
    """Override DRAM energy to 200 pJ/byte for Eyeriss."""
    result = _orig_eyeriss_energy(config)
    print(f"  [PATCH] Eyeriss DRAM energy: {result['dram_energy']:.2f} -> {DRAM_OVERRIDE} pJ/byte", flush=True)
    result['dram_energy'] = DRAM_OVERRIDE
    return result

thesis_arch.get_energy_values_from_accelergy = patched_depfin_energy
thesis_arch.get_eyeriss_energy_values = patched_eyeriss_energy

# Now import experiment runner (it will use patched functions)
from experiment_runner import ExperimentRunner

runner = ExperimentRunner()

# ── Run DepFiN Only ──
print("=" * 80, flush=True)
print(f"[1/2] DepFiN FSRCNN — DRAM = {DRAM_OVERRIDE} pJ/byte", flush=True)
print("=" * 80, flush=True)
result_depfin = runner.run_fusion_comparison(
    network='fsrcnn',
    arch_type='depfin',
    fmem_size_kb=576,
    wmem_size_kb=19,
    pe_rows=16,
    pe_cols=128,
    tile_size=120,
    scale_bandwidth=True,
    base_tile_size=128,
    verbose=False,
)
sys.stdout.flush()

# ── Run Eyeriss Only ──
print("\n" + "=" * 80, flush=True)
print(f"[2/2] Eyeriss FSRCNN — DRAM = {DRAM_OVERRIDE} pJ/byte", flush=True)
print("=" * 80, flush=True)
result_eyeriss = runner.run_fusion_comparison(
    network='fsrcnn',
    arch_type='eyeriss',
    gb_size_kb=128,
    pe_rows=128,
    pe_cols=16,
    input_reg_entries=234,
    weight_reg_entries=384,
    intermediate_reg_entries=200,
    output_reg_entries=64,
    tile_size=120,
    verbose=False,
)
sys.stdout.flush()
print("  (from FUSION_AUTO dict)")
print(f"  DepFiN FSRCNN original:  full E=8.352e+03, single E=6.536e+03, partial E=9.150e+03")
print(f"  Eyeriss FSRCNN original: full E=7.150e+04, single E=1.117e+04, partial E=7.203e+04")
