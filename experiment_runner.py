"""
Thesis Experiment Runner

A systematic framework for sweeping independent variables and collecting results
for layer fusion experiments.

Independent Variables:
- Workload: FSRCNN, MC-CNN, VGG16, ResNet18
- Fusion Level: single-layer, block-fused, fully-fused
- Architecture: Memory sizes, PE array dimensions

Dependent Variables (collected):
- Energy (μJ)
- Latency (cycles)
- EDP (Energy-Delay Product)
- Memory Operations (MOPs) per level
- Utilization
- Mapping search time (s)

Usage:
    python experiment_runner.py --help
    python experiment_runner.py --workload fsrcnn --fusion-level block
    python experiment_runner.py --sweep-all --output results.csv

    
Sample Commands:
    Single Workload, full fusion, fixed architecture (memory sizes, PE config):
    
        python3 experiment_runner.py \
        --workload fsrcnn --fusion full --variant 8layer \
        --fmem-size 256 --wmem-size 512 --pe-rows 16 --pe-cols 128 \
        --verbose 2>&1 | tee results/fsrcnn_8layer_fmem256_wmem512_pe16x128.log
    or
        python3 experiment_runner.py --sweep-arch \
        --workload fsrcnn --fusion full --variant 8layer \
        --fmem-size 256 --wmem-size 512 --pe-rows 16 --pe-cols 128 \
        --verbose --output results/fsrcnn_8layer_fmem256_wmem512_pe16x128.csv 2>&1 | tee results/fsrcnn_8layer_fmem256_wmem512_pe16x128.log

    Single Workload, full fusion, sweep arch (memory sizes and PE config): 3x2x3 = 18 runs
        python3 experiment_runner.py --sweep-arch \
        --workload fsrcnn --fusion full --variant 8layer \
        --fmem-sizes 128 256 512 --wmem-sizes 256 512 \
        --pe-configs 8x64 16x128 32x256 \
        --verbose 2>&1 | tee results/fsrcnn_8layer_arch_sweep.log

    Single Workload, Fusion Level, sweep-arch Architecture, FC:
        python3 experiment_runner.py --sweep-arch --fmem-sizes 128 --pe-configs 64x128 --wmem-sizes 1024 
        --workload fsrcnn --fusion full --variant 8layer --verbose 2>&1 | tee results/fsrcnn_8layer_fmem128_pe64x256.log


    Single Workload, Fusion Level, Architecture Sweep:
        python3 experiment_runner.py --sweep-arch --workload fsrcnn --fusion full --variant 8layer --verbose --output fsrcnn_8layer_fmem_sweep.csv 2>&1 | tee results/fsrcnn_8layer_fmem_sweep.log


"""

import os
import sys
import csv
import json
import time
import argparse
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
from itertools import product

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factors import Shape, Coupling
from settings import Settings
from engine import run_engine
from cost_model import EDP, Energy, Latency, MOPs
from arch import Arch
from prints import factorsString

# Import workloads from case_studies_computations
from case_studies_computations import (
    # FSRCNN-TDC
    fsrcnn_tdc_single_layers,
    fsrcnn_tdc_2layer_fused,
    fsrcnn_tdc_3layer_fused,
    fsrcnn_tdc_8layer_fused,
    fsrcnn_tdc_8layer_coupling,
    # MC-CNN
    mccnn_single_layers,
    mccnn_2layer_fused,
    mccnn_4layer_fused,
    mccnn_4layer_coupling,
    # VGG16
    vgg16_single_layers,
    vgg16_block_fused,
    vgg16_block_couplings,
    vgg16_full_fused,
    vgg16_full_coupling,
    # ResNet18
    resnet18_single_layers,
    resnet18_block_fused,
    resnet18_block_couplings,
    resnet18_2layer_fused,
    resnet18_full_fused,
    resnet18_full_coupling,
)

from computations import (
    conv_coupling,
    conv_2layers_coupling,
    conv_3layers_coupling,
    conv_4layers_coupling,
    conv_8layers_coupling,
)

# Import architecture builder
# For now, we use pre-built architectures from architectures.py
# and the thesis_arch module for custom configurations
from architectures.architectures import arch_eyeriss_conv
try:
    from architectures.thesis_arch import (
        create_thesis_architecture,
        create_thesis_architecture_10layers,  # Legacy, for backward compatibility
        ThesisArchConfig,
        get_baseline_config,
    )
    THESIS_ARCH_AVAILABLE = True
except ImportError:
    THESIS_ARCH_AVAILABLE = False


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    # Experiment identification
    experiment_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Workload specification
    workload_name: str = ""          # e.g., "fsrcnn", "vgg16"
    workload_variant: str = ""       # e.g., "L0", "block1", "full"
    fusion_level: str = ""           # "single", "block", "full"
    num_fused_layers: int = 1        # Actual number of layers fused
    
    # Architecture specification
    arch_name: str = "thesis_arch"
    fmem_size_kB: int = 1056
    wmem_size_kB: int = 524
    pe_rows: int = 16
    pe_cols: int = 128
    
    # Settings
    bias_read: bool = False
    verbose: bool = False
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = f"{self.workload_name}_{self.workload_variant}_{self.timestamp}"


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    # Configuration reference
    config: ExperimentConfig = None
    
    # Success/failure
    success: bool = False
    error_message: str = ""
    
    # Performance metrics
    energy_uJ: float = 0.0
    latency_cycles: int = 0
    edp: float = 0.0
    mops: int = 0
    utilization: float = 0.0
    
    # Timing
    mapping_time_s: float = 0.0
    
    # Additional details (optional)
    mops_per_level: Dict[str, int] = field(default_factory=dict)
    mapping_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        result = {
            # From config
            "experiment_id": self.config.experiment_id if self.config else "",
            "workload_name": self.config.workload_name if self.config else "",
            "workload_variant": self.config.workload_variant if self.config else "",
            "fusion_level": self.config.fusion_level if self.config else "",
            "num_fused_layers": self.config.num_fused_layers if self.config else 0,
            "arch_name": self.config.arch_name if self.config else "",
            "fmem_size_kB": self.config.fmem_size_kB if self.config else 0,
            "wmem_size_kB": self.config.wmem_size_kB if self.config else 0,
            "pe_rows": self.config.pe_rows if self.config else 0,
            "pe_cols": self.config.pe_cols if self.config else 0,
            # Results
            "success": self.success,
            "error_message": self.error_message,
            "energy_uJ": f"{self.energy_uJ:.3e}",
            "latency_cycles": f"{self.latency_cycles:.3e}",
            "edp": f"{self.edp:.2e}",
            "mops": self.mops,
            "utilization": self.utilization,
            "mapping_time_s": self.mapping_time_s,
            "mapping": self.mapping_summary,
        }
        return result


# =============================================================================
# WORKLOAD REGISTRY
# =============================================================================

class WorkloadRegistry:
    """
    Registry of all available workloads organized by network and fusion level.
    
    Structure:
        workloads[network_name][fusion_level] = {
            variant_name: (shape, coupling, num_layers)
        }
    """
    
    def __init__(self):
        self.workloads = self._build_registry()
    
    def _build_registry(self) -> Dict[str, Dict[str, Dict[str, Tuple[Shape, Coupling, int]]]]:
        """Build the complete workload registry."""
        
        registry = {
            # =========================================================
            # FSRCNN-TDC (Activation-Dominant)
            # =========================================================
            "fsrcnn": {
                "single": {
                    name: (shape, conv_coupling, 1)
                    for name, shape in fsrcnn_tdc_single_layers.items()
                },
                "2layer": {
                    name: (shape, conv_2layers_coupling, 2)
                    for name, shape in fsrcnn_tdc_2layer_fused.items()
                },
                "3layer": {
                    name: (shape, conv_3layers_coupling, 3)
                    for name, shape in fsrcnn_tdc_3layer_fused.items()
                },
                "full": {
                    "8layer": (fsrcnn_tdc_8layer_fused, fsrcnn_tdc_8layer_coupling, 8)
                },
            },
            
            # =========================================================
            # MC-CNN (Activation-Dominant)
            # =========================================================
            "mccnn": {
                "single": {
                    name: (shape, conv_coupling, 1)
                    for name, shape in mccnn_single_layers.items()
                },
                "2layer": {
                    name: (shape, conv_2layers_coupling, 2)
                    for name, shape in mccnn_2layer_fused.items()
                },
                "full": {
                    "4layer": (mccnn_4layer_fused, mccnn_4layer_coupling, 4)
                },
            },
            
            # =========================================================
            # VGG16 (Weight-Dominant)
            # =========================================================
            "vgg16": {
                "single": {
                    name: (shape, conv_coupling, 1)
                    for name, shape in vgg16_single_layers.items()
                },
                "block": {
                    name: (shape, vgg16_block_couplings[name], 
                           2 if name in ['block1', 'block2'] else 3)
                    for name, shape in vgg16_block_fused.items()
                },
                "full": {
                    "13layer": (vgg16_full_fused, vgg16_full_coupling, 13)
                },
            },
            
            # =========================================================
            # ResNet18 (Weight-Dominant)
            # =========================================================
            "resnet18": {
                "single": {
                    name: (shape, conv_coupling, 1)
                    for name, shape in resnet18_single_layers.items()
                },
                "2layer": {
                    name: (shape, conv_2layers_coupling, 2)
                    for name, shape in resnet18_2layer_fused.items()
                },
                "block": {
                    name: (shape, resnet18_block_couplings[name],
                           4 if name == 'stage1' else 2)
                    for name, shape in resnet18_block_fused.items()
                },
                "full": {
                    "17layer": (resnet18_full_fused, resnet18_full_coupling, 17)
                },
            },
        }
        
        return registry
    
    def get_workload(self, network: str, fusion_level: str, variant: str) -> Tuple[Shape, Coupling, int]:
        """Get a specific workload by network, fusion level, and variant."""
        if network not in self.workloads:
            raise ValueError(f"Unknown network: {network}. Available: {list(self.workloads.keys())}")
        if fusion_level not in self.workloads[network]:
            raise ValueError(f"Unknown fusion level: {fusion_level} for {network}. "
                           f"Available: {list(self.workloads[network].keys())}")
        if variant not in self.workloads[network][fusion_level]:
            raise ValueError(f"Unknown variant: {variant} for {network}/{fusion_level}. "
                           f"Available: {list(self.workloads[network][fusion_level].keys())}")
        return self.workloads[network][fusion_level][variant]
    
    def list_workloads(self, network: Optional[str] = None, fusion_level: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """List all workloads, optionally filtered by network and/or fusion level."""
        result = []
        for net_name, fusion_levels in self.workloads.items():
            if network and net_name != network:
                continue
            for fuse_name, variants in fusion_levels.items():
                if fusion_level and fuse_name != fusion_level:
                    continue
                for variant_name in variants.keys():
                    result.append((net_name, fuse_name, variant_name))
        return result
    
    def get_networks(self) -> List[str]:
        """Get list of available networks."""
        return list(self.workloads.keys())
    
    def get_fusion_levels(self, network: str) -> List[str]:
        """Get list of fusion levels for a network."""
        return list(self.workloads.get(network, {}).keys())


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Main experiment runner that executes workloads and collects results.
    
    Features:
    - Run single experiments or sweep multiple variables
    - Collect and export results to CSV/JSON
    - Resume interrupted experiments
    - Progress tracking and logging
    """
    
    def __init__(self, output_dir: str = "results"):
        self.workload_registry = WorkloadRegistry()
        self.output_dir = output_dir
        self.results: List[ExperimentResult] = []
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
    
    def create_architecture(self, config: ExperimentConfig, coupling: Coupling, shape: Shape = None) -> Arch:
        """
        Create an architecture instance based on configuration.
        
        Uses create_thesis_architecture for thesis experiments (supports any number
        of fused layers), or falls back to eyeriss_conv for basic single layer tests.
        """
        # Use default architecture 
        arch = deepcopy(arch_eyeriss_conv)
        
        # If thesis architecture is available and we need custom config, use it
        if THESIS_ARCH_AVAILABLE and config.arch_name == "thesis_arch":
            try:
                arch_config = ThesisArchConfig(
                    feature_memory_size_B=config.fmem_size_kB * 1024,  # Convert KB to Bytes
                    weight_memory_size_B=config.wmem_size_kB * 1024,   # Convert KB to Bytes
                    pe_rows=config.pe_rows,
                    pe_cols=config.pe_cols,
                    num_fused_layers=config.num_fused_layers,
                )
                # Use generic architecture factory with proper layer count and shape
                arch = create_thesis_architecture(
                    config=arch_config,
                    coupling=coupling,
                    shape=shape,
                    num_layers=config.num_fused_layers
                )
            except Exception as e:
                print(f"Warning: Could not create thesis_arch ({e}), using eyeriss_conv")
                arch = deepcopy(arch_eyeriss_conv)
        
        return arch
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.
        
        Steps:
        1. Get workload (shape + coupling) from registry
        2. Create architecture with specified parameters
        3. Run the mapping engine
        4. Collect and return results
        """
        result = ExperimentResult(config=config)
        
        try:
            # Step 1: Get workload
            shape, coupling, num_layers = self.workload_registry.get_workload(
                config.workload_name,
                config.fusion_level,
                config.workload_variant
            )
            config.num_fused_layers = num_layers
            
            # Step 2: Create architecture (pass coupling and shape for thesis_arch compatibility)
            arch = self.create_architecture(config, coupling, shape)
            
            # Step 3: Check compatibility and fit constraints
            arch.checkCouplingCompatibility(coupling, shape, verbose=config.verbose)
            arch.fitConstraintsToComp(shape, enforce=True)
            
            # Step 4: Run mapping engine
            # Temporarily adjust verbosity
            original_verbose = Settings.VERBOSE
            Settings.VERBOSE = config.verbose
            
            edp, mops, energy, latency, utilization, mapping_time, final_arch = run_engine(
                arch, shape, coupling, config.bias_read, verbose=config.verbose
            )
            
            Settings.VERBOSE = original_verbose
            
            # Step 5: Collect results
            result.success = True
            result.energy_uJ = energy
            result.latency_cycles = latency
            result.edp = edp
            result.mops = mops
            result.utilization = utilization
            result.mapping_time_s = mapping_time
            result.mapping_summary = factorsString(final_arch)
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            if config.verbose:
                import traceback
                traceback.print_exc()
        
        self.results.append(result)
        return result
    
    def run_workload_sweep(
        self,
        networks: Optional[List[str]] = None,
        fusion_levels: Optional[List[str]] = None,
        arch_config: Optional[ExperimentConfig] = None,
        progress_callback=None
    ) -> List[ExperimentResult]:
        """
        Sweep over multiple workloads.
        
        Args:
            networks: List of networks to test (None = all)
            fusion_levels: List of fusion levels to test (None = all)
            arch_config: Base architecture configuration
            progress_callback: Optional callback(current, total, config) for progress
        
        Returns:
            List of ExperimentResult for all runs
        """
        if arch_config is None:
            arch_config = ExperimentConfig()
        
        # Get all workloads to run
        workloads = []
        for net in (networks or self.workload_registry.get_networks()):
            for fuse in (fusion_levels or self.workload_registry.get_fusion_levels(net)):
                for (net_name, fuse_name, variant) in self.workload_registry.list_workloads(net, fuse):
                    workloads.append((net_name, fuse_name, variant))
        
        total = len(workloads)
        results = []
        
        print(f"\n{'='*60}")
        print(f"Starting workload sweep: {total} experiments")
        print(f"{'='*60}\n")
        
        for i, (network, fusion_level, variant) in enumerate(workloads):
            # Create config for this experiment
            config = ExperimentConfig(
                workload_name=network,
                workload_variant=variant,
                fusion_level=fusion_level,
                arch_name=arch_config.arch_name,
                fmem_size_kB=arch_config.fmem_size_kB,
                wmem_size_kB=arch_config.wmem_size_kB,
                pe_rows=arch_config.pe_rows,
                pe_cols=arch_config.pe_cols,
                bias_read=arch_config.bias_read,
                verbose=arch_config.verbose,
            )
            
            print(f"[{i+1}/{total}] Running: {network}/{fusion_level}/{variant}")
            
            if progress_callback:
                progress_callback(i, total, config)
            
            result = self.run_single_experiment(config)
            results.append(result)
            
            if result.success:
                print(f"  ✓ Energy: {result.energy_uJ:.2e} μJ, "
                      f"Latency: {result.latency_cycles:.2e} cycles, "
                      f"Time: {result.mapping_time_s:.2f}s")
            else:
                print(f"  ✗ Failed: {result.error_message[:50]}...")
        
        print(f"\n{'='*60}")
        print(f"Completed: {sum(1 for r in results if r.success)}/{total} successful")
        print(f"{'='*60}\n")
        
        return results
    
    def run_architecture_sweep(
        self,
        workload: Tuple[str, str, str],  # (network, fusion_level, variant)
        fmem_sizes_kb: Optional[List[int]] = None,
        wmem_sizes_kb: Optional[List[int]] = None,
        pe_configs: Optional[List[Tuple[int, int]]] = None,  # (rows, cols)
        progress_callback=None,
        verbose: bool = False
    ) -> List[ExperimentResult]:
        """
        Sweep over architecture parameters for a fixed workload.
        
        Args:
            workload: (network, fusion_level, variant) tuple
            fmem_sizes_kb: List of FMEM sizes to try
            wmem_sizes_kb: List of WMEM sizes to try
            pe_configs: List of (rows, cols) PE configurations
            progress_callback: Optional callback for progress
        
        Returns:
            List of ExperimentResult for all runs
        """
        network, fusion_level, variant = workload
        
        # Default sweep values
        if fmem_sizes_kb is None:
            fmem_sizes_kb = [128, 512, 1024]
        if wmem_sizes_kb is None:
            wmem_sizes_kb = [256, 512, 1024]
        if pe_configs is None:
            pe_configs = [(8, 64), (16, 128), (32, 256)]
        
        # Generate all combinations
        combinations = list(product(fmem_sizes_kb, wmem_sizes_kb, pe_configs))
        total = len(combinations)
        results = []
        
        print(f"\n{'='*60}")
        print(f"Architecture sweep for {network}/{fusion_level}/{variant}")
        print(f"{total} configurations to test")
        print(f"{'='*60}\n")
        
        for i, (fmem, wmem, (pe_r, pe_c)) in enumerate(combinations):
            config = ExperimentConfig(
                workload_name=network,
                workload_variant=variant,
                fusion_level=fusion_level,
                fmem_size_kB=fmem,
                wmem_size_kB=wmem,
                pe_rows=pe_r,
                pe_cols=pe_c,
                verbose=verbose,
            )
            
            print(f"[{i+1}/{total}] FMEM={fmem}KB, WMEM={wmem}KB, PE={pe_r}x{pe_c}")
            
            if progress_callback:
                progress_callback(i, total, config)
            
            result = self.run_single_experiment(config)
            results.append(result)
            
            if result.success:
                print(f"  ✓ EDP: {result.edp:.2e}")
            else:
                print(f"  ✗ Failed")
        
        return results
    
    def export_results_csv(self, filename: Optional[str] = None) -> str:
        """Export results to CSV file."""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not self.results:
            print("No results to export")
            return filepath
        
        # Get field names from first result
        fieldnames = list(self.results[0].to_dict().keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:                
                writer.writerow(result.to_dict())
        
        print(f"Results exported to: {filepath}")
        return filepath
    
    def export_results_json(self, filename: Optional[str] = None) -> str:
        """Export results to JSON file."""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "num_experiments": len(self.results),
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results exported to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a summary of all results."""
        if not self.results:
            print("No results to summarize")
            return
        
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        print(f"\nTotal experiments: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print(f"\n{'─'*80}")
            print("Results by workload:")
            print(f"{'─'*80}")
            
            # Group by network
            from collections import defaultdict
            by_network = defaultdict(list)
            for r in successful:
                by_network[r.config.workload_name].append(r)
            
            for network, results in by_network.items():
                print(f"\n{network.upper()}:")
                for r in results:
                    print(f"  {r.config.fusion_level}/{r.config.workload_variant}, fmem_size: {r.config.fmem_size_kB}kB, wmem_size: {r.config.wmem_size_kB}kB, pe_rows: {r.config.pe_rows}, pe_cols: {r.config.pe_cols} -> "
                          f"E={r.energy_uJ:.2e}μJ, L={r.latency_cycles:.2e}cc, "
                          f"EDP={r.edp:.2e}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Thesis Experiment Runner - Sweep workloads and collect results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available workloads
  python experiment_runner.py --list
  
  # Run a single experiment
  python experiment_runner.py --workload fsrcnn --fusion single --variant L0
  
  # Sweep all workloads with default architecture
  python experiment_runner.py --sweep-workloads
  
  # Sweep only VGG16 block fusions
  python experiment_runner.py --sweep-workloads --networks vgg16 --fusion-levels block
  
  # Sweep architecture parameters for a specific workload
  python experiment_runner.py --sweep-arch --workload fsrcnn --fusion full --variant 8layer
  
  # Full sweep with custom output
  python experiment_runner.py --sweep-workloads --output my_results.csv
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--list", action="store_true",
                           help="List all available workloads")
    mode_group.add_argument("--sweep-workloads", action="store_true",
                           help="Sweep over all workloads")
    mode_group.add_argument("--sweep-arch", action="store_true",
                           help="Sweep architecture parameters for a workload")
    mode_group.add_argument("--single", action="store_true",
                           help="Run a single experiment")
    
    # Workload specification
    parser.add_argument("--workload", "-w", type=str,
                       help="Workload network name (fsrcnn, mccnn, vgg16, resnet18)")
    parser.add_argument("--fusion", "-f", type=str,
                       help="Fusion level (single, 2layer, 3layer, block, full)")
    parser.add_argument("--variant", "-v", type=str,
                       help="Workload variant name")
    
    # Sweep filters
    parser.add_argument("--networks", nargs="+", type=str,
                       help="Networks to include in sweep")
    parser.add_argument("--fusion-levels", nargs="+", type=str,
                       help="Fusion levels to include in sweep")
    
    # Architecture parameters (single values)
    parser.add_argument("--fmem-size", type=int, default=1056,
                       help="Feature memory size in KB (default: 1056)")
    parser.add_argument("--wmem-size", type=int, default=524,
                       help="Weight memory size in KB (default: 524)")
    parser.add_argument("--pe-rows", type=int, default=16,
                       help="PE array rows (default: 16)")
    parser.add_argument("--pe-cols", type=int, default=128,
                       help="PE array columns (default: 128)")
    
    # Architecture sweep ranges (for --sweep-arch mode)
    parser.add_argument("--fmem-sizes", nargs="+", type=int,
                       help="FMEM sizes to sweep in KB (default: 512 1024 2048)")
    parser.add_argument("--wmem-sizes", nargs="+", type=int,
                       help="WMEM sizes to sweep in KB (default: 256 512 1024)")
    parser.add_argument("--pe-configs", nargs="+", type=str,
                       help="PE configs to sweep as ROWSxCOLS (default: 8x64 16x128 32x256)")
    
    # Output options
    parser.add_argument("--output", "-o", type=str,
                       help="Output filename for results (CSV)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory (default: results)")
    parser.add_argument("--json", action="store_true",
                       help="Also export results as JSON")
    
    # Verbosity
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output during experiments")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress progress output")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    # List mode
    if args.list:
        print("\nAvailable workloads:")
        print("=" * 60)
        for network in runner.workload_registry.get_networks():
            print(f"\n{network.upper()}:")
            for fusion in runner.workload_registry.get_fusion_levels(network):
                variants = runner.workload_registry.list_workloads(network, fusion)
                print(f"  {fusion}:")
                for _, _, variant in variants:
                    shape, coupling, num_layers = runner.workload_registry.get_workload(
                        network, fusion, variant
                    )
                    print(f"    - {variant} ({num_layers} layers)")
        return
    
    # Create base config from args
    base_config = ExperimentConfig(
        fmem_size_kB=args.fmem_size,
        wmem_size_kB=args.wmem_size,
        pe_rows=args.pe_rows,
        pe_cols=args.pe_cols,
        verbose=args.verbose,
    )
    
    # Run experiments based on mode
    if args.sweep_workloads:
        runner.run_workload_sweep(
            networks=args.networks,
            fusion_levels=args.fusion_levels,
            arch_config=base_config,
        )
    
    elif args.sweep_arch:
        if not all([args.workload, args.fusion, args.variant]):
            print("Error: --sweep-arch requires --workload, --fusion, and --variant")
            return
        
        # Parse PE configs if provided (format: "8x64 16x128")
        pe_configs = None
        if args.pe_configs:
            pe_configs = []
            for cfg in args.pe_configs:
                rows, cols = cfg.lower().split('x')
                pe_configs.append((int(rows), int(cols)))
        
        runner.run_architecture_sweep(
            workload=(args.workload, args.fusion, args.variant),
            fmem_sizes_kb=args.fmem_sizes,
            wmem_sizes_kb=args.wmem_sizes,
            pe_configs=pe_configs,
            verbose=args.verbose,
        )
    
    elif args.single or (args.workload and args.fusion and args.variant):
        if not all([args.workload, args.fusion, args.variant]):
            print("Error: Single run requires --workload, --fusion, and --variant")
            return
        config = ExperimentConfig(
            workload_name=args.workload,
            workload_variant=args.variant,
            fusion_level=args.fusion,
            fmem_size_kB=args.fmem_size,
            wmem_size_kB=args.wmem_size,
            pe_rows=args.pe_rows,
            pe_cols=args.pe_cols,
            verbose=True,  # Always verbose for single runs
        )
        result = runner.run_single_experiment(config)
        if result.success:
            print(f"\n{'='*60}")
            print("RESULT")
            print(f"{'='*60}")
            print(f"Energy: {result.energy_uJ:.4e} μJ")
            print(f"Latency: {result.latency_cycles:.4e} cycles")
            print(f"EDP: {result.edp:.4e}")
            print(f"Utilization: {result.utilization:.2%}")
            print(f"Mapping time: {result.mapping_time_s:.2f} s")
            if result.mapping_summary:
                print(f"\nFinal Mapping:")
                print(result.mapping_summary)
        else:
            print(f"\nExperiment failed: {result.error_message}")
    
    else:
        print("No mode specified. Use --help for usage information.")
        return
    
    # Export results
    if runner.results:
        runner.export_results_csv(args.output)
        if args.json:
            runner.export_results_json()
        runner.print_summary()


if __name__ == "__main__":
    main()
