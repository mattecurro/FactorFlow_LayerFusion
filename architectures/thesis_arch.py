"""
Thesis Architecture Configuration

A parameterized architecture for layer fusion case studies.
Combines features from Eyeriss and DepFiN for systematic experiments.

Energy values are calculated dynamically using Accelergy based on memory sizes.
Bandwidth values are fixed parameters (depend on physical interface, not size).
"""

from dataclasses import dataclass
from typing import Optional
import math
import sys
import os

# Handle imports for both direct execution and module import
try:
    from computations import *
    from levels import *
    from arch import *
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from computations import *
    from levels import *
    from arch import *

"""
    Configuration class for thesis experiments.
    
    Parameters are organized by category:
    - Memory sizes (in KB)
    - PE array configuration  
    - Bandwidth (fixed, depends on physical interface)
    - Technology node
    - Workload settings
"""
@dataclass
class ThesisArchConfig:
    # === Memory Sizes (in KB) ===
    feature_memory_size_B: int = 1056 * 1024  # FMEM size (activations)
    weight_memory_size_B: int = 524 * 1024    # WMEM size (weights)
    accumulation_reg_entries: int = 16*2  # Entries per accumulation register
    
    # === Register Sizes (entries) ===
    weight_reg_entries: int = 192*2          # Weight register (3x3 kernel = 9 weights)
    input_reg_entries: int = 12*2           # Input activation register
    intermediate_output_reg_entries: int = 100*2  # Intermediate outputs between layers
    # NOTE: No output_reg_entries - AccumulationReg serves as output storage
    
    # === PE Array Configuration ===
    pe_rows: int = 16      # Number of rows (typically for output channels Z)
    pe_cols: int = 128     # Number of columns (typically for spatial X)
    
    # === Bandwidth (bytes/cycle) - FIXED based on physical interface ===
    # These do NOT change with memory size
    dram_read_bandwidth: int = 12      # DRAM read bandwidth
    dram_write_bandwidth: int = 12     # DRAM write bandwidth
    fmem_read_bandwidth: int = 132     # FMEM read bandwidth (132 bytes = 1056 bits)
    fmem_write_bandwidth: int = 128    # FMEM write bandwidth
    wmem_read_bandwidth: int = 16           # WMEM read bandwidth 
    wmem_write_bandwidth: int = 16          # WMEM write bandwidth
    
    # === Technology Parameters ===
    technology: str = "22nm"           # Technology node for Accelergy
    technology_scale: float = 0.5      # Scale factor (e.g., 0.5 for 12nm from 22nm)
    
    # === Memory Organization (for Accelergy) ===
    fmem_word_bits: int = 1056         # FMEM wordline width (132 bytes)
    fmem_banks: int = 2                # FMEM banks (from DepFiN Fig 3: "2·132·8b")
    wmem_word_bits: int = 128          # WMEM wordline width (16 bytes)
    wmem_banks: int = 4                # WMEM banks (from DepFiN Fig 3: "4·16·8b")
    
    # === Compute Parameters ===
    precision: int = 8                 # Operand precision (bits)
    accumulator_precision: int = 32    # Accumulator precision (bits)
    
    # === Workload ===
    num_fused_layers: int = 10         # Number of layers to fuse
    
    # === Tile Size Override (for case studies) ===
    # tile_size_override specifies the OUTPUT tile size (Q dimension).
    # For multi-layer fusion with strides (e.g., ResNet, VGG), input layer tile sizes
    # are automatically derived: input_tile = output_tile * cumulative_stride.
    # Example: ResNet18 full (cum_stride=16), tile_size=7 → input tile X0 = 112
    tile_size_override: Optional[int] = None  # OUTPUT tile size override (Q dimension)
    scale_bandwidth_with_tile: bool = False   # If True, scale FMEM bandwidth proportionally to tile size
    base_tile_size_for_scaling: int = 128     # Reference tile size for bandwidth scaling
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.feature_memory_size_B > 0, "FMEM size must be positive"
        assert self.weight_memory_size_B > 0, "WMEM size must be positive"
        assert self.pe_rows > 0 and self.pe_cols > 0, "PE dimensions must be positive"
        if self.tile_size_override is not None:
            assert self.tile_size_override > 0, "Tile size override must be positive"
            assert self.tile_size_override <= self.pe_cols, \
                f"Tile size override ({self.tile_size_override}) cannot exceed pe_cols ({self.pe_cols})"
    
    def get_scaled_fmem_bandwidth(self) -> tuple[int, int]:
        """
        Get FMEM bandwidth, optionally scaled by tile size.
        
        Returns:
            (read_bandwidth, write_bandwidth) in bytes/cycle
        """
        if not self.scale_bandwidth_with_tile or self.tile_size_override is None:
            return self.fmem_read_bandwidth, self.fmem_write_bandwidth
        
        # Scale bandwidth proportionally to tile size
        scale_factor = self.tile_size_override / self.base_tile_size_for_scaling
        scaled_read = max(1, int(self.fmem_read_bandwidth * scale_factor))
        scaled_write = max(1, int(self.fmem_write_bandwidth * scale_factor))
        return scaled_read, scaled_write


def get_tile_size_divisors(dim_size: int, max_tile: int) -> list[int]:
    """
    Get all divisors of dim_size that are <= max_tile, sorted descending.
    
    This is used to enumerate valid tile sizes for a dimension when sweeping.
    
    Args:
        dim_size: The dimension size (e.g., Q=960 for FSRCNN)
        max_tile: Maximum tile size (usually pe_cols, e.g., 256)
    
    Returns:
        List of valid tile sizes (divisors of dim_size <= max_tile), largest first
    
    Example:
        >>> get_tile_size_divisors(960, 256)
        [240, 192, 160, 128, 120, 96, 80, 64, 60, 48, 40, 32, 30, 24, 20, 16, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1]
    """
    divisors = []
    for d in range(1, min(dim_size, max_tile) + 1):
        if dim_size % d == 0:
            divisors.append(d)
    return sorted(divisors, reverse=True)


def get_min_tile_size_for_multilayer(shape: 'Shape', num_layers: int) -> int:
    """
    Get the minimum common tile size across all layers for multi-layer fusion.
    
    For ResNet18 with varying spatial dimensions (56, 28, 14, 7), the minimum
    common tile size is gcd(56, 28, 14, 7) = 7.
    
    Args:
        shape: The workload shape containing all dimension sizes
        num_layers: Number of fused layers
    
    Returns:
        Minimum common tile size (GCD of all spatial widths)
    """
    from math import gcd
    from functools import reduce
    
    # Collect all spatial widths (Q for output, X{i} for intermediate layers)
    widths = []
    
    # Output layer width
    q_size = shape.get('Q', 1)
    if q_size > 1:
        widths.append(q_size)
    
    # Intermediate layer widths
    for layer in range(num_layers - 1):
        x_dim = f'X{layer}'
        x_size = shape.get(x_dim, 1)
        if x_size > 1:
            widths.append(x_size)
    
    if not widths:
        return 1
    
    return reduce(gcd, widths)


def get_layer_stride(shape: 'Shape', layer_idx: int) -> tuple[int, int]:
    """
    Get the (Pstride, Qstride) for a given layer index.
    Returns (1, 1) if strides are not specified (default).
    
    Strides are stored in Shape as Pstride{i} and Qstride{i}.
    
    Args:
        shape: Multi-layer Shape containing stride info
        layer_idx: Layer index (0 = first/input layer)
    
    Returns:
        (pstride, qstride) tuple
    """
    pstride = shape.get(f'Pstride{layer_idx}', 1)
    qstride = shape.get(f'Qstride{layer_idx}', 1)
    return (pstride, qstride)


def get_cumulative_stride(shape: 'Shape', num_layers: int) -> tuple[int, int]:
    """
    Calculate the cumulative stride across all layers.
    This is the product of all individual strides.
    
    For depth-first processing: input_tile = output_tile * cumulative_stride
    (when all strides <= kernel sizes)
    
    Args:
        shape: Multi-layer Shape with stride info
        num_layers: Number of layers
        
    Returns:
        (cum_pstride, cum_qstride) tuple
    """
    cum_pstride = 1
    cum_qstride = 1
    
    for layer_idx in range(num_layers):
        pstride, qstride = get_layer_stride(shape, layer_idx)
        cum_pstride *= pstride
        cum_qstride *= qstride
    
    return (cum_pstride, cum_qstride)


def calculate_per_layer_tile_sizes(
    shape: 'Shape',
    num_layers: int,
    output_tile_h: int,
    output_tile_w: int
) -> list[tuple[int, int]]:
    """
    Calculate the required tile sizes at each layer for depth-first processing.
    Propagates backwards from output layer to input layer.
    
    Formula: new_input = output * stride (when stride <= kernel, i.e., halo is reusable)
             new_input = (output - 1) * stride + kernel (when stride > kernel)
    
    Args:
        shape: Multi-layer Shape with stride info (Pstride{i}, Qstride{i})
        num_layers: Number of layers
        output_tile_h: Desired output tile height (P dimension)
        output_tile_w: Desired output tile width (Q dimension)
    
    Returns:
        List of (height, width) tuples for each layer, from layer 0 to layer num_layers-1,
        plus the output tile at index num_layers.
        tile_sizes[i] = tile size *after* layer i (i.e., its output).
        tile_sizes[0] is the input to layer 0.
    
    Example for ResNet18 full fusion (17 layers) with output tile 7x7:
        Layer 13-16: 7x7 (no stride)
        Layer 12: 7x7 (no stride)
        ...
        Before layer 13 (stride=2): 14x14
        Before layer 9 (stride=2): 28x28
        Before layer 5 (stride=2): 56x56
        Before layer 0 (stride=2): 112x112
    """
    # Result: tile_sizes[0] = input to layer 0, tile_sizes[i] = output of layer i-1
    # tile_sizes[num_layers] = final output
    tile_sizes = [(output_tile_h, output_tile_w)]
    
    curr_h, curr_w = output_tile_h, output_tile_w
    
    # Propagate backwards from last layer to first layer
    for layer_idx in range(num_layers - 1, -1, -1):
        pstride, qstride = get_layer_stride(shape, layer_idx)
        
        # Get kernel size for this layer (for stride > kernel case)
        r = shape.get(f'R{layer_idx}', 1)
        s = shape.get(f'S{layer_idx}', 1)
        
        # Calculate input tile size for this layer
        # Formula: new_input = output * stride (when stride <= kernel)
        if pstride <= r:
            new_h = curr_h * pstride
        else:
            # stride > kernel: no halo overlap, need full receptive field
            new_h = (curr_h - 1) * pstride + r
            
        if qstride <= s:
            new_w = curr_w * qstride
        else:
            new_w = (curr_w - 1) * qstride + s
        
        tile_sizes.insert(0, (new_h, new_w))
        curr_h, curr_w = new_h, new_w
    
    return tile_sizes


def get_per_layer_tile_size_constraints(
    shape: 'Shape',
    num_layers: int,
    output_tile_h: int,
    output_tile_w: int,
    pe_cols: int
) -> dict[str, int]:
    """
    Get fanout constraints for each layer based on stride-aware tile sizes.
    
    This is used to set different tile sizes for Q, X0, X1, ... based on
    the cumulative stride up to each layer.
    
    Args:
        shape: Multi-layer Shape with stride info
        num_layers: Number of layers
        output_tile_h: Desired output tile height
        output_tile_w: Desired output tile width
        pe_cols: Maximum tile size (PE array width)
    
    Returns:
        Dict mapping dimension names to tile sizes:
        {'Q': 7, 'X15': 7, ..., 'X0': 112}
    """
    tile_sizes = calculate_per_layer_tile_sizes(
        shape, num_layers, output_tile_h, output_tile_w
    )
    
    constraints = {}
    
    # Output layer: Q gets output tile width
    constraints['Q'] = min(output_tile_w, pe_cols)
    constraints['P'] = output_tile_h  # P is not typically constrained to PE, but track it
    
    # Intermediate layers: X{i} gets input tile width to layer i+1
    # tile_sizes[i] = input to layer i = output of layer i-1
    for layer_idx in range(num_layers - 1):
        x_dim = f'X{layer_idx}'
        y_dim = f'Y{layer_idx}'
        
        # tile_sizes[layer_idx + 1] = output of layer layer_idx = input to layer layer_idx+1
        # We want the input tile to layer layer_idx, which is tile_sizes[layer_idx]
        input_h, input_w = tile_sizes[layer_idx]
        
        # Clamp to PE array width
        constraints[x_dim] = min(input_w, pe_cols)
        constraints[y_dim] = input_h
    
    return constraints


def get_valid_output_tile_sizes(
    shape: 'Shape',
    num_layers: int,
    pe_cols: int,
    max_input_tile: int = None
) -> list[int]:
    """
    Get list of valid OUTPUT tile sizes for stride-aware multi-layer fusion.
    
    A valid output tile size must satisfy:
    1. Divides the output dimension Q evenly
    2. Results in input tile sizes that divide each layer's dimension evenly
    3. Does not exceed pe_cols
    4. (Optional) Results in input tile <= max_input_tile
    
    For networks with strides (ResNet, VGG), the input tile grows by
    cumulative_stride, so valid output tiles may be limited.
    
    Args:
        shape: Multi-layer Shape with stride info
        num_layers: Number of layers  
        pe_cols: Maximum tile size (PE array width)
        max_input_tile: Optional max input tile size (e.g., for memory constraints)
    
    Returns:
        List of valid output tile sizes, sorted descending (largest first)
    
    Example:
        VGG16 full (13 layers, cum_stride=16): Only 7 is valid because:
        - Output tile 14 → input tile 224, but 112 % 224 != 0
        - Output tile 7 → input tile 112, which divides 224, 112, 56, 28, 14 ✓
    """
    q_size = shape.get('Q', pe_cols)
    cum_pstride, cum_qstride = get_cumulative_stride(shape, num_layers)
    
    # Get all divisors of Q that are <= pe_cols
    divisors = get_tile_size_divisors(q_size, pe_cols)
    
    valid_tiles = []
    for output_tile in divisors:
        # Check if input tile would exceed max_input_tile
        input_tile = output_tile * cum_qstride
        if max_input_tile is not None and input_tile > max_input_tile:
            continue
            
        # Calculate per-layer tile sizes and verify they all divide evenly
        tile_sizes = calculate_per_layer_tile_sizes(
            shape, num_layers, output_tile, output_tile
        )
        
        is_valid = True
        for layer_idx in range(num_layers):
            if layer_idx < num_layers - 1:
                x_dim = f'X{layer_idx}'
                # Check if dimension exists in shape (skip if not present)
                if x_dim not in shape:
                    continue  # Dimension not defined, skip validation
                dim_size = shape[x_dim]
            else:
                dim_size = q_size  # Output layer uses Q
            
            # tile_sizes[layer_idx] is input to layer layer_idx
            # For output layer check, we use tile_sizes[num_layers] = output
            if layer_idx < num_layers - 1:
                tile_w = tile_sizes[layer_idx][1]
            else:
                tile_w = output_tile
            
            if dim_size % tile_w != 0:
                is_valid = False
                break
        
        if is_valid:
            valid_tiles.append(output_tile)
    
    return sorted(valid_tiles, reverse=True)


"""
    Calculate energy values using Accelergy based on memory sizes.
    
    Energy depends on:
    - Memory size (depth)
    - Wordline width
    - Number of banks
    - Technology node
    
    Returns energy values in pJ/byte (scaled by technology_scale).
    """
def get_energy_values_from_accelergy(config: ThesisArchConfig) -> dict:
    # Import here to avoid circular imports and slow startup
    from architectures.arch_hw_data import (
        DepFinParamInfo, 
        aclg_energy_mem, 
        aclg_energy_mul, 
        aclg_energy_add,
        smartbuffer_registerfile,
        accelergy_estimate_energy
    )
    
    cycle_seconds = 1.075e-09  # ~930 MHz
    arguments = {"global_cycle_seconds": cycle_seconds}
    
    # === DRAM Energy ===
    # DRAM energy is based on technology type, not size
    DRAM_attributes = {
        "type": "LPDDR4",
        "width": 64,  # 64-bit bus
        "technology": config.technology,
        "cycle_seconds": cycle_seconds
    }
    dram_energy_per_access = aclg_energy_mem("DRAM", DRAM_attributes, "read", arguments)
    dram_energy_per_byte = dram_energy_per_access / 8  # 8 bytes per access
    
    # === FMEM Energy ===
    fmem_size_bits = config.feature_memory_size_B *  8
    fmem_depth = math.ceil(fmem_size_bits / config.fmem_word_bits)
    
    FMEM_attributes = {
        "n_rd_ports": 1,
        "n_wr_ports": 1,
        "n_rdwr_ports": 0,
        "depth": fmem_depth,
        "width": config.fmem_word_bits,
        "technology": config.technology,
        "cycle_seconds": cycle_seconds,
        "global_cycle_seconds": cycle_seconds,
        "n_banks": config.fmem_banks
    }
    fmem_energy_per_access = aclg_energy_mem("SRAM", FMEM_attributes, "read", arguments)
    fmem_bytes_per_access = config.fmem_word_bits // 8
    fmem_energy_per_byte = fmem_energy_per_access / fmem_bytes_per_access
    
    # === WMEM Energy ===
    wmem_size_bits = config.weight_memory_size_B *  8
    wmem_depth = math.ceil(wmem_size_bits / config.wmem_word_bits)
    
    WMEM_attributes = {
        "n_rd_ports": 1,
        "n_wr_ports": 1,
        "n_rdwr_ports": 0,
        "depth": wmem_depth,
        "width": config.wmem_word_bits,
        "technology": config.technology,
        "cycle_seconds": cycle_seconds,
        "global_cycle_seconds": cycle_seconds,
        "n_banks": config.wmem_banks
    }
    wmem_energy_per_access = aclg_energy_mem("SRAM", WMEM_attributes, "read", arguments)
    wmem_bytes_per_access = config.wmem_word_bits // 8
    wmem_energy_per_byte = wmem_energy_per_access / wmem_bytes_per_access
    
    # === Accumulation Register Energy ===
    acc_depth = config.accumulation_reg_entries
    acc_word_bits = config.accumulator_precision
    acc_energy_per_access = smartbuffer_registerfile(
        acc_depth, acc_word_bits, config.precision,
        cycle_seconds, config.technology, "read"
    )
    acc_bytes_per_access = acc_word_bits // 8
    acc_energy_per_byte = acc_energy_per_access / acc_bytes_per_access
    
    # === Weight Register Energy ===
    wreg_depth = max(1, config.weight_reg_entries)
    wreg_word_bits = config.precision  # 8-bit weights
    wreg_energy_per_access = smartbuffer_registerfile(
        wreg_depth, wreg_word_bits, config.precision,
        cycle_seconds, config.technology, "read"
    )
    wreg_bytes_per_access = max(1, wreg_word_bits // 8)
    wreg_energy_per_byte = wreg_energy_per_access / wreg_bytes_per_access
    
    # === Input Register Energy ===
    inreg_depth = max(1, config.input_reg_entries)
    inreg_word_bits = config.precision  # 8-bit activations
    inreg_energy_per_access = smartbuffer_registerfile(
        inreg_depth, inreg_word_bits, config.precision,
        cycle_seconds, config.technology, "read"
    )
    inreg_bytes_per_access = max(1, inreg_word_bits // 8)
    inreg_energy_per_byte = inreg_energy_per_access / inreg_bytes_per_access
    
    # === Intermediate Output Register Energy (between fused layers) ===
    intoutreg_depth = max(1, config.intermediate_output_reg_entries)
    intoutreg_word_bits = config.precision  # 8-bit (quantized intermediate)
    intoutreg_energy_per_access = smartbuffer_registerfile(
        intoutreg_depth, intoutreg_word_bits, config.precision,
        cycle_seconds, config.technology, "read"
    )
    intoutreg_bytes_per_access = max(1, intoutreg_word_bits // 8)
    intoutreg_energy_per_byte = intoutreg_energy_per_access / intoutreg_bytes_per_access
    
    # NOTE: No separate OutputRegister - AccumulationReg serves as output storage
    
    # === Compute Energy ===
    type_multiplier = "aladdin_multiplier"
    width_multiplier = 2 * config.precision  # 8x8 -> 16-bit result
    multiplier_energy = aclg_energy_mul(
        type_multiplier, width_multiplier, config.precision,
        config.technology, "read", arguments
    )
    
    type_adder = "aladdin_adder"
    adder_energy = aclg_energy_add(
        type_adder, config.accumulator_precision,
        config.technology, "read", arguments
    )
    
    fma_energy = multiplier_energy + adder_energy
    
    # Apply technology scaling
    scale = config.technology_scale
    
    return {
        'dram_energy_per_byte': dram_energy_per_byte * scale,
        'fmem_energy_per_byte': fmem_energy_per_byte * scale,
        'wmem_energy_per_byte': wmem_energy_per_byte * scale,
        'accreg_energy_per_byte': acc_energy_per_byte * scale,
        'wreg_energy_per_byte': wreg_energy_per_byte * scale,
        'inreg_energy_per_byte': inreg_energy_per_byte * scale,
        'intoutreg_energy_per_byte': intoutreg_energy_per_byte * scale,
        # NOTE: No outreg - AccumulationReg serves as output storage
        'compute_energy_per_mac': fma_energy * scale,
        # Raw values (before scaling) for reference
        'raw': {
            'dram': dram_energy_per_byte,
            'fmem': fmem_energy_per_byte,
            'wmem': wmem_energy_per_byte,
            'accreg': acc_energy_per_byte,
            'wreg': wreg_energy_per_byte,
            'inreg': inreg_energy_per_byte,
            'intoutreg': intoutreg_energy_per_byte,
            'compute': fma_energy,
        },
        # Configuration used
        'config': {
            'fmem_depth': fmem_depth,
            'wmem_depth': wmem_depth,
            'technology': config.technology,
            'scale': scale,
        }
    }


# =============================================================================
# GENERIC N-LAYER ARCHITECTURE FACTORY
# =============================================================================
"""
Create an architecture for N fused layers.

This is the main factory function that creates architectures for any number
of fused layers (1, 2, 3, 4, 8, 10, 13, 17, etc.).

Args:
    config: ThesisArchConfig with memory sizes, PE configuration, etc.
    coupling: Coupling for the workload (determines layer count if num_layers not given)
    shape: Shape with dimension sizes (used for computing factor constraints)
    num_layers: Number of fused layers (if None, inferred from coupling)
    use_accelergy_energy: If True, calculate energy from Accelergy
    custom_energy: Optional dict with custom energy values to override

Returns:
    Arch object configured for the experiment

Architecture Structure (matches arch_depfin_10layers_F1S):
    - DRAM (bypasses int_in, int_out)
    - FeatureMemory (FMEM) - bypasses w, NO factor constraints
    - WeightMemory (WMEM) - bypasses in/int_in/int_out/out, ALL weight iterations
    - FanoutLevels (2 per layer: SACols + SARows)
    - WeightRegister - bypasses in/int_in/int_out/out
    - AccumulationIntermediateOutputRegister (for multi-layer) - bypasses in/w/out/int_in
    - AccumulationOutputRegister - bypasses in/w/int_in/int_out
    - Compute
"""
def create_thesis_architecture(
    config: ThesisArchConfig,
    coupling: Coupling,
    shape: Shape = None,
    num_layers: int = None,
    use_accelergy_energy: bool = True,
    custom_energy: Optional[dict] = None
) -> Arch:
    # Infer number of layers from coupling if not provided
    if num_layers is None:
        num_layers = coupling.getNumLayers()
    
    # Get energy values
    if use_accelergy_energy:
        energy = get_energy_values_from_accelergy(config)
        dram_energy = energy['dram_energy_per_byte']
        fmem_energy = energy['fmem_energy_per_byte']
        wmem_energy = energy['wmem_energy_per_byte']
        accreg_energy = energy['accreg_energy_per_byte']
        compute_energy = energy['compute_energy_per_mac']
    else:
        # Use default/custom values
        dram_energy = custom_energy.get('dram', 64.0) if custom_energy else 64.0
        fmem_energy = custom_energy.get('fmem', 1.61) if custom_energy else 1.61
        wmem_energy = custom_energy.get('wmem', 1.58) if custom_energy else 1.58
        accreg_energy = custom_energy.get('accreg', 0.40) if custom_energy else 0.40
        compute_energy = custom_energy.get('compute', 0.45) if custom_energy else 0.45
    
    # Memory sizes
    fmem_size = config.feature_memory_size_B 
    wmem_size = config.weight_memory_size_B
    # PE array dimensions
    pe_cols = config.pe_cols
    pe_rows = config.pe_rows
    
    # Build architecture based on number of layers
    if num_layers == 1:
        return _create_single_layer_architecture(
            config, coupling, shape, dram_energy, fmem_energy, wmem_energy, 
            accreg_energy, compute_energy, fmem_size, wmem_size, pe_cols, pe_rows
        )
    else:
        return _create_multi_layer_architecture(
            config, coupling, shape, num_layers, dram_energy, fmem_energy, wmem_energy,
            accreg_energy, compute_energy, fmem_size, wmem_size, pe_cols, pe_rows
        )

"""
Create architecture for single-layer (unfused) workload.

Uses standard convolution dimensions: N, M, P, Q, C, R, S
Matches arch_depfin_10layers_F1S structure for bypasses and constraints.
"""
def _create_single_layer_architecture(
    config: ThesisArchConfig,
    coupling: Coupling,
    shape: Shape,
    dram_energy: float,
    fmem_energy: float,
    wmem_energy: float,
    accreg_energy: float,
    compute_energy: float,
    fmem_size: int,
    wmem_size: int,
    pe_cols: int,
    pe_rows: int
) -> Arch:    
    # Helper function: compute GCD constraint (highest common divisor <= mesh)
    def gcd_constraint(dim_size: int, mesh: int) -> int:
        """Find highest factor of dim_size that is <= mesh."""
        if dim_size <= mesh:
            return dim_size
        # Find the highest divisor of dim_size that is <= mesh
        for d in range(mesh, 0, -1):
            if dim_size % d == 0:
                return d
        return 1
    
    # Get all dimensions from coupling (dims is a list, not a dict)
    all_dims = list(coupling.dims) if hasattr(coupling, 'dims') else ['M', 'P', 'Q', 'C', 'R', 'S']
    
    # Get dimension sizes from shape (if available)
    if shape is not None:
        q_size = shape.get('Q', pe_cols)
        p_size = shape.get('P', 1)
        m_size = shape.get('M', pe_rows)
        c_size = shape.get('C', 1)
        r_size = shape.get('R', 3)
        s_size = shape.get('S', 3)
    else:
        # Default values if shape not provided
        q_size, p_size, m_size, c_size, r_size, s_size = pe_cols, 1, pe_rows, 1, 3, 3
    
    # Compute factor constraints for FanoutLevels
    q_fanout = gcd_constraint(q_size, pe_cols)
    m_fanout = gcd_constraint(m_size, pe_rows)
    
    # DRAM constraints: remaining Q iterations not in fanout, plus all P
    q_dram = q_size // q_fanout
    p_dram = p_size
    
    # WMEM constraints: all weight dimensions (filter to those in coupling)
    wmem_dims = [d for d in ['M', 'C', 'R', 'S'] if d in all_dims]
    wmem_factors = {}
    if 'M' in all_dims:
        wmem_factors['M'] = m_size // m_fanout
    if 'C' in all_dims:
        wmem_factors['C'] = c_size
    if 'R' in all_dims:
        wmem_factors['R'] = r_size
    if 'S' in all_dims:
        wmem_factors['S'] = s_size
    
    levels = [
        # === DRAM ===
        MemLevel(
            name="DRAM",
            size=2**64-1,
            read_value_access_energy=dram_energy,
            write_value_access_energy=dram_energy,
            read_bandwidth=config.dram_read_bandwidth,
            write_bandwidth=config.dram_write_bandwidth,
            bypasses=[],  # Single layer: no intermediates to bypass
            dataflow_constraints=all_dims,  # Use coupling's dimensions
            factors_constraints={'Q': q_dram, 'P': p_dram}
        ),
        
        # === Feature Memory (FMEM) - NO factor constraints ===
        MemLevel(
            name="FeatureMemory",
            size=fmem_size,
            value_access_energy=fmem_energy,
            read_bandwidth=config.fmem_read_bandwidth,
            write_bandwidth=config.fmem_write_bandwidth,
            bypasses=['w'],
            dataflow_constraints=[],
            factors_constraints={}  
        ),
        
        # === Weight Memory (WMEM) - ALL weight iterations ===
        MemLevel(
            name="WeightMemory",
            size=wmem_size,
            value_access_energy=wmem_energy,
            read_bandwidth=config.wmem_read_bandwidth,
            write_bandwidth=config.wmem_write_bandwidth,
            bypasses=['in', 'out'],
            dataflow_constraints=wmem_dims,
            factors_constraints=wmem_factors
        ),
        
        # === Spatial Levels (PE Array) ===
        FanoutLevel(
            name="SACols",
            mesh=pe_cols,
            dims=['Q'],
            factors_constraints={'Q': q_fanout},
            spatial_reduction_support=True
        ),
        FanoutLevel(
            name="SARows",
            mesh=pe_rows,
            dims=['M'],
            factors_constraints={'M': m_fanout}
        ),
        
        # === Weight Register (bypasses in, out - same as WMEM for single layer) ===
        MemLevel(
            name="WeightRegister",
            size=config.weight_reg_entries,
            value_access_energy=wmem_energy,
            read_bandwidth=1,
            write_bandwidth=1,
            bypasses=['in', 'out'],
            dataflow_constraints=['C', 'R', 'S'] if 'C' in all_dims else wmem_dims,
            factors_constraints={}
        ),
        
        # === Output Register (Accumulation) ===
        MemLevel(
            name="OutputRegister",
            size=config.accumulation_reg_entries,
            value_access_energy=accreg_energy,
            read_bandwidth=1,
            write_bandwidth=1,
            bypasses=['in', 'w'],
            dataflow_constraints=['M', 'Q', 'P'] if 'M' in all_dims else [],
            factors_constraints={'M': 1, 'Q': 1, 'P': 1}
        ),
        
        # === Compute Level ===
        ComputeLevel(
            name="Compute",
            mesh=1,
            compute_energy=compute_energy,
            leakage_energy=0.001,
            cycles=1,
            factors_constraints={}
        )
    ]
    
    return Arch(levels, coupling=coupling,
                name=f"Thesis_1Layer_{config.feature_memory_size_B}B_FMEM_{pe_rows}x{pe_cols}_PEs")

"""
Create architecture for multi-layer (fused) workload.

Matches arch_depfin_10layers_F1S structure:
- DRAM: bypasses int_in/int_out, factors for Q/P and all X/Y
- FeatureMemory: bypasses w, NO factor constraints
- WeightMemory: bypasses in/int_in/int_out/out, ALL weight iterations
- FanoutLevels: exact factor constraints (GCD with shape, <= mesh)
- WeightRegister: bypasses in/int_in/int_out/out
- AccumulationIntermediateOutputRegister: bypasses in/w/out/int_in
- AccumulationOutputRegister: bypasses in/w/int_in/int_out
"""
def _create_multi_layer_architecture(
    config: ThesisArchConfig,
    coupling: Coupling,
    shape: Shape,
    num_layers: int,
    dram_energy: float,
    fmem_energy: float,
    wmem_energy: float,
    accreg_energy: float,
    compute_energy: float,
    fmem_size: int,
    wmem_size: int,
    pe_cols: int,
    pe_rows: int
) -> Arch:
    
    # Pre-compute per-layer tile sizes based on stride info (if tile_size_override is set)
    # This allows different layers to have different tile sizes based on stride
    per_layer_tile_constraints = None
    if config.tile_size_override is not None and shape is not None:
        # Use the tile_size_override as the OUTPUT tile size, then propagate backwards
        per_layer_tile_constraints = get_per_layer_tile_size_constraints(
            shape, num_layers, 
            output_tile_h=config.tile_size_override,
            output_tile_w=config.tile_size_override,
            pe_cols=pe_cols
        )
        
        # Debug: show per-layer tile sizes if they differ
        cum_pstride, cum_qstride = get_cumulative_stride(shape, num_layers)
        if cum_pstride > 1 or cum_qstride > 1:
            print(f"[Stride-Aware Tiling] output_tile={config.tile_size_override}, "
                  f"cumulative_stride={cum_pstride}x{cum_qstride}")
            input_tile = per_layer_tile_constraints.get('X0', config.tile_size_override)
            print(f"  Input tile (X0): {input_tile}, Output tile (Q): {config.tile_size_override}")
    
    # Helper function: compute GCD constraint (highest common divisor <= mesh)
    def gcd_constraint(dim_size: int, mesh: int, is_spatial_width: bool = False, dim_name: str = None) -> int:
        """
        Find highest factor of dim_size that is <= mesh.
        
        If tile_size_override is set and this is a spatial width dimension (Q, X0, X1, etc.),
        use the stride-aware per-layer tile size instead of computing GCD.
        
        Args:
            dim_size: The dimension size
            mesh: Maximum tile size (e.g., pe_cols)
            is_spatial_width: If True, this is a width dimension that can be overridden
            dim_name: The dimension name (Q, X0, X1, etc.) for looking up stride-aware sizes
        """
        # Use stride-aware per-layer tile size for spatial width dimensions if available
        if is_spatial_width and per_layer_tile_constraints is not None and dim_name is not None:
            tile = per_layer_tile_constraints.get(dim_name, config.tile_size_override)
            # Verify the tile size is valid (divides dim_size and <= mesh)
            if dim_size % tile == 0 and tile <= mesh:
                return tile
            else:
                print(f"Warning: stride-aware tile {tile} for {dim_name} is not valid for dim_size {dim_size} "
                      f"(must divide evenly and be <= {mesh}). Using GCD fallback.")
        
        # Fallback: use simple tile_size_override if set
        if is_spatial_width and config.tile_size_override is not None and per_layer_tile_constraints is None:
            tile = config.tile_size_override
            if dim_size % tile == 0 and tile <= mesh:
                return tile
            else:
                print(f"Warning: tile_size_override {tile} is not valid for dim_size {dim_size} "
                      f"(must divide evenly and be <= {mesh}). Using GCD fallback.")
        
        # Default: find highest divisor <= mesh
        if dim_size <= mesh:
            return dim_size
        for d in range(mesh, 0, -1):
            if dim_size % d == 0:
                return d
        return 1
    
    # Last layer index (output layer)
    last_layer = num_layers - 1
    
    # All N-layer couplings now use indexed dimensions: X0, Y0, Z0, X1, Y1, Z1, etc.
    def get_dim_name(base, layer_idx):
        """Get dimension name with layer index."""
        return f'{base}{layer_idx}'
    
    # === Build factor constraints based on shape ===
    # DRAM: Q, P for output layer + all Xi, Yi for intermediate layers
    dram_dataflow = ['Q', 'P']
    dram_factors = {}
    
    # WMEM: all weight dimensions for all layers
    wmem_dataflow = []
    wmem_factors = {}
    
    # Fanout constraints per layer
    fanout_constraints = {}
    
    if shape is not None:
        # Output layer dimensions
        q_size = shape.get('Q', pe_cols)
        p_size = shape.get('P', 1)
        
        # Fanout for output layer (Q in cols, Z{last_layer} in rows)
        # Q is a spatial width dimension - can be overridden
        q_fanout = gcd_constraint(q_size, pe_cols, is_spatial_width=True, dim_name='Q')
        # Output Z dimension is always Z{last_layer}
        z_last_dim = f'Z{last_layer}'
        z_last_size = shape.get(z_last_dim, pe_rows)
        z_last_fanout = gcd_constraint(z_last_size, pe_rows, is_spatial_width=False, dim_name=z_last_dim)
        fanout_constraints['Q'] = q_fanout
        fanout_constraints[z_last_dim] = z_last_fanout
        
        # DRAM: remaining Q iterations + all P
        dram_factors['Q'] = q_size // q_fanout
        dram_factors['P'] = p_size
        
        # Intermediate layers (0 to last_layer-1)
        for layer in range(last_layer - 1, -1, -1):
            x_dim = get_dim_name('X', layer)
            y_dim = get_dim_name('Y', layer)
            z_dim = get_dim_name('Z', layer)
            
            dram_dataflow.extend([x_dim, y_dim])
            
            # Get dimension sizes
            x_size = shape.get(x_dim, pe_cols)
            y_size = shape.get(y_dim, 1)
            z_size = shape.get(z_dim, pe_rows)
            
            # Fanout constraints - X is a spatial width dimension, Z is not
            # Use stride-aware tile sizes via dim_name lookup
            x_fanout = gcd_constraint(x_size, pe_cols, is_spatial_width=True, dim_name=x_dim)
            z_fanout = gcd_constraint(z_size, pe_rows, is_spatial_width=False, dim_name=z_dim)
            fanout_constraints[x_dim] = x_fanout
            fanout_constraints[z_dim] = z_fanout
            
            # DRAM: remaining X iterations + all Y
            dram_factors[x_dim] = x_size // x_fanout
            dram_factors[y_dim] = y_size
        
        # WMEM: all weight dimensions for all layers
        for layer in range(last_layer, -1, -1):
            z_dim = f'Z{layer}'
            c_dim = f'C{layer}'
            r_dim = f'R{layer}'
            s_dim = f'S{layer}'
            
            wmem_dataflow.extend([z_dim, c_dim, r_dim, s_dim])
            
            # Get all weight dimension sizes
            z_size = shape.get(z_dim, 1)
            c_size = shape.get(c_dim, 1)
            r_size = shape.get(r_dim, 3)
            s_size = shape.get(s_dim, 3)
            
            # All weight iterations go to WMEM
            # Z is already partly in fanout, so WMEM gets the rest
            z_fanout = fanout_constraints.get(z_dim, 1)
            wmem_factors[z_dim] = z_size // z_fanout if z_size // z_fanout > 0 else 1
            wmem_factors[c_dim] = c_size
            wmem_factors[r_dim] = r_size
            wmem_factors[s_dim] = s_size
    else:
        print("Warning: Shape not provided, using default factor constraints.")
        # Default fallback when shape not provided
        for layer in range(last_layer - 1, -1, -1):
            x_dim = get_dim_name('X', layer)
            y_dim = get_dim_name('Y', layer)
            dram_dataflow.extend([x_dim, y_dim])
        for layer in range(last_layer, -1, -1):
            z_dim = f'Z{layer}'
            wmem_dataflow.extend([z_dim, f'C{layer}', f'R{layer}', f'S{layer}'])
    
    # === Weight register dataflow ===
    wreg_dataflow = []
    for layer in range(last_layer, -1, -1):
        wreg_dataflow.extend([f'C{layer}', f'R{layer}', f'S{layer}'])
    
    # === Intermediate output register constraints ===
    int_out_dataflow = []
    int_out_factors = {}
    for layer in range(last_layer - 1, -1, -1):
        z_dim = get_dim_name('Z', layer)
        y_dim = get_dim_name('Y', layer)
        x_dim = get_dim_name('X', layer)
        int_out_dataflow.extend([z_dim, y_dim, x_dim])
        int_out_factors[z_dim] = 1
        int_out_factors[y_dim] = 1
        int_out_factors[x_dim] = 1
    # Output layer Z dimension
    int_out_factors[f'Z{last_layer}'] = 1

    # Get (potentially scaled) FMEM bandwidth
    fmem_read_bw, fmem_write_bw = config.get_scaled_fmem_bandwidth()
    if config.tile_size_override is not None:
        print(f"[Tile Size Override] tile_size={config.tile_size_override}, "
              f"scale_bandwidth={config.scale_bandwidth_with_tile}, "
              f"fmem_bw=({fmem_read_bw}, {fmem_write_bw})")

    levels = [
        # === DRAM (bypasses int_in, int_out) ===
        MemLevel(
            name="DRAM",
            size=2**64-1,
            read_value_access_energy=dram_energy,
            write_value_access_energy=dram_energy,
            read_bandwidth=config.dram_read_bandwidth,
            write_bandwidth=config.dram_write_bandwidth,
            bypasses=['int_in', 'int_out'],
            dataflow_constraints=dram_dataflow,
            factors_constraints=dram_factors
        ),
        
        # === Feature Memory (FMEM) - bypasses w, NO factor constraints ===
        MemLevel(
            name="FeatureMemory",
            size=fmem_size,
            value_access_energy=fmem_energy,
            read_bandwidth=fmem_read_bw,
            write_bandwidth=fmem_write_bw,
            bypasses=['w'],
            dataflow_constraints=[],
            factors_constraints={}  # NO constraints as per arch_depfin_10layers_F1S
        ),
        
        # === Weight Memory (WMEM) - bypasses in/int_in/int_out/out, ALL weight iterations ===
        MemLevel(
            name="WeightMemory",
            size=wmem_size,
            value_access_energy=wmem_energy,
            read_bandwidth=config.wmem_read_bandwidth,
            write_bandwidth=config.wmem_write_bandwidth,
            bypasses=['in', 'int_in', 'int_out', 'out'],
            dataflow_constraints=wmem_dataflow,
            factors_constraints=wmem_factors
        ),
    ]
    
    # === Spatial Levels (PE Array) for each layer ===
    # Output layer uses Q for columns
    q_fanout = fanout_constraints.get('Q', pe_cols)
    z_last_dim = f'Z{last_layer}'
    z_last_fanout = fanout_constraints.get(z_last_dim, pe_rows)
    
    levels.extend([
        FanoutLevel(
            name=f"SACols_{last_layer}",
            mesh=pe_cols,
            dims=['Q'],
            factors_constraints={'Q': q_fanout}
        ),
        FanoutLevel(
            name=f"SARows_{last_layer}",
            mesh=pe_rows,
            dims=[z_last_dim],
            factors_constraints={z_last_dim: z_last_fanout}
        ),
    ])
    
    # Intermediate and input layers use Xi for columns
    for layer in range(last_layer - 1, -1, -1):
        x_dim = get_dim_name('X', layer)
        z_dim = get_dim_name('Z', layer)
        x_fanout = fanout_constraints.get(x_dim, pe_cols)
        z_fanout = fanout_constraints.get(z_dim, pe_rows)
        
        levels.extend([
            FanoutLevel(
                name=f"SACols_{layer}",
                mesh=pe_cols,
                dims=[x_dim],
                factors_constraints={x_dim: x_fanout}
            ),
            FanoutLevel(
                name=f"SARows_{layer}",
                mesh=pe_rows,
                dims=[z_dim],
                factors_constraints={z_dim: z_fanout}
            ),
        ])
    
    # === Weight Register (bypasses in/int_in/int_out/out) ===
    levels.append(
        MemLevel(
            name="WeightRegister",
            size=config.weight_reg_entries,
            value_access_energy=wmem_energy,
            read_bandwidth=1,
            write_bandwidth=1,
            bypasses=['in', 'int_in', 'int_out', 'out'],
            dataflow_constraints=wreg_dataflow,
            factors_constraints={}
        )
    )
    
    # === Accumulation Intermediate Output Register (bypasses in/w/out/int_in) ===
    levels.append(
        MemLevel(
            name="AccumulationIntermediateOutputRegister",
            size=config.accumulation_reg_entries,
            value_access_energy=accreg_energy,
            read_bandwidth=1,
            write_bandwidth=1,
            bypasses=['in', 'w', 'out', 'int_in'],
            dataflow_constraints=int_out_dataflow,
            factors_constraints=int_out_factors
        )
    )
    
    # === Accumulation Output Register (bypasses in/w/int_in/int_out) ===
    levels.append(
        MemLevel(
            name="AccumulationOutputRegister",
            size=1,
            value_access_energy=accreg_energy,
            read_bandwidth=1,
            write_bandwidth=1,
            bypasses=['in', 'w', 'int_in', 'int_out'],
            dataflow_constraints=['Q', 'P', z_last_dim],
            factors_constraints={'Q': 1, 'P': 1, z_last_dim: 1}
        )
    )
    
    # === Compute Level ===
    levels.append(
        ComputeLevel(
            name="Compute",
            mesh=1,
            compute_energy=compute_energy,
            leakage_energy=0.001,
            cycles=1,
            factors_constraints={}
        )
    )
    
    return Arch(levels, coupling=coupling,
                name=f"Thesis_{num_layers}Layers_{config.feature_memory_size_B}KB_FMEM_{pe_rows}x{pe_cols}_PEs")


# =============================================================================
# EYERISS-LIKE ARCHITECTURE CONFIGURATION
# =============================================================================
"""
Eyeriss-like Architecture for Layer Fusion

A parameterized architecture based on Eyeriss for layer fusion case studies.
Key differences from DepFiN-based thesis_arch:
- Unified GlobalBuffer instead of separate FMEM + WMEM
- Eyeriss-style spatial processing (SACols/SARows)
- IntermediateRegister for partial sums of intermediate outputs
"""

@dataclass
class EyerissArchConfig:
    """
    Configuration class for Eyeriss-like architecture experiments.
    
    Uses a unified GlobalBuffer instead of separate FMEM/WMEM.
    """
    # === Global Buffer Size (in KB) ===
    global_buffer_size_B: int = 131072  # 128KB default (16384*8 entries from arch_eyeriss_conv)
    
    # === Register Sizes (entries) ===
    input_reg_entries: int = 50              # Input register (increased for fusion)
    weight_reg_entries: int = 15        # Weight register (192*2 from Eyeriss)
    intermediate_out_reg_entries: int = 32*10  # Intermediate output register (for layer fusion)
    output_reg_entries: int = 32          # Output register (16*2 from Eyeriss)
    
    # === PE Array Configuration (Eyeriss: 14×12 = 168 PEs) ===
    pe_cols: int = 14          # Number of columns (SACols mesh)
    pe_rows: int = 12          # Number of rows (SARows mesh)
    
    # === Bandwidth (bytes/cycle) ===
    dram_bandwidth: int = 8            # DRAM bandwidth
    global_buffer_bandwidth: int = 32  # GlobalBuffer bandwidth
    register_bandwidth: int = 4        # Register bandwidth
    
    # === Technology Parameters ===
    technology: str = "22nm"
    technology_scale: float = 0.5  # Scale factor (e.g., 0.5 for 12nm from 22nm)
    
    # === Workload ===
    num_fused_layers: int = 1
    
    # === Tile Size Override ===
    tile_size_override: Optional[int] = None
    scale_bandwidth_with_tile: bool = False
    base_tile_size_for_scaling: int = 128
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.global_buffer_size_B > 0, "GlobalBuffer size must be positive"
        assert self.pe_rows > 0 and self.pe_cols > 0, "PE dimensions must be positive"
        if self.tile_size_override is not None:
            assert self.tile_size_override > 0, "Tile size override must be positive"
            # tile_size_override defines the output tile size for DRAM depth-first
            # It can be larger or smaller than PE mesh dimensions
    
    def get_scaled_gb_bandwidth(self) -> int:
        """
        Get GlobalBuffer bandwidth, optionally scaled by tile size.
        
        Returns:
            bandwidth in bytes/cycle
        """
        if not self.scale_bandwidth_with_tile or self.tile_size_override is None:
            return self.global_buffer_bandwidth
        
        # Scale bandwidth proportionally to tile size
        scale_factor = self.tile_size_override / self.base_tile_size_for_scaling
        scaled_bw = max(1, int(self.global_buffer_bandwidth * scale_factor))
        return scaled_bw


def get_eyeriss_energy_values(config: EyerissArchConfig) -> dict:
    """
    Get energy values for Eyeriss-like architecture using Accelergy.
    
    Derives energy-per-access from actual memory sizes using the same Accelergy
    estimators used by DepFiN (smartbuffer_registerfile for registers, 
    aclg_energy_mem/SRAM for GlobalBuffer, aclg_energy_mem/DRAM for DRAM).
    
    This makes Eyeriss energy size-dependent: larger registers/buffers have
    higher energy per access, enabling fair comparisons when register sizes
    differ between fusion levels.
    
    Falls back to hardcoded arch_eyeriss_conv values if Accelergy is unavailable.
    """
    try:
        print("------ loading Accelergy for Eyeriss ------")
        from architectures.arch_hw_data import (
            aclg_energy_mem,
            aclg_energy_mul,
            aclg_energy_add,
            smartbuffer_registerfile,
            accelergy_estimate_energy
        )
    except (ImportError, Exception) as e:
        print(f"WARNING: Accelergy not available for Eyeriss ({e}), using hardcoded energy values")
        return _get_eyeriss_energy_values_hardcoded(config)
    
    try:
        cycle_seconds = 1.075e-09  # ~930 MHz (same as DepFiN)
        arguments = {"global_cycle_seconds": cycle_seconds}
        precision = 8  # 8-bit data
        
        # === DRAM Energy ===
        DRAM_attributes = {
            "type": "LPDDR4",
            "width": 64,
            "technology": config.technology,
            "cycle_seconds": cycle_seconds
        }
        dram_energy_per_access = aclg_energy_mem("DRAM", DRAM_attributes, "read", arguments)
        # Eyeriss uses per-operand energy (not per-byte), DRAM access = 8 bytes
        dram_energy_per_operand = dram_energy_per_access / 8
        
        # === GlobalBuffer Energy (SRAM) ===
        gb_size_bits = config.global_buffer_size_B
        gb_word_bits = 64  # 64-bit bus (same as arch_eyeriss_conv bandwidth=32 at 16b)
        gb_depth = math.ceil(gb_size_bits / gb_word_bits)
        
        GB_attributes = {
            "n_rd_ports": 1,
            "n_wr_ports": 1,
            "n_rdwr_ports": 0,
            "depth": gb_depth,
            "width": gb_word_bits,
            "technology": config.technology,
            "cycle_seconds": cycle_seconds,
            "global_cycle_seconds": cycle_seconds,
            "n_banks": 1
        }
        gb_energy_per_access = aclg_energy_mem("SRAM", GB_attributes, "read", arguments)
        # Convert to per-operand (1 operand = precision/8 bytes = 1 byte for 8-bit)
        gb_bytes_per_access = gb_word_bits // 8
        gb_energy_per_operand = gb_energy_per_access / gb_bytes_per_access
        
        # === Weight Register Energy ===
        wreg_depth = max(1, config.weight_reg_entries)
        wreg_word_bits = precision
        wreg_energy_per_access = smartbuffer_registerfile(
            wreg_depth, wreg_word_bits, precision,
            cycle_seconds, config.technology, "read"
        )
        wreg_energy_per_operand = wreg_energy_per_access / max(1, wreg_word_bits // 8)
        
        # === Input Register Energy ===
        inreg_depth = max(1, config.input_reg_entries)
        inreg_word_bits = precision
        inreg_energy_per_access = smartbuffer_registerfile(
            inreg_depth, inreg_word_bits, precision,
            cycle_seconds, config.technology, "read"
        )
        inreg_energy_per_operand = inreg_energy_per_access / max(1, inreg_word_bits // 8)
        
        # === Output Register Energy ===
        outreg_depth = max(1, config.output_reg_entries)
        outreg_word_bits = precision
        outreg_energy_per_access = smartbuffer_registerfile(
            outreg_depth, outreg_word_bits, precision,
            cycle_seconds, config.technology, "read"
        )
        outreg_energy_per_operand = outreg_energy_per_access / max(1, outreg_word_bits // 8)
        
        # === Intermediate Register Energy (for layer fusion) ===
        intreg_depth = max(1, config.intermediate_out_reg_entries)
        intreg_word_bits = precision
        intreg_energy_per_access = smartbuffer_registerfile(
            intreg_depth, intreg_word_bits, precision,
            cycle_seconds, config.technology, "read"
        )
        intreg_energy_per_operand = intreg_energy_per_access / max(1, intreg_word_bits // 8)
        
        # === Compute Energy (FMA) ===
        type_multiplier = "aladdin_multiplier"
        width_multiplier = 2 * precision  # 8x8 -> 16-bit result
        multiplier_energy = aclg_energy_mul(
            type_multiplier, width_multiplier, precision,
            config.technology, "read", arguments
        )
        type_adder = "aladdin_adder"
        adder_energy = aclg_energy_add(
            type_adder, 2 * precision,  # 16-bit accumulator
            config.technology, "read", arguments
        )
        compute_energy = multiplier_energy + adder_energy
        
        # Apply technology scaling
        scale = config.technology_scale
        
        return {
            'dram_energy': dram_energy_per_operand * scale,
            'global_buffer_energy': gb_energy_per_operand * scale,
            'input_reg_energy': inreg_energy_per_operand * scale,
            'weight_reg_energy': wreg_energy_per_operand * scale,
            'output_reg_energy': outreg_energy_per_operand * scale,
            'intermediate_reg_energy': intreg_energy_per_operand * scale,
            'compute_energy': compute_energy * scale,
        }
    except Exception as e:
        print(f"WARNING: Accelergy estimation failed for Eyeriss ({e}), using hardcoded values")
        return _get_eyeriss_energy_values_hardcoded(config)


def _get_eyeriss_energy_values_hardcoded(config: EyerissArchConfig) -> dict:
    """
    Fallback: hardcoded energy values from arch_eyeriss_conv.
    Used when Accelergy is not available.
    """
    base_energy = {
        'dram': 64.00,
        'global_buffer': 2.02,
        'input_reg': 0.69,
        'weight_reg': 1.97,
        'output_reg': 1.34,
        'intermediate_reg': 1.34,
        'compute': 0.21,
    }
    scale = config.technology_scale
    return {
        'dram_energy': base_energy['dram'] * scale,
        'global_buffer_energy': base_energy['global_buffer'] * scale,
        'input_reg_energy': base_energy['input_reg'] * scale,
        'weight_reg_energy': base_energy['weight_reg'] * scale,
        'output_reg_energy': base_energy['output_reg'] * scale,
        'intermediate_reg_energy': base_energy['intermediate_reg'] * scale,
        'compute_energy': base_energy['compute'] * scale,
    }


def create_eyeriss_architecture(
    config: EyerissArchConfig,
    coupling: Coupling = None,
    shape: Shape = None,
    num_layers: int = None,
    use_default_energy: bool = True,
    custom_energy: Optional[dict] = None
) -> Arch:
    """
    Create an Eyeriss-like architecture for layer fusion experiments.
    
    Args:
        config: EyerissArchConfig with architecture parameters
        coupling: Coupling for the workload (conv_coupling, conv_2layers_coupling, etc.)
        shape: Shape of the workload (for factor constraints)
        num_layers: Number of fused layers (overrides config.num_fused_layers if provided)
        use_default_energy: Use default Eyeriss energy values
        custom_energy: Optional dict with custom energy values
    
    Returns:
        Arch instance for the specified configuration
    """
    if coupling is None:
        from computations import conv_coupling
        coupling = conv_coupling
    
    if num_layers is None:
        num_layers = config.num_fused_layers
    
    # Get energy values
    if custom_energy:
        energy = custom_energy
    else:
        energy = get_eyeriss_energy_values(config)
    
    # Create architecture based on number of layers
    if num_layers == 1:
        return _create_eyeriss_single_layer(config, coupling, shape, energy)
    else:
        return _create_eyeriss_multi_layer(config, coupling, shape, num_layers, energy)


def _create_eyeriss_single_layer(
    config: EyerissArchConfig,
    coupling: Coupling,
    shape: Shape,
    energy: dict
) -> Arch:
    """Create single-layer Eyeriss architecture (matches arch_eyeriss_conv structure)."""
    
    pe_cols = config.pe_cols
    pe_rows = config.pe_rows
    gb_size = config.global_buffer_size_B / 8
    
    levels = [
        # === DRAM ===
        MemLevel(
            name="DRAM",
            dataflow_constraints=[],
            size=2**64-1,
            value_access_energy=energy['dram_energy'],
            bandwidth=config.dram_bandwidth,
            factors_constraints={},
            bypasses=[]
        ),
        
        # === GlobalBuffer (bypasses w) ===
        # P=1 constraint: P dimension entirely at DRAM
        MemLevel(
            name="GlobalBuffer",
            dataflow_constraints=[],
            size=gb_size,
            value_access_energy=energy['global_buffer_energy'],
            bandwidth=config.get_scaled_gb_bandwidth(),
            factors_constraints={'P': 1},
            bypasses=['w']
        ),
        
        # === SACols (spatial distribution across columns) ===
        FanoutLevel(
            name="SACols",
            mesh=pe_cols,
            dims=['Q', 'M'],
            factors_constraints={}
        ),
        
        # === SARows (spatial distribution across rows) ===
        FanoutLevel(
            name="SARows",
            mesh=pe_rows,
            dims=['S', 'C', 'M'],
            factors_constraints={}
        ),
        
        # === InRegister (bypasses w, out) ===
        MemLevel(
            name="InRegister",
            dataflow_constraints=['S', 'R', 'Q', 'P', 'C', 'M'],
            size=config.input_reg_entries,
            value_access_energy=energy['input_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints={'M': 1, 'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
            bypasses=['w', 'out']
        ),
        
        # === WRegister (bypasses in, out) ===
        MemLevel(
            name="WRegister",
            dataflow_constraints=['R', 'C', 'S', 'Q', 'P', 'M'],
            size=config.weight_reg_entries,
            value_access_energy=energy['weight_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints={'M': 1, 'P': 1, 'Q': 1, 'S': 1},
            bypasses=['in', 'out']
        ),
        
        # === OutRegister (bypasses in, w) ===
        MemLevel(
            name="OutRegister",
            dataflow_constraints=['M', 'S', 'R', 'Q', 'P', 'C'],
            size=config.output_reg_entries,
            value_access_energy=energy['output_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints={'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
            bypasses=['in', 'w']
        ),
        
        # === Compute ===
        ComputeLevel(
            name="Compute",
            mesh=1,
            compute_energy=energy['compute_energy'],
            leakage_energy=0.001,
            cycles=1,
            factors_constraints={}
        )
    ]
    
    return Arch(levels, coupling=coupling,
                name=f"Eyeriss_1Layer_{gb_size}B_GB_{pe_cols}x{pe_rows}_PEs")


# =============================================================================
# FULLY CONSTRAINED EYERISS ARCHITECTURE (Single deterministic mapping)
# =============================================================================

def _highest_divisor_leq(total: int, max_val: int) -> int:
    """Find the highest divisor of `total` that is <= `max_val`."""
    for d in range(min(total, max_val), 0, -1):
        if total % d == 0:
            return d
    return 1


def compute_eyeriss_mapping_constraints(
    shape: 'Shape',
    output_tile_size: int,
    pe_cols: int,
    pe_rows: int,
    num_layers: int = 1
) -> dict:
    """
    Compute all factor constraints for a fully-constrained Eyeriss mapping.
    
    Policy:
    - P: all iterations at DRAM
    - Q: distributed across DRAM, GlobalBuffer, SACols (bottom-up greedy)
    - SARows: S first, then C, then M/Z (greedy allocation)
    - InRegister: no iterations (all = 1)
    - WRegister: remaining C and R
    - OutRegister: remaining M
    - IntermediateReg (fusion): Zi = pe_rows (for layer i < last)
    
    For fusion: Xi, Yi follow same policy as Q, P for each layer.
    
    Args:
        shape: Workload shape
        output_tile_size: User-specified output tile size (GlobalBuffer tile)
        pe_cols: Number of PE columns (SACols mesh)
        pe_rows: Number of PE rows (SARows mesh)
        num_layers: Number of fused layers
    
    Returns:
        dict with constraints for each level
    """
    constraints = {}
    
    if num_layers == 1:
        # === Single Layer ===
        Q_shape = shape.get('Q', 1)
        P_shape = shape.get('P', 1)
        M_shape = shape.get('M', 1)
        C_shape = shape.get('C', 1)
        R_shape = shape.get('R', 1)
        S_shape = shape.get('S', 1)
        
        remaining_col_mesh = pe_cols
        # SACols: Q = highest divisor of output_tile_size <= remaining_col_mesh
        sacols_Q = _highest_divisor_leq(output_tile_size, remaining_col_mesh)
        remaining_col_mesh = remaining_col_mesh // sacols_Q if sacols_Q > 0 else remaining_col_mesh

        sacols_M = _highest_divisor_leq(M_shape, remaining_col_mesh)
        remaining_col_mesh = remaining_col_mesh // sacols_M if sacols_M > 0 else remaining_col_mesh
        remaining_M_shape = M_shape // sacols_M if sacols_M > 0 else M_shape
        
        # GlobalBuffer: Q = output_tile_size / sacols_Q
        gb_Q = output_tile_size // sacols_Q
        
        # DRAM: Q = total / output_tile_size, P = all
        dram_Q = Q_shape // output_tile_size
        dram_P = P_shape
        
        # SARows: greedy allocation (S -> C -> M)
        remaining_row_mesh = pe_rows
        sarows_S = min(S_shape, remaining_row_mesh)
        remaining_row_mesh = remaining_row_mesh // sarows_S if sarows_S > 0 else remaining_row_mesh
        
        sarows_C = _highest_divisor_leq(C_shape, remaining_row_mesh)
        remaining_row_mesh = remaining_row_mesh // sarows_C if sarows_C > 0 else remaining_row_mesh
        
        sarows_M = _highest_divisor_leq(remaining_M_shape, remaining_row_mesh)
        
        # WRegister: remaining C and R
        wreg_C = C_shape // sarows_C if sarows_C > 0 else C_shape
        wreg_R = R_shape  # All R at WRegister
        
        # OutRegister: remaining M
        outreg_M = remaining_M_shape // sarows_M if sarows_M > 0 else remaining_M_shape
        
        constraints = {
            'DRAM': {'Q': dram_Q, 'P': dram_P, 'M': 1, 'C': 1, 'R': 1, 'S': 1},
            'GlobalBuffer': {'Q': gb_Q, 'P': 1, 'M': 1, 'C': 1, 'R': 1, 'S': 1},
            'SACols': {'Q': sacols_Q, 'M': sacols_M},
            'SARows': {'S': sarows_S, 'C': sarows_C, 'M': sarows_M},
            'InRegister': {'M': 1, 'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
            'WRegister': {'C': wreg_C, 'R': wreg_R, 'M': 1, 'P': 1, 'Q': 1, 'S': 1},
            'OutRegister': {'M': outreg_M, 'C': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1},
        }
        
    else:
        # === Multi-Layer Fusion ===
        last_layer = num_layers - 1
        
        # Output layer dimensions (last layer uses Q, P)
        Q_shape = shape.get('Q', 1)
        P_shape = shape.get('P', 1)
        Z_shape = shape.get(f'Z{last_layer}', 1)
        
        remaining_col_mesh = pe_cols
        # SACols: Q = highest divisor of output_tile_size <= remaining_col_mesh
        sacols_Q = _highest_divisor_leq(output_tile_size, remaining_col_mesh)
        remaining_col_mesh = remaining_col_mesh // sacols_Q if sacols_Q > 0 else remaining_col_mesh

        sacols_Z = _highest_divisor_leq(Z_shape, remaining_col_mesh)
        remaining_Z_last_layer_shape = Z_shape // sacols_Z if sacols_Z > 0 else Z_shape        
        

        gb_Q = output_tile_size // sacols_Q
        dram_Q = Q_shape // output_tile_size
        dram_P = P_shape
        
        constraints['DRAM'] = {'Q': dram_Q, 'P': dram_P}
        constraints['GlobalBuffer'] = {'Q': gb_Q, 'P': 1}
        constraints['SACols'] = {'Q': sacols_Q, f'Z{last_layer}': sacols_Z}
        
        # For each layer (0 to last_layer)
        for layer in range(num_layers):
            if layer < last_layer:
                # Intermediate layer: uses Xi, Yi, Zi, Ci, Ri, Si
                X_dim = f'X{layer}'
                Y_dim = f'Y{layer}'
                Z_dim = f'Z{layer}'
                C_dim = f'C{layer}'
                R_dim = f'R{layer}'
                S_dim = f'S{layer}'
                
                X_shape = shape.get(X_dim, Q_shape)  # Same as output if stride=1
                Y_shape = shape.get(Y_dim, P_shape)
                Z_shape = shape.get(Z_dim, 1)
                C_shape_layer = shape.get(C_dim, 1)
                R_shape_layer = shape.get(R_dim, 1)
                S_shape_layer = shape.get(S_dim, 1)
                
                # Xi follows same policy as Q (output_tile_size, for stride=1)
                remaining_col_mesh = pe_cols
                sacols_X = _highest_divisor_leq(output_tile_size, remaining_col_mesh)
                remaining_col_mesh = remaining_col_mesh // sacols_X if sacols_X > 0 else remaining_col_mesh
                
                sacols_Z_i = _highest_divisor_leq(Z_shape, remaining_col_mesh)
                remaining_col_mesh = remaining_col_mesh // sacols_Z_i if sacols_Z_i > 0 else remaining_col_mesh
                remaining_Z_shape = Z_shape // sacols_Z_i if sacols_Z_i > 0 else Z_shape


                gb_X = output_tile_size // sacols_X
                dram_X = X_shape // output_tile_size
                
                # Yi follows same policy as P (all at DRAM)
                dram_Y = Y_shape
                
                # Update DRAM and GlobalBuffer constraints
                constraints['DRAM'][X_dim] = dram_X
                constraints['DRAM'][Y_dim] = dram_Y
                constraints['GlobalBuffer'][X_dim] = gb_X
                constraints['GlobalBuffer'][Y_dim] = 1
        

                # SACols_i for this layer
                sacols_name = f'SACols_{layer}'
                constraints[sacols_name] = {X_dim: sacols_X, Z_dim: sacols_Z_i}

                # SARows_i: greedy allocation (Si -> Ci -> Zi)
                sarows_name = f'SARows_{layer}'
                remaining_row_mesh = pe_rows
                
                sarows_S = min(S_shape_layer, remaining_row_mesh)
                remaining_row_mesh = remaining_row_mesh // sarows_S if sarows_S > 0 else remaining_row_mesh
                
                sarows_C = _highest_divisor_leq(C_shape_layer, remaining_row_mesh)
                remaining_row_mesh = remaining_row_mesh // sarows_C if sarows_C > 0 else remaining_row_mesh
                
                sarows_Z = _highest_divisor_leq(remaining_Z_shape, remaining_row_mesh)
                
                constraints[sarows_name] = {S_dim: sarows_S, C_dim: sarows_C, Z_dim: sarows_Z}
                
                # IntermediateRegister_i: Zi = remaining after SARows
                intreg_name = f'IntermediateRegister_{layer}'
                intreg_Z = remaining_Z_shape // sarows_Z if sarows_Z > 0 else remaining_Z_shape
                constraints[intreg_name] = {Z_dim: intreg_Z}
                
                # Store layer-specific constraints for WRegister and registers
                # WRegister: remaining C and R for this layer
                wreg_C = C_shape_layer // sarows_C
                wreg_R = R_shape_layer
                
                # Add to DRAM constraints (dimensions with factor=1)
                constraints['DRAM'][Z_dim] = 1
                constraints['DRAM'][C_dim] = 1
                constraints['DRAM'][R_dim] = 1
                constraints['DRAM'][S_dim] = 1
                
                # Add to GlobalBuffer constraints
                constraints['GlobalBuffer'][Z_dim] = 1
                constraints['GlobalBuffer'][C_dim] = 1
                constraints['GlobalBuffer'][R_dim] = 1
                constraints['GlobalBuffer'][S_dim] = 1                
            else:
                # Last layer: uses Q, P, Z{last}, C{last}, R{last}, S{last}
                # But Q, P already handled above
                Z_dim = f'Z{layer}'
                C_dim = f'C{layer}'
                R_dim = f'R{layer}'
                S_dim = f'S{layer}'
                
                C_shape_layer = shape.get(C_dim, 1)
                R_shape_layer = shape.get(R_dim, 1)
                S_shape_layer = shape.get(S_dim, 1)
                
                # SARows for last layer (same as single-layer policy)
                sarows_name = f'SARows_{layer}'
                remaining_mesh = pe_rows
                
                sarows_S = min(S_shape_layer, remaining_mesh)
                remaining_mesh = remaining_mesh // sarows_S if sarows_S > 0 else remaining_mesh
                
                sarows_C = _highest_divisor_leq(C_shape_layer, remaining_mesh)
                remaining_mesh = remaining_mesh // sarows_C if sarows_C > 0 else remaining_mesh
                
                sarows_Z = _highest_divisor_leq(remaining_Z_last_layer_shape, remaining_mesh)
                
                constraints[sarows_name] = {S_dim: sarows_S, C_dim: sarows_C, Z_dim: sarows_Z}
                
                # OutRegister: remaining Z
                outreg_Z = Z_shape // sarows_Z
                constraints['OutRegister'] = {Z_dim: outreg_Z}
                                
                # Add to DRAM constraints
                constraints['DRAM'][Z_dim] = 1
                constraints['DRAM'][C_dim] = 1
                constraints['DRAM'][R_dim] = 1
                constraints['DRAM'][S_dim] = 1
                
                # Add to GlobalBuffer constraints
                constraints['GlobalBuffer'][Z_dim] = 1
                constraints['GlobalBuffer'][C_dim] = 1
                constraints['GlobalBuffer'][R_dim] = 1
                constraints['GlobalBuffer'][S_dim] = 1
    
    return constraints


def create_constrained_eyeriss_single_layer(
    config: EyerissArchConfig,
    coupling: 'Coupling',
    shape: 'Shape',
    output_tile_size: int,
    energy: dict
) -> 'Arch':
    """
    Create a fully-constrained single-layer Eyeriss architecture.
    
    All factor_constraints are set to exact values, resulting in a single
    deterministic mapping (no search required).
    """
    from levels import MemLevel, FanoutLevel, ComputeLevel
    from arch import Arch
    
    pe_cols = config.pe_cols
    pe_rows = config.pe_rows
    gb_size = config.global_buffer_size_B
    
    # Compute all constraints
    constraints = compute_eyeriss_mapping_constraints(
        shape, output_tile_size, pe_cols, pe_rows, num_layers=1
    )
    
    print(f"[Constrained Eyeriss] Single layer, tile_size={output_tile_size}")
    print(f"  DRAM: {constraints['DRAM']}")
    print(f"  GlobalBuffer: {constraints['GlobalBuffer']}")
    print(f"  SACols: {constraints['SACols']}")
    print(f"  SARows: {constraints['SARows']}")
    print(f"  WRegister: {constraints['WRegister']}")
    print(f"  OutRegister: {constraints['OutRegister']}")
    
    levels = [
        # === DRAM ===
        MemLevel(
            name="DRAM",
            dataflow_constraints=['P', 'Q'],  # Depth-first: P outermost, then Q
            size=2**64-1,
            value_access_energy=energy['dram_energy'],
            bandwidth=config.dram_bandwidth,
            factors_constraints=constraints['DRAM'],
            bypasses=[]
        ),
        
        # === GlobalBuffer (bypasses w) ===
        MemLevel(
            name="GlobalBuffer",
            dataflow_constraints=['Q', 'S', 'C', 'R'],
            size=gb_size,
            value_access_energy=energy['global_buffer_energy'],
            bandwidth=config.get_scaled_gb_bandwidth(),
            factors_constraints=constraints['GlobalBuffer'],
            bypasses=['w']
        ),
        
        # === SACols ===
        FanoutLevel(
            name="SACols",
            mesh=pe_cols,
            dims=['Q', 'M'],
            factors_constraints=constraints['SACols']
        ),
        
        # === SARows ===
        FanoutLevel(
            name="SARows",
            mesh=pe_rows,
            dims=['S', 'C', 'M'],
            factors_constraints=constraints['SARows']
        ),
        
        # === InRegister (bypasses w, out) ===
        MemLevel(
            name="InRegister",
            dataflow_constraints=['S', 'R', 'Q', 'P', 'C', 'M'],
            size=config.input_reg_entries,
            value_access_energy=energy['input_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints=constraints['InRegister'],
            bypasses=['w', 'out']
        ),
        
        # === WRegister (bypasses in, out) ===
        MemLevel(
            name="WRegister",
            dataflow_constraints=['C', 'R', 'S', 'Q', 'P', 'M'],
            size=config.weight_reg_entries,
            value_access_energy=energy['weight_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints=constraints['WRegister'],
            bypasses=['in', 'out']
        ),
        
        # === OutRegister (bypasses in, w) ===
        MemLevel(
            name="OutRegister",
            dataflow_constraints=['M', 'S', 'R', 'Q', 'P', 'C'],
            size=config.output_reg_entries,
            value_access_energy=energy['output_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints=constraints['OutRegister'],
            bypasses=['in', 'w']
        ),
        
        # === Compute ===
        ComputeLevel(
            name="Compute",
            mesh=1,
            compute_energy=energy['compute_energy'],
            leakage_energy=0.001,
            cycles=1,
            factors_constraints={}
        )
    ]
    
    return Arch(levels, coupling=coupling,
                name=f"Eyeriss_Constrained_1L_tile{output_tile_size}_{pe_cols}x{pe_rows}")


def create_constrained_eyeriss_architecture(
    config: EyerissArchConfig,
    coupling: 'Coupling' = None,
    shape: 'Shape' = None,
    output_tile_size: int = None,
    num_layers: int = None,
    use_default_energy: bool = True,
    custom_energy: dict = None
) -> 'Arch':
    """
    Create a fully-constrained Eyeriss architecture with deterministic mapping.
    
    All factor_constraints are computed from the output_tile_size, resulting
    in a single valid mapping. The mapper will find this mapping without search.
    
    Args:
        config: EyerissArchConfig with architecture parameters
        coupling: Coupling for the workload
        shape: Shape of the workload (required for computing constraints)
        output_tile_size: User-specified output tile size (required)
        num_layers: Number of fused layers
        use_default_energy: Use default energy values
        custom_energy: Optional custom energy values
    
    Returns:
        Fully-constrained Arch instance
    """
    if coupling is None:
        from computations import conv_coupling
        coupling = conv_coupling
    
    if shape is None:
        raise ValueError("Shape is required for constrained architecture")
    
    if output_tile_size is None:
        raise ValueError("output_tile_size is required for constrained architecture")
    
    if num_layers is None:
        num_layers = config.num_fused_layers
    
    # Get energy values
    if custom_energy:
        energy = custom_energy
    else:
        energy = get_eyeriss_energy_values(config)
    
    # Create architecture based on number of layers
    if num_layers == 1:
        return create_constrained_eyeriss_single_layer(
            config, coupling, shape, output_tile_size, energy
        )
    else:
        return create_constrained_eyeriss_multi_layer(
            config, coupling, shape, output_tile_size, num_layers, energy
        )

# 1.89e+07

def create_constrained_eyeriss_multi_layer(
    config: EyerissArchConfig,
    coupling: 'Coupling',
    shape: 'Shape',
    output_tile_size: int,
    num_layers: int,
    energy: dict
) -> 'Arch':
    """
    Create a fully-constrained multi-layer Eyeriss architecture for fusion.
    
    Each layer i has:
    - SACols_i with Xi constraints
    - SARows_i with Si, Ci, Zi constraints  
    - IntermediateRegister_i with Zi constraints (for layers < last)
    """
    from levels import MemLevel, FanoutLevel, ComputeLevel
    from arch import Arch
    
    pe_cols = config.pe_cols
    pe_rows = config.pe_rows
    gb_size = config.global_buffer_size_B
    last_layer = num_layers - 1
    
    # Compute all constraints
    constraints = compute_eyeriss_mapping_constraints(
        shape, output_tile_size, pe_cols, pe_rows, num_layers
    )
    
    print(f"[Constrained Eyeriss] {num_layers} layers, tile_size={output_tile_size}")
    print(f"  DRAM: {constraints['DRAM']}")
    print(f"  GlobalBuffer: {constraints['GlobalBuffer']}")
    print(f"  SACols: {constraints['SACols']}")
    for layer in range(num_layers):
        sarows_name = f'SARows_{layer}'
        print(f"  {sarows_name}: {constraints.get(sarows_name, {})}")
    
    # Build depth-first dataflow order:
    # Outermost: P, Q (output spatial)
    # Then layer N-1 (last layer) dims: Z, C, R, S
    # Then intermediate Y, X for layer N-2
    # Then layer N-2 dims
    # ...
    # Innermost: layer 0 dims (input layer)
    dram_dataflow = ['P', 'Q']
    # Add layers from last to first (last layer outermost, first layer innermost)
    for layer in range(last_layer, -1, -1):
        # Layer dimensions
        dram_dataflow.extend([f'Z{layer}', f'C{layer}', f'R{layer}', f'S{layer}'])
        # Intermediate spatial (only for layers before the last)
        if layer > 0:
            dram_dataflow.extend([f'Y{layer-1}', f'X{layer-1}'])
    
    # All levels use the same depth-first dataflow order as DRAM
    gb_dataflow = dram_dataflow.copy()
    inreg_dataflow = dram_dataflow.copy()
    wreg_dataflow = dram_dataflow.copy()
    intreg_dataflow = dram_dataflow.copy()
    outreg_dataflow = dram_dataflow.copy()
    
    levels = [
        # === DRAM ===
        MemLevel(
            name="DRAM",
            dataflow_constraints=dram_dataflow,
            size=2**64-1,
            value_access_energy=energy['dram_energy'],
            bandwidth=config.dram_bandwidth,
            factors_constraints=constraints['DRAM'],
            bypasses=['int_in', 'int_out']
        ),
        
        # === GlobalBuffer (bypasses w) ===
        MemLevel(
            name="GlobalBuffer",
            dataflow_constraints=gb_dataflow,
            size=gb_size,
            value_access_energy=energy['global_buffer_energy'],
            bandwidth=config.get_scaled_gb_bandwidth(),
            factors_constraints=constraints['GlobalBuffer'],
            bypasses=['w']
        ),
    ]
    
    # Add SACols (output layer uses Q, intermediate layers handled by SACols_i)
    sacols_Z_last = constraints['SACols'].get(f'Z{last_layer}', 1)
    levels.append(
        FanoutLevel(
            name="SACols",
            mesh=pe_cols,
            dims=['Q'] + [f'Z{last_layer}'],  # Output spatial
            factors_constraints={'Q': constraints['SACols']['Q'], f'Z{last_layer}': sacols_Z_last}
        )
    )
    
    # Add SARows for output layer
    sarows_last = constraints[f'SARows_{last_layer}']
    levels.append(
        FanoutLevel(
            name="SARows",
            mesh=pe_rows,
            dims=[f'S{last_layer}', f'C{last_layer}', f'Z{last_layer}'],
            factors_constraints=sarows_last
        )
    )
    
    # Add per-layer spatial levels (in reverse order: from last-1 to 0)
    for layer in range(last_layer - 1, -1, -1):
        X_dim = f'X{layer}'
        Z_dim = f'Z{layer}'
        S_dim = f'S{layer}'
        C_dim = f'C{layer}'
        
        sacols_name = f'SACols_{layer}'
        sarows_name = f'SARows_{layer}'
        
        # SACols_i
        sacols_constraints = constraints.get(sacols_name, {X_dim: 1, Z_dim: 1})
        levels.append(
            FanoutLevel(
                name=sacols_name,
                mesh=pe_cols,
                dims=[X_dim, Z_dim],
                factors_constraints={X_dim: sacols_constraints.get(X_dim, 1), Z_dim: sacols_constraints.get(Z_dim, 1)}
            )
        )
        
        # SARows_i
        sarows_constraints = constraints.get(sarows_name, {})
        levels.append(
            FanoutLevel(
                name=sarows_name,
                mesh=pe_rows,
                dims=[S_dim, C_dim, Z_dim],
                factors_constraints=sarows_constraints
            )
        )
    
    # Add register levels (shared structure for all layers)
    # InRegister: all factors = 1
    inreg_constraints = {'P': 1, 'Q': 1}
    for layer in range(num_layers):
        inreg_constraints[f'C{layer}'] = 1
        inreg_constraints[f'R{layer}'] = 1
        inreg_constraints[f'S{layer}'] = 1
        inreg_constraints[f'Z{layer}'] = 1
        if layer < last_layer:
            inreg_constraints[f'X{layer}'] = 1
            inreg_constraints[f'Y{layer}'] = 1
    
    levels.append(
        MemLevel(
            name="InRegister",
            dataflow_constraints=inreg_dataflow,
            size=config.input_reg_entries,
            value_access_energy=energy['input_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints=inreg_constraints,
            bypasses=['w', 'out', 'int_out']
        )
    )
    
    # WRegister: C_i and R_i for each layer
    wreg_constraints = {'P': 1, 'Q': 1}
    for layer in range(num_layers):
        S_dim = f'S{layer}'
        C_dim = f'C{layer}'
        R_dim = f'R{layer}'
        Z_dim = f'Z{layer}'
        
        C_shape = shape.get(C_dim, 1)
        R_shape = shape.get(R_dim, 1)
        
        sarows_constraints = constraints.get(f'SARows_{layer}', {})
        sarows_C = sarows_constraints.get(C_dim, 1)
        
        wreg_constraints[C_dim] = C_shape // sarows_C
        wreg_constraints[R_dim] = R_shape
        wreg_constraints[S_dim] = 1
        wreg_constraints[Z_dim] = 1
        
        if layer < last_layer:
            wreg_constraints[f'X{layer}'] = 1
            wreg_constraints[f'Y{layer}'] = 1
    
    levels.append(
        MemLevel(
            name="WRegister",
            dataflow_constraints=wreg_dataflow,
            size=config.weight_reg_entries,
            value_access_energy=energy['weight_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints=wreg_constraints,
            bypasses=['in', 'out', 'int_in', 'int_out']
        )
    )
    
    # IntermediateRegister: Zi for intermediate layers
    intreg_constraints = {'P': 1, 'Q': 1}
    for layer in range(num_layers):
        Z_dim = f'Z{layer}'
        C_dim = f'C{layer}'
        R_dim = f'R{layer}'
        S_dim = f'S{layer}'
        
        if layer < last_layer:
            intreg_name = f'IntermediateRegister_{layer}'
            Z_shape = shape.get(Z_dim, 1)
            sarows_constraints = constraints.get(f'SARows_{layer}', {})
            sarows_Z = sarows_constraints.get(Z_dim, 1)
            
            intreg_constraints[Z_dim] = Z_shape // sarows_Z
            intreg_constraints[f'X{layer}'] = 1
            intreg_constraints[f'Y{layer}'] = 1
        else:
            intreg_constraints[Z_dim] = 1
        
        intreg_constraints[C_dim] = 1
        intreg_constraints[R_dim] = 1
        intreg_constraints[S_dim] = 1
    
    levels.append(
        MemLevel(
            name="IntermediateRegister",
            dataflow_constraints=intreg_dataflow,
            size=config.intermediate_out_reg_entries,
            value_access_energy=energy['intermediate_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints=intreg_constraints,
            bypasses=['in', 'w', 'out', 'int_in']
        )
    )
    
    # OutRegister: Z{last_layer}
    outreg_constraints = {'P': 1, 'Q': 1}
    for layer in range(num_layers):
        Z_dim = f'Z{layer}'
        C_dim = f'C{layer}'
        R_dim = f'R{layer}'
        S_dim = f'S{layer}'
        
        if layer == last_layer:
            Z_shape = shape.get(Z_dim, 1)
            sarows_constraints = constraints.get(f'SARows_{layer}', {})
            sarows_Z = sarows_constraints.get(Z_dim, 1)
            outreg_constraints[Z_dim] = Z_shape // sarows_Z
        else:
            outreg_constraints[Z_dim] = 1
            outreg_constraints[f'X{layer}'] = 1
            outreg_constraints[f'Y{layer}'] = 1
        
        outreg_constraints[C_dim] = 1
        outreg_constraints[R_dim] = 1
        outreg_constraints[S_dim] = 1
    
    levels.append(
        MemLevel(
            name="OutRegister",
            dataflow_constraints=outreg_dataflow,
            size=config.output_reg_entries,
            value_access_energy=energy['output_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints=outreg_constraints,
            bypasses=['in', 'w', 'int_in', 'int_out']
        )
    )
    
    # Compute level
    levels.append(
        ComputeLevel(
            name="Compute",
            mesh=1,
            compute_energy=energy['compute_energy'],
            leakage_energy=0.001,
            cycles=1,
            factors_constraints={}
        )
    )
    
    return Arch(levels, coupling=coupling,
                name=f"Eyeriss_Constrained_{num_layers}L_tile{output_tile_size}_{pe_cols}x{pe_rows}")


def _create_eyeriss_multi_layer(
    config: EyerissArchConfig,
    coupling: Coupling,
    shape: Shape,
    num_layers: int,
    energy: dict
) -> Arch:
    """
    Create multi-layer Eyeriss architecture for layer fusion.
    
    Key features:
    - Replicated spatial levels (SACols_i, SARows_i) for each fused layer
    - IntermediateRegister for partial sums of intermediate outputs
    - Unified GlobalBuffer
    """
    
    pe_cols = config.pe_cols
    pe_rows = config.pe_rows
    gb_size = config.global_buffer_size_B
    last_layer = num_layers - 1
    
    # Pre-compute per-layer tile sizes based on stride info
    per_layer_tile_constraints = None
    if config.tile_size_override is not None and shape is not None:
        per_layer_tile_constraints = get_per_layer_tile_size_constraints(
            shape, num_layers,
            output_tile_h=config.tile_size_override,
            output_tile_w=config.tile_size_override,
            pe_cols=pe_cols
        )
        
        cum_pstride, cum_qstride = get_cumulative_stride(shape, num_layers)
        if cum_pstride > 1 or cum_qstride > 1:
            print(f"[Eyeriss Stride-Aware Tiling] output_tile={config.tile_size_override}, "
                  f"cumulative_stride={cum_pstride}x{cum_qstride}")
            input_tile = per_layer_tile_constraints.get('X0', config.tile_size_override)
            print(f"  Input tile (X0): {input_tile}, Output tile (Q): {config.tile_size_override}")
    
    # Helper: compute GCD constraint
    def gcd_constraint(dim_size: int, mesh: int, is_spatial_width: bool = False, dim_name: str = None) -> int:
        if is_spatial_width and per_layer_tile_constraints is not None and dim_name is not None:
            tile = per_layer_tile_constraints.get(dim_name, config.tile_size_override)
            if dim_size % tile == 0 and tile <= mesh:
                return tile
        
        if is_spatial_width and config.tile_size_override is not None and per_layer_tile_constraints is None:
            tile = config.tile_size_override
            if dim_size % tile == 0 and tile <= mesh:
                return tile
        
        if dim_size <= mesh:
            return dim_size
        for d in range(mesh, 0, -1):
            if dim_size % d == 0:
                return d
        return 1
    
    # Dimension name helper
    def get_dim_name(base, layer_idx):
        return f'{base}{layer_idx}'
    
    # === Build DRAM depth-first dataflow and factor constraints (like DepFiN) ===
    # DRAM iterates over output tiles (Q, P) and intermediate tiles (Xi, Yi) in outermost loops
    # This enables depth-first execution: process each output tile completely before moving on
    dram_dataflow = ['Q', 'P']
    dram_factors = {}
    
    # Fanout constraints per layer (for computing remaining iterations at DRAM)
    fanout_constraints = {}
    
    if shape is not None:
        # Output layer dimensions
        q_size = shape.get('Q', pe_cols)
        p_size = shape.get('P', 1)
        
        # Fanout for output layer (Q in cols, Z{last_layer} in rows)
        q_fanout = gcd_constraint(q_size, pe_cols, is_spatial_width=True, dim_name='Q')
        z_last_dim = f'Z{last_layer}'
        z_last_size = shape.get(z_last_dim, pe_rows)
        z_last_fanout = gcd_constraint(z_last_size, pe_rows, is_spatial_width=False, dim_name=z_last_dim)
        fanout_constraints['Q'] = q_fanout
        fanout_constraints[z_last_dim] = z_last_fanout
        
        # DRAM depth-first: number of output tiles
        # If tile_size_override is given, use it; otherwise tile=full image
        if config.tile_size_override is not None:
            output_tile_q = config.tile_size_override
            output_tile_p = config.tile_size_override
            # Number of tiles = ceil(size / tile_size)
            import math
            dram_factors['Q'] = math.ceil(q_size / output_tile_q)
            dram_factors['P'] = math.ceil(p_size / output_tile_p)
        else:
            # No tile constraint: single tile covering full image
            dram_factors['Q'] = 1
            dram_factors['P'] = 1
        
        # Intermediate layers (0 to last_layer-1) - depth-first over intermediate tiles
        # For FSRCNN with all 1x1 convolutions (stride=1), intermediate tile = output tile
        for layer in range(last_layer - 1, -1, -1):
            x_dim = get_dim_name('X', layer)
            y_dim = get_dim_name('Y', layer)
            z_dim = get_dim_name('Z', layer)
            
            # Add to DRAM dataflow (depth-first ordering)
            dram_dataflow.extend([x_dim, y_dim])
            
            # Get dimension sizes
            x_size = shape.get(x_dim, pe_cols)
            y_size = shape.get(y_dim, 1)
            z_size = shape.get(z_dim, pe_rows)
            
            # Fanout constraints - X is spatial width, Z is not
            x_fanout = gcd_constraint(x_size, pe_cols, is_spatial_width=True, dim_name=x_dim)
            z_fanout = gcd_constraint(z_size, pe_rows, is_spatial_width=False, dim_name=z_dim)
            fanout_constraints[x_dim] = x_fanout
            fanout_constraints[z_dim] = z_fanout
            
            # DRAM depth-first: number of intermediate tiles
            # For 1x1 convolutions (stride=1), intermediate tile count = output tile count
            if config.tile_size_override is not None:
                import math
                # Same tile size for intermediate layers (stride=1 assumption)
                dram_factors[x_dim] = math.ceil(x_size / config.tile_size_override)
                dram_factors[y_dim] = math.ceil(y_size / config.tile_size_override)
            else:
                # No tile constraint: single tile
                dram_factors[x_dim] = 1
                dram_factors[y_dim] = 1
        
        print(f"[Eyeriss DRAM Depth-First] dataflow={dram_dataflow}")
        print(f"[Eyeriss DRAM Depth-First] factors={dram_factors}")
    else:
        print("Warning: Shape not provided, using default DRAM constraints.")
        # Default fallback when shape not provided
        for layer in range(last_layer - 1, -1, -1):
            x_dim = get_dim_name('X', layer)
            y_dim = get_dim_name('Y', layer)
            dram_dataflow.extend([x_dim, y_dim])
    
    # Build levels
    levels = [
        # === DRAM (bypasses int_in, int_out) - DEPTH-FIRST DATAFLOW ===
        MemLevel(
            name="DRAM",
            dataflow_constraints=dram_dataflow,
            size=2**64-1,
            value_access_energy=energy['dram_energy'],
            bandwidth=config.dram_bandwidth,
            factors_constraints=dram_factors,
            bypasses=['int_in', 'int_out']
        ),
        
        # === GlobalBuffer (bypasses w) ===
        MemLevel(
            name="GlobalBuffer",
            dataflow_constraints=[],
            size=gb_size,
            value_access_energy=energy['global_buffer_energy'],
            bandwidth=config.get_scaled_gb_bandwidth(),
            factors_constraints={},
            bypasses=['w']
        ),
    ]
    
    # === Spatial Levels (SACols_i, SARows_i) for each layer ===
    # Output layer uses Q, M for columns and S, C, M for rows
    levels.extend([
        FanoutLevel(
            name=f"SACols_{last_layer}",
            mesh=pe_cols,
            dims=['Q', f'Z{last_layer}'],  # Q and output channels
            factors_constraints={}
        ),
        FanoutLevel(
            name=f"SARows_{last_layer}",
            mesh=pe_rows,
            dims=[f'S{last_layer}', f'C{last_layer}', f'Z{last_layer}'],
            factors_constraints={}
        ),
    ])
    
    # Intermediate and input layers use Xi for columns
    for layer in range(last_layer - 1, -1, -1):
        x_dim = get_dim_name('X', layer)
        z_dim = get_dim_name('Z', layer)
        s_dim = get_dim_name('S', layer)
        c_dim = get_dim_name('C', layer)
        
        levels.extend([
            FanoutLevel(
                name=f"SACols_{layer}",
                mesh=pe_cols,
                dims=[x_dim, z_dim],
                factors_constraints={}
            ),
            FanoutLevel(
                name=f"SARows_{layer}",
                mesh=pe_rows,
                dims=[s_dim, c_dim, z_dim],
                factors_constraints={}
            ),
        ])
    
    # === InRegister (bypasses w, out, int_out) ===
    # Build dataflow and factor constraints for all layers
    in_reg_dataflow = []
    in_reg_factors = {}
    for layer in range(last_layer, -1, -1):
        z_dim = f'Z{layer}'
        c_dim = f'C{layer}'
        r_dim = f'R{layer}'
        s_dim = f'S{layer}'
        in_reg_dataflow.extend([s_dim, r_dim])
        in_reg_factors[z_dim] = 1
        in_reg_factors[c_dim] = 1
        in_reg_factors[r_dim] = 1
        in_reg_factors[s_dim] = 1
    
    # Add spatial dims
    in_reg_dataflow.extend(['Q', 'P'])
    in_reg_factors['P'] = 1
    in_reg_factors['Q'] = 1
    for layer in range(last_layer - 1, -1, -1):
        in_reg_dataflow.extend([f'X{layer}', f'Y{layer}'])
        in_reg_factors[f'X{layer}'] = 1
        in_reg_factors[f'Y{layer}'] = 1
    
    levels.append(
        MemLevel(
            name="InRegister",
            dataflow_constraints=in_reg_dataflow,
            size=config.input_reg_entries,
            value_access_energy=energy['input_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints=in_reg_factors,
            bypasses=['w', 'out', 'int_out']
        )
    )
    
    # === WRegister (bypasses in, out, int_in, int_out) ===
    w_reg_dataflow = []
    w_reg_factors = {}
    for layer in range(last_layer, -1, -1):
        r_dim = f'R{layer}'
        c_dim = f'C{layer}'
        s_dim = f'S{layer}'
        w_reg_dataflow.extend([r_dim, c_dim, s_dim])
        w_reg_factors[f'Z{layer}'] = 1
        w_reg_factors[s_dim] = 1
    
    w_reg_dataflow.extend(['Q', 'P'])
    w_reg_factors['P'] = 1
    w_reg_factors['Q'] = 1
    for layer in range(last_layer - 1, -1, -1):
        w_reg_dataflow.extend([f'X{layer}', f'Y{layer}'])
        w_reg_factors[f'X{layer}'] = 1
        w_reg_factors[f'Y{layer}'] = 1
    
    levels.append(
        MemLevel(
            name="WRegister",
            dataflow_constraints=w_reg_dataflow,
            size=config.weight_reg_entries,
            value_access_energy=energy['weight_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints=w_reg_factors,
            bypasses=['in', 'out', 'int_in', 'int_out']
        )
    )
    
    # === IntermediateRegister (bypasses in, w, out, int_in) - NEW for layer fusion ===
    # This register holds intermediate outputs between layers (partial sums)
    # Only passes int_out through
    int_reg_dataflow = []
    int_reg_factors = {}
    for layer in range(last_layer - 1, -1, -1):
        z_dim = get_dim_name('Z', layer)
        y_dim = get_dim_name('Y', layer)
        x_dim = get_dim_name('X', layer)
        int_reg_dataflow.extend([z_dim, y_dim, x_dim])
        int_reg_factors[z_dim] = 1
        int_reg_factors[y_dim] = 1
        int_reg_factors[x_dim] = 1
    # Output layer Z dimension
    int_reg_factors[f'Z{last_layer}'] = 1
    
    levels.append(
        MemLevel(
            name="IntermediateRegister",
            dataflow_constraints=int_reg_dataflow,
            size=config.intermediate_out_reg_entries,
            value_access_energy=energy['intermediate_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints=int_reg_factors,
            bypasses=['in', 'w', 'out', 'int_in']  # Only passes int_out
        )
    )
    
    # === OutRegister (bypasses in, w, int_in, int_out) ===
    out_reg_dataflow = [f'Z{last_layer}', 'Q', 'P']
    out_reg_factors = {
        f'Z{last_layer}': 1,
        'Q': 1,
        'P': 1,
    }
    # Add constraints for all weight dims (should be 1)
    for layer in range(last_layer, -1, -1):
        out_reg_factors[f'C{layer}'] = 1
        out_reg_factors[f'R{layer}'] = 1
        out_reg_factors[f'S{layer}'] = 1
    
    levels.append(
        MemLevel(
            name="OutRegister",
            dataflow_constraints=out_reg_dataflow,
            size=config.output_reg_entries,
            value_access_energy=energy['output_reg_energy'],
            bandwidth=config.register_bandwidth,
            factors_constraints=out_reg_factors,
            bypasses=['in', 'w', 'int_in', 'int_out']
        )
    )
    
    # === Compute ===
    levels.append(
        ComputeLevel(
            name="Compute",
            mesh=1,
            compute_energy=energy['compute_energy'],
            leakage_energy=0.001,
            cycles=1,
            factors_constraints={}
        )
    )
    
    return Arch(levels, coupling=coupling,
                name=f"Eyeriss_{num_layers}Layers_{gb_size//1024}KB_GB_{pe_cols}x{pe_rows}_PEs")


# =============================================================================
# LEGACY WRAPPER (for backward compatibility)
# =============================================================================
"""
Create a 10-layer fused architecture for thesis experiments.

DEPRECATED: Use create_thesis_architecture() instead.

This is a legacy wrapper for backward compatibility.
"""
def create_thesis_architecture_10layers(
    config: ThesisArchConfig,
    coupling = None,
    use_accelergy_energy: bool = True,
    custom_energy: Optional[dict] = None
) -> Arch:
    if coupling is None:
        coupling = conv_10layers_coupling
    
    return create_thesis_architecture(
        config=config,
        coupling=coupling,
        num_layers=10,
        use_accelergy_energy=use_accelergy_energy,
        custom_energy=custom_energy
    )


def print_config_summary(config: ThesisArchConfig, energy: dict = None):
    """Print a summary of the configuration and energy values."""
    print("=" * 70)
    print("THESIS ARCHITECTURE CONFIGURATION")
    print("=" * 70)
    
    print(f"\n--- Memory Sizes ---")
    print(f"  Feature Memory (FMEM): {config.feature_memory_size_B} B")
    print(f"  Weight Memory (WMEM):  {config.weight_memory_size_B} B")
    
    print(f"\n--- Register Sizes ---")
    print(f"  Accumulation Reg:      {config.accumulation_reg_entries} entries (32-bit)")
    print(f"  Weight Reg:            {config.weight_reg_entries} entries (8-bit)")
    print(f"  Input Reg:             {config.input_reg_entries} entries (8-bit)")
    print(f"  IntermediateOut Reg:   {config.intermediate_output_reg_entries} entries (8-bit)")
    print(f"  (Output stored in AccReg - no separate OutputReg)")
    
    print(f"\n--- PE Array ---")
    print(f"  Rows × Cols: {config.pe_rows} × {config.pe_cols} = {config.pe_rows * config.pe_cols} PEs")
    
    print(f"\n--- Bandwidth (bytes/cycle) ---")
    print(f"  DRAM Read/Write:  {config.dram_read_bandwidth} / {config.dram_write_bandwidth}")
    print(f"  FMEM Read/Write:  {config.fmem_read_bandwidth} / {config.fmem_write_bandwidth}")
    print(f"  WMEM Read/Write:  {config.wmem_read_bandwidth} / {config.wmem_write_bandwidth}")
    
    print(f"\n--- Technology ---")
    print(f"  Node:  {config.technology}")
    print(f"  Scale: {config.technology_scale}x")
    
    if energy:
        print(f"\n--- Energy (pJ/byte, scaled by {config.technology_scale}x) ---")
        print(f"  DRAM:          {energy['dram_energy_per_byte']:.4f} pJ/byte")
        print(f"  FMEM:          {energy['fmem_energy_per_byte']:.4f} pJ/byte")
        print(f"  WMEM:          {energy['wmem_energy_per_byte']:.4f} pJ/byte")
        print(f"  AccReg:        {energy['accreg_energy_per_byte']:.4f} pJ/byte")
        print(f"  WeightReg:     {energy['wreg_energy_per_byte']:.4f} pJ/byte")
        print(f"  InputReg:      {energy['inreg_energy_per_byte']:.4f} pJ/byte")
        print(f"  IntOutReg:     {energy['intoutreg_energy_per_byte']:.4f} pJ/byte")
        print(f"  (OutputReg = AccReg)")
        print(f"  Compute:       {energy['compute_energy_per_mac']:.4f} pJ/MAC")
        
        print(f"\n--- Memory Configuration (for Accelergy) ---")
        print(f"  FMEM depth: {energy['config']['fmem_depth']} rows")
        print(f"  WMEM depth: {energy['config']['wmem_depth']} rows")
    
    print("=" * 70)


# === Predefined Configurations for Experiments ===

def get_baseline_config() -> ThesisArchConfig:
    """Baseline configuration matching DepFiN specs."""
    return ThesisArchConfig(
        feature_memory_size_B=1056 * 1024,
        weight_memory_size_B=524 * 1024,
        pe_rows=16,
        pe_cols=128,
        technology="22nm",
        technology_scale=0.5,  # Scale to 12nm
    )






# === Main for testing ===

if __name__ == "__main__":
    print("\nTesting thesis architecture configuration...\n")
    
    # Test baseline config
    config = get_baseline_config()
    energy = get_energy_values_from_accelergy(config)
    print_config_summary(config, energy)
    
    # Create architecture
    arch = create_thesis_architecture_10layers(config)
    print(f"\nArchitecture created: {arch.name}")
    print(f"Architecture ready for experiments!")
