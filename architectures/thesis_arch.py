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
    accumulation_reg_entries: int = 10  # Entries per accumulation register
    
    # === Register Sizes (entries) ===
    weight_reg_entries: int = 9          # Weight register (3x3 kernel = 9 weights)
    input_reg_entries: int = 1           # Input activation register
    intermediate_output_reg_entries: int = 10  # Intermediate outputs between layers
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
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.feature_memory_size_B > 0, "FMEM size must be positive"
        assert self.weight_memory_size_B > 0, "WMEM size must be positive"
        assert self.pe_rows > 0 and self.pe_cols > 0, "PE dimensions must be positive"

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
        q_fanout = gcd_constraint(q_size, pe_cols)
        # Output Z dimension is always Z{last_layer}
        z_last_dim = f'Z{last_layer}'
        z_last_size = shape.get(z_last_dim, pe_rows)
        z_last_fanout = gcd_constraint(z_last_size, pe_rows)
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
            
            # Fanout constraints
            x_fanout = gcd_constraint(x_size, pe_cols)
            z_fanout = gcd_constraint(z_size, pe_rows)
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
            read_bandwidth=config.fmem_read_bandwidth,
            write_bandwidth=config.fmem_write_bandwidth,
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
            factors_constraints={'Q': q_fanout},
            spatial_reduction_support=True
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
        spatial_reduction = (layer == last_layer - 1)
        x_dim = get_dim_name('X', layer)
        z_dim = get_dim_name('Z', layer)
        x_fanout = fanout_constraints.get(x_dim, pe_cols)
        z_fanout = fanout_constraints.get(z_dim, pe_rows)
        
        levels.extend([
            FanoutLevel(
                name=f"SACols_{layer}",
                mesh=pe_cols,
                dims=[x_dim],
                factors_constraints={x_dim: x_fanout},
                spatial_reduction_support=spatial_reduction
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
