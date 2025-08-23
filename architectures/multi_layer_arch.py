from computations import create_nlayer_conv_coupling
from levels import *
from arch import *

# Create a 3-layer convolution coupling for the architecture
multilayer_conv_coupling = create_nlayer_conv_coupling(num_layers=3, with_stride=False, with_batches=True)

arch_multilayer_conv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2'],  # ONLY final layer + output dims
        size = 2**64-1,
        value_access_energy = 64.00,
        bandwidth = 8,
        factors_constraints = {'Q': 16},  # Remove conflicting constraints        
        #factors_constraints = {'S1': 1, 'S0': 1, 'R1': 1, 'R0': 1, 'C1': 1, 'C2': 1},  # Remove conflicting constraints
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2'],  # ONLY final layer + output dims
        size = 16384*8,
        value_access_energy = 2.02,
        bandwidth = 32,
        factors_constraints = {'Q': 2, 'P': 64},  # Remove conflicting constraints
        bypasses = []
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 14,
        dims = ['Q', 'K'],  # Only final layer dimensions
        factors_constraints = {'Q': 8}
    ),
    FanoutLevel(
        name = "SARows", 
        mesh = 12,
        dims = ['M'],  # Only output dimensions
        factors_constraints = {'M': 3}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = [],  # Empty = all dimensions allowed
        size = 48*2,  # Reduced size since handling all dimensions
        value_access_energy = 0.69,
        bandwidth = 4,
        factors_constraints = {
            'C1': 2, 'C2': 1},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['R0', 'R1', 'C2'],  # Empty = all dimensions allowed
        size = 192*2,
        value_access_energy = 1.97,
        bandwidth = 4,
        factors_constraints = {'R0': 3, 'R1': 3, 'C2>=': 2},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = [],  # Empty = all dimensions allowed
        size = 16*2,
        value_access_energy = 1.34,
        bandwidth = 4,
        factors_constraints = {'S1': 3, 'S0': 3, 'P': 4, 'K': 8},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21,
        cycles = 1,
        factors_constraints = {}
    )
], coupling=multilayer_conv_coupling, name="Multi-Layer Convolution Architecture")

fixed_multilayer_mapping_conv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2', 'R0', 'R1', 'C2', 'C1', 'S1', 'S0'],  # ONLY final layer + output dims
        size = 2**64-1,
        value_access_energy = 64.00,
        bandwidth = 8,
        factors_constraints = {'Q': 16, 'R2': 1, 'S2': 3},  # Remove conflicting constraints        
        #factors_constraints = {'S1': 1, 'S0': 1, 'R1': 1, 'R0': 1, 'C1': 1, 'C2': 1},  # Remove conflicting constraints
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2', 'R0', 'R1', 'C2', 'C1', 'S1', 'S0'],  # ONLY final layer + output dims
        size = 16384*8,
        value_access_energy = 2.02,
        bandwidth = 32,
        factors_constraints = {'Q': 2, 'P': 64, 'R2': 3},  # Remove conflicting constraints
        bypasses = []
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 14,
        dims = ['Q', 'K'],  # Only final layer dimensions
        factors_constraints = {'Q': 8}
    ),
    FanoutLevel(
        name = "SARows", 
        mesh = 12,
        dims = ['M'],  # Only output dimensions
        factors_constraints = {'M': 3}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2', 'R0', 'R1', 'C2', 'C1', 'S1', 'S0'],  # Empty = all dimensions allowed
        size = 48*2,  # Reduced size since handling all dimensions
        value_access_energy = 0.69,
        bandwidth = 4,
        factors_constraints = {
            'C1': 2, 'C2': 2},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2', 'R0', 'R1', 'C2', 'C1', 'S1', 'S0'],  # Empty = all dimensions allowed
        size = 192*2,
        value_access_energy = 1.97,
        bandwidth = 4,
        factors_constraints = {'R0': 3, 'R1': 3, 'C2': 2},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2', 'R0', 'R1', 'C2', 'C1', 'S1', 'S0'],  # Empty = all dimensions allowed
        size = 16*2,
        value_access_energy = 1.34,
        bandwidth = 4,
        factors_constraints = {'S1': 3, 'S0': 3, 'P': 4, 'K': 8},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21,
        cycles = 1,
        factors_constraints = {}
    )
], coupling = multilayer_conv_coupling, name="Fixed Multi-Layer Convolution Architecture with Mapping Constraints",
)




small_fixed_multilayer_mapping_conv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2', 'R0', 'R1', 'C2', 'C1', 'S1', 'S0'],  # ONLY final layer + output dims
        size = 2**64-1,
        value_access_energy = 64.00,
        bandwidth = 8,
        factors_constraints = {'Q': 2, 'R2': 1, 'S2': 3},  # Remove conflicting constraints        
        #factors_constraints = {'S1': 1, 'S0': 1, 'R1': 1, 'R0': 1, 'C1': 1, 'C2': 1},  # Remove conflicting constraints
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2', 'R0', 'R1', 'C2', 'C1', 'S1', 'S0'],  # ONLY final layer + output dims
        size = 16384*8,
        value_access_energy = 2.02,
        bandwidth = 32,
        factors_constraints = {'Q': 2, 'P': 64, 'R2': 3},  # Remove conflicting constraints
        bypasses = []
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 3,
        dims = ['Q', 'K'],  # Only final layer dimensions
        factors_constraints = {'Q': 3}
    ),
    FanoutLevel(
        name = "SARows", 
        mesh = 12,
        dims = ['M'],  # Only output dimensions
        factors_constraints = {'M': 3}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2', 'R0', 'R1', 'C2', 'C1', 'S1', 'S0'],  # Empty = all dimensions allowed
        size = 48*2,  # Reduced size since handling all dimensions
        value_access_energy = 0.69,
        bandwidth = 4,
        factors_constraints = {
            'C1': 2, 'C2': 2},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2', 'R0', 'R1', 'C2', 'C1', 'S1', 'S0'],  # Empty = all dimensions allowed
        size = 192*2,
        value_access_energy = 1.97,
        bandwidth = 4,
        factors_constraints = {'R0': 3, 'R1': 3, 'C2': 2},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2', 'R0', 'R1', 'C2', 'C1', 'S1', 'S0'],  # Empty = all dimensions allowed
        size = 16*2,
        value_access_energy = 1.34,
        bandwidth = 4,
        factors_constraints = {'S1': 3, 'S0': 3, 'P': 4, 'K': 8},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21,
        cycles = 1,
        factors_constraints = {}
    )
], coupling = multilayer_conv_coupling, name="Fixed Multi-Layer Convolution Architecture with Mapping Constraints",
)

"""
arch = arch_multilayer_conv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2'],  # ONLY final layer + output dims
        size = 2**64-1,
        value_access_energy = 64.00,
        bandwidth = 8,
        factors_constraints = {'S1': 1, 'S0': 1, 'R1': 1, 'R0': 1, 'C1': 1, 'C2': 1},  # Remove conflicting constraints
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R2', 'S2'],  # ONLY final layer + output dims
        size = 16384*8,
        value_access_energy = 2.02,
        bandwidth = 32,
        factors_constraints = {},  # Remove conflicting constraints
        bypasses = ['w']
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 14,
        dims = ['Q', 'K'],  # Only final layer dimensions
        factors_constraints = {}
    ),
    FanoutLevel(
        name = "SARows", 
        mesh = 12,
        dims = ['M'],  # Only output dimensions
        factors_constraints = {}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = [],  # Empty = all dimensions allowed
        size = 48*2,  # Reduced size since handling all dimensions
        value_access_energy = 0.69,
        bandwidth = 4,
        factors_constraints = {
            'C1': 2, 'C2': 4},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = [],  # Empty = all dimensions allowed
        size = 192*2,
        value_access_energy = 1.97,
        bandwidth = 4,
        factors_constraints = {'R0': 3, 'R1': 3},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = [],  # Empty = all dimensions allowed
        size = 16*2,
        value_access_energy = 1.34,
        bandwidth = 4,
        factors_constraints = {'S1': 3, 'S0': 3},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21,
        cycles = 1,
        factors_constraints = {}
    )
], coupling=multilayer_conv_coupling, name="Multi-Layer Convolution Architecture")
"""

# Alternative: 2-layer version for smaller workloads
multilayer_conv_coupling_2layer = create_nlayer_conv_coupling(num_layers=2, with_stride=True, with_batches=True)

arch_multilayer_conv_2layer = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = [],
        size = 2**64-1,
        value_access_energy = 64.00,
        bandwidth = 8,
        factors_constraints = {},
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = [],
        size = 512*1024, # 512KB buffer
        value_access_energy = 3.50,
        bandwidth = 32,
        factors_constraints = {},
        bypasses = []
    ),
    FanoutLevel(
        name = "LayerPEs",
        dims = ['K'],
        mesh = 4,
        factors_constraints = {}
    ),
    MemLevel(
        name = "IntermediateBuffer",
        dataflow_constraints = [],
        size = 16*1024,
        value_access_energy = 1.75,
        bandwidth = 16,
        factors_constraints = {},
        bypasses = []
    ),
    FanoutLevel(
        name = "SpatialPEs", 
        dims = ['P', 'Q'],
        mesh = 8,
        factors_constraints = {}
    ),
    MemLevel(
        name = "WeightCache",
        dataflow_constraints = [],
        size = 4*1024,
        value_access_energy = 1.00,
        bandwidth = 8,
        factors_constraints = {},
        bypasses = ['in', 'out']
    ),
    FanoutLevel(
        name = "FilterPEs",
        dims = ['R0', 'S0', 'R1', 'S1'],
        mesh = 4, # 2x2 for each layer
        factors_constraints = {}
    ),
    MemLevel(
        name = "ChannelBuffer",
        dataflow_constraints = [],
        size = 256,
        value_access_energy = 0.25,
        bandwidth = 4,
        factors_constraints = {},
        bypasses = []
    ),
    FanoutLevel(
        name = "ChannelPEs",
        dims = ['M', 'C1'],
        mesh = 2,
        factors_constraints = {}
    ),
    MemLevel(
        name = "RegisterFile",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'C1', 'R0', 'S0', 'R1', 'S1'],
        size = 4,
        value_access_energy = 0.05,
        bandwidth = 2,
        factors_constraints = {
            'N': 1, 'M': 1, 'P': 1, 'Q': 1, 'K': 1, 'C1': 1,
            'R0': 1, 'S0': 1, 'R1': 1, 'S1': 1
        },
        bypasses = []
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.25,
        cycles = 1,
        factors_constraints = {}
    )
], coupling=multilayer_conv_coupling_2layer, name="2-Layer Convolution Architecture")

arch = small_2_layer_fixed_multilayer_mapping_conv = Arch([
        MemLevel(
        name = "DRAM",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R0', 'R1', 'C1', 'S1', 'S0'],  # ONLY final layer + output dims
        size = 2**64-1,
        value_access_energy = 64.00,
        bandwidth = 8,
        factors_constraints = {'Q': 3, 'P':2, 'K': 2},  # Remove conflicting constraints
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R0', 'R1', 'C1', 'S1', 'S0'],  # ONLY final layer + output dims
        size = 16384*8,
        value_access_energy = 2.02,
        bandwidth = 32,
        factors_constraints = {'P': 3, 'R1': 3},  # Remove conflicting constraints
        bypasses = []
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 6,
        dims = ['Q', 'K'],  # Only final layer dimensions
        factors_constraints = {'Q': 2}
    ),
    FanoutLevel(
        name = "SARows", 
        mesh = 12,
        dims = ['M'],  # Only output dimensions
        factors_constraints = {'M': 3}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R0', 'R1', 'C1', 'S1', 'S0'],  # Empty = all dimensions allowed
        size = 48*2,  # Reduced size since handling all dimensions
        value_access_energy = 0.69,
        bandwidth = 4,
        factors_constraints = {
            'C1': 2},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R0', 'R1', 'C1', 'S1', 'S0'],  # Empty = all dimensions allowed
        size = 192*2,
        value_access_energy = 1.97,
        bandwidth = 4,
        factors_constraints = {'R0': 3},
        bypasses = ['in', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['N', 'M', 'P', 'Q', 'K', 'R0', 'R1', 'C1', 'S1', 'S0'],  # Empty = all dimensions allowed
        size = 16*2,
        value_access_energy = 1.34,
        bandwidth = 4,
        factors_constraints = {'S0': 3, 'S1': 3},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21,
        cycles = 1,
        factors_constraints = {}
    )
], coupling = multilayer_conv_coupling_2layer, name="Fixed Multi-Layer Convolution Architecture with Mapping Constraints",
)
