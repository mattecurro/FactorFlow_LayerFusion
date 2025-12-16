from computations import *
from levels import *
from arch import *


# Create a 3-layer convolution coupling for the architecture
#multilayer_conv_coupling = create_nlayer_conv_coupling(3)
arch_ez_depfin = Arch([
    MemLevel(
        name = "DRAM",
        size = 2**64-1, 
        # 50 pJ/transferred bit (?)
        value_access_energy = 50.0, 
        bandwidth = 8, # as eyeriss (?) 
        bypasses = [],
        dataflow_constraints = ['C2', 'Z2', 'S2', 'Q'], 
        factors_constraints = {'C2': 32, 'Z2': 2, 'S2': 3, 'Q': 3}
    ),
    
    # On-Chip SRAM: DepFiN ha memorie fisicamente separate per pesi e feature
    # "1056kB Features, 524kB Weights" table 11
    # "Both memories consist of multiple banks of single-port SRAMs" 
   
    # input reuse
    FanoutLevel(
        name = "SACols",
        mesh = 128, 
        dims = ['Q'], # Corrisponde a Ox (Output Width)
        factors_constraints = {'Q': 128}
    ),

    # Livello Spaziale 2: Righe (Parallelismo su Output Channels M)
    # "spatially unrolling the output channel (OC) loop" 
    # "16 equivalent weights for 16 different OCs are broadcasted horizontally" 
    FanoutLevel(
        name = "SARows", 
        mesh = 16,
        dims = ['Z2'], # Corrisponde a OC (Output Channels/K)
        factors_constraints = {'Z2': 16}
    ),
   
    MemLevel(
        name = "FeatureMemory", # FMEM
        size = 1056 * 1024, # 1056 kB in bits
        value_access_energy = 2.02, # SRAM eyeriss 
        bandwidth = 128,
        bypasses = ['w'],
        dataflow_constraints = ['C1', 'Z1', 'S1', 'X1', 'Z0', 'X0'],
        factors_constraints = {'C1': 32, 'Z1': 32, 'S1': 3, 'X1': 130, 'Z0': 32, 'X0': 22}                                    
    ),
    
    MemLevel(
        name = "WeightMemory", # WMEM
        size = 524 * 1024, # 524 kB in bits
        value_access_energy = 2.02, # SRAM eyeriss
        bandwidth = 16, # "16 weights are provided in parallel" 
        bypasses = ['in', 'int', 'out'],
        dataflow_constraints = ['C0', 'S0'],
        factors_constraints = {'C0': 32, 'S0': 3}
    ),

    # Registri nei PE
    # "Accumulation: fully in PE" 
    # "Each PE therefore has ten accumulation registers" 
    MemLevel(
        name = "AccumulationOutRegister",
        size = 10, # "Accumulation REGF (10x32b)"
        value_access_energy = 1.34, # As per eyeriss
        bandwidth = 1,
        # Output stationarity: si accumula qui prima di scrivere in FMEM
        bypasses = ['in', 'w'],
        dataflow_constraints = ['X0'],
        factors_constraints = {'X0': 6}
    ),
    

    ComputeLevel(
        name = "Compute",
        mesh = 1, # 1 MAC per PE
        compute_energy = 0.21, # As per eyeriss 12nm
        cycles = 1,
        factors_constraints = {}
    )
], coupling=easy_conv_3layers_coupling, name="DepFiN Architecture")


arch = arch_depfin_complete = Arch([
    MemLevel(
        name = "DRAM",
        size = 2**64-1, 
        # 50 pJ/transferred bit (?)
        value_access_energy = 50.0, 
        bandwidth = 8, # as eyeriss (?) 
        bypasses = [],
        dataflow_constraints = ['C2', 'Z2', 'R2', 'S2', 'Q', 'P'], 
        factors_constraints = {'C2': 1, 'Z2': 1, 'R2': 1, 'S2': 1, 'Q': 10, 'P': 704}
    ),
    
    # On-Chip SRAM: DepFiN ha memorie fisicamente separate per pesi e feature
    # "1056kB Features, 524kB Weights" table 11
    # "Both memories consist of multiple banks of single-port SRAMs" 
      
    MemLevel(
        name = "FeatureMemory", # FMEM
        size = 1056 * 1024, # 1056 kB in bits
        value_access_energy = 2.02, # SRAM eyeriss 
        bandwidth = 128,
        bypasses = ['w'],
        dataflow_constraints = ['C2', 'R2', 'S2', 'C1', 'S1', 'R1', 'C0', 'R0', 'S0', 'Q', 'Z2', 'X1', 'Z1', 'X0', 'Z0'],
        factors_constraints = {'Z2': 1, 'R2': 1, 'S2': 1, 'Q': 1, 'C2': 32, 'C1': 32, 'Z1': 2, 'S1': 3, 'R1': 3, 'X1': 1, 'S0': 3, 'R0': 3, 'Z0': 2, 'X0': 1, 'C0': 3}
    ),
    
    MemLevel(
        name = "WeightMemory", # WMEM
        size = 524 * 1024, # 524 kB in bits
        value_access_energy = 2.02, # SRAM eyeriss
        bandwidth = 16, # "16 weights are provided in parallel" 
        bypasses = ['in', 'int', 'out'],
        dataflow_constraints = ['R0','S0'],
        factors_constraints = {'R0': 1, 'S0': 1}
    ),

    
    # input reuse
    FanoutLevel(
        name = "SACols_2",
        mesh = 128, 
        dims = ['Q'], # Corrisponde a Ox (Output Width)
        #factors_constraints = {'Q': 128}
        factors_constraints = {'Q': 128}
    ),

    # Livello Spaziale 2: Righe (Parallelismo su Output Channels M)
    # "spatially unrolling the output channel (OC) loop" 
    # "16 equivalent weights for 16 different OCs are broadcasted horizontally" 
    FanoutLevel(
        name = "SARows_2", 
        mesh = 16,
        dims = ['Z2'], # Corrisponde a OC (Output Channels/K)
        factors_constraints = {'Z2': 16}
    ),

    FanoutLevel(
        name = "SACols_1",
        mesh = 128, 
        dims = ['X1'], # Corrisponde a Ox (Output Width)
        #factors_constraints = {'Q': 128}
        factors_constraints = {'X1': 128}
    ),

    # Livello Spaziale 2: Righe (Parallelismo su Output Channels M)
    # "spatially unrolling the output channel (OC) loop" 
    # "16 equivalent weights for 16 different OCs are broadcasted horizontally" 
    FanoutLevel(
        name = "SARows_1", 
        mesh = 16,
        dims = ['Z1'], # Corrisponde a OC (Output Channels/K)
        factors_constraints = {'Z1': 16}
    ),

    FanoutLevel(
        name = "SACols_0",
        mesh = 128, 
        dims = ['X0'], # Corrisponde a Ox (Output Width)
        #factors_constraints = {'Q': 128}
        factors_constraints = {'X0': 128}
    ),

    # Livello Spaziale 2: Righe (Parallelismo su Output Channels M)
    # "spatially unrolling the output channel (OC) loop" 
    # "16 equivalent weights for 16 different OCs are broadcasted horizontally" 
    FanoutLevel(
        name = "SARows_0", 
        mesh = 16,
        dims = ['Z0'], # Corrisponde a OC (Output Channels/K)
        factors_constraints = {'Z0': 16}
    ),

    # Registri nei PE
    # "Accumulation: fully in PE" 
    # "Each PE therefore has ten accumulation registers" 
    MemLevel(
        name = "AccumulationOutRegister",
        size = 10, # "Accumulation REGF (10x32b)"
        value_access_energy = 1.34, # As per eyeriss
        bandwidth = 1,
        # Output stationarity: si accumula qui prima di scrivere in FMEM
        bypasses = ['in', 'w'],
        dataflow_constraints = ['X0'],
        factors_constraints = {'X0': 1}
    ),


    ComputeLevel(
        name = "Compute",
        mesh = 1, # 1 MAC per PE
        compute_energy = 0.21, # As per eyeriss 12nm
        cycles = 1,
        factors_constraints = {}
    )
], coupling=conv_3layers_coupling, name="DepFiN Architecture")

"""
arch_depfin_mapping_changed = Arch([
    MemLevel(
        name = "DRAM",
        size = 2**64-1, 
        # 50 pJ/transferred bit (?)
        value_access_energy = 50.0, 
        bandwidth = 8, # as eyeriss (?) 
        bypasses = [],
        dataflow_constraints = ['C2', 'Z2', 'S2', 'Q'], 
        factors_constraints = {'C2': 32, 'Z2': 2, 'S2': 3, 'Q': 3}
    ),
    
    # On-Chip SRAM: DepFiN ha memorie fisicamente separate per pesi e feature
    # "1056kB Features, 524kB Weights" table 11
    # "Both memories consist of multiple banks of single-port SRAMs" 
   
    # input reuse
    
    FanoutLevel(
        name = "SACols_0",
        mesh = 128, 
        dims = ['Q'], # Corrisponde a Ox (Output Width)
        factors_constraints = {'Q': 128}
    ),

    # Livello Spaziale 2: Righe (Parallelismo su Output Channels M)
    # "spatially unrolling the output channel (OC) loop" 
    # "16 equivalent weights for 16 different OCs are broadcasted horizontally" 
    FanoutLevel(
        name = "SARows_0", 
        mesh = 16,
        dims = ['Z2'], # Corrisponde a OC (Output Channels/K)
        factors_constraints = {'Z2': 16}
    ),
   
    # innermost Cout, Q
    MemLevel(
        name = "FeatureMemory", # FMEM
        size = 1056 * 1024, # 1056 kB in bits
        value_access_energy = 2.02, # SRAM eyeriss 
        bandwidth = 128,
        bypasses = ['w'],
        dataflow_constraints = ['C1', 'Z1', 'Z0', 'C0'],
        factors_constraints = {'C1': 32, 'Z1': 2, 'Z0': 2, 'C0': 32}                                    
    ),
    # weight reuse
 
    # R1234, S1234
    # Y1234, X12424
    MemLevel(
        name = "WeightMemory", # WMEM
        size = 524 * 1024, # 524 kB in bits
        value_access_energy = 2.02, # SRAM eyeriss
        bandwidth = 16, # "16 weights are provided in parallel" 
        bypasses = ['in', 'int', 'out'],
        dataflow_constraints = ['S1', 'S0'],
        factors_constraints = {'S1': 3, 'S0': 3}
    ),

    # Livello Spaziale 1: Colonne (Parallelismo su Output Width Q)
    # "applying 128x spatial unrolling of the horizontal FM dimension (Ox)" 
    # "128 horizontally neighboring input features are broadcasted vertically" 
    
    FanoutLevel(
        name = "SACols_1",
        mesh = 128,
        dims = ['X1'], # Corrisponde a Ox (Output Width)
        factors_constraints = {'X1': 128}
    ),

    FanoutLevel(
        name = "SARows_1", 
        mesh = 16,
        dims = ['Z1'], # Corrisponde a OC (Output Channels/K)
        factors_constraints = {'Z1': 16}
    ),

    FanoutLevel(
        name = "SACols_2",
        mesh = 128,
        dims = ['X0'], # Corrisponde a Ox (Output Width)
        factors_constraints = {'X0': 128}
    ),

    FanoutLevel(
        name = "SARows_2",
        mesh = 16,
        dims = ['Z0'], # Corrisponde a OC (Output Channels/K)
        factors_constraints = {'Z0': 16}
    ),

    # Registri nei PE
    # "Accumulation: fully in PE" 
    # "Each PE therefore has ten accumulation registers" 
    MemLevel(
        name = "AccumulationOutRegister",
        size = 10, # "Accumulation REGF (10x32b)"
        value_access_energy = 1.34, # As per eyeriss
        bandwidth = 1,
        # Output stationarity: si accumula qui prima di scrivere in FMEM
        bypasses = ['in', 'w']
    ),
    

    ComputeLevel(
        name = "Compute",
        mesh = 1, # 1 MAC per PE
        compute_energy = 0.21, # As per eyeriss 12nm
        cycles = 1,
        factors_constraints = {}
    )
], coupling=easy_conv_2layers_coupling, name="DepFiN Architecture")

fixed_multilayer_mapping_conv = Arch([
    MemLevel(
        name = "MainMemory",
        size = 4096 * 32,  # depth * block_size (in bytes, assuming 8-bit words)
        # Calculate value_access_energy from Timeloop's energy model
        # You'll need to run Timeloop's energy calculations or use default values
        # For DRAM, typical values range from 50-200 pJ per access
        value_access_energy = 2048.0,  # Placeholder - adjust based on your needs
        # Calculate bandwidth: width / cycle_time
        # 256 bits / 1ns = 256 Gbps = 32 GB/s = 32 bytes/cycle (assuming 1GHz)
        bandwidth = 32,  # bytes per cycle (256 bits / 8)
        leakage_energy = 0.0,  # Add if you have leakage data
        dataflow_constraints = ['P', 'C0', 'Y', 'Z', 'C1', 'C2'],  
        factors_constraints = {'P': 128, 'C0': 64, 'Y': 128, 'Z': 64, 'C1': 64 , 'C2': 64}, 
        bypasses = ['int']
    ),
    MemLevel(
        name = "GlobalBuffer",
        size = 8192 * 32,  # depth * block_size (in bytes)
        # For SRAM, typical values range from 2-10 pJ per access
        value_access_energy = 121,  # Placeholder - adjust based on your needs
        # With 2 read/write ports, bandwidth might be doubled
        bandwidth = 64,  # 2 ports * (256 bits / 8) = 64 bytes/cycle
        leakage_energy = 0.0,  # Add if you have leakage data
        dataflow_constraints = [],  
        factors_constraints = {},  
        bypasses = []
    ),
    ComputeLevel(
        name = "MACC",
        mesh = 1,  # Number of concurrent MACs - adjust based on your design
        # For 8-bit integer MAC, typical values range from 0.2-0.5 pJ
        compute_energy = 0.85,  # Placeholder - adjust based on your needs
        cycles = 1,  # 1 cycle per MAC operation
        leakage_energy = 0.0,  # Add if you have leakage data
        factors_constraints = {}  # You mentioned you'll fill this
    )
], coupling = gemm_2layers_coupling, name = "Fixed Multi-Layer GEMM Architecture with Mapping Constraints")
"""
"""
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
"""
"""
fixed_multilayer_mapping_conv = Arch([
    MemLevel(
        name = "DRAM",
        dataflow_constraints = ['P', 'Q', 'C1'],  
        size = 2**64-1,
        value_access_energy = 64.00,
        bandwidth = 8,
        factors_constraints = {'Q': 3, 'P': 2, 'C1': 2},  
        bypasses = []
    ),
    MemLevel(
        name = "GlobalBuffer",
        dataflow_constraints = ['P', 'X', 'R1', 'Y'],  
        size = 16384*100,
        value_access_energy = 2.02,
        bandwidth = 32,
        factors_constraints = {'P': 3, 'X': 2, 'R1': 3, 'Y': 4},  
        bypasses = []
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 4,
        dims = ['Y', 'Q'],  
        factors_constraints = {'Y': 2, 'Q': 2}
    ),
    FanoutLevel(
        name = "SARows", 
        mesh = 2,
        dims = ['C1'],  # Only output dimensions
        factors_constraints = {'C1': 2}
    ),
    MemLevel(
        name = "InRegister",
        dataflow_constraints = ['S1', 'Z'],
        size = 48*2,  # Reduced size since handling all dimensions
        value_access_energy = 0.69,
        bandwidth = 4,
        factors_constraints = {
            'S1': 3, 'Z': 4},
        bypasses = ['w', 'out']
    ),
    MemLevel(
        name = "WRegister",
        dataflow_constraints = ['C2'],
        size = 192*2,
        value_access_energy = 1.97,
        bandwidth = 4,
        factors_constraints = {'C2': 2},
        bypasses = ['in', 'int', 'out']
    ),
    MemLevel(
        name = "OutRegister",
        dataflow_constraints = ['X','S0','R0','C0'],
        size = 16*2,
        value_access_energy = 1.34,
        bandwidth = 4,
        factors_constraints = {'X': 4, 'S0': 3, 'R0': 3, 'C0': 3},
        bypasses = ['in', 'w']
    ),
    ComputeLevel(
        name = "Compute",
        mesh = 1,
        compute_energy = 0.21,
        cycles = 1,
        factors_constraints = {}
    )
], coupling = conv_2layers_coupling, name="Fixed Multi-Layer Convolution Architecture with Mapping Constraints",
)
""""""
"""
"""
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
#multilayer_conv_coupling_2layer = create_nlayer_conv_coupling(num_layers=2, with_stride=True, with_batches=True)
"""
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

small_2_layer_fixed_multilayer_mapping_conv = Arch([
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
        dataflow_constraints = ['N', 'Q', 'K', 'S1', 'C1', 'R0', 'S0', 'M', 'P', 'R1'],  # ONLY final layer + output dims
        size = 16384*8,
        value_access_energy = 2.02,
        bandwidth = 32,
        factors_constraints = {'P': 3, 'R1': 3},  # Remove conflicting constraints
        bypasses = []
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 2,
        dims = ['Q', 'K'],  # Only final layer dimensions
        factors_constraints = {'Q': 2}
    ),
    FanoutLevel(
        name = "SARows", 
        mesh = 3,
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
            'C1': 2, 'S1': 3},
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
        factors_constraints = {'S0': 3},
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

no_bypass_small_2_layer_fixed_multilayer_mapping_conv = Arch([
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
        dataflow_constraints = ['N', 'Q', 'K', 'S1', 'C1', 'R0', 'S0', 'M', 'P', 'R1'],  # ONLY final layer + output dims
        size = 16384*8,
        value_access_energy = 2.02,
        bandwidth = 32,
        factors_constraints = {'P': 3, 'R1': 3},  # Remove conflicting constraints
        bypasses = []
    ),
    FanoutLevel(
        name = "SACols",
        mesh = 2,
        dims = ['Q', 'K'],  # Only final layer dimensions
        factors_constraints = {'Q': 2}
    ),
    FanoutLevel(
        name = "SARows", 
        mesh = 3,
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
            'C1': 2, 'S1': 3},
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
        factors_constraints = {'S0': 3},
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
"""