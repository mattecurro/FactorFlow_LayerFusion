from computations import *
from levels import *
from arch import *
from architectures.arch_hw_data import get_depfin_energy_per_byte, DepFinParamInfo

# Validation CL: python3 main_cli.py architectures/multi_layer_arch.py architectures/10layer_comp.py > output_FW_int.log 2>&1

def get_scaled_energy_values(
    feature_memory_size_kb: int,
    weight_memory_size_kb: int,
    technology: str = "22nm"
) -> dict:
    """
    Calculate energy values that scale with memory sizes using Accelergy/CACTI.
    """
    param = DepFinParamInfo()
    param.FMEM_size = feature_memory_size_kb * 1024 * 8  # Convert KB to bits
    param.WMEM_size = weight_memory_size_kb * 1024 * 8
    param.technology = technology    
    energies = get_depfin_energy_per_byte(param)
    
    return {
        'dram_energy': energies['DRAM']['read_energy_per_byte'],
        'fmem_energy': energies['FeatureMemory']['read_energy_per_byte'],
        'wmem_energy': energies['WeightMemory']['read_energy_per_byte'],
        'accreg_energy': energies['AccumulationRegister']['read_energy_per_byte'],
        'compute_energy': energies['Compute']['fma_energy'],
    }

""" 
    MemLevel(
        name = "WeightMemory", # WMEM
        size = 524 * 1024, 
        value_access_energy = 0.5*8, 
        bandwidth = 2*16,                 #FORZATURA read_bandwidth = 16, write_bandwidth = 16
        bypasses = ['in', 'int', 'out'],
        dataflow_constraints = [           
            'C10', 'R10', 'S10',
            'C9', 'R9', 'S9', 'C8', 'S8', 'R8', 'C7', 'S7', 'R7', 'C6', 'S6', 'R6', 
            'C5', 'S5', 'R5', 'C4', 'S4', 'R4', 'C3', 'S3', 'R3', 'C2', 'S2', 'R2', 
            'C1', 'S1', 'R1', 'C0', 
            'Q', 'Z9', 'X8', 'Z8', 'X7', 'Z7', 'X6', 'Z6', 'X5', 'Z5', 
            'X4', 'Z4', 'X3', 'Z3', 'X2', 'Z2', 'X1', 'Z1', 'X0', 'Z0', 'R0','S0'], # Just innermost weights
        # Constraints for all weights
        factors_constraints = {            
            'Z9': 2,
            'Z8': 2,
            'Z7': 2,
            'Z6': 2,
            'Z5': 2,
            'Z4': 2,
            'Z3': 2,
            'Z2': 2,
            'Z1': 2,
            'Z0': 2, 
            'R9': 3, 'S9': 3, 'S8': 3, 'R8': 3, 'S7': 3, 'R7': 3, 'S6': 3, 'R6': 3, 'S5': 3, 'R5': 3,
            'S4': 3, 'R4': 3, 'S3': 3, 'R3': 3, 'S2': 3, 'R2': 3, 'S1': 3, 'R1': 3,
            'R0': 3, 'S0': 3
        }
    ),


MemLevel(
    name = "AccumulationReg_10",
    size = 322,
    value_access_energy = 1.34, 
    read_bandwidth = 1,
    write_bandwidth = 1,
    bypasses = ['in', 'w'], 
    dataflow_constraints = ['Q', 'P', 'Z10', 'C10', 'R10', 'S10'],
    factors_constraints = {'C10': 32}
),
MemLevel(
    name = "AccumulationReg_9",
    size = 332,
    value_access_energy = 1.34, 
    read_bandwidth = 1,
    write_bandwidth = 1,
    bypasses = ['in', 'w', 'out'], 
    dataflow_constraints = ['X9', 'Y9', 'Z9', 'C9', 'R9', 'S9'],
    factors_constraints = {'C9': 32}
),
MemLevel(
    name = "AccumulationReg_8",
    size = 332,
    value_access_energy = 1.34, 
    read_bandwidth = 1,
    write_bandwidth = 1,
    bypasses = ['in', 'w', 'out'], 
    dataflow_constraints = ['X8', 'Y8', 'Z8', 'C8', 'R8', 'S8'],
    factors_constraints = {'C8': 32}
),
MemLevel(
    name = "AccumulationReg_7",
    size = 256,
    value_access_energy = 1.34, 
    read_bandwidth = 1,
    write_bandwidth = 1,
    bypasses = ['in', 'w', 'out'], 
    dataflow_constraints = ['X7', 'Y7', 'Z7', 'C7', 'R7', 'S7'],
    factors_constraints = {'C7': 32}
),
MemLevel(
    name = "AccumulationReg_6",
    size = 256,
    value_access_energy = 1.34, 
    read_bandwidth = 1,
    write_bandwidth = 1,
    bypasses = ['in', 'w', 'out'], 
    dataflow_constraints = ['X6', 'Y6', 'Z6', 'C6', 'R6', 'S6'],
    factors_constraints = {'C6': 32}
),
MemLevel(
    name = "AccumulationReg_5",
    size = 256,
    value_access_energy = 1.34, 
    read_bandwidth = 1,
    write_bandwidth = 1,
    bypasses = ['in', 'w', 'out'], 
    dataflow_constraints = ['X5', 'Y5', 'Z5', 'C5', 'R5', 'S5'],
    factors_constraints = {'C5': 32}
),
MemLevel(
    name = "AccumulationReg_4",
    size = 256,
    value_access_energy = 1.34, 
    read_bandwidth = 1,
    write_bandwidth = 1,
    bypasses = ['in', 'w'], 
    dataflow_constraints = ['X4', 'Y4', 'Z4', 'C4', 'R4', 'S4'],
    factors_constraints = {'C4': 32}
),
MemLevel(
    name = "AccumulationReg_3",
    size = 256,
    value_access_energy = 1.34, 
    read_bandwidth = 1,
    write_bandwidth = 1,
    bypasses = ['in', 'w', 'out'], 
    dataflow_constraints = ['X3', 'Y3', 'Z3', 'C3', 'R3', 'S3'],
    factors_constraints = {'C3': 32}
),
MemLevel(
    name = "AccumulationReg_2",
    size = 256,
    value_access_energy = 1.34, 
    read_bandwidth = 1,
    write_bandwidth = 1,
    bypasses = ['in', 'w', 'out'], 
    dataflow_constraints = ['X2', 'Y2', 'Z2', 'C2', 'R2', 'S2'],
    factors_constraints = {'C2': 32}
),
MemLevel(
    name = "AccumulationReg_1",
    size = 256,
    value_access_energy = 1.34, 
    read_bandwidth = 1,
    write_bandwidth = 1,
    bypasses = ['in', 'w', 'out'], 
    dataflow_constraints = ['X1', 'Y1', 'Z1', 'C1', 'R1', 'S1'],
    factors_constraints = {'C1': 32}
),
MemLevel(
    name = "AccumulationReg_0",
    size = 256,
    value_access_energy = 1.34, 
    read_bandwidth = 1,
    write_bandwidth = 1,
    bypasses = ['in', 'w', 'out'], 
    dataflow_constraints = ['X0', 'Y0', 'Z0', 'C0', 'R0', 'S0'],
    factors_constraints = {'C0': 3}
),
"""
""" 
To perform: Home-made Workload by DepFiN
DRAM BandwidthCode: 
External IO bandwidth of 12 B (96/8) as stated in the figure.
Frequency: The core runs at 190/680 MHz per clock cycle == 930 M cycles/s 

# 17GB/s / 930MHz = 18 B per cycle
"""
arch = arch_depfin_10layers_F1S = Arch([
    MemLevel(
        name = "DRAM",
        size = 2**64-1, 
#        value_access_energy = 400.0, 
        read_value_access_energy = 160,
        write_value_access_energy = 160,
        # The Depth-First mapping ensure to me that read are done before writes, so I can assume 12 for both.
        read_bandwidth = 18,
        write_bandwidth = 18, 
        bypasses = ['int_in', 'int_out'],
        # Constraints for the outermost layer (Layer 9)
        dataflow_constraints = ['Q', 'P', 'X9', 'Y9', 'X8', 'Y8', 'X7', 'Y7', 'X6', 'Y6', 
                                 'X5', 'Y5', 'X4', 'Y4', 'X3', 'Y3', 
                                 'X2', 'Y2', 'X1', 'Y1', 'X0', 'Y0'], 
        factors_constraints = {'Q': 10, 'P': 720, 'X9': 10, 'Y9': 720, 'X8': 10, 'Y8': 720, 'X7': 10, 'Y7': 720, 'X6': 10, 'Y6': 720, 
                               'X5': 10, 'Y5': 720, 'X4': 10, 'Y4': 720, 'X3': 10, 'Y3': 720, 
                               'X2': 10, 'Y2': 720, 'X1': 10, 'Y1': 720, 'X0': 10,  'Y0': 720}
    ),

    MemLevel(
        name = "FeatureMemory", # FMEM
        size = 1056 * 1024, 
        value_access_energy = 0.75, 
        bandwidth = 132*2,
        read_bandwidth = 132,
        write_bandwidth = 128,   
        bypasses = ['w'],
        dataflow_constraints = [],
        factors_constraints = {}
    ),


    MemLevel(
        name = "WeightMemory", # WMEM
        size = 524 * 1024, 
        value_access_energy = 0.71, 
        bandwidth = 2*16,                 #FORZATURA read_bandwidth = 16, write_bandwidth = 16
        bypasses = ['in', 'int_in', 'int_out', 'out'],
        dataflow_constraints = [           
            'Z10', 'C10', 'R10', 'S10', 
            'Z9', 'C9', 'R9', 'S9',
            'Z8', 'C8', 'S8', 'R8', 
            'Z7', 'C7', 'S7', 'R7', 
            'Z6', 'C6', 'S6', 'R6',
            'Z5', 'C5', 'S5', 'R5',
            'Z4', 'C4', 'S4', 'R4',
            'Z3', 'C3', 'S3', 'R3',
            'Z2', 'C2', 'S2', 'R2',
            'Z1', 'C1', 'S1', 'R1',
            'Z0', 'C0', 'R0', 'S0'], # Just innermost weights
        # Constraints for all weights
        factors_constraints = {            
            'C10': 32, 
            'C9': 32, 'Z9': 2,
            'C8': 32, 'Z8': 2,
            'C7': 32, 'Z7': 2,
            'C6': 32, 'Z6': 2,
            'C5': 32, 'Z5': 2,
            'C4': 32, 'Z4': 2,
            'C3': 32, 'Z3': 2,
            'C2': 32, 'Z2': 2,
            'C1': 32, 'Z1': 2,
            'Z0': 2, 'C0': 3, 
            'R9': 3, 'S9': 3, 'S8': 3, 'R8': 3, 'S7': 3, 'R7': 3, 'S6': 3, 'R6': 3, 'S5': 3, 'R5': 3,
            'S4': 3, 'R4': 3, 'S3': 3, 'R3': 3, 'S2': 3, 'R2': 3, 'S1': 3, 'R1': 3,
            'R0': 3, 'S0': 3
        }
    ),


    # --- Layer 10 ---
    FanoutLevel(name = "SACols_10", mesh = 128, dims = ['Q'], factors_constraints = {'Q': 128}, spatial_reduction_support = True),
    FanoutLevel(name = "SARows_10", mesh = 16, dims = ['Z10'], factors_constraints = {'Z10': 16}),

    # --- Layer 9  ---
    FanoutLevel(name = "SACols_9", mesh = 128, dims = ['X9'], factors_constraints = {'X9': 128}, spatial_reduction_support = True),
    FanoutLevel(name = "SARows_9", mesh = 16, dims = ['Z9'], factors_constraints = {'Z9': 16}), 

    # --- Layer 8 ---
    FanoutLevel(name = "SACols_8", mesh = 128, dims = ['X8'], factors_constraints = {'X8': 128}),
    FanoutLevel(name = "SARows_8", mesh = 16, dims = ['Z8'], factors_constraints = {'Z8': 16}),

    # --- Layer 7 ---
    FanoutLevel(name = "SACols_7", mesh = 128, dims = ['X7'], factors_constraints = {'X7': 128}),
    FanoutLevel(name = "SARows_7", mesh = 16, dims = ['Z7'], factors_constraints = {'Z7': 16}),

    # --- Layer 6 ---
    FanoutLevel(name = "SACols_6", mesh = 128, dims = ['X6'], factors_constraints = {'X6': 128}),
    FanoutLevel(name = "SARows_6", mesh = 16, dims = ['Z6'], factors_constraints = {'Z6': 16}),

    # --- Layer 5 ---
    FanoutLevel(name = "SACols_5", mesh = 128, dims = ['X5'], factors_constraints = {'X5': 128}),
    FanoutLevel(name = "SARows_5", mesh = 16, dims = ['Z5'], factors_constraints = {'Z5': 16}),

    # --- Layer 4 ---
    FanoutLevel(name = "SACols_4", mesh = 128, dims = ['X4'], factors_constraints = {'X4': 128}),
    FanoutLevel(name = "SARows_4", mesh = 16, dims = ['Z4'], factors_constraints = {'Z4': 16}),

    # --- Layer 3 ---
    FanoutLevel(name = "SACols_3", mesh = 128, dims = ['X3'], factors_constraints = {'X3': 128}),
    FanoutLevel(name = "SARows_3", mesh = 16, dims = ['Z3'], factors_constraints = {'Z3': 16}),

    # --- Layer 2 ---
    FanoutLevel(name = "SACols_2", mesh = 128, dims = ['X2'], factors_constraints = {'X2': 128}),
    FanoutLevel(name = "SARows_2", mesh = 16, dims = ['Z2'], factors_constraints = {'Z2': 16}),

    # --- Layer 1 ---
    FanoutLevel(name = "SACols_1", mesh = 128, dims = ['X1'], factors_constraints = {'X1': 128}),
    FanoutLevel(name = "SARows_1", mesh = 16, dims = ['Z1'], factors_constraints = {'Z1': 16}),

    # --- Layer 0 (Input) ---
    FanoutLevel(name = "SACols_0", mesh = 128, dims = ['X0'], factors_constraints = {'X0': 128}),
    FanoutLevel(name = "SARows_0", mesh = 16, dims = ['Z0'], factors_constraints = {'Z0': 16}),

    MemLevel(
        name = "AccumulationIntermediateOutputRegister",
        size = 11,
        value_access_energy = 0.16,
        read_bandwidth = 1,
        write_bandwidth = 1,
        bypasses = ['in', 'w', 'int_in'],
        dataflow_constraints = ['Z10', 'P', 'Q', 'Z9', 'Y9', 'X9', 'Z8', 'Y8', 'X8', 'Z7', 'Y7', 'X7', 'Z6', 'Y6', 'X6', 'Z5', 'Y5', 'X5', 'Z4', 'Y4', 'X4', 'Z3', 'Y3', 'X3', 'Z2', 'Y2', 'X2', 'Z1', 'Y1', 'X1', 'Z0', 'Y0', 'X0'],
        factors_constraints = {'Z10': 1, 'P': 1, 'Q': 1, 'Z9': 1, 'Y9': 1, 'X9': 1, 'Z8': 1, 'Y8': 1, 'X8': 1, 'Z7': 1, 'Y7': 1, 'X7': 1, 'Z6': 1, 'Y6': 1, 'X6': 1, 'Z5': 1, 'Y5': 1, 'X5': 1, 'Z4': 1, 'Y4': 1, 'X4': 1, 'Z3': 1, 'Y3': 1, 'X3': 1, 'Z2': 1, 'Y2': 1, 'X2': 1, 'Z1': 1, 'Y1': 1, 'X1': 1,        'Z0': 1, 'Y0': 1, 'X0': 1}
    ),

    ComputeLevel(
        name = "Compute",
        mesh = 1, 
        compute_energy = 0.20, 
        leakage_energy = 0.001,
        cycles = 1,
        factors_constraints = {}
    )
], coupling=conv_10layers_coupling, name="DepFiN 10-Layer Architecture")


# NOTE: For parameterized thesis experiments, use thesis_arch.py instead
# The create_thesis_architecture function has been moved to architectures/thesis_arch.py
