from computations import *
from levels import *
from arch import *

"""
Int in FMem
"""
arch_depfin_10layers_F1S_in_FMem = Arch([
    MemLevel(
        name = "DRAM",
        size = 2**64-1, 
#        value_access_energy = 400.0, 
        read_value_access_energy = 400.0,
        write_value_access_energy = 400.0,
        # The Depth-First mapping ensure to me that read are done before writes, so I can assume 12 for both.
        read_bandwidth = 12,
        write_bandwidth = 12, 
        bypasses = ['int_in', 'int_out'],
        # Constraints for the outermost layer (Layer 9)
        dataflow_constraints = ['Q', 'P', 'X0', 'Y0'], 
        factors_constraints = {'Q': 10, 'P': 720, 'X0': 10,  'Y0': 720}
    ),

    MemLevel(
        name = "FeatureMemory", # FMEM
        size = 1056 * 10240000000, 
        value_access_energy = 8, 
        bandwidth = 132*2,
        read_bandwidth = 132,
        write_bandwidth = 128,   
        bypasses = ['w'],
        dataflow_constraints = ['X9', 'Y9', 'X8', 'Y8', 'X7', 'Y7', 'X6', 'Y6', 
                                 'X5', 'Y5', 'X4', 'Y4', 'X3', 'Y3', 
                                 'X2', 'Y2', 'X1', 'Y1'],
        factors_constraints = {'X9': 10, 'Y9': 720, 'X8': 10, 'Y8': 720, 'X7': 10, 'Y7': 720, 'X6': 10, 'Y6': 720, 
                               'X5': 10, 'Y5': 720, 'X4': 10, 'Y4': 720, 'X3': 10, 'Y3': 720, 
                               'X2': 10, 'Y2': 720, 'X1': 10, 'Y1': 720}
    ),

    MemLevel(
        name = "WeightMemory", # WMEM
        size = 524 * 1024, 
        value_access_energy = 0.5*8, 
        bandwidth = 2*16,                 #FORZATURA read_bandwidth = 16, write_bandwidth = 16
        bypasses = ['in', 'int_in', 'int_out', 'out'],
        dataflow_constraints = [           
            'C10', 'R10', 'S10',
            'C9', 'R9', 'S9', 'C8', 'S8', 'R8', 'C7', 'S7', 'R7', 'C6', 'S6', 'R6', 
            'C5', 'S5', 'R5', 'C4', 'S4', 'R4', 'C3', 'S3', 'R3', 'C2', 'S2', 'R2', 
            'C1', 'S1', 'R1', 'C0', 
            'Q', 'Z9', 'X8', 'Z8', 'X7', 'Z7', 'X6', 'Z6', 'X5', 'Z5', 
            'X4', 'Z4', 'X3', 'Z3', 'X2', 'Z2', 'X1', 'Z1', 'X0', 'Z0', 'R0','S0'], # Just innermost weights
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
    FanoutLevel(name = "SACols_10", mesh = 128, dims = ['Q'], factors_constraints = {'Q': 128}),
    FanoutLevel(name = "SARows_10", mesh = 16, dims = ['Z10'], factors_constraints = {'Z10': 16}),

    # --- Layer 9  ---
    FanoutLevel(name = "SACols_9", mesh = 128, dims = ['X9'], factors_constraints = {'X9': 128}),
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

#    MemLevel(
#        name = "AccumulationOutRegister",
#        size = 28,                  # FORZATURA "Accumulation REGF (28x32b)"      128/8 
#        value_access_energy = 1.34, 
#        bandwidth = 2*2,
#        bypasses = ['in', 'w'],
#        dataflow_constraints = ['Q'],
#        factors_constraints = {'Q': 10}
#    ),

    ComputeLevel(
        name = "Compute",
        mesh = 1, 
        compute_energy = 0.21, 
        cycles = 1,
        factors_constraints = {}
    )
], coupling=conv_10layers_coupling, name="DepFiN 10-Layer Architecture")


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
        read_value_access_energy = 200,
        write_value_access_energy = 200,
        # The Depth-First mapping ensure to me that read are done before writes, so I can assume 12 for both.
        read_bandwidth = 2,
        write_bandwidth = 4, 
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
        size = 10,
        value_access_energy = 0.16,
        read_bandwidth = 1,
        write_bandwidth = 1,
        bypasses = ['in', 'w', 'out', 'int_in'],
        dataflow_constraints = ['Z9', 'Y9', 'X9', 'Z8', 'Y8', 'X8', 'Z7', 'Y7', 'X7', 'Z6', 'Y6', 'X6', 'Z5', 'Y5', 'X5', 'Z4', 'Y4', 'X4', 'Z3', 'Y3', 'X3', 'Z2', 'Y2', 'X2', 'Z1', 'Y1', 'X1',        'Z0', 'Y0', 'X0'],
        factors_constraints = {'Z9': 1, 'Y9': 1, 'X9': 1, 'Z8': 1, 'Y8': 1, 'X8': 1, 'Z7': 1, 'Y7': 1, 'X7': 1, 'Z6': 1, 'Y6': 1, 'X6': 1, 'Z5': 1, 'Y5': 1, 'X5': 1, 'Z4': 1, 'Y4': 1, 'X4': 1, 'Z3': 1, 'Y3': 1, 'X3': 1, 'Z2': 1, 'Y2': 1, 'X2': 1, 'Z1': 1, 'Y1': 1, 'X1': 1,        'Z0': 1, 'Y0': 1, 'X0': 1}
    ),

    MemLevel(
        name = "AccumulationOutputRegister",
        size = 1,
        value_access_energy = 0.16,
        read_bandwidth = 1,
        write_bandwidth = 1,
        bypasses = ['in', 'w', 'int_in', 'int_out'],
        dataflow_constraints = ['Q', 'P', 'Z10'],
        factors_constraints = {'Q': 1, 'P': 1, 'Z10': 1}
    ),

    ComputeLevel(
        name = "Compute",
        mesh = 1, 
        compute_energy = 0.202, 
        leakage_energy = 0.001,
        cycles = 1,
        factors_constraints = {}
    )
], coupling=conv_10layers_coupling, name="DepFiN 10-Layer Architecture")



arch_depfin_10layers = Arch([
    MemLevel(
        name = "DRAM",
        size = 2**64-1, 
        value_access_energy = 400.0, 
        read_value_access_energy = 400.0,
        write_value_access_energy = 0.000000001,
        # The Depth-First mapping ensure to me that read are done before writes, so I can assume 12 for both.
        read_bandwidth = 12,
        write_bandwidth = 12, 
        bypasses = ['int'],
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
        value_access_energy = 0.5*8, 
        bandwidth = 132*2,
        read_bandwidth = 132,
        write_bandwidth = 128,   
        bypasses = ['w'],
        dataflow_constraints = [
            'C10', 'R10', 'S10',
            'C9', 'R9', 'S9', 'C8', 'S8', 'R8', 'C7', 'S7', 'R7', 'C6', 'S6', 'R6', 
            'C5', 'S5', 'R5', 'C4', 'S4', 'R4', 'C3', 'S3', 'R3', 'C2', 'S2', 'R2', 
            'C1', 'S1', 'R1', 'C0', 'R0', 'S0', 
            'Q', 'Z9', 'X8', 'Z8', 'X7', 'Z7', 'X6', 'Z6', 'X5', 'Z5', 
            'X4', 'Z4', 'X3', 'Z3', 'X2', 'Z2', 'X1', 'Z1', 'X0', 'Z0'
        ],
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
            'Z0': 2, 'C0': 3, 'R9': 3, 'S9': 3, 'S8': 3, 'R8': 3, 'S7': 3, 'R7': 3, 'S6': 3, 'R6': 3, 'S5': 3, 'R5': 3,
            'S4': 3, 'R4': 3, 'S3': 3, 'R3': 3, 'S2': 3, 'R2': 3, 'S1': 3, 'R1': 3
        }
    ),

    MemLevel(
        name = "WeightMemory", # WMEM
        size = 524 * 1024, 
        value_access_energy = 0.5*8, 
        bandwidth = 2*16,                 #FORZATURA read_bandwidth = 16, write_bandwidth = 16
        bypasses = ['in', 'int', 'out'],
        dataflow_constraints = ['R0','S0'], # Just innermost weights
        # Constraints for all weights
        factors_constraints = {
            'R0': 3, 'S0': 3
        }
    ),

    # --- Layer 10 ---
    FanoutLevel(name = "SACols_10", mesh = 128, dims = ['Q'], factors_constraints = {'Q': 128}),
    FanoutLevel(name = "SARows_10", mesh = 16, dims = ['Z10'], factors_constraints = {'Z10': 16}),

    # --- Layer 9  ---
    FanoutLevel(name = "SACols_9", mesh = 128, dims = ['X9'], factors_constraints = {'X9': 128}),
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

#    MemLevel(
#        name = "AccumulationOutRegister",
#        size = 28,                  # FORZATURA "Accumulation REGF (28x32b)"      128/8 
#        value_access_energy = 1.34, 
#        bandwidth = 2*2,
#        bypasses = ['in', 'w'],
#        dataflow_constraints = ['Q'],
#        factors_constraints = {'Q': 10}
#    ),

    ComputeLevel(
        name = "Compute",
        mesh = 1, 
        compute_energy = 0.21, 
        cycles = 1,
        factors_constraints = {}
    )
], coupling=conv_10layers_coupling, name="DepFiN 10-Layer Architecture")

arch_depfin_10layers_FSRCNN = Arch([
    MemLevel(
        name = "DRAM",
        size = 2**64-1, 
        value_access_energy = 50.0, 
        #bandwidth = 18*2,                           # 17GB/s / 930MHz = 18 B per cycle
        read_bandwidth = 17,
        write_bandwidth = 2,
        bypasses = ['int'],
        # Constraints for the outermost layer (Layer 9)
        dataflow_constraints = ['Q', 'P', 'X8', 'Y8', 'X7', 'Y7', 'X6', 'Y6', 
                                 'X5', 'Y5', 'X4', 'Y4', 'X3', 'Y3', 
                                 'X2', 'Y2', 'X1', 'Y1', 'X0', 'Y0'], 
        factors_constraints = {'Q': 10, 'P': 720, 'X8': 10, 'Y8': 720, 'X7': 10, 'Y7': 720, 'X6': 10, 'Y6': 720, 
                               'X5': 10, 'Y5': 720, 'X4': 10, 'Y4': 720, 'X3': 10, 'Y3': 720, 
                               'X2': 10, 'Y2': 720, 'X1': 10, 'Y1': 720, 'X0': 10,  'Y0': 720}
    ),


    MemLevel(
        name = "FeatureMemory", # FMEM
        size = 1056 * 1024, 
        value_access_energy = 2.02, 
        bandwidth = 132*2,
        bypasses = ['w'],
        dataflow_constraints = [
            'C9', 'R9', 'S9', 'C8', 'S8', 'R8', 'C7', 'S7', 'R7', 'C6', 'S6', 'R6', 
            'C5', 'S5', 'R5', 'C4', 'S4', 'R4', 'C3', 'S3', 'R3', 'C2', 'S2', 'R2', 
            'C1', 'S1', 'R1', 'C0', 'R0', 'S0', 
            'Q', 'Z9', 'X8', 'Z8', 'X7', 'Z7', 'X6', 'Z6', 'X5', 'Z5', 
            'X4', 'Z4', 'X3', 'Z3', 'X2', 'Z2', 'X1', 'Z1', 'X0', 'Z0'
        ],
        factors_constraints = {
            'C9': 32,
            'C8': 32, 'Z8': 2,
            'C7': 32, 'Z7': 2,
            'C6': 32, 'Z6': 2,
            'C5': 32, 'Z5': 2,
            'C4': 32, 'Z4': 2,
            'C3': 32, 'Z3': 2,
            'C2': 32, 'Z2': 2,
            'C1': 32, 'Z1': 2,
            'Z0': 2, 'C0': 3, 'S8': 3, 'R8': 3, 'S7': 3, 'R7': 3, 'S6': 3, 'R6': 3, 'S5': 3, 'R5': 3,
            'S4': 3, 'R4': 3, 'S3': 3, 'R3': 3, 'S2': 3, 'R2': 3, 'S1': 3, 'R1': 3
        }
    ),

    MemLevel(
        name = "WeightMemory", # WMEM
        size = 524 * 1024, 
        value_access_energy = 2.02, 
        bandwidth = 2*16,                 #FORZATURA read_bandwidth = 16, write_bandwidth = 16
        bypasses = ['in', 'int', 'out'],
        dataflow_constraints = ['R0','S0'], # Just innermost weights
        # Constraints for all weights
        factors_constraints = {
            'R0': 3, 'S0': 3
        }
    ),


    # --- Layer 9  ---
    FanoutLevel(name = "SACols_9", mesh = 128, dims = ['Q'], factors_constraints = {'Q': 128}),
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

#    MemLevel(
#        name = "AccumulationOutRegister",
#        size = 28,                  # FORZATURA "Accumulation REGF (28x32b)"      128/8 
#        value_access_energy = 1.34, 
#        bandwidth = 2*2,
#        bypasses = ['in', 'w'],
#        dataflow_constraints = ['Q'],
#        factors_constraints = {'Q': 10}
#    ),

    ComputeLevel(
        name = "Compute",
        mesh = 1, 
        compute_energy = 0.21, 
        cycles = 1,
        factors_constraints = {}
    )
], coupling=conv_10layers_coupling, name="DepFiN 10-Layer Architecture")

arch_depfin_10layers_without_intermediate_but_I_think_it_is_wrong = Arch([
    MemLevel(
        name = "DRAM",
        size = 2**64-1, 
        #value_access_energy = 400.0,
        read_value_access_energy = 400,
        write_value_access_energy = 400, 
        bandwidth = 18*2,                           # 17GB/s / 930MHz = 18 B per s
        read_bandwidth = 14,
        write_bandwidth = 4,
        
        bypasses = ['int'],
        dataflow_constraints = ['Q', 'P', 'X8', 'Y8', 'X7', 'Y7', 'X6', 'Y6', 
                                 'X5', 'Y5', 'X4', 'Y4', 'X3', 'Y3', 
                                 'X2', 'Y2', 'X1', 'Y1', 'X0', 'Y0'], 
        factors_constraints = {'Q': 10, 'P': 720}
    ),

    MemLevel(
        name = "FeatureMemory", # FMEM
        size = 1056 * 1024, 
        value_access_energy = 2.02, 
        bandwidth = 132*2,
        bypasses = ['w'],
        dataflow_constraints = [
            'C10', 'R10', 'S10',
            'C9', 'R9', 'S9', 'C8', 'S8', 'R8', 'C7', 'S7', 'R7', 'C6', 'S6', 'R6', 
            'C5', 'S5', 'R5', 'C4', 'S4', 'R4', 'C3', 'S3', 'R3', 'C2', 'S2', 'R2', 
            'C1', 'S1', 'R1', 'C0', 'R0', 'S0', 
            'Q', 'Z9', 'X8', 'Z8', 'X7', 'Z7', 'X6', 'Z6', 'X5', 'Z5', 
            'X4', 'Z4', 'X3', 'Z3', 'X2', 'Z2', 'X1', 'Z1', 'X0', 'Z0'
        ],
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
            'Z0': 2, 'C0': 3, 'R9': 3, 'S9': 3, 'S8': 3, 'R8': 3, 'S7': 3, 'R7': 3, 'S6': 3, 'R6': 3, 'S5': 3, 'R5': 3,
            'S4': 3, 'R4': 3, 'S3': 3, 'R3': 3, 'S2': 3, 'R2': 3, 'S1': 3, 'R1': 3
        }
    ),

    MemLevel(
        name = "WeightMemory", # WMEM
        size = 524 * 1024, 
        value_access_energy = 2.02, 
        bandwidth = 2*16,                 #FORZATURA read_bandwidth = 16, write_bandwidth = 16
        bypasses = ['in', 'int', 'out'],
        dataflow_constraints = ['R0','S0'], # Just innermost weights
        # Constraints for all weights
        factors_constraints = {
            'R0': 3, 'S0': 3
        }
    ),

    # --- Layer 10 ---
    FanoutLevel(name = "SACols_10", mesh = 128, dims = ['Q'], factors_constraints = {'Q': 128}),
    FanoutLevel(name = "SARows_10", mesh = 16, dims = ['Z10'], factors_constraints = {'Z10': 16}),

    # --- Layer 9  ---
    FanoutLevel(name = "SACols_9", mesh = 128, dims = ['X9'], factors_constraints = {'X9': 128}),
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

    ComputeLevel(
        name = "Compute",
        mesh = 1, 
        compute_energy = 0.21, 
        cycles = 1,
        factors_constraints = {}
    )
], coupling=conv_10layers_coupling, name="DepFiN 10-Layer Architecture")


"""
arch_depfin_complete = Arch([
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
        factors_constraints = {'Z2': 1, 'R2': 1, 'S2': 1, 'Q': 1, 'C2': 32, 'C1': 32, 'Z1': 2, 'S1': 3, 'R1': 3, 'X1': 1, 'S0': 1, 'R0': 1, 'Z0': 2, 'X0': 1, 'C0': 3}
    ),
    
    MemLevel(
        name = "WeightMemory", # WMEM
        size = 524 * 1024, # 524 kB in bits
        value_access_energy = 2.02, # SRAM eyeriss
        bandwidth = 16, # "16 weights are provided in parallel" 
        bypasses = ['in', 'int', 'out'],
        dataflow_constraints = ['R0','S0'],
        factors_constraints = {'R0': 3, 'S0': 3}
    ),

    FanoutLevel(
        name = "SACols_2",
        mesh = 128, 
        dims = ['Q'], # Corrisponde a Ox (Output Width)
        factors_constraints = {'Q': 128}
    ),

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
        factors_constraints = {'X1': 128}
    ),

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
        factors_constraints = {'X0': 128}
    ),

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

arch_depfin_without_spatial = Arch([
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
        dataflow_constraints = ['C2', 'R2', 'S2', 'Q', 'Z2', 'C1', 'S1', 'R1', 'X1', 'Z1', 'C0', 'R0', 'S0', 'X0', 'Z0'],
        factors_constraints = {'Z2': 16, 'R2': 1, 'S2': 1, 'Q': 128, 'C2': 32, 'C1': 32, 'Z1': 32, 'S1': 3, 'R1': 3, 'X1': 128, 'S0': 3, 'R0': 3, 'Z0': 32, 'X0': 128, 'C0': 3}
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

arch_depfin_without_spatial_as_if = Arch([
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
        factors_constraints = {'Z2': 16, 'R2': 1, 'S2': 1, 'Q': 128, 'C2': 32, 'C1': 32, 'Z1': 32, 'S1': 3, 'R1': 3, 'X1': 128, 'S0': 3, 'R0': 3, 'Z0': 32, 'X0': 128, 'C0': 3}
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
"""
"""
arch_depfin_10layers_without_intermediate_but_I_think_it_is_wrong = Arch([
        MemLevel(
        name = "DRAM",
        size = 2**64-1, 
        value_access_energy = 50.0, 
        bandwidth = 18*2,                           # 17GB/s / 930MHz = 18 B per s
        bypasses = [],
        # Constraints for the outermost layer (Layer 9)
        dataflow_constraints = ['Q', 'P', 'X8', 'Y8', 'X7', 'Y7', 'X6', 'Y6', 
                                 'X5', 'Y5', 'X4', 'Y4', 'X3', 'Y3', 
                                 'X2', 'Y2', 'X1', 'Y1', 'X0', 'Y0'], 
        factors_constraints = {'Q': 1, 'P': 720}
    ),


    MemLevel(
        name = "FeatureMemory", # FMEM
        size = 1056 * 1024, 
        value_access_energy = 2.02, 
        bandwidth = 132*2,
        bypasses = ['w'],
        dataflow_constraints = [
            'C9', 'R9', 'S9', 'C8', 'S8', 'R8', 'C7', 'S7', 'R7', 'C6', 'S6', 'R6', 
            'C5', 'S5', 'R5', 'C4', 'S4', 'R4', 'C3', 'S3', 'R3', 'C2', 'S2', 'R2', 
            'C1', 'S1', 'R1', 'C0', 'R0', 'S0', 
            'Q', 'Z9', 'X8', 'Z8', 'X7', 'Z7', 'X6', 'Z6', 'X5', 'Z5', 
            'X4', 'Z4', 'X3', 'Z3', 'X2', 'Z2', 'X1', 'Z1', 'X0', 'Z0'
        ],
        factors_constraints = {
            'C9': 32,
            'C8': 32, 'Z8': 2,
            'C7': 32, 'Z7': 2,
            'C6': 32, 'Z6': 2,
            'C5': 32, 'Z5': 2,
            'C4': 32, 'Z4': 2,
            'C3': 32, 'Z3': 2,
            'C2': 32, 'Z2': 2,
            'C1': 32, 'Z1': 2,
            'Z0': 2, 'C0': 3,'S8': 3, 'R8': 3, 'S7': 3, 'R7': 3, 'S6': 3, 'R6': 3, 'S5': 3, 'R5': 3,
            'S4': 3, 'R4': 3, 'S3': 3, 'R3': 3, 'S2': 3, 'R2': 3, 'S1': 3, 'R1': 3
        }
    ),

    MemLevel(
        name = "WeightMemory", # WMEM
        size = 524 * 1024, 
        value_access_energy = 2.02, 
        bandwidth = 2*16,                 #FORZATURA read_bandwidth = 16, write_bandwidth = 16
        bypasses = ['in', 'int', 'out'],
        dataflow_constraints = ['R0','S0'], # Just innermost weights
        # Constraints for all weights
        factors_constraints = {
            'R0': 3, 'S0': 3
        }
    ),


    # --- Layer 9  ---
    FanoutLevel(name = "SACols_9", mesh = 128, dims = ['Q'], factors_constraints = {'Q': 128}),
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
        name = "AccumulationOutRegister",
        size = 28,                  # FORZATURA "Accumulation REGF (28x32b)"      
        value_access_energy = 1.34, 
        bandwidth = 2*2,
        bypasses = ['in', 'w'],
        dataflow_constraints = ['Q'],
        factors_constraints = {'Q': 10}
    ),

    ComputeLevel(
        name = "Compute",
        mesh = 1, 
        compute_energy = 0.21, 
        cycles = 1,
        factors_constraints = {}
    )
], coupling=conv_10layers_coupling, name="DepFiN 10-Layer Architecture")

"""


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