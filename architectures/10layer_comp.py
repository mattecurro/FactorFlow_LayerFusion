from computations import *
from factors import Shape

coupling = conv_10layers_coupling

# Define your computation with the parameters you specified
# K = 8, P = 256, Q = 256, R1 = 3, S1 = 3, C1 = 2, R0 = 3, S0 = 3, M = 3
#comp = Shape(
#    P = 256,
#    Q = 256,
#    K = 8,
#    R0 = 3,
#    S0 = 3,
#    C1 = 2,
#    R1 = 3,
#    S1 = 3,
#    R2 = 3,  # You'll need to specify R2 and S2 for the third layer
#    S2 = 3,
#    C2 = 4,  # You'll need to specify C2 for the connection between layers
#    M = 3
#)

comp_FSRCNN = Shape(
    # --- Layer 7 (Output) ---
    Y7 = 540,
    X7 = 960,
    Z7 = 16,
    C7 = 56,
    R7 = 1,
    S7 = 1,

    # --- Layer 6 ---
    Y6 = 540,
    X6 = 960,
    Z6 = 56,
    C6 = 12,
    R6 = 1,
    S6 = 1,

    # --- Layer 5 ---
    Y5 = 540,
    X5 = 960,
    Z5 = 12,
    C5 = 12,
    R5 = 3,
    S5 = 3,

    # --- Layer 4 ---
    Y4 = 540,
    X4 = 960,
    Z4 = 12,
    C4 = 12,
    R4 = 3,
    S4 = 3,

    # --- Layer 3 ---
    Y3 = 540,
    X3 = 960,
    Z3 = 12,
    C3 = 12,
    R3 = 3,
    S3 = 3,

    # --- Layer 2 ---
    Y2 = 540,
    X2 = 960,
    Z2 = 12,
    C2 = 12,
    R2 = 3,
    S2 = 3,

    # --- Layer 1 ---
    Y1 = 540,
    X1 = 960,
    Z1 = 12,
    C1 = 56,
    R1 = 1,
    S1 = 1,

    # --- Layer 0 (Input) ---
    Y0 = 540,
    X0 = 960, 
    Z0 = 56,
    R0 = 5,
    S0 = 5,
    C0 = 3
)

comp = comp_home_without_int = Shape(
    # --- Layer 10 (Output) ---
    P = 720,            
    Q = 1280,
    Z10 = 16,
    C10 = 32,
    R10 = 1,
    S10 = 1,
    # --- Layer 9 ---
    Y9 = 1,
    X9 = 128,
    Z9 = 32,
    C9 = 32,
    R9 = 3,
    S9 = 3,
    # --- Layer 8 ---
    Y8 = 1,
    X8 = 128,
    Z8 = 32,
    C8 = 32,
    R8 = 3,
    S8 = 3,
    # --- Layer 7 ---
    Y7 = 1,
    X7 = 128,
    Z7 = 32,
    C7 = 32,
    R7 = 3,
    S7 = 3,
    # --- Layer 6 ---
    Y6 = 1,
    X6 = 128,
    Z6 = 32,
    C6 = 32,
    R6 = 3,
    S6 = 3,
    # --- Layer 5 ---
    Y5 = 1,
    X5 = 128,
    Z5 = 32,
    C5 = 32,
    R5 = 3,
    S5 = 3,     
    # --- Layer 4 ---
    Y4 = 1,
    X4 = 128,
    Z4 = 32,
    C4 = 32,
    R4 = 3,
    S4 = 3,
    # --- Layer 3 ---
    Y3 = 1,
    X3 = 128,
    Z3 = 32,
    C3 = 32,
    R3 = 3,
    S3 = 3,
    # --- Layer 2 ---
    Y2 = 1,
    X2 = 128,
    Z2 = 32,
    C2 = 32,
    R2 = 3,
    S2 = 3,
    # --- Layer 1 ---
    Y1 = 1,
    X1 = 128,
    Z1 = 32,
    C1 = 32,
    R1 = 3,
    S1 = 3,
    # --- Layer 0 (Input) ---
    Y0 = 720,
    X0 = 1280, 
    Z0 = 32,
    R0 = 3,
    S0 = 3,
    C0 = 3
)

comp = comp_home = Shape(
    # --- Layer 10 (Output) ---
    P = 720,            
    Q = 1280,
    Z10 = 16,
    C10 = 32,
    R10 = 1,
    S10 = 1,

    # --- Layer 9 ---
    Y9 = 720,
    X9 = 1280,
    Z9 = 32,
    C9 = 32,
    R9 = 3,
    S9 = 3,

    # --- Layer 8 ---
    Y8 = 720,
    X8 = 1280,
    Z8 = 32,
    C8 = 32,
    R8 = 3,
    S8 = 3,

    # --- Layer 7 ---
    Y7 = 720,
    X7 = 1280,
    Z7 = 32,
    C7 = 32,
    R7 = 3,
    S7 = 3,

    # --- Layer 6 ---
    Y6 = 720,
    X6 = 1280,
    Z6 = 32,
    C6 = 32,
    R6 = 3,
    S6 = 3,

    # --- Layer 5 ---
    Y5 = 720,
    X5 = 1280,
    Z5 = 32,
    C5 = 32,
    R5 = 3,
    S5 = 3,

    # --- Layer 4 ---
    Y4 = 720,
    X4 = 1280,
    Z4 = 32,
    C4 = 32,
    R4 = 3,
    S4 = 3,

    # --- Layer 3 ---
    Y3 = 720,
    X3 = 1280,
    Z3 = 32,
    C3 = 32,
    R3 = 3,
    S3 = 3,

    # --- Layer 2 ---
    Y2 = 720,
    X2 = 1280,
    Z2 = 32,
    C2 = 32,
    R2 = 3,
    S2 = 3,

    # --- Layer 1 ---
    Y1 = 720,
    X1 = 1280,
    Z1 = 32,
    C1 = 32,
    R1 = 3,
    S1 = 3,

    # --- Layer 0 (Input) ---
    Y0 = 720,
    X0 = 1280, 
    Z0 = 32,
    R0 = 3,
    S0 = 3,
    C0 = 3
)