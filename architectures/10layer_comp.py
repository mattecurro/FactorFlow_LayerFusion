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

#     0 <= P1 < 128 and 0 <= M1 < 64 and 0 <= C1 < 64
#     0 <= P2 < 128 and 0 <= M2 < 64 and 0 <= C2 < 64
# C0: Filter0 depth/Input depth
# Y: Intermediate Out height/Input height
# Z: Filter0 num/Intermediate Out depth  
# C1: Filter1 depth/Intermediate In depth
# P: Out height
# C2: Filter1 num/Out depth
# MAC1: Intermediate_Out[z][y] += W0[z][c0] * In[c0][y]
# MAC2: Out[c2][p] += W1[c2][c1] * Intermediate_In[c1][p]
comp_2 = Shape(
    # --- Layer 9 (Output) ---
    P = 720,            
    Q = 1280,
    Z9 = 16,
    C9 = 32,
    R9 = 1,
    S9 = 1,

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
    Y0 = 1,
    X0 = 128,
    Z0 = 32,
    R0 = 3,
    S0 = 3,
    C0 = 3
)

comp = Shape(
    # --- Layer 9 (Output) ---
    P = 720,            
    Q = 1280,
    Z9 = 16,
    C9 = 32,
    R9 = 1,
    S9 = 1,

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