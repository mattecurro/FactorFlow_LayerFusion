from computations import gemm_2layers_coupling, create_nlayer_conv_coupling
from factors import Shape

# Create the coupling for 3-layer convolution
#coupling = create_nlayer_conv_coupling(num_layers=2)

#coupling = gemm_2layers_coupling 

# Define your computation with the parameters you specified
# K = 8, P = 256, Q = 256, R1 = 3, S1 = 3, C1 = 2, R0 = 3, S0 = 3, M = 3
#comp = Shape(
#    P = 6,
#    Q = 6,
#    Y = 8,
#    X = 8,
#    Z = 4,
#    C0 = 3,
#    C1 = 4,
#    S0 = 3,
#    R0 = 3,
#    R1 = 3,
#    S1 = 3,
#    C2 = 2
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
comp = Shape(
    P = 128,
    C1 = 64,
    C2 = 64,
    Y = 128,
    Z = 64,
    C0 = 64
)
