from computations import *
from factors import Shape

# Create the coupling for 3-layer convolution
coupling = easy_conv_3layers_coupling


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
comp = Shape(
    P = 704,            
    Q = 1280,           
    Z2 = 16,
    C2 = 32,
    R2 = 1,
    S2 = 1,
    X1 = 128,
    Y1 = 1,
    Z1 = 32,
    R1 = 3,
    S1 = 3,
    X0 = 128,
    Y0 = 1,
    C1 = 32,
    Z0 = 32,
    R0 = 3,
    S0 = 3,
    C0 = 3
)
"""
comp = Shape(
    Q = 384,
    Z2 = 32,
    C2 = 32,
    S2 = 3,
    X1 = 130,
    Z1 = 32,
    S1 = 3,
    X0 = 132,
    C1 = 32,
    Z0 = 32,
    S0 = 3,
    C0 = 32
)
"""