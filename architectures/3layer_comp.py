from computations import create_nlayer_conv_coupling
from factors import Shape

# Create the coupling for 3-layer convolution
coupling = create_nlayer_conv_coupling(num_layers=3)

# Define your computation with the parameters you specified
# K = 8, P = 256, Q = 256, R1 = 3, S1 = 3, C1 = 2, R0 = 3, S0 = 3, M = 3
comp = Shape(
    P = 256,
    Q = 256,
    K = 8,
    R0 = 3,
    S0 = 3,
    C1 = 2,
    R1 = 3,
    S1 = 3,
    R2 = 3,  # You'll need to specify R2 and S2 for the third layer
    S2 = 3,
    C2 = 4,  # You'll need to specify C2 for the connection between layers
    M = 3
)