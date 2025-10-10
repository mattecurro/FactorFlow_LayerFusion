from computations import create_nlayer_conv_coupling
from factors import Shape

# Create the coupling for 3-layer convolution
coupling = create_nlayer_conv_coupling(num_layers=2)

# Define your computation with the parameters you specified
# K = 8, P = 256, Q = 256, R1 = 3, S1 = 3, C1 = 2, R0 = 3, S0 = 3, M = 3
comp = Shape(
    P = 6,
    Q = 6,
    Y = 8,
    X = 8,
    Z = 4,
    C0 = 3,
    C1 = 4,
    S0 = 3,
    R0 = 3,
    R1 = 3,
    S1 = 3,
    C2 = 2
)