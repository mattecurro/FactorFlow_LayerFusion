"""
Thesis Case Study Workloads

This file defines workloads for layer fusion experiments, categorized as:
- Activation-Dominant: FSRCNN, MC-CNN (benefit more from layer fusion)
- Weight-Dominant: VGG16, ResNet18 (less benefit from layer fusion)

For each network, we provide:
1. Single-layer versions (for baseline comparisons)
2. Multi-layer fused versions (for fusion experiments)

Multi-layer Shape Convention:
- Layer i has dimensions: Yi, Xi (input spatial), Zi (output channels), Ci (input channels), Ri, Si (kernel)
- Output layer uses P, Q instead of Y, X (no layer index)
- Strides are stored as: Pstride{i}, Qstride{i} for layer i (default 1 if not specified)
- For layer fusion tile sizing with Fully Cached: input_tile = output_tile * stride (when kernel >= stride)
"""

from factors import Shape, Coupling
from computations import (
    conv_coupling, 
    conv_coupling_with_stride,
    create_nlayer_conv_coupling,
    conv_2layers_coupling,
    conv_3layers_coupling,
    conv_4layers_coupling,
    conv_5layers_coupling,
    conv_8layers_coupling,
    conv_13layers_coupling,
    conv_17layers_coupling
)
from typing import List, Tuple


# =============================================================================
# HELPER FUNCTIONS FOR STRIDE-AWARE TILE SIZING
# =============================================================================

def get_layer_stride(shape: Shape, layer_idx: int) -> Tuple[int, int]:
    """
    Get the (Pstride, Qstride) for a given layer index.
    Returns (1, 1) if strides are not specified (default).
    
    Args:
        shape: Multi-layer Shape containing stride info
        layer_idx: Layer index (0 = first/input layer)
    
    Returns:
        (pstride, qstride) tuple
    """
    pstride = shape.get(f'Pstride{layer_idx}', 1)
    qstride = shape.get(f'Qstride{layer_idx}', 1)
    return (pstride, qstride)


def get_layer_kernel(shape: Shape, layer_idx: int) -> Tuple[int, int]:
    """
    Get the (R, S) kernel size for a given layer index.
    
    Args:
        shape: Multi-layer Shape
        layer_idx: Layer index
    
    Returns:
        (R, S) tuple
    """
    r = shape.get(f'R{layer_idx}', 1)
    s = shape.get(f'S{layer_idx}', 1)
    return (r, s)


def calculate_per_layer_tile_sizes(
    shape: Shape,
    num_layers: int,
    output_tile_h: int,
    output_tile_w: int
) -> List[Tuple[int, int]]:
    """
    Calculate the required tile sizes at each layer for depth-first processing in Fully Cached mode.
    Propagates backwards from output layer to input layer.
    
    Formula: new_input = output * stride (when stride <= kernel)
    
    Args:
        shape: Multi-layer Shape with stride info
        num_layers: Number of layers
        output_tile_h: Desired output tile height
        output_tile_w: Desired output tile width
    
    Returns:
        List of (height, width) tuples for each layer (index 0 = input layer)
    """
    # Start with output tile size
    sizes = [(output_tile_h, output_tile_w)]
    curr_h, curr_w = output_tile_h, output_tile_w
    
    # Propagate backwards from last layer to first layer
    for layer_idx in range(num_layers - 1, -1, -1):
        pstride, qstride = get_layer_stride(shape, layer_idx)
        r, s = get_layer_kernel(shape, layer_idx)
        
        # Calculate new input size for this layer
        # new_input = output * stride (when stride <= kernel, halo is reused)
        if pstride <= r:
            new_h = curr_h * pstride
        else:
            # stride > kernel: no halo overlap, need full input
            new_h = (curr_h - 1) * pstride + r
            
        if qstride <= s:
            new_w = curr_w * qstride
        else:
            new_w = (curr_w - 1) * qstride + s
        
        sizes.insert(0, (new_h, new_w))
        curr_h, curr_w = new_h, new_w
    
    return sizes


def get_cumulative_stride(shape: Shape, num_layers: int) -> Tuple[int, int]:
    """
    Calculate the cumulative stride across all layers.
    This is the product of all individual strides.
    
    For depth-first: input_tile = output_tile * cumulative_stride
    """
    cum_pstride = 1
    cum_qstride = 1
    
    for layer_idx in range(num_layers):
        pstride, qstride = get_layer_stride(shape, layer_idx)
        cum_pstride *= pstride
        cum_qstride *= qstride
    
    return (cum_pstride, cum_qstride)

# =============================================================================
# ACTIVATION-DOMINANT WORKLOADS
# These networks have small filters and large feature maps, making them
# ideal candidates for layer fusion (intermediate activations dominate memory)
# =============================================================================

# -----------------------------------------------------------------------------
# FSRCNN (Fast Super-Resolution CNN)
# Reference: "Accelerating the Super-Resolution Convolutional Neural Network"
# Architecture: 5x5 -> 1x1 -> 3x3 -> 3x3 -> 3x3 -> 3x3 -> 1x1 -> 9x9
# Characteristics: Small channels, large spatial dimensions
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# FSRCNN TDC Version (from TDC[21] paper)
# Input: 960×540×3 (RGB)
# Architecture: 5x5(56) -> 1x1(12) -> [3x3(12)]×4 -> 1x1(56) -> 3x3(16)
# Total: 8 layers
# -----------------------------------------------------------------------------

# 8-layer shape for FSRCNN-TDC (for 8-layer fusion with conv_8layers_coupling)
comp_FSRCNN_TDC = Shape(
    # --- Layer 7 (Output): 3×3 conv, 56→16 ---
    P = 540,            # Output height
    Q = 960,            # Output width
    Z7 = 16,            # Output channels
    C7 = 56,            # Input channels (from L6)
    R7 = 3,             # 3×3 kernel (TDC version uses 3×3, not 1×1)
    S7 = 3,

    # --- Layer 6 (Expanding): 1×1 PW, 12→56 ---
    Y6 = 540,
    X6 = 960,
    Z6 = 56,
    C6 = 12,
    R6 = 1,
    S6 = 1,

    # --- Layer 5 (Mapping 4): 3×3, 12→12 ---
    Y5 = 540,
    X5 = 960,
    Z5 = 12,
    C5 = 12,
    R5 = 3,
    S5 = 3,

    # --- Layer 4 (Mapping 3): 3×3, 12→12 ---
    Y4 = 540,
    X4 = 960,
    Z4 = 12,
    C4 = 12,
    R4 = 3,
    S4 = 3,

    # --- Layer 3 (Mapping 2): 3×3, 12→12 ---
    Y3 = 540,
    X3 = 960,
    Z3 = 12,
    C3 = 12,
    R3 = 3,
    S3 = 3,

    # --- Layer 2 (Mapping 1): 3×3, 12→12 ---
    Y2 = 540,
    X2 = 960,
    Z2 = 12,
    C2 = 12,
    R2 = 3,
    S2 = 3,

    # --- Layer 1 (Shrinking): 1×1 PW, 56→12 ---
    Y1 = 540,
    X1 = 960,
    Z1 = 12,
    C1 = 56,
    R1 = 1,
    S1 = 1,

    # --- Layer 0 (Feature Extraction): 5×5, 3→56 ---
    Y0 = 540,
    X0 = 960, 
    Z0 = 56,
    R0 = 5,
    S0 = 5,
    C0 = 3              
)

# Single-layer shapes for FSRCNN-TDC (for non-fused baseline)
fsrcnn_tdc_single_layers = {
    'L0_feature_extraction': Shape(
        C=3, M=56, P=540, Q=960, R=5, S=5,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L1_shrinking': Shape(
        C=56, M=12, P=540, Q=960, R=1, S=1,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L2_mapping1': Shape(
        C=12, M=12, P=540, Q=960, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L3_mapping2': Shape(
        C=12, M=12, P=540, Q=960, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L4_mapping3': Shape(
        C=12, M=12, P=540, Q=960, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L5_mapping4': Shape(
        C=12, M=12, P=540, Q=960, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L6_expanding': Shape(
        C=12, M=56, P=540, Q=960, R=1, S=1,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L7_output': Shape(
        C=56, M=16, P=540, Q=960, R=3, S=3,  # 3×3 output (TDC version)
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
}

# 2-layer fused FSRCNN-TDC segments (for 2-layer fusion experiments)
# Uses indexed dimensions: Y0, X0, Z0 for intermediate, Z1 for output channels
# This matches the N-layer pattern (Z{layer} for each layer's output channels)
fsrcnn_tdc_2layer_fused = {
    # L0 + L1: Feature extraction (5×5) + Shrinking (1×1)
    'L0_L1_extraction_shrinking': Shape(
        P=540, Q=960,           # Output spatial
        C0=3, R0=5, S0=5,       # L0: 5×5, 3→56
        Y0=540, X0=960, Z0=56,  # Intermediate (layer 0 output)
        C1=56, R1=1, S1=1, Z1=12  # L1: 1×1, 56→12 (Z1 = output channels)
    ),
    # L2 + L3: Mapping layers 1-2 (both 3×3, 12→12)
    'L2_L3': Shape(
        P=540, Q=960,
        C0=12, R0=3, S0=3,
        Y0=540, X0=960, Z0=12,
        C1=12, R1=3, S1=3, Z1=12
    ),
    # L4 + L5: Mapping layers 3-4 (both 3×3, 12→12)
    'L4_L5': Shape(
        P=540, Q=960,
        C0=12, R0=3, S0=3,
        Y0=540, X0=960, Z0=12,
        C1=12, R1=3, S1=3, Z1=12
    ),
    # L6 + L7: Expanding (1×1) + Output (3×3)
    'L6_L7_expanding_output': Shape(
        P=540, Q=960,
        C0=12, R0=1, S0=1,       # L6: 1×1, 12→56
        Y0=540, X0=960, Z0=56,
        C1=56, R1=3, S1=3, Z1=16  # L7: 3×3, 56→16
    ),
}

# 3-layer fused FSRCNN-TDC segments (for 3-layer fusion experiments)
# Uses conv_3layers_coupling from computations.py
fsrcnn_tdc_3layer_fused = {
    # L0 + L1 + L2: Feature extraction + Shrinking + Mapping1
    'L0_L1_L2': Shape(
        P=540, Q=960,
        C0=3, R0=5, S0=5,        # L0: 5×5, 3→56
        Y0=540, X0=960, Z0=56,
        C1=56, R1=1, S1=1,       # L1: 1×1, 56→12
        Y1=540, X1=960, Z1=12,
        C2=12, R2=3, S2=3, Z2=12  # L2: 3×3, 12→12
    ),
    # L3 + L4 + L5: Mapping layers 2-4 (all 3×3, 12→12)
    'L3_L4_L5': Shape(
        P=540, Q=960,
        C0=12, R0=3, S0=3,
        Y0=540, X0=960, Z0=12,
        C1=12, R1=3, S1=3,
        Y1=540, X1=960, Z1=12,
        C2=12, R2=3, S2=3, Z2=12
    ),
    # L6 + L7: Expanding (1×1) + Output (3×3)
    'L6_L7_expanding_output': Shape(
        P=540, Q=960,
        C0=12, R0=1, S0=1,       # L6: 1×1, 12→56
        Y0=540, X0=960, Z0=56,
        C1=56, R1=3, S1=3, Z1=16  # L7: 3×3, 56→16
    ),
}

# Per-variant couplings for FSRCNN-TDC 3-layer level (L6_L7 is only 2 layers)
fsrcnn_tdc_3layer_couplings = {
    'L0_L1_L2': conv_3layers_coupling,
    'L3_L4_L5': conv_3layers_coupling,
    'L6_L7_expanding_output': conv_2layers_coupling,
}

# 4-layer fused FSRCNN-TDC (all mapping layers)
# NOTE: Disabled because create_nlayer_conv_coupling(num_layers=4) has bugs
# and conv_4layers_coupling is not available. Define conv_4layers_coupling
# in computations.py if needed.
# fsrcnn_tdc_4layer_fused = {
#     # L2 + L3 + L4 + L5: All 4 mapping layers (all 3×3, 12→12)
#     'L2_L3_L4_L5_all_mapping': Shape(
#         P=540, Q=960,
#         C0=12, R0=3, S0=3,
#         Y0=540, X0=960, Z0=12,
#         C1=12, R1=3, S1=3,
#         Y1=540, X1=960, Z1=12,
#         C2=12, R2=3, S2=3,
#         Y2=540, X2=960, Z2=12,
#         C3=12, R3=3, S3=3, Z3=12
#     ),
# }

# 8-layer fully fused FSRCNN-TDC (entire network fused)
# Uses conv_8layers_coupling from computations.py
# This is the same as comp_FSRCNN_TDC defined above, provided as an alias for consistency
fsrcnn_tdc_8layer_fused = comp_FSRCNN_TDC

# Coupling for 8-layer FSRCNN-TDC fusion
# conv_8layers_coupling is imported at the top of this file
fsrcnn_tdc_8layer_coupling = conv_8layers_coupling

# Single-layer shapes for FSRCNN (for non-fused baseline)
fsrcnn_single_layers = {
    'L0_feature_extraction': Shape(
        C=1, M=56, P=540, Q=960, R=5, S=5,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L1_shrinking': Shape(
        C=56, M=12, P=540, Q=960, R=1, S=1,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L2_mapping1': Shape(
        C=12, M=12, P=540, Q=960, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L3_mapping2': Shape(
        C=12, M=12, P=540, Q=960, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L4_mapping3': Shape(
        C=12, M=12, P=540, Q=960, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L5_mapping4': Shape(
        C=12, M=12, P=540, Q=960, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L6_expanding': Shape(
        C=12, M=56, P=540, Q=960, R=1, S=1,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L7_deconv': Shape(
        C=56, M=1, P=1080, Q=1920, R=9, S=9,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
}

# 2-layer fused FSRCNN segments (for 2-layer fusion experiments)
fsrcnn_2layer_coupling = conv_2layers_coupling

fsrcnn_2layer_fused = {
    'L0_L1': Shape(
        # Layer 0: 5x5 conv, C0=1 -> Z=56
        # Layer 1: 1x1 conv, C1=56 -> C2=12
        C0=1, Z=56, R0=5, S0=5,
        Y=540, X=960,  # Intermediate spatial
        C1=56, C2=12, R1=1, S1=1,
        P=540, Q=960   # Output spatial
    ),
    'L2_L3': Shape(
        C0=12, Z=12, R0=3, S0=3,
        Y=540, X=960,
        C1=12, C2=12, R1=3, S1=3,
        P=540, Q=960
    ),
    'L4_L5': Shape(
        C0=12, Z=12, R0=3, S0=3,
        Y=540, X=960,
        C1=12, C2=12, R1=3, S1=3,
        P=540, Q=960
    ),
}

# 3-layer fused FSRCNN (mapping layers)
# Note: Use conv_3layers_coupling instead of create_nlayer_conv_coupling (which has bugs)
fsrcnn_3layer_coupling = conv_3layers_coupling

fsrcnn_3layer_fused = {
    'L2_L3_L4': Shape(
        # Mapping layers 1-3
        P=540, Q=960,
        C0=12, R0=3, S0=3,
        Y0=540, X0=960, Z0=12,
        C1=12, R1=3, S1=3,
        Y1=540, X1=960, Z1=12,
        C2=12, R2=3, S2=3, Z2=12
    ),
}

# -----------------------------------------------------------------------------
# MC-CNN (Matching Cost CNN for Stereo Vision)
# Reference: "Computing the Stereo Matching Cost with a CNN"
# Architecture: 3x3 -> 3x3 -> 3x3 -> 3x3 (4 layers, 32 channels each)
# Characteristics: Very large spatial dimensions (1242x376), uniform channels
# -----------------------------------------------------------------------------

# Single-layer shapes for MC-CNN
mccnn_single_layers = {
    'L0': Shape(
        C=1, M=32, P=376, Q=1242, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L1': Shape(
        C=32, M=32, P=376, Q=1242, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L2': Shape(
        C=32, M=32, P=376, Q=1242, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
    'L3': Shape(
        C=32, M=32, P=376, Q=1242, R=3, S=3,
        Pstride=1, Qstride=1, Rdilation=1, Sdilation=1
    ),
}

# 2-layer fused MC-CNN
mccnn_2layer_fused = {
    'L0_L1': Shape(
        P=376, Q=1242,
        C0=1, R0=3, S0=3,
        Y0=376, X0=1242, Z0=32,
        C1=32, R1=3, S1=3, Z1=32
    ),
    'L2_L3': Shape(
        P=376, Q=1242,
        C0=32, R0=3, S0=3,
        Y0=376, X0=1242, Z0=32,
        C1=32, R1=3, S1=3, Z1=32
    ),
}

# 4-layer fully fused MC-CNN (entire network fused)
# Uses conv_4layers_coupling from computations.py
mccnn_4layer_fused = Shape(
    # Layer 3 (output layer)
    P=376, Q=1242, Z3=32, C3=32, R3=3, S3=3,
    # Layer 2
    Y2=376, X2=1242, Z2=32, C2=32, R2=3, S2=3,
    # Layer 1
    Y1=376, X1=1242, Z1=32, C1=32, R1=3, S1=3,
    # Layer 0 (input layer)
    Y0=376, X0=1242, Z0=32, R0=3, S0=3, C0=1
)

# Coupling for 4-layer MC-CNN fusion
mccnn_4layer_coupling = conv_4layers_coupling


# =============================================================================
# WEIGHT-DOMINANT WORKLOADS
# These networks have large filters and many channels, making weight reuse
# more important than activation reuse. Layer fusion has less benefit.
# =============================================================================

# -----------------------------------------------------------------------------
# VGG16 (for fusion experiments)
# Reference: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Characteristics: 13 conv layers + 3 FC, 3x3 filters, blocks separated by maxpool
# Note: Fusion makes sense WITHIN blocks (layers share spatial dimensions)
# -----------------------------------------------------------------------------

# Single-layer VGG16 shapes (from computations.py)
from computations import comp_vgg_16 as vgg16_single_layers


# Block-based fusion for VGG16 (layers within each block share spatial dimensions)
# Block 1: L0+L1 (2 layers, 224x224, 64 channels) - uses conv_2layers_coupling
vgg16_block1_fused = Shape(
    P=224, Q=224, Z1=64,
    C1=64, R1=3, S1=3,
    Y0=224, X0=224, Z0=64, R0=3, S0=3, C0=3
)
    
# Block 2: L2+L3 (2 layers, 112x112, 128 channels) - uses conv_2layers_coupling
vgg16_block2_fused = Shape(
    P=112, Q=112, Z1=128, C1=128, R1=3, S1=3, 
    Y0=112, X0=112, Z0=128, R0=3, S0=3, C0=64, Pstride0=2, Qstride0=2,
)

# Block 3: L4+L5 (2 layers, 56x56, 256 channels) - uses conv_2layers_coupling
vgg16_block3_fused = Shape(
    P=56, Q=56, Z1=256, C1=256, R1=3, S1=3,
    Y0=56, X0=56, Z0=256, R0=3, S0=3, C0=128, Pstride0=2, Qstride0=2
)

# Block 4: L7+L8 (2 layers, 28x28, 512 channels) - uses conv_2layers_coupling
vgg16_block4_fused = Shape(
    P=28, Q=28, Z1=512, C1=512, R1=3, S1=3,
    Y0=28, X0=28, Z0=512, R0=3, S0=3, C0=256, Pstride0=2, Qstride0=2
)

    

# Block 5: L10+L11 (2 layers, 14x14, 512 channels) - uses conv_2layers_coupling
vgg16_block5_fused = Shape(
    P=14, Q=14, Z1=512, C1=512, R1=3, S1=3,
    Y0=14, X0=14, Z0=512, R0=3, S0=3, C0=512, Pstride0=2, Qstride0=2
)

# Dictionary of all VGG16 block fusions
vgg16_block_fused = {
    'block1': vgg16_block1_fused,  # L0+L1, 2 layers
    'block2': vgg16_block2_fused,  # L2+L3, 2 layers
    'block3': vgg16_block3_fused,  # L4+L5, 2 layers
    'L6': Shape(C=256, M=256, P=56, Q=56, R=3, S=3),
    'block4': vgg16_block4_fused,  # L7+L8, 2 layers
    'L9': Shape(C=512, M=512, P=28, Q=28, R=3, S=3),
    'block5': vgg16_block5_fused,  # L10+L11, 2 layers
    'L12': Shape(C=512, M=512, P=14, Q=14, R=3, S=3),
}

# Couplings for each block
vgg16_block_couplings = {
    'block1': conv_2layers_coupling,
    'block2': conv_2layers_coupling,
    'block3': conv_2layers_coupling,
    'L6': conv_coupling,
    'block4': conv_2layers_coupling,
    'L9': conv_coupling,
    'block5': conv_2layers_coupling,
    'L12': conv_coupling  
}

# =============================================================================
# VGG16 FULL FUSION (all 13 conv layers)
# This is impractical but useful for theoretical case studies
# Uses conv_13layers_coupling
#
# STRIDE INFO: VGG16 uses MaxPool (stride=2) between blocks, not strided convolutions.
# For tile size calculation, we model the pooling as effective stride=2 on the first
# layer of each new block:
#   - Layer 2 (conv2_1): after pool1, effective Pstride2=2, Qstride2=2
#   - Layer 4 (conv3_1): after pool2, effective Pstride4=2, Qstride4=2
#   - Layer 7 (conv4_1): after pool3, effective Pstride7=2, Qstride7=2
#   - Layer 10 (conv5_1): after pool4, effective Pstride10=2, Qstride10=2
# Cumulative stride = 2*2*2*2 = 16
# =============================================================================
vgg16_full_fused = Shape(
    # Output (after L12 = conv5_3)
    P=14, Q=14, Z12=512, C12=512, R12=3, S12=3,
    # Layer 11 (conv5_2): 512->512, 14x14
    Y11=14, X11=14, Z11=512, C11=512, R11=3, S11=3,
    # Layer 10 (conv5_1): 512->512, 14x14 (after pool4 from 28x28)
    Y10=14, X10=14, Z10=512, C10=512, R10=3, S10=3, Pstride10=2, Qstride10=2,
    # Layer 9 (conv4_3): 512->512, 28x28
    Y9=28, X9=28, Z9=512, C9=512, R9=3, S9=3,
    # Layer 8 (conv4_2): 512->512, 28x28
    Y8=28, X8=28, Z8=512, C8=512, R8=3, S8=3,
    # Layer 7 (conv4_1): 256->512, 28x28 (after pool3 from 56x56)
    Y7=28, X7=28, Z7=512, C7=256, R7=3, S7=3, Pstride7=2, Qstride7=2,
    # Layer 6 (conv3_3): 256->256, 56x56
    Y6=56, X6=56, Z6=256, C6=256, R6=3, S6=3,
    # Layer 5 (conv3_2): 256->256, 56x56
    Y5=56, X5=56, Z5=256, C5=256, R5=3, S5=3,
    # Layer 4 (conv3_1): 128->256, 56x56 (after pool2 from 112x112)
    Y4=56, X4=56, Z4=256, C4=128, R4=3, S4=3, Pstride4=2, Qstride4=2,
    # Layer 3 (conv2_2): 128->128, 112x112
    Y3=112, X3=112, Z3=128, C3=128, R3=3, S3=3,
    # Layer 2 (conv2_1): 64->128, 112x112 (after pool1 from 224x224)
    Y2=112, X2=112, Z2=128, C2=64, R2=3, S2=3, Pstride2=2, Qstride2=2,
    # Layer 1 (conv1_2): 64->64, 224x224
    Y1=224, X1=224, Z1=64, C1=64, R1=3, S1=3,
    # Layer 0 (conv1_1): 3->64, 224x224
    Y0=224, X0=224, Z0=64, R0=3, S0=3, C0=3
)

vgg16_full_coupling = conv_13layers_coupling

# Full VGG16 fusion (all 13 conv layers) would require conv_13layers_coupling
# This is computationally very expensive and rarely practical
# Note: FC layers are typically not fused with conv layers due to different data patterns

# -----------------------------------------------------------------------------
# ResNet18 (for fusion experiments)  
# Reference: "Deep Residual Learning for Image Recognition"
# 
# ResNet18 Structure (21 layers total):
# - L0: Initial 7x7 conv, stride 2 → 112x112x64
# - L1-L4: Conv2_x (4 conv layers), 56x56, 64 channels
# - L5-L9: Conv3_x (4 conv + 1 projection), 28x28, 128 channels
# - L10-L14: Conv4_x (4 conv + 1 projection), 14x14, 256 channels  
# - L15-L19: Conv5_x (4 conv + 1 projection), 7x7, 512 channels
# - L20: FC 512→1000
#
# Fusion strategy: Fuse within stages (same spatial dimension)
# Projections (1x1) are computed separately
# -----------------------------------------------------------------------------

# Single-layer ResNet18 shapes (from computations.py)
from computations import comp_resnet_18 as resnet18_single_layers

# Block-based fusion for ResNet18
# Stage 1 (Conv2_x): L0+L1+L2+L3+L4 (5 layers, 56x56, 64 channels) - uses conv_4layers_coupling
resnet18_stage1_fused = Shape(
    # Layer 4 (L4_conv2_2_2): 64->64, 56x56
    P=56, Q=56, Z4=64, C4=64, R4=3, S4=3,
    # Layer 3 (L3_conv2_2_1): 64->64, 56x56
    Y3=56, X3=56, Z3=64, C3=64, R3=3, S3=3,
    # Layer 2 (L2_conv2_1_2): 64->64, 56x56
    Y2=56, X2=56, Z2=64, C2=64, R2=3, S2=3,
    # Layer 1 (L1_conv2_1_1): 64->64, 56x56
    Y1=56, X1=56, Z1=64, C1=64, R1=3, S1=3,
    # Layer 0 (L0_conv1): 3->64, 112x112 (stride 2 from 112x112)
    Y0=112, X0=112, Z0=64, R0=7, S0=7, C0=3, Pstride0=2, Qstride0=2
)

# Stage 2: L5+L6+L7+L8(4 layers, 28x28, 128 channels)
resnet18_stage2_b2_fused = Shape(
    # Layer 8 (L9_conv3_2_2): 128->128, 28x28
    P=28, Q=28, Z3=128, C3=128, R3=3, S3=3,
    # Layer 7 (L8_conv3_2_1): 128->128, 28x28
    Y2=28, X2=28, Z2=128, C2=128, R2=3, S2=3,
    # Layer 6 (L6_conv3_1_2): 128->128, 28x28
    Y1=28, X1=28, Z1=128, C1=128, R1=3, S1=3,
    # Layer 5 (L5_conv3_1_1): 64->128, 28x28 (stride 2 from 56x56)
    Y0=56, X0=56, Z0=128, C0=64, R0=3, S0=3, Pstride0=2, Qstride0=2,
)


# Stage 3: L9+L10+L11+L12 (4 layers, 14x14, 256 channels)
resnet18_stage3_b2_fused = Shape(
    # Layer 12 (L14_conv4_2_2): 256->256, 14x14
    P=14, Q=14, Z3=256, C3=256, R3=3, S3=3,
    # Layer 11 (L13_conv4_2_1): 256->256, 14x14
    Y2=14, X2=14, Z2=256, C2=256, R2=3, S2=3,
    # Layer 10 (L11_conv4_1_2): 256->256, 14x14
    Y1=14, X1=14, Z1=256, C1=256, R1=3, S1=3,
    # Layer 9 (L10_conv4_1_1): 128->256, 14x14 (stride 2 from 28x28)
    Y0=28, X0=28, Z0=256, C0=128, R0=3, S0=3, Pstride0=2, Qstride0=2,
    
    
)

# Stage 4: L13+L14+L15+L16 (4 layers, 7x7, 512 channels)
resnet18_stage4_b2_fused = Shape(
    # Output (7x7, 512 channels)
    P=7, Q=7, Z3=512, C3=512, R3=3, S3=3,
    # Layer 15 (L18_conv5_2_1): 512->512, 7x7
    Y2=7, X2=7, Z2=512, C2=512, R2=3, S2=3,
    # Layer 14 (L16_conv5_1_2): 512->512, 7x7
    Y1=7, X1=7, Z1=512, C1=512, R1=3, S1=3,
    # Layer 13 (L15_conv5_1_1): 256->512, 7x7 (stride 2 from 14x14)
    Y0=14, X0=14, Z0=512, C0=256, R0=3, S0=3, Pstride0=2, Qstride0=2,
    )

# Dictionary of all ResNet18 block fusions
resnet18_block_fused = {
    'stage1': resnet18_stage1_fused,       # L0+L1+L2+L3+L4, 5 layers, 56x56
    'stage2_b2': resnet18_stage2_b2_fused, # L5+L6+L7+L8, 4 layers, 28x28
    'stage3_b2': resnet18_stage3_b2_fused, # L9+L10+L11+L12, 4 layers, 14x14
    'stage4_b2': resnet18_stage4_b2_fused, # L13+L14+L15+L16, 4 layers, 7x7
}

# Couplings for each block
resnet18_block_couplings = {
    'stage1': conv_5layers_coupling,
    'stage2_b2': conv_4layers_coupling,
    'stage3_b2': conv_4layers_coupling,
    'stage4_b2': conv_4layers_coupling,
}

# 2-layer fused segments (for finer granularity)
# Covers all 17 fused layers without overlap:
# s1b1(2) + s1b2(3) + s2b1(2) + s2b2(2) + s3b1(2) + s3b2(2) + s4b1(2) + s4b2(2) = 17
resnet18_2layer_fused = {
    # Stage 1, block 1: L0+L1 (7x7 stride2 + 3x3)
    's1b1': Shape(
        P=56, Q=56, Z1=64, C1=64, R1=3, S1=3,
        Y0=112, X0=112, Z0=64, R0=7, S0=7, C0=3, Pstride0=2, Qstride0=2
    ),
    # Stage 1, block 2: L2+L3+L4 (3 layers, all 56x56, 64ch — odd remainder)
    's1b2': Shape(
        P=56, Q=56, Z2=64, C2=64, R2=3, S2=3,
        Y1=56, X1=56, Z1=64, C1=64, R1=3, S1=3,
        Y0=56, X0=56, Z0=64, R0=3, S0=3, C0=64
    ),
    # Stage 2, block 1: L5+L6 (stride 2 on L5, 64→128)
    's2b1': Shape(
        P=28, Q=28, Z1=128, C1=128, R1=3, S1=3,
        Y0=56, X0=56, Z0=128, C0=64, R0=3, S0=3, Pstride0=2, Qstride0=2
    ),
    # Stage 2, block 2: L8+L9 (28x28, 128ch)
    's2b2': Shape(
        P=28, Q=28, Z1=128, C1=128, R1=3, S1=3,
        Y0=28, X0=28, Z0=128, C0=128, R0=3, S0=3
    ),
    # Stage 3, block 1: L10+L11 (stride 2 on L10, 128→256)
    's3b1': Shape(
        P=14, Q=14, Z1=256, C1=256, R1=3, S1=3,
        Y0=28, X0=28, Z0=256, C0=128, R0=3, S0=3, Pstride0=2, Qstride0=2
    ),
    # Stage 3, block 2: L13+L14 (14x14, 256ch)
    's3b2': Shape(
        P=14, Q=14, Z1=256, C1=256, R1=3, S1=3,
        Y0=14, X0=14, Z0=256, C0=256, R0=3, S0=3
    ),
    # Stage 4, block 1: L15+L16 (stride 2 on L15, 256→512)
    's4b1': Shape(
        P=7, Q=7, Z1=512, C1=512, R1=3, S1=3,
        Y0=14, X0=14, Z0=512, C0=256, R0=3, S0=3, Pstride0=2, Qstride0=2
    ),
    # Stage 4, block 2: L18+L19 (7x7, 512ch)
    's4b2': Shape(
        P=7, Q=7, Z1=512, C1=512, R1=3, S1=3,
        Y0=7, X0=7, Z0=512, C0=512, R0=3, S0=3
    ),
}

# Per-variant couplings for ResNet18 2-layer level (s1b2 is 3-layer)
resnet18_2layer_couplings = {
    's1b1': conv_2layers_coupling,
    's1b2': conv_3layers_coupling,   # 3 layers (odd remainder from stage1)
    's2b1': conv_2layers_coupling,
    's2b2': conv_2layers_coupling,
    's3b1': conv_2layers_coupling,
    's3b2': conv_2layers_coupling,
    's4b1': conv_2layers_coupling,
    's4b2': conv_2layers_coupling,
}

# =============================================================================
# ResNet18 FULL FUSION (17 main conv layers, excluding projections and FC)
# Layers: L0 (initial) + L1-L4 (stage1) + L5,L6,L8,L9 (stage2) + 
#         L10,L11,L13,L14 (stage3) + L15,L16,L18,L19 (stage4)
# This is impractical but useful for theoretical case studies
# Uses conv_17layers_coupling
#
# STRIDE INFO: Stride=2 at layers 0, 5, 9, 13 (downsampling layers)
#              Cumulative stride = 2*2*2*2 = 16 (so 7x7 output needs 112x112 input tile)
# =============================================================================
resnet18_full_fused = Shape(
    # Output (7x7, 512 channels)
    P=7, Q=7, Z16=512,
    # Layer 16 (L19_conv5_2_2): 512->512, 7x7
    C16=512, R16=3, S16=3,
    # Layer 15 (L18_conv5_2_1): 512->512, 7x7
    Y15=7, X15=7, Z15=512, C15=512, R15=3, S15=3,
    # Layer 14 (L16_conv5_1_2): 512->512, 7x7
    Y14=7, X14=7, Z14=512, C14=512, R14=3, S14=3,
    # Layer 13 (L15_conv5_1_1): 256->512, 7x7 (stride 2 from 14x14)
    Y13=14, X13=14, Z13=512, C13=256, R13=3, S13=3, Pstride13=2, Qstride13=2,
    # Layer 12 (L14_conv4_2_2): 256->256, 14x14
    Y12=14, X12=14, Z12=256, C12=256, R12=3, S12=3,
    # Layer 11 (L13_conv4_2_1): 256->256, 14x14
    Y11=14, X11=14, Z11=256, C11=256, R11=3, S11=3,
    # Layer 10 (L11_conv4_1_2): 256->256, 14x14
    Y10=14, X10=14, Z10=256, C10=256, R10=3, S10=3,
    # Layer 9 (L10_conv4_1_1): 128->256, 14x14 (stride 2 from 28x28)
    Y9=28, X9=28, Z9=256, C9=128, R9=3, S9=3, Pstride9=2, Qstride9=2,
    # Layer 8 (L9_conv3_2_2): 128->128, 28x28
    Y8=28, X8=28, Z8=128, C8=128, R8=3, S8=3,
    # Layer 7 (L8_conv3_2_1): 128->128, 28x28
    Y7=28, X7=28, Z7=128, C7=128, R7=3, S7=3,
    # Layer 6 (L6_conv3_1_2): 128->128, 28x28
    Y6=28, X6=28, Z6=128, C6=128, R6=3, S6=3,
    # Layer 5 (L5_conv3_1_1): 64->128, 28x28 (stride 2 from 56x56)
    Y5=56, X5=56, Z5=128, C5=64, R5=3, S5=3, Pstride5=2, Qstride5=2,
    # Layer 4 (L4_conv2_2_2): 64->64, 56x56
    Y4=56, X4=56, Z4=64, C4=64, R4=3, S4=3,
    # Layer 3 (L3_conv2_2_1): 64->64, 56x56
    Y3=56, X3=56, Z3=64, C3=64, R3=3, S3=3,
    # Layer 2 (L2_conv2_1_2): 64->64, 56x56
    Y2=56, X2=56, Z2=64, C2=64, R2=3, S2=3,
    # Layer 1 (L1_conv2_1_1): 64->64, 56x56
    Y1=56, X1=56, Z1=64, C1=64, R1=3, S1=3,
    # Layer 0 (L0_conv1): 3->64, 112x112 (stride 2 from 112x112)
    Y0=112, X0=112, Z0=64, R0=7, S0=7, C0=3, Pstride0=2, Qstride0=2
)

resnet18_full_coupling = conv_17layers_coupling


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_workload_stats(shape: Shape, coupling: Coupling = None) -> dict:
    """
    Calculate statistics for a workload to classify it as
    activation-dominant or weight-dominant.
    """
    total_macs = shape.FLOPs() // 2
    
    # Estimate activation size (simplified)
    if 'P' in shape and 'Q' in shape:
        # Convolution
        if 'M' in shape:
            output_size = shape.get('M', 1) * shape['P'] * shape['Q']
        elif 'Z' in shape:  # Multi-layer
            output_size = shape.get('Z', 1) * shape.get('Y', shape['P']) * shape.get('X', shape['Q'])
        else:
            output_size = shape['P'] * shape['Q']
            
        input_channels = shape.get('C', shape.get('C0', 1))
        filter_r = shape.get('R', shape.get('R0', 1))
        filter_s = shape.get('S', shape.get('S0', 1))
        input_size = input_channels * (shape['P'] + filter_r - 1) * (shape['Q'] + filter_s - 1)
    else:
        output_size = 0
        input_size = 0
    
    # Estimate weight size
    weight_size = 0
    for key in shape:
        if key.startswith('C') or key == 'M' or key.startswith('Z'):
            if key.startswith('R') or key.startswith('S'):
                continue
            # This is a rough estimate
            pass
    
    # For simplicity, use the ratio of spatial dimensions to channel dimensions
    spatial_dims = shape.get('P', 1) * shape.get('Q', 1)
    channel_dims = shape.get('C', shape.get('C0', 1)) * shape.get('M', shape.get('Z', 1))
    
    activation_weight_ratio = spatial_dims / max(channel_dims, 1)
    
    return {
        'total_macs': total_macs,
        'spatial_dims': spatial_dims,
        'channel_dims': channel_dims,
        'activation_weight_ratio': activation_weight_ratio,
        'classification': 'activation-dominant' if activation_weight_ratio > 10 else 'weight-dominant'
    }


def print_workload_summary():
    """Print a summary of all defined workloads."""
    print("=" * 80)
    print("THESIS WORKLOADS SUMMARY")
    print("=" * 80)
    
    print("\n--- ACTIVATION-DOMINANT (benefit from layer fusion) ---")
    print("\nFSRCNN Single Layers:")
    for name, shape in fsrcnn_single_layers.items():
        stats = get_workload_stats(shape)
        print(f"  {name}: {shape['P']}x{shape['Q']}, C={shape['C']}, M={shape['M']}, "
              f"R={shape['R']}x{shape['S']}, ratio={stats['activation_weight_ratio']:.1f}")
    
    print("\nMC-CNN Single Layers:")
    for name, shape in mccnn_single_layers.items():
        stats = get_workload_stats(shape)
        print(f"  {name}: {shape['P']}x{shape['Q']}, C={shape['C']}, M={shape['M']}, "
              f"R={shape['R']}x{shape['S']}, ratio={stats['activation_weight_ratio']:.1f}")
    
    print("\n--- WEIGHT-DOMINANT (less benefit from layer fusion) ---")
    print("\nVGG16 Block Fused:")
    for name, shape in vgg16_block_fused.items():
        print(f"  {name}: {shape['P']}x{shape['Q']}, C0={shape['C0']}")
    
    print("\nResNet18 2-Layer Fused:")
    for name, shape in resnet18_2layer_fused.items():
        print(f"  {name}: {shape['P']}x{shape['Q']}, C0={shape['C0']}")


# =============================================================================
# WORKLOAD COLLECTIONS FOR EXPERIMENTS
# =============================================================================

# All single-layer workloads (for non-fused baselines)
all_single_layer_workloads = {
    'fsrcnn': fsrcnn_single_layers,
    'mccnn': mccnn_single_layers,
    'vgg16': vgg16_single_layers,
    'resnet18': resnet18_single_layers,
}

# All block/2-layer fused workloads
all_fused_workloads = {
    'fsrcnn': fsrcnn_tdc_2layer_fused,
    'mccnn': mccnn_2layer_fused,
    'vgg16': vgg16_block_fused,
    'resnet18': resnet18_2layer_fused,
}

# Representative workloads for quick experiments
representative_workloads = {
    # Activation-dominant
    'fsrcnn_mapping': fsrcnn_tdc_2layer_fused['L2_L3'],  # Small channels, large spatial
    'mccnn_full': mccnn_4layer_fused,                              # Full MC-CNN fused
    
    # Weight-dominant  
    'vgg16_block5': vgg16_block5_fused,              # 512 channels, 14x14, 3 layers
    'resnet18_stage1': resnet18_stage1_fused,        # 64 channels, 56x56, 4 layers
}


if __name__ == "__main__":
    print_workload_summary()