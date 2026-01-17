from factors import Shape, Coupling

# DIMENSIONS and COUPLING for 2 Layer GEMM:
# C0: Filter0 depth/Input depth
# Y: Intermediate Out height/Input height
# Z: Filter0 num/Intermediate Out depth  
# C1: Filter1 depth/Intermediate In depth
# P: Out height
# C2: Filter1 num/Out depth
# MAC1: Intermediate_Out[z][y] += W0[z][c0] * In[c0][y]
# MAC2: Out[c2][p] += W1[c2][c1] * Intermediate_In[c1][p]
gemm_2layers_coupling = Coupling(
                                dims = ['C0', 'Y', 'Z', 'C1', 'P', 'C2'],
                                in_coupling = ['C0', 'Y'],        # In
                                w_coupling = {
                                    0: ['Z', 'C0'],               # W0
                                    1: ['C2', 'C1']               # W1
                                },
                                int_in_coupling = { 0: ['C1', 'P']},       # Intermediate_Out
                                int_out_coupling = { 0: ['Z', 'Y'] },      # Intermediate_In
                                out_coupling = ['C2', 'P'])                # Out


# DIMENSION and COUPLING for 2 Layers Convolution:
# C0: Filter0 depth/Input depth
# Y: Intermediate Out height
# X: Intermediate Out width
# R0: Filter0 height
# S0: Filter0 width
# C1: Filter1 depth/Intermediate In depth
# Z: Filter0 num/Intermediate Out depth
# R1: Filter1 height
# S1: Filter1 width
# C2: Filter1 num/Out depth
# P: Out height
# Q: Out width
# => Y+R0-1: Intermediate Out height
# => X+S0-1: Intermediate Out width
# MAC: Intermediate_Out[z][y][x] += W0[z][c0][r0][s0] * In[c0][y+r0][x+s0]
# => P+R1-1: Intermediate Input height
# => Q+S1-1: Intermediate Input width
# MAC: Out[c2][p][q] += W1[c2][c1][r1][s1] * Intermediate_In[c1][p+r1][q+s1]
conv_2layers_coupling = Coupling(
                                dims = ['C0', 'Y', 'X', 'R0', 'S0', 'Z', 'C1', 'R1', 'S1', 'C2', 'P', 'Q'],
                                in_coupling = ['C0', ['Y', 'R0'], ['X', 'S0']],      # In
                                w_coupling = {
                                    0: ['Z', 'C0', 'R0', 'S0'],               # W0
                                    1: ['C2', 'C1', 'R1', 'S1']            # W1
                                },
                                int_in_coupling = { 0: ['C1', ['P', 'R1'], ['Q', 'S1']]},                      # Intermediate_Out
                                int_out_coupling = { 0: ['Z', 'Y', 'X'] },      # Intermediate_In
                                out_coupling = ['C2', 'P', 'Q'])                       # Out

conv_3layers_coupling = Coupling(
    dims = ['C0', 'Y0', 'X0', 'R0', 'S0', 'Z0', 'C1', 'R1', 'S1', 'Y1', 'X1', 'Z1', 'C2', 'R2', 'S2', 'P', 'Q', 'Z2'],
    in_coupling = ['C0', ['Y0', 'R0'], ['X0', 'S0']],               # In
    w_coupling= {
        0: ['Z0', 'C0', 'R0', 'S0'],                                # W0
        1: ['Z1', 'C1', 'R1', 'S1'],                                # W1
        2: ['Z2', 'C2', 'R2', 'S2']                                 # W2
    },
    int_in_coupling = {
        0: ['C1', ['Y1', 'R1'], ['X1', 'S1']],                      # Intermediate_Out L1
        1: ['C2', ['P', 'R2'], ['Q', 'S2']]                         # Intermediate_Out L2
    },
    int_out_coupling = {
        0: ['Z0', 'Y0', 'X0'],                                      # Intermediate_In L1
        1: ['Z1', 'Y1', 'X1']                                       # Intermediate_In L2
    },
    out_coupling = ['Z2', 'P', 'Q']                                 # Out
)

conv_10layers_coupling = Coupling(
    dims = ['C0', 'Y0', 'X0', 'R0', 'S0', 'Z0', 'C1', 'R1', 'S1', 'Y1', 'X1', 'Z1', 'C2', 'R2', 'S2', 'Y2', 'X2', 'Z2', 'C3', 'R3', 'S3',
            'Y3', 'X3', 'Z3', 'C4', 'R4', 'S4', 'Y4', 'X4', 'Z4', 'C5', 'R5', 'S5', 'Y5', 'X5', 'Z5', 'C6', 'R6', 'S6', 'Y6', 'X6', 'Z6',
            'C7', 'R7', 'S7', 'Y7', 'X7', 'Z7', 'C8', 'R8', 'S8', 'Y8', 'X8', 'Z8', 'C9', 'R9', 'S9', 'Y9', 'X9', 'Z9', 'C10', 'R10', 'S10', 'P', 'Q', 'Z10'],
    in_coupling = ['C0', ['Y0', 'R0'], ['X0', 'S0']],               # In
    w_coupling= {
        0: ['Z0', 'C0', 'R0', 'S0'],                                # W0
        1: ['Z1', 'C1', 'R1', 'S1'],                                # W1
        2: ['Z2', 'C2', 'R2', 'S2'],                                # W2
        3: ['Z3', 'C3', 'R3', 'S3'],                                # W3
        4: ['Z4', 'C4', 'R4', 'S4'],                                # W4
        5: ['Z5', 'C5', 'R5', 'S5'],                                # W5
        6: ['Z6', 'C6', 'R6', 'S6'],                                # W6
        7: ['Z7', 'C7', 'R7', 'S7'],                                # W7
        8: ['Z8', 'C8', 'R8', 'S8'],                                # W8
        9: ['Z9', 'C9', 'R9', 'S9'],                                # W9
        10: ['Z10', 'C10', 'R10', 'S10']                            # W10   
    },
    int_in_coupling = {
        0: ['C1', ['Y1', 'R1'], ['X1', 'S1']],                      # Intermediate_In L1
        1: ['C2', ['Y2', 'R2'], ['X2', 'S2']],                      # Intermediate_In L2
        2: ['C3', ['Y3', 'R3'], ['X3', 'S3']],                      # Intermediate_In L3
        3: ['C4', ['Y4', 'R4'], ['X4', 'S4']],                      # Intermediate_In L4
        4: ['C5', ['Y5', 'R5'], ['X5', 'S5']],                      # Intermediate_In L5
        5: ['C6', ['Y6', 'R6'], ['X6', 'S6']],                      # Intermediate_In L6
        6: ['C7', ['Y7', 'R7'], ['X7', 'S7']],                      # Intermediate_In L7
        7: ['C8', ['Y8', 'R8'], ['X8', 'S8']],                      # Intermediate_In L8
        8: ['C9', ['Y9', 'R9'], ['X9', 'S9']],                      # Intermediate_In L9
        9: ['C10', ['P', 'R10'], ['Q', 'S10']]                      # Intermediate_In L10
    },
    int_out_coupling = {
        0: ['Z0', 'Y0', 'X0'],                                      # Intermediate_Out L0
        1: ['Z1', 'Y1', 'X1'],              
        2: ['Z2', 'Y2', 'X2'],
        3: ['Z3', 'Y3', 'X3'],
        4: ['Z4', 'Y4', 'X4'],
        5: ['Z5', 'Y5', 'X5'],
        6: ['Z6', 'Y6', 'X6'],
        7: ['Z7', 'Y7', 'X7'],
        8: ['Z8', 'Y8', 'X8'],
        9: ['Z9', 'Y9', 'X9']
    },
    out_coupling = ['Z10', 'P', 'Q']
)

easy_conv_3layers_coupling = Coupling(
    dims = ['Q', 'Z2', 'C2', 'S2', 'X1', 'Z1', 'C1', 'S1', 'X0', 'Z0', 'S0', 'C0'],
    in_coupling = ['C0', ['X0', 'S0']],      # In
    w_coupling= {
        0: ['Z0', 'C0', 'S0'],               # W0
        1: ['Z1', 'C1', 'S1'],               # W1
        2: ['Z2', 'C2', 'S2']                # W2
    },
    int_in_coupling = {
        0: ['C1', ['X1', 'S1']],                      # Intermediate_Out L1
        1: ['C2', ['Q', 'S2']]                        # Intermediate_Out L2
    },
    int_out_coupling = {
        0: ['Z0', 'X0'],      # Intermediate_In L1
        1: ['Z1', 'X1']       # Intermediate_In L2
    },
    out_coupling = ['Z2', 'Q']                        # Out
)

# DIMENSIONS and COUPLING for GEMMS:
# M: Weight/Out rows
# K: Inner dimension, Weight cols/In rows
# N: In/Out cols
# ==> MAC: Out[m][n] += W[m][k] * In[k][n]
gemm_coupling = Coupling(dims = ['M', 'K', 'N'], in_coupling = ['K', 'N'], w_coupling = {0: ['M', 'K']}, out_coupling = ['M', 'N'])

# DIMENSIONS and COUPLING for CONVOLUTIONS:
# M: Filter num/Out depth
# P: Out height
# Q: Out width
# C: Filter/Input depth
# R: Filter height
# S: Filter width
# => P+R-1: Input height
# => Q+S-1: Input width
# ==> MAC: Out[m][p][q] += W[m][c][r][s] * In[c][p+r][q+s]
# VGG16-L0 -> C: 3, M: 64, P: 224, Q: 224, R: 3, S: 3
# Output shape: [64, 222, 222], Weight shape: [64, 3, 3, 3], Input shape: [3, 224, 224]
conv_coupling = Coupling(
    dims = ['M', 'P', 'Q', 'C', 'R', 'S'], 
    in_coupling = ['C', ['P', 'R'], ['Q', 'S']], 
    w_coupling = {0: ['M', 'C', 'R', 'S']},  # Changed from list to dict
    out_coupling = ['M', 'P', 'Q']
)
# WITH STRIDE the indexing becomes:
# => Pstride*P+Rdilation*R-1: Input height
# => Qstride*Q+Sdilation*S-1: Input width
# ==> MAC: Out[m][p][q] += W[m][c][r][s] * In[c][p*Pstride+r*Rdilation][q*Qstride+s*Sdilation]
conv_coupling_with_stride = Coupling(
    dims = ['M', 'P', 'Q', 'C', 'R', 'S'], 
    in_coupling = ['C', ['P', 'R'], ['Q', 'S']], 
    w_coupling = {0: ['M', 'C', 'R', 'S']},  # Changed from list to dict
    out_coupling = ['M', 'P', 'Q'], 
    in_strides = {'P': 'Pstride', 'R': 'Rdilation', 'Q': 'Qstride', 'S': 'Sdilation'}
)
# WITH BATCHES too we get:
# N: Batch size
# ==> MAC: Out[n][m][p][q] += W[m][c][r][s] * In[n][c][p*Pstride+r*Rdilation][q*Qstride+s*Sdilation]
conv_coupling_with_stride_and_batches = Coupling(
    dims = ['N', 'M', 'P', 'Q', 'C', 'R', 'S'], 
    in_coupling = ['N', 'C', ['P', 'R'], ['Q', 'S']], 
    w_coupling = {0: ['M', 'C', 'R', 'S']},  # Changed from list to dict
    out_coupling = ['N', 'M', 'P', 'Q'], 
    in_strides = {'P': 'Pstride', 'R': 'Rdilation', 'Q': 'Qstride', 'S': 'Sdilation'}
)
# In a TRANSPOSED CONVOLUTION DIMENSIONS become:
# P: Input height
# Q: Input width
# => P+R-1: Out height
# => Q+S-1: Out width
# ==> MAC: Out[m][p+r][q+s] += W[m][c][r][s] * In[c][p][q] (stride and dilation omitted for clarity)
transposed_conv_coupling = Coupling(
    dims = ['M', 'P', 'Q', 'C', 'R', 'S'], 
    in_coupling = ['C', 'P', 'Q'], 
    w_coupling = {0: ['M', 'C', 'R', 'S']},  # Changed from list to dict
    out_coupling = ['M', ['P', 'R'], ['Q', 'S']], 
    out_strides = {'P': 'Pstride', 'R': 'Rdilation', 'Q': 'Qstride', 'S': 'Sdilation'}
)
# WITH BATCHES too we get:
# N: Batch size
# ==> MAC: Out[n][m][p+r][q+s] += W[m][c][r][s] * In[n][c][p][q] (stride and dilation omitted for clarity)
transposed_conv_coupling_with_batches = Coupling(
    dims = ['N', 'M', 'P', 'Q', 'C', 'R', 'S'], 
    in_coupling = ['N', 'C', 'P', 'Q'], 
    w_coupling = {0: ['M', 'C', 'R', 'S']},  # Changed from list to dict
    out_coupling = ['N', 'M', ['P', 'R'], ['Q', 'S']], 
    out_strides = {'P': 'Pstride', 'R': 'Rdilation', 'Q': 'Qstride', 'S': 'Sdilation'}
)
# NOTE: each comp must be strictly compatible with its coupling, that is, it must assign a value to each of the coupling's dimensions.
#       Then, the comp's coupling may happen to be a subcoupling of the one used to define the current architecture.


"""
Generates computation instances for each GEMM of a BERT Transformer
with arbitrary parameters/dimensions. See:
"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
"""
def comp_BERT(embedding : int, seq_length : int, heads : int, ff_dim : int) -> dict[str, Shape]:
    assert embedding % heads == 0, f"Embedding dim ({embedding}) must be divisible by the number of heads ({heads})."
    return {
        'KQV': Shape(
            M = embedding*3,
            K = embedding,
            N = seq_length
            ),
        'KTQ': Shape(
            M = seq_length,
            K = embedding//heads,
            N = seq_length
            ),
        'VScores': Shape(
            M = embedding//heads,
            K = seq_length,
            N = seq_length
            ),
        'Out': Shape(
            M = embedding,
            K = embedding,
            N = seq_length
            ),
        'FF1': Shape(
            M = ff_dim,
            K = embedding,
            N = seq_length
            ),
        'FF2': Shape(
            M = embedding,
            K = ff_dim,
            N = seq_length
            )
    }

comp_BERT_base = comp_BERT(768, 1024, 12, 3072)
comp_BERT_large = comp_BERT(1024, 4096, 16, 4096)

comp_harsh_factos_1 = Shape(
    M = 4000,
    K = 6032,
    N = 12000
    )

comp_harsh_factos_2 = Shape(
    M = 7000,
    K = 1440,
    N = 4224
    )

comp_requiring_padding = Shape(
    M = 4037,
    K = 6011,
    N = 12071
    )

"""
GEMMs coming from scientific applications, taken from previous literature:
"Evaluating Spatial Accelerator Architectures with Tiled Matrix-Matrix Multiplication"
"""
comp_maestro_blas = {
    'MB1': Shape(
        M = 8192,
        K = 8192,
        N = 8192
    ),
    'MB2': Shape(
        M = 1024,
        K = 8192,
        N = 1024
    ),
    'MB3': Shape(
        M = 8,
        K = 8192,
        N = 8
    ),
    'MB4': Shape(
        M = 8,
        K = 1024,
        N = 8192
    ),
    'MB5': Shape(
        M = 8192,
        K = 1024,
        N = 8
    ),
    'MB6': Shape(
        M = 512,
        K = 256,
        N = 256
    )
}

"""
Convolutions from the layers of VGG16. See:
"Very Deep Convolutional Networks for Large-Scale Image Recognition"
"""
comp_vgg_16 = {
    'L0': Shape(C = 3, M = 64, P = 224, Q = 224, R = 3, S = 3),
    'L1': Shape(C = 64, M = 64, P = 224, Q = 224, R = 3, S = 3),
    'L2': Shape(C = 64, M = 128, P = 112, Q = 112, R = 3, S = 3),
    'L3': Shape(C = 128, M = 128, P = 112, Q = 112, R = 3, S = 3),
    'L4': Shape(C = 128, M = 256, P = 56, Q = 56, R = 3, S = 3),
    'L5': Shape(C = 256, M = 256, P = 56, Q = 56, R = 3, S = 3),
    #'L6': Shape(C = 256, M = 256, P = 56, Q = 56, R = 3, S = 3),
    'L7': Shape(C = 256, M = 512, P = 28, Q = 28, R = 3, S = 3),
    'L8': Shape(C = 512, M = 512, P = 28, Q = 28, R = 3, S = 3),
    #'L9': Shape(C = 512, M = 512, P = 28, Q = 28, R = 3, S = 3),
    'L10': Shape(C = 512, M = 512, P = 14, Q = 14, R = 3, S = 3),
    #'L11': Shape(C = 512, M = 512, P = 14, Q = 14, R = 3, S = 3),
    #'L12': Shape(C = 512, M = 512, P = 14, Q = 14, R = 3, S = 3),
    'L13': Shape(C = 25088, M = 4096, P = 1, Q = 1, R = 1, S = 1), # fully connected
    'L14': Shape(C = 4096, M = 4096, P = 1, Q = 1, R = 1, S = 1), # fully connected
    'L15': Shape(C = 4096, M = 1000, P = 1, Q = 1, R = 1, S = 1), # fully connected
    'L3+': Shape(C = 128, M = 128, P = 112, Q = 112, R = 9, S = 9) # large filter experiment
}

"""
Convolutions from the layers of ResNet18.
"""
comp_resnet_18 = {
    'L0': Shape(C = 3, M = 64, P = 112, Q = 112, R = 7, S = 7, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1),
    'L1': Shape(C = 64, M = 64, P = 56, Q = 56, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    #'L2': Shape(C = 64, M = 64, P = 56, Q = 56, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    #'L3': Shape(C = 64, M = 64, P = 56, Q = 56, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    #'L4': Shape(C = 64, M = 64, P = 56, Q = 56, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'L5': Shape(C = 64, M = 128, P = 28, Q = 28, R = 3, S = 3, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1),
    'L6': Shape(C = 128, M = 128, P = 28, Q = 28, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'L7': Shape(C = 64, M = 128, P = 28, Q = 28, R = 1, S = 1, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1), # point-wise
    #'L8': Shape(C = 128, M = 128, P = 28, Q = 28, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    #'L9': Shape(C = 128, M = 128, P = 28, Q = 28, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'L10': Shape(C = 128, M = 128, P = 14, Q = 14, R = 3, S = 3, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1),
    'L11': Shape(C = 256, M = 256, P = 14, Q = 14, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'L12': Shape(C = 128, M = 256, P = 14, Q = 14, R = 1, S = 1, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1), # point-wise
    #'L13': Shape(C = 256, M = 256, P = 14, Q = 14, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    #'L14': Shape(C = 256, M = 256, P = 14, Q = 14, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'L15': Shape(C = 256, M = 512, P = 7, Q = 7, R = 3, S = 3, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1),
    'L16': Shape(C = 512, M = 512, P = 7, Q = 7, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'L17': Shape(C = 256, M = 512, P = 7, Q = 7, R = 1, S = 1, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1), # point-wise
    #'L18': Shape(C = 512, M = 512, P = 7, Q = 7, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    #'L19': Shape(C = 512, M = 512, P = 7, Q = 7, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'L20': Shape(C = 512, M = 1000, P = 1, Q = 1, R = 1, S = 1, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1), # fully connected
    'L1+': Shape(C = 256, M = 256, P = 56, Q = 56, R = 3, S = 3, Pstride = 2, Qstride = 2, Rdilation = 3, Sdilation = 3), # 2D dilation experiment
    'L3+': Shape(C = 128, M = 128, P = 112, Q = 112, R = 9, S = 9, Pstride = 1, Qstride = 4, Rdilation = 1, Sdilation = 3) # 1D dilation experiment
}

"""
Convolutions chosen as benchmark for the tool.
"""
benchmark_convs = {
    # VGG16
    'I': Shape(C = 128, M = 256, P = 56, Q = 56, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'II': Shape(C = 512, M = 512, P = 28, Q = 28, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    # ResNet18 and 50
    'III': Shape(C = 3, M = 64, P = 112, Q = 112, R = 7, S = 7, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1),
    'IV': Shape(C = 64, M = 64, P = 56, Q = 56, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'V': Shape(C = 128, M = 128, P = 28, Q = 28, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'VI': Shape(C = 256, M = 256, P = 14, Q = 14, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'VII': Shape(C = 256, M = 512, P = 7, Q = 7, R = 3, S = 3, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1),
    'VIII': Shape(C = 64, M = 256, P = 56, Q = 56, R = 1, S = 1, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1), # point-wise
    # MobileNetV3
    'IX': Shape(C = 3, M = 96, P = 176, Q = 176, R = 3, S = 3, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1),
    'X': Shape(C = 72, M = 72, P = 28, Q = 28, R = 3, S = 3, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1),
    'XI': Shape(C = 576, M = 576, P = 7, Q = 7, R = 5, S = 5, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'XII': Shape(C = 24, M = 88, P = 28, Q = 28, R = 1, S = 1, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1), # point-wise
    # Strides and Dilation
    'XIII': Shape(C = 16, M = 16, P = 224, Q = 224, R = 3, S = 3, Pstride = 3, Qstride = 3, Rdilation = 4, Sdilation = 4),
    'XIV': Shape(C = 128, M = 128, P = 112, Q = 112, R = 9, S = 9, Pstride = 4, Qstride = 4, Rdilation = 3, Sdilation = 3),
    'XV': Shape(C = 256, M = 256, P = 56, Q = 56, R = 3, S = 3, Pstride = 2, Qstride = 2, Rdilation = 3, Sdilation = 3)
}
benchmark_convs_transposed = {
    # Transposed convs
    'XVI': Shape(C = 128, M = 256, P = 32, Q = 32, R = 4, S = 4, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'XVII': Shape(C = 576, M = 576, P = 7, Q = 7, R = 5, S = 5, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1)
}
benchmark_convs_batched = {
    # Batched convs
    'XVIII': Shape(N = 64, C = 256, M = 256, P = 14, Q = 14, R = 3, S = 3, Pstride = 1, Qstride = 1, Rdilation = 1, Sdilation = 1),
    'XIX': Shape(N = 128, C = 72, M = 72, P = 28, Q = 28, R = 3, S = 3, Pstride = 2, Qstride = 2, Rdilation = 1, Sdilation = 1),
    'XX': Shape(N = 32, C = 256, M = 256, P = 56, Q = 56, R = 5, S = 5, Pstride = 2, Qstride = 2, Rdilation = 3, Sdilation = 3)
}

"""
    Creates a coupling for an N-layer convolution.
    
    Args:
        num_layers: Number of convolution layers
        with_stride: Whether to include stride parameters
        with_batches: Whether to include batch dimension
    
    Returns:
        A Coupling object representing an N-layer convolution
"""    
def create_nlayer_conv_coupling(num_layers: int, with_stride: bool = False, with_batches: bool = False) -> Coupling:
    # Build dimensions list
    dims = ['P', 'Q', 'C0']
    
    # Add batch dimension if needed
    if with_batches:
        dims.append('N')

    for i in range(num_layers-1):
        dims.extend([f'Y{i}', f'X{i}'])
        
    # Add dimensions for each layer
    for i in range(num_layers):
        # Add filter dimensions for this layer
        dims.extend([f'R{i}', f'S{i}'])
        # Add channel dimensions 
        dims.append(f'C{i+1}')


    # Initialize coupling variables
    in_coupling = None
    w_coupling = {}
    int_in_coupling = {}
    int_out_coupling = {}
    out_coupling = None

    if num_layers == 1:
        # Single layer convolution
        dims = ['M', 'P', 'Q', 'C', 'R', 'S']
        in_coupling = ['C', ['P', 'R'], ['Q', 'S']]
        w_coupling = {0: ['M', 'C', 'R', 'S']}
        out_coupling = ['M', 'P', 'Q']
    elif num_layers == 2:
        # For 2 layers, use the existing conv_2layers_coupling structure
        dims = ['C0', 'Y', 'X', 'R0', 'S0', 'Z', 'C1', 'R1', 'S1', 'C2', 'P', 'Q']
        in_coupling = ['C0', ['Y', 'R0'], ['X', 'S0']]
        w_coupling = {
            0: ['Z', 'C0', 'R0', 'S0'],
            1: ['C2', 'C1', 'R1', 'S1']
        }
        int_out_coupling = {0: ['Z', 'Y', 'X']}
        int_in_coupling = {0: ['C1', ['P', 'R1'], ['Q', 'S1']]}
        out_coupling = ['C2', 'P', 'Q']    
    # For more than 2 layers, set up intermediate layers        
    else:
        in_coupling = ['C0', ['Y0', 'R0'], ['X0', 'S0']]
        for i in range(num_layers):
            w_coupling[i] = [f'C{i+1}', f'C{i}', f'R{i}', f'S{i}']
        for i in range(num_layers - 1):
            int_out_coupling[i] = [f'C{i+1}', f'Y{i}', f'X{i}']
            int_in_coupling[i] = [f'C{i+1}', [f'Y{i+1}', f'R{i+1}'], [f'X{i+1}', f'S{i+1}']]
        for i in range(num_layers):
            if i == 0:
                # First layer: input to first intermediate
                w_coupling[i] = ['C1', 'C0', 'R0', 'S0']
                if i < num_layers - 1:
                    int_out_coupling[i] = ['C1', 'Y0', 'X0']
            elif i == num_layers - 1:
                # Last layer: final intermediate to output
                in_coupling = ['C' + str(i), ['Y' + str(i-1), 'R' + str(i)], ['X' + str(i-1), 'S' + str(i)]]
                w_coupling[i] = ['C' + str(i+1), 'C' + str(i), 'R' + str(i), 'S' + str(i)]
                out_coupling = ['C' + str(i+1), 'P', 'Q']
                dims.extend(['C' + str(i+1)])
            else:
                # Middle layers: intermediate to intermediate
                int_in_coupling[i-1] = ['C' + str(i), ['Y' + str(i-1), 'R' + str(i)], ['X' + str(i-1), 'S' + str(i)]]
                w_coupling[i] = ['C' + str(i+1), 'C' + str(i), 'R' + str(i), 'S' + str(i)]
                int_out_coupling[i-1] = ['C' + str(i), 'Y' + str(i-1), 'X' + str(i-1)]
                dims.extend(['Y' + str(i-1), 'X' + str(i-1), 'C' + str(i)])
    
    if with_batches and num_layers > 1:
        in_coupling.insert(0, 'N')
        out_coupling.insert(0, 'N')
        for i in range(num_layers - 1):
            int_in_coupling[i].insert(0, 'N')
            int_out_coupling[i].insert(0, 'N')
       
        
    # Create strides if needed
    in_strides = None
    w_strides = None
    out_strides = None
    
    if with_stride:
        in_strides = {}
        in_strides['P'] = 'Pstride'
        in_strides['Q'] = 'Qstride'
        for i in range(num_layers):
            in_strides[f'R{i}'] = f'Rdilation{i}'
            in_strides[f'S{i}'] = f'Sdilation{i}'
    
    return Coupling(
        dims = dims,
        in_coupling = in_coupling,
        w_coupling = w_coupling,
        int_in_coupling = int_in_coupling if int_in_coupling else None,
        int_out_coupling = int_out_coupling if int_out_coupling else None,
        out_coupling = out_coupling,
        in_strides = in_strides,
        w_strides = w_strides,
        out_strides = out_strides
    )

# 3-layer convolution example
#conv_3layers_coupling = create_nlayer_conv_coupling(num_layers=3)

# 4-layer convolution example
#conv_4layers_coupling = create_nlayer_conv_coupling(num_layers=4)

# 3-layer convolution with stride and batches
#conv_3layer_with_stride_and_batches = create_nlayer_conv_coupling(num_layers=3, with_stride=True, with_batches=True)

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

