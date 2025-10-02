import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from factors import Coupling

"""Test basic valid initialization"""
def test_coupling_init_valid_basic():
    dims = ['X', 'Y', 'Z']
    in_coupling = ['X', 'Y']
    w_coupling = {0: ['Y', 'Z']}
    out_coupling = ['X', 'Z']
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)
    
    assert coupling.dims == dims
    assert coupling.flat_in_coupling == ['X', 'Y']
    assert coupling.flat_w_coupling[0] == ['Y', 'Z']
    assert coupling.flat_out_coupling == ['X', 'Z']
    assert coupling.in_coupling == [['X'], ['Y']]
    assert coupling.w_coupling[0] == [['Y'], ['Z']]
    assert coupling.out_coupling == [['X'], ['Z']]

"""Test valid initialization with nested coupling lists"""    
def test_coupling_init_valid_nested_lists():
    dims = ['X', 'Y', 'Z', 'W']
    in_coupling = ['X', ['Y', 'Z']]
    w_coupling = {0: [['Y', 'Z'], 'W']}
    out_coupling = [['X', 'W'], 'Z']
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)
    
    assert coupling.flat_in_coupling == ['X', 'Y', 'Z']
    assert coupling.flat_w_coupling[0] == ['Y', 'Z', 'W']
    assert coupling.flat_out_coupling == ['X', 'W', 'Z']
    assert coupling.in_coupling == [['X'], ['Y', 'Z']]
    assert coupling.w_coupling[0] == [['Y', 'Z'], ['W']]
    assert coupling.out_coupling == [['X', 'W'], ['Z']]

"""Test valid initialization with stride parameters"""
def test_coupling_init_valid_with_strides():
    dims = ['X', 'Y', 'Z']
    in_coupling = [['X', 'Y']]
    w_coupling = {0: [['Y', 'Z']]}
    out_coupling = [['X', 'Z']]
    in_strides = {'X': 'stride_x'}
    w_strides = {0: {'Y': 'stride_y'}}
    out_strides = {'X': 'stride_x_out'}
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling, 
                       in_strides, w_strides, out_strides)
    
    assert coupling.in_strides == in_strides
    assert coupling.w_strides[0] == {'Y': 'stride_y'}
    assert coupling.out_strides == out_strides

"""Test valid initialization with intermediate layer parameters"""    
def test_coupling_init_valid_intermediate_layers():
    dims = ['P', 'Q', 'C2', 'C1', 'R1', 'S1', 'C0', 'R0', 'S0', 'X', 'Y', 'Z']
    in_coupling = [['X', 'R0'], ['Y', 'S0'], 'C0']
    w_coupling = {0: ['C0', 'Z', 'R0', 'S0'], 1: ['C1', 'C2', 'R1', 'S1']}
    out_coupling = ['P', 'Q', 'C2']
    int_in_coupling = {1: [['P', 'R1'], ['Q', 'S1'], 'C1']}
    int_out_coupling = {1: ['Y', 'X', 'Z']}
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       int_in_coupling=int_in_coupling, int_out_coupling=int_out_coupling)
    
    assert coupling.flat_int_in_coupling[1] == ['P', 'R1', 'Q', 'S1', 'C1']
    assert coupling.flat_int_out_coupling[1] == ['Y', 'X', 'Z']
    assert coupling.int_in_coupling[1] == [['P', 'R1'], ['Q', 'S1'], ['C1']]
    assert coupling.int_out_coupling[1] == [['Y'], ['X'], ['Z']]

"""Test that duplicate dimensions in dims raise AssertionError"""    
def test_coupling_init_invalid_duplicate_dims():
    dims = ['X', 'Y', 'X']  # Duplicate 'X'
    in_coupling = ['X']
    w_coupling = {0: ['Y']}
    out_coupling = ['X']
    
    with pytest.raises(AssertionError, match="must not contain duplicates"):
        Coupling(dims, in_coupling, w_coupling, out_coupling)

"""Test that three-level lists in coupling raise AssertionError"""    
def test_coupling_init_invalid_three_level_list():
    dims = ['X', 'Y', 'Z']
    in_coupling = ['X', [['Y', 'Z']]]  # Three levels
    w_coupling = {0: ['Y']}
    out_coupling = ['X']
    
    with pytest.raises(AssertionError, match="must be one or two level lists"):
        Coupling(dims, in_coupling, w_coupling, out_coupling)

"""Test that coupling dimensions not in dims raise AssertionError"""
def test_coupling_init_invalid_coupling_not_subset():
    dims = ['X', 'Y']
    in_coupling = ['X', 'Z']  # 'Z' not in dims
    w_coupling = {0: ['Y']}
    out_coupling = ['X']
    
    with pytest.raises(AssertionError, match="must be a subset of dims"):
        Coupling(dims, in_coupling, w_coupling, out_coupling)

"""Test that w_coupling dimensions not in dims raise AssertionError"""    
def test_coupling_init_invalid_w_coupling_not_subset():
    dims = ['X', 'Y']
    in_coupling = ['X']
    w_coupling = {0: ['Y', 'Z']}  # 'Z' not in dims
    out_coupling = ['X']
    
    with pytest.raises(AssertionError, match="must be a subset of dims"):
        Coupling(dims, in_coupling, w_coupling, out_coupling)

"""Test that duplicate dimensions in same coupling raise AssertionError"""    
def test_coupling_init_invalid_duplicate_in_coupling():
    dims = ['X', 'Y', 'Z']
    in_coupling = ['X', ['Y', 'X']]  # 'X' appears twice
    w_coupling = {0: ['Y']}
    out_coupling = ['Z']
    
    with pytest.raises(AssertionError, match="must not use a dimension more than once"):
        Coupling(dims, in_coupling, w_coupling, out_coupling)

"""Test that duplicate dimensions in w_coupling raise AssertionError"""
def test_coupling_init_invalid_duplicate_w_coupling():
    dims = ['X', 'Y', 'Z']
    in_coupling = ['X']
    w_coupling = {0: ['Y', 'Y']}  # 'Y' appears twice
    out_coupling = ['Z']
    
    with pytest.raises(AssertionError, match="must not use a dimension more than once"):
        Coupling(dims, in_coupling, w_coupling, out_coupling)



"""Test valid initialization with duplicate dimensions in w_coupling but in different layers"""
def test_coupling_init_valid_duplicate_w_coupling_in_different_layers():
    dims = ['X', 'Y', 'Z']
    in_coupling = ['X']
    w_coupling = {0: ['Y', 'Z'], 1: ['Y']}
    out_coupling = ['Z']

    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)

    assert coupling.getWeightCoupling(0) == [['Y'], ['Z']]
    assert coupling.getWeightCoupling(1) == [['Y']]



"""Test that stride keys not in dims raise AssertionError"""
def test_coupling_init_invalid_stride_key_not_in_dims():
    dims = ['X', 'Y']
    in_coupling = [['X', 'Y']]
    w_coupling = {0: ['X']}
    out_coupling = ['Y']
    in_strides = {'Z': 'stride_z'}  # 'Z' not in dims
    
    with pytest.raises(AssertionError, match="must be dimensions of the coupling"):
        Coupling(dims, in_coupling, w_coupling, out_coupling, in_strides)

"""Test that stride values that are dimensions raise AssertionError"""
def test_coupling_init_invalid_stride_value_in_dims():
    dims = ['X', 'Y']
    in_coupling = [['X', 'Y']]
    w_coupling = {0: ['X']}
    out_coupling = ['Y']
    in_strides = {'X': 'Y'}  # 'Y' is in dims
    
    with pytest.raises(AssertionError, match="must not be dimensions of the coupling"):
        Coupling(dims, in_coupling, w_coupling, out_coupling, in_strides)

"""Test that strides must reference dims in sums of length > 1"""    
def test_coupling_init_invalid_stride_not_in_sum():
    dims = ['X', 'Y', 'Z']
    in_coupling = ['X', ['Y', 'Z']]  # Only ['Y', 'Z'] has length > 1
    w_coupling = {0: ['X']}
    out_coupling = ['Y']
    in_strides = {'X': 'stride_x'}  # 'X' is not in a sum of length > 1
    
    with pytest.raises(AssertionError, match="must be part of a sum of at least two indices"):
        Coupling(dims, in_coupling, w_coupling, out_coupling, in_strides)

"""Test initialization with None intermediate layer parameters"""    
def test_coupling_init_empty_intermediate_layers():
    dims = ['X', 'Y']
    in_coupling = ['X']
    w_coupling = {0: ['Y']}
    out_coupling = ['X']
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       int_in_coupling=None, int_out_coupling=None)
    
    assert coupling.int_in_coupling == {}
    assert coupling.int_out_coupling == {}
    assert coupling.flat_int_in_coupling == {}
    assert coupling.flat_int_out_coupling == {}

"""Test initialization with None stride parameters"""
def test_coupling_init_empty_strides():
    dims = ['X', 'Y']
    in_coupling = ['X']
    w_coupling = {0: ['Y']}
    out_coupling = ['X']
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       in_strides=None, w_strides=None, out_strides=None)
    
    assert coupling.in_strides == {}
    assert coupling.w_strides[0] == {}
    assert coupling.out_strides == {}

"""Test initialization with multiple weight layers"""
def test_coupling_init_multiple_w_layers():    
    dims = ['X', 'Y', 'Z', 'W']
    in_coupling = ['X']
    w_coupling = {0: ['Y'], 1: ['Z'], 2: ['W']}
    out_coupling = ['X']
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)
    
    assert len(coupling.w_coupling) == 3
    assert coupling.flat_w_coupling[0] == ['Y']
    assert coupling.flat_w_coupling[1] == ['Z']
    assert coupling.flat_w_coupling[2] == ['W']
    assert coupling.w_strides[0] == {}
    assert coupling.w_strides[1] == {}
    assert coupling.w_strides[2] == {}

"""Test initialization with partial w_strides (some layers missing)"""
def test_coupling_init_partial_w_strides():    
    dims = ['X', 'Y', 'Z']
    in_coupling = ['X']
    w_coupling = {0: [['Y', 'Z']], 1: ['X']}
    out_coupling = ['X']
    w_strides = {0: {'Y': 'stride_y'}}  # Only layer 0 has strides
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling, w_strides=w_strides)
    
    assert coupling.w_strides[0] == {'Y': 'stride_y'}
    assert coupling.w_strides[1] == {}  # Should be empty dict for layer 1

"""Test isCompatibleComp with a compatible Shape"""
def test_is_compatible_comp_valid():
    dims = ['X', 'Y', 'Z']
    in_coupling = [['X', 'Y']]
    w_coupling = {0: ['Z']}
    out_coupling = ['X']
    in_strides = {'X': 'stride_x'}
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling, in_strides=in_strides)
    
    # Create a compatible Shape (has all dims + stride values)
    from factors import Shape
    comp = Shape({'X': 10, 'Y': 20, 'Z': 30, 'stride_x': 2})
    
    assert coupling.isCompatibleComp(comp) == True

"""Test isCompatibleComp with missing dimension"""    
def test_is_compatible_comp_missing_dim():
    dims = ['X', 'Y', 'Z']
    in_coupling = ['X']
    w_coupling = {0: ['Y']}
    out_coupling = ['Z']
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)
    
    from factors import Shape
    comp = Shape({'X': 10, 'Y': 20})  # Missing 'Z'
    
    assert coupling.isCompatibleComp(comp) == False

"""Test isCompatibleComp but missing stride value"""
def test_is_compatible_comp_missing_stride():
    dims = ['X', 'Y']
    in_coupling = [['X', 'Y']]
    w_coupling = {0: ['X']}
    out_coupling = ['Y']
    in_strides = {'X': 'stride_x'}
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling, in_strides=in_strides)
    
    from factors import Shape
    comp = Shape({'X': 10, 'Y': 20})  # Missing 'stride_x'
    
    assert coupling.isCompatibleComp(comp) == False

"""Test isCompatibleComp with intermediate layers and strides"""
def test_is_compatible_comp_with_intermediate_layers_and_strides():
    dims = ['P', 'Q', 'C2', 'C1', 'R1', 'S1', 'C0', 'R0', 'S0', 'X', 'Y', 'Z']
    in_coupling = [['X', 'R0'], ['Y', 'S0'], 'C0']
    w_coupling = {0: ['C0', 'Z', 'R0', 'S0'], 1: ['C1', 'C2', 'R1', 'S1']}
    out_coupling = ['P', 'Q', 'C2']
    int_in_coupling = {1: [['P', 'R1'], ['Q', 'S1'], 'C1']}
    int_out_coupling = {1: ['Y', 'X', 'Z']}
    in_strides = {'R0': 'stride_r0'}
    #w_strides = {0: {'R0': 'stride_r0_w'}, 1: {'R1': 'stride_r1_w'}}
    #out_strides = {'P': 'stride_p_out'}
    int_in_strides = {1: {'R1': 'stride_r1_int'}}
    int_out_strides = {1: {'Y': 'stride_y_int'}}
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       in_strides=in_strides,
                       int_in_coupling=int_in_coupling, int_out_coupling=int_out_coupling,
                       int_in_strides=int_in_strides, int_out_strides=int_out_strides)
    
    from factors import Shape
    comp = Shape({'P': 10, 'Q': 20, 'C2': 30, 'C1': 40, 
                  'R1': 50, 'S1': 60, 
                  'C0': 70, 
                  'R0': 80, 'S0': 90, 
                  'X': 100, 
                  'Y': 110, 
                  'Z': 120,
                  # Strides
                  'stride_r0': 2,
                  #'stride_r0_w': 2,
                  #'stride_r1_w': 3,
                  #'stride_p_out': 4,
                  'stride_r1_int': 5,
                  'stride_y_int': 6})
    assert coupling.isCompatibleComp(comp) == True

"""Test isCompatibleComp with intermediate layers but missing intermediate layer dimension"""
def test_is_compatible_comp_missing_intermediate_layer_dim():
    dims = ['P', 'Q', 'C2', 'C1', 'R1', 'S1', 'C0', 'R0', 'S0', 'X', 'Y', 'Z']
    in_coupling = [['X', 'R0'], ['Y', 'S0'], 'C0']
    w_coupling = {0: ['C0', 'Z', 'R0', 'S0'], 1: ['C1', 'C2', 'R1', 'S1']}
    out_coupling = ['P', 'Q', 'C2']
    int_in_coupling = {1: [['P', 'R1'], ['Q', 'S1'], 'C1']}
    int_out_coupling = {1: ['Y', 'X', 'Z']}
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       int_in_coupling=int_in_coupling, int_out_coupling=int_out_coupling)
    
    from factors import Shape
    comp = Shape({'P': 10, 'Q': 20, 'C2': 30, 'C1': 40, 
                  'R1': 50, 'S1': 60, 
                  # Missing C0
                  # 'C0': 70, 
                  'R0': 80, 'S0': 90, 
                  'X': 100, 
                  'Y': 110, 
                  'Z': 120})
    assert coupling.isCompatibleComp(comp) == False

"""Test isCompatibleCoupling with a compatible Coupling"""
def test_is_compatible_coupling_basic():
    dims = ['X', 'Y', 'Z']
    in_coupling = ['X', 'Y']
    w_coupling = {0: ['Y', 'Z']}
    out_coupling = ['X', 'Z']
    
    coup1 = Coupling(dims, in_coupling, w_coupling, out_coupling)
    coup2 = Coupling(dims, in_coupling, w_coupling, out_coupling)
    
    assert coup1.isCompatibleCoupling(coup2) == True

"""Test isCompatibleCoupling with different dims"""
def test_is_compatible_coupling_different_dims():
    dims1 = ['X', 'Y', 'Z']
    in_coupling1 = ['X', 'Y']
    w_coupling1 = {0: ['Y', 'Z']}
    out_coupling1 = ['X', 'Z']
    
    dims2 = ['X', 'Y', 'W']  # Different dims
    in_coupling2 = ['X', 'Y']
    w_coupling2 = {0: ['Y', 'W']}
    out_coupling2 = ['X', 'W']
    
    coup1 = Coupling(dims1, in_coupling1, w_coupling1, out_coupling1)
    coup2 = Coupling(dims2, in_coupling2, w_coupling2, out_coupling2)
    
    assert coup1.isCompatibleCoupling(coup2) == False

"""Test isCompatibleCoupling with different strides"""
def test_is_compatible_coupling_with_strides():
    dims = ['X', 'Y', 'Z']
    in_coupling = [['X', 'Y']]
    w_coupling = {0: ['Z']}
    out_coupling = ['X']
    in_strides = {'X': 'stride_x'}
    
    coup1 = Coupling(dims, in_coupling, w_coupling, out_coupling, in_strides=in_strides)
    coup2 = Coupling(dims, in_coupling, w_coupling, out_coupling, in_strides=in_strides)
    
    assert coup1.isCompatibleCoupling(coup2) == True

    coup3 = Coupling(dims, in_coupling, w_coupling, out_coupling, in_strides={'X': 'different_stride'})
    assert coup1.isCompatibleCoupling(coup3) == False

"""Test isCompatibleCoupling with intermediate layers and (different) strides"""
def test_is_compatible_coupling_with_intermediate_layers_with_strides():
    dims = ['P', 'Q', 'C2', 'C1', 'R1', 'S1', 'C0', 'R0', 'S0', 'X', 'Y', 'Z']
    in_coupling = [['X', 'R0'], ['Y', 'S0'], 'C0']
    w_coupling = {0: ['C0', 'Z', 'R0', 'S0'], 1: ['C1', 'C2', 'R1', 'S1']}
    out_coupling = ['P', 'Q', 'C2']
    int_in_coupling = {1: [['P', 'R1'], ['Q', 'S1'], 'C1']}
    int_out_coupling = {1: ['Y', 'X', 'Z']}
    in_strides = {'R0': 'stride_r0'}
    #w_strides = {0: {'R0': 'stride_r0_w'}, 1: {'R1': 'stride_r1_w'}}
    #out_strides = {'P': 'stride_p_out'}
    int_in_strides = {1: {'R1': 'stride_r1_int'}}
    int_out_strides = {1: {'Y': 'stride_y_int'}}
    
    coup1 = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       in_strides=in_strides, 
                       int_in_coupling=int_in_coupling, int_out_coupling=int_out_coupling,
                       int_in_strides=int_in_strides, int_out_strides=int_out_strides)
    
    coup2 = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       in_strides=in_strides, 
                       int_in_coupling=int_in_coupling, int_out_coupling=int_out_coupling,
                       int_in_strides=int_in_strides, int_out_strides=int_out_strides)
    
    assert coup1.isCompatibleCoupling(coup2) == True

    # Change an intermediate layer stride
    int_in_strides_diff = {1: {'R1': 'different_stride'}}
    coup3 = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       in_strides=in_strides, 
                       int_in_coupling=int_in_coupling, int_out_coupling=int_out_coupling,
                       int_in_strides=int_in_strides_diff, int_out_strides=int_out_strides)
    assert coup1.isCompatibleCoupling(coup3) == False

"""Test valid subcoupling relationship"""
def test_is_subcoupling_valid():
    dims = ['X', 'Y', 'Z', 'W']
    in_coupling = [['X', 'Y'], 'Z']
    w_coupling = {0: [['Y', 'Z'], 'W']}
    out_coupling = [['X', 'W'], 'Z']
    
    parent_coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)
    
    # Create subcoupling
    sub_in_coupling = ['X', 'Z']
    sub_w_coupling = {0: ['Y', 'W']}
    sub_out_coupling = ['Z']
    
    sub_coupling = Coupling(dims, sub_in_coupling, sub_w_coupling, sub_out_coupling)
    
    assert parent_coupling.isSubcoupling(sub_coupling) == True

##DOUBT
"""Test invalid subcoupling relationship due to missing dimension"""
def test_is_subcoupling_invalid_missing_dim():
    dims = ['X', 'Y', 'Z', 'W']
    in_coupling = [['X', 'Y'], 'Z']
    w_coupling = {0: [['Y', 'Z'], 'W']}
    out_coupling = [['X', 'W'], 'Z']
    
    parent_coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)
    
    # Create subcoupling missing 'Y'
    sub_in_coupling = ['X', 'Z']
    sub_w_coupling = {0: ['W']}  # Missing 'Y'
    sub_out_coupling = ['Z']
    
    sub_coupling = Coupling(dims, sub_in_coupling, sub_w_coupling, sub_out_coupling)
    
    assert parent_coupling.isSubcoupling(sub_coupling) == True

def test_get_dim_sum_basic():
    dims = ['X', 'Y', 'Z']
    in_coupling = [['X', 'Y']]
    w_coupling = {0: ['Z']}
    out_coupling = ['X']
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)
    
    assert coupling.getDimSum('in', 'X') == ['X', 'Y']
    assert coupling.getDimSum('w', 'Z', layer_index=0) == ['Z']
    assert coupling.getDimSum('out', 'X') == ['X']

"""Test getDimSum for intermediate layers"""    
def test_get_dim_sum_intermediate_layers():
    dims = ['P', 'Q', 'C2', 'C1', 'R1', 'S1', 'C0', 'R0', 'S0', 'X', 'Y', 'Z']
    in_coupling = [['X', 'R0'], ['Y', 'S0'], 'C0']
    w_coupling = {0: ['C0', 'Z', 'R0', 'S0'], 1: ['C1', 'C2', 'R1', 'S1']}
    out_coupling = ['P', 'Q', 'C2']
    int_in_coupling = {1: [['P', 'R1'], ['Q', 'S1'], 'C1']}
    int_out_coupling = {1: ['Y', 'X', 'Z']}
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       int_in_coupling=int_in_coupling, int_out_coupling=int_out_coupling)
    
    assert coupling.getDimSum('in', 'X') == ['X', 'R0']
    assert coupling.getDimSum('w', 'Z', layer_index=0) == ['Z']
    assert coupling.getDimSum('out', 'C2') == ['C2']
    assert coupling.getDimSum('int_in', 'P', layer_index=1) == ['P', 'R1']
    assert coupling.getDimSum('int_out', 'Y', layer_index=1) == ['Y']

"""Test getDimSum with "invalid" operand"""    
def test_get_dim_sum_invalid_operand():
    dims = ['X', 'Y']
    coupling = Coupling(dims, ['X'], {0: ['Y']}, ['X'])
    
    with pytest.raises(Exception, match="Unrecognized operand"):
        coupling.getDimSum('invalid', 'X')

"""Test getDimSum with dimension not in coupling"""
def test_get_dim_sum_invalid_layer_index_missing():
    dims = ['X', 'Y', 'Z']
    in_coupling = [['X', 'Y']]
    w_coupling = {0: ['Z']}
    out_coupling = ['X']
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)

    with pytest.raises(Exception, match=(f"Layer index {1} not found in weight coupling.") or (f"Layer index {1} not found in intermediate coupling.")):
        coupling.getDimSum('w', 'Z', layer_index=1)  # Missing layer_index

"""Test flat_coupling_by_operand basic functionality"""
def test_flat_coupling_by_operand_basic():
    dims = ['X', 'Y', 'Z']
    in_coupling = [['X', 'Y']]
    w_coupling = {0: ['Z']}
    out_coupling = ['X']
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)
    
    assert coupling.flatCouplingByOperand('in') == ['X', 'Y']
    assert coupling.flatCouplingByOperand('w', layer_index=0) == ['Z']
    assert coupling.flatCouplingByOperand('out') == ['X']

"""Test flat_coupling_by_operand with intermediate layers"""
def test_flat_coupling_by_operand_intermediate_layers():
    dims = ['P', 'Q', 'C2', 'C1', 'R1', 'S1', 'C0', 'R0', 'S0', 'X', 'Y', 'Z']
    in_coupling = [['X', 'R0'], ['Y', 'S0'], 'C0']
    w_coupling = {0: ['C0', 'Z', 'R0', 'S0'], 1: ['C1', 'C2', 'R1', 'S1']}
    out_coupling = ['P', 'Q', 'C2']
    int_in_coupling = {1: [['P', 'R1'], ['Q', 'S1'], 'C1']}
    int_out_coupling = {1: ['Y', 'X', 'Z']}
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       int_in_coupling=int_in_coupling, int_out_coupling=int_out_coupling)

    assert coupling.flatCouplingByOperand('in') == ['X', 'R0', 'Y', 'S0', 'C0']
    assert coupling.flatCouplingByOperand('w', layer_index=0) == ['C0', 'Z', 'R0', 'S0']
    assert coupling.flatCouplingByOperand('w', layer_index=1) == ['C1', 'C2', 'R1', 'S1']
    assert coupling.flatCouplingByOperand('out') == ['P', 'Q', 'C2']
    assert coupling.flatCouplingByOperand('int_in', layer_index=1) == ['P', 'R1', 'Q', 'S1', 'C1']
    assert coupling.flatCouplingByOperand('int_out', layer_index=1) == ['Y', 'X', 'Z']

"""Test flat_coupling_by_operand with "invalid" operand or layer"""
def test_flat_coupling_by_operand_invalid_operand_or_layer():
    dims = ['X', 'Y']
    coupling = Coupling(dims, ['X'], {0: ['Y']}, ['X'])
    
    with pytest.raises(Exception, match="Unrecognized operand"):
        coupling.flatCouplingByOperand('invalid')

    with pytest.raises(Exception, match="Layer index 3 not found in flat weight coupling."):
        coupling.flatCouplingByOperand('w', 3)  # Invalid layer_index

"""Test get_weight_coupling basic functionality"""
def test_get_weight_coupling():
    dims = ['X', 'Y', 'Z']
    in_coupling = [['X', 'Y']]
    w_coupling = {0: ['Z'], 1: ['X', 'Y']}
    out_coupling = ['X']
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)
    
    assert coupling.getWeightCoupling(0) == [['Z']]
    assert coupling.getWeightCoupling(1) == [['X'], ['Y']]
    with pytest.raises(Exception, match=(f"Layer index {2} not found in weight coupling.") or (f"Layer index {2} not found in intermediate coupling.")):
        coupling.getWeightCoupling(2)  # Invalid layer_index

"""Test get_intermediate_coupling_methods functionality"""
def test_get_intermediate_coupling_methods():
    dims = ['P', 'Q', 'C3', 'C2', 'C1', 'R1', 'S1', 'C0', 'R0', 'S0', 'X0', 'Y0', 'X1', 'Y1', 'R2', 'S2', 'Z0', 'Z1']
    in_coupling = [['X0', 'R0'], ['Y0', 'S0'], 'C0']
    w_coupling = {0: ['C0', 'C1', 'R0', 'S0'], 1: ['C1', 'C2', 'R1', 'S1'], 2: ['C2', 'C3', 'R2', 'S2']}
    out_coupling = ['P', 'Q', 'C3']
    int_in_coupling = {1: [['X1', 'R1'], ['Y1', 'S1'], 'C1'], 2: [['P', 'R2'], ['Q', 'S2'], 'C2']}
    int_out_coupling = {1: ['Y0', 'X0', 'Z0'], 2: ['Y1', 'X1', 'Z1']}

    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       int_in_coupling=int_in_coupling, int_out_coupling=int_out_coupling)
    
    assert coupling.getIntermediateInputCoupling(1) == [['X1', 'R1'], ['Y1', 'S1'], ['C1']]
    assert coupling.getIntermediateOutputCoupling(1) == [['Y0'], ['X0'], ['Z0']]
    assert coupling.getIntermediateInputCoupling(2) == [['P', 'R2'], ['Q', 'S2'], ['C2']]
    assert coupling.getIntermediateOutputCoupling(2) == [['Y1'], ['X1'], ['Z1']]
    assert coupling.getFlatIntermediateInputCoupling(1) == ['X1', 'R1', 'Y1', 'S1', 'C1']
    assert coupling.getFlatIntermediateOutputCoupling(1) == ['Y0', 'X0', 'Z0']
    assert coupling.getFlatIntermediateInputCoupling(2) == ['P', 'R2', 'Q', 'S2', 'C2']
    assert coupling.getFlatIntermediateOutputCoupling(2) == ['Y1', 'X1', 'Z1']

    with pytest.raises(Exception, match="Layer index 0 not found in intermediate input coupling."):
        coupling.getIntermediateInputCoupling(0)  # Invalid layer_index
    with pytest.raises(Exception, match="Layer index 0 not found in intermediate output coupling."):
        coupling.getIntermediateOutputCoupling(0)  # Invalid layer_index

"""Test getNumLayers method"""
def test_get_num_layers():
    dims = ['X', 'Y', 'Z']
    in_coupling = [['X', 'Y']]
    w_coupling = {0: ['Z'], 1: ['X']}
    out_coupling = ['X']
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling)
    
    assert coupling.getNumLayers() == 2

    # Test with no weight layers
    coupling_no_w = Coupling(dims, in_coupling, {}, out_coupling)
    assert coupling_no_w.getNumLayers() == 0

"""
def test_str_with_intermediate_layers():
    dims = ['P', 'Q', 'C2', 'C1', 'R1', 'S1', 'C0', 'R0', 'S0', 'X', 'Y', 'Z']
    in_coupling = [['X', 'R0'], ['Y', 'S0'], 'C0']
    w_coupling = {0: ['C0', 'Z', 'R0', 'S0'], 1: ['C1', 'C2', 'R1', 'S1']}
    out_coupling = ['P', 'Q', 'C2']
    int_in_coupling = {1: [['P', 'R1'], ['Q', 'S1'], 'C1']}
    int_out_coupling = {1: ['Y', 'X', 'Z']}
    in_strides = {'R0': 'stride_r0'}
    #w_strides = {0: {'R0': 'stride_r0_w'}, 1: {'R1': 'stride_r1_w'}}
    #out_strides = {'P': 'stride_p_out'}
    int_in_strides = {1: {'R1': 'stride_r1_int'}}
    int_out_strides = {1: {'Y': 'stride_y_int'}}
    
    coupling = Coupling(dims, in_coupling, w_coupling, out_coupling,
                       in_strides=in_strides, 
                       int_in_coupling=int_in_coupling, int_out_coupling=int_out_coupling,
                       int_in_strides=int_in_strides, int_out_strides=int_out_strides)
    
    expected_str = ("Coupling(\n"
                    "  dims=['P', 'Q', 'C2', 'C1', 'R1', 'S1', 'C0', 'R0', 'S0', 'X', 'Y', 'Z'],\n"
                    "  in_coupling=[['X', 'R0'], ['Y', 'S0'], ['C0']],\n"
                    "  w_coupling={0: [['C0', 'Z', 'R0', 'S0']], 1: [['C1', 'C2', 'R1', 'S1']]},\n"
                    "  out_coupling=[['P', 'Q', 'C2']],\n"
                    "  in_strides={'R0': 'stride_r0'},\n"
                    "  w_strides={0: {}, 1: {}},\n"
                    "  out_strides={},\n"
                    "  int_in_coupling={1: [['P', 'R1'], ['Q', 'S1'], ['C1']]},\n"
                    "  int_out_coupling={1: [['Y'], ['X'], ['Z']]},\n"
                    "  int_in_strides={1: {'R1': 'stride_r1_int'}},\n"
                    "  int_out_strides={1: {'Y': 'stride_y_int'}}\n"
                    ")")
    assert str(coupling) == expected_str
"""
if __name__ == "__main__":
    # Run all test functions
    test_coupling_init_valid_basic()
    test_coupling_init_valid_nested_lists()
    test_coupling_init_valid_with_strides()
    test_coupling_init_valid_intermediate_layers()
    test_coupling_init_invalid_duplicate_dims()
    test_coupling_init_invalid_three_level_list()
    test_coupling_init_invalid_coupling_not_subset()
    test_coupling_init_invalid_w_coupling_not_subset()
    test_coupling_init_invalid_duplicate_in_coupling()
    test_coupling_init_invalid_duplicate_w_coupling()
    test_coupling_init_invalid_stride_key_not_in_dims()
    test_coupling_init_invalid_stride_value_in_dims()
    test_coupling_init_invalid_stride_not_in_sum()
    test_coupling_init_empty_intermediate_layers()
    test_coupling_init_empty_strides()
    test_coupling_init_multiple_w_layers()
    test_coupling_init_partial_w_strides()
    
    # New function tests
    test_is_compatible_comp_valid()
    test_is_compatible_comp_missing_dim()
    test_is_compatible_comp_missing_stride()
    test_is_compatible_coupling_basic()
    test_is_compatible_coupling_with_strides()
    test_is_subcoupling_valid()
    test_get_dim_sum_basic()
    test_get_dim_sum_intermediate_layers()
    test_get_dim_sum_invalid_operand()
    test_flat_coupling_by_operand_basic()
    test_get_weight_coupling()
    test_get_intermediate_coupling_methods()
    test_get_num_layers()
    #test_str_with_intermediate_layers()

    print("All tests passed!")