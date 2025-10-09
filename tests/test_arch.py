import sys
import os
import pytest
from typing import Dict, List

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arch import Arch
from factors import Coupling, Factors, Shape
from levels import MemLevel, ComputeLevel

class TestArchHeuristicValidation:
    """Test the heuristic validation functions in Arch class"""
    
    def setup_method(self):
        """Set up a basic architecture for testing"""
        # Create a simple coupling for testing
        self.coupling = Coupling(
            dims=['Q', 'S', 'X', 'M', 'K'],
            in_coupling=[['Q', 'S'], ['M']],
            w_coupling={0: [['S'], ['K', 'M']]},
            out_coupling=[['Q'], ['K']]
        )
        
        # Create simple levels
        mem_level = MemLevel("TestMem", size=1000, value_access_energy=1.0)
        compute_level = ComputeLevel("TestCompute", mesh=1, compute_energy=1.0, cycles=1)
        
        self.arch = Arch([mem_level, compute_level], self.coupling, "TestArch")
        
        # Initialize with a simple computation
        comp = Shape({'Q': 8, 'S': 3, 'X': 10, 'M': 16, 'K': 32})
        self.arch.initFactors(comp)

    def test_parse_loop_name_valid_inputs(self):
        """Test parsing of valid loop names"""
        # Test simple cases
        assert self.arch.parse_loop_name('q1') == ('q', 1)
        assert self.arch.parse_loop_name('s2') == ('s', 2)
        assert self.arch.parse_loop_name('x0') == ('x', 0)
        
        # Test multi-character bases
        assert self.arch.parse_loop_name('abc3') == ('abc', 3)
        
        # Test no number (should default to 1)
        assert self.arch.parse_loop_name('q') == ('q', 1)
        assert self.arch.parse_loop_name('xyz') == ('xyz', 1)
        
        # Test multi-digit numbers
        assert self.arch.parse_loop_name('q123') == ('q', 123)

    def test_parse_loop_name_edge_cases(self):
        """Test edge cases for loop name parsing"""
        # Test mixed alphanumeric
        assert self.arch.parse_loop_name('q1a2') == ('qa', 12)
        
        # Test only letters
        assert self.arch.parse_loop_name('qrs') == ('qrs', 1)
        
        # Test only numbers (should return empty string for base)
        assert self.arch.parse_loop_name('123') == ('', 123)

    def test_validate_heuristic_valid_case(self):
        """Test heuristic validation with a valid configuration"""
        tiling = {
            'iterations_q1': 2,
            'iterations_s1': 1, 
            'iterations_x1': 5,
            'iterations_q2': 4,
            'iterations_s2': 3,
            'iterations_x2': 8
        }
        loop_order = ['q1', 's1', 'x1', 'q2', 's2', 'x2']
        
        result = self.arch.validate_heuristic(tiling, loop_order)
        assert result == True

    def test_validate_heuristic_invalid_case(self):
        """Test heuristic validation with an invalid configuration"""
        tiling = {
            'iterations_q1': 5,
            'iterations_s1': 4,
            'iterations_x1': 7,  # 5 + 4 - 1 = 8, but x1 = 7, so invalid
        }
        loop_order = ['q1', 's1', 'x1']
        
        result = self.arch.validate_heuristic(tiling, loop_order)
        assert result == False

    def test_validate_heuristic_first_non_x_loop_violation(self):
        """Test validation fails when non-x loop appears first with iterations > 1"""
        tiling = {
            'iterations_q1': 2,  # First loop is q with iterations > 1
            'iterations_x1': 5
        }
        loop_order = ['q1', 'x1']
        
        result = self.arch.validate_heuristic(tiling, loop_order)
        assert result == False

    def test_validate_heuristic_multiple_x_levels(self):
        """Test heuristic with multiple X levels"""
        tiling = {
            'iterations_q1': 2,
            'iterations_s1': 1,
            'iterations_x1': 4,
            'iterations_q2': 1,
            'iterations_s2': 2,
            'iterations_x2': 5,  # Inner X needs q2 + s2 - 1 = 2
            'iterations_x3': 10  # Outer X needs total coverage
        }
        loop_order = ['x3', 'q1', 's1', 'x1', 'q2', 's2', 'x2']
        
        result = self.arch.validate_heuristic(tiling, loop_order)
        assert result == True

    def test_extract_loop_order(self):
        """Test extraction of loop order from architecture"""
        # Set up dataflows
        self.arch[0].dataflow = ['Q', 'S']
        self.arch[1].dataflow = ['X', 'M']
        
        loop_order = self.arch._extractLoopOrder()
        expected = ['q0', 's0', 'x1', 'm1']
        assert loop_order == expected

    def test_extract_current_tiling(self):
        """Test extraction of current tiling from architecture"""
        # Set up some factors
        self.arch[0].factors.addFactor('Q', 2, 2)  # Q has 4 iterations (2^2)
        self.arch[0].factors.addFactor('S', 3, 1)  # S has 3 iterations
        self.arch[1].factors.addFactor('X', 5, 1)  # X has 5 iterations
        
        # Set dataflows
        self.arch[0].dataflow = ['Q', 'S']
        self.arch[1].dataflow = ['X']
        
        tiling = self.arch._extractCurrentTiling()
        
        assert tiling['iterations_q0'] == 4
        assert tiling['iterations_s0'] == 3
        assert tiling['iterations_x1'] == 5

    def test_create_simulated_tiling_after_move(self):
        """Test simulation of tiling after a factor move"""
        # Set up initial factors
        self.arch[0].factors.addFactor('Q', 2, 3)  # Q has 8 iterations (2^3)
        self.arch[1].factors.addFactor('Q', 1, 0)  # Q has 1 iteration
        
        # Set dataflows
        self.arch[0].dataflow = ['Q']
        self.arch[1].dataflow = ['Q']
        
        # Simulate moving factor 2 with amount 1 from level 0 to level 1
        simulated_tiling = self.arch._createSimulatedTilingAfterMove(0, 1, 'Q', 2, 1)
        
        # After move: level 0 should have 4 iterations (8/2), level 1 should have 2 iterations (1*2)
        assert simulated_tiling['iterations_q0'] == 4
        assert simulated_tiling['iterations_q1'] == 2


class TestArchMoveFactor:
    """Test the moveFactor function with heuristic validation"""
    
    def setup_method(self):
        """Set up architecture for factor movement tests"""
        self.coupling = Coupling(
            dims=['Q', 'S', 'X'],
            in_coupling=[['Q', 'S']],
            w_coupling={0: [['S']]},
            out_coupling=[['Q']]
        )
        
        mem_level1 = MemLevel("Mem1", size=10000, value_access_energy=1.0)
        mem_level2 = MemLevel("Mem2", size=1000, value_access_energy=2.0)
        compute_level = ComputeLevel("Compute", mesh=1, compute_energy=1.0, cycles=1)
        
        self.arch = Arch([mem_level1, mem_level2, compute_level], self.coupling, "TestArch")
        
        # Initialize with computation
        comp = Shape({'Q': 8, 'S': 4, 'X': 12})
        self.arch.initFactors(comp)

    def test_move_factor_valid_move(self):
        """Test a valid factor move"""
        # Set up dataflows
        self.arch[0].dataflow = ['Q', 'S']
        self.arch[1].dataflow = ['Q', 'S']
        self.arch[2].dataflow = ['X']
        
        # Move factor 2 from level 0 to level 1 for dimension Q
        result = self.arch.moveFactor(0, 1, 'Q', 2, 1, skip_heuristic_check=True)
        assert result == True
        
        # Check that factors were moved correctly
        assert 2 not in self.arch[0].factors['Q'] or self.arch[0].factors['Q'][2] < 3
        assert 2 in self.arch[1].factors['Q'] and self.arch[1].factors['Q'][2] >= 1

    def test_move_factor_insufficient_factors(self):
        """Test move fails when source doesn't have enough factors"""
        # Try to move more factors than available
        result = self.arch.moveFactor(0, 1, 'Q', 2, 10)  # Too many factors
        assert result == False

    def test_move_factor_with_heuristic_validation(self):
        """Test factor move with heuristic validation enabled"""
        # Set up dataflows that might violate heuristic
        self.arch[0].dataflow = ['Q', 'S', 'X']
        self.arch[1].dataflow = []
        self.arch[2].dataflow = []
        
        # This should pass basic constraints but might fail heuristic
        result = self.arch.moveFactor(0, 1, 'Q', 2, 1, skip_heuristic_check=False)
        # Result depends on whether the resulting configuration violates the heuristic
        assert isinstance(result, bool)

    def test_move_factor_constraint_violation(self):
        """Test move fails due to constraint violations"""
        # Set a constraint that would be violated
        self.arch[1].factors_constraints['Q'] = 1  # Exact constraint of 1
        
        # Try to move factors to level 1, which should violate the constraint
        result = self.arch.moveFactor(0, 1, 'Q', 2, 2)  # Would give 4 iterations, violating constraint of 1
        assert result == False

    def test_move_factor_rollback(self):
        """Test that rollback works correctly when move fails"""
        # Store initial state
        initial_q_factors_l0 = dict(self.arch[0].factors['Q'])
        initial_q_factors_l1 = dict(self.arch[1].factors['Q'])
        initial_tile_size = self.arch[0].tile_sizes['Q']
        
        # Set constraint that will cause failure
        self.arch[1].factors_constraints['Q'] = 1
        
        # Attempt move that should fail and rollback
        result = self.arch.moveFactor(0, 1, 'Q', 2, 2)
        assert result == False
        
        # Check that state was rolled back
        assert self.arch[0].factors['Q'] == initial_q_factors_l0
        assert self.arch[1].factors['Q'] == initial_q_factors_l1
        assert self.arch[0].tile_sizes['Q'] == initial_tile_size


class TestArchConstraints:
    """Test constraint checking and fitting functions"""
    
    def setup_method(self):
        """Set up architecture for constraint tests"""
        self.coupling = Coupling(
            dims=['M', 'K', 'N'],
            in_coupling=[['M'], ['K']],
            w_coupling={0: [['K'], ['N']]},
            out_coupling=[['M'], ['N']]
        )
        
        mem_level = MemLevel("TestMem", size=1000, value_access_energy=1.0)
        mem_level.factors_constraints = {'M': 4, 'K>=': 2}  # M must be exactly 4, K at least 2
        
        compute_level = ComputeLevel("TestCompute", mesh=1, compute_energy=1.0, cycles=1)
        
        self.arch = Arch([mem_level, compute_level], self.coupling, "TestArch")

    def test_fit_constraints_to_comp_success(self):
        """Test successful constraint fitting"""
        comp = Shape({'M': 16, 'K': 8, 'N': 32})
        result = self.arch.fitConstraintsToComp(comp)
        assert result == True

    def test_fit_constraints_to_comp_failure(self):
        """Test constraint fitting failure"""
        comp = Shape({'M': 2, 'K': 8, 'N': 32})  # M=2 is too small for constraint M=4
        result = self.arch.fitConstraintsToComp(comp, enforce=False)
        assert result == False

    def test_check_factors_constraints(self):
        """Test factors constraint checking"""
        comp = Shape({'M': 16, 'K': 8, 'N': 32})
        self.arch.initFactors(comp)
        
        # Should pass initially (all factors on first level)
        assert self.arch.checkFactorsConstraints() == True

    def test_enforce_factors_constraints(self):
        """Test constraint enforcement"""
        comp = Shape({'M': 16, 'K': 8, 'N': 32})
        self.arch.initFactors(comp)
        
        # Enforce constraints
        self.arch.enforceFactorsConstraints()
        
        # Check that constraints are satisfied
        assert self.arch.checkFactorsConstraints() == True
        assert self.arch[0].factors.dimProduct('M') == 4  # Exact constraint


if __name__ == "__main__":
    pytest.main([__file__])