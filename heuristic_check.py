"""
Convolution Loop Scheduling Validator
Validates loop orderings and tiling configurations for convolution operations
based on dependency heuristics to prevent reading from Int before it's written.
"""

import random
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import json

# Feature map dimensions
FEATURE_MAP_Q = 40  # Output dimension
FEATURE_MAP_S = 3   # Filter dimension
FEATURE_MAP_X = 42  # Intermediate dimension (S+Q-1)
FEATURE_MAP_R = 1   # Input filter dimension

@dataclass
class ValidationResult:
    """Stores validation result for a single X loop level"""
    level_name: str
    q_iterations: int
    s_iterations: int
    x_iterations: int
    is_valid: bool
    calculation_str: str

class ConvolutionScheduleValidator:
    """Main validator class for convolution loop scheduling"""

    def __init__(self):
        self.feature_q = FEATURE_MAP_Q
        self.feature_s = FEATURE_MAP_S
        self.feature_x = FEATURE_MAP_X
        self.feature_r = FEATURE_MAP_R

    def parse_loop_name(self, name: str) -> Tuple[str, int]:
        """Parse 'q1' -> ('q', 1), 's2' -> ('s', 2)"""
        base = ''.join(c for c in name if c.isalpha())
        level = ''.join(c for c in name if c.isdigit())
        level = int(level) if level else 1
        return base, level

    def validate_heuristic(self, tiling: Dict[str, int], loop_order: List[str]) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate the heuristic for given tiling and loop order.

        Heuristic: For each X loop level:
        Q_iterations + S_iterations - 1 <= X_iterations

        Returns: (is_valid_overall, list_of_validation_results)
        """
        results = []

        # Check if the dimension in loop_order before the first 'x' are with iteration equal to 1
        for i, loop in enumerate(reversed(loop_order)):
            base, level = self.parse_loop_name(loop)
            if base != 'x':
                # base_iter is the number of iterations of the position in the loop order correspondent to base
                iter_key = f'iterations_{loop}'
                if iter_key in tiling:
                    if tiling[iter_key] != 1:
                        return False, [ValidationResult(
                        level_name="N/A",
                        q_iterations=0,
                        s_iterations=0,
                        x_iterations=0,
                        is_valid=False,
                        calculation_str="First loop with an iteration > 1 must be an 'x' loop"
                        )]
            else: break

        # Extract X loop positions and identify their levels
        x_positions = []
        for i, loop in enumerate(loop_order):
            #print(f"i: {i}, loop: {loop}")
            base, level = self.parse_loop_name(loop)
            if base == 'x':
                x_positions.append((i, loop, level))


        #print(f"x_position: {x_positions}")
        # Sort by level (innermost first for processing)
        x_positions.sort(key=lambda x: -x[2])
        overall_valid = True

        for x_pos_idx, (pos, x_loop, x_level) in enumerate(x_positions):
            # Find the next outer X loop position (if exists)
            next_outer_x_pos = None
            if x_pos_idx < len(x_positions) - 1:
                next_outer_x_pos = x_positions[x_pos_idx + 1][0]
            else:
                next_outer_x_pos = -1  # Before all loops

            # Calculate Q_iterations: product of Q loops between current X and next outer X
            q_iterations = 1
            q_loops_used = []

            # Calculate S_iterations: product of S loops between current X and next outer X
            s_iterations = 1
            s_loops_used = []

            # Look at loops between next_outer_x_pos and current pos
            for i in range(next_outer_x_pos + 1, len(loop_order)):
                loop = loop_order[i]
                base, level = self.parse_loop_name(loop)

                if base == 'q':
                    iter_key = f'iterations_{loop}'
                    if iter_key in tiling:
                        q_iterations *= tiling[iter_key]
                        q_loops_used.append(f"{loop}={tiling[iter_key]}")
                elif base == 's':
                    iter_key = f'iterations_{loop}'
                    if iter_key in tiling:
                        s_iterations *= tiling[iter_key]
                        s_loops_used.append(f"{loop}={tiling[iter_key]}")


            # Calculate X_iterations: current X and all inner X loops
            x_iterations = tiling[f'iterations_{x_loop}']
            x_loops_used = [f"{x_loop}={x_iterations}"]

            # Add inner X loops
            for inner_x_pos, inner_x_loop, _ in x_positions:
                if inner_x_pos > pos:
                    inner_x_iter = tiling[f'iterations_{inner_x_loop}']
                    x_iterations *= inner_x_iter
                    x_loops_used.append(f"{inner_x_loop}={inner_x_iter}")

            # Special case: outermost X level should check against total dimensions
            if x_pos_idx == len(x_positions) - 1:
                # For outermost level, check total coverage
                total_q = 1
                total_s = 1
                total_x = 1

                for loop in loop_order:
                    base, _ = self.parse_loop_name(loop)
                    iter_key = f'iterations_{loop}'
                    if iter_key in tiling:
                        if base == 'q':
                            total_q *= tiling[iter_key]
                        elif base == 's':
                            total_s *= tiling[iter_key]
                        elif base == 'x':
                            total_x *= tiling[iter_key]

                q_iterations = total_q
                s_iterations = total_s
                x_iterations = total_x

            # Check the heuristic
            lhs = q_iterations + s_iterations - 1
            is_valid = lhs <= x_iterations

            # Create calculation string
            if q_loops_used or s_loops_used:
                q_calc = " * ".join(q_loops_used) if q_loops_used else "1"
                s_calc = " * ".join(s_loops_used) if s_loops_used else "1"
            else:
                q_calc = str(q_iterations)
                s_calc = str(s_iterations)

            x_calc = " * ".join(x_loops_used)

            calc_str = f"Q_iter={q_iterations} ({q_calc}), S_iter={s_iterations} ({s_calc}), X_iter={x_iterations} ({x_calc}) → {lhs} <= {x_iterations}? {is_valid}"

            result = ValidationResult(
                level_name=x_loop,
                q_iterations=q_iterations,
                s_iterations=s_iterations,
                x_iterations=x_iterations,
                is_valid=is_valid,
                calculation_str=calc_str
            )

            results.append(result)
            if not is_valid:
                overall_valid = False

        # Reverse to show from innermost to outermost
        results.reverse()

        return overall_valid, results


    def print_validation_report(self, config_name: str, tiling: Dict[str, int],
                               loop_order: List[str], expected_result: str = None):
        """Print detailed validation report for a configuration"""
        print(f"\n{'='*60}")
        print(f"Testing Configuration: {config_name}")
        print(f"{'='*60}")
        print(f"Loop Order: {loop_order}")
        print(f"Tiling: {json.dumps(tiling, indent=2)}")

        is_valid, results = self.validate_heuristic(tiling, loop_order)

        print(f"\nHeuristic Validation:")
        print("-" * 40)

        for result in results:
            status = "✓" if result.is_valid else "❌"
            print(f"- {result.level_name} level: {result.calculation_str} {status}")

        overall_status = "VALID" if is_valid else "INVALID"
        failed_levels = [r.level_name for r in results if not r.is_valid]

        if failed_levels:
            print(f"\nOverall Result: {overall_status} (fails at {', '.join(failed_levels)} level(s))")
        else:
            print(f"\nOverall Result: {overall_status} ✓")

        if expected_result:
            matches = (expected_result == "VALID" and is_valid) or (expected_result == "FAIL" and not is_valid)
            match_str = "✓ MATCHES EXPECTED" if matches else "❌ DOES NOT MATCH EXPECTED"
            print(f"Expected: {expected_result} - {match_str}")

        return is_valid

def main():
    """Main entry point"""
    print("Convolution Loop Scheduling Validator")
    print("=====================================")

    validator = ConvolutionScheduleValidator()

    # Example of custom configuration
    custom_tiling = {
        'iterations_q1': 5, 'iterations_x1': 6, 'iterations_s1': 1,
        'iterations_q2': 8, 'iterations_s2': 3, 'iterations_x2': 7,
        'iterations_q3': 1, 'iterations_s3': 1, 'iterations_x3': 1
    }

    custom_order = ['x3', 'q2', 'q3', 's1', 's2', 'x2', 'q1', 'x1', 's3']

    # Validate with heuristic
    is_valid, _ = validator.validate_heuristic(custom_tiling, custom_order)
    print(f"\nCustom Configuration Validation Result: {'VALID' if is_valid else 'INVALID'}")

if __name__ == "__main__":
    main()
