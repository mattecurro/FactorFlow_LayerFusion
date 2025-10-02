# filepath: /home/matteo/Desktop/FactorFlow_LayerFusion/test_output.py

import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict

class LoopNestSimulator:
	"""Simulates the execution of a loop nest to verify MOPs calculations"""
	
	def __init__(self):
		self.computation_shape = {
			'P': 6, 'Q': 6, 'K': 2, 'R0': 3, 'S0': 3, 
			'C1': 2, 'R1': 3, 'S1': 3, 'M': 3
		}
		
		# Mapping from the output
		self.mapping = {
			'DRAM': ['P', 'Q', 'K'],
			'GlobalBuffer': ['R1', 'P'], 
			'SACols': ['Q'],
			'SARows': ['M'],
			'InRegister': ['C1', 'S1'],
			'WRegister': ['R0'],
			'OutRegister': ['S0'],
			'Compute': []
		}
		
		# Factors assigned to each level
		self.level_factors = {
			'DRAM': {'P': 2, 'Q': 3, 'K': 2},
			'GlobalBuffer': {'R1': 3, 'P': 3},
			'SACols': {'Q': 2},
			'SARows': {'M': 3},
			'InRegister': {'C1': 2, 'S1': 3},
			'WRegister': {'R0': 3},
			'OutRegister': {'S0': 3},
			'Compute': {}
		}
		
		# Spatial instances
		self.spatial_instances = {
			'SACols': 14,  # mesh size
			'SARows': 12   # mesh size
		}
		
		# Track memory operations
		self.mops = defaultdict(lambda: {
			'in_reads_l0': 0, 'in_reads_l1': 0,
			'w_reads_l0': 0, 'w_reads_l1': 0,
			'out_reads': 0, 'out_drains': 0,
			'out_writes': 0, 'out_updates': 0, 'out_fills': 0,
			'in_writes': 0, 'w_writes': 0
		})
		
		# Bypassing - from the architecture definition
		self.bypasses = {
			'InRegister': ['w', 'out'],
			'WRegister': ['in', 'out'], 
			'OutRegister': ['in', 'w']
		}

	def get_loop_order(self) -> List[Tuple[str, str, int]]:
		"""Get the loop order from outermost to innermost"""
		loop_order = []
		
		# Build loop nest from mapping
		for level_name in ['DRAM', 'GlobalBuffer', 'SACols', 'SARows', 
						  'InRegister', 'WRegister', 'OutRegister', 'Compute']:
			if level_name in self.level_factors:
				for dim, factor in self.level_factors[level_name].items():
					loop_order.append((level_name, dim, factor))
		
		return loop_order

	def simulate_memory_accesses(self):
		"""Simulate the actual loop nest execution and track memory accesses"""
		
		loop_order = self.get_loop_order()
		
		# Calculate spatial parallelism
		spatial_parallel = 1
		for level in ['SACols', 'SARows']:
			if level in self.level_factors:
				for dim, factor in self.level_factors[level].items():
					spatial_parallel *= factor
		
		print(f"Spatial parallelism: {spatial_parallel}")
		print(f"Loop order: {loop_order}")
		
		# Simulate nested loops
		self._simulate_loops(loop_order, 0, {}, spatial_parallel)
		
		return self.mops

	def _simulate_loops(self, loop_order: List[Tuple[str, str, int]], 
					   depth: int, current_coords: Dict[str, int], 
					   spatial_instances: int):
		"""Recursively simulate nested loops"""
		
		if depth >= len(loop_order):
			# At the innermost level - perform computation
			self._perform_computation(current_coords, spatial_instances)
			return
			
		level_name, dim, factor = loop_order[depth]
		
		# For each iteration of this loop
		for i in range(factor):
			new_coords = current_coords.copy()
			new_coords[dim] = i
			# advance to the next loop
			self._simulate_loops(loop_order, depth + 1, new_coords, spatial_instances)

	def _perform_computation(self, current_coords, spatial_instances: int):
        	# TEMP: just count computations so the script runs.
	        self.mops["Compute"]["ops"] = self.mops["Compute"].get("ops", 0) + spatial_instances

if __name__ == "__main__":
    sim = LoopNestSimulator()
    stats = sim.simulate_memory_accesses()
    print("Done. Example counters:", {k: v for k, v in stats.items() if k == "Compute"})