from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any
from math import prod

from factors import *
#from prints import *


# fix static typechecking without recursive imports
if TYPE_CHECKING:
    from arch import Arch
 
# Remember: on the present level you store everything for the iterations you have there, while
# stationarity means finding the operands that remaing constant (and don't need to be read again
# to either be used or sent to a level below) during consecutive (thus, innermost) iterations.

"""
NOTE: the hierarchy is read just as the architecture is specified, with
"below" levels being closer to the computation and "above" ones closer
to DRAM or slower memories.


Class with the minimal information characterizing a level's mapping (dataflow, factors, tile_sizes).
"""
class LevelCore:
    # IMPORTANT:
    # Use the entries of "dataflow" to access "factors", so that you only
    # see the factors over which you are actually supposed to iterate!
    dataflow : list[str] # order of the loops | e.g. ['M', 'K', 'N']
    dataflow_per_layer : dict[int, str]
    factors : Factors # iterations done for the dimensions at this level
    tile_sizes : Shape # indicate the size of a tile used in the level BELOW (closer to PEs)
    # NOTE: also, this is the data sent inward at each iterations
    # NOTE: tile sizes are, in other words, the number of elements jumped
    #       by one iteration at this level an any dimension
    # NOTE: another interpretation is that tile sizes are the amount by
    #       which inner levels will move along a dimension
    # NOTE: finally, tile sizes can be seen as the coefficient to be applied
    #       to a level's index along the respective dimension
    # NOTE: tile sizes in fanout levels represent the size of each tile
    #       stored in an instance below them
    # NOTE: tile sizes are updated in "moveFactor"
    
    def __init__(self, dataflow : list[str], factors : Factors, tile_sizes : Shape):
        self.dataflow = dataflow
        self.dataflow_per_layer = {}
        for layer_id in range(self.arch.coupling.getNumLayers()):
            layer_relevant_dims = self.arch.coupling.relevantDimsForLayer(layer_id)
            self.dataflow_per_layer[layer_id] = [
                dim for dim in dataflow if dim in layer_relevant_dims
            ]
        self.tile_sizes = tile_sizes
    
    def __str__(self) -> str:
        return f"dataflow per layer: {self.dataflow_per_layer},dataflow: {self.dataflow}, factors: {self.factors}, tile_sizes: {self.tile_sizes}"

"""
Abstract class representing a level of the accelerator's architecture.
"""
class Level(LevelCore):
    name : str
    arch : Arch
    factors_constraints : dict[str, int]
    area : Optional[float]

    """
    Sets up a pointer back to the whole architecture.
    Ultimates the initialization of the level and validates its attributes.
    """
    def initArch(self, arch : Arch):
        self.arch = arch


    """
    Add "amount" instances of the provided factor to those of
    "dimension" in the current level.
    """
    def addFactor(self, dimension : str, factor : int, amount : int = 1) -> None:
        self.factors.addFactor(dimension, factor, amount)

    """
    Removes "amount" instances of the provided factor from those of
    "dimension" in the current level.
    Return False if the removal failed because the current level does not
    have at least "amount" instances of "factor" along "dimension".
    """
    def removeFactor(self, dimension : str, factor : int, amount : int = 1) -> bool:
        return self.factors.removeFactor(dimension, factor, amount)

    """
    Returns True iif factors present on this level satisfy all of its constraints.
    """ 
    ## aggiungere constraints
    def checkFactorsConstraints(self) -> bool:
        # NOTE: full condition kept for readability:
        #return (all([(dim not in self.factors_constraints or self.factors_constraints[dim] == self.factors.dimProduct(dim)) for dim in self.dataflow]) and
        #        all([((dim + '<=') not in self.factors_constraints or self.factors_constraints[dim + '<='] >= self.factors.dimProduct(dim)) for dim in self.dataflow]) and
        #        all([((dim + '>=') not in self.factors_constraints or self.factors_constraints[dim + '>='] <= self.factors.dimProduct(dim)) for dim in self.dataflow]) and
        #        all([len(self.factors[dim]) == 0 for dim in self.arch.coupling.dims if dim not in self.dataflow]))
        # Extract dimension name correctly for multi-character dimensions
        """
        Separate the dimension name from the constraint type in the keys of factors_constraints.
        'M>=5' -> ('M', '>=')
        """
        def extract_dimension_and_constraint_type(constraint_key):
            if constraint_key.endswith('>='):
                return constraint_key[:-2], '>='
            elif constraint_key.endswith('<='):
                return constraint_key[:-2], '<='
            else:
                return constraint_key, '=='
        
        """
        Validates the dimension againtst the constraints specified in factors_constraints.
        """
        for constraint_key, constraint_value in self.factors_constraints.items():
            dim_name, constraint_type = extract_dimension_and_constraint_type(constraint_key)
            
            if constraint_type == '==':
                if constraint_value != self.factors.dimProduct(dim_name):
                    return False
            elif constraint_type == '>=':
                if constraint_value > self.factors.dimProduct(dim_name):
                    return False
            elif constraint_type == '<=':
                if constraint_value < self.factors.dimProduct(dim_name):
                    return False
        
        # Check that dimensions not in dataflow have 0 iterations
        return all(len(self.factors[dim]) == 0 for dim in self.arch.coupling.dims if dim not in self.dataflow)

    """
    Returns a string describing the current violation of constraints, if any.
    """
    def logConstraintsViolation(self) -> str:
        if not self.checkFactorsConstraints():
            # Helper function to extract dimension names correctly
            def extract_dimension_and_constraint_type(constraint_key):
                if constraint_key.endswith('>='):
                    return constraint_key[:-2], '>='
                elif constraint_key.endswith('<='):
                    return constraint_key[:-2], '<='
                else:
                    return constraint_key, '=='
            
            violations = []
            
            # Check exact constraints (==)
            for constraint_key, constraint_value in self.factors_constraints.items():
                dim_name, constraint_type = extract_dimension_and_constraint_type(constraint_key)
                
                if constraint_type == '==' and dim_name in self.dataflow:
                    if constraint_value != self.factors.dimProduct(dim_name):
                        violations.append(f"constrained {dim_name} == {constraint_value} VS obtained {dim_name}: {self.factors.dimProduct(dim_name)}")
            
            # Check upper bound constraints (<=)
            for constraint_key, constraint_value in self.factors_constraints.items():
                dim_name, constraint_type = extract_dimension_and_constraint_type(constraint_key)
                
                if constraint_type == '<=' and dim_name in self.dataflow:
                    if constraint_value < self.factors.dimProduct(dim_name):
                        violations.append(f"constrained {dim_name} <= {constraint_value} VS obtained {dim_name}: {self.factors.dimProduct(dim_name)}")
            
            # Check lower bound constraints (>=)
            for constraint_key, constraint_value in self.factors_constraints.items():
                dim_name, constraint_type = extract_dimension_and_constraint_type(constraint_key)
                
                if constraint_type == '>=' and dim_name in self.dataflow:
                    if constraint_value > self.factors.dimProduct(dim_name):
                        violations.append(f"constrained {dim_name} >= {constraint_value} VS obtained {dim_name}: {self.factors.dimProduct(dim_name)}")
            
            # Check dimensions not in dataflow but with iterations
            for dim in self.arch.coupling.dims:
                if dim not in self.dataflow and len(self.factors[dim]) != 0:
                    violations.append(f"dimension {dim} is not in the dataflow ({self.dataflow}), but still received some iterations ({dim}: {self.factors.dimProduct(dim)}) due to constraints")
            
            if violations:
                return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: " + ", ".join(violations)
        
        return ""

    def __getitem__(self, key : str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key : str, value : Any) -> None:
        setattr(self, key, value)

    def __str__(self) -> str:
        return f"{self.name}: ({super().__str__()}, factors_constraints: {self.factors_constraints})"


"""
A Memory Level within the architecture, with the possibility to store
data and provide it as tiles to the levels below it.

Constructor arguments:
- name: the level's name
- size: the capacity (in number-of-operands, disregarding bits per operand)
- value_access_energy: energy required by each access of one value (in pJ)
                       [w.r.t. Timeloop this is the vector access energy / elements per
                       vector, also called energy-per-scalar-access]
  - Note: if specified, it has priority over wordline_access_energy. At least one of
          value_access_energy and wordline_access_energy must be specified.
- wordline_access_energy: energy required by each wordline access (in pJ)
    - Note: at least one of value_access_energy and wordline_access_energy must be specified.
          Specifying wordline_access_energy requires word_bits and value_bits to be both
          specified as well.
- word_bits: size in bits of the memory's wordlines.
- value_bits: size in bits of the values stored on the memory. This is the same for all
          operands, castings are implicitly assumed to take place whenever needed.
- leakage_energy: energy leaked each clock cycle by the component (in pJ/cc)
- area: the area occupied by the memory (in um^2).
- bandwidth: the bandwidth for reads and writes, it will be divided in 1/2 for
             read and 1/2 for write (in operands/clock-cycle)
- dataflow: specifies the dimensions over which to iterate, defaults to all dimensions
- factors: specifies the initial factors for this level, should not be normally
           specified aside for initializing MSE from a specific configuration
- tile_sizes: specifies the initial tile sizes for this level, should not be normally
              specified aside for initializing MSE from a specific configuration,
              in which case it must be consistent with any other factors initialization
- factors_constraints: constraints on the factors that must be placed on this level.
                      Valid dictionary keys use dimension names, e.g. for a GEMM:
                          - 'M', 'K', and 'N' for exact values;
                          - 'M<=', 'K<=', and 'N<=' for upper bounds;
                          - 'M>=', 'K>=', and 'N>=' for lower bounds;
                      NOTE: the use of the '<=' and '>=' constraints does not shorten the
                            mapper's runtime as much as exact constraints.
- dataflow_constraints: constraints for the order of loops at this level, for any dim not
                        specified here, all permutations are tried while keeping fixed the
                        relative order of constrained dimensions. A placeholder "_" can be
                        used to mark the place where non-specified dimensions are allowed
                        to be inserted. If any "_" is used, the total number of specified
                        dimensions or placeholders must be equal to those in the coupling.
                        E.g., for GEMMs, valid strings are 'M', 'K', 'N', and "_", ["N", "M"]
                        and ["N", "_", "_"] are valid constraints, while ["_", "N"] is not.
                        With ["N", "M"], valid dataflows are NMK, NKM, and KNM.
                        With ["N", "_", "_"] valid dataflows are NMK, NKM.
- bypasses: list of operands which should bypass this level (i.o.w. not be stored here),
            valid strings are 'in', 'int_in', 'int_out', 'w', and 'out'.
- multiple_buffering: factor of multiple buffering employed by this level, must be >1
- multiple_reuses: if True (default) both stationarity reuse (loops skip) and halo reuse can be
                   leveraged at once. Otherwise only the one bound to the innermost loop is leveraged.
- read_value_access_energy: energy required for reads accesses, if specified overrides value_access_energy
- write_value_access_energy: energy required for write accesses, if specified overrides value_access_energy
  - Note: either both or none of read_value_access_energy and write_value_access_energy must be specified.
- read_wordline_access_energy: energy required for reads accesses, if specified overrides wordline_access_energy
- write_wordline_access_energy: energy required for write accesses, if specified overrides wordline_access_energy
  - Note: either both or none of read_wordline_access_energy and write_wordline_access_energy must be specified.
- read_bandwidth: bandwidth allocated for reads, if specified overrides "bandwidth"
- write_bandwidth: bandwidth allocated for writes, if specified overrides "bandwidth"
  - Note: either both or none of read_bandwidth and write_bandwidth must be specified
"""
class MemLevel(Level):
    size : int
    values_per_wordline : int
    read_access_energy : float
    write_access_energy : float
    leakage_energy : float
    read_bandwidth : float
    write_bandwidth : float
    dataflow_constraints : list[str]
    bypasses : list[str]
    in_bp : bool
    w_bp : bool
    out_bp : bool
    int_in_bp : bool
    int_out_bp : bool
    multiple_buffering : int
    multiple_reuses : bool
    
    # POINTERS TO OTHER LEVELS:

    ## flag indicating if this is the last memory before the compute level 
    #  (initialized in setupSpatialLevelPointers)
    next_is_compute : bool 
    ## list of spatial levels following this one (can be empty, initialized in
    #  setupSpatialLevelPointers)
    next_spatials : Optional[list[SpatialLevel]] 
    ## per-operand list of all immediately following levels that back-to-back bypass an operand
    #  not hereby (on this level) bypassed (initialized in setupBypasses)
    next_levels_with_bypass : dict[str, Optional[list[Level]]]
    num_layers : int
    # STATISTICS:
    active_instances_per_layer : dict[int, int]
    active_instances : int
    ## WHEN DO I USE TEMPORAL_ITERATIONS in LEVELS.py?       
    temporal_iterations_per_layer : dict[int, int]
    temporal_iterations : int
    in_reads = 0
    w_reads = 0
    out_reads = 0
    in_writes = 0
    w_writes = 0
    int_in_reads = 0
    int_in_writes = 0
    int_out_reads = 0
    int_out_writes = 0
    last_out_reads = 0 
    last_out_writes = 0 
    per_layer_w_reads : dict[int, int]
    per_layer_w_writes : dict[int, int]
    per_layer_int_in_reads : dict[int, int]
    per_layer_int_in_writes : dict[int, int]
    per_layer_int_out_reads : dict[int, int]
    per_layer_int_out_writes : dict[int, int]
    last_per_layer_int_out_reads : dict[int, int]
    last_per_layer_int_out_writes : dict[int, int]
    stall_cycles_per_layer : dict[int, int]
    latency_read_drain = 0
    latency_fill_update = 0
    cc_per_tile = 0
    stall_cycles = 0
    ideal_bandwidth_read = 0
    ideal_bandwidth_update = 0
    ideal_bandwidth_fill = 0
    ideal_bandwidth_drain = 0
    ideal_bandwidth_read_per_layer : dict[int,int]
    ideal_bandwidth_update_per_layer : dict[int, int]
    ideal_bandwidth_fill_per_layer : dict[int, int]
    ideal_bandwidth_drain_per_layer : dict[int, int]
    #bp_stationarity_solved_here: dict[bool] # tracks if this level’s loops are the ones dictating the dataflow for a bypassed operand going over it

    cc_per_tile_per_layer = 0
    latency_read_drain_per_layer : dict[int, int]
    latency_fill_update_per_layer : dict[int, int]

    def __init__(self, name : str, size : int, value_access_energy : Optional[float] = None, wordline_access_energy : Optional[float] = None, word_bits : Optional[int] = None, value_bits : Optional[int] = None, leakage_energy : float = 0, area : Optional[float] = None, bandwidth : Optional[float] = None, dataflow : Optional[list[str]] = None, factors : Optional[Factors] = None, tile_sizes : Optional[Shape] = None, factors_constraints : Optional[dict[str, int]] = None, dataflow_constraints : Optional[list[str]] = None, bypasses : Optional[list[str]] = None, multiple_buffering : int = 1, multiple_reuses : bool = True, read_value_access_energy : Optional[float] = None, write_value_access_energy : Optional[float] = None, read_wordline_access_energy : Optional[float] = None, write_wordline_access_energy : Optional[float] = None, read_bandwidth : Optional[float] = None, write_bandwidth : Optional[float] = None):
        self.name = name
        self.dataflow = dataflow
        self.size = size
        self._word_bits = word_bits
        self._value_bits = value_bits
        self._wordline_access_energy = wordline_access_energy
        self._value_access_energy = value_access_energy
        self._read_wordline_access_energy = read_wordline_access_energy
        self._write_wordline_access_energy = write_wordline_access_energy
        self._read_value_access_energy = read_value_access_energy
        self._write_value_access_energy = write_value_access_energy
        self.leakage_energy = leakage_energy
        self.area = area
        self._bandwidth = bandwidth
        self.read_bandwidth = read_bandwidth
        self.write_bandwidth = write_bandwidth
        self.factors = factors
        self.tile_sizes = tile_sizes
        self.factors_constraints = factors_constraints if factors_constraints else {}
        self.dataflow_constraints = dataflow_constraints if dataflow_constraints else []
        self.bypasses = bypasses if bypasses else []
        self.in_bp = 0 if (bypasses and 'in' in bypasses) else 1
        self.w_bp = 0 if (bypasses and 'w' in bypasses) else 1
        self.int_in_bp = 0 if (bypasses and 'int_in' in bypasses) else 1
        self.int_out_bp = 0 if (bypasses and 'int_out' in bypasses) else 1
        self.out_bp = 0 if (bypasses and 'out' in bypasses) else 1
        self.multiple_buffering = multiple_buffering
        self.multiple_reuses = multiple_reuses
        

    """
    Sets up a pointer back to the whole architecture.
    Ultimates the initialization of the level and validates its attributes.
    """
    def initArch(self, arch : Arch):
        self.arch = arch
        self.dataflow = self.dataflow if self.dataflow else arch.coupling.dims # dimensions over which to iterate
        assert all(dim in arch.coupling.dims for dim in self.dataflow), f"Arch: {arch.name} -> Level: {self.name}: accepted names for dimensions, as per the present coupling, are solely {arch.coupling.dims} provided ones were {self.dataflow}."
        assert self.size >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative size ({self.size}) does not mean anything."
        # read_access_energy and write_access_energy are intended always for one value, remember to bring accessed values to a multiple of values_per_wordline for the correct total energy
        assert (self._value_access_energy or (self._read_value_access_energy and self._write_value_access_energy)) or (self._word_bits and self._value_bits and (self._wordline_access_energy or (self._read_wordline_access_energy and self._write_wordline_access_energy))), f"Arch: {arch.name} -> Level: {self.name}: either value_access_energy ({self._value_access_energy}) or read_value_access_energy ({self._read_value_access_energy}) and write_value_access_energy ({self._write_value_access_energy}) must be specified, alternatively, you can specify word_bits ({self._word_bits}) and value_bits ({self._value_bits}) and either wordline_access_energy ({self._wordline_access_energy}) or read_wordline_access_energy ({self._read_wordline_access_energy}) and write_wordline_access_energy ({self._write_wordline_access_energy}). In any case, when if either of read_*_access_energy or write_*_access_energy is specified, the other must be present as well."
        if (self._value_access_energy or (self._read_value_access_energy and self._write_value_access_energy)):
            self.values_per_wordline = 1
            self.read_access_energy = self._read_value_access_energy if self._read_value_access_energy else self._value_access_energy
            self.write_access_energy = self._write_value_access_energy if self._write_value_access_energy else self._value_access_energy
        else:
            assert self._word_bits >= self._value_bits, f"Arch: {arch.name} -> Level: {self.name}: word_bits ({self._word_bits}) must be more than value_bits ({self._value_bits}), otherwise a value cannot fit on a single wordline."
            self.values_per_wordline = self._word_bits // self._value_bits
            self.read_access_energy = (self._read_wordline_access_energy if self._read_wordline_access_energy else self._wordline_access_energy) / self.values_per_wordline
            self.write_access_energy = (self._write_wordline_access_energy if self._write_wordline_access_energy else self._wordline_access_energy) / self.values_per_wordline
        del self._word_bits, self._value_bits, self._wordline_access_energy, self._value_access_energy, self._read_wordline_access_energy, self._write_wordline_access_energy, self._read_value_access_energy, self._write_value_access_energy
        assert self.read_access_energy >= 0 and self.write_access_energy >= 0 and self.leakage_energy >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative access energy ({self.read_access_energy} read, {self.read_access_energy} write), ({self.leakage_energy} leak), does not mean anything (unless you are into sci-fi stuff)."
        assert not self.area or self.area >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative area ({self.area}) does not mean anything."
        # NOTE: 1/2 split of bandwidth for consistency with Timeloop - not a true must...
        assert (self._bandwidth and not self.read_bandwidth and not self.write_bandwidth) or (self.read_bandwidth and self.write_bandwidth), f"Arch: {arch.name} -> Level: {self.name}: either bandwidth ({self._bandwidth}) or read_bandwidth ({self.read_bandwidth}) and write_bandwidth ({self.write_bandwidth}) must be specified, if either of read_bandwidth or write_bandwidth is specified, the other must be specified as well."
        self.read_bandwidth = self.read_bandwidth if self.read_bandwidth else self._bandwidth/2
        self.write_bandwidth = self.write_bandwidth if self.write_bandwidth else self._bandwidth/2
        del self._bandwidth
        self.factors = self.factors if self.factors else Factors(arch.coupling.dims)
        self.tile_sizes = self.tile_sizes if self.tile_sizes else Shape({dim: 1 for dim in arch.coupling.dims})
        assert self.read_bandwidth >= 0 and self.write_bandwidth >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative bandwidth ({self.read_bandwidth} R, {self.write_bandwidth} W) does not mean anything."
        ## changed due to the 2 char
        assert all([
    constr in self.dataflow or 
    (constr.endswith('<=') and constr[:-2] in self.dataflow) or 
    (constr.endswith('>=') and constr[:-2] in self.dataflow) 
    for constr in self.factors_constraints.keys()
]), f"Arch: {arch.name} -> Level: {self.name}: all keys within factor constraints ({list(self.factors_constraints.keys())}) must be a dimension of the dataflow ({self.dataflow}) and in the form 'dim', 'dim<=', or 'dim>='."
        assert all([sum((constr == dim) + (constr == dim + '<=') + (constr == dim + '>=') for constr in self.factors_constraints.keys()) <= 1 for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: each dimension must occur at most once in factor constraints ({list(self.factors_constraints.keys())}), regardless of the use of '>=' or '<='."        
        assert all([value > 0 for value in self.factors_constraints.values()]), f"Arch: {arch.name} -> Level: {self.name}: all factor constraints ({self.factors_constraints}) must have a value strictly > 0."
        assert all([constr == '_' or constr in self.dataflow for constr in self.dataflow_constraints]), f"Arch: {arch.name} -> Level: {self.name}: all dims specified as dataflow constraints ({self.dataflow_constraints}) must be part of the dataflow ({self.dataflow}) or be placeholders ('_')."
        assert all([sum(constr == dim for constr in self.dataflow_constraints) <= 1 for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: each dimension must appear at most once in dataflow constraints ({self.dataflow_constraints})."
        assert '_' not in self.dataflow_constraints or len(self.dataflow_constraints) == len(self.dataflow), f"Arch: {arch.name} -> Level: {self.name}: when using placeholders ('_') in dataflow constraints ({self.dataflow_constraints}), the total number of specified dimensions or placeholders ({len(self.dataflow_constraints)}) must be equal to those in the dataflow ({len(self.dataflow)})."
        # NOTE: this way of constructing the dataflow from the constraints is redundant, but useful if one wants to skip the
        # exploration of permutations since with this method the dataflow will be immediately consistent with constraints.
        if self.dataflow_constraints and '_' not in self.dataflow_constraints:
            self.dataflow = self.dataflow_constraints + [dim for dim in self.dataflow if dim not in self.dataflow_constraints]
        elif self.dataflow_constraints and '_' in self.dataflow_constraints:
            gen = (dim for dim in self.dataflow if dim not in self.dataflow_constraints)
            self.dataflow = [dim if dim != '_' else next(gen) for dim in self.dataflow_constraints]
        assert self.multiple_buffering >= 1, f"Arch: {arch.name} -> Level: {self.name}: multiple buffering ({self.multiple_buffering}) must be at least 1."
        self.next_is_compute = False
        self.next_spatials = None
        self.next_levels_with_bypass = {'in': None, 'w': None, 'out': None, 'int_in': None, 'int_out': None}
        #self.bp_stationarity_solved_here = {'in': False, 'w': False, 'out': False}

    """
    Initializes the bypasses which start from this level.
    Let "levels" be all levels starting from the next one going downward
    up until and including the last one bypassing "operand".

    => This method must be invoked while iterating from outer to inner levels
    as it updates the level's notion of bypassed operations.
    """
    def initBypass(self, operand : str, levels : list[Level]) -> None:
        self.next_levels_with_bypass[operand] = levels
        if operand == 'in':
            self.in_bp = 0
        elif operand == 'w':
            self.w_bp = 0
        elif operand == 'int_in':
            self.int_in_bp = 0
        elif operand == 'int_out':
            self.int_out_bp = 0
        elif operand == 'out':
            self.out_bp = 0

    """
    Sets MOPs statistics for this level.
    Those must include both operations with the above and below level.
    """
    def setMOPs(self, in_reads : int, in_writes : int, per_layer_w_reads : dict[int, int], per_layer_w_writes : dict[int, int], per_layer_int_in_reads : dict[int, int], per_layer_int_in_writes : dict[int, int], per_layer_int_out_reads : dict[int, int], per_layer_int_out_writes : dict[int, int], out_reads : int, out_writes : int) -> None:
        # Store the input reads (single value, not per-layer)
        self.in_reads = in_reads
        self.in_writes = in_writes
        # Store per-layer weight reads dictionary
        self.per_layer_w_reads = per_layer_w_reads.copy()
        self.per_layer_w_writes = per_layer_w_writes.copy()
        # Store per-layer intermediate reads dictionaries
        self.per_layer_int_in_reads = per_layer_int_in_reads.copy()
        self.per_layer_int_in_writes = per_layer_int_in_writes.copy()
        self.per_layer_int_out_reads = per_layer_int_out_reads.copy()
        self.per_layer_int_out_writes = per_layer_int_out_writes.copy()
        # Store output reads and writes (single values)
        self.out_reads = out_reads
        self.out_writes = out_writes    
        # Calculate totals for backward compatibility
        self.w_reads = sum(per_layer_w_reads.values()) if per_layer_w_reads else 0
        self.w_writes = sum(per_layer_w_writes.values()) if per_layer_w_writes else 0
        self.int_in_reads = sum(per_layer_int_in_reads.values()) if per_layer_int_in_reads else 0
        self.int_in_writes = sum(per_layer_int_in_writes.values()) if per_layer_int_in_writes else 0
        self.int_out_reads = sum(per_layer_int_out_reads.values()) if per_layer_int_out_reads else 0
        self.int_out_writes = sum(per_layer_int_out_writes.values()) if per_layer_int_out_writes else 0
        

    # fill, drain, read, and update are intended in the "Buffet" sense, and to be
    # computed they require "last_out_writes"  and "last_out_reads" from the level
    # ABOVE to distinguish the components of "out_reads" and "out_writes"!
    """
    Sets MOPs statistics for the interations between this level and
    the MemLevel immediately above it.
    """
    def setAboveMOPs(self, last_out_reads : Optional[int] = None, last_out_writes : Optional[int] = None, last_per_layer_int_out_reads : dict[int, int] = None, last_per_layer_int_out_writes : dict[int, int] = None) -> None:
        if last_out_reads is not None:
            self.last_out_reads = last_out_reads
        if last_out_writes is not None:
            self.last_out_writes = last_out_writes
        if last_per_layer_int_out_reads is not None:
            self.last_per_layer_int_out_reads = last_per_layer_int_out_reads.copy()
        if last_per_layer_int_out_writes is not None:
            self.last_per_layer_int_out_writes = last_per_layer_int_out_writes.copy()

    """
    NOTE:
    Fill	DOWN ↓	Data written TO this level FROM the level above (e.g., DRAM → FeatureMemory)
    Drain	UP ↑	Data read FROM this level TO the level above (e.g., FeatureMemory → DRAM)
    Read	DOWN ↓	Data read FROM this level TO the level below (e.g., FeatureMemory → Compute)
    Update	UP ↑	Data written TO this level FROM the level below (e.g., Compute → FeatureMemory)
    """
    
    """
    Returns "Fills" as intended for Buffets, 
    thus the incoming writes
    from an higher level.
    Fill	DOWN ↓	Data written TO this level FROM the level above (e.g., DRAM → FeatureMemory)
    """
    def getFill(self) -> int:
        return self.in_writes + self.w_writes + self.last_out_reads #include fills for outputs
    
    def getFillPerLayer(self, layer_id) -> int:
        if self.arch.coupling.getNumLayers() > 1:
            if layer_id == 0:
                return self.in_writes + self.per_layer_w_writes[layer_id] + self.last_per_layer_int_out_reads[layer_id]
            elif layer_id == self.arch.coupling.getNumLayers() - 1:
                return self.per_layer_int_in_writes[layer_id - 1] + self.per_layer_w_writes[layer_id] + self.last_out_reads
            else:
                return self.per_layer_int_in_writes[layer_id - 1] + self.per_layer_w_writes[layer_id] + self.last_per_layer_int_out_reads[layer_id]
        else: 
            return self.getFill()

    ## Counters of Output Writes (in order to free up the space)
    """
    Returns "Drains" as intended for Buffets, thus the outgoing reads
    towards an higher level.
    Drain	UP ↑	Data read FROM this level TO the level above (e.g., FeatureMemory → DRAM)
    """
    def getDrain(self) -> int:
        return self.last_out_writes
    
    """
    Returns the drains (reads going UP to outer level) for EXTERNAL bandwidth calculation.
    
    For FeatureMemory:
    - Only the LAST layer has a drain (final output → DRAM)
    - The drain size = final output size (NOT partial sum accumulation size)
    - This is the size of the complete output feature map
    """
    def getDrainPerLayer(self, layer_id) -> int:
        if self.arch.coupling.getNumLayers() > 1:
            if layer_id == self.arch.coupling.getNumLayers() - 1:
                return self.last_out_writes
            else:
                return self.last_per_layer_int_out_writes[layer_id]
        else: 
            return self.getDrain()

    

    ## Counters of R, without the writes done to drain
    """
    Returns "Reads" as intended for Buffets, thus the outgoing reads
    towards a lower level.
    Read	DOWN ↓	Data read FROM this level TO the level below (e.g., FeatureMemory → Compute)
    """
    def getRead(self) -> int:
        return self.in_reads + self.w_reads + (self.out_reads - self.last_out_writes) #ignore reads done to drain

    
    def getReadPerLayerWithoutPartialAccumulation(self, layer_id) -> int:
        if self.arch.coupling.getNumLayers() > 1:
            if layer_id == 0:
                return self.in_reads + self.per_layer_w_reads[layer_id] 
            elif layer_id == self.arch.coupling.getNumLayers() - 1:
                return self.per_layer_int_in_reads[layer_id - 1] + self.per_layer_w_reads[layer_id]
            else:
                return self.per_layer_int_in_reads[layer_id - 1] + self.per_layer_w_reads[layer_id]
        else: 
            return self.getRead()


    """
    Read	DOWN ↓	Data read FROM this level TO the level below (e.g., FeatureMemory → Compute)

    Returns the reads.
    
    For bandwidth, EXCLUDE the partial sum accumulation reads (per_layer_int_out_reads and out_reads)
    because those are handled by the PE.
    
    INCLUDE:
    - in_reads (layer 0 input)
    - int_in_reads (intermediate inputs from previous layers)  
    - w_reads (weights)
    
    EXCLUDE:
    - int_out_reads (partial sum accumulation - handled by registers)
    
    This uses getReadPerLayerWithoutPartialAccumulation which excludes int_out_reads.
    """
    def getReadPerLayer(self, layer_id) -> int:
        if self.arch.coupling.getNumLayers() > 1:
            return self.getReadPerLayerTotal(layer_id)
        else: 
            return self.getRead()


    """ The if for "FeatureMemory" is for a special case of DepFin feature shifter. """
    def getReadPerLayerTotal(self, layer_id) -> int:
        if self.arch.coupling.getNumLayers() > 1:
            if layer_id == 0:
                if self.name == "FeatureMemory":
                    # DEBUG print(f"self.in_reads: {self.in_reads*1/5}, self.per_layer_w_reads[{layer_id}]: {self.per_layer_w_reads[layer_id]}, self.per_layer_int_out_reads[{layer_id}]: {self.per_layer_int_out_reads[layer_id]}")
                    return self.in_reads*1/5 + self.per_layer_w_reads[layer_id] + self.per_layer_int_out_reads[layer_id]
                return self.in_reads + self.per_layer_w_reads[layer_id] + self.per_layer_int_out_reads[layer_id]
            elif layer_id == self.arch.coupling.getNumLayers() - 1:
                if self.name == "FeatureMemory":
                    # DEBUG print(f"DEBUG: {self.name} layer_id={layer_id}, per_layer_int_in_reads[{layer_id - 1}]={self.per_layer_int_in_reads[layer_id - 1]*1/5:,.0f}, out_reads={self.out_reads:,.0f}, last_out_writes={self.last_out_writes:,.0f}, per_layer_w_reads[{layer_id}]={self.per_layer_w_reads[layer_id]:,.0f}")
                    return self.per_layer_int_in_reads[layer_id - 1]*1/5 + (self.out_reads - self.last_out_writes) + self.per_layer_w_reads[layer_id]
                return self.per_layer_int_in_reads[layer_id - 1] + (self.out_reads - self.last_out_writes) + self.per_layer_w_reads[layer_id]
                
            else:
                if self.name == "FeatureMemory":
                    # DEBUG print(f"{self.name} layer_id={layer_id}: getReadPerLayer: {self.per_layer_int_in_reads[layer_id - 1]*1/5 + self.per_layer_w_reads[layer_id] + (self.per_layer_int_out_reads[layer_id] - self.last_per_layer_int_out_writes[layer_id])}, per_layer_int_in_reads[{layer_id - 1}]={self.per_layer_int_in_reads[layer_id - 1]:,.0f}, per_layer_w_reads[{layer_id}]={self.per_layer_w_reads[layer_id]:,.0f}, per_layer_int_out_reads[{layer_id}]={self.per_layer_int_out_reads[layer_id]:,.0f}, last_per_layer_int_out_writes[{layer_id}]={self.last_per_layer_int_out_writes[layer_id]:,.0f}")
                    return self.per_layer_int_in_reads[layer_id - 1]*1/5 + self.per_layer_w_reads[layer_id] + (self.per_layer_int_out_reads[layer_id] - self.last_per_layer_int_out_writes[layer_id])
                return self.per_layer_int_in_reads[layer_id - 1] + self.per_layer_w_reads[layer_id] + (self.per_layer_int_out_reads[layer_id] - self.last_per_layer_int_out_writes[layer_id])   
        else: 
            return self.getRead()

    """
    Update	UP ↑	Data written TO this level FROM the level below (e.g., Compute → FeatureMemory)

    Returns the updates (writes from below) for bandwidth calculation purposes.
    
    Updates = writes FROM inner (below) level TO the current level.
    
    For bandwidth, we count the FINAL output writes (not partial sum accumulation writes),
    assuming there's an implicit accumulation inside the PE.
    Only the final result is written to this level.
    
    We compute the one-time write by dividing per_layer_int_out_writes by the 
    accumulation factor. The accumulation factor is derived from layer 0:
    acc_factor = per_layer_int_out_writes[0] / in_reads
    
    Then for any layer L:
    one_time_write[L] = per_layer_int_out_writes[L] / acc_factor
    """
    def getUpdatePerLayer(self, layer_id) -> int:
        if self.arch.coupling.getNumLayers() > 1:
            if layer_id == self.arch.coupling.getNumLayers() - 1:
                return self.out_writes - self.last_out_reads
            else:
                #if layer_id == 1 or layer_id == 9:
                #    print(f"UpdatePerLayer: self.per_layer_int_out_writes[{layer_id}]: {self.per_layer_int_out_writes[layer_id]} - self.last_per_layer_int_out_reads[{layer_id}]: {self.last_per_layer_int_out_reads[layer_id]}")
                #print(f"DEBUG: getUpdatePerLayerTotal called for layer_id={layer_id}, per_layer_int_out_writes[{layer_id}]={self.per_layer_int_out_writes[layer_id]:,.0f}, last_per_layer_int_out_reads[{layer_id}]={self.last_per_layer_int_out_reads[layer_id]:,.0f}")
                return self.per_layer_int_out_writes[layer_id] - self.last_per_layer_int_out_reads[layer_id]
        else: 
            return self.getUpdate()
        


    ## Counters of W of Outputs (without the updates coming from)
    """
    Returns "Updates" as intended for Buffets, thus the incoming writes
    from a lower level. Includes full accumulation count (for debugging/analysis).
    """
    def getUpdate(self) -> int:
        return self.out_writes - self.last_out_reads #ignore updates coming from fills

    def getUpdatePerLayerTotal(self, layer_id) -> int:
        if self.arch.coupling.getNumLayers() > 1:
            if layer_id == self.arch.coupling.getNumLayers() - 1:
                return self.out_writes - self.last_out_reads
            else:
                #if layer_id == 1 or layer_id == 9:
                #    print(f"UpdatePerLayer: self.per_layer_int_out_writes[{layer_id}]: {self.per_layer_int_out_writes[layer_id]} - self.last_per_layer_int_out_reads[{layer_id}]: {self.last_per_layer_int_out_reads[layer_id]}")
                #print(f"DEBUG: getUpdatePerLayerTotal called for layer_id={layer_id}, per_layer_int_out_writes[{layer_id}]={self.per_layer_int_out_writes[layer_id]:,.0f}, last_per_layer_int_out_reads[{layer_id}]={self.last_per_layer_int_out_reads[layer_id]:,.0f}")
                return self.per_layer_int_out_writes[layer_id] - self.last_per_layer_int_out_reads[layer_id]
        else: 
            return self.getUpdate()

    ## Modify getSettedMOPs
    """
    Returns the total MOPs previoulsy stored by setMOPs().
    """
    def getSettedMOPs(self) -> int:
        return (self.in_reads + self.w_reads + self.int_in_reads + self.int_out_reads + self.out_reads + 
            self.in_writes + self.w_writes + self.int_in_writes + self.int_out_writes + self.out_writes)
 
    """
    Sets Latency and related statistics for this level.
    """
    def setLatency(self, latency_read_drain : int, latency_fill_update : int, cc_per_tile : int, stall_cycles : int, ideal_bandwidth_read : float, ideal_bandwidth_update : float, ideal_bandwidth_fill : float, ideal_bandwidth_drain : float) -> None:
        self.latency_read_drain = latency_read_drain
        self.latency_fill_update = latency_fill_update
        self.cc_per_tile = cc_per_tile
        self.stall_cycles = stall_cycles
        self.ideal_bandwidth_read = ideal_bandwidth_read
        self.ideal_bandwidth_update = ideal_bandwidth_update
        self.ideal_bandwidth_fill = ideal_bandwidth_fill
        self.ideal_bandwidth_drain = ideal_bandwidth_drain

    """
    Sets Latency and related statistics for this level.
    """
    def setLatencyPerLayer(self, latency_read_drain_per_layer : dict[int, int], latency_fill_update_per_layer : dict[int, int], cc_per_tile_per_layer : dict[int, int], stall_cycles_per_layer : dict[int, int], ideal_bandwidth_read_per_layer : float, ideal_bandwidth_update_per_layer : float, ideal_bandwidth_fill_per_layer : float, ideal_bandwidth_drain_per_layer : float) -> None:
        self.latency_read_drain_per_layer = latency_read_drain_per_layer.copy()
        self.latency_fill_update_per_layer = latency_fill_update_per_layer.copy()
        self.cc_per_tile_per_layer = cc_per_tile_per_layer.copy()
        self.stall_cycles_per_layer = stall_cycles_per_layer.copy()
        self.ideal_bandwidth_read_per_layer = ideal_bandwidth_read_per_layer.copy()
        self.ideal_bandwidth_update_per_layer = ideal_bandwidth_update_per_layer.copy()
        self.ideal_bandwidth_fill_per_layer = ideal_bandwidth_fill_per_layer.copy()
        self.ideal_bandwidth_drain_per_layer = ideal_bandwidth_drain_per_layer.copy()


    """
    Returns the Latency previoulsy stored by setLatency().
    """
    def getSettedLatency(self) -> int:
        return max(self.latency_read_drain, self.latency_fill_update)

    """
    Returns the Latency previoulsy stored by setLatencyPerLayer().
    """
    def getSettedLatencyPerLayer(self, layer_id) -> int:
        return max(self.latency_read_drain_per_layer[layer_id], self.latency_fill_update_per_layer[layer_id])
 

    """
    If the architecture is like DepFiN, this MOPs at regime:
    Returns the memory operations between this level and the one below it, at regime phase.
    Returns: reads going downward, writes coming upward.
    In other words, reads and writes between this level and the one(s) below it.
    """

    """
    Returns the memory operations between this level and the one below it in a single iteration.
    Specifically: returns reads going downward from this level and writes
    coming upward from lower levels. In other words, reads and writes
    between this level and the one(s) below it.

    => The returned value must be multiplied by the iterations happening
       on levels above this one in order to get the true total, that will be done in updateStats in model.py.

    Bypasses are handled like this:
    If this level is the last before a bypass of a certain operand, it
    will be invoking this same method on the last level before the next
    one which does not bypass the operand in question.

    For example:
    We have a memory hierarchy with levels L1, L2, L3, L4, and L5, where:

        L2 is configured to bypass weights
        L3 is configured to bypass weights
        L4 stores weights again
        In this case, when L1 needs to send weight data downward,
        instead of communicating with L2 (which would normally be next), 
        it communicates directly with L4 (the next level that actually handles weights).
        This creates a "virtual connection" that jumps over the bypassed levels

    This way MOPs are computed w.r.t. the tile size of such level, which
    corresponds with the actual tiles being accessed from this level.
    For each level encountered in between, MOPs are scaled accordingly.
    

    Lastly, factors encountered on dimensions orthogonal to the output
    are returned too, to allow the caller to remove one such iteration
    to account for the presence or absence of the bias (bias_read flag).
    
    Arguments:
    - in_bp, w_bp, out_bp, int_in_bp, int_out_bp: whether the respective operands are bypassed or not,
                           defaults to the instance's setting if not provided.
                           Values are '0' for bypass, '1' for keep.
    - ignore_bypasses: if True, MOPs are returned only relative to this level's
                       accesses from strictly adjacent levels, bypassed operands
                       will thus show 0 MOPs.
    """
    ## in_bp = 1 == Keep, 0 == Bypass
    ## MOPs is composed by 3 parts
        ## 1. actual_dataflow for 'this' MemLevel
        ## 2. calculation of stationarity for 'this' MemLevel (change the min length w.r.t. the level)
        ##    2.1 if next level is not a compute one: multiplication of the Tile size[i]*Num_it[i]+ Tile_size[j] for all j in innermost_dim_sum
        ##    2.2 
        ###           2.2.1 if next level is not a spatial one: prod( for dim_sum in in_coupling if dim_sum is not innermost_dim_sum: multiplication of Tile size[dim_sum] )
        ###           2.2.2 else: prod( for dim_sum in in_coupling if dim_sum is not innermost_dim_sum: \\next_spatial_unroll[dim])
        ##    2.3 for dim in actual_dataflow[:i+1] Mul n_iterations[dim] 
        ##            2.3.1 out_reads_factors only the orthogonal  
        ## 3. handle bypasses
        ##    3.1 call to MOps of this level
        ##    3.2 stationarity_to_address == T = None of dim of this operand level's dataflow have n_iterations > 1; F = any dim has n_iterations > 1
        ##    3.3 for operand
        ###           3.3.1  for in_btwn upper to lower (since when s_t_a==FALSE will be false to all below level)
        #                      if None DIM of input of the above's in_btwn (and level) dataflows have > 1 iteration
        ####                   3.3.1.1 if stationarity_is_to_address = i<0 is False and not spatial level below: Handle this
        #                      divide by the product of tile_sizes of the innermost_dim_sum        
        ####                   3.3.1.2 same as before  
        #                      else prod of n_iteration of all the dim of in_btwn
        ###           3.3.2  level
        #                      if None DIM of input of the above's in_btwn (and level) dataflows have > 1 iteration
        ####                   3.3.2.1 if stationarity_is_to_address = i<0 is False and not spatial level below: Handle this
        #                      divide by the product of tile_sizes of the innermost_dim_sum
        ####                   3.3.2.2 same as before
        #                      else prod of n_iteration of all the dim of level
        ## 4. 
    ## CAMBIARE forma input
    ## input diventa una lista di input intermedi e girare con loop
    ##
    ## in_reads, per_layer_w_reads_bp, per_layer_int_in_reads_bp, per_layer_int_out_reads_bp, out_reads_bp, out_writes_bp, out_reads_bp_factors
    def MOPs(self, in_bp : Optional[bool] = None, w_bp : Optional[bool] = None, out_bp : Optional[bool] = None, int_in_bp : Optional[bool] = None, int_out_bp : Optional[bool] = None, ignore_bypasses : bool = False, verbose : bool = False) -> tuple[int, dict[int, int], dict[int, int], dict[int, int], dict[int, int], int, int, int]:
        # Helper function to conditionally vprint inside MOPs
        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        vprint(f"arch.name: {self.arch.name}, level.name: {self.name}")
        in_bp = in_bp if in_bp != None else self.in_bp
        w_bp = w_bp if w_bp != None else self.w_bp
        out_bp = out_bp if out_bp != None else self.out_bp
        int_in_bp = int_in_bp if int_in_bp != None else self.int_in_bp
        int_out_bp = int_out_bp if int_out_bp != None else self.int_out_bp
        num_layers = self.arch.coupling.getNumLayers()               
        if not ignore_bypasses:
            vprint(f"Level: {self.name}: Starting MOPs calculation with in_bp: {in_bp}, w_bp: {w_bp}, out_bp: {out_bp}, int_in_bp: {int_in_bp}, int_out_bp: {int_out_bp}, ignore_bypasses: {ignore_bypasses}, dataflow: {self.dataflow}, factors: {self.factors}, tile_sizes: {self.tile_sizes}, factors_constraints: {self.factors_constraints}, bypasses: {self.bypasses}, multiple_buffering: {self.multiple_buffering}, multiple_reuses: {self.multiple_reuses}")
        else:
            vprint(f"Level (handling bypass): {self.name}: Starting MOPs calculation with in_bp: {in_bp}, w_bp: {w_bp}, out_bp: {out_bp}, int_in_bp: {int_in_bp}, int_out_bp: {int_out_bp}, dataflow: {self.dataflow}, factors: {self.factors}, tile_sizes: {self.tile_sizes}, factors_constraints: {self.factors_constraints}, bypasses: {self.bypasses}, multiple_buffering: {self.multiple_buffering}, multiple_reuses: {self.multiple_reuses}")
        ## List of dim in df with loops > one
        actual_dataflow = list(filter(lambda dim : self.factors.dimProduct(dim) > 1, self.dataflow))
        # separate the actual_dataflow per layer: dict[n_layer, order of loop of that ]
        actual_dataflow_per_layer: dict[int, list[str]] = {}
        if self.arch.coupling.getNumLayers() > 1:
          for layer_id in range(self.arch.coupling.getNumLayers()):
            # For each layer, filter dimensions that have loops > 1 AND are relevant to that layer
            layer_relevant_dims = self.arch.coupling.relevantDimsForLayer(layer_id)
            actual_dataflow_per_layer[layer_id] = [
                dim for dim in actual_dataflow
                if dim in layer_relevant_dims
            ]
            vprint(f"actual_dataflow_per_layer[{layer_id}]: {actual_dataflow_per_layer[layer_id]}")
        else:
            actual_dataflow_per_layer[0] = actual_dataflow
        vprint(f"Level: {self.name}: actual_dataflow: {actual_dataflow}, dataflow: {self.dataflow}, actual_dataflow_per_layer: {actual_dataflow_per_layer}, factors: {self.factors}, num layers: {num_layers}")
        vprint(f"tile size for each dim in actual_dataflow: {[ (dim, self.tile_sizes[dim]) for dim in actual_dataflow ]}")
        vprint(f"tile size for each dim in actual_dataflow_per_layer[0]: {[ (dim, self.tile_sizes[dim]) for dim in actual_dataflow_per_layer[0] ]}")
        vprint(f"tile size for each dim in actual_dataflow_per_layer[1]: { [ (dim, self.tile_sizes[dim]) for dim in actual_dataflow_per_layer[1] ] if 1 in actual_dataflow_per_layer else 'N/A' }")
        vprint(f"tile size for each dim in actual_dataflow_per_layer[2]: { [ (dim, self.tile_sizes[dim]) for dim in actual_dataflow_per_layer[2] ] if 2 in actual_dataflow_per_layer else 'N/A' }")
        vprint(f"...")
        vprint(f"tile size for each dim in actual_dataflow_per_layer[10]: { [ (dim, self.tile_sizes[dim]) for dim in actual_dataflow_per_layer[10] ] if 10 in actual_dataflow_per_layer else 'N/A' }")
        vprint(f"\n\nStationarity calculation for Level: {self.name}: in_bp = {in_bp}, w_bp = {w_bp}, int_in_bp = {int_in_bp}, int_out_bp = {int_out_bp}, out_bp = {out_bp}\n")
        # stationarity calculation for inputs
        in_reads = int(in_bp)
        if in_bp:
            vprint(f"\nInitialization Level: {self.name}: in_reads = {in_reads}")
            i = len(actual_dataflow_per_layer[0]) - 1
            # dimensions part of a sum of indices with the innermost iterated dimension
            innermost_dim_sum = None
            ## Calculate the num of el reads for the innermost element of innermost_dim_sum
            ## Num of unique input elements reads
                ## n_iteration(innermost_of_innermost_dim_sum) * tile_sizes(innermost_of_innermost_dim_sum) +
                ## + tile_sizes(other_dims_in_innermost_dim_sum) , stride                                       
            # if the next level is not a compute one, we can reuse the halo left by the 
            # innermost iterated dimension                
            if not self.next_is_compute:
                skipped = False
                # skip contigous innermost orthogonal dimensions (determining stationarity)
                # search for the first dimension in the dataflow which is part of the coupling
                vprint(f"Level: {self.name}: flat_in_coupling = {self.arch.coupling.flat_in_coupling}, i = {i}, actual_dataflow_per_layer[0] = {actual_dataflow_per_layer[0]}")
                while i >= 0 and (actual_dataflow_per_layer[0][i] not in self.arch.coupling.flat_in_coupling):
                    i -= 1
                    skipped = True

                    innermost_dim_sum = self.arch.coupling.getDimSum('in', actual_dataflow_per_layer[0][i], 2) if i >= 0 else None                    
                    vprint(f"Level: {self.name}: innermost_dim_sum = {innermost_dim_sum}, i = {i}, actual_dataflow_per_layer[0] = {actual_dataflow_per_layer[0]}")
                    # Check for sliding window reuse necessary condition: none of the innermost_dim_sum dimensions are spatially reused
                    # if any dimension in innermost_dim_sum is spatially unrolled after this level,
                    # no halo reuse can occur because the instance
                    # storing reusable data from the previous iteration differs from the instance
                    # which needs that data for the next iteration

                    # Halo reuse is disabled if any dimension in the sliding window is spatially parallelized at any level below this one
                    # (distributed across multiple hardware units). 
                    
                    # This is because the overlapping data would be physically stored in different hardware instances, 
                    # making reuse impossible.
                    # NOTE: halo reuse, when a sum of indices is involved, after reading the first tile, 
                    # all subsequent ones are only read for the part not in common
                    # with the preceeding tile 
                    # [here a tile is intended as deriving from the product of sum of tile sizes matching 
                    # the product of sum in the coupling]
                    # TODO: think about how to consider memories with all iterations at 1. 
                    # Should "next_spatials" cut through them and see fanouts beyond, 
                    # or stop there as if they were used (this also has repercussions on bypass handling)?
                    
                    # All spatial levels below this one have exactly one factor per dimension,
                    # all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum): 
                    #   true if ALL dimensions in sliding window are not spatially parallelized
                    # not all( ALL dimensions in sliding window are not spatially parallelized for all spatial levels below this one)
                    #   true if at least one dimension in sliding window is spatially parallelized
                    if innermost_dim_sum and (not all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) 
                                           for sp_level in self.next_spatials) or (skipped and not self.multiple_reuses)):
                        innermost_dim_sum = None
                    ## HALO REUSE
                    # reuse the halo left by the innermost iterated dimension on the iterations on dimensions part of a sum of indices with it
                    if innermost_dim_sum:
                        in_reads *= distinct_values([self.factors.dimProduct(actual_dataflow_per_layer[0][i])*self.tile_sizes[actual_dataflow_per_layer[0][i]]] + 
                                                      [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[0][i]], 
                                                      [self.arch.getInStride(actual_dataflow_per_layer[0][i])] + 
                                                      [self.arch.getInStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[0][i]])
                        i -= 1
                ## calculate the num of unique in el reads for 1 iteration of a tile
                ## for dim_sum in in_coupling 
                ##   if dim_sum is not innermost_dim_sum
                ##      in_r *= d_v( tile_sizes(dims_in_dim_sum // tot_iteration_of_dim), stride )
                ##      in_r *= tot_iteration_of_dim                
            vprint(f"input read after innermost_dim_sum handling: {in_reads}")               
            if not self.next_spatials:
                for dim_sum in self.arch.coupling.in_coupling:
                    if dim_sum is not innermost_dim_sum:
                        pass  # DEBUG print(f"dim_sum: {dim_sum}, tile sizes[dim] for dim in dim_sum: {[self.tile_sizes[dim] for dim in dim_sum]}")

                ## actual tile size on every fanout level, regardless of how many iterations are there on the next spatial 
                in_reads *= prod(distinct_values([self.tile_sizes[dim] for dim in dim_sum], [self.arch.getInStride(dim) for dim in dim_sum]) 
                                for dim_sum in self.arch.coupling.in_coupling if dim_sum is not innermost_dim_sum)
                vprint(f"Level: {self.name}, Layer {0}: after no Spatial, Input in_reads = {in_reads}, i = {i}, innermost_dim_sum = {innermost_dim_sum}")
            else:
                ## WHY ALL THE DIM? and not only input
                # total iterations per dimension in the following consecutive fanout levels
                ## handle MULTICAST and halo 
                next_spatial_unroll = {dim: prod(level.factors.dimProduct(dim) for level in self.next_spatials 
                                if not level.selective_multicast_support) for dim in self.arch.coupling.getFlatInputCoupling()}
                vprint(f"innermost_dim_sum: {innermost_dim_sum}, next_spatial_unroll: {next_spatial_unroll}")
                vprint(f"tile sizes[dim] for dim in dim_sum: {[[self.tile_sizes[dim] for dim in dim_sum] for dim_sum in self.arch.coupling.in_coupling if dim_sum is not innermost_dim_sum]}")
                # elements per tile - however, if immediately following fanouts don't support the multicasting of only certain parts of tiles (selective multicast), shared between instances, each seeing a different step
                # of the moving window, then the window required by each instance needs to be accessed independently, making the final tiles at this level contain duplicate elements for those which can't be multicasted.
                # NOTE: moving window, the access patter deriving from a sum of indices, where the inner iterated one creates a window, which is slid forward by the outer iterated index.                
                ## divide by iterations on next spatial
                ## actual tile size on every fanout level, regardless of how many iterations are there on the next spatial 
                in_reads *= prod(distinct_values(
                    [self.tile_sizes[dim]//next_spatial_unroll[dim] for dim in dim_sum], 
                    [self.arch.getInStride(dim) for dim in dim_sum]
                ) * prod(next_spatial_unroll[dim] for dim in dim_sum) 
                for dim_sum in self.arch.coupling.in_coupling if dim_sum is not innermost_dim_sum)
                vprint(f"Level: {self.name}, Layer {0}: after Spatial, Input in_reads = {in_reads}, i = {i}, innermost_dim_sum = {innermost_dim_sum}")
            vprint(f"input remaining dim in actual_dataflow_per_layer[0][:{i+1}]: {actual_dataflow_per_layer[0][:i+1]}")
            vprint(f"actual_dataflow_per_layer[0]: {actual_dataflow_per_layer[0]}, i = {i}")
            for dim in actual_dataflow_per_layer[0][:i+1]:
                vprint(f"Level: {self.name}, Layer {0}: multiplying Input in_reads = {in_reads} by factor of dim {dim} with factor {self.factors.dimProduct(dim)}")
                in_reads *= self.factors.dimProduct(dim)
            vprint(f"Level: {self.name}, Layer {0}: final Input in_reads = {in_reads:,.0f}\n")    
        # stationarity calculation for weights
        w_reads = int(w_bp)
        per_layer_w_reads: dict[int, int] = {}
        for layer_id in range(num_layers):
            per_layer_w_reads[layer_id] = int(w_bp)
        if w_bp:
            vprint(f"\nInitialization Level: {self.name}: w_reads = {w_reads}, per_layer_w_reads = {per_layer_w_reads}")
            ## Handle each weight Layer separately      
            for layer_id in range(num_layers):
                layer_read = 1
                i = len(actual_dataflow_per_layer[layer_id]) - 1
                innermost_dim_sum = None
                if not self.next_is_compute:
                    skipped = False
                    vprint(f"actual_dataflow_per_layer[{layer_id}] = {actual_dataflow_per_layer[layer_id]}, flat_weight_coupling[{layer_id}] = {self.arch.coupling.getFlatWeightCoupling(layer_id)}")
                    vprint(f"tile size for each dim of actual_dataflow_per_layer[{layer_id}]: {[self.tile_sizes[dim] for dim in actual_dataflow_per_layer[layer_id]]}")
                    while i >= 0 and (actual_dataflow_per_layer[layer_id][i] not in self.arch.coupling.getFlatWeightCoupling(layer_id)):
                        i -= 1
                        skipped = True
                    innermost_dim_sum = self.arch.coupling.getDimSum('w', actual_dataflow_per_layer[layer_id][i], 2, layer_id) if i >= 0 else None
                    vprint(f"Level: {self.name}: innermost_dim_sum = {innermost_dim_sum}, i = {i}, actual_dataflow_per_layer[{layer_id}] = {actual_dataflow_per_layer[layer_id]}")
                    if innermost_dim_sum and (not all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) or (skipped and not self.multiple_reuses)):
                        innermost_dim_sum = None
                    if innermost_dim_sum:
                        layer_read *= distinct_values([self.factors.dimProduct(actual_dataflow_per_layer[layer_id][i])*self.tile_sizes[actual_dataflow_per_layer[layer_id][i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id][i]], [self.arch.getWStride(actual_dataflow_per_layer[layer_id][i], layer_id)] + [self.arch.getWStride(dim, layer_id) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id][i]])
                        i -= 1
                vprint(f"weight layer {layer_id} read after innermost_dim_sum handling: {layer_read}")
                if not self.next_spatials:
                    vprint(f"innermost_dim_sum: {innermost_dim_sum}, dim_sum: dim_sum in self.arch.coupling.getWeightCoupling({layer_id}) if dim_sum is not innermost_dim_sum: {[dim_sum for dim_sum in self.arch.coupling.getWeightCoupling(layer_id) if dim_sum is not innermost_dim_sum]}")
                    vprint(f"self.tile_size[dim] for dim in dim_sum: {[[self.tile_sizes[dim] for dim in dim_sum] for dim_sum in self.arch.coupling.getWeightCoupling(layer_id) if dim_sum is not innermost_dim_sum]}")
                    layer_read *= prod(distinct_values([self.tile_sizes[dim] for dim in dim_sum], [self.arch.getWStride(dim, layer_id) for dim in dim_sum]) for dim_sum in self.arch.coupling.getWeightCoupling(layer_id) if dim_sum is not innermost_dim_sum)
                    vprint(f"weight layer {layer_id} read after no next_spatials handling: {layer_read}")
                else:
                    next_spatial_unroll = {dim: prod(level.factors.dimProduct(dim) for level in self.next_spatials if not level.selective_multicast_support) for dim in self.arch.coupling.getFlatWeightCoupling(layer_id)}
                    vprint(f"innermost_dim_sum: {innermost_dim_sum}, next_spatial_unroll: {next_spatial_unroll}")
                    vprint(f"tile sizes[dim] for dim in dim_sum: {[[self.tile_sizes[dim] for dim in dim_sum] for dim_sum in self.arch.coupling.getWeightCoupling(layer_id) if dim_sum is not innermost_dim_sum]}")
                    layer_read *= prod(distinct_values([self.tile_sizes[dim]//next_spatial_unroll[dim] for dim in dim_sum], [self.arch.getWStride(dim, layer_id) for dim in dim_sum])*prod(next_spatial_unroll[dim] for dim in dim_sum) for dim_sum in self.arch.coupling.getWeightCoupling(layer_id) if dim_sum is not innermost_dim_sum)
                vprint(f"weight layer {layer_id} read after next_spatials handling: {layer_read}")
                vprint(f"weight remaining dim in actual_dataflow_per_layer[{layer_id}][:{i+1}]: {actual_dataflow_per_layer[layer_id][:i+1]}")
                vprint(f"actual_dataflow_per_layer[{layer_id}]: {actual_dataflow_per_layer[layer_id]}, i = {i}")
                for dim in actual_dataflow_per_layer[layer_id][:i+1]:
                    layer_read *= self.factors.dimProduct(dim)
                vprint(f"\nLevel: {self.name}, Layer {layer_id}: final Weight layer_read = {layer_read:,.0f}\n\n")    
                per_layer_w_reads[layer_id] = layer_read
            w_reads = sum(per_layer_w_reads.values())
        ## ATTENTION: actual_dataflow_per_layer[layer_id+1] is the one for intermediate_inputs
        # stationarity calculation for intermediates inputs
        int_in_reads = int(int_in_bp)
        per_layer_int_in_reads: dict[int, int] = {}
        # Initialize per_layer_in_reads_bp for each layer
        for layer_id in range(num_layers-1):
            per_layer_int_in_reads[layer_id] = int(int_in_bp)
        if int_in_bp and per_layer_int_in_reads:
            for layer_id in range(num_layers - 1):
                vprint(f"\nInitialization Level: {self.name}, Layer {layer_id}: int_in_reads = {int_in_reads}, per_layer_int_in_reads = {per_layer_int_in_reads}")
                ## Handle each intermediate Layer separately
                layer_read = 1
                i = len(actual_dataflow_per_layer[layer_id + 1]) - 1
                vprint(f"i = {i} for actual_dataflow_per_layer[{layer_id + 1}] = {actual_dataflow_per_layer[layer_id + 1]}")
                innermost_dim_sum = None
                if not self.next_is_compute:
                    skipped = False
                    vprint(f"actual_dataflow_per_layer[{layer_id+1}] = {actual_dataflow_per_layer[layer_id + 1]}, flat_intermediate_input_coupling[{layer_id + 1}] = {self.arch.coupling.getFlatIntermediateInputCoupling(layer_id)}")
                    vprint(f"tile size for each dim of actual_dataflow_per_layer[{layer_id + 1}]: {[self.tile_sizes[dim] for dim in actual_dataflow_per_layer[layer_id + 1]]}")
                    while i >= 0 and (actual_dataflow_per_layer[layer_id + 1][i] not in self.arch.coupling.getFlatIntermediateInputCoupling(layer_id)):
                        i -= 1
                        skipped = True
                    innermost_dim_sum = self.arch.coupling.getDimSum('int_in', actual_dataflow_per_layer[layer_id + 1][i], 2, layer_id) if i >= 0 else None
                    vprint(f"Level: {self.name}: innermost_dim_sum = {innermost_dim_sum}, i = {i}, actual_dataflow = {actual_dataflow}")
                    if innermost_dim_sum and (not all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) or (skipped and not self.multiple_reuses)):
                        innermost_dim_sum = None
                    if innermost_dim_sum:
                        vprint(f"self.factors.dimProduct(actual_dataflow_per_layer[{layer_id + 1}][{i}]): {self.factors.dimProduct(actual_dataflow_per_layer[layer_id + 1][i])}, self.tile_sizes[actual_dataflow_per_layer[{layer_id + 1}][{i}]]: {self.tile_sizes[actual_dataflow_per_layer[layer_id + 1][i]]}")
                        vprint(f"[self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[{layer_id + 1}][{i}]]: {[self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id + 1][i]]}")
                        layer_read *= distinct_values([self.factors.dimProduct(actual_dataflow_per_layer[layer_id + 1][i])*self.tile_sizes[actual_dataflow_per_layer[layer_id + 1][i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id + 1][i]], [self.arch.getIntermediateInputStride(actual_dataflow_per_layer[layer_id + 1][i], layer_id)] + [self.arch.getIntermediateInputStride(dim, layer_id) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id + 1][i]])
                        i -= 1
                vprint(f"layer read after innermost_dim_sum handling: {layer_read}")
                if not self.next_spatials:
                    vprint(f"Self.next_spatials: {self.next_spatials}")
                    vprint(f"innermost_dim_sum: {innermost_dim_sum}, dim_sum: dim_sum in self.arch.coupling.getIntermediateInputCoupling({layer_id}) if dim_sum is not innermost_dim_sum: {[dim_sum for dim_sum in self.arch.coupling.getIntermediateInputCoupling(layer_id) if dim_sum is not innermost_dim_sum]}")
                    vprint(f"self.tile_size[dim] for dim in dim_sum: {[[self.tile_sizes[dim] for dim in dim_sum] for dim_sum in self.arch.coupling.getIntermediateInputCoupling(layer_id) if dim_sum is not innermost_dim_sum]}")
                    layer_read *= prod(distinct_values([self.tile_sizes[dim] for dim in dim_sum], [self.arch.getIntermediateInputStride(dim, layer_id) for dim in dim_sum]) for dim_sum in self.arch.coupling.getIntermediateInputCoupling(layer_id) if dim_sum is not innermost_dim_sum)
                    vprint(f"intermediate in layer {layer_id} read after no next_spatials handling: {layer_read}")
                else:
                    next_spatial_unroll = {dim: prod(level.factors.dimProduct(dim) for level in self.next_spatials if not level.selective_multicast_support) for dim in self.arch.coupling.getFlatIntermediateInputCoupling(layer_id)}
                    layer_read *= prod(distinct_values([self.tile_sizes[dim]//next_spatial_unroll[dim] for dim in dim_sum], [self.arch.getIntermediateInputStride(dim, layer_id) for dim in dim_sum])*prod(next_spatial_unroll[dim] for dim in dim_sum) for dim_sum in self.arch.coupling.getIntermediateInputCoupling(layer_id) if dim_sum is not innermost_dim_sum)
                    vprint(f"innermost_dim_sum: {innermost_dim_sum}, next_spatial_unroll: {next_spatial_unroll}")
                    vprint(f" dims in dim_sum: {[dim_sum for dim_sum in self.arch.coupling.getIntermediateInputCoupling(layer_id) if dim_sum is not innermost_dim_sum]}")
                    vprint(f"tile sizes[dim] for dim in dim_sum: {[[self.tile_sizes[dim] for dim in dim_sum] for dim_sum in self.arch.coupling.getIntermediateInputCoupling(layer_id) if dim_sum is not innermost_dim_sum]}")
                    vprint(f"intermediate in layer {layer_id} read after next_spatials handling: {layer_read}")
                vprint(f"Level {self.name}: intermediate in remaining dim in actual_dataflow_per_layer[{layer_id + 1}][:{i+1}]: {actual_dataflow_per_layer[layer_id + 1][:i+1]}")
                ## DOUBT
                for dim in actual_dataflow_per_layer[layer_id + 1][:i+1]:
                    layer_read *= self.factors.dimProduct(dim)
                vprint(f"intermediate in layer {layer_id} final read: {layer_read:,.0f}")
                per_layer_int_in_reads[layer_id] = layer_read
            int_in_reads = sum(per_layer_int_in_reads.values())
        # stationarity calculation for intermediate outputs
        int_out_reads = int(int_out_bp)
        per_layer_int_out_reads: dict[int, int] = {}
        # Initialize per-layer intermediate output reads for layers that have intermediate outputs
        for layer_id in range(num_layers - 1):
            if self.arch.coupling.getIntermediateOutputCoupling(layer_id):
                per_layer_int_out_reads[layer_id] = int(int_out_bp)
        if int_out_bp and per_layer_int_out_reads:
            ## Handle each intermediate output layer separately    
            for layer_id in range(num_layers - 1):               
                layer_read = 1
                i = len(actual_dataflow_per_layer[layer_id]) - 1
                innermost_dim_sum = None
                vprint(f"\nInitialization Level: {self.name}, Layer {layer_id}: int_out_reads = {int_out_reads}, per_layer_int_out_reads = {per_layer_int_out_reads}")
                if not self.next_is_compute:
                    skipped = False
                    vprint(f"actual_dataflow_per_layer[{layer_id}] = {actual_dataflow_per_layer[layer_id]}, flat_intermediate_output_coupling[{layer_id}] = {self.arch.coupling.getFlatIntermediateOutputCoupling(layer_id)}")
                    vprint(f"tile size for each dim of actual_dataflow_per_layer[{layer_id}]: {[self.tile_sizes[dim] for dim in actual_dataflow_per_layer[layer_id]]}")
                    while i >= 0 and (actual_dataflow_per_layer[layer_id][i] not in self.arch.coupling.getFlatIntermediateOutputCoupling(layer_id)):
                        i -= 1
                        skipped = True
                    innermost_dim_sum = self.arch.coupling.getDimSum('int_out', actual_dataflow_per_layer[layer_id][i], 2, layer_id) if i >= 0 else None
                    vprint(f"Level: {self.name}: innermost_dim_sum = {innermost_dim_sum}, i = {i}, actual_dataflow = {actual_dataflow_per_layer[layer_id]}") 
                    if innermost_dim_sum and (not all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) or (skipped and not self.multiple_reuses)):
                        innermost_dim_sum = None
                    if innermost_dim_sum:
                        layer_read *= distinct_values([self.factors.dimProduct(actual_dataflow_per_layer[layer_id][i])*self.tile_sizes[actual_dataflow_per_layer[layer_id][i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id][i]], [self.arch.getIntermediateOutputStride(actual_dataflow_per_layer[layer_id][i], layer_id)] + [self.arch.getIntermediateOutputStride(dim, layer_id) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id][i]])
                        i -= 1
                vprint(f"layer read after innermost_dim_sum handling: {layer_read}")                        
                if not self.next_spatials:
                    vprint(f"innermost_dim_sum: {innermost_dim_sum}, dim_sum: dim_sum in self.arch.coupling.getIntermediateOutputCoupling({layer_id}) if dim_sum is not innermost_dim_sum: {[dim_sum for dim_sum in self.arch.coupling.getIntermediateOutputCoupling(layer_id) if dim_sum is not innermost_dim_sum]}")
                    vprint(f"self.tile_size[dim] for dim in dim_sum: {[[self.tile_sizes[dim] for dim in dim_sum] for dim_sum in self.arch.coupling.getIntermediateOutputCoupling(layer_id) if dim_sum is not innermost_dim_sum]}")
                    layer_read *= prod(distinct_values([self.tile_sizes[dim] for dim in dim_sum], [self.arch.getIntermediateOutputStride(dim, layer_id) for dim in dim_sum]) for dim_sum in self.arch.coupling.getIntermediateOutputCoupling(layer_id) if dim_sum is not innermost_dim_sum)
                    vprint(f"intermediate out layer {layer_id} read after no next_spatials handling: {layer_read}")
                else:                   
                    next_spatial_unroll = {dim: prod(level.factors.dimProduct(dim) for level in self.next_spatials if not level.selective_multicast_support) for dim in self.arch.coupling.getFlatIntermediateOutputCoupling(layer_id)}
                    layer_read *= prod(distinct_values([self.tile_sizes[dim]//next_spatial_unroll[dim] for dim in dim_sum], [self.arch.getIntermediateOutputStride(dim, layer_id) for dim in dim_sum])*prod(next_spatial_unroll[dim] for dim in dim_sum) for dim_sum in self.arch.coupling.getIntermediateOutputCoupling(layer_id) if dim_sum is not innermost_dim_sum)
                    vprint(f"innermost_dim_sum: {innermost_dim_sum}, next_spatial_unroll: {next_spatial_unroll}")
                    vprint(f"tile sizes[dim] for dim in dim_sum: {[[self.tile_sizes[dim] for dim in dim_sum] for dim_sum in self.arch.coupling.getIntermediateOutputCoupling(layer_id) if dim_sum is not innermost_dim_sum]}")
                    vprint(f"intermediate out layer {layer_id} read after next_spatials handling: {layer_read}")
                vprint(f"intermediate out remaining dim in actual_dataflow_per_layer[{layer_id}][:{i+1}]: {actual_dataflow_per_layer[layer_id][:i+1]}")
                for dim in actual_dataflow_per_layer[layer_id][:i+1]:
                    layer_read *= self.factors.dimProduct(dim)
                vprint(f"\nintermediate out layer {layer_id} final read: {layer_read:,.0f}")
                per_layer_int_out_reads[layer_id] = layer_read
            int_out_reads = sum(per_layer_int_out_reads.values())
        # Intermediate output writes (same as reads for now)
        int_out_writes = int_out_reads
        per_layer_int_out_writes = per_layer_int_out_reads.copy()    
        # stationarity calculation for outputs        
        out_reads = int(out_bp)
        ## iterations orthogonal to the output, handle the presence/absence of the bias
        out_reads_factors = out_reads 
        if out_bp:
            out_layer_id = self.arch.coupling.getNumLayers() - 1
            i = len(actual_dataflow_per_layer[out_layer_id]) - 1
            innermost_dim_sum = None
            vprint(f"\nInitialization Level: {self.name}, Layer {out_layer_id}: out_reads = {out_reads:,.0f}, innermost_dim_sum = {innermost_dim_sum}, i = {i}, actual_dataflow = {actual_dataflow_per_layer[out_layer_id]}")
            if not self.next_is_compute:
                skipped = False
                # DEBUG print(f"CIAO i = {i}")
                while i >= 0 and (actual_dataflow_per_layer[out_layer_id][i] not in self.arch.coupling.getFlatOutputCoupling()):
                    # DEBUG print(f"Skipping dim {actual_dataflow_per_layer[out_layer_id][i]} not in output coupling {self.arch.coupling.getFlatOutputCoupling()}")
                    i -= 1
                    skipped = True
                ## DIFFERENCE but should be ok
                innermost_dim_sum = self.arch.coupling.getDimSum('out', actual_dataflow_per_layer[out_layer_id][i], 2) if i >= 0 else None
                if innermost_dim_sum and (not all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) or (skipped and not self.multiple_reuses)):
                    innermost_dim_sum = None
                if innermost_dim_sum:
                    out_reads *= distinct_values([self.factors.dimProduct(actual_dataflow_per_layer[out_layer_id][i])*self.tile_sizes[actual_dataflow_per_layer[out_layer_id][i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[out_layer_id][i]], [self.arch.getOutStride(actual_dataflow_per_layer[out_layer_id][i])] + [self.arch.getOutStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[out_layer_id][i]])
                    i -= 1
            vprint(f"out layer read after innermost_dim_sum handling: {out_reads:,.0f}")
            if not self.next_spatials:
                vprint(f"innermost_dim_sum: {innermost_dim_sum}, dim_sum: dim_sum in self.arch.coupling.out_coupling if dim_sum is not innermost_dim_sum: {[dim_sum for dim_sum in self.arch.coupling.out_coupling if dim_sum is not innermost_dim_sum]}")
                vprint(f"self.tile_size[dim] for dim in dim_sum: {[[self.tile_sizes[dim] for dim in dim_sum] for dim_sum in self.arch.coupling.out_coupling if dim_sum is not innermost_dim_sum]}")  
                out_reads *= prod(distinct_values([self.tile_sizes[dim] for dim in dim_sum], [self.arch.getOutStride(dim) for dim in dim_sum]) for dim_sum in self.arch.coupling.out_coupling if dim_sum is not innermost_dim_sum)
                vprint(f"output layer {out_layer_id} read after no next_spatials handling: {out_reads}")
            else:
                next_spatial_unroll = {dim: prod(level.factors.dimProduct(dim) for level in self.next_spatials if not level.selective_multicast_support) for dim in self.arch.coupling.getFlatOutputCoupling()}
                out_reads *= prod(distinct_values([self.tile_sizes[dim]//next_spatial_unroll[dim] for dim in dim_sum], [self.arch.getOutStride(dim) for dim in dim_sum])*prod(next_spatial_unroll[dim] for dim in dim_sum) for dim_sum in self.arch.coupling.getOutputCoupling() if dim_sum is not innermost_dim_sum)
                vprint(f"innermost_dim_sum: {innermost_dim_sum}, next_spatial_unroll: {next_spatial_unroll}")
                vprint(f"tile sizes[dim] for dim in dim_sum: {[[self.tile_sizes[dim] for dim in dim_sum] for dim_sum in self.arch.coupling.getOutputCoupling() if dim_sum is not innermost_dim_sum]}")
                vprint(f"output layer {out_layer_id} read after next_spatials handling: {out_reads}")
            vprint(f"out remaining dim in actual_dataflow_per_layer[{out_layer_id}][:{i+1}]: {actual_dataflow_per_layer[out_layer_id][:i+1]}")
            vprint(f"actual_dataflow_per_layer[{out_layer_id}]: {actual_dataflow_per_layer[out_layer_id]}, i = {i}")
            for dim in actual_dataflow_per_layer[out_layer_id][:i+1]:
                out_reads *= self.factors.dimProduct(dim)
                if dim not in self.arch.coupling.getFlatOutputCoupling():
                    # DEBUG print(f"Multiplying out_reads by factor of dim {dim} with factor {self.factors.dimProduct(dim)}")
                    out_reads_factors *= self.factors.dimProduct(dim)
            vprint(f"\noutput layer {out_layer_id} final read: {out_reads}, out_reads_factors (orthogonal to output) = {out_reads_factors}")
        out_writes = out_reads

        # vprint bypasses
        vprint(f"\nBEFORE BYPASS: ignore_bypasses: {ignore_bypasses}," +
              f" In reads: {in_reads:,.0f}, Per Layer W Reads: {per_layer_w_reads}, w_reads: {w_reads}," +
              f"\n Per Layer Intermediate Input reads: {per_layer_int_in_reads}, int_in_reads: {int_in_reads}, Per Layer Intermediate Output reads: {per_layer_int_out_reads}, int_out_reads: {int_out_reads}, Out reads: {out_reads}")

        # handle bypasses
        if not ignore_bypasses:        
            vprint("I AM HANDLING BYPASS NOW")    
            for operand, levels in self.next_levels_with_bypass.items():
                vprint(f"\n\nLevel: {self.name}: Operand = {operand}")
                if levels != None:
                    # 'level' is the level immediately above the one that stores the bypassed operand after the self level
                    ## in_between == all in between levels bypassed                     
                    in_between, level = levels[:-1], levels[-1]
                    vprint(f"Final level name before not bypassing: {level.name}\nI call MOPs on {level.name}")
                    ## True at the ignore_bypasses attribute => no recursive call to this line
                    in_reads_bp, per_layer_w_reads_bp, per_layer_int_in_reads_bp, per_layer_int_out_reads_bp, per_layer_int_out_writes_bp, out_reads_bp, out_writes_bp, out_reads_bp_factors = level.MOPs(operand == 'in', operand == 'w', operand == 'out', operand == 'int_in', operand == 'int_out', True)
                    w_reads_bp = sum(per_layer_w_reads_bp.values())
                    int_in_reads_bp = sum(per_layer_int_in_reads_bp.values())
                    int_out_reads_bp = sum(per_layer_int_out_reads_bp.values())
                    int_out_writes_bp = sum(per_layer_int_out_writes_bp.values())
                    vprint(f"After non recursive call for Level: {level.name}: MOPs for per_layer_int_in_reads_bp={per_layer_int_in_reads_bp} in_bp={in_reads_bp}, per_layer_w_reads_bp={per_layer_w_reads_bp} w_bp={w_reads_bp}, out_bp={out_reads_bp}, ignore_bypasses={True} (bypass operand {operand})")
                    # build the inner-most dataflow, by piling one against the other all non-1 loops, then look at which is the innermost dimension, that is the one that matters!
                    # TL;DR: consider the dataflow only w.r.t. the innermost loop, ignoring those with 1 iteration!
                    
                    # True = none of dim of this operand level's dataflow have n_iterations > 1
                    # False = at least one dim has n_iterations > 1
                    stationarity_to_address = not (any(level.factors.dimProduct(dim) > 1 and dim in self.arch.coupling.flatCouplingByOperand(operand) for dim in level.dataflow) if self.multiple_reuses else any(level.factors.dimProduct(dim) > 1 for dim in level.dataflow))
                    ## DEBUG
                    if stationarity_to_address == False:
                        vprint(f"stationarity false due to:")
                        for dim in level.dataflow:
                            if level.factors.dimProduct(dim) > 1:
                                if dim in self.arch.coupling.flatCouplingByOperand(operand):
                                    vprint(f" dim {dim} with factor {level.factors.dimProduct(dim)} is in flatCouplingByOperand({operand})")
                    # DEBUG print(f"Level: {self.name}: stationarity_to_address for operand {operand} = {stationarity_to_address}")
                    # DEBUG print(f"level.factors.dimProduct for level {level.name} dataflow: {[level.factors.dimProduct(dim) for dim in level.dataflow]}, level.dataflow: {level.dataflow}, flatCouplingByOperand: {self.arch.coupling.flatCouplingByOperand(operand)}")
                    # I'm going from the down (the bottom is Compute Level) to the top (current processing level)
                    vprint("Start in_btwn for")
                    vprint(f"I'm going from the down (Compute) to the top")
                    for in_btwn in in_between[::-1]:
                        vprint(f"Level: {self.name}: in_btwn = {in_btwn.name}, stationarity_to_address = {stationarity_to_address}, in_reads_bp = {in_reads_bp}, w_reads_bp = {w_reads_bp}, out_reads_bp = {out_reads_bp}, ignore_bypasses = {ignore_bypasses}")
                        # get the actual dataflow for the in_btwn level, filtering out dimensions with only one iteration
                        actual_dataflow_bp = list(filter(lambda dim : in_btwn.factors.dimProduct(dim) > 1, in_btwn.dataflow))
                        actual_dataflow_per_layer_bp: dict[int, list[str]] = {}
                        if self.arch.coupling.getNumLayers() > 1:        
                            for layer_id in range(self.arch.coupling.getNumLayers()):
                                # For each layer, filter dimensions that have loops > 1 AND are relevant to that layer
                                layer_relevant_dims = self.arch.coupling.relevantDimsForLayer(layer_id)
                                actual_dataflow_per_layer_bp[layer_id] = [
                                    dim for dim in actual_dataflow_bp
                                    if dim in layer_relevant_dims
                                ]
                        else:
                            actual_dataflow_per_layer_bp[0] = actual_dataflow_bp
                        if isinstance(in_btwn, MemLevel):
                            # DEBUG print(f"actual_dataflow_bp for level {in_btwn.name} = {actual_dataflow_bp}")
                            # DEBUG print(f"actual_dataflow_per_layer_bp for level {in_btwn.name} = {actual_dataflow_per_layer_bp}")
                            ## actual_dataflow_bp for the in_btwn level
                            # ignore loops at one
                            # precompute the full factors product per layer for the in_btwn level
                            in_btwn_factors_full_per_layer: dict[int, int] = {}
                            for layer_id in range(num_layers):
                                in_btwn_factors_full_per_layer[layer_id] = 1
                                for dim in actual_dataflow_per_layer_bp[layer_id]:
                                    in_btwn_factors_full_per_layer[layer_id] *= in_btwn.factors.dimProduct(dim)                            
                            # Handle Input Bypasses
                            if in_reads_bp:
                                ## None DIM of input of the above's in_btwn (and level) dataflows have > 1 iteration
                                if stationarity_to_address:
                                    # all inner loops were 1s or orthogonal, deal with the dataflow now!              
                                    i = len(actual_dataflow_per_layer_bp[0]) - 1
                                    while i >= 0 and (actual_dataflow_per_layer_bp[0][i] not in self.arch.coupling.getFlatInputCoupling()):
                                        i -= 1
                                    ## SINCE NOW stationarity_to_address is False if there is IN in df
                                    ## TRUE if: None DIM of actual_df in IN > 1 iterations
                                    ## FALSE if: Any DIM of actual_df in IN > 1 iterations
                                    stationarity_to_address = i < 0 # postpone stationarity evaluation since not input-coupled dimension is iterated here
                                    vprint(f"in_btwn name: {in_btwn.name}, Handling Input Bypass with stationarity_to_address = {stationarity_to_address}")
                                    # dimensions that partake in a sum of indices for the input operand together with the innermost iterated dimension on the currently traversed level between bypasses
                                    innermost_dim_sum = self.arch.coupling.getDimSum('in', actual_dataflow_per_layer_bp[0][i], 2) if i >= 0 else None                                    
                                    vprint(f"Level: {self.name}: innermost_dim_sum = {innermost_dim_sum}, i = {i}, actual_dataflow_per_layer_bp[0] = {actual_dataflow_per_layer_bp[0]}")

                                    ## ALL dim in sliding window have NO spatial parallelization in ANY spatial level below
                                    if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in in_btwn.next_spatials) and (i == len(actual_dataflow_per_layer_bp[0]) - 1 or self.multiple_reuses)):
                                        # before updating the tile size, remove from the original one the component relative
                                        # to the dimension involved in the sum of indices
                                        # WARNING: this ignores potential reuse in the case in which (e.g.) R is iterated immediately around P, 
                                        # and part of the input can be reused on inner levels
                                        ## remove the tile size
                                        ## conceptually divide by the level.tile_sizes[dim] should be equal
                                        in_reads_bp //= distinct_values([in_btwn.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getInStride(dim) for dim in innermost_dim_sum])
                                        in_reads_bp *= distinct_values([in_btwn.factors.dimProduct(actual_dataflow_per_layer_bp[0][i])*in_btwn.tile_sizes[actual_dataflow_per_layer_bp[0][i]]] + [in_btwn.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer_bp[0][i]], [self.arch.getInStride(actual_dataflow_per_layer_bp[0][i])] + [self.arch.getInStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer_bp[0][i]])
                                        i -= 1
                                    vprint(f"Remaining dim: actual_dataflow_per_layer_bp[0][:i+1] = {actual_dataflow_per_layer_bp[0][:i+1]}")
                                    for dim in actual_dataflow_per_layer_bp[0][:i+1]:
                                        in_reads_bp *= in_btwn.factors.dimProduct(dim)
                                    #vprint(f"Mem Level: {in_btwn}  Layer {layer_id} In reads Bypass: {layer_read}")
                                    #vprint(f"Total In reads Bypass: {in_reads_bp}")
                                else:
                                    # dataflow already handled among inner loops
                                    in_reads_bp = in_btwn_factors_full_per_layer[0]*in_reads_bp
                            # Handle Weight Bypasses
                            if w_reads_bp:
                                vprint(f"Stationarity to address from above w_reads_bp: {stationarity_to_address}")
                                if stationarity_to_address:
                                    for layer_id in range(num_layers):
                                        layer_read = per_layer_w_reads_bp[layer_id]
                                        i = len(actual_dataflow_per_layer_bp[layer_id]) - 1
                                        while i >= 0 and (actual_dataflow_per_layer_bp[layer_id][i] not in self.arch.coupling.getFlatWeightCoupling(layer_id)):
                                            i -= 1
                                        stationarity_to_address = i < 0
                                        vprint(f"in_btwn name: {in_btwn.name}, Handling Weight Bypass with stationarity_to_address = {stationarity_to_address}")
                                        innermost_dim_sum = self.arch.coupling.getDimSum('w', actual_dataflow_per_layer_bp[layer_id][i], 2, layer_id) if i >= 0 else None
                                        vprint(f"Level: {self.name}: innermost_dim_sum_w = {innermost_dim_sum}, i = {i}")
                                        if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in in_btwn.next_spatials) and (i == len(actual_dataflow_per_layer_bp[layer_id]) - 1 or self.multiple_reuses)):
                                            layer_read //= distinct_values([in_btwn.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getWStride(dim) for dim in innermost_dim_sum])
                                            layer_read *= distinct_values([in_btwn.factors.dimProduct(actual_dataflow_per_layer_bp[layer_id][i])*in_btwn.tile_sizes[actual_dataflow_per_layer_bp[layer_id][i]]] + [in_btwn.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer_bp[layer_id][i]], [self.arch.getWStride(actual_dataflow_per_layer_bp[layer_id][i])] + [self.arch.getWStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer_bp[layer_id][i]])
                                            i -= 1
                                        for dim in actual_dataflow_per_layer_bp[layer_id][:i+1]:
                                            layer_read *= in_btwn.factors.dimProduct(dim)
                                        per_layer_w_reads_bp[layer_id] = layer_read
                                    #vprint(f"per_layer_w_reads_bp = {per_layer_w_reads_bp}")
                                    #vprint(f"Total W reads: {w_reads_bp}")
                                    w_reads_bp = sum(per_layer_w_reads_bp.values())
                                else:
                                    vprint(f"Level: {self.name}: Handling Weight Bypass with stationarity_to_address = {stationarity_to_address}")
                                    for layer_id in range(num_layers):
                                        per_layer_w_reads_bp[layer_id] *= in_btwn_factors_full_per_layer[layer_id]
                                    w_reads_bp = sum(per_layer_w_reads_bp.values())                           
                            # Handle Intermediate Input Bypasses
                            if int_in_reads_bp:
                                # DEBUG print(f"per_layer_int_in_reads_bp before handling bypass: {per_layer_int_in_reads_bp}")
                                # DEBUG print(f"stationarity_to_address before handling int_in bypass: {stationarity_to_address}")
                                if stationarity_to_address:
                                    for layer_id in range(num_layers-1):
                                        layer_read = per_layer_int_in_reads_bp[layer_id]
                                        i = len(actual_dataflow_per_layer_bp[layer_id + 1]) - 1
                                        while i >= 0 and (actual_dataflow_per_layer_bp[layer_id + 1][i] not in self.arch.coupling.getFlatIntermediateInputCoupling(layer_id)):
                                            vprint(f"Skipping dim {actual_dataflow_per_layer_bp[layer_id + 1][i]} not in intermediate input coupling {self.arch.coupling.getFlatIntermediateInputCoupling(layer_id)}")
                                            i -= 1
                                        stationarity_to_address = i < 0
                                        innermost_dim_sum = self.arch.coupling.getDimSum('int_in', actual_dataflow_per_layer_bp[layer_id + 1][i], 2, layer_id) if i >= 0 else None
                                        if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in in_btwn.next_spatials) and (i == len(actual_dataflow_per_layer_bp[layer_id + 1]) - 1 or self.multiple_reuses)):
                                            layer_read //= distinct_values([in_btwn.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getIntermediateInputStride(dim, layer_id) for dim in innermost_dim_sum])
                                            layer_read *= distinct_values([in_btwn.factors.dimProduct(actual_dataflow_per_layer_bp[layer_id + 1][i]) * in_btwn.tile_sizes[actual_dataflow_per_layer_bp[layer_id + 1][i]]] + [in_btwn.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer_bp[layer_id + 1][i]], [self.arch.getIntermediateInputStride(actual_dataflow_per_layer_bp[layer_id + 1][i], layer_id)] + [self.arch.getIntermediateInputStride(dim, layer_id) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer_bp[layer_id + 1][i]])
                                            i -= 1
                                        # DEBUG print(f"Layer: {layer_id}: actual_dataflow_per_layer_bp[{layer_id+1}]: {actual_dataflow_per_layer_bp[layer_id + 1]} i after innermost= {i}")                                        
                                        for dim in actual_dataflow_per_layer_bp[layer_id + 1][:i+1]:
                                            # DEBUG print(f"Multiplying layer_read by factor of dim {dim} with factor {in_btwn.factors.dimProduct(dim)}")
                                            layer_read *= in_btwn.factors.dimProduct(dim)
                                        per_layer_int_in_reads_bp[layer_id] = layer_read
                                    # DEBUG print(f"per_layer_int_in_reads_bp after handling bypass: {per_layer_int_in_reads_bp}")
                                    int_in_reads_bp = sum(per_layer_int_in_reads_bp.values())
                                else:
                                    for layer_id in range(num_layers-1):
                                        per_layer_int_in_reads_bp[layer_id] *= in_btwn_factors_full_per_layer[layer_id]
                                    int_in_reads_bp = sum(per_layer_int_in_reads_bp.values())                            
                            # Handle intermediate output bypasses                         
                            if int_out_reads_bp:
                                # DEBUG print(f"\nper_layer_int_out_reads_bp before handling bypass: {per_layer_int_out_reads_bp}, per_layer_int_out_writes_bp before handling bypass: {per_layer_int_out_writes_bp}")
                                # DEBUG print(f"int_out_reads_bp: {sum(per_layer_int_out_reads_bp.values())}, int_out_writes: {sum(per_layer_int_out_writes_bp.values())}")
                                # DEBUG print(f"stationarity_to_address before handling int_out bypass: {stationarity_to_address}")
                                if stationarity_to_address:
                                    for layer_id in range(num_layers-1):
                                        layer_read = per_layer_int_out_reads_bp[layer_id]
                                        layer_write = per_layer_int_out_writes_bp[layer_id]
                                        i = len(actual_dataflow_per_layer_bp[layer_id]) - 1
                                        while i >= 0 and (actual_dataflow_per_layer_bp[layer_id][i] not in self.arch.coupling.getFlatIntermediateOutputCoupling(layer_id)):
                                            vprint(f"Skipping dim {actual_dataflow_per_layer_bp[layer_id][i]} not in intermediate output coupling {self.arch.coupling.getFlatIntermediateOutputCoupling(layer_id)}")
                                            i -= 1
                                        stationarity_to_address = i < 0
                                        innermost_dim_sum = self.arch.coupling.getDimSum('int_out', actual_dataflow_per_layer_bp[layer_id][i], 2, layer_id) if i >= 0 else None
                                        if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in in_btwn.next_spatials) and (i == len(actual_dataflow_per_layer_bp[layer_id]) - 1 or self.multiple_reuses)):
                                            dvs = distinct_values([in_btwn.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getIntermediateOutputStride(dim, layer_id) for dim in innermost_dim_sum])
                                            layer_read //= dvs
                                            layer_write //= dvs
                                            dvs = distinct_values([in_btwn.factors.dimProduct(actual_dataflow_per_layer_bp[layer_id][i])*in_btwn.tile_sizes[actual_dataflow_per_layer_bp[layer_id][i]]] + [in_btwn.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer_bp[layer_id][i]], [self.arch.getIntermediateOutputStride(actual_dataflow_per_layer_bp[layer_id][i], layer_id)] + [self.arch.getIntermediateOutputStride(dim, layer_id) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer_bp[layer_id][i]])
                                            layer_read *= dvs
                                            layer_write *= dvs
                                            i -= 1
                                        # DEBUG print(f"Layer: {layer_id}: actual_dataflow_per_layer_bp[{layer_id+1}]: {actual_dataflow_per_layer_bp[layer_id + 1]} i after innermost= {i}")                                        
                                        for dim in actual_dataflow_per_layer_bp[layer_id][:i+1]:
                                            # DEBUG print(f"Multiplying layer_read by factor of dim {dim} with factor {in_btwn.factors.dimProduct(dim)}")
                                            layer_read *= in_btwn.factors.dimProduct(dim)
                                            layer_write *= in_btwn.factors.dimProduct(dim)
                                        per_layer_int_out_reads_bp[layer_id] = layer_read
                                        per_layer_int_out_writes_bp[layer_id] = layer_write
                                    int_out_reads_bp = sum(per_layer_int_out_reads_bp.values())
                                    int_out_writes_bp = sum(per_layer_int_out_writes_bp.values())
                                    # DEBUG print(f"per_layer_int_out_reads_bp after handling bypass: {per_layer_int_out_reads_bp}, per_layer_int_out_writes_bp after handling bypass: {per_layer_int_out_writes_bp}")
                                else:
                                    for layer_id in range(num_layers-1):
                                        per_layer_int_out_reads_bp[layer_id] *= in_btwn_factors_full_per_layer[layer_id]
                                        per_layer_int_out_writes_bp[layer_id] *= in_btwn_factors_full_per_layer[layer_id]
                                    int_out_reads_bp = in_btwn_factors_full_per_layer[layer_id] * int_out_reads_bp
                                    int_out_writes_bp = in_btwn_factors_full_per_layer[layer_id] * int_out_writes_bp
                            ## senza bias => out_read < out_writes_bp
                            out_layer_id = num_layers - 1
                            if out_reads_bp:
                                if stationarity_to_address:
                                    i = len(actual_dataflow_per_layer_bp[out_layer_id]) - 1
                                    while i >= 0 and (actual_dataflow_per_layer_bp[out_layer_id][i] not in self.arch.coupling.getFlatOutputCoupling()):
                                        i -= 1
                                    stationarity_to_address = i < 0
                                    innermost_dim_sum = self.arch.coupling.getDimSum('out', actual_dataflow_per_layer_bp[out_layer_id][i], 2) if i >= 0 else None
                                    #vprint(f"Level: {self.name}: innermost_dim_sum_out = {innermost_dim_sum}, i = {i}, actual_dataflow = {actual_dataflow_bp}")
                                    if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in in_btwn.next_spatials) and (i == len(actual_dataflow_per_layer_bp[out_layer_id]) - 1 or self.multiple_reuses)):
                                        out_reads_bp, out_writes_bp = out_reads_bp//(dvs := distinct_values([in_btwn.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getOutStride(dim) for dim in innermost_dim_sum])), out_writes_bp//dvs
                                        out_reads_bp, out_writes_bp = out_reads_bp*(dvs := distinct_values([in_btwn.factors.dimProduct(actual_dataflow_per_layer_bp[out_layer_id][i])*in_btwn.tile_sizes[actual_dataflow_per_layer_bp[out_layer_id][i]]] + [in_btwn.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer_bp[out_layer_id][i]], [self.arch.getOutStride(actual_dataflow_per_layer_bp[out_layer_id][i])] + [self.arch.getOutStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer_bp[out_layer_id][i]])), out_writes_bp*dvs
                                        i -= 1
                                    for dim in actual_dataflow_per_layer_bp[out_layer_id][:i+1]:
                                        out_reads_bp *= in_btwn.factors.dimProduct(dim)
                                        out_writes_bp *= in_btwn.factors.dimProduct(dim)
                                        if dim not in self.arch.coupling.getFlatOutputCoupling():
                                            out_reads_bp_factors *= in_btwn.factors.dimProduct(dim)
                                else:
                                    out_reads_bp = in_btwn_factors_full_per_layer[num_layers-1]*out_reads_bp
                                    out_writes_bp = in_btwn_factors_full_per_layer[num_layers-1]*out_writes_bp
                                    out_reads_bp_factors *= prod(in_btwn.factors.dimProduct(dim) for dim in in_btwn.dataflow if dim not in self.arch.coupling.flat_out_coupling)
                            #in_btwn.bp_stationarity_solved_here[operand] = stationarity_to_address != old_stationarity_to_address
                        # Fanout
                        else:
                            vprint(f"in_btwn Level: {in_btwn.name}, FanoutLevel for operand {operand}")
                            ## mulByDim calculates MOPs for spatial levels
                            ## it multiplies the MOPs by the factors of the dimensions (i.e. how many parallel instances are there)
                            in_reads_bp, per_layer_w_reads_bp, per_layer_int_in_reads_bp, per_layer_int_out_reads_bp, per_layer_int_out_writes_bp, out_reads_bp, out_writes_bp = in_btwn.mulByDim(in_reads_bp, per_layer_w_reads_bp, per_layer_int_in_reads_bp, per_layer_int_out_reads_bp, per_layer_int_out_writes_bp, out_reads_bp, out_writes_bp)
#                            vprint(f"per_layer_in_reads_bp = {per_layer_in_reads_bp}, per_layer_w_reads_bp = {per_layer_w_reads_bp}")
                            w_reads_bp = sum(per_layer_w_reads_bp.values())
                            int_in_reads_bp = sum(per_layer_int_in_reads_bp.values())
                            int_out_reads_bp = sum(per_layer_int_out_reads_bp.values())
                            int_out_writes_bp = sum(per_layer_int_out_writes_bp.values())
                            vprint(f"After FanoutLevel {in_btwn.name} for operand {operand}: in_reads_bp={in_reads_bp}, per_layer_w_reads_bp={per_layer_w_reads_bp}, w_reads_bp={w_reads_bp}, out_reads_bp={out_reads_bp}, int_in_reads_bp={int_in_reads_bp}, int_out_reads_bp={int_out_reads_bp}, int_out_writes_bp={int_out_writes_bp}, ignore_bypasses={ignore_bypasses}")
                            # do not update out_reads_bp_factors here, because in it go only iterations of which the first one is skipped,
                            # while in a fanout all fanned-out copies of the inner loop behave the same, there isn't a first different spatial iteration or anything
                        #vprint("IN BETWEEN BYPASS:\n", f"{in_btwn.name}:{chr(9) * (2 - len(in_btwn.name)//8)}{in_reads_bp} In_R, {w_reads_bp} W_R, {out_reads_bp} Our_R, {in_reads_bp + w_reads_bp + out_reads_bp} Tot_R, {out_writes_bp} Out_W, {out_reads_bp_factors} Out_R_Fac")
                    ## now we can update the MOPs for the current level
                    ## factors_full is the product of all factors in the
                    ## current level
                    vprint("End in_btwn for")
                    vprint(f"Level: {self.name}: after all in_btwn levels, stationarity_to_address = {stationarity_to_address}, operand = {operand} in_reads_bp = {in_reads_bp}, w_reads_bp = {w_reads_bp}, out_reads_bp = {out_reads_bp}, int_in_reads_bp = {int_in_reads_bp}, int_out_reads_bp = {int_out_reads_bp}, int_out_writes_bp = {int_out_writes_bp}, ignore_bypasses = {ignore_bypasses}")
                    factors_full = self.factors.fullProduct()
                    factors_full_per_layer: dict[int, int] = {}
                    
                    for layer_id in range(num_layers):
                        factors_full_per_layer[layer_id] = 1
                        for dim in actual_dataflow_per_layer[layer_id]:
                            factors_full_per_layer[layer_id] *= self.factors.dimProduct(dim)
                    #self.bp_stationarity_solved_here[operand] = stationarity_to_address
                    vprint(f"Now I will do MyLevel: {self.name}: Final handling of bypassed operand {operand} with stationarity_to_address = {stationarity_to_address}")
                    vprint(f"in_reads_bp before final handling: {in_reads_bp}, w_reads_bp before final handling: {w_reads_bp}")
                    vprint(f"int_in_reads_bp before final handling: {int_in_reads_bp}, int_out_reads_bp before final handling: {int_out_reads_bp}, out_reads_bp before final handling: {out_reads_bp}")
                    if in_reads_bp:
                        ## if is true mean that starting from the last in_btwn: there is IN in 
                        ## actual_dataflow_db => stationarity_to_address is False  
                        ## SINCE NOW stationarity_to_address is False if there is IN in df
                        if stationarity_to_address:
                            # all inner loops were 1s or orthogonal, deal with the dataflow now!
                            i = len(actual_dataflow_per_layer[0]) - 1
                            #vprint(f"Level: {self.name}: flat_in_coupling[{layer_id}] = {flat_in_coupling[layer_id]}")
                            while i >= 0 and (actual_dataflow_per_layer[0][i] not in self.arch.coupling.getFlatInputCoupling()):
                                i -= 1
                            innermost_dim_sum = self.arch.coupling.getDimSum('in', actual_dataflow_per_layer[0][i], 2, layer_id) if i >= 0 else None
                            #vprint(f"Level: {self.name}: innermost_dim_sum_in_after_inbtwn = {innermost_dim_sum}, i = {i}, actual_dataflow = {actual_dataflow}")
                            if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) and (i == len(actual_dataflow_per_layer[0]) - 1 or self.multiple_reuses)):
                                in_reads_bp //= distinct_values([self.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getInStride(dim) for dim in innermost_dim_sum])
                                in_reads_bp *= distinct_values([self.factors.dimProduct(actual_dataflow_per_layer[0][i])*self.tile_sizes[actual_dataflow_per_layer[0][i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[0][i]], [self.arch.getInStride(actual_dataflow_per_layer[0][i])] + [self.arch.getInStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[0][i]])
                                i -= 1
                            for dim in actual_dataflow_per_layer[0][:i+1]:
                                in_reads_bp *= self.factors.dimProduct(dim)
                            #vprint(f"Mem Level: {level}")
                            #vprint(f"Layer {layer_id} In reads Bypass Current level: {layer_read_bp}")
                            #vprint(f"Total In reads Bypass: {in_reads_bp}")
                        else:
                            # dataflow handled among inner loops
                            in_reads_bp *= factors_full_per_layer[0]
                    if w_reads_bp:
                        if stationarity_to_address:
                            for layer_id in range(num_layers):
                                layer_w_reads_bp = per_layer_w_reads_bp[layer_id]
                                # all inner loops were 1s or orthogonal, deal with the dataflow now!
                                i = len(actual_dataflow_per_layer[layer_id]) - 1
                                while i >= 0 and (actual_dataflow_per_layer[layer_id][i] not in self.arch.coupling.getFlatWeightCoupling(layer_id)):
                                    i -= 1
                                innermost_dim_sum = self.arch.coupling.getDimSum('w', actual_dataflow_per_layer[layer_id][i], 2, layer_id) if i >= 0 else None
                                if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) and (i == len(actual_dataflow_per_layer[layer_id]) - 1 or self.multiple_reuses)):
                                    layer_w_reads_bp //= distinct_values([self.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getWStride(dim) for dim in innermost_dim_sum])
                                    layer_w_reads_bp *= distinct_values([self.factors.dimProduct(actual_dataflow_per_layer[layer_id][i])*self.tile_sizes[actual_dataflow_per_layer[layer_id][i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id][i]], [self.arch.getWStride(actual_dataflow_per_layer[layer_id][i])] + [self.arch.getWStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id][i]])
                                    i -= 1
                                vprint(f"layer = {layer_id}, layer_w_reads_bp before inner df = {layer_w_reads_bp}")
                                for dim in actual_dataflow_per_layer[layer_id][:i+1]:
                                    layer_w_reads_bp *= self.factors.dimProduct(dim)
                                vprint(f"Level: {self.name}: layer_id = {layer_id}, layer_w_reads_bp = {layer_w_reads_bp}")
                                per_layer_w_reads_bp[layer_id] = layer_w_reads_bp
                            w_reads_bp = sum(per_layer_w_reads_bp.values())
                        else:
                            vprint(f"Level: {self.name}: stationarity_to_address is False, dataflow handled among inner loops")
                            for layer_id in range(num_layers):
                                vprint(f"Before: per_layer_w_reads_bp[{layer_id}] = {per_layer_w_reads_bp[layer_id]}")
                                per_layer_w_reads_bp[layer_id] *= factors_full_per_layer[layer_id]
                                vprint(f"After: per_layer_w_reads_bp[{layer_id}] = {per_layer_w_reads_bp[layer_id]}")
                                w_reads_bp = sum(per_layer_w_reads_bp.values())
                    if int_in_reads_bp:
                        # DEBUG print(f"int_in_reads_bp before handling current level: {int_in_reads_bp}")
                        # DEBUG print(f"stationarity_to_address before handling int_in current level: {stationarity_to_address}")
                        if stationarity_to_address:
                            vprint(f"Level: {self.name}: Handling Intermediate Input with stationarity_to_address = {stationarity_to_address}")
                            for layer_id in range(num_layers - 1):
                                # DEBUG print(f"Actual dataflow per layer {layer_id + 1}: {actual_dataflow_per_layer[layer_id + 1]}")
                                # DEBUG print(f"getFlatIntermediateInputCoupling for layer {layer_id}: {self.arch.coupling.getFlatIntermediateInputCoupling(layer_id)}")
                                layer_int_in_reads_bp = per_layer_int_in_reads_bp[layer_id]
                                # all inner loops were 1s or orthogonal, deal with the dataflow now!
                                i = len(actual_dataflow_per_layer[layer_id + 1]) - 1
                                # DEBUG print(f"i start: {i}")
                                while i >= 0 and (actual_dataflow_per_layer[layer_id + 1][i] not in self.arch.coupling.getFlatIntermediateInputCoupling(layer_id)):
                                    i -= 1
                                # DEBUG print(f"i after while: {i}")
                                innermost_dim_sum = self.arch.coupling.getDimSum('int_in', actual_dataflow_per_layer[layer_id + 1][i], 2, layer_id) if i >= 0 else None
                                if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) and (i == len(actual_dataflow_per_layer[layer_id + 1]) - 1 or self.multiple_reuses)):
                                    layer_int_in_reads_bp //= distinct_values([self.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getIntermediateInputStride(dim, layer_id) for dim in innermost_dim_sum])
                                    layer_int_in_reads_bp *= distinct_values([self.factors.dimProduct(actual_dataflow_per_layer[layer_id + 1][i])*self.tile_sizes[actual_dataflow_per_layer[layer_id + 1][i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id + 1][i]], [self.arch.getIntermediateInputStride(actual_dataflow_per_layer[layer_id + 1][i], layer_id)] + [self.arch.getIntermediateInputStride(dim, layer_id) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id + 1][i]])
                                    i -= 1
                                # DEBUG print(f"actual_dataflow_per_layer[{layer_id + 1}][:{i+1}] = {actual_dataflow_per_layer[layer_id + 1][:i+1]}")
                                for dim in actual_dataflow_per_layer[layer_id + 1][:i+1]:
                                    layer_int_in_reads_bp *= self.factors.dimProduct(dim)
                                # DEBUG print(f"Level: {self.name}: layer_id = {layer_id}, layer_int_in_reads_bp = {layer_int_in_reads_bp}")    
                                per_layer_int_in_reads_bp[layer_id] = layer_int_in_reads_bp
                            int_in_reads_bp = sum(per_layer_int_in_reads_bp.values())
                        else:
                            vprint(f"Level: {self.name}: stationarity_int_in_to_address is False, dataflow handled among inner loops")
                            for layer_id in range(num_layers-1):
                                per_layer_int_in_reads_bp[layer_id] *= factors_full_per_layer[layer_id]
                            int_in_reads_bp *= factors_full_per_layer[layer_id]
                    if int_out_reads_bp:
                        # DEBUG print(f"\nint_out_reads_bp before handling current level: {int_out_reads_bp}, int_out_writes_bp before handling current level: {int_out_writes_bp}")
                        # DEBUG print(f"stationarity_to_address before handling int_out current level: {stationarity_to_address}")
                        if stationarity_to_address:
                            for layer_id in range(num_layers-1):
                                layer_int_out_reads_bp = per_layer_int_out_reads_bp[layer_id]
                                layer_int_out_writes_bp = per_layer_int_out_writes_bp[layer_id]
                                # all inner loops were 1s or orthogonal, deal with the dataflow now!
                                i = len(actual_dataflow_per_layer[layer_id]) - 1
                                while i >= 0 and (actual_dataflow_per_layer[layer_id][i] not in self.arch.coupling.getFlatIntermediateOutputCoupling(layer_id)):
                                    i -= 1
                                innermost_dim_sum = self.arch.coupling.getDimSum('int_out', actual_dataflow_per_layer[layer_id][i], 2, layer_id) if i >= 0 else None

                                if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) and (i == len(actual_dataflow_per_layer[layer_id]) - 1 or self.multiple_reuses)):
                                    dvs = distinct_values([self.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getIntermediateOutputStride(dim, layer_id) for dim in innermost_dim_sum])
                                    layer_int_out_reads_bp //= dvs
                                    layer_int_out_writes_bp //= dvs
                                    dvs = distinct_values([self.factors.dimProduct(actual_dataflow_per_layer[layer_id][i])*self.tile_sizes[actual_dataflow_per_layer[layer_id][i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id][i]], [self.arch.getIntermediateOutputStride(actual_dataflow_per_layer[layer_id][i], layer_id)] + [self.arch.getIntermediateOutputStride(dim, layer_id) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[layer_id][i]])
                                    layer_int_out_reads_bp *= dvs
                                    layer_int_out_writes_bp *= dvs
                                    i -= 1
                                for dim in actual_dataflow_per_layer[layer_id][:i+1]:
                                    layer_int_out_reads_bp *= self.factors.dimProduct(dim)
                                    layer_int_out_writes_bp *= self.factors.dimProduct(dim)
                                per_layer_int_out_reads_bp[layer_id] = layer_int_out_reads_bp
                                per_layer_int_out_writes_bp[layer_id] = layer_int_out_writes_bp
                            int_out_reads_bp = sum(per_layer_int_out_reads_bp.values())
                            int_out_writes_bp = sum(per_layer_int_out_writes_bp.values())
                        else:
                            vprint(f"Level: {self.name}: stationarity_int_out_to_address is False, dataflow handled among inner loops")
                            for layer_id in range(num_layers-1):
                                per_layer_int_out_reads_bp[layer_id] *= factors_full_per_layer[layer_id]
                                per_layer_int_out_writes_bp[layer_id] *= factors_full_per_layer[layer_id]
                            int_out_reads_bp *= factors_full_per_layer[layer_id]
                            int_out_writes_bp *= factors_full_per_layer[layer_id]
                    if out_reads_bp:
                        if stationarity_to_address:
                            out_layer_id = self.arch.coupling.getNumLayers() - 1
                            i = len(actual_dataflow_per_layer[out_layer_id]) - 1
                            while i >= 0 and (actual_dataflow_per_layer[out_layer_id][i] not in self.arch.coupling.flat_out_coupling):
                                i -= 1
                            ## DOUBT
                            innermost_dim_sum = self.arch.coupling.getDimSum('out', actual_dataflow_per_layer[out_layer_id][i], 2) if i >= 0 else None
                            #vprint(f"Level: {self.name}: innermost_dim_sum_out_after_inbtwn = {innermost_dim_sum}, i = {i}, actual_dataflow = {actual_dataflow}")
                            if innermost_dim_sum and (all(all(sp_level.factors.dimProduct(dim) == 1 for dim in innermost_dim_sum) for sp_level in self.next_spatials) and (i == len(actual_dataflow_per_layer[out_layer_id]) - 1 or self.multiple_reuses)):
                                out_reads_bp, out_writes_bp = out_reads_bp//(dvs := distinct_values([self.tile_sizes[dim] for dim in innermost_dim_sum], [self.arch.getOutStride(dim) for dim in innermost_dim_sum])), out_writes_bp//dvs
                                out_reads_bp, out_writes_bp = out_reads_bp*(dvs := distinct_values([self.factors.dimProduct(actual_dataflow_per_layer[out_layer_id][i])*self.tile_sizes[actual_dataflow_per_layer[out_layer_id][i]]] + [self.tile_sizes[dim] for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[out_layer_id][i]], [self.arch.getOutStride(actual_dataflow_per_layer[out_layer_id][i])] + [self.arch.getOutStride(dim) for dim in innermost_dim_sum if dim != actual_dataflow_per_layer[out_layer_id][i]])), out_writes_bp*dvs
                                i -= 1
                            for dim in actual_dataflow_per_layer[out_layer_id][:i+1]:
                                out_reads_bp *= self.factors.dimProduct(dim)
                                out_writes_bp *= self.factors.dimProduct(dim)
                                if dim not in self.arch.coupling.flat_out_coupling:
                                    out_reads_bp_factors *= self.factors.dimProduct(dim)
                        else:
                            out_reads_bp = factors_full_per_layer[out_layer_id]*out_reads_bp
                            out_writes_bp = factors_full_per_layer[out_layer_id]*out_writes_bp
                            out_reads_bp_factors *= prod(self.factors.dimProduct(dim) for dim in self.dataflow if dim not in self.arch.coupling.getFlatOutputCoupling())
                    vprint(f"Level: {self.name}: MOPs after bypasses for in_reads_bp = {in_reads_bp}, per_layer_w_reads_bp = {per_layer_w_reads_bp}, w_reads_bp = {w_reads_bp}, per_layer_int_in_reads_bp = {per_layer_int_in_reads_bp}, int_in_reads_bp = {int_in_reads_bp}, per_layer_int_out_reads_bp = {per_layer_int_out_reads_bp}, int_out_reads_bp = {int_out_reads_bp}, int_out_writes_bp = {int_out_writes_bp}, out_reads_bp = {out_reads_bp}, out_writes_bp = {out_writes_bp} (bypass operand {operand})")
                    for layer_id in range(num_layers-1):
                        # accumulate the reads for the current level but it is iterable
                        per_layer_int_in_reads[layer_id] += per_layer_int_in_reads_bp[layer_id]
                        per_layer_int_out_reads[layer_id] += per_layer_int_out_reads_bp[layer_id]
                        per_layer_int_out_writes[layer_id] += per_layer_int_out_writes_bp[layer_id]                        
                    for layer_id in range(num_layers):     
                        per_layer_w_reads[layer_id] += per_layer_w_reads_bp[layer_id]
                    in_reads += in_reads_bp
                    int_in_reads += sum(per_layer_int_in_reads_bp.values())
                    int_out_reads += sum(per_layer_int_out_reads_bp.values())
                    int_out_writes += sum(per_layer_int_out_writes_bp.values())
                    w_reads += sum(per_layer_w_reads_bp.values())
                    out_reads += out_reads_bp
                    out_writes += out_writes_bp
#                    vprint(f"Level: {self.name}: MOPs after bypasses for per_layer_in_reads = {per_layer_in_reads} in_bp={in_reads}, per_layer_w_reads = {per_layer_w_reads} w_bp={w_reads}, out_bp={out_reads}, ignore_bypasses={ignore_bypasses} (bypass operand {operand})")
                    if operand == 'out':
                        out_reads_factors = out_reads_bp_factors
                else:
                    vprint("No levels in between for bypass operand " + operand)
            # vprint the results of handling bypasses
            vprint("\nAFTER BYPASS of all operands: ", f"{self.name}: in_reads: {in_reads}, per_layer_w_reads: {per_layer_w_reads}, w_reads: {w_reads}, per_layer_int_in_reads: {per_layer_int_in_reads}, int_in_reads: {int_in_reads}, per_layer_int_out_reads: {per_layer_int_out_reads}, int_out_reads: {int_out_reads}, out_reads: {out_reads}, out_writes: {out_writes}, out_reads_factors: {out_reads_factors}\n")
        vprint(f"Start spatials for Level: {self.name}\n")
        for spatial_level in self.next_spatials:
            vprint(f"Level: {self.name}, handling spatial level: {spatial_level.name}, spatial_level_multicast_support = {spatial_level.spatial_multicast_support}, spatial_level_spatial_reduction_support = {spatial_level.spatial_reduction_support}")
            for dim in spatial_level.dataflow:
                # no multicast => copy n_levels times
                if not spatial_level.spatial_multicast_support and dim not in self.arch.coupling.getFlatInputCoupling(): # the next fanout doesn't have spatial multicast capabilities, increment the reads on the memory before such fanout
                        in_reads *= spatial_level.factors.dimProduct(dim)
                for layer_id in range(num_layers):
                    if not spatial_level.spatial_multicast_support and dim not in self.arch.coupling.getFlatWeightCoupling(layer_id): # the next fanout doesn't have spatial multicast capabilities, increment the reads on the memory before such fanout
                        per_layer_w_reads[layer_id] *= spatial_level.factors.dimProduct(dim)                            
                for layer_id in range(num_layers - 1):
                    if not spatial_level.spatial_multicast_support and dim not in self.arch.coupling.getFlatIntermediateInputCoupling(layer_id): # the next fanout doesn't have spatial multicast capabilities, increment the reads on the memory before such fanout
                        per_layer_int_in_reads[layer_id] *= spatial_level.factors.dimProduct(dim)                            
                    if not spatial_level.spatial_multicast_support and dim not in self.arch.coupling.getFlatIntermediateOutputCoupling(layer_id): # the next fanout doesn't have spatial multicast capabilities, increment the reads on the memory before such fanout
                        per_layer_int_out_reads[layer_id] *= spatial_level.factors.dimProduct(dim)                            
                        ## ATTENTION DOUBT spatial reduction?????? I don't think so
                        per_layer_int_out_writes[layer_id] *= spatial_level.factors.dimProduct(dim)                            
                if dim not in self.arch.coupling.getFlatOutputCoupling():
                    # no multicast for reading
                    if not spatial_level.spatial_multicast_support: # we don't have spatial multicast capabilities, increment retroactively the reads on the above level
                        out_reads *= spatial_level.factors.dimProduct(dim) # no need to account for bias_read here, as the first read is skipped by all instances of the fanout
                    # no spatial reduction for writing
                    if not spatial_level.spatial_reduction_support: # we don't have spatial reduction capabilities, increment retroactively the writes on the above level
                        out_writes *= spatial_level.factors.dimProduct(dim)
        vprint(f"After Spatials: Level: {self.name}: MOPs in_reads = {in_reads}, per_layer_w_reads = {per_layer_w_reads}, w_reads = {w_reads}, per_layer_int_in_reads = {per_layer_int_in_reads}, int_in_reads = {int_in_reads}, per_layer_int_out_reads = {per_layer_int_out_reads}, int_out_reads = {int_out_reads}, out_reads = {out_reads}, out_writes = {out_writes}, out_reads_factors = {out_reads_factors}\n")
        if not ignore_bypasses:
            vprint("End MOPs with Bypasses for Level: " + self.name + "\n\n")
        # Change the return in such a way I have also per_
        return in_reads, per_layer_w_reads, per_layer_int_in_reads, per_layer_int_out_reads, per_layer_int_out_writes, out_reads, out_writes, out_reads_factors

    """
    Returns the provided MOPs (or newly calculated MOPs for this level)
    scaled by the MOPs's weight/energy at this level.
    """
    def WMOPs(self, reads : Optional[int] = None, writes : Optional[int] = None) -> float:
        if  reads == None or writes == None:
            reads = reads if reads else self.in_reads + self.w_reads + self.int_in_reads + self.int_out_reads + self.out_reads
            writes = writes if writes else self.in_writes + self.w_writes + self.int_in_writes + self.int_out_writes + self.out_writes
        return self.read_access_energy * reads + self.write_access_energy * writes

    """
    Returns the total leaked energy during the provided clock cycles.
    """
    def Leakage(self, cycles : int) -> float:
        return cycles*self.leakage_energy

    """
    Returns True iif factors present on this level satisfy all of
    its constraints, including fitting in the available memory.
    """
    def checkFactorsConstraints(self) -> bool:
        # Existing memory footprint check
        # DEBUG print(f"Level: {self.name}: Checking factors constraints...")
        base_check = self.factors.memFootprint(self.tile_sizes, self.arch, not self.bypasses or 'in' not in self.bypasses, not self.bypasses or 'w' not in self.bypasses, not self.bypasses or 'out' not in self.bypasses, not self.bypasses or 'int_in' not in self.bypasses, not self.bypasses or 'int_out' not in self.bypasses) <= self.size/self.multiple_buffering and super().checkFactorsConstraints()
        if not base_check:
            return False
        """    
        ## FIX THIS    
        # Layer fusion constraint: no iterations on intermediate layer dimensions (except last layer)
        if hasattr(self.arch.coupling, 'w_coupling') and len(self.arch.coupling.w_coupling) > 1:
            # Get the number of layers
            num_layers = self.arch.coupling.getNumLayers()

            forbidden_dims = set()
            #for i in range(num_layers -1):
            #    forbidden_dims.update([f'C{i+1}', f'R{i}', f'S{i}'])
        
            if num_layers > 1:
                ## # Add C1, cause is an intermediate
                forbidden_dims.add('C1')  

            compute_index = next(i for i, level in enumerate(self.arch) if isinstance(level, ComputeLevel))
            level_index = self.arch.index(self)
            is_register_level = (compute_index - level_index <= 3)
            if not is_register_level:
                ## Check if forbidden have at most 1 iteration
                for dim in forbidden_dims:
                    if dim in self.dataflow and self.factors.dimProduct(dim) > 1:
                        return False      
            """      
        return True

    """
    Returns True iif this level's dataflow satisfies all of its constraints.
    """
    def checkDataflowConstraints(self) -> bool:
        # no "_" in dataflow constraints, enforce only the relative order of constrained dimensions
        if '_' not in self.dataflow_constraints:
            dim_idx = 0
            for dim in self.dataflow_constraints:
                while dim_idx < len(self.dataflow):
                    if dim == self.dataflow[dim_idx]:
                        break
                    dim_idx += 1
                if dim_idx == len(self.dataflow):
                    return False
        else: # "_" in dataflow constraints, enforce unspecified dimensions in the position of placeholders
            for i in range(len(self.dataflow)):
                if self.dataflow_constraints[i] != '_' and self.dataflow[i] != self.dataflow_constraints[i]:
                    return False
        return True

    """
    Returns a string describing the current violation of constraints, if any.
    """
    def logConstraintsViolation(self) -> str:
        if not super().checkFactorsConstraints():
            return super().logConstraintsViolation()
        elif not self.checkFactorsConstraints():
            if self.name == "AccumulationOutRegister":
                pass  # DEBUG print(f"self.bypasses: {self.bypasses}")
            mem_footprint = self.factors.memFootprint(self.tile_sizes, self.arch, not self.bypasses or 'in' not in self.bypasses, not self.bypasses or 'w' not in self.bypasses, not self.bypasses or 'out' not in self.bypasses, not self.bypasses or 'int_in' not in self.bypasses, not self.bypasses or 'int_out' not in self.bypasses)
            ## CONSTRAINT on mem_footprint
            if mem_footprint > self.size/self.multiple_buffering:
                # Print factors and tile sizes for debugging the mapping that failed
                print(f"\n{'='*80}")
                print(f"MEMORY CONSTRAINT VIOLATION DEBUG - Level: {self.name}")
                print(f"{'='*80}")
                print(f"Memory footprint: {mem_footprint:,.0f} bytes")
                print(f"Memory available: {self.size/self.multiple_buffering:,.0f} bytes")
                print(f"Overflow: {mem_footprint - self.size/self.multiple_buffering:,.0f} bytes")
                print(f"\nFull architecture mapping:")
                for level in self.arch:
                    if hasattr(level, 'factors') and level.factors:
                        factors_str = ", ".join([f"{dim}:{level.factors.dimProduct(dim)}" for dim in level.dataflow if level.factors.dimProduct(dim) > 1])
                        if factors_str:
                            print(f"  {level.name}: {factors_str}")
                print(f"{'='*80}\n")
                return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: memory footprint: {mem_footprint} VS memory available: {self.size/self.multiple_buffering:.0f}"
                 
            ## Check layer fusion constraint violation
            if hasattr(self.arch.coupling, 'w_coupling') and len(self.arch.coupling.w_coupling) > 1:
                # Get the number of layers
                num_layers = self.arch.coupling.getNumLayers()
                forbidden_dims = set()
                for i in range(num_layers - 1):
                    forbidden_dims.update([f'C{i+1}', f'R{i}', f'S{i}'])
                if num_layers > 1:
                    forbidden_dims.add('C1')

                ## CHECK IN ALL LEVEL \ last 
                level_index = self.arch.index(self)
                compute_index = next(i for i, level in enumerate(self.arch) if isinstance(level, ComputeLevel))
                is_register_level = (compute_index - level_index <= 3)  
  
                if not is_register_level:
                    violated_dims = [dim for dim in forbidden_dims if dim in self.dataflow and self.factors.dimProduct(dim) > 1]
                    if violated_dims:
                        return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: layer fusion constraint violated, dimensions with more than 1 iteration: {', '.join(violated_dims)}"           
        elif not self.checkDataflowConstraints():
            return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: dataflow: {self.dataflow} VS " + (f"dataflow constraints for relative order: {self.dataflow_constraints}" if '_' not in self.dataflow_constraints else f"positional dataflow constraints: {self.dataflow_constraints} ('_' are placeholders)")
        return ""

    def __str__(self) -> str:
        return f"{super().__str__()}, size: {self.size}, read_access_energy: {self.read_access_energy}, write_access_energy: {self.write_access_energy}, leakage_energy: {self.leakage_energy}, read_bandwidth: {self.read_bandwidth}, write_bandwidth: {self.write_bandwidth}, bypasses: {self.bypasses}, multiple_buffering: {self.multiple_buffering}"


"""
Abstract class for a level introducing multiple spatial instances.
"""
class SpatialLevel(Level):
    dims : list[str]
    mesh : int
    spatial_multicast_support : bool
    spatial_reduction_support : bool
    selective_multicast_support : bool
    selective_reduction_support : bool


"""
A Spatial Fanout Level within the architecture, the core of a spatial architecture,
all subsequent levels will be replicated "mesh" times, each replica executing one
of the iterations done by this level, by each running the same inner loops as the
others, with partially different data.

Constructor arguments:
- name: the level's name
- mesh: the maximum spatial fanout available at this level
- dim: single dimension to spatially unroll at this level, if dim is specified,
       dims must not be specified
- dims: list of dimensions to spatially unroll at this level, if dims is specified,
        dim must not be specified. The order in dims does not matter.
- area: the area occupied by the entire interconnect (in um^2).
- pe_to_pe: if True, data is not multicasted in one shot, but sent to only the first
            fanout entry, which then forwards it to the second, and so on, just like
            the operands flowing in/out of a systolic array (or, like in a pipeline).
            This adds a warmup overhead to latency, but does not affect the total MOPs.
- spatial_multicast_support: True (default) if this level supports spatial multicast.
        By itself, this only allows the same identical tile of an operand to be sent
        to each instance at once identically.
- spatial_reduction_support: same as 'spatial_multicast_support' but for spatial reduction.
- selective_multicast_support: enables sending parts of a read tile selectively to one
        or more different instances. In other words, if enabled, all values required
        by all instances are read once, with each instance receiving only the part of its
        interest. When disabled, each instance [involved in the spatial unrolling of a
        dimension present in a sum of indices] issues its own independent read for the
        values it needs. Obviously, this is useful only when fanouts unfold spatially
        dimensions involved in a sum of indices (e.g. those of the input tensor of a
        convolution), it does not make a difference otherwise (e.g. on GEMMs).
        Can only be True if 'spatial_multicast_support' is already True. Default is False.
- selective_reduction_support: same as 'selective_multicast_support' but for reduction.
        Can only be True if 'spatial_reduction_support' is already True. Default is False.
- power_gating_support: True if instances immediately following this level can be
                        power-gated when not in use, saving leakage power.
- factors: specifies the initial factors for this level, should not be normally
           specified aside for initializing MSE from a specific configuration
- tile_sizes: specifies the initial tile sizes for this level, should not be normally
              specified aside for initializing MSE from a specific configuration,
              in which case it must be consistent with any other factors initialization
- factors_constraints: constraints on the factors that must be placed on this level.
                      Valid dictionary keys use dimension names, e.g. for a GEMM:
                          - 'M', 'K', and 'N' for exact values;
                          - 'M<=', 'K<=', and 'N<=' for upper bounds;
                          - 'M>=', 'K>=', and 'N>=' for lower bounds;
                      NOTE: the use of the '<=' and '>=' constraints does not shorten the
                            mapper's runtime as much as exact constraints.
"""
# IMPORTANT:
# Currently fanout levels reuse all operands mapped on them, period. However this should be up to hardware support.
# Therefore add here a value N which determines how many operands can be spatially reused.
# Practically, if N = 1 and you have 2 loops with some iterations, for the inner loop operate as if spatial_multicast_support
# and spatial_reduction_support were True (if they were False to begin with, let them be False (&&)), for the second set
# them both to false.
# Obviously, in case N < |dims| you need to change the "iterate permutations" step to actually permute spatial loops!!!
# BETTER: make spatial_reduction_support and spatial_multicast_support be specified per-dimension!
class FanoutLevel(SpatialLevel):
    pe_to_pe : bool # True in all cases where the operand independent of "dim" (e.g.: in a GEMM, if dim = M, such operand is the input) is forwarded pe->pe rather than multicasted
    power_gating_support : bool

    def __init__(self, name : str, mesh : int, dim : Optional[str] = None, dims : Optional[list[str]] = None, area : Optional[float] = None, pe_to_pe : bool = False, spatial_multicast_support : bool = True, spatial_reduction_support : bool = True, selective_multicast_support : bool = False, selective_reduction_support : bool = False, power_gating_support : bool = False, factors : Optional[Factors] = None, tile_sizes : Optional[Shape] = None, factors_constraints : Optional[dict[str, int]] = None):
        self.name = name
        self._dim = dim
        self.dims =  dims
        self.mesh = mesh
        self.area = area
        self.pe_to_pe = pe_to_pe
        self.spatial_multicast_support = spatial_multicast_support
        self.spatial_reduction_support = spatial_reduction_support
        self.selective_multicast_support = selective_multicast_support
        self.selective_reduction_support = selective_reduction_support
        self.power_gating_support = power_gating_support
        self.factors = factors
        self.tile_sizes = tile_sizes
        self.factors_constraints = factors_constraints if factors_constraints else {}

    """
    Sets up a pointer back to the whole architecture.
    Ultimates the initialization of the level and validates its attributes.
    """
    def initArch(self, arch : Arch):
        self.arch = arch
        assert (self._dim and not self.dims) or (self.dims and not self._dim), f"Arch: {arch.name} -> Level: {self.name}: exactly one of dim ({self._dim}) or dims ({self.dims}) must be specified."
        self.dims = [self._dim] if self._dim else self.dims
        del self._dim
        self.dataflow = self.dims
        assert all([dim in arch.coupling.dims for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: accepted names for dimensions are solely M, K and N, provided ones were {self.dataflow}."
        num_layers = arch.coupling.getNumLayers()
        
        assert self.mesh > 0, f"Arch: {arch.name} -> Level: {self.name}: a spatial fanout must have a mesh ({self.mesh}) of at least 1."
        assert not self.area or self.area >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative area ({self.area}) does not mean anything."
        assert not self.pe_to_pe or (self.spatial_multicast_support and self.spatial_reduction_support), f"Arch: {arch.name} -> Level: {self.name}: pe-to-pe ({self.pe_to_pe}) forwarding is a form of spatial multicast ({self.spatial_multicast_support}) or reduction ({self.spatial_reduction_support}), which must then both be supported to enable it."
        assert not self.selective_multicast_support or self.spatial_multicast_support, f"Arch: {arch.name} -> Level: {self.name}: selective multicast ({self.selective_multicast_support}) is a form of spatial multicast ({self.spatial_multicast_support}), which must then be supported to enable it."
        assert not self.selective_reduction_support or self.spatial_reduction_support, f"Arch: {arch.name} -> Level: {self.name}: selective reduction ({self.selective_reduction_support}) is a form of spatial reduction ({self.spatial_reduction_support}), which must then be supported to enable it."
        self.factors = self.factors if self.factors else Factors(arch.coupling.dims)
        self.tile_sizes = self.tile_sizes if self.tile_sizes else Shape({dim: 1 for dim in arch.coupling.dims})
        assert all([
    constr in self.dataflow or 
    (constr.endswith('<=') and constr[:-2] in self.dataflow) or 
    (constr.endswith('>=') and constr[:-2] in self.dataflow) 
    for constr in self.factors_constraints.keys()
]), f"Arch: {arch.name} -> Level: {self.name}: all keys within factor constraints ({list(self.factors_constraints.keys())}) must be a dimension of the dataflow ({self.dataflow}) and in the form 'dim', 'dim<=', or 'dim>='."
        assert all([sum((constr == dim) + (constr == dim + '<=') + (constr == dim + '>=') for constr in self.factors_constraints.keys()) <= 1 for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: each dimension must occur at most once in factor constraints ({list(self.factors_constraints.keys())}), regardless of the use of '>=' or '<='."        
        assert all([value > 0 for value in self.factors_constraints.values()]), f"Arch: {arch.name} -> Level: {self.name}: all factor constraints ({self.factors_constraints}) must have a value strictly > 0."

    """
    Let inputs be the amount of operations occuring on a level below this fanout,
    this method returns the amount of operations seen from above the fanout,
    accounting for spatial multicast and spatial reduction support for operands.
    """
    # here you receive the reads/writes done by an instance, and need to
    # return the reads/writes that are needed for all instances.    
    def mulByDim(self, in_reads : int, per_layer_w_reads : dict[int, int], per_layer_int_in_reads : dict[int, int], per_layer_int_out_reads : dict[int, int], per_layer_int_out_writes : dict[int, int], out_reads : int, out_writes : int) -> tuple[int, dict[int, int], dict[int, int], dict[int, int], dict[int, int], int, int]:
        if self.selective_multicast_support:
            for dim_sum in self.arch.coupling.getInputCoupling():
                if len(dim_sum) > 1:
                    strides = [self.arch.getInStride(dim) for dim in dim_sum]
                    in_reads //= distinct_values([self.tile_sizes[dim] for dim in dim_sum], strides)
                    in_reads *= distinct_values([self.factors.dimProduct(dim)*self.tile_sizes[dim] for dim in dim_sum], strides)
                else:
                    in_reads *= self.factors.dimProduct(dim_sum[0])
            for layer_id in range(self.arch.coupling.getNumLayers()):
                for dim_sum in self.arch.coupling.getWeightCoupling(layer_id):
                    if len(dim_sum) > 1:
                        strides = [self.arch.getWStride(dim, layer_id) for dim in dim_sum]
                        per_layer_w_reads[layer_id] //= distinct_values([self.tile_sizes[dim] for dim in dim_sum], strides)
                        per_layer_w_reads[layer_id] *= distinct_values([self.factors.dimProduct(dim)*self.tile_sizes[dim] for dim in dim_sum], strides)
                    else:
                        per_layer_w_reads[layer_id] *= self.factors.dimProduct(dim_sum[0])
            for layer_id in range(self.arch.coupling.getNumLayers() - 1):
                for dim_sum in self.arch.coupling.getIntermediateInputCoupling(layer_id):
                    if len(dim_sum) > 1:
                        strides = [self.arch.getIntermediateInputStride(dim, layer_id) for dim in dim_sum]
                        per_layer_int_in_reads[layer_id] //= distinct_values([self.tile_sizes[dim] for dim in dim_sum], strides)
                        per_layer_int_in_reads[layer_id] *= distinct_values([self.factors.dimProduct(dim)*self.tile_sizes[dim] for dim in dim_sum], strides)
                    else:
                        per_layer_int_in_reads[layer_id] *= self.factors.dimProduct(dim_sum[0])    
            for dim_sum in self.arch.coupling.getOutputCoupling():
                if len(dim_sum) > 1:
                    strides = [self.arch.getOutStride(dim) for dim in dim_sum]
                    out_reads //= distinct_values([self.tile_sizes[dim] for dim in dim_sum], strides)
                    out_reads *= distinct_values([self.factors.dimProduct(dim)*self.tile_sizes[dim] for dim in dim_sum], strides)
                else:
                    out_reads *= self.factors.dimProduct(dim_sum[0])
        else:
            in_reads *= prod(self.factors.dimProduct(dim) for dim in self.arch.coupling.getFlatInputCoupling())
            
            for layer_id in range(self.arch.coupling.getNumLayers()):
                per_layer_w_reads[layer_id] *= prod(self.factors.dimProduct(dim) for dim in self.arch.coupling.getFlatWeightCoupling(layer_id))
            for layer_id in range(self.arch.coupling.getNumLayers() - 1):
                per_layer_int_in_reads[layer_id] *= prod(self.factors.dimProduct(dim) for dim in self.arch.coupling.getFlatIntermediateInputCoupling(layer_id))
                per_layer_int_out_reads[layer_id] *= prod(self.factors.dimProduct(dim) for dim in self.arch.coupling.getFlatIntermediateOutputCoupling(layer_id))

            out_reads *= prod(self.factors.dimProduct(dim) for dim in self.arch.coupling.getFlatOutputCoupling())
        # DEBUG print(f"Level: {self.name}: out_writes before selective_reduction_support = {out_writes}, per_layer_int_out_reads = {per_layer_int_out_reads} per_layer_int_out_writes = {per_layer_int_out_writes}")
        if self.selective_reduction_support:
            for dim_sum in self.arch.coupling.getOutputCoupling():
                if len(dim_sum) > 1:
                    strides = [self.arch.getOutStride(dim) for dim in dim_sum]
                    out_writes //= distinct_values([self.tile_sizes[dim] for dim in dim_sum], strides)
                    out_writes *= distinct_values([self.factors.dimProduct(dim)*self.tile_sizes[dim] for dim in dim_sum], strides)
                else:
                    out_writes *= self.factors.dimProduct(dim_sum[0])
            for layer_id, int_out_coupling in self.arch.coupling.int_out_coupling.items():
                if layer_id in per_layer_int_out_writes:
                    for dim_sum in int_out_coupling:
                        if len(dim_sum) > 1:
                            strides = [self.arch.getIntermediateOutputStride(dim, layer_id) for dim in dim_sum]
                            per_layer_int_out_writes[layer_id] //= distinct_values([self.tile_sizes[dim] for dim in dim_sum], strides)
                            per_layer_int_out_writes[layer_id] *= distinct_values([self.factors.dimProduct(dim)*self.tile_sizes[dim] for dim in dim_sum], strides)
                        else:
                            # DEBUG print(f"Level: {self.name}: selective_reduction_support out_writes for layer {layer_id} dim_sum {dim_sum}, dim={dim_sum[0]}, factor={self.factors.dimProduct(dim_sum[0])}")
                            per_layer_int_out_writes[layer_id] *= self.factors.dimProduct(dim_sum[0])
        else:
            out_writes *= prod(self.factors.dimProduct(dim) for dim in self.arch.coupling.getFlatOutputCoupling())

            for layer_id in per_layer_int_out_writes.keys():
                per_layer_int_out_writes[layer_id] *= prod(self.factors.dimProduct(dim) for dim in self.arch.coupling.getFlatIntermediateOutputCoupling(layer_id))        
        # DEBUG print(f"Level: {self.name}: out_writes after selective_reduction_support = {out_writes}, per_layer_int_out_reads = {per_layer_int_out_reads} per_layer_int_out_writes = {per_layer_int_out_writes}")

        # Initialize dataflow_per_layer before both spatial_multicast_support and spatial_reduction_support blocks
        # so it's available for both conditional paths
        dataflow_per_layer: dict[int, list[str]] = {}
        if self.arch.coupling.getNumLayers() > 1:
            for layer_id in range(self.arch.coupling.getNumLayers()):
                # For each layer, filter dimensions that have loops > 1 AND are relevant to that layer
                layer_relevant_dims = self.arch.coupling.relevantDimsForLayer(layer_id)
                dataflow_per_layer[layer_id] = [
                    dim for dim in self.dataflow
                    if dim in layer_relevant_dims
                ]
        else:
            dataflow_per_layer[0] = self.dataflow.copy()
        
        if not self.spatial_multicast_support:
            in_reads *= prod(self.factors.dimProduct(dim) for dim in dataflow_per_layer[0] if dim not in self.arch.coupling.getFlatInputCoupling())
            for layer_id in range(self.arch.coupling.getNumLayers()):
                per_layer_w_reads[layer_id] *= prod(self.factors.dimProduct(dim) for dim in dataflow_per_layer[layer_id] if dim not in self.arch.coupling.getFlatWeightCoupling(layer_id))
            for layer_id in range(self.arch.coupling.getNumLayers() - 1):
                per_layer_int_in_reads[layer_id] *= prod(self.factors.dimProduct(dim) for dim in dataflow_per_layer[layer_id + 1] if dim not in self.arch.coupling.getFlatIntermediateInputCoupling(layer_id))
                per_layer_int_out_reads[layer_id] *= prod(self.factors.dimProduct(dim) for dim in dataflow_per_layer[layer_id] if dim not in self.arch.coupling.getFlatIntermediateOutputCoupling(layer_id))
            out_reads *= prod(self.factors.dimProduct(dim) for dim in dataflow_per_layer[self.arch.coupling.getNumLayers() - 1] if dim not in self.arch.coupling.getFlatOutputCoupling())
        if not self.spatial_reduction_support:
            out_writes *= prod(self.factors.dimProduct(dim) for dim in dataflow_per_layer[0] if dim not in self.arch.coupling.getFlatOutputCoupling())
            for layer_id in range(self.arch.coupling.getNumLayers() - 1):
                per_layer_int_out_writes[layer_id] *= prod(self.factors.dimProduct(dim) for dim in dataflow_per_layer[layer_id] if dim not in self.arch.coupling.getFlatIntermediateOutputCoupling(layer_id))
        
        # DEBUG print(f"Level: {self.name}: after spatial_multicast_support and spatial_reduction_support blocks: in_reads = {in_reads}, per_layer_w_reads = {per_layer_w_reads}, per_layer_int_in_reads = {per_layer_int_in_reads}, per_layer_int_out_reads = {per_layer_int_out_reads}, per_layer_int_out_writes = {per_layer_int_out_writes}, out_reads = {out_reads}, out_writes = {out_writes}\n")
        return in_reads, per_layer_w_reads, per_layer_int_in_reads, per_layer_int_out_reads, per_layer_int_out_writes, out_reads, out_writes

    """
    Returns the clock cycles required by this fanout level to sustain the bandwidth
    to move all operands across the above and below memory/compute levels.
    
    TODO: implement this w.r.t. a NoC model.
    
    => The returned value must be multiplied by the factors above it.
    """
    def latency(self) -> int:
        return 0 # change this if we model the network's latency

    """
    Returns True iif factors present on this level satisfy all of its constraints,
    including not exceeding the physical mesh.
    """
    def checkFactorsConstraints(self) -> bool:
        return self.factors.fullProduct() <= self.mesh and super().checkFactorsConstraints()

    """
    Returns a string describing the current violation of constraints, if any.
    """
    def logConstraintsViolation(self) -> str:
        if not super().checkFactorsConstraints():
            return super().logConstraintsViolation()
        elif not self.checkFactorsConstraints():
            return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: spatial iterations used: {self.factors.fullProduct()} VS available instances (mesh): {self.mesh}"
        return ""

    def __str__(self) -> str:
        return f"{super().__str__()}, mesh: {self.mesh}, pe_to_pe: {self.pe_to_pe}, spatial_multicast_support: {self.spatial_multicast_support}, spatial_reduction_support: {self.spatial_reduction_support}, power_gating_support: {self.power_gating_support}"


"""
A Compute Level within the architecture, it is a placeholder for any processing
element (PE) capable of multiply and accumulate (MAC).

NOTE: iterations at this level are equivalent to requiring that a PE can, in the
clock-cycles specified in "cycles", execute all such iterations. It is also assumed
that the data required for all iterations done at this level is held within hardware
registers whose energy cost is modeled by the "compute energy", together with the
actual compute cost. As such, iterations done here are equivalent to a PE capable
of multiple concurrent (parallel or pipelined (chaining accumulation output)) MACs.
Then intuitively, increasing the number of iterations done at this level linearly
increases the required bandwidth of all memory levels feeding it, as they need to
keep up with the concurrent MACs done here.

NOTE: no value is kept stationary on a ComputeLevel, the values for each MAC
operation are read/written every time. PE-level stationarity is realized by the
first Memory Level above compute.
NOTE: values that partake in multiple MAC operations performed concurrently in the
iterations of the ComputeLevel are read/written once per MAC.
TODO: move the multicast/reduction and selective_multicast/selective_reduction
to the SpatialLevel and make them available on ComputeLevel too. Then make the
above NOTEs conditional on the True or False of those properties too! These would
take effect during MemLevel.MOPs, just like they do for FanoutLevel, since also
the ComputeLevel is included in MemLevel.next_spatials!

Constructor arguments:
- name: the level's name
- mesh: how many concurrent (parallel or pipelined) MACs the compute element
        can perform within "cycles" clock-cycles.
- compute_energy: the energy required for a single MAC (regardless of how many
                  you run concurrently), accounting for all computation-related
                  costs at the PE level.
- cycles: the number of clock cycles of latency required to execute "size" MACs
- leakage_energy: energy leaked each clock cycle by the component (in pJ/cc)
- area: the area occupied by the entire PE (in um^2).
- dim: single dimension along which MAC operations are picked to run concurrently
       on this level, if dim is specified, dims must not be specified
       NOTE: when mesh is 1, both dim and dims can be omitted
- dims: list of dimensions from which MAC operations are picked to run concurrently
       on this level, if dims is specified, dim must not be specified.
       The order in dims does not matter.
       NOTE: when mesh is 1, both dim and dims can be omitted
- factors: specifies the initial factors for this level, should not be normally
           specified aside for initializing MSE from a specific configuration
- tile_sizes: specifies the initial tile sizes for this level, should not be normally
              specified aside for initializing MSE from a specific configuration,
              in which case it must be consistent with any other factors initialization
- factors_constraints: constraints on the factors that must be placed on this level.
                      Valid dictionary keys use dimension names, e.g. for a GEMM:
                          - 'M', 'K', and 'N' for exact values;
                          - 'M<=', 'K<=', and 'N<=' for upper bounds;
                          - 'M>=', 'K>=', and 'N>=' for lower bounds;
                      NOTE: the use of the '<=' and '>=' constraints does not shorten the
                            mapper's runtime as much as exact constraints.
"""
class ComputeLevel(SpatialLevel):
    compute_energy : float
    leakage_energy : float
    cycles : int # clock cycles used per element in the inner dimension (latency of one MAC)

    # STATISTICS:
    active_instances_per_layer : dict[int, int]
    active_instances : int
    temporal_iterations_per_layer : dict[int, int]
    temporal_iterations : int
    
    def __init__(self, name : str, mesh : int, compute_energy : float, cycles : int, dim : Optional[str] = None, dims : Optional[list[str]] = None, leakage_energy : float = 0, area : Optional[float] = None, factors : Optional[Factors] = None, tile_sizes : Optional[Shape] = None, factors_constraints : Optional[dict[str, int]] = None):
        self.name = name
        self._dim = dim
        self.dims = dims
        self.mesh = mesh # for a systolic array, this is the length of the operand buffers
        self.compute_energy = compute_energy
        self.leakage_energy = leakage_energy
        self.area = area
        self.spatial_multicast_support = False
        self.spatial_reduction_support = False
        self.selective_multicast_support = False
        self.selective_reduction_support = False
        self.cycles = cycles # clock cycles used per element in the inner dimension (latency of one MAC)
        self.factors = factors
        self.tile_sizes = tile_sizes
        self.factors_constraints = factors_constraints if factors_constraints else {}

    """
    Sets up a pointer back to the whole architecture.
    Ultimates the initialization of the level and validates its attributes.
    """
    def initArch(self, arch : Arch):
        self.arch = arch
        assert self.mesh == 1 or (self._dim and not self.dims) or (self.dims and not self._dim), f"Arch: {arch.name} -> Level: {self.name}: when mesh ({self.mesh}) is > 1, exactly one of dim ({self._dim}) or dims ({self.dims}) must be specified."
        self.dims = ([self._dim] if self._dim else self.dims) if self._dim or self.dims else []
        del self._dim
        self.dataflow = self.dims
        assert all([dim in arch.coupling.dims for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: accepted names for dimensions are solely M, K and N, provided ones were {self.dataflow}."
        assert self.mesh > 0, f"Arch: {arch.name} -> Level: {self.name}: a zero or negative size ({self.mesh}) does not make sense."
        assert self.compute_energy >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative compute energy ({self.compute_energy}) does not mean anything (unless you watched too much Gundam and discovered Minovsky particles...)."
        assert self.leakage_energy >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative leakage energy ({self.leakage_energy}) does not mean anything (unless you watched too much Gundam 00 and discovered GN particles...)."
        assert not self.area or self.area >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative area ({self.area}) does not mean anything."
        assert self.cycles >= 0, f"Arch: {arch.name} -> Level: {self.name}: a negative number of clock-cycles per MAC ({self.cycles}) does not mean anything."
        self.factors = self.factors if self.factors else Factors(arch.coupling.dims)
        self.tile_sizes = self.tile_sizes if self.tile_sizes else Shape({dim: 1 for dim in arch.coupling.dims})
        assert all([
    constr in self.dataflow or 
    (constr.endswith('<=') and constr[:-2] in self.dataflow) or 
    (constr.endswith('>=') and constr[:-2] in self.dataflow) 
    for constr in self.factors_constraints.keys()
]), f"Arch: {arch.name} -> Level: {self.name}: all keys within factor constraints ({list(self.factors_constraints.keys())}) must be a dimension of the dataflow ({self.dataflow}) and in the form 'dim', 'dim<=', or 'dim>='."
        assert all([sum((constr == dim) + (constr == dim + '<=') + (constr == dim + '>=') for constr in self.factors_constraints.keys()) <= 1 for dim in self.dataflow]), f"Arch: {arch.name} -> Level: {self.name}: each dimension must occur at most once in factor constraints ({list(self.factors_constraints.keys())}), regardless of the use of '>=' or '<='."
        assert all([value > 0 for value in self.factors_constraints.values()]), f"Arch: {arch.name} -> Level: {self.name}: all factor constraints ({self.factors_constraints}) must have a value strictly > 0."

    """
    Returns the clock cycles required by this compute level to perform ALL its
    allocated iterations. The reasoning being that such iterations occur all either
    within the same set of cycles or in pipeline with one step requiring the hereby
    returned amount of cycles.
    
    => The returned value must be multiplied by the factors above it.
    """
    def latency(self) -> int:
        return self.cycles


    """
    Returns the energy required by this level to perform all MACs across its internal
    iterations, times "iterations", which represents the number of iterations done by
    the hierarchy of levels on top of this one.
    """
    def computeCost(self, iterations : int = 1) -> float:
        return self.compute_energy*float(self.factors.fullProduct()*iterations)

    """
    Returns the energy required by this level to perform all MACs across its internal
    iterations, times "iterations", which represents the number of iterations done by
    the hierarchy of levels on top of this one, for a specific layer (convolution).
    """
    def computeCostPerLayer(self, layer_id : int, iterations : int = 1) -> float:
        # Only multiply by factors relevant to this specific layer
        layer_factors_product = 1
        
        # Get the relevant dimensions for this layer
        layer_relevant_dims = self.arch.coupling.relevantDimsForLayer(layer_id)

        # Calculate the product of factors only for dimensions relevant to this layer
        for dim in self.dataflow:
            if dim in layer_relevant_dims:
                layer_factors_product *= self.factors.dimProduct(dim)
        
        return self.compute_energy * float(layer_factors_product * iterations)

    """
    Returns the total leaked energy during the provided clock cycles.
    """
    def Leakage(self, cycles : int) -> float:
        return cycles*self.leakage_energy

    """
    Returns True iif factors present on this level satisfy all of its constraints,
    including not exceeding the phisical number of concurrent MACs performed here.
    """
    def checkFactorsConstraints(self) -> bool:
        return self.factors.fullProduct() <= self.mesh and super().checkFactorsConstraints()

    """
    Returns a string describing the current violation of constraints, if any.
    """
    def logConstraintsViolation(self) -> str:
        if not super().checkFactorsConstraints():
            return super().logConstraintsViolation()
        elif not self.checkFactorsConstraints():
            return f"CONSTRAINTS VIOLATION: Arch: {self.arch.name} -> Level: {self.name}: concurrent MACs used: {self.factors.fullProduct()} VS concurrent MACs available: {self.mesh}"
        return ""

    def __str__(self) -> str:
        return f"{super().__str__()}, mesh: {self.mesh}, compute_energy: {self.compute_energy}, leakage_energy: {self.leakage_energy}, cycles: {self.cycles}"