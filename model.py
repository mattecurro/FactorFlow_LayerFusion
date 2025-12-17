import math

from settings import *
from factors import *
from levels import *
from prints import *
from utils import *
from arch import *

"""
Entry point for the analytical model.
Updates the MOPs and Latency data of each level w.r.t. the current mapping.
"""
def updateStats(arch : Arch, bias_read : bool) -> tuple[float, int]:
    print("é stato chiamato update stats! \n\n")
    assert arch.initialized, f"Arch {arch.name}: architecture not initialized, ensure to call 'initFactors' first."
    
    WMOPs_per_layer = {layer_id: 0 for layer_id in range(arch.coupling.getNumLayers())}
    WMOPs = 0
    num_layers = arch.coupling.getNumLayers()
    temporal_iterations = 1
    temporal_iterations_per_layer = {layer_id: 1 for layer_id in range(num_layers)}
    spatial_iterations = 1
    spatial_iterations_per_layer = {layer_id: 1 for layer_id in range(num_layers)}
    last_in_reads, last_w_reads, last_out_reads, last_out_writes = 0, 0, 0, 0
    last_int_in_reads, last_int_out_reads = 0, 0
    acc_out_reads_factors = 1
    # NOTE: here we compute total MOPs, not per-instance
    for i in range(len(arch)):
        level = arch[i]
        # Initialize dataflow per layer
        dataflow_per_layer = {}
        if num_layers > 1:
            for layer_idx in range(num_layers):
                if layer_idx == 0:
                    layer_relevant_dims = (
                        set(arch.coupling.getFlatInputCoupling()) |
                        set(arch.coupling.getFlatWeightCoupling(layer_idx))|
                        set(arch.coupling.getFlatIntermediateOutputCoupling(layer_idx))
                    )
                elif layer_idx == num_layers - 1:
                    layer_relevant_dims = (
                        set(arch.coupling.getFlatOutputCoupling()) |
                        set(arch.coupling.getFlatWeightCoupling(layer_idx))|
                        set(arch.coupling.getFlatIntermediateInputCoupling(layer_idx - 1))
                    )
                else:
                    layer_relevant_dims = (
                        set(arch.coupling.getFlatIntermediateInputCoupling(layer_idx - 1)) |
                        set(arch.coupling.getFlatWeightCoupling(layer_idx))|
                        set(arch.coupling.getFlatIntermediateOutputCoupling(layer_idx))
                    )
                dataflow_per_layer[layer_idx] = [dim for dim in level.dataflow if dim in layer_relevant_dims]
        else:
            dataflow_per_layer[0] = level.dataflow.copy()

        # Results after scaling        
        if isinstance(level, MemLevel):
            # multiply by spatial_iterations too because memory is replicated spatially
            print("\n\nQuesto mops è chiamato da update stats")     
            print(f"Level: {level.name}, arch bypasses for the level: {level.bypasses}, in_bp: {level.in_bp}, w_bp: {level.w_bp}, out_bp: {level.out_bp}, int_bp: {level.int_bp}")      
            ## Get base MOPs for this level
            in_reads, per_layer_w_reads, per_layer_int_in_reads, per_layer_int_out_reads, per_layer_int_out_writes, out_reads, out_writes, out_reads_factors = level.MOPs()
            w_reads = sum(per_layer_w_reads.values()) 
            int_in_reads = sum(per_layer_int_in_reads.values()) if per_layer_int_in_reads else 0
            int_out_reads = sum(per_layer_int_out_reads.values()) if per_layer_int_out_reads else 0
            int_out_writes = sum(per_layer_int_out_writes.values()) if per_layer_int_out_writes else 0
            print(f"DEBUG updateStats pre scaling: Level {level.name}:"
                  f"\n  in_reads: {in_reads}"
                  f"\n  per_layer_w_reads: {per_layer_w_reads}"
                  f"\n  w_reads: {w_reads}"
                  f"\n  per_layer_int_in_reads: {per_layer_int_in_reads}"
                  f"\n  per_layer_int_out_reads: {per_layer_int_out_reads}"
                  f"\n  out_reads: {out_reads}"
                  f"\n  out_reads_factors: {out_reads_factors}")
            print(f"temporal_iterations_per_layer before update: {temporal_iterations_per_layer}, spatial_iterations_per_layer before update: {spatial_iterations_per_layer}")
            # Scale MOPs according to temporal and spatial iterations per layer
            scale_per_layer = {layer_id: temporal_iterations_per_layer[layer_id]*spatial_iterations_per_layer[layer_id] for layer_id in range(num_layers)}
            print(f"scale per layer: {scale_per_layer}")
            in_reads = in_reads * scale_per_layer[0]
            print(f"scaled in_reads: {in_reads}")
            for layer_id, w_read in per_layer_w_reads.items():
                per_layer_w_reads[layer_id] = w_read * scale_per_layer[layer_id]
            print(f"scaled per_layer_w_reads before sum: {per_layer_w_reads}")
            w_reads = sum(per_layer_w_reads.values())
            for layer_id, int_in_read in per_layer_int_in_reads.items():
                per_layer_int_in_reads[layer_id] = int_in_read * scale_per_layer[layer_id+1]
            print(f"scaled per_layer_int_in_reads before sum: {per_layer_int_in_reads}")
            int_in_reads = sum(per_layer_int_in_reads.values()) 
            for layer_id, int_out_read in per_layer_int_out_reads.items():
                per_layer_int_out_reads[layer_id] = int_out_read * scale_per_layer[layer_id]
            print(f"scaled per_layer_int_out_reads before sum: {per_layer_int_out_reads}")
            int_out_reads = sum(per_layer_int_out_reads.values())
            for layer_id, int_out_write in per_layer_int_out_writes.items():
                per_layer_int_out_writes[layer_id] = int_out_write * scale_per_layer[layer_id]
            int_out_writes = sum(per_layer_int_out_writes.values())
            out_reads = out_reads * scale_per_layer[num_layers - 1]
            print(f"scaled out_reads: {out_reads}")
            out_writes = out_writes * scale_per_layer[num_layers - 1]
            out_reads_factors = out_reads_factors * acc_out_reads_factors
            
            # Adjust for bias read if needed, update writes, DRAINS management
            if not bias_read and out_reads_factors != 0:
                out_reads = (out_reads*(out_reads_factors - 1))//out_reads_factors
            if 'in' not in level.bypasses:
                in_writes = last_in_reads # reads above are written here
                last_in_reads = in_reads
            else:
                in_writes = 0
            if 'w' not in level.bypasses:
                w_writes = last_w_reads # reads above are written here
                last_w_reads = w_reads
            else:
                w_writes = 0
            if 'int' not in level.bypasses:
                int_in_writes = last_int_in_reads   # intermediate input reads above are written here
                int_out_writes += last_int_out_reads # intermediate output reads above are written here
                last_int_in_reads = int_in_reads    # update for next level
                last_int_out_reads = int_out_reads
            else:
                int_in_writes = 0
            if 'out' not in level.bypasses:
                level.setAboveMOPs(last_out_reads, last_out_writes)
                out_writes += last_out_reads # reads above are written here
                last_out_reads = out_reads
                if not Settings.FREE_DRAINS:
                    out_reads += last_out_writes # writes above where read here
                    last_out_writes = out_writes
            else:
                level.setAboveMOPs(0, 0)
            print(f"DEBUG updateStats post scaling: Level {level.name}:"
                  f"\n  in_reads: {in_reads}"
                  f"\n  per_layer_w_reads: {per_layer_w_reads}"
                  f"\n  w_reads: {w_reads}"
                  f"\n  per_layer_int_in_reads: {per_layer_int_in_reads}"
                  f"\n  int_in_reads: {int_in_reads}"
                  f"\n  per_layer_int_out_reads: {per_layer_int_out_reads}"
                  f"\n  int_out_reads: {int_out_reads}"
                  f"\n  out_reads: {out_reads}"
                  f"\n  out_reads_factors: {out_reads_factors}")
            ## Update level MOPs
            level.setMOPs(
                in_reads=in_reads,
                per_layer_w_reads=per_layer_w_reads,
                per_layer_int_in_reads=per_layer_int_in_reads,
                per_layer_int_out_reads=per_layer_int_out_reads,
                per_layer_int_out_writes=per_layer_int_out_writes,
                out_reads=out_reads,
                out_writes=out_writes
            )
            level.temporal_iterations_per_layer = temporal_iterations_per_layer.copy()  
            level.temporal_iterations = temporal_iterations
            # Calculate total reads and writes for energy calculation
            total_reads = in_reads + w_reads + int_in_reads + int_out_reads + out_reads
            total_writes = in_writes + w_writes + int_in_writes + int_out_writes + out_writes 
            total_reads_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
            if num_layers > 1:
                total_reads_per_layer[0] = in_reads + per_layer_w_reads.get(0, 0) + per_layer_int_out_reads.get(0, 0)
                for layer_idx in range(1, num_layers - 1):
                    total_reads_per_layer[layer_idx] = per_layer_int_in_reads.get(layer_idx - 1, 0) + per_layer_w_reads.get(layer_idx, 0) + per_layer_int_out_reads.get(layer_idx, 0)
                total_reads_per_layer[num_layers - 1] = per_layer_int_in_reads.get(num_layers - 2, 0) + per_layer_w_reads.get(num_layers - 1, 0) + out_reads
                for layer_id in range(num_layers):
                    print(f"Level {level.name} total_reads_per_layer[{layer_id}]: {total_reads_per_layer[layer_id]}")
                    WMOPs_per_layer[layer_id] += level.WMOPs(total_reads_per_layer[layer_id], total_writes)
                WMOPs = sum(WMOPs_per_layer.values())
                real = level.WMOPs(total_reads, total_writes)
                print(f"Level {level.name} total_WMOPs: {WMOPs}")
                print(f"Level {level.name} real WMOPs: {real}")
            else:
                total_reads = in_reads + w_reads + out_reads
                WMOPs_per_layer[0] += level.WMOPs(total_reads, total_writes)
                WMOPs = WMOPs_per_layer[0]
            level.active_instances_per_layer = spatial_iterations_per_layer.copy()
            level.active_instances = spatial_iterations
            # Update temporal iterations per layer
            for layer_idx in range(num_layers):
                layer_factors_product = 1
                for dim in dataflow_per_layer[layer_idx]:
                    layer_factors_product *= level.factors.dimProduct(dim)
#                    print(f"Updated considering level: {level.name}, this dim: {dim} is considered in temporal_iterations_per_layer[{layer_idx}]")
                temporal_iterations_per_layer[layer_idx] *= layer_factors_product
            ## ATTENTION: this temporal_iterations don't consider the intermediate escamotage
            temporal_iterations *= level.factors.fullProduct()
            acc_out_reads_factors *= math.prod(level.factors.dimProduct(dim) for dim in dataflow_per_layer[num_layers-1] if dim not in level.arch.coupling.getFlatOutputCoupling())
        # update of last reads
        elif isinstance(level, FanoutLevel):
            spatial_iterations *= level.factors.fullProduct()
            for layer_idx in range(num_layers):
                # spatial reuse of an operand occurs if the fanout is along a dimension not coupled to such operand,
                # hence, the operand is read once, but written once per instance (modeled by last_XX_reads)
                # TODO: add NoC modeling and accumulate data transfer energy here!
                layer_spatial_iterations = 1
                for dim in level.dataflow_per_layer[layer_idx]:
                    layer_spatial_iterations *= level.factors.dimProduct(dim)
                spatial_iterations_per_layer[layer_idx] *= layer_spatial_iterations
                # Update last_operand_reads accordingly
                for dim in level.dataflow_per_layer[layer_idx]:
                    layer_iterations = level.factors.dimProduct(dim)
                    # Layer 0: has input and intermediate output
                    if layer_idx == 0:
                        if dim not in arch.coupling.getFlatInputCoupling():
                            last_in_reads *= layer_iterations
                        if dim not in arch.coupling.getFlatIntermediateOutputCoupling(layer_idx):
                            last_int_out_reads *= layer_iterations
                        if dim not in arch.coupling.getFlatWeightCoupling(layer_idx):
                            last_w_reads *= layer_iterations                            
                    # Last layer: has intermediate input and final output
                    elif layer_idx == num_layers - 1:
                        if dim not in arch.coupling.getFlatOutputCoupling():
                            last_out_reads *= layer_iterations
                            last_out_writes *= layer_iterations
                        if dim not in arch.coupling.getFlatIntermediateInputCoupling(layer_idx - 1):
                            last_int_in_reads *= layer_iterations
                        if dim not in arch.coupling.getFlatWeightCoupling(layer_idx):
                            last_w_reads *= layer_iterations                           
                    # Middle layers: have both intermediate input and output
                    else:
                        if dim not in arch.coupling.getFlatIntermediateInputCoupling(layer_idx - 1):
                            last_int_in_reads *= layer_iterations
                        if dim not in arch.coupling.getFlatIntermediateOutputCoupling(layer_idx):
                            last_int_out_reads *= layer_iterations
                        if dim not in arch.coupling.getFlatWeightCoupling(layer_idx):
                            last_w_reads *= layer_iterations                    
            print(f"Updated considering level: {level.name}, this dim: {dim} is considered in spatial_iterations_per_layer[{layer_idx}]")
        # WMOPs
        elif isinstance(level, ComputeLevel):
            # TODO: remove cost of first output accumulate if bias_read is False!
            # => not needed because the cost of the add is << than the multiply!
            level.temporal_iterations_per_layer = temporal_iterations_per_layer.copy()
            level.active_instances_per_layer = spatial_iterations_per_layer.copy()
            level.temporal_iterations = temporal_iterations
            level.active_instances = spatial_iterations
            for layer_id in range(num_layers):
                WMOPs_per_layer[layer_id] += level.computeCostPerLayer(layer_id, temporal_iterations_per_layer[layer_id]*level.active_instances_per_layer[layer_id])
                WMOPs += WMOPs_per_layer[layer_id]
            # compute is meant to be the innermost level
            break
    
    # Latency:
    max_latency = 0
    cc_per_tile = arch[len(arch)-1].latency()
    # IMPROVEMENT:
    # - data for the operand which is stationary can be drained/filled immediately as all iterations unfold (cc_per_tile * fullProduct)
    # - data for the non-stationary operands can be filled/drained only during the last iteration of the outermost loop (cc_per_tile * product of only the two inner dimensions)
    # - if double buffering, add the outermost dimension to the second product too
    # WARNING:
    # - the stationary operand only ever occupies a PART of its allotted space, so the size constraint shall be relaxed according to the number of instance
    #   of such operand which are scheduled to be kept at the same time
    for i in range(len(arch) - 2, -1, -1):
        level = arch[i]
        previous_fanout_pe_to_pe_warmup = 0
        if isinstance(level, MemLevel):
            scaling = level.active_instances*level.temporal_iterations
            cc_per_all_tiles = cc_per_tile*level.factors.fullProduct()
            scaled_cc_per_all_tiles = scaling*cc_per_all_tiles
            ideal_bandwidth_read = level.getRead()/scaled_cc_per_all_tiles # original -more readable- formulation: (level.getRead()/scaling)/(cc_per_tile*level.factors.fullProduct())
            ideal_bandwidth_update = level.getUpdate()/scaled_cc_per_all_tiles
            # TODO: this should get divided per-operand, as depending on the dataflow, an operand may have more or less iterations to be loaded
            # TODO: support double buffering on a per-operand basis
            # NOTE: the current implementation coincides with Timeloop's notion of Buffets, but it is not exact...
            # NOTE: my bandwidth is already a bit more accurate since Timeloop ignores drains...
            #outermost_available_iterations = 1 if level.multiple_buffering == 1 else level.factors.dimProduct(level.dataflow[0])*(level.multiple_buffering - 1)
            #ideal_bandwidth_fill = (level.getFill()/scaling)/(cc_per_tile*level.factors.dimProduct(level.dataflow[1])*level.factors.dimProduct(level.dataflow[2])*outermost_available_iterations)
            #ideal_bandwidth_drain = (level.getDrain()/scaling)/(cc_per_tile*level.factors.dimProduct(level.dataflow[1])*level.factors.dimProduct(level.dataflow[2])*outermost_available_iterations)
            ideal_bandwidth_fill = level.getFill()/scaled_cc_per_all_tiles
            ideal_bandwidth_drain = level.getDrain()/scaled_cc_per_all_tiles if not Settings.FREE_DRAINS else 0
            # bandwidth is statically divided between reads and writes
            # NOTE: warmup cycles cannot be used to compensate for a lack of bandiwidth at regime
            if ideal_bandwidth_read + ideal_bandwidth_drain <= level.read_bandwidth:
                latency_read_drain = cc_per_all_tiles + previous_fanout_pe_to_pe_warmup*cc_per_tile
            else:
                latency_read_drain = ((level.getRead() + level.getDrain())/scaling)*(1/level.read_bandwidth) if not Settings.FREE_DRAINS else (level.getRead()/scaling)*(1/level.read_bandwidth)
            if ideal_bandwidth_fill + ideal_bandwidth_update <= level.write_bandwidth:
                latency_fill_update = cc_per_all_tiles + previous_fanout_pe_to_pe_warmup*cc_per_tile
            else:
                latency_fill_update = ((level.getFill() + level.getUpdate())/scaling)*(1/level.write_bandwidth)
            latency = max(latency_read_drain, latency_fill_update)
            stall_cycles = latency - cc_per_all_tiles
            level.setLatency(latency_read_drain = latency_read_drain*level.temporal_iterations, latency_fill_update = latency_fill_update*level.temporal_iterations, cc_per_tile = cc_per_tile, stall_cycles = stall_cycles*level.temporal_iterations, ideal_bandwidth_read = ideal_bandwidth_read, ideal_bandwidth_update = ideal_bandwidth_update, ideal_bandwidth_fill = ideal_bandwidth_fill, ideal_bandwidth_drain = ideal_bandwidth_drain)
            #Timeloop does this (loosing knowledge of true behaviour): cc_per_tile = cc_per_tile*level.factors.fullProduct()
            previous_fanout_pe_to_pe_warmup = 0
            cc_per_tile = latency
        elif isinstance(level, FanoutLevel) and level.pe_to_pe:
            # pe-to-pe forwarding implies that the SA operates as a PIPELINE, which has overall latency equal to that of an operation but a warmup dependent on the mesh size
            previous_fanout_pe_to_pe_warmup = level.mesh - 1    
    # Active instances, leakage, and final latency:
    temporal_iterations_per_layer = {layer_id: 1 for layer_id in range(num_layers)}
    temporal_iterations = 1
    powered_instances_per_layer = {layer_id: 1 for layer_id in range(num_layers)}
    powered_instances = 1
    for i in range(len(arch)):
        level = arch[i]
        if isinstance(level, MemLevel):
            max_latency = max(max_latency, level.getSettedLatency())
            #print(f"Leakage level {level.name}: {level.Leakage(level.getSettedLatency())*powered_instances}")
            # Add leakage energy per layer 
            WMOPs += level.Leakage(level.getSettedLatency()) * sum(powered_instances_per_layer.values())
            temporal_iterations *= level.factors.fullProduct()
        elif isinstance(level, FanoutLevel):
            # Calculate max temporal iterations across all layers
            max_latency = max(max_latency, level.latency() * temporal_iterations)
            if level.power_gating_support:
                powered_instances *= level.factors.fullProduct()
            else:
                powered_instances *= level.mesh
        elif isinstance(level, ComputeLevel):
            max_latency = max(max_latency, level.latency() * temporal_iterations)
            WMOPs += level.Leakage(level.latency()) * powered_instances
            break

    return WMOPs, max_latency

"""
Weighted Arithmetic Intensity (WART)

It is equivalent to FLOPs/EDPoU, where EDPoU = (Energy * Latency) / Utilization
=> Maximizing the WART minimizes the EDPoU.
"""
def Wart(arch : Arch, comp : Shape, bias_read : bool, utilization_exponent : int = 1) -> float:
    FLOPs = comp.FLOPs()
    WMOPs, max_latency = updateStats(arch, bias_read)
    utilization = arch.spatialUtilization()**utilization_exponent if Settings.UTILIZATION_IN_WART else 1
    return (FLOPs/(WMOPs*max_latency))*utilization

"""
Energy-Delay Product [pJ*cc]

If pJ_to_J is True, the returned value is in [J*cc].
"""
def EDP(arch : Arch, bias_read : bool, pJ_to_J : bool = False) -> float:
    WMOPs, max_latency = updateStats(arch, bias_read)
    return WMOPs*max_latency*(10**-12 if pJ_to_J else 1)

"""
Latency [cc]
"""
def Latency(arch : Arch) -> int:
    max_latency = 0
    for level in arch:
        if isinstance(level, MemLevel):
            if max_latency <= level.getSettedLatency():
                max_latency = level.getSettedLatency()
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            break
    return max_latency

"""
Energy [pJ]

If pJ_to_uJ is True, the returned value is in [uJ].
"""
def Energy(arch : Arch, pJ_to_uJ : bool = False) -> float:
    WMOPs = 0
    for level in arch:
        if isinstance(level, MemLevel):
            reads = level.in_reads + level.w_reads + level.out_reads
            writes = level.in_writes + level.w_writes + level.out_writes
            WMOPs += level.WMOPs(reads, writes)
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            for layer_id in range(arch.coupling.getNumLayers()):
                WMOPs += level.computeCostPerLayer(layer_id, level.temporal_iterations_per_layer[layer_id] * level.active_instances_per_layer[layer_id])
            break
    return WMOPs * (10**-6 if pJ_to_uJ else 1)

"""
Total read and write Memory Operations (MOPs)
"""
def MOPs(arch : Arch) -> tuple[int, int]:
    tot_reads, tot_writes = 0, 0
    for level in arch:
        if isinstance(level, MemLevel):
            reads = level.in_reads + level.w_reads + level.out_reads
            writes = level.in_writes + level.w_writes + level.out_writes
            tot_reads += reads
            tot_writes += writes
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            break
    return tot_reads, tot_writes