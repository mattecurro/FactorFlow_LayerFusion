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
    vprint("é stato chiamato update stats! \n\n")
    assert arch.initialized, f"Arch {arch.name}: architecture not initialized, ensure to call 'initFactors' first."
    
    WMOPs_per_layer = {layer_id: 0 for layer_id in range(arch.coupling.getNumLayers())}
    WMOPs = 0
    num_layers = arch.coupling.getNumLayers()
    temporal_iterations = 1
    temporal_iterations_per_layer = {layer_id: 1 for layer_id in range(num_layers)}
    spatial_iterations = 1
    spatial_iterations_per_layer = {layer_id: 1 for layer_id in range(num_layers)}
    last_in_reads, last_out_reads, last_out_writes, last_w_reads = 0, 0, 0, 0
    last_per_layer_w_reads = {layer_id: 0 for layer_id in range(num_layers)}
    last_per_layer_int_in_reads = {layer_id: 0 for layer_id in range(num_layers - 1)}
    last_per_layer_int_out_reads = {layer_id: 0 for layer_id in range(num_layers - 1)}
    last_per_layer_int_out_writes = {layer_id: 0 for layer_id in range(num_layers - 1)}
    per_layer_w_reads = {layer_id: 0 for layer_id in range(num_layers)}
    per_layer_w_writes = {layer_id: 0 for layer_id in range(num_layers)}
    per_layer_int_in_writes = {layer_id: 0 for layer_id in range(num_layers - 1)}
    per_layer_int_out_writes = {layer_id: 0 for layer_id in range(num_layers - 1)}
    acc_out_reads_factors = 1
    ## Added by me
    total_reads_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    total_writes_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    scaling_per_layer = {layer_id: 1 for layer_id in range(num_layers)}
    scaled_cc_per_all_tiles_per_layer = {layer_id: 1 for layer_id in range(num_layers)}
    cc_per_tile_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    cc_per_all_tiles_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    ideal_bandwidth_read_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    ideal_bandwidth_update_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    ideal_bandwidth_fill_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    ideal_bandwidth_drain_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    latency_read_drain_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    latency_fill_update_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    stall_cycles_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    latency_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    
    WMOPs_another_check = 0
    # NOTE: here we compute total MOPs, not per-instance
    # Starting from Top (DRAM) to Bottom (Compute)
    for i in range(len(arch)):
        level = arch[i]
        # Initialize dataflow per layer per level 
        dataflow_per_layer = {}
        if num_layers > 1:
            for layer_idx in range(num_layers):
                layer_relevant_dims = arch.coupling.relevantDimsForLayer(layer_idx)
                dataflow_per_layer[layer_idx] = [dim for dim in level.dataflow if dim in layer_relevant_dims]
        else:
            dataflow_per_layer[0] = level.dataflow.copy()
        # Calculate results after scaling        
        if isinstance(level, MemLevel):
            # multiply by spatial_iterations too, because memory is replicated spatially
            vprint("\n\nQuesto mops è chiamato da update stats")     
            vprint(f"Level: {level.name}, arch bypasses for the level: {level.bypasses}, in_bp: {level.in_bp}, w_bp: {level.w_bp}, out_bp: {level.out_bp}, int_in_bp: {level.int_in_bp}, int_out_bp: {level.int_out_bp}")      
            #regime_mops = level.MOPs_Regime()
            level_mops = level.MOPs()
            in_reads, per_layer_w_reads, per_layer_int_in_reads, per_layer_int_out_reads, per_layer_int_out_writes, out_reads, out_writes, out_reads_factors = level_mops[:8]
            vprint(f"DEBUG updateStats pre scaling: Level {level.name}:"
                  f"\n  in_reads: {in_reads}"
                  f"\n  per_layer_w_reads: {per_layer_w_reads}"
                  f"\n  per_layer_int_in_reads: {per_layer_int_in_reads}"
                  f"\n  per_layer_int_out_reads: {per_layer_int_out_reads}"
                  f"\n  per_layer_int_out_writes: {per_layer_int_out_writes}"
                  f"\n  out_reads: {out_reads}"
                  f"\n  out_writes: {out_writes}"
                  f"\n  out_reads_factors: {out_reads_factors}"
                  )
            scale_per_layer = {layer_id: temporal_iterations_per_layer[layer_id]*spatial_iterations_per_layer[layer_id] for layer_id in range(num_layers)}
            if level.name == "WeightMemory":
                vprint(f"scale_per_layer[0] = {scale_per_layer[0]}, temporal_iterations_per_layer[0] = {temporal_iterations_per_layer[0]}, spatial_iterations_per_layer[0] = {spatial_iterations_per_layer[0]}")
            in_reads = in_reads * scale_per_layer[0]
            vprint(f"scale_per_layer[0]: {scale_per_layer[0]}, temporal_iterations_per_layer[0]: {temporal_iterations_per_layer[0]}, spatial_iterations_per_layer[0]: {spatial_iterations_per_layer[0]}")
            for layer_id, w_read in per_layer_w_reads.items():
                per_layer_w_reads[layer_id] = w_read * scale_per_layer[layer_id]
            w_reads = sum(per_layer_w_reads.values())
            for layer_id, int_in_read in per_layer_int_in_reads.items():
                per_layer_int_in_reads[layer_id] = int_in_read * scale_per_layer[layer_id+1]
            int_in_reads = sum(per_layer_int_in_reads.values()) 
            for layer_id, int_out_read in per_layer_int_out_reads.items():
                per_layer_int_out_reads[layer_id] = int_out_read * scale_per_layer[layer_id]
            int_out_reads = sum(per_layer_int_out_reads.values())
            for layer_id, int_out_write in per_layer_int_out_writes.items():
                per_layer_int_out_writes[layer_id] = int_out_write * scale_per_layer[layer_id]
            int_out_writes = sum(per_layer_int_out_writes.values())
            out_writes = out_writes * scale_per_layer[num_layers - 1]
            out_reads = out_reads * scale_per_layer[num_layers - 1] 
            out_reads_factors = level_mops[7]*acc_out_reads_factors
            ## TODO: check if there is a per_layer_int_out_reads_factors 
            vprint(f"DEBUG: results after scaling: Level {level.name}: in_reads={in_reads}, w_reads={w_reads}, int_in_reads={int_in_reads}, int_out_reads={int_out_reads}, int_out_writes={int_out_writes}, out_writes={out_writes}, out_reads={out_reads}, out_reads_factors={out_reads_factors}")            
            # Adjust for bias read if needed, update writes, DRAINS management
            if not bias_read and out_reads_factors != 0:
                out_reads = (out_reads*(out_reads_factors - 1))//out_reads_factors
                vprint(f"DEBUG: Level {level.name} adjusted out_reads for no bias read: {out_reads}")
            ## TODO: possible add of bias read for intermediate outputs? I think so
            if 'in' not in level.bypasses:
                in_writes = last_in_reads # reads of the above level are written here
                last_in_reads = in_reads
            else:
                in_writes = 0
            if 'w' not in level.bypasses:
                for layer_id in range(num_layers):
                    per_layer_w_writes[layer_id] = last_per_layer_w_reads.get(layer_id, 0) # reads of the above level are written here
                    last_per_layer_w_reads[layer_id] = per_layer_w_reads[layer_id]
            else:
                per_layer_w_writes = {layer_id: 0 for layer_id in range(num_layers)}
            if 'out' not in level.bypasses:
                level.setAboveMOPs(last_out_reads=last_out_reads, last_out_writes=last_out_writes)
                out_writes += last_out_reads # reads above are written here
                last_out_reads = out_reads
                if not Settings.FREE_DRAINS:
                    out_reads += last_out_writes # writes above where read here
                    last_out_writes = out_writes
            else:
                level.setAboveMOPs(0, 0)
            if 'int_in' not in level.bypasses:
                for layer_id in range(num_layers - 1):
                    per_layer_int_in_writes[layer_id] = last_per_layer_int_in_reads.get(layer_id, 0) # reads of the above level are written here
                    last_per_layer_int_in_reads[layer_id] = per_layer_int_in_reads.get(layer_id, 0)
            else:
                per_layer_int_in_writes = {layer_id: 0 for layer_id in range(num_layers - 1)}
            if 'int_out' not in level.bypasses:
                level.setAboveMOPs(last_per_layer_int_out_reads=last_per_layer_int_out_reads, last_per_layer_int_out_writes=last_per_layer_int_out_writes)
                for layer_id in range(num_layers - 1):
                    per_layer_int_out_writes[layer_id] += last_per_layer_int_out_reads.get(layer_id, 0) # reads of the above level are written here
                    last_per_layer_int_out_reads[layer_id] = per_layer_int_out_reads.get(layer_id, 0)
                    if not Settings.FREE_DRAINS:
                        per_layer_int_out_reads[layer_id] += last_per_layer_int_out_writes.get(layer_id, 0) # writes above are read here
                        last_per_layer_int_out_writes[layer_id] = per_layer_int_out_writes.get(layer_id, 0)
            else:
                vprint(f"DEBUG: Level {level.name} bypasses intermediate storage")
                level.setAboveMOPs(last_per_layer_int_out_reads={layer_id: 0 for layer_id in range(num_layers - 1)}, last_per_layer_int_out_writes={layer_id: 0 for layer_id in range(num_layers - 1)})
            vprint(f"DEBUG updateStats post scaling: Level {level.name}:"
                  f"\n  in_reads: {in_reads:,.0f}"
                  f"\n  per_layer_w_reads: {per_layer_w_reads}"
                  f"\n  w_reads: {w_reads:,.0f}"
                  f"\n  per_layer_int_in_reads: {per_layer_int_in_reads}"
                  f"\n  per_layer_int_in_writes: {per_layer_int_in_writes}"
                  f"\n  int_in_reads: {int_in_reads:,.0f}"
                  f"\n  per_layer_int_out_reads: {per_layer_int_out_reads}"
                  f"\n  per_layer_int_out_writes: {per_layer_int_out_writes}"
                  f"\n  int_out_reads: {int_out_reads:,.0f}"
                  f"\n  out_reads: {out_reads:,.0f}"
                  f"\n  out_writes: {out_writes:,.0f}"
                  f"\n  out_reads_factors: {out_reads_factors}")
            ## Update level MOPs
            level.setMOPs(
                in_reads=in_reads,
                in_writes=in_writes,
                per_layer_w_reads=per_layer_w_reads.copy(),
                per_layer_w_writes=per_layer_w_writes.copy(),
                per_layer_int_in_reads=per_layer_int_in_reads.copy(),
                per_layer_int_in_writes=per_layer_int_in_writes.copy(),
                per_layer_int_out_reads=per_layer_int_out_reads.copy(),
                per_layer_int_out_writes=per_layer_int_out_writes.copy(),
                out_reads=out_reads,
                out_writes=out_writes
            )
            level.temporal_iterations_per_layer = temporal_iterations_per_layer.copy()
            level.temporal_iterations = temporal_iterations
            # Calculate total reads and writes for energy calculation
            total_reads = in_reads + w_reads + int_in_reads + int_out_reads + out_reads
            total_writes = in_writes + sum(per_layer_w_writes.values()) + sum(per_layer_int_in_writes.values()) + int_out_writes + out_writes
            vprint(f"in_writes: {in_writes}, w_writes: {sum(per_layer_w_writes.values())}, int_in_writes: {sum(per_layer_int_in_writes.values())}, int_out_writes: {int_out_writes}, out_writes: {out_writes}")
            level.active_instances_per_layer = spatial_iterations_per_layer.copy()
            level.active_instances = spatial_iterations
            vprint(f"Name of the level: {level.name}, level active instances per layer: {level.active_instances_per_layer}, level active instances: {level.active_instances}")
            # Calculate WMOPs per layer
            if num_layers > 1:    
                for layer_id in range(num_layers):
#                    print(f"WMOPs_per_layer[{layer_id}]: {WMOPs_per_layer[layer_id]:,.3f} before adding level {level.name} contribution")
                    if layer_id == 0:
                        layer_total_reads = in_reads + per_layer_w_reads[layer_id] + per_layer_int_out_reads[layer_id]
                        layer_total_writes = in_writes + per_layer_w_writes[layer_id] + per_layer_int_out_writes[layer_id]
                    elif layer_id == num_layers - 1:
                        layer_total_reads = per_layer_int_in_reads[layer_id - 1] + per_layer_w_reads[layer_id] + out_reads
                        vprint(f"DEBUG: Level {level.name}, Layer {layer_id}, per_layer_int_in_reads{layer_id - 1}: {per_layer_int_in_reads[layer_id - 1]}, per_layer_w_reads{layer_id}: {per_layer_w_reads[layer_id]}, out_reads: {out_reads}")
                        layer_total_writes = per_layer_int_in_writes[layer_id - 1] + per_layer_w_writes[layer_id] + out_writes
                    else:
                        layer_total_reads = per_layer_int_in_reads[layer_id-1] + per_layer_w_reads[layer_id] + per_layer_int_out_reads[layer_id]
                        layer_total_writes = per_layer_int_in_writes[layer_id-1] + per_layer_w_writes[layer_id] + per_layer_int_out_writes[layer_id]
                    total_reads_per_layer[layer_id] = layer_total_reads
                    total_writes_per_layer[layer_id] = layer_total_writes
                    vprint(f"Level {level.name}, Layer {layer_id}: total_reads = {layer_total_reads}, total_writes = {layer_total_writes}")
                    WMOPs_per_layer[layer_id] += level.WMOPs(layer_total_reads, layer_total_writes)
                    #print(f"Level {level.name} WMOPs_per_layer[{layer_id}]: {level.WMOPs(layer_total_reads, layer_total_writes):,.3f}")
                    #print(f"Level {level.name} Accumulated WMOPs_per_layer: {WMOPs_per_layer[layer_id]:,.3f}")
                WMOPs = sum(WMOPs_per_layer.values())
                WMOPs_another_check += level.WMOPs(total_reads, total_writes)
                vprint(f"Check: Per Layer Total reads: {total_reads_per_layer}, total: {sum(total_reads_per_layer.values())}")  
                vprint(f"Check: Per Layer Total writes: {total_writes_per_layer}, \n total: {sum(total_writes_per_layer.values())}")          
                vprint(f"Level {level.name} total WMOPs from per layer sum: {WMOPs:,.3f}")
                vprint(f"Level {level.name} total WMOPs from total reads/writes: {WMOPs_another_check:,.3f}")
            else:
                total_reads = in_reads + w_reads + out_reads
                WMOPs += level.WMOPs(total_reads, total_writes)            
            # temporal iterations' update
            for layer_id in range(num_layers):
                layer_factors_product = 1
                for dim in dataflow_per_layer[layer_id]:
                    layer_factors_product *= level.factors.dimProduct(dim)
                temporal_iterations_per_layer[layer_id] *= layer_factors_product            
            temporal_iterations *= level.factors.fullProduct()
            acc_out_reads_factors *= math.prod(level.factors.dimProduct(dim) for dim in dataflow_per_layer[num_layers-1] if dim not in level.arch.coupling.getFlatOutputCoupling())
            vprint(f"End first part of update for level: {level.name}\n")
        # update of last reads
        elif isinstance(level, FanoutLevel):
            vprint(f"\n\nCalculating Fanout Level: {level.name}, updating spatial and last reads")
            spatial_iterations *= level.factors.fullProduct()
            if num_layers > 1:
                actual_dataflow_per_layer: dict[int, list[str]] = {}
                # actual_dataflow_per_layer and spatial_iterations_per_layer: Initialization
                for layer_id in range(num_layers):
                    spatial_iterations_per_layer[layer_id] *= level.factors.fullLayerProduct(layer_id, arch)
                    # spatial reuse of an operand occurs if the fanout is along a dimension not coupled to such operand,
                    # hence, the operand is read once, but written once per instance (modeled by last_XX_reads)
                    # TODO: add NoC modeling and accumulate data transfer energy here!
                    layer_relevant_dims = arch.coupling.relevantDimsForLayer(layer_id)
                    actual_dataflow_per_layer[layer_id] = [
                        dim for dim in level.dataflow
                        if dim in layer_relevant_dims
                    ]
                # Update last_XX_reads considering spatial reuse
                for layer_id in range(num_layers):
                    for dim in actual_dataflow_per_layer[layer_id]:
                        layer_iterations = level.factors.dimProduct(dim)
                        vprint(f"Fanout Level {level.name}, Layer {layer_id}, dim {dim}, layer_iterations: {layer_iterations}")
                        if layer_id == 0:
                            if dim not in arch.coupling.getFlatInputCoupling():
                                last_in_reads *= layer_iterations
                            if dim not in arch.coupling.getFlatIntermediateOutputCoupling(layer_id):
                                last_per_layer_int_out_reads[layer_id] *= layer_iterations
                                last_per_layer_int_out_writes[layer_id] *= layer_iterations
                            if dim not in arch.coupling.getFlatWeightCoupling(layer_id):
                                last_per_layer_w_reads[layer_id] *= layer_iterations                            
                        # Last layer: has intermediate input and final output
                        elif layer_id == num_layers - 1:
                            if dim not in arch.coupling.getFlatOutputCoupling():
                                last_out_reads *= layer_iterations
                                last_out_writes *= layer_iterations
                            if dim not in arch.coupling.getFlatIntermediateInputCoupling(layer_id - 1):
                                last_per_layer_int_in_reads[layer_id - 1] *= layer_iterations
                            if dim not in arch.coupling.getFlatWeightCoupling(layer_id):
                                last_per_layer_w_reads[layer_id] *= layer_iterations                           
                        # Middle layers: have both intermediate input and output
                        else:
                            if dim not in arch.coupling.getFlatIntermediateInputCoupling(layer_id - 1):
                                last_per_layer_int_in_reads[layer_id - 1] *= layer_iterations
                            if dim not in arch.coupling.getFlatIntermediateOutputCoupling(layer_id):
                                last_per_layer_int_out_reads[layer_id] *= layer_iterations
                                last_per_layer_int_out_writes[layer_id] *= layer_iterations
                            if dim not in arch.coupling.getFlatWeightCoupling(layer_id):
                                last_per_layer_w_reads[layer_id] *= layer_iterations 
            else:
                for dim in level.dataflow:
                    layer_iterations = level.factors.dimProduct(dim)
                    if dim not in arch.coupling.getFlatInputCoupling():
                        last_in_reads *= layer_iterations
                    if dim not in arch.coupling.getFlatOutputCoupling():
                        last_out_reads *= layer_iterations
                        last_out_writes *= layer_iterations
                    if dim not in arch.coupling.getFlatWeightCoupling(0):
                        last_w_reads *= layer_iterations
            vprint(f"End of the first part of Update Stats for fanout level: {level.name}")
        elif isinstance(level, ComputeLevel):
            vprint("\n\nCalculating WMOPs for Compute Level")
            # TODO: remove cost of first output accumulate if bias_read is False!
            # => not needed because the cost of the add is << than the multiply!
            level.temporal_iterations_per_layer = temporal_iterations_per_layer.copy()
            level.active_instances_per_layer = spatial_iterations_per_layer.copy()
            level.temporal_iterations = temporal_iterations
            level.active_instances = spatial_iterations
            for layer_id in range(num_layers):
                WMOPs_per_layer[layer_id] += level.computeCostPerLayer(layer_id, temporal_iterations_per_layer[layer_id]*level.active_instances_per_layer[layer_id])
            vprint(f"Compute Level {level.name} WMOPs_per_layer: {WMOPs_per_layer}")
            ## TODO Comparison check
            vprint(f"WMOPs_till_now before compute level: {WMOPs:,.3f}")
            sum_WMOPS = sum(WMOPs_per_layer.values())
            vprint(f"Compute Level {level.name} total WMOPs from per layer sum: {sum_WMOPS}")
            standard_WMOPs = level.computeCost(temporal_iterations*spatial_iterations)
            vprint(f"Compute Level {level.name} total WMOPs: {standard_WMOPs}")
            ## TODO check if this is the correect one, I think so
            WMOPs += sum_WMOPS
            vprint(f"Total WMOPs after adding compute level: {WMOPs}")
            if WMOPs == standard_WMOPs:
                vprint("Check OK: WMOPs from per-layer sum matches standard WMOPs")
            # compute is meant to be the innermost level
            break
    vprint("END FIRST PART OF UPDATE STATS\n\n")
    # Latency:
    max_latency = 0
    # Compute level latency
    ## DOUBT can ComputeLevel.latency() be != to 1? In that case is it a problem?
    cc_per_tile = arch[len(arch)-1].latency()
    vprint(f"First cc_per_tile, it is from compute = {cc_per_tile}")
    cc_per_tile_per_layer = {layer_id: arch[len(arch)-1].latency() for layer_id in range(num_layers)}
    vprint(f"First cc_per_tile_per_layer it is from compute = {cc_per_tile_per_layer}")
    # Calculate latency from Bottom (Compute) to Top (DRAM)
    for i in range(len(arch) - 2, -1, -1):
        level = arch[i]
        previous_fanout_pe_to_pe_warmup = 0
        if isinstance(level, MemLevel):
            vprint(f"\n\nCalculating Latency for Level: {level.name}")
            scaling = level.active_instances*level.temporal_iterations
            # Version with multi layer
            if num_layers > 1:
                
                #  Calculation: ideal bandwidth per layer, ideal latency per layer (no stalls)
                for layer_id in range(num_layers):
                    vprint(f"\nLatency for layer: {layer_id} of Level: {level.name}")
                    vprint(f"level.active_instances_per_layer[{layer_id}]: {level.active_instances_per_layer[layer_id]}")
                    vprint(f"level.temporal_iterations_per_layer[{layer_id}]: {level.temporal_iterations_per_layer[layer_id]}")
                    scaling_per_layer[layer_id] = level.active_instances_per_layer[layer_id]*level.temporal_iterations_per_layer[layer_id]
                    vprint(f"Scaling per layer[{layer_id}]: {scaling_per_layer[layer_id]:,.0f}")
                    vprint(f"Single Tile latency: cc_per_tile_per_layer[{layer_id}]: {cc_per_tile_per_layer[layer_id]}")
                    cc_per_all_tiles_per_layer[layer_id] = cc_per_tile_per_layer[layer_id]*level.factors.fullLayerProduct(layer_id, arch)
                    vprint(f"Ideal Case, Latency at this level without stalls: cc_per_all_tiles_per_layer[{layer_id}]: {cc_per_all_tiles_per_layer[layer_id]}")
                    scaled_cc_per_all_tiles_per_layer[layer_id] = scaling_per_layer[layer_id]*cc_per_all_tiles_per_layer[layer_id]
                    vprint(f"SCALED cc_per_all_tiles_per_layer[{layer_id}]: {scaled_cc_per_all_tiles_per_layer[layer_id]:,.0f}")                    
                    ideal_bandwidth_read_per_layer[layer_id] = level.getReadPerLayer(layer_id) / scaled_cc_per_all_tiles_per_layer[layer_id]
                    vprint(f"level.getReadPerLayer[{layer_id}](): {level.getReadPerLayer(layer_id):,.0f}")
                    vprint(f"  (For comparison - full reads including accumulation: {level.getReadPerLayerTotal(layer_id):,.0f})")
                    vprint(f"ideal_bandwidth_read_per_layer[{layer_id}]: {ideal_bandwidth_read_per_layer[layer_id]}")

                    Settings.FREE_DRAINS = False
                    ideal_bandwidth_drain_per_layer[layer_id] = level.getDrainPerLayer(layer_id)/scaled_cc_per_all_tiles_per_layer[layer_id] if not Settings.FREE_DRAINS else 0
                    vprint(f"level.getDrainPerLayer[{layer_id}](): {level.getDrainPerLayer(layer_id):,.0f}")       
                    vprint(f"ideal_bandwidth_drain_per_layer[{layer_id}]: {ideal_bandwidth_drain_per_layer[layer_id]}")             
                    
                    ideal_bandwidth_update_per_layer[layer_id] = level.getUpdatePerLayer(layer_id) / scaled_cc_per_all_tiles_per_layer[layer_id]
                    #print(f"level.getUpdatePerLayer[{layer_id}](): {level.getUpdatePerLayer(layer_id):,.0f}")
                    vprint(f"ideal_bandwidth_update_per_layer[{layer_id}]: {ideal_bandwidth_update_per_layer[layer_id]}")
                    # Debug: also show the full update (including accumulation) vs actual update
                    vprint(f"  (For comparison - full updates including accumulation: {level.getUpdatePerLayerTotal(layer_id):,.0f})")

                    ideal_bandwidth_fill_per_layer[layer_id] = level.getFillPerLayer(layer_id)/scaled_cc_per_all_tiles_per_layer[layer_id]
                    vprint(f"level.getFillPerLayer[{layer_id}](): {level.getFillPerLayer(layer_id):,.0f}")
                    vprint(f"ideal_bandwidth_fill_per_layer[{layer_id}]: {ideal_bandwidth_fill_per_layer[layer_id]}")
                # Calculate latency per layer considering stalls due to bandwidth limits
                # Two cases: Sequential Layer Execution or Parallel Layer Execution
                # Sequential Layer Execution Case: sum latencies per layer
                if getattr(Settings, 'SEQUENTIAL_LAYER_EXECUTION', True):
                    vprint("\nStart of calculation of latency:")
                    if level.name == "FeatureMemory":
                        vprint("I am in FeatureMemory level, so I use different formula for latency calculation")
                        total_latency = 0
                        latency_until_now = 0
                    for layer_id in range(num_layers):
                        if level.name == "FeatureMemory":
                            vprint(f"\n\nLayer_id: {layer_id} in {level.name}\n")
                            vprint(f"scaling_per_layer[{layer_id}]: {scaling_per_layer[layer_id]}")
                            total_data_rd = int(level.getReadPerLayer(layer_id)) + level.getDrainPerLayer(layer_id)
                            vprint(f"I AM HERE total_data_rd: {total_data_rd}")
                            latency_read_drain_per_layer[layer_id] = (total_data_rd/scaling_per_layer[layer_id])*(1/level.read_bandwidth) if not Settings.FREE_DRAINS else (int(level.getReadPerLayer(layer_id))/scaling_per_layer[layer_id])*(1/level.read_bandwidth)
                            total_data_fu = level.getFillPerLayer(layer_id) + level.getUpdatePerLayer(layer_id)
                            latency_fill_update_per_layer[layer_id] = (total_data_fu/scaling_per_layer[layer_id])*(1/level.write_bandwidth)                            
                            latency_per_layer[layer_id] = latency_read_drain_per_layer[layer_id] + latency_fill_update_per_layer[layer_id]
                            total_latency += latency_per_layer[layer_id]
                            vprint(f"latency_per_layer[{layer_id}]: {latency_per_layer[layer_id]}, latency_read_drain_per_layer[{layer_id}]: {latency_read_drain_per_layer[layer_id]}, latency_fill_update_per_layer[{layer_id}]: {latency_fill_update_per_layer[layer_id]}")
                            vprint(f"Total latency so far: {total_latency}")
                        else:
                            vprint(f"\n\nLayer_id: {layer_id} in {level.name}\n")
                            vprint(f"ideal_bandwidth_read_per_layer[{layer_id}] + ideal_bandwidth_drain_per_layer[{layer_id}]: {ideal_bandwidth_read_per_layer[layer_id] + ideal_bandwidth_drain_per_layer[layer_id]} vs level.read_bandwidth: {level.read_bandwidth}")
                            if ideal_bandwidth_read_per_layer[layer_id] + ideal_bandwidth_drain_per_layer[layer_id] <= level.read_bandwidth:
                                vprint("I am in Compute Bound")
                                latency_read_drain_per_layer[layer_id] = cc_per_all_tiles_per_layer[layer_id] + previous_fanout_pe_to_pe_warmup*cc_per_tile_per_layer[layer_id]
                                vprint(f"cc_per_all_tiles_per_layer[{layer_id}]: {cc_per_all_tiles_per_layer[layer_id]}")
                                vprint(f"STANDARD CASE latency_read_drain_per_layer[{layer_id}]: {latency_read_drain_per_layer[layer_id]}")
                            else:
                                vprint("I am in Slowdown: Memory Bound")
                                vprint(f"ideal_bandwidth_read_per_layer[{layer_id}]: {ideal_bandwidth_read_per_layer[layer_id]}")
                                vprint(f"ideal_bandwidth_drain_per_layer[{layer_id}]: {ideal_bandwidth_drain_per_layer[layer_id]}")
                                vprint(f"level.read_bandwidth: {level.read_bandwidth}")
                                total_data = level.getReadPerLayer(layer_id) + level.getDrainPerLayer(layer_id)
                                latency_read_drain_per_layer[layer_id] = (total_data/scaling_per_layer[layer_id])*(1/level.read_bandwidth) if not Settings.FREE_DRAINS else (level.getReadPerLayer(layer_id)/scaling_per_layer[layer_id])*(1/level.read_bandwidth)
                                vprint(f"total data (for bandwidth): {total_data}")
                                vprint(f"scaling_per_layer[{layer_id}]: {scaling_per_layer[layer_id]}")
                                vprint(f"SLOWDOWN CASE latency_read_drain_per_layer[{layer_id}]: {latency_read_drain_per_layer[layer_id]}")
                            vprint(f"\nideal_bandwidth_fill_per_layer[{layer_id}] + ideal_bandwidth_update_per_layer[{layer_id}]: {ideal_bandwidth_fill_per_layer[layer_id] + ideal_bandwidth_update_per_layer[layer_id]} vs level.write_bandwidth: {level.write_bandwidth}")
                            if ideal_bandwidth_fill_per_layer[layer_id] + ideal_bandwidth_update_per_layer[layer_id] <= level.write_bandwidth:
                                vprint(f"previous_fanout_pe_to_pe_warmup: {previous_fanout_pe_to_pe_warmup}")
                                latency_fill_update_per_layer[layer_id] = cc_per_all_tiles_per_layer[layer_id] + previous_fanout_pe_to_pe_warmup*cc_per_tile_per_layer[layer_id]
                                vprint(f"STANDARD CASE latency_fill_update_per_layer[{layer_id}]: {latency_fill_update_per_layer[layer_id]}")
                            else: 
                                vprint(f"ideal_bandwidth_fill_per_layer[{layer_id}]: {ideal_bandwidth_fill_per_layer[layer_id]}")
                                vprint(f"ideal_bandwidth_update_per_layer[{layer_id}]: {ideal_bandwidth_update_per_layer[layer_id]}")
                                vprint(f"level.write_bandwidth: {level.write_bandwidth}")
                                # Use bandwidth-specific method (excludes internal accumulation traffic)
                                total_data = level.getFillPerLayer(layer_id) + level.getUpdatePerLayer(layer_id)
                                latency_fill_update_per_layer[layer_id] = (total_data/scaling_per_layer[layer_id])*(1/level.write_bandwidth)
                                vprint(f"total data (for bandwidth): {total_data}")
                                vprint(f"scaling_per_layer[{layer_id}]: {scaling_per_layer[layer_id]}")                        
                                vprint(f"SLOWDOWN CASE latency_fill_update_per_layer[{layer_id}]: {latency_fill_update_per_layer[layer_id]}")
                    vprint(f"\nEnd of calculation of latency.\n")
                else:
                    total_read_bandwidth_demand = sum(ideal_bandwidth_read_per_layer.values()) + sum(ideal_bandwidth_drain_per_layer.values())
                    total_write_bandwidth_demand = sum(ideal_bandwidth_fill_per_layer.values()) + sum(ideal_bandwidth_update_per_layer.values())
                    read_slowdown = 1.0
                    if total_read_bandwidth_demand > level.read_bandwidth:
                        read_slowdown = total_read_bandwidth_demand / level.read_bandwidth
                    write_slowdown = 1.0
                    if total_write_bandwidth_demand > level.write_bandwidth:
                        write_slowdown = total_write_bandwidth_demand / level.write_bandwidth
                    for layer_id in range(num_layers):
                        latency_read_drain_per_layer[layer_id] = cc_per_all_tiles_per_layer[layer_id]*read_slowdown
                        latency_fill_update_per_layer[layer_id] = cc_per_all_tiles_per_layer[layer_id]*write_slowdown
                for layer_id in range(num_layers):
                    if level.name == "FeatureMemory":
                        vprint(f"Calculating per tile final latency and stall for layer {layer_id} in {level.name}: latency_read_drain_per_layer[{layer_id}] = {latency_read_drain_per_layer[layer_id]}, latency_fill_update_per_layer[{layer_id}] = {latency_fill_update_per_layer[layer_id]}")
                        summed_latency = latency_read_drain_per_layer[layer_id] + latency_fill_update_per_layer[layer_id] 
                        latency_per_layer[layer_id] = max(summed_latency, cc_per_all_tiles_per_layer[layer_id])
                        stall_cycles_per_layer[layer_id] = latency_per_layer[layer_id] - cc_per_all_tiles_per_layer[layer_id]
                        vprint(f"Final per tile latency_per_layer[{layer_id}]: {latency_per_layer[layer_id]:,.0f}")
                        vprint(f"Final per tile stall_cycles_per_layer[{layer_id}]: {stall_cycles_per_layer[layer_id]:,.0f}")
                        vprint(f"Ideal case: no stall introduced by this level per this layer: cc_per_all_tiles_per_layer[{layer_id}]: {cc_per_all_tiles_per_layer[layer_id]:,.0f}")
                        
                    else:
                        # Slower between: Read or Write
                        vprint(f"Calculating per tile final latency and stall for layer {layer_id} in {level.name}: latency_read_drain_per_layer[{layer_id}] = {latency_read_drain_per_layer[layer_id]}, latency_fill_update_per_layer[{layer_id}] = {latency_fill_update_per_layer[layer_id]}")
                        latency_per_layer[layer_id] = max(latency_read_drain_per_layer[layer_id], latency_fill_update_per_layer[layer_id])
                        stall_cycles_per_layer[layer_id] = latency_per_layer[layer_id] - cc_per_all_tiles_per_layer[layer_id]
                        vprint(f"Final per tile latency_per_layer[{layer_id}]: {latency_per_layer[layer_id]:,.0f}")
                        vprint(f"Final per tile stall_cycles_per_layer[{layer_id}]: {stall_cycles_per_layer[layer_id]:,.0f}")
                        vprint(f"Ideal case: no stall introduced by this level per this layer: cc_per_all_tiles_per_layer[{layer_id}]: {cc_per_all_tiles_per_layer[layer_id]:,.0f}")
                      
                latency_read_drain_per_layer[layer_id] = latency_read_drain_per_layer[layer_id]*level.temporal_iterations_per_layer[layer_id]
                latency_fill_update_per_layer[layer_id] = latency_fill_update_per_layer[layer_id]*level.temporal_iterations_per_layer[layer_id]
                ideal_bandwidth_read = max(ideal_bandwidth_read_per_layer.values())
                ideal_bandwidth_update = max(ideal_bandwidth_update_per_layer.values())
                ideal_bandwidth_drain = max(ideal_bandwidth_drain_per_layer.values())
                ideal_bandwidth_fill = max(ideal_bandwidth_fill_per_layer.values())
                level.setLatencyPerLayer(
                    latency_read_drain_per_layer = latency_read_drain_per_layer,
                    latency_fill_update_per_layer = latency_fill_update_per_layer,
                    cc_per_tile_per_layer = cc_per_tile_per_layer,
                    stall_cycles_per_layer = stall_cycles_per_layer,
                    ideal_bandwidth_read_per_layer = ideal_bandwidth_read_per_layer,
                    ideal_bandwidth_update_per_layer = ideal_bandwidth_update_per_layer,
                    ideal_bandwidth_fill_per_layer = ideal_bandwidth_fill_per_layer,
                    ideal_bandwidth_drain_per_layer = ideal_bandwidth_drain_per_layer
                )
                previous_fanout_pe_to_pe_warmup = 0
                cc_per_tile_per_layer = {layer_id: latency_per_layer[layer_id] for layer_id in range(num_layers)}
            else:
                scaling_per_layer[0] = level.active_instances*level.temporal_iterations
                cc_per_all_tiles_per_layer[0] = cc_per_tile_per_layer[0]*level.factors.fullProduct()
                scaled_cc_per_all_tiles_per_layer[0] = scaling_per_layer[0]*cc_per_all_tiles_per_layer[0]
                ideal_bandwidth_read = level.getRead()/scaled_cc_per_all_tiles_per_layer[0] # original -more readable- formulation: (level.getRead()/scaling)/(cc_per_tile*level.factors.fullProduct())
                ideal_bandwidth_update = level.getUpdate()/scaled_cc_per_all_tiles_per_layer[0]
                ideal_bandwidth_fill = level.getFill()/scaled_cc_per_all_tiles_per_layer[0]
                ideal_bandwidth_drain = level.getDrain()/scaled_cc_per_all_tiles_per_layer[0] if not Settings.FREE_DRAINS else 0
                vprint(f"ideal_bandwidth_read + ideal_bandwidth_drain: {ideal_bandwidth_read + ideal_bandwidth_drain} vs level.read_bandwidth: {level.read_bandwidth}")
                if ideal_bandwidth_read + ideal_bandwidth_drain <= level.read_bandwidth:
                    vprint("entro nel caso ideale read + drain")
                    latency_read_drain = cc_per_all_tiles_per_layer[0] + previous_fanout_pe_to_pe_warmup*cc_per_tile
                else:
                    latency_read_drain = ((level.getRead() + level.getDrain())/scaling)*(1/level.read_bandwidth) if not Settings.FREE_DRAINS else (level.getRead()/scaling)*(1/level.read_bandwidth)
                vprint(f"ideal_bandwidth_fill + ideal_bandwidth_update: {ideal_bandwidth_fill + ideal_bandwidth_update} vs level.write_bandwidth: {level.write_bandwidth}")
                if ideal_bandwidth_fill + ideal_bandwidth_update <= level.write_bandwidth:
                    vprint("entro nel caso ideale fill + update")
                    latency_fill_update = cc_per_all_tiles_per_layer[0] + previous_fanout_pe_to_pe_warmup*cc_per_tile
                else:
                    latency_fill_update = ((level.getFill() + level.getUpdate())/scaling)*(1/level.write_bandwidth)
                latency = max(latency_read_drain, latency_fill_update)
                stall_cycles = latency - cc_per_all_tiles_per_layer[0]
                level.setLatency(latency_read_drain = latency_read_drain*level.temporal_iterations, latency_fill_update = latency_fill_update*level.temporal_iterations, cc_per_tile = cc_per_tile, stall_cycles = stall_cycles*level.temporal_iterations, ideal_bandwidth_read = ideal_bandwidth_read, ideal_bandwidth_update = ideal_bandwidth_update, ideal_bandwidth_fill = ideal_bandwidth_fill, ideal_bandwidth_drain = ideal_bandwidth_drain)
                previous_fanout_pe_to_pe_warmup = 0
                cc_per_tile_per_layer[0] = latency
        elif isinstance(level, FanoutLevel) and level.pe_to_pe:
            # pe-to-pe forwarding implies that the SA operates as a PIPELINE, which has overall latency equal to that of an operation but a warmup dependent on the mesh size
            previous_fanout_pe_to_pe_warmup = level.mesh - 1    
    # Active instances, leakage, and final latency:
    temporal_iterations_per_layer = {layer_id: 1 for layer_id in range(num_layers)}
    temporal_iterations = 1
    powered_instances_per_layer = {layer_id: 1 for layer_id in range(num_layers)}
    powered_instances = 1
    max_latency_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
    for i in range(len(arch)):
        level = arch[i]
        vprint(f"\n\nFinal WMOPs and Latency calculation for Level: {level.name}")
        if isinstance(level, MemLevel):
                if num_layers > 1:
                    for layer_id in range(num_layers):
                        vprint(f"Level: {level.name}, Layer: {layer_id}")
                        vprint(f"max_latency_per_layer[{layer_id}]: {max_latency_per_layer[layer_id]:,.0f} before update")
                        vprint(f"level.getSettedLatencyPerLayer[{layer_id}](): {level.getSettedLatencyPerLayer(layer_id):,.0f}")
                        max_latency_per_layer[layer_id] = max(max_latency_per_layer[layer_id], level.getSettedLatencyPerLayer(layer_id))
                        WMOPs_per_layer[layer_id] += level.Leakage(level.getSettedLatencyPerLayer(layer_id)) * powered_instances_per_layer[layer_id]
                        temporal_iterations_per_layer[layer_id] *= level.factors.fullLayerProduct(layer_id, arch) 
                    if getattr(Settings, 'SEQUENTIAL_LAYER_EXECUTION', True):
                        max_latency = sum(max_latency_per_layer.values())
                        vprint(f"Level {level.name}: {arch.name}, max_latency (sequential layers): {max_latency:,.0f}")
                    else:
                        max_latency = max(max_latency_per_layer.values())                        
                    WMOPs = sum(WMOPs_per_layer.values())
                else:
                    max_latency = max(max_latency, level.getSettedLatency())
                    WMOPs += level.Leakage(level.getSettedLatency()) * powered_instances
                    temporal_iterations *= level.factors.fullProduct()
        elif isinstance(level, FanoutLevel):
            if num_layers > 1:
                for layer_id in range(num_layers):
                    if level.power_gating_support:
                        powered_instances_per_layer[layer_id] *= level.factors.fullProductPerLayer(layer_id, arch)
                    else:
                        ## TODO: big doubt: mesh per layer????
                        powered_instances_per_layer[layer_id] *= level.mesh
                ## TODO: big big big doubt, max latency===??????
                # For latency, use the maximum temporal iterations across all layers
                max_temporal = max(temporal_iterations_per_layer.values())
                max_latency = max(max_latency, level.latency() * max_temporal)
            else:
                ## TODO: Problem of latency calculation with multiple layers
    #            if num_layers > 1:
                max_latency = max(max_latency, level.latency() * temporal_iterations)
                if level.power_gating_support:
                    powered_instances *= level.factors.fullProduct()
                else:
                    powered_instances *= level.mesh
        elif isinstance(level, ComputeLevel):
            ## TODO: Problem of latency calculation with multiple layers
            ## TODO: Ideal compute latency per layer and then sum or max according to execution mode: 
            vprint(f"Compute level latency: {level.latency():,.0f}")
            if num_layers > 1: 
                compute_latencies = []
                for layer_id in range(num_layers):
                    latency = level.latency() * temporal_iterations_per_layer[layer_id]
                    vprint(f"Compute latency for layer {layer_id}: {latency:,.0f}")
                    compute_latencies.append(latency)
                if getattr(Settings, 'SEQUENTIAL_LAYER_EXECUTION', True):
                    compute_latency = sum(compute_latencies)
                else:
                    compute_latency = max(compute_latencies)
                vprint(f"max_latency initially: {max_latency}")
                vprint(f"compute_latency from compute level: {compute_latency:,.0f}")    
                max_latency = max(max_latency, compute_latency)
                vprint(f"max_latency after compute: {max_latency:,.0f}, before fanout_level")
            else:
                max_latency = max(max_latency, level.latency() * temporal_iterations)
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
    vprint(f"utilization used in WART calculation: {utilization}")
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
    num_layers = arch.coupling.getNumLayers()
    max_latency = 0

    if num_layers > 1:
        max_latency_per_layer = {layer_id: 0 for layer_id in range(num_layers)}
        for level in arch:
            if isinstance(level, MemLevel):
                for layer_id in range(num_layers):
                    # Tile overhead is already included in latency_fill_update from updateStats()
                    max_latency_per_layer[layer_id] = max(max_latency_per_layer[layer_id], level.getSettedLatencyPerLayer(layer_id))
                        
        if getattr(Settings, 'SEQUENTIAL_LAYER_EXECUTION', True):
            max_latency = sum(max_latency_per_layer.values())
        else:
            max_latency = max(max_latency_per_layer.values())
        
        for level in arch:
            if isinstance(level, ComputeLevel):
                compute_latencies = []
                for layer_id in range(num_layers):
                    latency = level.latency() * level.temporal_iterations_per_layer[layer_id]
                    compute_latencies.append(latency)
                if getattr(Settings, 'SEQUENTIAL_LAYER_EXECUTION', True):
                    compute_latency = sum(compute_latencies)
                else:
                    compute_latency = max(compute_latencies)
                max_latency = max(max_latency, compute_latency)
                break
    else:
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
            num_layers = arch.coupling.getNumLayers()
            if level.name == "DRAM":
                #level.in_reads = 2760960+12870
                #level.w_reads = 84320            
                vprint(f"DRAM in_reads set to: {level.in_reads}, w_reads set to: {level.w_reads}, out_reads: {level.out_reads}, int_in_reads: {level.int_in_reads}, int_out_reads: {level.int_out_reads}")
            reads = level.in_reads + level.w_reads + level.out_reads + level.int_in_reads + level.int_out_reads
            writes = level.in_writes + level.w_writes + level.out_writes + level.int_in_writes + level.int_out_writes
        
            if level.name == "DRAM":
                vprint(f"DRAM Reads: {reads}: in:{level.in_reads}, w:{level.w_reads}, out:{level.out_reads}, int_in:{level.int_in_reads}, int_out:{level.int_out_reads}")
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
            reads = level.in_reads + level.w_reads + level.out_reads + level.int_in_reads + level.int_out_reads
            writes = level.in_writes + level.w_writes + level.out_writes + level.int_in_writes + level.int_out_writes
            tot_reads += reads
            tot_writes += writes
        elif isinstance(level, FanoutLevel):
            continue
        elif isinstance(level, ComputeLevel):
            break
    return tot_reads, tot_writes

"""
DRAM-level Memory Operations only (first MemLevel in the architecture).
Returns (reads, writes) for the DRAM level specifically, including
in_reads, w_reads, out_reads, int_in_reads, int_out_reads and their write counterparts.
"""
def DRAM_MOPs(arch : Arch) -> tuple[int, int]:
    for level in arch:
        if isinstance(level, MemLevel):
            reads = level.in_reads + level.w_reads + level.out_reads + level.int_in_reads + level.int_out_reads
            writes = level.in_writes + level.w_writes + level.out_writes + level.int_in_writes + level.int_out_writes
            return reads, writes
    return 0, 0