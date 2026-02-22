"""
Results:

================================================================================
DEPFIN CASE STUDIES 
================================================================================
  Architecture: Depfin-like.
  Feature Memory Bandwidth scales with output tile size tried: BW_scaled = BW_base × (tile_size / 128)
  PE-level registers: WReg (weights), IntReg (intermediates), OutReg (outputs), InReg (inputs).
  Spatial mapping: SARows map Z_i factors; SACols map Q/X (tile).
  The output tile size is automatically set as the largest divisor of Q that fits in pe_cols

================================================================================
DEPFIN CASE STUDIES — FSRCNN 8-layer full fusion
================================================================================


Case Study: Tile Size Width Sweep with Bandwidth Scaling ----- FSRCNN 8-layer full fusion
  DepFiN 16×128 PEs, FMEM=1056KB, WMEM=524KB
  FMEM BW scales as: BW_scaled = BW_base × (tile_size / 128)
    
    python3 experiment_runner.py --sweep-tile-sizes  \
      --workload fsrcnn --fusion full --variant 8layer \
      --wmem-size 19 --fmem-size 72 --pe-rows 16 --pe-cols 128 \
      --tile-sizes 120 64 32 16 8 2>&1 | tee results/DF_CS1_FSRCNN.log

    Tried tile sizes: 120 - 64 - 32 - 16 - 8

    Tile  PE Util   Energy(μJ)    Latency(cc)          EDP     Util   DRAM R   DRAM W
  ----------------------------------------------------------------------
      120    93.8%    8.356e+03      6.156e+06     6.01e+04   10.8%  1573992  8294400
      64     50.0%    8.364e+03      1.154e+07     1.13e+05    0.1%  1573992  8294400
      32     25.0%    8.381e+03      2.308e+07     2.26e+05    0.0%  1573992  8294400
      16     12.5%    8.433e+03      4.622e+07     4.55e+05    0.0%  1573992  8294400
      8      6.2%     8.524e+03      9.244e+07     9.18e+05    0.0%  1573992  8294400

    What changes in the mapping: DRAM Q/X iterations increase, SACols decrease
    
    PE Utilization: decrease (93.8% → 6.2%)
    Energy: slight increase (+6%, from WeightMemory 15 times re-reads)
    Latency: increase ~15× (6.16M → 92.4M cc), more temporal iterations in DRAM.
    EDP: increase ~15.5× (dominated by latency)


    Why: Smaller tiles mean more DRAM Q iterations (more temporal passes over the
    output space). Each pass re-reads ALL weights from WeightMemory to WeightRegister.
    The weight data volume is fixed, but it's re-read 15× more often.

    The ENTIRE +6% in energy increase comes from WeightMemory reads:
    tile=120: WMem reads = 81,181,440    (weights fetched 1× per tile pass)
    tile=8:   WMem reads = 1,217,721,600 (15× more — weights re-fetched every tile)
    WeightMemory               64.3 μJ          963.7 μJ          +899.5 μJ (15×)

    Bandwidth does NOT affect energy — it only affects WHEN data transfers occur
    (latency/stalls), not HOW MANY transfers occur (which determines energy).

    
NOT IN THESIS; USEFUL FOR DEBUG: Case Study: Tile Size Width Sweep NO Bandwidth Scaling
  FSRCNN 8-layer full fusion, DepFiN 16×128 PEs, FMEM=1056KB, WMEM=524KB

    python3 experiment_runner.py --sweep-tile-sizes \
    --workload fsrcnn --fusion full --variant 8layer \
    --tile-sizes 120 64 32 16 8 --no-scale-bandwidth 2>&1

    Tried tile sizes: 120 - 64 - 32 - 16 - 8

    What change in the mapping: DRAM Q/X iterations (increase as tile size decrease), SACols iterations (decrease as tile size decrease)

    PE Utilization: decrease because I map tile_size onto PE columns and the tiles size decrease.
    Energy: increase
    Latency: increase (more temporal iterations in DRAM)
    EDP: increase

Case Study: Architecture Sweep: FMEM Size ----- FSRCNN 8-layer full fusion 
  FSRCNN 8-layer full fusion, DepFiN 16×128 PEs, FMEM= sweep, WMEM=524KB

    python3 experiment_runner.py --sweep-arch \
    --workload fsrcnn --fusion full --variant 8layer \
    --fmem-sizes 256 512 1024 2048 2>&1
    
    Mapping identical across all FMEM sizes

    256KB - 512KB - 1024KB
    Energy: increase : due to the size. Can be useful only when the FMEM is too small and doesn't allow fusion
    Latency: constant



    
Case Study: PE Array Sweep — Increasing PE Rows (fixed cols=128) ----- FSRCNN 8-layer full fusion
  FSRCNN 8-layer full fusion, DepFiN, FMEM=576KB, WMEM=19KB

    python3 experiment_runner.py --sweep-arch  \
    --workload fsrcnn --fusion full --variant 8layer  \
    --fmem-sizes 576 --wmem-sizes 19     --pe-configs 8x128 16x128 32x128  2>&1 | tee results/DF_CS2_FSRCNN.log

    Config   | PEs   | Energy(μJ) | Latency(cc) | EDP
    8×128    | 1024  | 8.67e+03   | 1.148e+07   | 1.16e+05
    16×128   | 2048  | 8.35e+03   | 6.17e+06    | 5.98e+04
    32×128   | 4096  | 8.31e+03   | 5.909e+06   | 5.74e+04

    What changes in the mapping: 
      SARows Z values increase (more output channels parallelized spatially).
      DRAM iterations unchanged (Q=15 for all, same tile_size=64).
      WMEM content unchanged (all weights fit in WMEM for all configs).
      Intermediate output register and output register iterations decrease (more spatial parallelism → fewer temporal iterations at WMEM/FMEM levels).

      FMEM Access Breakdown:
      PE Config  | in_reads      | int_in_reads    | int_out_reads | int_out_writes
      8x128      | 272,160,000   | 1,072,051,200   | 89,164,800    | 89,164,800
      16x128     | 155,520,000   | 539,136,000     | 89,164,800    | 89,164,800
      32x128     |  77,760,000   | 526,694,400     | 89,164,800    | 89,164,800

    

    Energy: slight decrease (~4%, 8→16 rows), then near-constant.
      The total DRAM reads are identical (in=1,555,200, w=18,792, out_w=8,294,400).
      The slight reduction comes from fewer temporal iterations at WMEM/FMEM levels
      (more spatial parallelism → fewer reads at intermediate memory levels).

    Latency: two-stage saturation pattern.
      16→32: ~3.6% improvement (1.110e+07 → 1.070e+07). L7 is already saturated, but
        L6 and L0 (Z_shape=56) are SECONDARY bottlenecks. Z6 goes from 14→28
        (temporal iterations halved from 4→2), producing a small gain.

    Analysis:
    - SARows maps Z0 (first layer output channels, Z0=56). More rows → more Z0
      processed per pass → fewer reloads of Layer 0 input features from FMEM.
    - in_reads halves each step: 272M → 156M → 78M (proportional to passes over Z0).
    - int_in_reads drops sharply 8→16 (1,072M→539M, ~2x) but barely 16→32
      (539M→527M, only 2.3%). The 16→32 benefit saturates because the large
      intermediate activations (layers 2-5 have Z=4) are already covered in 1 pass.
    - int_out_reads/writes constant at 89M (depend on spatial tiling, not Z).
    - WMEM and DRAM completely unchanged across configs.
    - Energy drops -3.7% (8→16) then only -0.5% (16→32): diminishing returns as
      register-level WMOPs (~17.3B total, ~82%) dominate the energy budget.
    - Latency drops sharply 8→16 (1.15e+07→6.12e+06, -46.8%) due to halved
      Z0 passes, then only -3.4% (16→32) as compute parallelism saturates.
    - The large Latency drop at 8→16 is because FSRCNN Layer 0 (C=1, Z=56, R=S=5)
      is the latency bottleneck: with 8 rows, it needs 7 passes; with 16 rows
      (Z0=14, a divisor of 56), it needs only 4 passes → nearly 2x speedup.


Case Study: PE Array Sweep — Increasing PE Cols (fixed rows=16) ----- FSRCNN 8-layer full fusion
  FSRCNN 8-layer full fusion, DepFiN, FMEM=576KB, WMEM=19KB

  In the DepFiN architecture, the tile size is automatically set as the largest divisor of Q that fits in pe_cols

    python3 experiment_runner.py --sweep-arch \
    --workload fsrcnn --fusion full --variant 8layer \
    --fmem-sizes 576 --wmem-sizes 19 \
    --pe-configs 16x64 16x128 16x256 --verbose 2>&1 | tee results/DF_CS3_FSRCNN.log

    Config   | PEs   | Energy(μJ) | Latency(cc) | EDP
    16×64    | 1024  | 8.355e+04  | 1.110e+07   | 1.08e+05    tile 64
    16×128   | 2048  | 8.352e+04  | 6.120e+06   | 5.98e+04    tile 120
    16×256   | 4096  | 8.352e+04  | 3.349e+06   | 3.27e+04    tile 240

    What changes in the mapping:
      SACols Q/X values increase (larger tile fits spatially).
      DRAM Q/X iterations DECREASE: Q=15 → 8 → 8 → 4.
      SARows Z unchanged (always Z7=16, Z6=14, Z5-Z2=12, Z0=14).
      WMEM content unchanged (all weights fit for all configs), but change the accesses:
      WMEM Access (changes with cols):
        16x64  → w_reads = 152,215,200
        16x128 → w_reads =  81,181,440   (-46.7%)
        16x256 → w_reads =  40,590,720   (-73.3%)
      FMEM acceses unchanged.



    Energy: near-constant across all configs
      
    Latency: consistent ~1.8× reduction each time tile size doubles.
      More cols → larger tile → fewer DRAM temporal iterations in Q/X dimensions.
      64→120: DRAM Q from 15→8 iterations. Latency halves (1.11e+07 → 6.12e+06).
      120→128: NO improvement. tile_size stays 120 (128 doesn't divide Q=960 evenly)
      128→256: DRAM Q from 8→4 iterations. Latency halves again (6.12e+06 → 3.35e+06).
    


    
Case Study: PE Array Sweep — Changing the aspect ratio (fixed total PEs=2048)
  FSRCNN 8-layer full fusion, DepFiN, FMEM=576KB, WMEM=19KB

    python3 experiment_runner.py --sweep-arch \
    --workload fsrcnn --fusion full --variant 8layer \
    --fmem-sizes 576 --wmem-sizes 19 \
    --pe-configs 2x1024 4x512 8x256 16x128 32x64 64x32 128x16 256x8 512x4 2x1024 --verbose 2>&1 | tee results/DF_CS4_FSRCNN.log
    


    Config   | Tile | DRAM Q | SARows Z7 | SARows Z6 | SARows Z1-5 
    2×1024   | 960  | 1      | 2         | 2         | 2           
    4×512    | 480  | 2      | 4         | 4         | 4           
    8×256    | 240  | 4      | 8         | 8         | 6           
    16×128   | 120  | 8      | 16        | 14        | 12          
    32×64    | 64   | 15     | 16        | 28        | 12          
    64×32    | 32   | 30     | 16        | 56        | 12          
    128×16   | 16   | 60     | 16        | 56        | 12
    256×8    | 8    | 120    | 16        | 56        | 12
    512×4    | 4    | 240    | 16        | 56        | 12

    PE         Energy (μJ)   Latency (cc)    EDP
    2x1024     1.04e+04      8.89e+06        1.05e+05
    4x512      9.19e+03      5.68e+06        6.02e+04
    8x256      8.66e+03      5.98e+06        6.03e+04
    16x128     8.35e+03      6.12e+06        5.98e+04    ← EDP minimum
    32x64      8.32e+03      1.07e+07        1.04e+05
    64x32      8.32e+03      2.11e+07        2.05e+05
    128x16     8.36e+03      4.22e+07        4.12e+05
    256x8      8.45e+03      8.43e+07        8.32e+05
    512x4      8.63e+03      1.69e+08        1.69e+06

    Energy:   decreases 2x1024→32x64 (–20.0 %), then nearly flat 32x64→64x32,
              then slowly rises again 64x32→512x4 (+3.7 %).
              Minimum at 32x64 / 64x32 (8.32e+03 μJ).

    Latency:  minimum at 4x512 (5.68e+06 cc), close second 8x256 (5.98e+06).
              Rises steeply with more rows: 64x32 = 2.11e+07, 512x4 = 1.69e+08.
              Latency from 4x512→512x4 grows ~30× (tile shrinks 480→4).

    EDP:      minimum at 16x128 (5.98e+04).
              4x512 and 8x256 are very close (6.02e+04, 6.03e+04).
              EDP degrades rapidly beyond 16x128 toward row-heavy configs.

    Column-heavy configs (2x1024, 4x512): FMEM energy is ~4× higher because
    few rows → many Z passes → FMEM features reloaded many times.
    Row-heavy configs (256x8, 512x4): WMEM energy rises because tiny tile
    → many spatial passes → weights reloaded many times.

    
    FeatureMemory:
      PE         in_reads        int_in_reads     int_out_reads  int_out_writes  out_writes
      2x1024     1,088,640,000   3,782,246,400    89,164,800     89,164,800      8,294,400
      4x512        544,320,000   1,891,123,200    89,164,800     89,164,800      8,294,400
      8x256        272,160,000   1,072,051,200    89,164,800     89,164,800      8,294,400
      16x128       155,520,000     539,136,000    89,164,800     89,164,800      8,294,400
      32x64         77,760,000     526,694,400    89,164,800     89,164,800      8,294,400
      64x32         38,880,000     520,473,600    89,164,800     89,164,800      8,294,400
      128x16        38,880,000     520,473,600    89,164,800     89,164,800      8,294,400
      256x8         38,880,000     520,473,600    89,164,800     89,164,800      8,294,400
      512x4         38,880,000     520,473,600    89,164,800     89,164,800      8,294,400

    
    WeightMemory w_reads:
      PE         w_reads
      2x1024      10,147,680
      4x512       20,295,360
      8x256       40,590,720
      16x128      81,181,440
      32x64      152,215,200
      64x32      304,430,400
      128x16     608,860,800
      256x8    1,217,721,600
      512x4    2,435,443,200

      Analysis
      --------
      This sweep reveals the fundamental ROWS vs COLS trade-off in DepFin
      at fixed PE budget (2048 PEs):

        MORE COLS (column-heavy, e.g. 2x1024):
        + Larger spatial tile → fewer tile passes → fewer WMEM w_reads
        + Lower latency per tile pass
        − Very few rows → Z mapped partially → many Z passes
        − FMEM in_reads and int_in_reads explode (features reloaded per Z pass)
        − Higher total energy from FMEM accesses

        MORE ROWS (row-heavy, e.g. 512x4):
        + Full Z mapping → single Z pass → FMEM accesses minimised
        + Lower FMEM energy
        − Tiny spatial tile → many tile passes → WMEM w_reads explode
        − Latency scales linearly with tile passes (169M cc at 512x4)
        − Higher WMEM energy

        SWEET SPOT:
        The EDP-optimal config is 16x128 (EDP=5.98e+04), closely followed
        by 4x512 (6.02e+04) and 8x256 (6.03e+04).

        Energy is minimised around 32x64–64x32 where Z0 is fully mapped (56)
        and the tile is still reasonable (32–64).

        Latency is minimised at 4x512 (5.68e+06 cc) where the tile (480)
        covers nearly half the spatial dimension in one pass.

        The 16x128 config achieves the best EDP by balancing moderate energy
        (only 0.4 % above minimum) with reasonable latency.

        Beyond 64x32, latency degrades rapidly (~2× per halving of cols)
        while energy savings plateau, making row-heavy configs inefficient.


================================================================================
DEPFIN CASE STUDIES — RESNET18 17-layer full fusion
================================================================================


Case Study: Tile Size Sweep with Bandwidth Scaling ----- ResNet18 17-layer full fusion
  DepFiN 16×128 PEs, FMEM=1056KB, WMEM=11264KB (~11MB to fit all 17 layers' weights)
  FMEM BW scales as: BW_scaled = BW_base × (tile_size / 128)
  NOTE: WMEM must be ~11MB for ResNet18 full fusion (vs 524KB for FSRCNN).
        17 layers with up to 512×512×3×3 weights = 10,994,880 bytes total.

    python3 experiment_runner.py --sweep-tile-sizes \
    --workload resnet18 --fusion full --variant 17layer \
    --wmem-size 10738 --fmem-size 266 --pe-rows 16 --pe-cols 128 \
    --tile-sizes 7 1 2>&1

    Only 2 valid tile sizes: Q=7 → divisors are 7 and 1.
    (Cumulative stride = 2⁴ = 16, so output_tile=7 → input_tile=112)

    Tile  | SACols Q | SACols X0 | DRAM Q iters | Energy(μJ) | Latency(cc) | EDP
    7     | Q=7      | X0=112    | 1            | 1.083e+04  | 8.998e+06   | 1.74e+05
    1     | Q=1      | X0=16     | 7            | 1.359e+04  | 6.299e+07   | 1.39e+06

    What changes in the mapping:
      tile=7: SACols maps full spatial dims: Q=7, X13=14, X9=28, X5=56, X0=112
      tile=1: Everything shrinks: Q=1, X13=2, X9=4, X5=8, X0=16
      DRAM Q iterations: 7/7=1 → 7/1=7 (7× more temporal passes)

    Energy: +25% (10,876 → 13,623 μJ).
      DRAM reads are IDENTICAL (11,032,512) at both tile sizes.
      The entire increase comes from WeightMemory re-reads:
        tile=7: WMem reads = 124,916,736  
        tile=1: WMem reads = 874,417,152  (7× — weights re-fetched every DRAM Q pass)

    Latency: 7× worse (8.998e+06 → 6.299e+07 cc).
      Directly proportional to DRAM Q iterations (1 → 7).
      DRAM is the latency bottleneck at both tile sizes.

    EDP: 8× worse (1.74e+05 → 1.39e+06).

    COMPARISON WITH FSRCNN TILE SWEEP:
      FSRCNN had many valid tile sizes (120, 64, 32, 16, 8) → gradual degradation.
      ResNet18 has only 2 valid tiles (7, 1) → binary choice, no middle ground.
      This is because Q=7 with cumulative stride=16 severely limits
      valid tiles.




      
Case Study: PE Array Sweep — Increasing PE Rows (fixed cols=128) ----- ResNet18 17-layer full fusion
  DepFiN, FMEM=1056KB, WMEM=11264KB

    python3 experiment_runner.py --sweep-arch \
    --workload resnet18 --fusion full --variant 17layer \
    --fmem-sizes 266 --wmem-sizes 10738 \
    --pe-configs 16x128 32x128 64x128 128x128  --tile-size 7 2>&1 | tee results/DF_CS2_ResNet18.log

    ResNet18 Z dimensions: Z0=64, Z1-4=64, Z5-8=128, Z9-12=256, Z13-16=512

    Config   | PEs    | SARows Z16 | SARows Z0 | FMem Reads | Energy(μJ) | Latency(cc) | Bottleneck | EDP
    16×128   | 2,048  | 16         | 16        | 147M       | 1.083e+04  | 7.812e+06    | DRAM       | 1.514e+05
    32×128   | 4,096  | 32         | 32        | 75M        | 1.081e+04  | 7.807e+06    | WMem       | 1.511e+05
    64×128   | 8,192  | 64         | 64        | 39M        | 1.079e+04  | 7.807e+06    | WMem       | 1.510e+05
    128×128  | 16,384 | 128        | 64        | 25M        | 1.079e+04  | 7.807e+06    | WMem       | 1.510e+05
    
    What changes in the mapping:
      SARows Z increases with pe_rows: each layer gets Z_i = min(pe_rows, Z_i_shape).
      SACols unchanged.
      DRAM unchanged.

    Latency: 


    Energy: tiny decrease (~2% total, from 10,992 → 10,773 μJ over 8→512 rows).
      What is CONSTANT across all configs (and why):
        - DRAM reads:  11,032,512 (all data fits in on-chip memories, loaded once)
        - WMem reads:  124,916,736 (all weights loaded once from DRAM→WMem→WReg)
        - WReg reads:  2,314,518,528 (total MAC operations, independent of rows)
        - IntReg R+W:  2,201,722,880 each (intermediate activations between layers)
        - OutReg R+W:  115,605,504 each (final output accumulation)
      What CHANGES (only FeatureMemory reads):
        16 rows:  FMem reads = 147,492,352  (halved)
        32 rows:  FMem reads = 75,163,648   (halved again)
        64 rows:  FMem reads = 38,999,296
        128 rows: FMem reads = 25,451,776
        
    ANALYSIS — Why FeatureMemory reads decrease with more PE rows:
      In fused execution, FMEM stores intermediate activations passed between layers.
      Each layer reads its input activations from FMEM (written by the previous layer).
      The input activation volume per layer is: Y_i × X_i × C_i (spatial × input channels).

      SARows distributes Z_i (output channels) across rows. The REMAINING Z_i factor
      that doesn't fit in SARows becomes a temporal loop at the WeightMemory level.
      Each temporal Z iteration requires RE-READING the same input activations from FMEM,
      because each output channel slice needs the full input volume.

      Example for L13 (C13=256, Y13=14, X13=14, Z13=512):
        8 rows:   SARows Z13=8   → temporal Z iterations = 512/8  = 64 → read inputs 64×
        16 rows:  SARows Z13=16  → temporal Z iterations = 512/16 = 32 → read inputs 32×
        64 rows:  SARows Z13=64  → temporal Z iterations = 512/64 = 8 → read inputs 8×
        512 rows: SARows Z13=512 → temporal Z iterations = 512/512= 1 → read inputs 1× (minimum)

      More rows → fewer temporal Z iterations → fewer re-reads of input activations from FMEM.
      But FMEM energy is a small fraction of total energy (~1-2%), so the total impact is tiny.

    COMPARISON WITH FSRCNN PE ROWS SWEEP:
      FSRCNN: Latency showed a two-stage saturation (1.9× at 16 rows, +3.6% at 32).
        Z dimensions are small (12-56), so they saturate quickly. The latency bottleneck
        was DRAM Q iterations (DRAM Q=15 at cols=64), which rows don't affect.
      ResNet18: For ResNet18 with tile=7, the computation is so heavily 
      register/compute-bound that even a dramatic 6.6x reduction in FMEM reads has almost
      zero impact on total Energy or Latency

        
    
Case Study: PE Array Sweep — Increasing PE Cols (fixed rows=16) ----- ResNet18 17-layer full fusion
  ResNet18 17-layer full fusion, DepFiN, FMEM=266KB, WMEM=10739KB

    python3 experiment_runner.py --sweep-arch \
      --workload resnet18 --fusion full --variant 17layer \
      --fmem-sizes 266 --wmem-sizes 10739 \
      --pe-configs 16x128 16x256 16x512 16x1024 \
      --verbose 2>&1 | tee results/DF_CS3_ResNet.log


  PE         Energy (μJ)   Latency (cc)    EDP
  16x128     1.08e+04      7.81e+06        1.51e+05
  16x256     1.08e+04      7.81e+06        1.51e+05
  16x512     1.08e+04      7.81e+06        1.51e+05
  16x1024    1.08e+04      7.81e+06        1.51e+05

   Energy:  0 % change across all configs.
   Latency: 0 % change across all configs.
   EDP:     0 % change across all configs.
    Analysis:
      ResNet18's spatial dimensions (112, 56, 28, 14, 7) all fit within
      the smallest tested pe_cols=128.  The auto-tile selector picks the
      full spatial width for every layer, producing exactly ONE spatial
      pass per layer regardless of how many columns are available.
      Because the tile never changes, no data-reuse or pass-count metrics
      are affected: access counts, WMOPs, latency, and energy are all
      identical across 16x128 through 16x1024.
     
      Conclusion: for ResNet18 17-layer full fusion with these memory
      sizes, increasing PE columns beyond 128 yields ZERO benefit.
      The bottleneck is elsewhere (PE rows / Z-dimension or memory BW).
      The extra columns are entirely wasted.

    
Case Study: PE Array Sweep — Aspect Ratio (fixed total PEs=2048) ----- ResNet18 17-layer full fusion
  ResNet18 17-layer full fusion, DepFiN, FMEM=266KB, WMEM=10738KB

    python3 experiment_runner.py --sweep-arch \
      --workload resnet18 --fusion full --variant 17layer \
      --fmem-sizes 266 --wmem-sizes 10738 \
      --pe-configs 2x1024 4x512 8x256 16x128 32x64 64x32 128x16 256x8 512x4 1024x2\
      --verbose 2>&1 | tee results/DF_CS4_ResNet18.log

    Results:
    
    PE         Energy (μJ)   Latency (cc)    EDP
    2x1024     1.12e+04      6.25e+07        1.23e+06
    4x512      1.10e+04      3.12e+07        6.10e+05
    8x256      1.09e+04      1.56e+07        3.04e+05
    16x128     1.08e+04      7.81e+06        1.51e+05    ← EDP minimum
    32x64      1.08e+04      7.87e+06        1.52e+05
    64x32      1.09e+04      8.78e+06        1.70e+05
    128x16     1.10e+04      1.18e+07        2.31e+05
    256x8      1.15e+04      1.98e+07        3.96e+05
    512x4      1.39e+04      6.15e+07        1.38e+06
    1024x2     1.52e+04      8.32e+07        1.98e+06

    Energy:   decreases 2x1024→16x128 (–3.6 %), nearly flat 16x128→32x64
            (≈0 %), then rises 32x64→1024x2 (+40.7 %).
            Minimum at 16x128 / 32x64 (1.08e+04 μJ).
            Sharp rise at 512x4 and 1024x2 driven by massive WMEM overhead.

    Latency:  minimum at 16x128 (7.81e+06 cc).
              2x1024→16x128: halves each step (perfect 2× scaling from Z passes).
              16x128→32x64: nearly identical (+0.8 %) — tile drops 112→56 for
              early layers but Z0 doubles 16→32, roughly cancelling.
              Beyond 64x32: rises steeply. 1024x2 = 8.32e+07 (10.7× worse).

    EDP:      minimum at 16x128 (1.51e+05), barely ahead of 32x64 (1.52e+05).
              Symmetric degradation toward both extremes.
              EDP at 2x1024 (1.23e+06) ≈ EDP at 512x4 (1.38e+06).

    FeatureMemory:
    PE         in_reads      int_in_reads     int_out_reads  int_out_writes
    2x1024     59,006,976    1,098,252,288    2,809,856      2,809,856
    4x512      29,503,488      549,126,144    2,809,856      2,809,856
    8x256      14,751,744      274,563,072    2,809,856      2,809,856
    16x128      7,375,872      137,281,536    2,809,856      2,809,856
    32x64       3,687,936       68,640,768    2,809,856      2,809,856
    64x32       1,843,968       34,320,384    2,809,856      2,809,856
    128x16      1,843,968       20,772,864    2,809,856      2,809,856
    256x8       1,843,968       16,257,024    2,809,856      2,809,856
    512x4       1,843,968       15,128,064    2,809,856      2,809,856
    1024x2      1,843,968       15,128,064    2,809,856      2,809,856

    in_reads:     drop 59M → 1.8M (–96.9 %). Saturates at 64x32 (Z0=64=full).
    int_in_reads: drop 1,098M → 15M (–98.6 %). Saturates at 512x4/1024x2.
                  Note: does NOT fully saturate at 64x32 because deeper layers
                  (Z=128,256,512) still need multiple Z passes even at rows=64.
                  Full saturation requires rows≥512 to cover Z16=512.
    int_out_reads/writes: constant at 2.8M across all configs.

  WeightMemory w_reads:
    PE         w_reads
    2x1024     124,916,736
    4x512      124,916,736
    8x256      124,916,736
    16x128     124,916,736     ← all 4 identical (all tiles ≤ pe_cols, no extra passes)
    32x64      125,970,432     (+0.8 % — X0=56<112 → 2 passes for layer 0)
    64x32      140,464,128     (+12.4 %)
    128x16     189,041,664     (+51.3 %)
    256x8      316,151,808     (+153.1 %)
    512x4      983,248,896     (+686.8 %)
    1024x2   1,330,667,520     (+964.9 %)

    At pe_cols ≥ 128 all ResNet18 spatial dims fit → constant w_reads.
    Below 128, tiles shrink and w_reads scale with spatial passes.
    512x4→1024x2: smaller jump because deepest layers (Q=7) still
    fit in a single pass; only early layers (X0=112) fragment heavily.


Analysis
--------
ResNet18's multi-scale spatial structure (112→56→28→14→7) creates a
distinctive pattern compared to single-scale workloads (FSRCNN, MC-CNN):

1. FMEM saturation is gradual: Z dimensions range from Z0=64 to Z16=512.
   Even at rows=64 (full Z0), deeper layers still have unsaturated Z.
   FMEM int_in_reads only fully plateau at rows≥512.

2. WMEM w_reads stay constant from 2x1024 through 16x128 because ALL
   layer spatial dims (max 112) fit within pe_cols=128. This makes the
   column-heavy configs unusually efficient — they gain nothing from
   larger tiles but also lose nothing.

3. The latency pattern 2x1024→16x128 shows perfect 2× scaling per
   row-doubling because latency is purely Z-pass-limited when all
   tiles fit in one spatial pass. At 32x64, layer 0's tile drops to
   56, introducing 2 spatial passes and breaking the scaling.

4. The energy "U-shape" is asymmetric:
   - Column-heavy side (2x1024→16x128): mild, only –3.6 % total
     because FMEM WMOPs (762M→407M) are small vs total (~59B).
   - Row-heavy side (64x32→1024x2): steep, +40.7 % because WMEM
     inflates (925M→5,292M).

5. The EDP-optimal 16x128 config coincides with the exact threshold
   where ALL spatial dims fit in pe_cols (max dim 112 < 128). Going
   wider adds unused columns; going narrower fragments tiles and
   inflates WMEM reads. This makes 16x128 a natural sweet spot for
   ResNet18-like architectures with small spatial dimensions.
    

    


================================================================================
DEPFIN CASE STUDIES — MC-CNN 4-layer full fusion
================================================================================

  Network: MC-CNN (Matching Cost CNN for Stereo Vision)
  Architecture: 4 conv layers, all 3×3 filters, all Z_i=32, all strides=1
  Spatial dims: P=376, Q=1242 (very large, stereo image resolution)
  Characteristics: ACTIVATION-DOMINANT — huge spatial dims (1242×376),
    tiny uniform channels (Z=32). Very similar to FSRCNN.

  Layer shapes (all strides=1, so all X_i = Q = 1242):
    Layer  C_i  Z_i  R_i  S_i  X_i    Y_i
    L0     1    32   3    3    1242   376
    L1     32   32   3    3    1242   376
    L2     32   32   3    3    1242   376
    L3     32   32   3    3    1242   376

  Total weights: L0=1×32×3×3=288 + L1-L3=3×32×32×3×3=27,648 = 27,936 params (~55KB)
    → WMEM=524KB is more than sufficient (9.4× overprovisioned)

  Valid tile sizes (divisors of Q=1242 = 2 × 3³ × 23):
    [1, 2, 3, 6, 9, 18, 23, 27, 46, 54, 69, 138, 207, 414, 621, 1242]


Case Study: Tile Size Width Sweep with Bandwidth Scaling ----- MC-CNN 4-layer full fusion
  DepFiN 16×128 PEs, FMEM=522KB, WMEM=29KB
  FMEM BW scales as: BW_scaled = BW_base × (tile_size / 128)
  Auto-selected tile = max divisor of 1242 ≤ 128 = 69

    python3 experiment_runner.py --sweep-tile-sizes \
    --workload mccnn --fusion full --variant 4layer \
  --wmem-size 28 --fmem-size 522 --pe-rows 16 --pe-cols 128     --tile-sizes 69 46 27 18 9 3 2>&1 | tee results/DF_CS1_MCCNN_tile_sweep.log

    Tried tile sizes: 69 - 46 - 27 - 18 - 9 - 3

    Tile  DRAM Q  Energy(μJ)  Latency(cc)       EDP    PE Util
    ----  ------  ----------  -----------  ---------   --------
    69      18    1.127e+04   1.218e+07    1.64e+05     53.9%
    46      27    1.133e+04   1.827e+07    2.46e+05     35.9%
    27      46    1.138e+04   3.114e+07    4.18e+05     21.1%
    18      69    1.142e+04   4.671e+07    6.31e+05     14.1%
     9     138    1.150e+04   9.343e+07    1.27e+06      7.0%
     3     414    1.201e+04   2.803e+08    3.95e+06      2.3%


    Memory reads:
      Tile  DRAM Reads     FMem Reads      WMem Reads
      69    1,979,712   3,485,628,288     756,283,392
      46    1,979,712   3,500,572,032   1,134,425,088
      27    1,979,712   3,500,572,032   1,932,724,224
      18    1,979,712   3,500,572,032   2,899,086,336
       9    1,979,712   3,500,572,032   5,798,172,672
       3    1,979,712   3,500,572,032  17,394,518,016

    What changes in the mapping: DRAM Q/X iterations increase, SACols decrease

    PE Utilization: decrease (53.9% → 2.3%) — smaller tile wastes more PE columns.
    Energy: increase (+6.5%)
    Latency: increase 23× (12.2M → 280M cc), dominated by more DRAM temporal iterations
      Latency scales nearly proportionally to the number of spatial passes:
      tile=69: 1242/69 = 18 passes → 1.218e+07 cc
      tile= 3: 1242/3  = 414 passes → 2.803e+08 cc  (ratio: 23x, close to 414/18 = 23x)
    EDP: increase 24× (dominated by latency)

    ANALYSIS:
      DRAM reads: CONSTANT at 1,979,712 across all tiles.
        All data (inputs + weights + outputs) is loaded once from DRAM regardless of tile.

      FMem reads: nearly CONSTANT (~3.50B) for tiles 46 and below.
        At tile=69, slightly lower (3.49B) — the larger tile provides marginally
        better data reuse within FMEM. But the difference is <0.5%.

      WMem reads: SCALE LINEARLY with 1/tile_size.
        tile=69: WMem reads = 756M
        tile=3:  WMem reads = 17.4B (23× more)
        Smaller tiles mean more DRAM Q iterations (more temporal passes over the
        output space). Each pass re-reads ALL weights from WeightMemory.
        The entire +16.5% energy increase comes from this WMem re-reading.

    COMPARISON WITH FSRCNN:
      Identical behavior pattern: DRAM constant, WMem scales with 1/tile.
      FSRCNN had Q=960 (16 divisors), tile sweep 120→8 gave +6% energy increase.
      MC-CNN has Q=1242 (16 divisors), tile sweep 69→3 gives +16.5% energy increase.
      The larger range (23× WMem increase vs 15×) explains the bigger energy impact.
      Both are activation-dominant: the tile size primarily affects weight re-reads,
      not activation memory traffic (which dominates and is tile-independent).


      
Case Study: PE Array Sweep — Increasing PE Rows (fixed cols=128), MC-CNN 4-layer full fusion
  DepFiN, FMEM=522KB, WMEM=28KB
  Auto tile = max divisor of 1242 ≤ 64 = 54

    python3 experiment_runner.py --sweep-arch \
    --workload mccnn --fusion full --variant 4layer \
    --fmem-sizes 522 --wmem-sizes 28 \
    --pe-configs 8x128 16x128 32x128 2>&1 | tee results/DF_CS2_MCCNN.log

    MC-CNN Z dimensions: Z0=Z1=Z2=Z3=32 (uniform across all layers)

    Config   PEs    Energy(μJ)  Latency(cc)  Bottleneck  FMem Reads       WMem Reads    DRAM Reads  EDP
    8×128    1024   1.170e+04   2.37e+07    DRAM        6,750,145,368     966,362,112   1,979,712    3.28e+05
    16×128   2048   1.132e+04   1.20e+07    WMem        3,500,572,032     966,362,112   1,979,712    1.61e+05
    32×128   4096   1.111e+04   1.19e+07    WMem        1,869,835,968     966,362,112   1,979,712    1.58e+05

    What changes in the mapping:
      SARows Z_i values increase (more output channels parallelized spatially).
      MC-CNN has ALL Z_i = 32, so:
        8 rows:  SARows Z_i = 8  → temporal Z = 32/8  = 4
        16 rows: SARows Z_i = 16 → temporal Z = 32/16 = 2
        32 rows: SARows Z_i = 32 → temporal Z = 32/32 = 1 (SATURATED)

    DRAM reads and WMem reads: CONSTANT across all configs (1.98M and 966M).
    Only FMem reads change: 6.75B → 3.50B → 1.87B (halving with rows, saturates at 32).

    Energy: slight decrease 
      Driven entirely by FMem reads reduction (fewer Z temporal iterations →
      fewer intermediate activation re-reads from FeatureMemory).


    COMPARISON WITH FSRCNN PE ROWS SWEEP:
      FSRCNN: Two-stage saturation (L7 Z=16 saturates at 16 rows, L6 Z=56 secondary).
      MC-CNN: Single-step saturation (all Z_i=32, uniform → all saturate simultaneously).
      Both show the same pattern: rows help little beyond Z_max. The latency bottleneck
      is not Z parallelism but DRAM/WMem bandwidth for activation-dominant workloads.


Case Study: PE Array Sweep — Increasing PE Cols (fixed rows=16), MC-CNN 4-layer full fusion
  DepFiN, FMEM=1056KB, WMEM=524KB

  In DepFiN, tile_size = max divisor of Q (=1242) that fits in pe_cols.
  Q = 1242 = 2 × 3³ × 23, so all chosen pe_cols are divisors of 1242.

    python3 experiment_runner.py --sweep-arch \
    --workload mccnn --fusion full --variant 4layer \
    --fmem-sizes 522 --wmem-sizes 28 \
    --pe-configs 16x64 16x128 16x256 --verbose 2>&1 | tee results/DF_CS3_MCCNN.log

    SACols mapping: Since all strides=1, X_i = Q = 1242 for all layers.
      SACols = min(pe_cols, X_i) = pe_cols (always, since pe_cols ≤ 1242).
      All layers have IDENTICAL spatial mapping (no heterogeneous stages).

    Config   PEs    Tile  DRAM Q  Energy(μJ)  Latency(cc) EDP
    16×64     1024  64     19     1.13e+04   1.52e+07     2.05e+05
    16×128    2048  128     9     1.13e+04   1.20e+07     1.61e+05
    16×256    4096  256     4     1.13e+04   4.15e+06     5.57e+04

    What changes in the mapping:
      SACols X_i/Q values increase (larger tile mapped spatially).
      DRAM Q iterations = Q / tile_size = 1242 / pe_cols → decreases.
      FMem reads: CONSTANT at ~3.50B (activation traffic dominates, unchanged by tile).
      DRAM reads: CONSTANT at 1.98M.
      WMem reads: DECREASE proportionally with tile (fewer weight re-reads per DRAM pass).

    Energy: very slight decrease
      Energy is overwhelmingly dominated by FMem reads (intermediates), which don't change.
      The WMem energy saving from fewer weight re-reads is marginal 

    Latency: monotonic decrease, proportional to 1/DRAM_Q.
      Each halving of DRAM Q iterations roughly halves latency.



Case Study: PE Array Sweep — Aspect Ratio (fixed total PEs=2048) ----- MC-CNN 4-layer full fusion
  DepFiN, FMEM=522KB, WMEM=28KB

    python3 experiment_runner.py --sweep-arch \
    --workload mccnn --fusion full --variant 4layer \
    --fmem-sizes 522 --wmem-sizes 28 \
    --pe-configs 2x1024 4x512 8x256 16x128 32x64 64x32 128x16 256x8 512x4 1024x2 --verbose 2>&1 | tee results/DF_CS4_MCCNN.log

    Tile = largest divisor of Q=1242 fitting in pe_cols.


    Auto-selected tile sizes and DRAM Q iterations:
      Config  Tile  DRAM Q
      2×1024  621     2   
      4×512   414     3   
      8×256   207     6   
      16×128   69    18   
      32×64    54    23   
      64×32    27    46   
      128×16   9    138
      256×8    6    207
      512×4    3    414
      1024×2   2    621

    Results:
      PE         Energy (μJ)   Latency (cc)    EDP
      2x1024     1.39e+04      1.08e+07        1.73e+05
      4x512      1.24e+04      8.10e+06        1.18e+05
      8x256      1.17e+04      8.07e+06        1.11e+05    ← EDP minimum
      16x128     1.13e+04      1.20e+07        1.61e+05
      32x64      1.11e+04      1.52e+07        2.01e+05
      64x32      1.12e+04      3.02e+07        4.01e+05
      128x16     1.13e+04      9.06e+07        1.22e+06
      256x8      1.14e+04      1.36e+08        1.84e+06
      512x4      1.18e+04      2.72e+08        3.78e+06
      1024x2     1.22e+04      4.08e+08        5.82e+06

    ANALYSIS — Trade-off between rows (Z parallelism) and cols (Q tile size):

      Energy:   decreases 2x1024→32x64 (–20.1 %), then nearly flat 32x64→64x32
            (+0.9 %), then slowly rises 64x32→1024x2 (+8.9 %).
            Minimum at 32x64 (1.11e+04 μJ).

      Latency:  minimum at 8x256 (8.07e+06 cc), very close to 4x512 (8.10e+06).
                Rises steeply with more rows: 1024x2 = 4.08e+08 cc (50× worse).
                From 8x256→1024x2: latency grows ~50×.

      EDP:      minimum at 8x256 (1.11e+05).
                4x512 close second (1.18e+05).
                EDP degrades rapidly beyond 16x128 toward row-heavy configs.

    Register WMOPs (WReg+AccumIntOut+AccumOut) dominate (~92-78 % of total).
    → Column-heavy configs (2x1024): FMEM energy 3,564M — 4.8× higher than
      the floor (738M at rows≥32) due to many Z passes reloading features.
    → Row-heavy configs (1024x2): WMEM energy 1,796M — 2.3× higher than
      the floor (778M at 32x64) due to many tile passes reloading weights.
    → FMEM saturates once rows≥32 (Z0=32, full mapping for this FMEM size).
    → Energy minimum at 32x64 where FMEM has saturated and WMEM is still low.

    Access counts per level
  -----------------------
  DRAM (constant across ALL configs):
    in_reads = 466,992   w_reads = 27,936   out_writes = 14,943,744

  AccumIntOutReg (constant across ALL configs):
    int_out_reads = 8,742,090,240   int_out_writes = 8,742,090,240

  AccumOutReg (constant across ALL configs):
    out_writes = 4,303,798,272   out_reads = 4,303,798,272

  WeightRegister w_reads (constant across ALL configs): 13,045,888,512

  FeatureMemory:
    PE         in_reads        int_in_reads     int_out_reads  int_out_writes
    2x1024      67,246,848     6,455,697,408    44,831,232     44,831,232
    4x512       33,623,424     3,227,848,704    44,831,232     44,831,232
    8x256       16,811,712     1,613,924,352    44,831,232     44,831,232
    16x128       8,405,856       806,962,176    44,831,232     44,831,232
    32x64        4,202,928       403,481,088    44,831,232     44,831,232
    64x32        4,202,928       403,481,088    44,831,232     44,831,232
    128x16       4,202,928       403,481,088    44,831,232     44,831,232
    256x8        4,202,928       403,481,088    44,831,232     44,831,232
    512x4        4,202,928       403,481,088    44,831,232     44,831,232
    1024x2       4,202,928       403,481,088    44,831,232     44,831,232

    in_reads:     drop 67M → 4.2M (–93.7 %) as rows increase. Saturates at 32x64.
    int_in_reads: drop 6,456M → 403M (–93.8 %). Saturates at 32x64 (Z0=32=full).
    int_out_reads/writes: constant at 44.8M (independent of aspect ratio).

  WeightMemory w_reads:
    PE         w_reads
    2x1024      21,007,872
    4x512       31,511,808
    8x256       63,023,616
    16x128     189,070,848
    32x64      241,590,528
    64x32      483,181,056
    128x16   1,449,543,168
    256x8    2,174,314,752
    512x4    4,348,629,504
    1024x2   6,522,944,256

    w_reads scale with spatial passes: each halving of pe_cols roughly
    doubles w_reads. From 2x1024→1024x2: 310× increase (21M → 6,523M).

  Analysis
--------
This sweep reveals the ROWS vs COLS trade-off for MC-CNN at fixed
PE budget (2048 PEs):

  MORE COLS (column-heavy, e.g. 2x1024):
  + Larger spatial tile (621) → fewer tile passes → fewer WMEM w_reads (21M)
  + Low latency per tile pass
  − Very few rows (2) → Z0 mapped as 2 → 56 Z passes
  − FMEM int_in_reads explode (6,456M) — features reloaded per Z pass
  − Total energy 25 % higher than minimum

  MORE ROWS (row-heavy, e.g. 1024x2):
  + Full Z mapping (32) from rows≥32 → FMEM accesses minimised
  − Tiny spatial tile (2) → 621 tile passes → WMEM w_reads explode (6,523M)
  − Latency scales linearly with passes (408M cc at 1024x2)
  − Energy rises 10 % above minimum due to WMEM overhead

  SWEET SPOT:
  The EDP-optimal config is 8x256 (EDP=1.11e+05), achieving the best
  balance: tile=207 keeps WMEM w_reads at 63M while Z0=8 keeps FMEM
  int_in_reads at 1,614M — neither extreme dominates.

  Energy minimum is at 32x64 (1.11e+04 μJ) where Z0 fully maps (32)
  saturating FMEM savings, and the tile (54) is still large enough to
  keep WMEM manageable.

  Latency minimum is at 8x256 (8.07e+06 cc) — nearly identical to
  4x512 — where large tiles minimise spatial passes.

  The 8x256 config wins on EDP because its latency advantage (8.07M cc)
  outweighs the slightly higher energy vs 32x64 (+5.4 %).

  Beyond 32x64, latency degrades ~2× per halving of cols while energy
  savings cease (FMEM already saturated), making row-heavy configs
  increasingly inefficient: 1024x2 has 50× worse latency than 8x256
  for only 4.2 % more energy than the minimum.


================================================================================
DEPFIN CASE STUDIES — VGG16 13-layer full fusion
================================================================================

  Architecture: DepFiN-like.
  FMEM = 1056KB, WMEM = 14400KB (~14.1MB to fit all 13 layers' weights).
  PE array = 16×128 (default). Bandwidth = 16 words/cc at each memory level.

  VGG16 13-layer full fusion shape (all R_i=S_i=3, strides via 4× MaxPool):
      Layer    | Name     | C    | Z    | Q×P       | Notes
      L0       | Conv1_1  | 3    | 64   | 224×224   |
      L1       | Conv1_2  | 64   | 64   | 224×224   |
      -------- | Pool1    | ---- | ---- | 224→112   | stride=2
      L2       | Conv2_1  | 64   | 128  | 112×112   |
      L3       | Conv2_2  | 128  | 128  | 112×112   |
      -------- | Pool2    | ---- | ---- | 112→56    | stride=2
      L4       | Conv3_1  | 128  | 256  | 56×56     |
      L5       | Conv3_2  | 256  | 256  | 56×56     |
      L6       | Conv3_3  | 256  | 256  | 56×56     |
      -------- | Pool3    | ---- | ---- | 56→28     | stride=2
      L7       | Conv4_1  | 256  | 512  | 28×28     |
      L8       | Conv4_2  | 512  | 512  | 28×28     |
      L9       | Conv4_3  | 512  | 512  | 28×28     |
      -------- | Pool4    | ---- | ---- | 28→14     | stride=2
      L10      | Conv5_1  | 512  | 512  | 14×14     |
      L11      | Conv5_2  | 512  | 512  | 14×14     |
      L12      | Conv5_3  | 512  | 512  | 14×14     |

    Z dimensions: Z0-Z1=64, Z2-Z3=128, Z4-Z6=256, Z7-Z12=512
    Cumulative stride = 2^4 = 16 (4 MaxPool layers)
    Output Q = 14, valid output tiles: {7, 1} (stride-aware)
    Total weights: 14,710,464 params (~14.1MB)
    WEIGHT-DOMINANT workload (similar to ResNet18, but deeper with 13 vs 17 layers).

    CLI: -w vgg16 -f full -v 13layer


Case Study: Tile Size Sweep with Bandwidth Scaling ----- VGG16 13-layer full fusion
  DepFiN 16×128 PEs, FMEM=568KB, WMEM=14400KB
  FMEM BW scales as: BW_scaled = BW_base × (tile_size / 128)
  NOTE: WMEM = 14400KB to fit all 13 layers' weights (14,710,464 total).

    python3 experiment_runner.py --sweep-tile-sizes \
    --workload vgg16 --fusion full --variant 13layer \
    --wmem-size 14400 --fmem-size 568 --pe-rows 16 --pe-cols 128 \
    --tile-sizes 7 2 1 2>&1

    Q=14, divisors={1,2,7,14}

    Tile  | DRAM Q iters | Energy(μJ) | Latency(cc) | EDP      | WMem Reads
    14    | 1            | 7.607e+04  | 2.728e+07   | 3.82e+06 | 388,878,696
    7     | 2            | 7.724e+04  | 5.024e+07   | 7.09e+06 | 669,634,560
    2     | 7            | 8.326e+04  | 1.705e+08   | 2.51e+07 | 2,116,903,680   
    1     | 14           | 9.396e+04  | 3.516e+08   | 5.55e+07 | 4,687,441,920

    What changes in the mapping:
      tile=14: SACols maps entire output width: Q=14 out of 14. Input tile=224.
      tile=7: SACols maps half the output: Q=7 out of 14. Input tile=112.
      tile=2: SACols maps 2 columns: Q=2 out of 14. Input tile=32.
      tile=1: SACols maps single column: Q=1. Input tile=16.
      DRAM Q iterations: 14/7=2 → 14/1=14 (7× more temporal passes)

    DRAM reads: IDENTICAL at both tiles (14,860,992 = 150,528 in + 14,710,464 w).
      All weights fit in WMEM (14.4MB > 14.1MB total weights), loaded once per tile.
      All inputs fit in FMEM, read once from DRAM.

    FMem reads: IDENTICAL (972,711,936). Intermediate activations are independent
      of tile size because FMEM is large enough to hold all per-tile intermediates.

    WMem reads: 14× increase (669M → 4,687M). Weights are re-fetched from WMEM
      every DRAM Q pass. With 14× more passes (1 → 14), WMem reads scale 14×.
      This drives the entire energy and latency degradation.

    Energy: +21.5% (77,578 → 94,292 μJ). Driven entirely by WMem re-reads.

    Latency: 14× worse (2.731e+07 → 3.516e+08 cc). Directly proportional to
      DRAM Q iterations. DRAM is the latency bottleneck at both tiles.

    EDP: 14.5× worse (3.82e+06 → 5.55e+07).

    COMPARISON WITH RESNET18 TILE SWEEP:
      ResNet18: tile=7 vs tile=1, 7× latency increase, +25% energy.
      VGG16: tile=7 vs tile=1, 7× latency increase, +21.5% energy.
      Nearly identical pattern — both are weight-dominant, stride=16, fully cached.
      The slight energy difference comes from VGG16's larger weight volume (14.7M vs 11M).
      Key insight: For weight-dominant workloads, tile=Q (full width) always best.
      

Case Study: PE Array Sweep — Increasing PE Rows (fixed cols=128) ----- VGG16 13-layer full fusion
  DepFiN, FMEM=568KB, WMEM=14400KB, Tile Size 14

    python3 experiment_runner.py --sweep-arch \
    --workload vgg16 --fusion full --variant 13layer \
    --fmem-sizes 568 --wmem-sizes 14400 \
    --pe-configs  16x128 32x128 64x128 128x128 --tile-size 14 2>&1 | tee DF_CS2_VGG16.log

    VGG16 Z dimensions: Z0-Z1=64, Z2-Z3=128, Z4-Z6=256, Z7-Z12=512

    Config   | PEs    | FMem Reads    | WMem Reads  | Energy(μJ) | Latency(cc) | EDP
    16×128   | 2,048  | 972,711,936   | 388,878,336 | 7.61e+04   | 2.43e+07    | 3.41e+06
    32×128   | 4,096  | 493,129,728   | 388,878,336 | 7.59e+04   | 2.43e+07    | 3.40e+06
    64×128   | 8,192  | 253,338,624   | 388,878,336 | 7.57e+04   | 2.43e+07    | 3.39e+06
    128×128  | 16,384 | 148,571,136   | 388,878,336 | 7.57e+04   | 2.43e+07    | 3.39e+06

    
    DRAM reads: CONSTANT at 14,860,992 across all configs.
    WMem reads: CONSTANT at 388,878,336 across all configs.
    FMem reads: Only FMem changes (halving pattern with doubling rows):
      16→32: 973M → 493M (halved)
      32→64: 493M → 253M (halved)
      64→128: 253M → 149M
      128→256: 149M → 107M
    Despite 7.1x FMEM int_in_reads reduction, total Energy drops only ~0.5%
    because register WMOPs (≈403B total, ~99%) dominate the energy budget.
    Latency is essentially flat (2.43e+07 cc) — compute-bound, not memory-bound.


    Latency: 
      16→512: Latency CONSTANT. 

    Energy: Tiny decrease (~2% total, from 77,200 → 75,700 μJ over 16→512 rows).
      Driven by decreasing FMem reads. More rows → more Z parallelism → fewer
      temporal Z iterations → fewer re-reads of input activations from FMEM.

    IDENTICAL PATTERN TO RESNET18:
      ResNet18: single 2× latency step at 8→16, then flat. Same FMem halving.
      The weight-dominant characteristic dominates: once DRAM and WMem are the
      bottleneck (at ≥16 rows), row parallelism is useless for latency.
      Energy improvement is marginal because FMem is a tiny fraction of total.

      

Case Study: PE Array Sweep — Increasing PE Cols (fixed rows=16) ----- VGG16 13-layer full fusion
  VGG16 13-layer full fusion, DepFiN, FMEM=568KB, WMEM=14400KB

    python3 experiment_runner.py --sweep-arch \
      --workload vgg16 --fusion full --variant 13layer \
      --fmem-sizes 568 --wmem-sizes 14400 \
      --pe-configs 16x128 16x256 16x512 16x1024 \
      --tile-size 14 \
      --verbose 2>&1 | tee results/DF_CS3_VGG16.log

      PE         Energy (μJ)   Latency (cc)    EDP
      16x128     7.61e+04      2.43e+07        3.41e+06
      16x256     7.61e+04      2.38e+07        3.33e+06
      16x512     7.61e+04      2.38e+07        3.33e+06
      16x1024    7.61e+04      2.38e+07        3.33e+06

      Energy:  0 % change across all configs (identical at 7.61e+04 μJ).
      Latency: 16x128→16x256  –2.1 %  |  16x256→16x512/1024  0 %.
      EDP:     16x128→16x256  –2.3 %  |  16x256→16x512/1024  0 %.

      WeightMemory w_reads:
      16x128  → 388,878,336
      16x256  → 380,233,728   (–2.2 %)
      16x512  → 380,233,728   (same as 256)
      16x1024 → 380,233,728   (same as 256)

      The tile change from 112→224 for only 2 of 13 layers produces a
      modest –2.2 % reduction in WMEM w_reads (389M → 380M) and –2.1 %
      latency improvement.  Energy is unmoved because register WMOPs
      (~99 % of total) are invariant to the tile size.

      Conclusion: for VGG16 13-layer full fusion with these memory sizes
      and --tile-size 14, increasing PE columns beyond 256 yields ZERO
      additional benefit.  Even the 128→256 step provides only marginal
      improvement (–2.1 % latency) because only 2 early layers are
      affected (X0, X1) and register-level energy completely dominates.


Case Study: PE Array Sweep — Aspect Ratio (fixed total PEs=2048) ----- VGG16 13-layer full fusion
  VGG16 13-layer full fusion, DepFiN, FMEM=568KB, WMEM=14400KB

    python3 experiment_runner.py --sweep-arch \
      --workload vgg16 --fusion full --variant 13layer \
      --fmem-sizes 568 --wmem-sizes 14400 \
      --pe-configs 2x1024 4x512 8x256 16x128 32x64 64x32 128x16 256x8 512x4 1024x2\
      --verbose 2>&1 | tee results/DF_vgg16_pe_aspect_ratio_sweep.log

    PE         rows  cols   X0_tile  Q_tile(L12)  Z0_mapped  Z0_passes
    2x1024       2  1024     224       14            2         32
    4x512        4   512     224       14            4         16
    8x256        8   256     224       14            8          8
    16x128      16   128     112       14           16          4
    32x64       32    64      56       14           32          2
    64x32       64    32      32       14           64          1
    128x16     128    16      16       14           64          1
    256x8      256     8       8        7           64          1
    512x4      512     4       4        2           64          1
    1024x2    1024     2       2        2           64          1

    Z0 saturates at 64 (= full Z0 dim) once rows ≥ 64.
    X0=224 fits fully at pe_cols ≥ 256. At 128, X0=112 (2 passes).
    Deep layers: Q=14 fits at pe_cols ≥ 16; at 8, Q→7 (2 passes);
    at 4, Q→2 (7 passes); at 2, Q→2 (7 passes).

      
    Results:
      PE         Energy (μJ)   Latency (cc)    EDP
      2x1024     7.93e+04      1.90e+08        2.72e+07
      4x512      7.74e+04      9.51e+07        1.34e+07
      8x256      7.65e+04      4.76e+07        6.68e+06
      16x128     7.61e+04      2.43e+07        3.41e+06
      32x64      7.60e+04      2.69e+07        3.77e+06
      64x32      7.66e+04      3.68e+07        5.17e+06
      128x16     7.85e+04      6.59e+07        9.38e+06
      256x8      8.25e+04      1.27e+08        1.85e+07
      512x4      9.15e+04      2.61e+08        4.06e+07
      1024x2     1.06e+05      4.80e+08        8.15e+07

    Energy:   decreases 2x1024→32x64 (–4.2 %), then rises steeply
            32x64→1024x2 (+39.5 %).
            Minimum at 32x64 (7.60e+04 μJ).

    Latency:  minimum at 16x128 (2.43e+07 cc).
              2x1024→16x128: halves each step (~2× scaling from Z passes).
              16x128→32x64: +10.7 % (X0 tile drops 112→56, adding passes).
              Beyond 64x32: rises steeply. 1024x2 = 4.80e+08 (19.8× worse).

    EDP:      minimum at 16x128 (3.41e+06).
              32x64 close second (3.77e+06, +10.6 %).
              Degrades rapidly toward both extremes, but much more
              steeply on the row-heavy side.
  

    FeatureMemory:
      PE         in_reads      int_in_reads     int_out_reads  int_out_writes
      2x1024     43,352,064    7,629,963,264    13,447,168     13,447,168
      4x512      21,676,032    3,814,981,632    13,447,168     13,447,168
      8x256      10,838,016    1,907,490,816    13,447,168     13,447,168
      16x128      5,419,008      953,745,408    13,447,168     13,447,168
      32x64       2,709,504      476,872,704    13,447,168     13,447,168
      64x32       1,354,752      238,436,352    13,447,168     13,447,168
      128x16      1,354,752      133,668,864    13,447,168     13,447,168
      256x8       1,354,752       92,123,136    13,447,168     13,447,168
      512x4       1,354,752       80,381,952    13,447,168     13,447,168
      1024x2      1,354,752       80,381,952    13,447,168     13,447,168

      in_reads:     drop 43M → 1.4M (–96.9 %). Saturates at 64x32 (Z0=64=full).
      int_in_reads: drop 7,630M → 80M (–98.9 %). Does NOT fully saturate at
                    64x32 — continues dropping to 512x4 because deeper layers
                    (Z=128,256,512) still need multiple Z passes at lower rows.
                    Full saturation at rows≥512.

    WeightMemory w_reads:
      PE         w_reads
      2x1024       380,233,728
      4x512        380,233,728
      8x256        380,233,728     ← all 3 identical (all tiles fit at cols≥256)
      16x128       388,878,336     (+2.3 % — X0=112 not 224, 2 passes for L0-1)
      32x64        430,940,160     (+13.3 %)
      64x32        588,994,560     (+54.9 %)
      128x16     1,054,126,080     (+177.2 %)
      256x8      2,025,676,800     (+432.6 %)
      512x4      4,183,474,176     (+1000.3 %)
      1024x2     7,673,315,328     (+1918.4 %)

      At pe_cols ≥ 256, all VGG16 spatial dims (max 224) fit → constant w_reads.
      Below 256, tiles shrink progressively and w_reads scale with spatial passes.
      The rise is dramatic: 2x1024→1024x2 = 20.2× increase.

  Analysis
    --------
    VGG16's large spatial dimensions (224→112→56→28→14) and deep Z
    dimensions (up to Z12=512) create a pronounced asymmetry:

    1. FMEM saturation is very gradual: Z dims span 64→512 across 13 layers.
      FMEM in_reads saturate at rows=64 (Z0=64 fully mapped), but
      int_in_reads continue dropping until rows=512 (covering Z12=512).
      This gives diminishing but persistent FMEM savings well into
      row-heavy territory.

    2. WMEM inflation is severe: VGG16's large early layers (224×224) at
      tiny tiles (2-8 at pe_cols ≤ 8) create extreme spatial pass counts
      (112 passes at tile=2 for layer 0 alone). WMEM w_reads at 1024x2
      are 20× vs 2x1024.

    3. Register WMOPs rise +23 % from column-heavy to row-heavy (130B→160B),
      larger than ResNet18's +24 % swing, because VGG16 has more layers
      and more total MACs affected by tile fragmentation.

    4. The EDP-optimal 16x128 barely edge out 32x64 (3.41M vs 3.77M)
      because:
      - 16x128 has 4 Z passes but X0=112=full for most layers
      - 32x64 has 2 Z passes but X0=56 (4 spatial passes for layer 0)
      - The latency advantage of fewer passes at 16x128 wins despite
        slightly higher FMEM energy.

    5. VGG16's large weight count
      (14.7M weights) amplifies WMEM reload costs at tiny tiles.

      
      
======================================================================================================
EYERISS CASE STUDIES — ResNet18 17-layer full fusion
======================================================================================================

  Architecture: Eyeriss-like with unified GlobalBuffer (128KB), no FMEM/WMEM split.
  PE-level registers: WReg (weights), IntReg (intermediates), OutReg (outputs), InReg (inputs).
  Spatial mapping: SARows map C_i × S_i × Z_i factors; SACols map Q/X (tile) and Z.
  tile_size=1 

  ResNet18 17-layer shape:
    Z0=64, Z1-Z4=64, Z5-Z8=128, Z9-Z12=256, Z13-Z16=512
    C0=3, C1-C4=64, C5-C8=128, C9-C12=256, C13-C16=512
    All R_i=S_i=3 (except R0=S0=7). Stride-2 at L0, L5, L9, L13.
    With tile=1: X0=16, X5=8, X9=4, X13=2, Q=1. 

  Register footprints at tile_size=1 (measured with unit-sized regs):
    PE config   InReg   WReg    IntReg  OutReg
    84×16       650     14311   209     32
    128×16      332     7159    209     32
    196×16      173     3583    209     32
    324×16      173     3583    209     32      ← footprints saturate at 196 rows
    512×16      101     1795    197     32
    512×64      101     460     52      8       ← more cols → smaller IntReg/OutReg
    84×64       650     3583    53      8

  Key insight: WReg dominates (460–14311 entries) because it stores ALL 17 layers'
  weight residuals (C×R factors not absorbed by SARows). IntReg (52–209) and
  OutReg (8–32) are secondary constraints affected mainly by pe_cols (more cols
  → SACols distributes Z → fewer residual Z iterations at register level).
  InReg (101–650) depends on pe_rows (more rows → fewer C/S iterations).


Eyeriss Case Study 0: Architecture Sweep GB Size:
    python3 experiment_runner.py --sweep-arch \
    --workload resnet --fusion full --variant 17layer \
    --arch-type eyeriss \
    --gb-sizes 48 64 128 256 512 1024 \
    --input-reg 64 --weight-reg 500 --intermediate-reg 500 --output-reg 32 \
    --tile-size 80 --pe-configs 84x16 2>&1 | tee results/eyeriss_gb_sweep.log


Eyeriss Case Study 1: WRegister Size Sweep (400–15000 entries)
  ResNet18 17-layer full fusion, Eyeriss, GB=128KB, tile_size=1
  Non-swept regs: InReg=400, IntReg=300, OutReg=64 (matching fusion canonical)

GlobalBuffer:   Total Reads = 8,141,056, Total Writes = 2,872,576
  
GlobalBuffer:   Total Reads = 8,141,056, Total Writes = 2,872,576
full/17layer, fmem_size: 1056kB, wmem_size: 524kB, pe_rows: 256, pe_cols: 64 -> E=4.71e+04μJ, L=3.04e+06cc, EDP=1.48e+05
  full/17layer, fmem_size: 1056kB, wmem_size: 524kB, pe_rows: 512, pe_cols: 32 -> E=4.73e+04μJ, L=3.04e+06cc, EDP=1.48e+05


  
    python3 experiment_runner.py --sweep-wreg-pe \
    -w resnet18 -f full -v 17layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 400 --intermediate-reg 300 --output-reg 64 \
    --weight-reg-sizes 902 1800 3600 

    SUMMARY (A) — Minimum PEs Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      900     N/A          N/A         N/A        N/A          N/A             N/A
      902     512x32       16,384      Yes        5.102e+04    3.042e+06    1.60e+05
      1800    256×32        8,192      Yes        7.245e+04    3.042e+06    2.18e+05
      3600    256×16         4,096      Yes        1.173e+05    3.059e+06    3.51e+05
      

    SUMMARY (B) — Minimum EDP Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      900     N/A          N/A         N/A        N/A          N/A             N/A
      902     512x32       16,384      Yes        5.102e+04    3.042e+06    1.60e+05      
      1800    256×32       8,192       Yes        7.245e+04    3.042e+06    2.18e+05
      3600    256×16       4,096       Yes        1.211e+05    3.042e+06    3.51e+05
      

    SUMMARY (C) — ALL Feasible Configurations per WReg Size:
      WReg=400: 0 feasible config(s)
      WReg=902: 2 feasible config(s)
         PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         256×   64       16,384        Yes      5.103e+04      3.042e+06       1.63e+05
         512×   32       16,384        Yes      5.062e+04      3.042e+06       1.60e+05
      WReg=1800:  9 configs  (min PEs= 8,192)
      WReg=3600: 18 configs  (min PEs= 4,096)
      WReg=7200: 21 configs  (min PEs= 2,048)


    ANALYSIS —
      Why Tile 7 don't work:
        DRAM: P:7, Y15:7, Y14:7, Y13:14, X13:2, Y12:14, X12:2, Y11:14, X11:2, Y10:14, X10:2, Y9:28, X9:4, Y8:28, X8:4, Y7:28, X7:4, Y6:28, X6:4, Y5:56, X5:8, Y4:56, X4:8, Y3:56, X3:8, Y2:56, X2:8, Y1:56, X1:8, Y0:112, X0:16
        SACols: Q:7, Z16:4
        SARows: S16:3, C16:128
        SACols_15: X15:7, Z15:4
        SARows_15: S15:3, C15:128
        SACols_14: X14:7, Z14:4
        SARows_14: S14:3, C14:128
        SACols_13: X13:7, Z13:4
        SARows_13: S13:3, C13:128
        SACols_12: X12:7, Z12:4
        SARows_12: S12:3, C12:128
        SACols_11: X11:7, Z11:4
        SARows_11: S11:3, C11:128
        SACols_10: X10:7, Z10:4
        SARows_10: S10:3, C10:128
        SACols_9: X9:7, Z9:4
        SARows_9: S9:3, C9:128
        SACols_8: X8:7, Z8:4
        SARows_8: S8:3, C8:128  
        SACols_7: X7:7, Z7:4
        SARows_7: S7:3, C7:128
        SACols_6: X6:7, Z6:4
        SARows_6: S6:3, C6:128
        SACols_5: X5:7, Z5:4
        SARows_5: S5:3, C5:64, Z5:2
        SACols_4: X4:7, Z4:4
        SARows_4: S4:3, C4:64, Z4:2
        SACols_3: X3:7, Z3:4
        SARows_3: S3:3, C3:64, Z3:2
        SACols_2: X2:7, Z2:4
        SARows_2: S2:3, C2:64, Z2:2
        SACols_1: X1:7, Z1:4
        SARows_1: S1:3, C1:64, Z1:2
        SACols_0: X0:7, Z0:4
        SARows_0: S0:7, C0:3, Z0:16
        WRegister: C16:4, R16:3, C15:4, R15:3, C14:4, R14:3, C13:2, R13:3, C12:2, R12:3, C11:2, R11:3, C10:2, R10:3, R9:3, R8:3, R7:3, R6:3, R5:3, R4:3, R3:3, R2:3, R1:3, R0:7
        IntermediateRegister: Z15:128, Z14:128, Z13:128, Z12:64, Z11:64, Z10:64, Z9:64, Z8:32, Z7:32, Z6:32, Z5:16, Z4:8, Z3:8, Z2:8, Z1:8
        OutRegister: Z16:128
    
        It means that Z must me handled by registers!
    
    
    Latency behavior:
      Min-latency saturates at 64×128 = 8,192 PEs (3.042e+06 cc) for WReg ≥ 2000.
      pe_rows=64 is sufficient because ResNet18's layer structure saturates row
      utilization at 64 rows. Beyond that, more rows don't improve latency.
      Unlike VGG16 which also saturates at 64 rows, ResNet18 needs fewer PEs
      for min-latency (8,192 vs 16,384) reflecting its lighter per-layer footprint.

    Energy: Increases with WReg (5.851e+04 → 3.997e+05 μJ at min-PEs across 1K–15K).
      Within each WReg, higher-rows configs have lower energy at same total PEs.
      Energy penalty at min-PEs vs min-latency is small (−2% to −3%).


Eyeriss Case Study 1_block: WRegister Size Sweep (400–8000 entries)
  ResNet18 4-layer block fusion (stage4_b2: L13+L14+L15+L16), Eyeriss, GB=128KB, tile_size=1
  Stage 4 block 2: 256→512→512→512→512, spatial 14×14→7×7→7×7→7×7 (stride-2 at L13)
  Non-swept regs: InReg=700, IntReg=300, OutReg=64 (generous, non-binding)

    python3 experiment_runner.py --sweep-wreg-pe \
    -w resnet18 -f block -v stage4_b2 \
    --tile-size 1 --gb-size 128 \
    --input-reg 700 --intermediate-reg 300 --output-reg 64 \
    --weight-reg-sizes 700 1000 2000 4000 8000

    SUMMARY (A) — Minimum PEs Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      400     N/A          N/A         N/A        N/A          N/A             N/A
      1000    64×256       16,384      Yes        1.312e+04    2.077e+06    2.84e+04
      2000    64×128        8,192      Yes        1.839e+04    2.077e+06    3.90e+04
      4000    64×64         4,096      Yes        3.050e+04    2.077e+06    6.43e+04
      8000    64×32         2,048      Yes        5.551e+04    2.077e+06    1.17e+05

    SUMMARY (B) — Minimum Latency Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      400     N/A          N/A         N/A        N/A          N/A             N/A
      1000    64×256       16,384      Yes        1.312e+04    2.077e+06    2.84e+04
      2000    64×128        8,192      Yes        1.839e+04    2.077e+06    3.90e+04
      4000    64×64         4,096      Yes        3.050e+04    2.077e+06    6.43e+04
      8000    64×32         2,048      Yes        5.551e+04    2.077e+06    1.17e+05

    NOTE: Summary A = Summary B for all WReg sizes. Every feasible PE config achieves
    the same latency (2.077e+06 cc). This is because stage4_b2 has very small spatial
    dimensions (7×7 output, tile=1) and only 4 layers, so there is minimal temporal
    iteration variation — the dataflow saturates regardless of PE shape.

    SUMMARY (C) — ALL Feasible Configurations per WReg Size:
      WReg=400: 0 feasible config(s)
      WReg=1000: 4 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         64×  256       16,384        Yes      1.312e+04      2.077e+06       2.84e+04
        128×  128       16,384        Yes      1.212e+04      2.077e+06       2.58e+04
        256×   64       16,384        Yes      1.170e+04      2.077e+06       2.47e+04
        512×   32       16,384        Yes      1.163e+04      2.077e+06       2.45e+04
      WReg=2000: 11 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         64×  128        8,192        Yes      1.839e+04      2.077e+06       3.90e+04
        128×   64        8,192        Yes      1.792e+04      2.077e+06       3.78e+04
        256×   32        8,192        Yes      1.775e+04      2.077e+06       3.74e+04
        512×   16        8,192        Yes      1.782e+04      2.077e+06       3.74e+04
         80×  128       10,240        Yes      1.839e+04      2.077e+06       3.90e+04
        128×   80       10,240        Yes      1.792e+04      2.077e+06       3.78e+04
        320×   32       10,240        Yes      1.775e+04      2.077e+06       3.74e+04
         64×  256       16,384        Yes      1.944e+04      2.077e+06       4.17e+04
        128×  128       16,384        Yes      1.844e+04      2.077e+06       3.91e+04
        256×   64       16,384        Yes      1.801e+04      2.077e+06       3.80e+04
        512×   32       16,384        Yes      1.795e+04      2.077e+06       3.78e+04
      WReg=4000: 20 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         64×   64        4,096        Yes      3.050e+04      2.077e+06       6.43e+04
        128×   32        4,096        Yes      3.029e+04      2.077e+06       6.38e+04
        256×   16        4,096        Yes      3.026e+04      2.077e+06       6.37e+04
        512×    8        4,096        Yes      3.039e+04      2.077e+06       6.39e+04
         64×   80        5,120        Yes      3.050e+04      2.077e+06       6.43e+04
         80×   64        5,120        Yes      3.050e+04      2.077e+06       6.43e+04
        320×   16        5,120        Yes      3.026e+04      2.077e+06       6.37e+04
        512×   12        6,144        Yes      3.039e+04      2.077e+06       6.39e+04
         80×   80        6,400        Yes      3.050e+04      2.077e+06       6.43e+04
         64×  128        8,192        Yes      3.103e+04      2.077e+06       6.57e+04
        128×   64        8,192        Yes      3.055e+04      2.077e+06       6.44e+04
        256×   32        8,192        Yes      3.039e+04      2.077e+06       6.40e+04
        512×   16        8,192        Yes      3.045e+04      2.077e+06       6.41e+04
         80×  128       10,240        Yes      3.103e+04      2.077e+06       6.57e+04
        128×   80       10,240        Yes      3.055e+04      2.077e+06       6.44e+04
        320×   32       10,240        Yes      3.039e+04      2.077e+06       6.40e+04
         64×  256       16,384        Yes      3.207e+04      2.077e+06       6.83e+04
        128×  128       16,384        Yes      3.108e+04      2.077e+06       6.58e+04
        256×   64       16,384        Yes      3.065e+04      2.077e+06       6.46e+04
        512×   32       16,384        Yes      3.058e+04      2.077e+06       6.44e+04
      WReg=8000: 27 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         64×   32        2,048        Yes      5.551e+04      2.077e+06       1.17e+05
        128×   16        2,048        Yes      5.543e+04      2.077e+06       1.17e+05
        256×    8        2,048        Yes      5.547e+04      2.077e+06       1.17e+05
         80×   32        2,560        Yes      5.551e+04      2.077e+06       1.17e+05
        320×    8        2,560        Yes      5.547e+04      2.077e+06       1.17e+05
        256×   12        3,072        Yes      5.547e+04      2.077e+06       1.17e+05
        320×   12        3,840        Yes      5.547e+04      2.077e+06       1.17e+05
         64×   64        4,096        Yes      5.578e+04      2.077e+06       1.18e+05
        128×   32        4,096        Yes      5.556e+04      2.077e+06       1.17e+05
        256×   16        4,096        Yes      5.553e+04      2.077e+06       1.17e+05
        512×    8        4,096        Yes      5.566e+04      2.077e+06       1.17e+05
         64×   80        5,120        Yes      5.578e+04      2.077e+06       1.18e+05
         80×   64        5,120        Yes      5.578e+04      2.077e+06       1.18e+05
        320×   16        5,120        Yes      5.553e+04      2.077e+06       1.17e+05
        512×   12        6,144        Yes      5.566e+04      2.077e+06       1.17e+05
         80×   80        6,400        Yes      5.578e+04      2.077e+06       1.18e+05
         64×  128        8,192        Yes      5.630e+04      2.077e+06       1.19e+05
        128×   64        8,192        Yes      5.583e+04      2.077e+06       1.18e+05
        256×   32        8,192        Yes      5.566e+04      2.077e+06       1.17e+05
        512×   16        8,192        Yes      5.573e+04      2.077e+06       1.17e+05
         80×  128       10,240        Yes      5.630e+04      2.077e+06       1.19e+05
        128×   80       10,240        Yes      5.583e+04      2.077e+06       1.18e+05
        320×   32       10,240        Yes      5.566e+04      2.077e+06       1.17e+05
         64×  256       16,384        Yes      5.735e+04      2.077e+06       1.22e+05
        128×  128       16,384        Yes      5.635e+04      2.077e+06       1.19e+05
        256×   64       16,384        Yes      5.592e+04      2.077e+06       1.18e+05
        512×   32       16,384        Yes      5.586e+04      2.077e+06       1.18e+05

    ANALYSIS — WReg is the dominant binding constraint:
      With uniform PE grid [4,8,12,16,32,64,80,128,256,320,512]:

      WReg=400: No feasible configuration. The 4-layer block (256→512→512→512→512)
        has a large combined weight volume; even at 512×32 = 16,384 PEs the
        per-PE weight footprint exceeds 400 entries.

      WReg=1000: 4 configs, all at 16,384 PEs (64×256, 128×128, 256×64, 512×32).
        Only the maximum total-PEs tier works. Higher rows → lower energy
        (512×32: 1.163e+04 μJ vs 64×256: 1.312e+04 μJ, −11%).

      WReg=2000: 11 configs, minimum PEs = 8,192 (64×128, 128×64, 256×32, 512×16).
        Best EDP: 256×32 or 512×16 at 3.74e+04.

      WReg=4000: 20 configs, minimum PEs = 4,096 (64×64, 128×32, 256×16, 512×8).
        Best EDP: 256×16 at 6.37e+04.

      WReg=8000: 27 configs, minimum PEs = 2,048 (64×32, 128×16, 256×8).
        Best EDP: 128×16 at 1.17e+05.

    Latency behavior — UNIQUE: constant 2.077e+06 cc across ALL feasible configs.
      Unlike the 17-layer full fusion (which has 3-4 latency tiers), the 4-layer
      block fusion achieves identical latency regardless of PE shape. This is because:
      (1) Small spatial dimensions (7×7 output, tile=1) minimize Q/X tiling variation.
      (2) Only 4 layers of Z=512 — the dataflow can always fully spatialize the
          available parallelism without temporal iteration overhead.
      (3) Summary A = Summary B: there is no PE-count vs latency trade-off.
      Implication: For block fusion, the ONLY design trade-off is PEs vs energy.
      More PEs = marginally more energy (idle PE overhead) with zero latency benefit.

    Energy: Monotonically increases with WReg (1.312e+04 → 5.551e+04 μJ at min-PEs).
      Within each WReg, higher-rows configs consistently have lower energy
      (512×32 < 256×64 < 128×128 < 64×256 at WReg=1000, −11% spread).
      This is because wider rows absorb more C/S factors spatially, reducing
      temporal weight iterations and hence register read/write energy.

    COMPARISON WITH 17-LAYER FULL FUSION (CS1):
      Block fusion (4 layers):  min PEs = 16,384 @ WReg=1000, 2,048 @ WReg=8000
      Full fusion (17 layers):  min PEs = 16,384 @ WReg=1000, 2,048 @ WReg=8000
      Identical min-PEs progression! This is because stage4_b2 contains the
      widest layers (Z=512) that dominate the 17-layer footprint. The other 13
      layers (Z=64-256) add register pressure but don't change the PE threshold.
      Block latency (2.077e+06 cc) is ~32% lower than full fusion (3.042e+06 cc)
      because of 4× fewer layers to process sequentially.


Eyeriss Case Study 1_2layer: WRegister Size Sweep (384–1600 entries)
  ResNet18 2-layer fusion (s4b2: L18+L19), Eyeriss, GB=128KB, tile_size=1
  Stage 4 block 2: 512→512→512, spatial 7×7→7×7 (both layers 3×3 conv, no stride)
  Non-swept regs: InReg=700, IntReg=300, OutReg=64 (generous, non-binding)

    python3 experiment_runner.py --sweep-wreg-pe \
    -w resnet18 -f 2layer -v s4b2 \
    --tile-size 1 --gb-size 128 \
    --input-reg 700 --intermediate-reg 300 --output-reg 64 \
    --weight-reg-sizes 384 770 1600

    SUMMARY (A) — Minimum PEs Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      384     32×512       16,384      Yes        4.261e+03    1.186e+06    5.43e+03
      770     32×256        8,192      Yes        4.638e+03    1.186e+06    5.73e+03
      1600    32×128        4,096      Yes        6.437e+03    1.186e+06    7.82e+03

    SUMMARY (B) — Minimum Latency Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      384     32×512       16,384      Yes        4.261e+03    1.186e+06    5.43e+03
      770     32×256        8,192      Yes        4.638e+03    1.186e+06    5.73e+03
      1600    32×128        4,096      Yes        6.437e+03    1.186e+06    7.82e+03

    NOTE: Summary A = Summary B for all WReg sizes. Every feasible PE config achieves
    the same latency (1.186e+06 cc). Same behavior as block fusion (CS1_block) — the
    small spatial dimensions (7×7, tile=1) and only 2 layers mean the dataflow saturates
    regardless of PE shape.

    SUMMARY (C) — ALL Feasible Configurations per WReg Size:
      WReg=384: 5 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         32×  512       16,384        Yes      4.261e+03      1.186e+06       5.43e+03
         64×  256       16,384        Yes      3.667e+03      1.186e+06       4.55e+03
        128×  128       16,384        Yes      3.376e+03      1.186e+06       4.12e+03
        256×   64       16,384        Yes      3.243e+03      1.186e+06       3.91e+03
        512×   32       16,384        Yes      3.201e+03      1.186e+06       3.84e+03
      WReg=770: 14 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         32×  256        8,192        Yes      4.638e+03      1.186e+06       5.73e+03
         64×  128        8,192        Yes      4.343e+03      1.186e+06       5.29e+03
        128×   64        8,192        Yes      4.202e+03      1.186e+06       5.08e+03
        256×   32        8,192        Yes      4.143e+03      1.186e+06       4.98e+03
        512×   16        8,192        Yes      4.139e+03      1.186e+06       4.97e+03
         32×  320       10,240        Yes      4.638e+03      1.186e+06       5.73e+03
         80×  128       10,240        Yes      4.343e+03      1.186e+06       5.29e+03
        128×   80       10,240        Yes      4.202e+03      1.186e+06       5.08e+03
        320×   32       10,240        Yes      4.143e+03      1.186e+06       4.98e+03
         32×  512       16,384        Yes      5.237e+03      1.186e+06       6.61e+03
         64×  256       16,384        Yes      4.642e+03      1.186e+06       5.73e+03
        128×  128       16,384        Yes      4.351e+03      1.186e+06       5.30e+03
        256×   64       16,384        Yes      4.218e+03      1.186e+06       5.09e+03
        512×   32       16,384        Yes      4.176e+03      1.186e+06       5.02e+03
      WReg=1600: 24 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         32×  128        4,096        Yes      6.437e+03      1.186e+06       7.82e+03
         64×   64        4,096        Yes      6.291e+03      1.186e+06       7.60e+03
        128×   32        4,096        Yes      6.225e+03      1.186e+06       7.50e+03
        256×   16        4,096        Yes      6.204e+03      1.186e+06       7.47e+03
        512×    8        4,096        Yes      6.218e+03      1.186e+06       7.48e+03
         64×   80        5,120        Yes      6.291e+03      1.186e+06       7.60e+03
         80×   64        5,120        Yes      6.291e+03      1.186e+06       7.60e+03
        320×   16        5,120        Yes      6.204e+03      1.186e+06       7.47e+03
        512×   12        6,144        Yes      6.218e+03      1.186e+06       7.48e+03
         80×   80        6,400        Yes      6.291e+03      1.186e+06       7.60e+03
         32×  256        8,192        Yes      6.736e+03      1.186e+06       8.26e+03
         64×  128        8,192        Yes      6.441e+03      1.186e+06       7.83e+03
        128×   64        8,192        Yes      6.299e+03      1.186e+06       7.61e+03
        256×   32        8,192        Yes      6.241e+03      1.186e+06       7.52e+03
        512×   16        8,192        Yes      6.236e+03      1.186e+06       7.51e+03
         32×  320       10,240        Yes      6.736e+03      1.186e+06       8.26e+03
         80×  128       10,240        Yes      6.441e+03      1.186e+06       7.83e+03
        128×   80       10,240        Yes      6.299e+03      1.186e+06       7.61e+03
        320×   32       10,240        Yes      6.241e+03      1.186e+06       7.52e+03
         32×  512       16,384        Yes      7.334e+03      1.186e+06       9.15e+03
         64×  256       16,384        Yes      6.740e+03      1.186e+06       8.27e+03
        128×  128       16,384        Yes      6.449e+03      1.186e+06       7.84e+03
        256×   64       16,384        Yes      6.316e+03      1.186e+06       7.63e+03
        512×   32       16,384        Yes      6.274e+03      1.186e+06       7.56e+03

    ANALYSIS — 2-layer fusion is much cheaper than block/full fusion:
      With uniform PE grid [4,8,12,16,32,64,80,128,256,320,512]:

      WReg=384: 5 configs, all at 16,384 PEs (32×512, 64×256, 128×128, 256×64, 512×32).
        Only the maximum total-PEs tier works. Higher rows → lower energy
        (512×32: 3.201e+03 μJ vs 32×512: 4.261e+03 μJ, −25%).

      WReg=770: 14 configs, minimum PEs = 8,192 (32×256, 64×128, 128×64, 256×32, 512×16).
        Best EDP: 512×16 at 4.97e+03.

      WReg=1600: 24 configs, minimum PEs = 4,096 (32×128, 64×64, 128×32, 256×16, 512×8).
        Best EDP: 256×16 at 7.47e+03.

    Latency — constant 1.186e+06 cc across ALL feasible configs (same as block fusion
      behavior). Summary A = Summary B: no PE-count vs latency trade-off.

    Energy: Monotonically increases with WReg (4.261e+03 → 6.437e+03 μJ at min-PEs).
      Within each WReg, higher-rows configs consistently have lower energy
      (512×32 < 256×64 < 128×128 < 64×256 < 32×512, −25% spread at WReg=384).

    COMPARISON WITH BLOCK FUSION (CS1_block, 4 layers, stage4_b2):
      2-layer fusion: min PEs = 16,384 @ WReg=384, vs block: 16,384 @ WReg=1000
      2-layer @ WReg=384 matches block @ WReg=1000 in PE requirement!
      This shows that 2 layers need ~2.6× less WReg per PE than 4 layers.
      2-layer latency (1.186e+06 cc) is 43% lower than block (2.077e+06 cc).
      2-layer energy at min-PEs (4.261e+03 μJ) is 67% lower than block (1.312e+04 μJ).

      
Eyeriss Case Study 2: IntermediateRegister Size Sweep (50–400 entries)
  ResNet18 17-layer full fusion, Eyeriss, GB=128KB, tile_size=1
  Non-swept regs: InReg=400, WReg=4000, OutReg=64

    python3 experiment_runner.py --sweep-intreg-pe \
    -w resnet18 -f full -v 17layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 400 --weight-reg 4000 --output-reg 64 \
    --intermediate-reg-sizes 30 56 100 200

    SUMMARY (A) — Minimum PEs Configuration:
      IntReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      100     256×64       16,384      Yes        6.944e+03    3.042e+06    2.15e+04
      200     256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04
      400     256×16        4,096      Yes        6.697e+03    3.059e+06    2.06e+04  ← saturation
     1000     256×16        4,096      Yes        6.697e+03    3.059e+06    2.06e+04
      
    SUMMARY (B) — Minimum Latency Configuration:
      IntReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      100     256×64       16,384      Yes        6.944e+03    3.042e+06    2.15e+04
      200     256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04  ← saturation
      400     256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04
      1000    256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04


    ANALYSIS — IntReg is a BINDING constraint for ResNet18 (unlike FSRCNN):
      IntReg stores intermediate activations between adjacent layers: the Z_i
      values not absorbed by SACols. Footprint = 209 at pe_cols=16, drops to
      52 at pe_cols=64 (SACols distributes Z → fewer residual Z iterations).

      IntReg=50: Even pe_cols=64 gives footprint=52 > 50 barely, so needs cols=128
        to push footprint below 50 → 256×128 = 32,768 PEs.

      IntReg=100: pe_cols=64 gives footprint=52 < 100 → 256×64 = 16,384 PEs.

      IntReg=200: pe_cols=32 gives footprint~104 < 200 → 256×32 = 8,192 PEs.

      IntReg=400: pe_cols=16 gives footprint=209 < 400 → 256×16 = 4,096 PEs.
        Saturation: beyond 400 entries, no further PE reduction.

      The pe_cols doubling pattern is identical to WReg: each halving of IntReg
      doubles the required pe_cols or of pe_rows (if other dimension are completely managed).
      This is because SACols distributes Z across column groups — doubling cols 
      halves the per-PE intermediate Z footprint.

    Contrast with FSRCNN: IntReg footprint was only 130 entries (small Z dims).
      For ResNet18 with Z up to 512, the per-PE footprint reaches 209 at pe_cols=16,
      

      
Eyeriss Case Study 3: OutRegister Size Sweep (8–64 entries)
  ResNet18 17-layer full fusion, Eyeriss, GB=128KB, tile_size=1
  Non-swept regs: InReg=400, WReg=4000, IntReg=300

    python3 experiment_runner.py --sweep-outreg-pe \
    -w resnet18 -f full -v 17layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 400 --weight-reg 4000 --intermediate-reg 300 \
    --output-reg-sizes 8 16 32 64

    SUMMARY (A) — Minimum PEs Configuration:
      OutReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      8       256×64       16,384      Yes        6.944e+03    3.042e+06    2.15e+04
      16      256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04
      32      256×16        4,096      Yes        6.697e+03    3.059e+06    2.06e+04  ← saturation
      64      256×16        4,096      Yes        6.697e+03    3.059e+06    2.06e+04

    SUMMARY (B) — Minimum Latency Configuration:
      OutReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      8       256×64       16,384      Yes        6.944e+03    3.042e+06    2.15e+04
      16      256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04
      32      256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04  ← saturation
      64      256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04

    ANALYSIS — OutReg follows the same SACols Z pattern:
      OutReg stores the final layer's output: Z16=512, but SACols distributes Z16
      across column groups. Footprint = 32 at pe_cols=16, drops to 8 at pe_cols=64.

      OutReg=8: footprint=32 at cols=16 → needs cols=64 to reduce to 8 → 256×64.
      OutReg=16: footprint=32 at cols=16 → needs cols=32 → 256×32.
      OutReg=32: footprint=32 ≤ 32 at cols=16 → 256×16 = 4,096 PEs.
      OutReg=64: same as OutReg=32 (excess unused).

      The footprint is small (Z16/SACols_Z_groups) because OutReg only stores
      the output of the LAST layer (L16). Contrast with WReg (ALL layers' weights)
      and IntReg (intermediate activations between ALL pairs of adjacent layers).


  COMBINED ANALYSIS — All three Eyeriss register types for ResNet18:
    pe_cols is the universal PE-reduction lever for all register types.
    Doubling pe_cols halves the per-PE footprint for WReg, IntReg, and OutReg
    because SACols distributes Z across more column groups.

    To reach the minimum PE count (256×16 = 4,096 PEs), ALL three must be satisfied:
      WReg ≥ 4000 (footprint 3583 at 256×16)
      IntReg ≥ 210 (footprint 209 at 256×16)
      OutReg ≥ 32 (footprint 32 at 256×16)
      InReg ≥ 173 (footprint 173 at 256×16, from subagent data)

    The binding hierarchy is: WReg >> IntReg > InReg > OutReg.
    WReg alone determines the PE count when it's undersized (e.g. WReg=400 → N/A),
    because its footprint is 17× larger than IntReg's.

    Comparison with FSRCNN:
      FSRCNN: WReg ≥ 300 sufficient, IntReg/OutReg non-binding. Min PEs = 512 (128×4).
      ResNet18: WReg ≥ 4000, IntReg ≥ 210, OutReg ≥ 32. Min PEs = 4,096 (64×64).
      The 8× PE increase reflects deeper fusion (17 vs 8 layers) and wider channels
      (Z up to 512 vs 56). Register requirements scale with network depth and width.


Eyeriss Case Study 5: PE Aspect Ratio Sweep (fixed total PEs=16384)
  ResNet18 17-layer full fusion, Eyeriss, GB=128KB, tile_size=1
  Registers: InReg=400, WReg=902, IntReg=300, OutReg=64

    python3 experiment_runner.py --sweep-pe-aspect \
    --workload resnet18 --fusion full --variant 17layer \
    --total-pes 16384 \
    --input-reg 400 --weight-reg 902 --intermediate-reg 350 --output-reg 64 \
    --tile-size 1 --gb-size 128 --verbose 2>&1 | tee results/EY_CS5_ResNet18.log

    Infeasible configs: column-heavy (1x4096 through 32x128) and
    extreme row-heavy (512x8 through 4096x1).
        1x16384 (1 row, 16384 cols): SARows has mesh=1, so it can't absorb any S, C, or Z spatially. All those factors go to WRegister, which needs footprint ~10,197 entries but has only 902. Also InRegister overflows (10,216 vs 700)
        through 128x128: SARows mesh absorbs S (3 or 7) but can't absorb enough C factors → WRegister footprint still exceeds 902

    Results:
        PE         Energy (μJ)   Latency (cc)    EDP
        256x64     5.103e+04      3.042e+06       1.63e+05   
        512x32     5.062e+04      3.042e+06       1.60e+05  
        1024x16    5.113e+04      3.042e+06       1.62e+05  
        2048x8     5.136e+04      3.042e+06       1.62e+05  
        4096x4     5.136e+04      3.042e+06       1.62e+05  
        8192x2     5.136e+04      3.042e+06       1.62e+05  
        16384x1    5.136e+04      3.042e+06       1.62e+05  

    Energy:   decreases 64x64→256x16 (–1.7 %).
              InReg reads drop 1.19B→0.47B (–60.3 %), driving savings.
              IntReg R+W rises 4.62B→5.20B (+12.5 %), partially offsetting.
              WReg reads constant (2.31B).
              Why energy varies only slightly (~1.5% range)
                The total MACs are identical (same workload, same total PEs). What changes is the distribution of factors across SARows vs WRegister:

                512x32 (best): SARows absorbs S=3, C=128 for deeper layers → fewer WReg reads (C residual is small), lower WReg energy. SACols Z=32 still covers enough output channels.
                256x64: SARows absorbs S=3, C=64 → more C residual in WReg → slightly more WReg accesses → slightly higher energy.
                2048x8 and beyond: SARows absorbs S=3, C=up to 512, plus some Z → minimal WReg accesses, but the per-access energy of the larger SARows fanout offsets the savings. Energy plateaus at 5.136e+04.

    Latency:  identical at 3.059e+06 cc. GlobalBuffer is the bottleneck
              and its access pattern is invariant.

    EDP:      minimum at 256x16 (6.31e+05). Monotonically decreasing
              toward more rows, same trend as VGG16.

    Per-level access counts (reads):
      PE         GB Reads    InReg Reads     WReg Reads      IntReg R+W
      256x16     8,141,056     472,055,808   2,314,518,528   5,202,247,680

    GlobalBuffer reads: constant (8.1M).
    WReg reads: constant (2.31B).
    InReg reads: drop 2.5× (1.19B→0.47B) — more rows map C×S more
    completely, reducing InReg residuals.
    IntReg R+W: rise +12.5 % — same mechanism as VGG16 (fewer cols
    push Z iterations to IntReg level).

    For ResNet18 full fusion at 16,384 PEs with tile=1, 
    the aspect ratio barely matters (~1.5% energy range). 
    The system is DRAM-bandwidth-bound — latency is completely flat,
    and energy differences come only from minor rebalancing of
    register-level reads driven by how SARows splits C vs Z across
    the row dimension.

    Key insight: ResNet18 and VGG16 share the same 3-config feasibility
    window at 4096 PEs. Both favour 256x16 (more rows, fewer cols).
    The underlying reason is identical: Eyeriss's SARows must accommodate
    C×S×Z factor products, which requires ≥64 rows for deep networks.
    Meanwhile, WReg capacity (storing C×R residuals for all 17 layers)
    caps row count at 256 before pe_cols becomes too small to distribute
    Z across SACols. This dual squeeze leaves a narrow operating region
    — a fundamental constraint of Eyeriss's fixed register hierarchy
    that DepFiN's dedicated WMEM avoids entirely.



================================================================================
EYERISS CASE STUDIES — VGG16 13-layer full fusion
================================================================================

  Architecture: Eyeriss-like with unified GlobalBuffer (128KB), no FMEM/WMEM split.
  PE-level registers: WReg (weights), IntReg (intermediates), OutReg (outputs), InReg (inputs).
  Spatial mapping: SARows map C_i × S_i × Z_i factors; SACols map Q/X (tile) and Z.
  tile_size=1 (same as ResNet18 — both have stride=16, Q=14, making tile=1 the
  worst-case analysis; tile=7 would halve footprints via SACols Q coverage).

  VGG16 13-layer shape:
    Z0-Z1=64, Z2-Z3=128, Z4-Z6=256, Z7-Z12=512
    C0=3, C1=64, C2=64, C3=128, C4=128, C5-C6=256, C7=256, C8-C12=512
    All R_i=S_i=3. Stride-2 via MaxPool after L1, L3, L6, L9.
    With tile=1: X0=16, X2=8, X4=4, X7=2, X10=1, Q=1.

  PE grid searched: rows=[4,8,12,16,32,64,80,128,256,320,512], cols=[4,8,12,16,32,64,80,128,256,320,512]
    121 combos per register size (filtered by max_total_pes=17000).

  Register footprints at tile_size=1 (measured with unit-sized regs):
    PE Config   InReg   WReg    IntReg  OutReg
    84×16       702     19,155  229     32
    84×32       702      9,579  115     16
    84×64       702      4,791   58      8
    84×128      702      2,403   30      4
    128×16      354      9,579  229     32
    128×32      354      4,791  115     16
    128×64      354      2,397   58      8
    128×128     354      1,203   30      4
    196×16      180      4,791  229     32      ← rows saturate at 196
    196×32      180      2,397  115     16
    196×64      180      1,200   58      8
    196×128     180        603   30      4
    256×16      180      4,791  229     32      ← same as 196
    324×16      180      4,791  229     32
    512×16      180      4,791  229     32

  Key insights:
    - WReg dominates (603–19,155 entries) — stores ALL 13 layers' weight residuals.
      VGG16 WReg footprint is 33% LARGER than ResNet18's (4,791 vs 3,583 at 256×16)
      despite fewer layers (13 vs 17). This is because VGG16 has more total weights
      (14.7M vs 11M) and the per-PE weight storage reflects total weight volume.
    - Rows saturate at 196: from 196→512 rows, all footprints are identical.
      This means SARows absorbs all available C×S×Z factors by 196 rows.
    - IntReg (30–229) depends only on pe_cols: pe_cols=16 → 229, cols=32 → 115, etc.
      Halving cols doubles IntReg footprint (SACols distributes Z across groups).
    - OutReg (4–32) follows same pe_cols pattern: 32/16/8/4 at 16/32/64/128 cols.
    - InReg (180–702) depends only on pe_rows: 702 at 84, 354 at 128, 180 at 196+.

  Comparison with ResNet18 footprints:
    ResNet18 at 256×16: WReg=3,583, IntReg=209, OutReg=32, InReg=180
    VGG16 at 256×16:    WReg=4,791, IntReg=229, OutReg=32, InReg=180
    WReg is 33% larger (+1,208), IntReg is 10% larger (+20), OutReg identical.
    The WReg increase reflects VGG16's heavier per-layer weight contribution
    (6 layers at 512×512×3×3 vs ResNet18's mixed projection/conv structure).


Eyeriss Case Study 1: WRegister Size Sweep (500–20000 entries)
  VGG16 13-layer full fusion, Eyeriss, GB=128KB, tile_size=1
  Non-swept regs: InReg=400, IntReg=300, OutReg=64 (matching fusion canonical)

    python3 experiment_runner.py --sweep-wreg-pe \
    -w vgg16 -f full -v 13layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 400 --intermediate-reg 300 --output-reg 64 \
    --weight-reg-sizes 1200 2400 4800 

    SUMMARY (A) — Minimum PEs Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      1200    512x32        16384      Yes        4.430e+05    5.455e+06    2.51e+06
      2400    256x32        8,192      Yes        6.232e+05    5.681e+06    3.58e+06
      4800    256x16        4,096      Yes        1.015e+06    6.715e+06    6.85e+06
     

    SUMMARY (B) — Minimum EDP Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      1200    512x32        16384      Yes        4.430e+05    5.455e+06    2.51e+06
      2400    512x32        16384      Yes        6.444e+05    5.455e+06    3.57e+06
      4800    512x32        16384      Yes        1.047e+06    5.455e+06    5.77e+06

          ANALYSIS — WReg is the dominant binding constraint:
      WReg footprint = Σ(13 layers) of residual C×R factors not absorbed by SARows.
      With uniform PE grid [4,8,12,16,32,64,80,128,256,320,512]:

      WReg=500/1000: No feasible configuration exists even at maximum grid size.
        VGG16's 13 conv layers have very large filter counts (up to 512 channels);
        the combined weight register pressure exceeds these budgets for all
        PE configurations in the grid.

      WReg=2500: 11 configs feasible, minimum PEs = 64×128 = 8,192.
        At 8,192 PEs four row×col combos work: 64×128, 128×64, 256×32, 512×16.
        All achieve same latency tier (5.681e+06 cc) except 256+ col configs
        which reach the optimal 5.455e+06 cc. Best EDP: 512×32 at 3.67e+06.

      WReg=5000: 20 configs feasible, minimum PEs = 4,096 (64×64, 128×32, 256×16, 512×8).
        Two latency tiers: 6.715e+06 cc (≤6,400 PEs) and 5.681e+06/5.455e+06 cc
        (≥8,192 PEs). Best EDP: 512×32 at 5.96e+06.

      WReg=10000: 27 configs feasible, minimum PEs = 2,048 (64×32, 128×16, 256×8).
        Three latency tiers: 1.103e+07 (≤3,840 PEs), 6.715e+06 (4,096–6,400 PEs),
        and 5.681e+06/5.455e+06 (≥8,192 PEs). Best EDP: 512×32 at 1.05e+07.

      WReg=20000: 31 configs feasible, minimum PEs = 1,024 (64×16, 128×8).
        Four latency tiers: 2.010e+07 (≤1,536 PEs), 1.103e+07 (2,048–3,840 PEs),
        6.715e+06 (4,096–6,400 PEs), 5.681e+06/5.455e+06 (≥8,192 PEs).
        Best EDP: 256×64 or 512×32 at 1.97e+07.

    Latency behavior:
      Min-latency saturates at 64×256 = 16,384 PEs (5.455e+06 cc) for WReg ≥ 2500.
      pe_rows=64 is the smallest row count that reaches this optimal latency
      when paired with 256 cols. Larger rows (128, 256, 512) also reach 5.455e+06 cc
      at 64+ cols, but require the same or more total PEs.

    Energy: Increases monotonically with WReg (6.716e+05 → 3.549e+06 μJ at min-PEs).
      Within each WReg, higher-rows configs (256, 512) have slightly lower energy
      than lower-rows configs (64, 80) at the same total PEs, because wider rows
      absorb more filter factors spatially. The EDP trade-off consistently favors
      the min-latency tier configs (5.455e+06 cc) over min-PEs configs.


Eyeriss Case Study 2: IntermediateRegister Size Sweep (50–500 entries)
  VGG16 13-layer full fusion, Eyeriss, GB=128KB, tile_size=1
  Non-swept regs: InReg=400, WReg=5000, OutReg=64

    python3 experiment_runner.py --sweep-intreg-pe \
    -w vgg16 -f full -v 13layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 400 --weight-reg 5000 --output-reg 64 \
    --intermediate-reg-sizes 50 100 230 500 \
    --pe-rows-grid 128 256 512 \
    --pe-cols-grid 16 32 64 128

    SUMMARY (A) — Minimum PEs Configuration:
      IntReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      50       84×128      10,752      Yes        4.159e+04    5.681e+06    2.42e+05
      100      84×64        5,376      Yes        4.048e+04    6.715e+06    2.76e+05
      230     196×16        3,136      Yes        4.114e+04    6.715e+06    2.77e+05  ← sat.

    SUMMARY (B) — Minimum Latency Configuration:
      IntReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      50      128×128      16,384      Yes        4.225e+04    5.455e+06    2.36e+05
      100     196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05
      230     196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05  ← sat.

    ANALYSIS — IntReg is a BINDING constraint for VGG16 (same as ResNet18):
      IntReg stores intermediate activations between adjacent layers: the Z_i
      values not absorbed by SACols. Footprint = 229 at pe_cols=16, drops to
      115 at cols=32, 58 at cols=64, 30 at cols=128.

      IntReg=50: Need pe_cols=128 to push footprint to 30 < 50.
        Min PEs = 84×128 = 10,752 (the search finds smaller rows since WReg=5000
        is generous, but large cols are mandatory).

      IntReg=100: pe_cols=64 gives footprint=58 < 100 → 84×64 = 5,376 PEs.
        Halving cols from 128 to 64 halves PEs when IntReg is the binding constraint.

      IntReg=230: pe_cols=16 gives footprint=229 < 230. Just barely fits!
        Min PEs = 196×16 = 3,136 (WReg=5000 is non-binding at this config).
        This is the saturation point for IntReg.

      IntReg=500: Same as 230 — excess unused. Already at 196×16.

    Min-latency behavior:
      IntReg=50: Needs 128 cols → 128×128 = 16,384 PEs. But not 512×128 because
        WReg=5000 is satisfied at 128 rows with 128 cols.
      IntReg=100: 196×64 = 12,544 PEs (same latency 5.455e+06 cc).
      IntReg≥230: Saturates at 196×64 = 12,544 PEs.

    Comparison with ResNet18 IntReg sweep:
      ResNet18 at pe_cols=16: footprint=209. Saturation at IntReg=400, 256×16=4,096 PEs.
      VGG16 at pe_cols=16: footprint=229. Saturation at IntReg=230, 196×16=3,136 PEs.
      VGG16 IntReg is slightly larger (229 vs 209, +10%) but saturates at fewer PEs
      because the row saturation point is lower (196 vs 256).


Eyeriss Case Study 3: OutRegister Size Sweep (4–64 entries)
  VGG16 13-layer full fusion, Eyeriss, GB=128KB, tile_size=1
  Non-swept regs: InReg=400, WReg=5000, IntReg=300

    python3 experiment_runner.py --sweep-outreg-pe \
    -w vgg16 -f full -v 13layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 400 --weight-reg 5000 --intermediate-reg 300 \
    --output-reg-sizes 4 8 16 32 64 \
    --pe-rows-grid 84 128 196 256 324 512 \
    --pe-cols-grid 4 8 16 32 64 128

    SUMMARY (A) — Minimum PEs Configuration:
      OutReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      4        84×128      10,752      Yes        4.159e+04    5.681e+06    2.42e+05
      8        84×64        5,376      Yes        4.048e+04    6.715e+06    2.76e+05
      16      128×32        4,096      Yes        4.026e+04    6.715e+06    2.72e+05
      32      196×16        3,136      Yes        4.114e+04    6.715e+06    2.77e+05  ← sat.
      64      196×16        3,136      Yes        4.114e+04    6.715e+06    2.77e+05

    SUMMARY (B) — Minimum Latency Configuration:
      OutReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      4       128×128      16,384      Yes        4.225e+04    5.455e+06    2.36e+05
      8       196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05
      16      196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05  ← sat.
      32      196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05
      64      196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05

    ANALYSIS — OutReg follows the same SACols Z pattern:
      OutReg stores the final layer's output: Z12=512, but SACols distributes Z12
      across column groups. Footprint = 32 at pe_cols=16, 16 at cols=32, 8 at cols=64,
      4 at cols=128.

      OutReg=4: Need pe_cols=128 (footprint=4 ≤ 4 exact fit) → 84×128 = 10,752 PEs.
      OutReg=8: pe_cols=64 (footprint=8 ≤ 8) → 84×64 = 5,376 PEs.
      OutReg=16: pe_cols=32 (footprint=16 ≤ 16) → 128×32 = 4,096 PEs.
        Note: search finds 128×32 (not 196×32) as min PEs config because WReg=5000
        is satisfied at 128 rows with 32 cols (footprint=4,791 < 5,000).
      OutReg=32: pe_cols=16 suffices (footprint=32 ≤ 32) → 196×16 = 3,136 PEs.
      OutReg=64: Same as 32 (excess unused).

      The OutReg constraint determines minimum pe_cols: each halving of OutReg
      doubles required pe_cols. The footprint is Z12/SACols_Z_groups and is
      identical to ResNet18 (Z16=512, same pattern).

    Min-latency:
      OutReg=4: 128×128 = 16,384 PEs (5.455e+06 cc).
      OutReg=8+: Saturates at 196×64 = 12,544 PEs (5.455e+06 cc).

    Comparison with ResNet18:
      Identical OutReg footprint (both have Z_last=512, same SACols Z pattern).
      ResNet18: OutReg=32 → 256×16=4,096 PEs. VGG16: OutReg=32 → 196×16=3,136 PEs.
      Less PEs for VGG16 because row saturation is at 196 (vs 256 for ResNet18).


    Eyeriss Case Study 5: PE Aspect Ratio Sweep (fixed total PEs=16384)
    VGG16 13-layer full fusion, Eyeriss, GB=128KB, tile_size=1
    Registers: InReg=400, WReg=1200, IntReg=300, OutReg=64

    python3 experiment_runner.py --sweep-pe-aspect \
    --workload vgg16 --fusion full --variant 13layer \
    --total-pes 16384 \
    --input-reg 400 --weight-reg 1200 --intermediate-reg 350 --output-reg 64 \
    --tile-size 1 --gb-size 128 2>&1 | tee results/EY_CS5_VGG16.log

        PE         Energy (μJ)  Latency (cc)    EDP
        256×64		3.686e+05	5.455e+06	    2.09e+06	
        512×32		3.641e+05	5.455e+06	    2.04e+06	
        1024×16	  	3.686e+05	5.455e+06   	2.06e+06

        Reads:
                    256x64          512x32          1024x16
        InRegister	5,635,768,320	4,248,502,272	3,843,883,008	↓ 32%
    IntermediateReg	16,878,403,584	17,918,853,120	19,074,908,160	↑ 13%
        
        The InRegister's 32% read reduction translates to a ΔWMOPs of ~1.5e+10 pJ, while IntermediateRegister's 13% read increase costs ~5e+09 pJ. 

        Why Latency is Identical (DRAM-Bandwidth-Bound)
        All three configs show:

        DRAM: 5,454,912 cc Read Drain (bottleneck)
        DRAM: 2,754,816 stall cycles (~50% of DRAM latency)
        Read bandwidth = 4 words/cycle, Max Ideal Read demand = 62.7 words/cycle → 15.7× oversubscribed
        All register-level drain/fill times are ≈ 38 k cc — negligible vs. DRAM
        With tile = 1, the entire 13-layer fused workload's weights (14.7M parameters) must flow through DRAM's 4-word read port. This fixed bottleneck makes latency completely insensitive to the PE aspect ratio.





    Eyeriss Case Study 5_2: PE Aspect Ratio Sweep (fixed total PEs=4096)
    VGG16 13-layer full fusion, Eyeriss, GB=128KB, tile_size=1
    Registers: InReg=400, WReg=4800, IntReg=300, OutReg=64

    python3 experiment_runner.py --sweep-pe-aspect \
    --workload vgg16 --fusion full --variant 13layer \
    --total-pes 4096 \
    --input-reg 400 --weight-reg 4800 --intermediate-reg 300 --output-reg 64 \
    --tile-size 1 --gb-size 128 --verbose 2>&1 | tee results/EY_CS5_2_VGG16.log

    Infeasible configs: column-heavy (1x4096 through 32x128) and
    extreme row-heavy (512x8 through 4096x1).
    Only 3 configs feasible: 64x64, 128x32, 256x16.

    Column-heavy fails: VGG16's large C dimensions (up to C12=512) and
    R=S=3 require SARows products ≥ C×S×Z_share. With ≤32 rows, the
    mapping constraints become ill-posed.

    Row-heavy fails: at ≥512 rows, WReg footprint exceeds 5000 entries.
    With tile_size=1 (Q=1), SACols carry only Z factors. At pe_cols=8,
    the residual Z iterations at WRegister level exceed capacity.

    Results:
      PE         Energy (μJ)   Latency (cc)    EDP
      64x64      9.960e+05     6.715e+06       6.79e+06
      128x32     9.782e+05     6.715e+06       6.62e+06
      256x16     9.743e+05     6.715e+06       6.57e+06    ← EDP minimum

    Energy:   decreases 64x64→256x16 (–2.2 %).
              Driven by InReg reads: 5.64B→1.82B (–67.7 %).
              IntReg R+W rises slightly: 30.8B→33.8B (+9.6 %).
              Net effect: fewer InReg accesses win.

    Latency:  identical at 6.715e+06 cc. GlobalBuffer is the bottleneck
              and its access pattern is invariant across these 3 configs.

    EDP:      minimum at 256x16 (6.57e+06). Monotonically decreasing
              toward more rows. The trend suggests even higher pe_rows
              would improve EDP, but WReg capacity limits prevent it.

    Per-level access counts (reads):
      PE         GB Reads     InReg Reads     WReg Reads      IntReg R+W
      64x64      40,793,088   5,635,768,320   15,346,630,656  30,808,866,816
      128x32     40,793,088   3,092,447,232   15,346,630,656  31,791,513,600
      256x16     40,793,088   1,820,786,688   15,346,630,656  33,756,807,168

    GlobalBuffer reads: constant (40.8M).
    WReg reads: constant (15.3B).
    InReg reads: drop 3.1× (5.64B→1.82B) — more rows better map C×S
    factors, reducing residual InReg iterations.
    IntReg R+W: rise +9.6 % — fewer cols means fewer Z factors on SACols,
    pushing more Z iterations to IntReg.

    Key insight: The narrow feasibility window (64–256 rows) reflects
    VGG16's dual constraint pressure. Large C dimensions (up to 512)
    demand sufficient rows for SARows mapping, while the enormous weight
    count (14.7M params) inflates WReg footprints at extreme row counts.
    256x16 hits the sweet spot: enough rows for C mapping, enough cols
    to keep WReg footprint within 5000 entries.      




















  COMBINED ANALYSIS — All three Eyeriss register types for VGG16:
    pe_cols is the universal PE-reduction lever for all register types,
    identical to the ResNet18 pattern. Doubling pe_cols halves the per-PE
    footprint for WReg, IntReg, and OutReg via SACols Z distribution.

    To reach the minimum PE count (196×16 = 3,136 PEs), ALL three must be satisfied:
      WReg ≥ 5000 (footprint 4,791 at 196×16)
      IntReg ≥ 230 (footprint 229 at 196×16)
      OutReg ≥ 32 (footprint 32 at 196×16)
      InReg ≥ 180 (footprint 180 at 196×16)

    The binding hierarchy is: WReg >> IntReg > InReg > OutReg (same as ResNet18).
    WReg alone determines the PE count when undersized because its footprint
    is 21× larger than IntReg's (4,791 vs 229).

    Comparison with ResNet18:
      ResNet18: WReg ≥ 4,000, IntReg ≥ 210, OutReg ≥ 32. Min PEs = 4,096 (256×16).
      VGG16:   WReg ≥ 5,000, IntReg ≥ 230, OutReg ≥ 32. Min PEs = 3,136 (196×16).
      VGG16 needs 25% more WReg (5000 vs 4000) and 10% more IntReg (230 vs 210),
      but achieves 23% fewer min PEs (3,136 vs 4,096). This is because VGG16's
      simpler layer structure (all 3×3 convs, no 7×7 or 1×1 projections) allows
      SARows to saturate at fewer rows (196 vs 256).

    Comparison with FSRCNN and MC-CNN (activation-dominant):
      FSRCNN: WReg ≥ 300, IntReg/OutReg non-binding. Min PEs = 336 (84×4).
      MC-CNN:  WReg ≥ 600, IntReg/OutReg non-binding. Min PEs = 192 (12×16).
      VGG16:   WReg ≥ 5,000, IntReg ≥ 230, OutReg ≥ 32. Min PEs = 3,136 (196×16).
      VGG16 needs 17× more WReg than FSRCNN, and IntReg/OutReg become genuinely
      binding constraints — unlike the activation-dominant workloads where only
      WReg matters. The 9× PE increase (336 → 3,136) reflects the weight-dominant
      character: larger channels (Z up to 512), more layers, and the critical
      difference that ALL register types scale with channel width.

    Min-latency comparison:
      ResNet18: 256×32 = 8,192 PEs → 3.042e+06 cc.
      VGG16:   196×64 = 12,544 PEs → 5.455e+06 cc.
      VGG16 needs 53% more PEs for min-latency (12,544 vs 8,192) and achieves
      79% worse latency (5.455M vs 3.042M cc). The latency difference comes from
      VGG16's much larger total weight volume (14.7M vs 11M params) requiring
      more GlobalBuffer streaming cycles.

      




















================================================================================
EYERISS CASE STUDIES — MC-CNN 4-layer full fusion
================================================================================

  Architecture: Eyeriss-like with unified GlobalBuffer (128KB), no FMEM/WMEM split.
  PE-level registers: WReg (weights), IntReg (intermediates), OutReg (outputs), InReg (inputs).
  Spatial mapping: SARows map C_i × S_i × Z_i factors; SACols map Q/X (tile) and Z.
  tile_size=69 (max divisor of Q=1242 that fits in 128 cols).

  MC-CNN 4-layer shape (all strides=1):
    Z0-Z3=32 (uniform), C0=1, C1-C3=32, R_i=S_i=3 for all layers
    All X_i = Q = 1242 (no strided reduction)
    Total weights: 27,936 params (~55KB)

  PE grid searched: rows=[4,8,12,16,32,64,80,128,256,320,512], cols=[4,8,12,16,32,64,80,128,256,320,512]
    121 combos per register size (filtered by max_total_pes=17000).

  Register footprints at tile_size=69:
    PE Config   InReg   WReg    IntReg  OutReg
    84×4        24      582     66      32
    84×8        24      291     33      16
    84×16       24      147     17       8
    84×32       24      582     66      32      ← SACols Z wraps (32=4×8)
    84×64       24      291     33      16      ← same as 84×8 pattern
    32×16       42      291     17       8
    16×16       78      582     18       8
    8×16        150    1164     20       8

  Key insights:
    - WReg dominates (147–1164 entries) — stores ALL 4 layers' weight residuals.
      But 4 layers with Z=32, C≤32, R=S=3 means much less than ResNet18 (WReg up to 14K).
    - InReg depends on pe_rows (24 at 84 rows → 150 at 8 rows): more rows absorb
      more C×S factors, leaving fewer residual C iterations at register level.
    - IntReg (17–66) and OutReg (8–32) are very small — they depend on pe_cols
      (SACols distributes Z across column groups).
    - Minimum feasible WReg = 147 (at 84×16). For FSRCNN it was 258 at 84×4.
      MC-CNN is simpler: only 4 layers (vs 8), all Z=32 (vs Z up to 56).


Eyeriss Case Study 1: WRegister Size Sweep (100–600 entries)
  MC-CNN 4-layer full fusion, Eyeriss, GB=128KB, tile_size=69
  InReg=34, IntReg=32, OutReg=64 (matching fusion canonical)

    python3 experiment_runner.py --sweep-wreg-pe \
    -w mccnn -f full -v 4layer \
    --tile-size 69 --gb-size 128 \
    --input-reg 34 --intermediate-reg 32 --output-reg 64 \
    --weight-reg-sizes 100 200 300 600

    SUMMARY (A) — Minimum PEs Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      100     128×12       1,536       No         6.649e+04    1.223e+07    8.28e+05
      200      64×12         768       No         7.730e+04    2.344e+07    1.84e+06
      300      32×12         384       No         8.983e+04    4.585e+07    4.18e+06
      600      12×12         144       No         1.315e+05    9.068e+07    1.20e+07

    SUMMARY (B) — Minimum Latency Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      100     128×12       1,536       No         6.649e+04    1.223e+07    8.28e+05
      200     128×12       1,536       No         8.075e+04    1.223e+07    1.00e+06
      300     128×12       1,536       No         9.501e+04    1.223e+07    1.18e+06
      600     128×12       1,536       No         1.378e+05    1.223e+07    1.70e+06

    SUMMARY (C) — ALL Feasible Configurations per WReg Size:
      WReg=100: 14 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
        128×   12        1,536         No      6.649e+04      1.223e+07       8.28e+05
        128×   16        2,048         No      6.649e+04      1.223e+07       8.28e+05
        256×    8        2,048         No      6.649e+04      1.223e+07       8.28e+05
        512×    4        2,048         No      6.649e+04      1.223e+07       8.28e+05
        320×    8        2,560         No      6.649e+04      1.223e+07       8.28e+05
        256×   12        3,072         No      6.896e+04      1.223e+07       8.74e+05
        320×   12        3,840         No      6.896e+04      1.223e+07       8.74e+05
        256×   16        4,096         No      6.896e+04      1.223e+07       8.74e+05
        512×    8        4,096         No      6.896e+04      1.223e+07       8.74e+05
        320×   16        5,120         No      6.896e+04      1.223e+07       8.74e+05
        512×   12        6,144         No      7.391e+04      1.223e+07       9.64e+05
        512×   16        8,192         No      7.391e+04      1.223e+07       9.64e+05
        256×   64       16,384         No      6.649e+04      1.223e+07       8.28e+05
        512×   32       16,384         No      6.649e+04      1.223e+07       8.28e+05
      WReg=200: 24 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         64×   12          768         No      7.730e+04      2.344e+07       1.84e+06
         80×   12          960         No      7.730e+04      2.344e+07       1.84e+06
         64×   16        1,024         No      7.730e+04      2.344e+07       1.84e+06
        128×    8        1,024         No      7.951e+04      2.344e+07       1.88e+06
        256×    4        1,024         No      7.951e+04      2.344e+07       1.88e+06
         80×   16        1,280         No      7.730e+04      2.344e+07       1.84e+06
        320×    4        1,280         No      7.951e+04      2.344e+07       1.88e+06
        128×   12        1,536         No      8.075e+04      1.223e+07       1.00e+06
        128×   16        2,048         No      8.075e+04      1.223e+07       1.00e+06
        256×    8        2,048         No      8.075e+04      1.223e+07       1.00e+06
        512×    4        2,048         No      8.075e+04      1.223e+07       1.00e+06
        320×    8        2,560         No      8.075e+04      1.223e+07       1.00e+06
        256×   12        3,072         No      8.322e+04      1.223e+07       1.05e+06
        320×   12        3,840         No      8.322e+04      1.223e+07       1.05e+06
        256×   16        4,096         No      8.322e+04      1.223e+07       1.05e+06
        512×    8        4,096         No      8.322e+04      1.223e+07       1.05e+06
        320×   16        5,120         No      8.322e+04      1.223e+07       1.05e+06
        512×   12        6,144         No      8.817e+04      1.223e+07       1.14e+06
        128×   64        8,192         No      7.951e+04      1.223e+07       9.80e+05
        256×   32        8,192         No      7.951e+04      1.223e+07       9.80e+05
        512×   16        8,192         No      8.817e+04      1.223e+07       1.14e+06
        320×   32       10,240         No      7.951e+04      1.223e+07       9.80e+05
        256×   64       16,384         No      8.075e+04      1.223e+07       1.00e+06
        512×   32       16,384         No      8.075e+04      1.223e+07       1.00e+06
      WReg=300: 37 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         32×   12          384         No      8.983e+04      4.585e+07       4.18e+06
         32×   16          512         No      8.983e+04      4.585e+07       4.18e+06
         64×    8          512         No      9.032e+04      4.585e+07       4.17e+06
        128×    4          512         No      9.316e+04      4.585e+07       4.29e+06
         80×    8          640         No      9.032e+04      4.585e+07       4.17e+06
         64×   12          768         No      9.156e+04      2.344e+07       2.17e+06
         80×   12          960         No      9.156e+04      2.344e+07       2.17e+06
         64×   16        1,024         No      9.156e+04      2.344e+07       2.17e+06
        128×    8        1,024         No      9.377e+04      2.344e+07       2.21e+06
        256×    4        1,024         No      9.377e+04      2.344e+07       2.21e+06
         80×   16        1,280         No      9.156e+04      2.344e+07       2.17e+06
        320×    4        1,280         No      9.377e+04      2.344e+07       2.21e+06
        128×   12        1,536         No      9.501e+04      1.223e+07       1.18e+06
        128×   16        2,048         No      9.501e+04      1.223e+07       1.18e+06
        256×    8        2,048         No      9.501e+04      1.223e+07       1.18e+06
        512×    4        2,048         No      9.501e+04      1.223e+07       1.18e+06
        320×    8        2,560         No      9.501e+04      1.223e+07       1.18e+06
        256×   12        3,072         No      9.748e+04      1.223e+07       1.22e+06
        320×   12        3,840         No      9.748e+04      1.223e+07       1.22e+06
         64×   64        4,096         No      9.032e+04      1.223e+07       1.11e+06
        128×   32        4,096         No      9.316e+04      1.223e+07       1.14e+06
        256×   16        4,096         No      9.748e+04      1.223e+07       1.22e+06
        512×    8        4,096         No      9.748e+04      1.223e+07       1.22e+06
         80×   64        5,120         No      9.032e+04      1.223e+07       1.11e+06
        320×   16        5,120         No      9.748e+04      1.223e+07       1.22e+06
        512×   12        6,144         No      1.024e+05      1.223e+07       1.31e+06
        128×   64        8,192         No      9.377e+04      1.223e+07       1.15e+06
        256×   32        8,192         No      9.377e+04      1.223e+07       1.15e+06
        512×   16        8,192         No      1.024e+05      1.223e+07       1.31e+06
         32×  320       10,240        Yes      8.983e+04      1.223e+07       1.11e+06
        128×   80       10,240         No      9.316e+04      1.223e+07       1.14e+06
        320×   32       10,240         No      9.377e+04      1.223e+07       1.15e+06
         32×  512       16,384        Yes      8.983e+04      1.223e+07       1.11e+06
         64×  256       16,384        Yes      9.032e+04      1.223e+07       1.11e+06
        128×  128       16,384         No      9.316e+04      1.223e+07       1.14e+06
        256×   64       16,384         No      9.501e+04      1.223e+07       1.18e+06
        512×   32       16,384         No      9.501e+04      1.223e+07       1.18e+06
      WReg=600: 56 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         12×   12          144         No      1.315e+05      9.068e+07       1.20e+07
         12×   16          192         No      1.315e+05      9.068e+07       1.20e+07
         16×   12          192         No      1.315e+05      9.068e+07       1.20e+07
         16×   16          256         No      1.315e+05      9.068e+07       1.20e+07
         32×    8          256         No      1.312e+05      9.068e+07       1.20e+07
         64×    4          256         No      1.323e+05      9.068e+07       1.20e+07
         80×    4          320         No      1.323e+05      9.068e+07       1.20e+07
         32×   12          384         No      1.326e+05      4.585e+07       6.14e+06
         32×   16          512         No      1.326e+05      4.585e+07       6.14e+06
         64×    8          512         No      1.331e+05      4.585e+07       6.13e+06
        128×    4          512         No      1.359e+05      4.585e+07       6.25e+06
         80×    8          640         No      1.331e+05      4.585e+07       6.13e+06
         64×   12          768         No      1.343e+05      2.344e+07       3.18e+06
         80×   12          960         No      1.343e+05      2.344e+07       3.18e+06
         64×   16        1,024         No      1.343e+05      2.344e+07       3.18e+06
        128×    8        1,024         No      1.366e+05      2.344e+07       3.21e+06
        256×    4        1,024         No      1.366e+05      2.344e+07       3.21e+06
         80×   16        1,280         No      1.343e+05      2.344e+07       3.18e+06
        320×    4        1,280         No      1.366e+05      2.344e+07       3.21e+06
        128×   12        1,536         No      1.378e+05      1.223e+07       1.70e+06
        128×   16        2,048         No      1.378e+05      1.223e+07       1.70e+06
        256×    8        2,048         No      1.378e+05      1.223e+07       1.70e+06
        512×    4        2,048         No      1.378e+05      1.223e+07       1.70e+06
         32×   64        2,048         No      1.312e+05      1.272e+07       1.68e+06
         64×   32        2,048         No      1.323e+05      1.272e+07       1.69e+06
        320×    8        2,560         No      1.378e+05      1.223e+07       1.70e+06
         80×   32        2,560         No      1.323e+05      1.272e+07       1.69e+06
        256×   12        3,072         No      1.403e+05      1.223e+07       1.75e+06
         12×  320        3,840        Yes      1.315e+05      1.223e+07       1.62e+06
        320×   12        3,840         No      1.403e+05      1.223e+07       1.75e+06
         64×   64        4,096         No      1.331e+05      1.223e+07       1.64e+06
        128×   32        4,096         No      1.359e+05      1.223e+07       1.67e+06
        256×   16        4,096         No      1.403e+05      1.223e+07       1.75e+06
        512×    8        4,096         No      1.403e+05      1.223e+07       1.75e+06
         16×  320        5,120        Yes      1.315e+05      1.223e+07       1.62e+06
         64×   80        5,120         No      1.323e+05      1.223e+07       1.62e+06
         80×   64        5,120         No      1.331e+05      1.223e+07       1.64e+06
        320×   16        5,120         No      1.403e+05      1.223e+07       1.75e+06
         12×  512        6,144        Yes      1.315e+05      1.223e+07       1.62e+06
        512×   12        6,144         No      1.452e+05      1.223e+07       1.84e+06
         80×   80        6,400         No      1.323e+05      1.223e+07       1.62e+06
         16×  512        8,192        Yes      1.315e+05      1.223e+07       1.62e+06
         32×  256        8,192        Yes      1.312e+05      1.223e+07       1.61e+06
         64×  128        8,192         No      1.323e+05      1.223e+07       1.62e+06
        128×   64        8,192         No      1.366e+05      1.223e+07       1.68e+06
        256×   32        8,192         No      1.366e+05      1.223e+07       1.68e+06
        512×   16        8,192         No      1.452e+05      1.223e+07       1.84e+06
         32×  320       10,240        Yes      1.326e+05      1.223e+07       1.64e+06
         80×  128       10,240         No      1.323e+05      1.223e+07       1.62e+06
        128×   80       10,240         No      1.359e+05      1.223e+07       1.67e+06
        320×   32       10,240         No      1.366e+05      1.223e+07       1.68e+06
         32×  512       16,384        Yes      1.326e+05      1.223e+07       1.64e+06
         64×  256       16,384        Yes      1.331e+05      1.223e+07       1.64e+06
        128×  128       16,384         No      1.359e+05      1.223e+07       1.67e+06
        256×   64       16,384         No      1.378e+05      1.223e+07       1.70e+06
        512×   32       16,384         No      1.378e+05      1.223e+07       1.70e+06

    ANALYSIS — WReg is the binding constraint (uniform grid [4..512]):
      WReg footprint = Σ(4 layers) of residual C×R factors not absorbed by SARows.
      At pe_rows=128: footprint ≈ 75 entries at pe_cols=12. ✓ for all WReg ≥ 100.
      At pe_rows=64:  footprint ≈ 147 at pe_cols=12 → fits WReg ≥ 200.
      At pe_rows=32:  footprint ≈ 291 at pe_cols=12 → fits WReg ≥ 300.
      At pe_rows=12:  footprint ≈ 582 at pe_cols=12 → fits WReg ≥ 600.

      WReg=100: Min PEs = 128×12 = 1,536. Footprint=75 < 100 ✓
      WReg=200: Min PEs = 64×12  =   768. Footprint=147 < 200 ✓
      WReg=300: Min PEs = 32×12  =   384. Footprint=291 < 300 ✓
      WReg=600: Min PEs = 12×12  =   144. Footprint=582 < 600 ✓

    Latency behavior:
      Min-latency = 1.223e+07 cc saturates at 128×12 = 1,536 PEs for all WReg sizes.
      Smaller PE arrays (lower rows) achieve same latency minimum at higher total PEs.
      All Summary B configs converge to 128×12 because this provides enough
      rows to absorb weight factors and enough cols for spatial coverage.

    Energy: Lower PEs → slightly lower energy (66,490→131,500 μJ as WReg increases
      from 100→600), reflecting the WReg size cost. Fewer PEs at WReg=600 but
      higher per-PE register energy due to larger WReg.


Eyeriss Case Study 2: IntermediateRegister Size Sweep (15–100 entries)
  MC-CNN 4-layer full fusion, Eyeriss, GB=128KB, tile_size=69
  Non-swept regs: InReg=34, WReg=600, OutReg=64

    python3 experiment_runner.py --sweep-intreg-pe \
    -w mccnn -f full -v 4layer \
    --tile-size 69 --gb-size 128 \
    --input-reg 34 --weight-reg 600 --output-reg 64 \
    --intermediate-reg-sizes 15 30 50 100 \
    --pe-rows-grid 4 8 12 14 16 28 32 56 84 \
    --pe-cols-grid 4 8 16 32 64 69 128 138

    SUMMARY (A) — Minimum PEs Configuration:
      IntReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      15      N/A          N/A         N/A        N/A          N/A             N/A
      30      12×16        192         No         3.488e+04    9.068e+07    3.18e+06
      50      12×16        192         No         3.488e+04    9.068e+07    3.18e+06
      100     12×16        192         No         3.488e+04    9.068e+07    3.18e+06

    SUMMARY (B) — Minimum Latency Configuration:
      IntReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      15      N/A          N/A         N/A        N/A          N/A             N/A
      30      56×16        896         No         3.635e+04    2.344e+07    8.56e+05
      50      56×64        3584        No         3.617e+04    1.223e+07    4.43e+05
      100     56×64        3584        No         3.617e+04    1.223e+07    4.43e+05

    ANALYSIS — IntReg is a WEAK constraint for MC-CNN:
      IntReg footprint is very small: 17 at pe_cols=16, 33 at pe_cols=8, 66 at pe_cols=4.
      This is because MC-CNN has only 3 intermediate activations (between 4 layers),
      and Z=32 is distributed across SACols Z groups when available.

      IntReg=15: INFEASIBLE — minimum footprint is 17 (at pe_cols=16).

      IntReg=30: Every config with pe_cols=16 is feasible (footprint=17 < 30).
        Min PEs = 12×16 = 192 (same as WReg=600 limit).
        The IntReg constraint doesn't add any restriction beyond WReg.

      IntReg=50/100: IDENTICAL to IntReg=30 — fully saturated.
        The IntReg constraint doesn't add any restriction beyond WReg.

      Min-latency: IntReg=30 gives 56×16=896 (not 56×64) because at pe_cols=64
        the footprint=33 > 30 (SACols wraps and changes the mapping).
        IntReg=50: 56×64 feasible (footprint=33 < 50). Latency 1.9× better.

    Conclusion: IntReg saturates immediately at ≥ 30 entries for min PEs.
      For MC-CNN (only 4 layers, Z=32), the intermediate activation footprint
      is tiny. This is analogous to FSRCNN where IntReg was also non-binding.


Eyeriss Case Study 3: OutRegister Size Sweep (8–64 entries)
  MC-CNN 4-layer full fusion, Eyeriss, GB=128KB, tile_size=69
  Non-swept regs: InReg=34, WReg=600, IntReg=32

    python3 experiment_runner.py --sweep-outreg-pe \
    -w mccnn -f full -v 4layer \
    --tile-size 69 --gb-size 128 \
    --input-reg 34 --weight-reg 600 --intermediate-reg 32 \
    --output-reg-sizes 8 16 32 64 \
    --pe-rows-grid 4 8 12 14 16 28 32 56 84 \
    --pe-cols-grid 4 8 16 32 64 69 128 138

    SUMMARY (A) — Minimum PEs Configuration:
      OutReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      8       12×16        192         No         3.488e+04    9.068e+07    3.18e+06
      16      12×16        192         No         3.488e+04    9.068e+07    3.18e+06
      32      12×16        192         No         3.488e+04    9.068e+07    3.18e+06
      64      12×16        192         No         3.488e+04    9.068e+07    3.18e+06

    SUMMARY (B) — Minimum Latency Configuration:
      OutReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      8       56×16        896         No         3.635e+04    2.344e+07    8.56e+05
      16      56×64        3584        No         3.617e+04    1.223e+07    4.43e+05
      32      56×64        3584        No         3.617e+04    1.223e+07    4.43e+05
      64      56×64        3584        No         3.617e+04    1.223e+07    4.43e+05

    ANALYSIS — OutReg is NEVER binding for MC-CNN:
      OutReg stores the final layer's output: Z3=32. Footprint = Z3/SACols_Z3.
        pe_cols=16: footprint = 8 (SACols Z3=4) → even OutReg=8 works.
        pe_cols=8: footprint = 16.
        pe_cols=4: footprint = 32.

      Min PEs: ALL register sizes give 12×16=192 PEs — fully constant.
        OutReg=8 at pe_cols=16: footprint=8 ≤ 8 
        The constraint never eliminates any config that WReg hasn't already eliminated.

      Min Latency: OutReg=8 gives 56×16=896 (not 56×64 because at pe_cols=64
        the footprint=16 > 8). OutReg=16: 56×64 feasible (footprint=16 ≤ 16). ✓

    Conclusion: OutReg is non-binding for MC-CNN min PEs determination.
      Even OutReg=8 (minimal) doesn't restrict the PE count at all.
      This matches FSRCNN behavior: small Z → small OutReg footprint.


  COMBINED ANALYSIS — All three Eyeriss register types for MC-CNN:
    The binding hierarchy is: WReg >> IntReg > OutReg (same as FSRCNN).

    To reach the minimum PE count (12×16 = 192 PEs), the requirements are:
      WReg ≥ 600 (footprint ~582 at 12×16)
      IntReg ≥ 17 (footprint 17 at pe_cols=16)
      OutReg ≥ 8 (footprint 8 at pe_cols=16)
      InReg ≥ ~78 (footprint ~78 at 16 rows, higher at 12 rows)

    WReg is the SOLE binding constraint — it alone determines the minimum PE count.
    IntReg and OutReg are effectively non-binding: their footprints are so small
    (17 and 8 entries) that any reasonable register sizing satisfies them.

    Comparison with FSRCNN:
      FSRCNN: WReg ≥ 258, min PEs = 336 (84×4). IntReg/OutReg non-binding.
      MC-CNN: WReg ≥ 600, min PEs = 192 (12×16). IntReg/OutReg non-binding.
      MC-CNN needs FEWER PEs (192 vs 336) despite higher WReg requirement (600 vs 258)
      because it has only 4 layers (vs 8) and all Z=32 (vs Z up to 56).
      The smaller pe_rows (12 vs 84) reflects the simpler layer structure:
      SARows only needs to distribute C×S×Z = 32×3×32 = 3072 max, and 12 rows
      can absorb enough of these factors to bring WReg footprint under 600.

    Comparison with ResNet18:
      ResNet18: WReg ≥ 4000, IntReg ≥ 210, OutReg ≥ 32. Min PEs = 4,096 (256×16).
      MC-CNN: WReg ≥ 600, IntReg ≥ 17, OutReg ≥ 8. Min PEs = 192 (12×16).
      The 21× PE reduction reflects 4 layers (vs 17) and Z=32 (vs Z up to 512).
      ResNet18's IntReg and OutReg were genuinely binding; MC-CNN's are not.

Eyeriss Case Study 5: PE Aspect Ratio Sweep (fixed total PEs=2048)
  MC-CNN 4-layer full fusion, Eyeriss, GB=128KB, tile_size=69
  Registers: InReg=34, WReg=600, IntReg=32, OutReg=64

    python3 experiment_runner.py --sweep-pe-aspect \
    --workload mccnn --fusion full --variant 4layer \
    --total-pes 2048 \
    --input-reg 34 --weight-reg 600 --intermediate-reg 32 --output-reg 64 \
    --tile-size 69 --gb-size 128 --verbose 2>&1 | tee results/EY_CS5_MCCNN.log

    Infeasible configs: 1x2048, 2x1024, 4x512, 8x256, 16x128
    (column-heavy configs fail: MC-CNN's Z=32 cannot fill SARows
    when pe_rows ≤ 16; constraint factor product C×S×Z < pe_rows)

    Results:
      PE         Energy (μJ)   Latency (cc)    EDP
      32x64      1.312e+05     1.272e+07       1.68e+06    ← EDP minimum
      64x32      1.323e+05     1.272e+07       1.69e+06
      128x16     1.378e+05     1.223e+07       1.70e+06
      256x8      1.378e+05     1.223e+07       1.70e+06
      512x4      1.378e+05     1.223e+07       1.70e+06
      1024x2     1.452e+05     1.261e+07       1.89e+06
      2048x1     1.452e+05     1.261e+07       1.89e+06

    Energy:   increases 32x64→2048x1 (+10.7 %).
              Two plateaus: 128x16=256x8=512x4 (1.378e+05) and
              1024x2=2048x1 (1.452e+05). Jump at 1024x2 driven by
              InReg reads doubling (8.74B vs 5.51B).
              Minimum at 32x64 (1.312e+05 μJ).

    Latency:  three distinct values:
              32x64, 64x32: 1.272e+07 cc
              128x16, 256x8, 512x4: 1.223e+07 cc (–3.9 %, best)
              1024x2, 2048x1: 1.261e+07 cc (+3.1 %)
              Mid-range configs have lowest latency.

    EDP:      minimum at 32x64 (1.68e+06). Very flat plateau from
              32x64 through 512x4 (1.68e+06→1.70e+06, +1.2 %).
              Sharp step at 1024x2 (1.89e+06, +12.5 %).

    Per-level access counts (reads):
      PE         GB Reads      InReg Reads     WReg Reads    IntReg R+W
      32x64      195,669,648   4,909,019,904   13,045,888,512  19,008,442,368
      64x32      195,669,648   4,640,032,512   13,045,888,512  20,443,041,792
      128x16     195,669,648   5,514,241,536   13,045,888,512  23,312,240,640
      256x8      195,669,648   5,514,241,536   13,045,888,512  23,312,240,640
      512x4      195,669,648   5,514,241,536   13,045,888,512  23,312,240,640
      1024x2     195,669,648   8,742,090,240   13,045,888,512  23,312,240,640
      2048x1     195,669,648   8,742,090,240   13,045,888,512  23,312,240,640

    GlobalBuffer reads: constant (196M) — tile_size=69 is invariant.
    WReg reads: constant (13.0B) — weight accesses independent of aspect.
    InReg reads: vary 4.64B→8.74B. Jump at 1024x2 (rows exceed all dim
    products, so excess rows add redundant InReg iterations).
    IntReg R+W: plateau at 23.3B for ≥128 rows; 32x64 and 64x32 lower
    (19.0B, 20.4B) because more cols distribute Z across SACols.

    Key insight: MC-CNN's uniform Z=32 across all 4 layers means 32 rows
    suffice to fully map all Z dimensions. Beyond 32 rows, extra rows are
    wasted on SARows and cause rising InReg and IntReg overhead. The
    EDP-optimal 32x64 exactly matches 32 rows = Z=32.










      

================================================================================
EYERISS CASE STUDIES — FSRCNN 8-layer full fusion
================================================================================

  Architecture: Eyeriss-like with unified GlobalBuffer (128KB), no FMEM/WMEM split.
  PE-level registers: WReg (weights), IntReg (intermediates), OutReg (outputs), InReg (inputs).
  Spatial mapping: SARows map C_i × S_i × Z_i factors; SACols map Q/X (tile) and Z.
  Constraint: pe_rows must satisfy ALL layers' SARows products simultaneously.
  SACols Z: Activates when pe_cols >= 2 × tile_size, splitting Z across column groups.


Eyeriss Case Study 1: WRegister Size Sweep (200, 300, 400 bytes) find: A) Minimum PEs configuration that fits WReg constraint, B) Minimum latency configuration
  FSRCNN 8-layer full fusion, Eyeriss, GB=128KB, InReg=34, IntReg=32, OutReg=64, tile_size=120
  PE grid: rows=[4,8,12,14,16,28,32,56,84], cols=[4,8,16,32,56,64,80,160,240,320]
    
    python3 experiment_runner.py --sweep-wreg-pe \
    --workload fsrcnn --fusion full --variant 8layer \
    --weight-reg-sizes 200 300 400 \
    --input-reg 34 --intermediate-reg 32 --output-reg 64 --tile-size 120

    SUMMARY (A) — Minimum PEs Configuration Feasible:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      200     128×4            512      No         1.203e+05    2.760e+07    3.33e+06
      300     128×4            512      No         1.310e+05    2.760e+07    3.62e+06
      400      80×4            320      No         1.412e+05    4.769e+07    6.75e+06

    SUMMARY (B) — Minimum Latency Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      200     128×12        1,536      No         1.203e+05    1.889e+07    2.28e+06
      300     128×12        1,536      No         1.310e+05    1.889e+07    2.48e+06
      400     128×12        1,536      No         1.416e+05    1.889e+07    2.68e+06

    SUMMARY (C) — ALL Feasible Configurations per WReg Size:
      WReg=200: 26 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
        128×    4          512         No      1.203e+05      2.760e+07       3.33e+06
        128×    8        1,024         No      1.203e+05      1.915e+07       2.31e+06
        256×    4        1,024         No      1.212e+05      1.944e+07       2.37e+06
        320×    4        1,280         No      1.212e+05      1.944e+07       2.37e+06
        128×   12        1,536         No      1.203e+05      1.889e+07       2.28e+06
        128×   16        2,048         No      1.203e+05      1.889e+07       2.28e+06
        256×    8        2,048         No      1.212e+05      1.889e+07       2.30e+06
        512×    4        2,048         No      1.229e+05      1.889e+07       2.34e+06
        320×    8        2,560         No      1.212e+05      1.889e+07       2.30e+06
        256×   12        3,072         No      1.212e+05      1.889e+07       2.30e+06
        320×   12        3,840         No      1.212e+05      1.889e+07       2.30e+06
        128×   32        4,096         No      1.203e+05      1.889e+07       2.28e+06
        256×   16        4,096         No      1.212e+05      1.889e+07       2.30e+06
        512×    8        4,096         No      1.229e+05      1.889e+07       2.34e+06
        320×   16        5,120         No      1.212e+05      1.889e+07       2.30e+06
        512×   12        6,144         No      1.229e+05      1.889e+07       2.34e+06
        128×   64        8,192         No      1.203e+05      1.889e+07       2.28e+06
        256×   32        8,192         No      1.212e+05      1.889e+07       2.30e+06
        512×   16        8,192         No      1.229e+05      1.889e+07       2.34e+06
        128×   80       10,240         No      1.203e+05      1.889e+07       2.28e+06
        320×   32       10,240         No      1.212e+05      1.889e+07       2.30e+06
         32×  512       16,384        Yes      1.140e+05      1.889e+07       2.17e+06
         64×  256       16,384        Yes      1.203e+05      1.889e+07       2.28e+06
        128×  128       16,384         No      1.203e+05      1.889e+07       2.28e+06
        256×   64       16,384         No      1.212e+05      1.889e+07       2.30e+06
        512×   32       16,384         No      1.229e+05      1.889e+07       2.34e+06
      WReg=300: 27 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
        128×    4          512         No      1.310e+05      2.760e+07       3.62e+06
        128×    8        1,024         No      1.310e+05      1.915e+07       2.51e+06
        256×    4        1,024         No      1.318e+05      1.944e+07       2.57e+06
        320×    4        1,280         No      1.318e+05      1.944e+07       2.57e+06
        128×   12        1,536         No      1.310e+05      1.889e+07       2.48e+06
        128×   16        2,048         No      1.310e+05      1.889e+07       2.48e+06
        256×    8        2,048         No      1.318e+05      1.889e+07       2.50e+06
        512×    4        2,048         No      1.335e+05      1.889e+07       2.54e+06
        320×    8        2,560         No      1.318e+05      1.889e+07       2.50e+06
        256×   12        3,072         No      1.318e+05      1.889e+07       2.50e+06
        320×   12        3,840         No      1.318e+05      1.889e+07       2.50e+06
        128×   32        4,096         No      1.310e+05      1.889e+07       2.48e+06
        256×   16        4,096         No      1.318e+05      1.889e+07       2.50e+06
        512×    8        4,096         No      1.335e+05      1.889e+07       2.54e+06
        320×   16        5,120         No      1.318e+05      1.889e+07       2.50e+06
        512×   12        6,144         No      1.335e+05      1.889e+07       2.54e+06
        128×   64        8,192         No      1.310e+05      1.889e+07       2.48e+06
        256×   32        8,192         No      1.318e+05      1.889e+07       2.50e+06
        512×   16        8,192         No      1.335e+05      1.889e+07       2.54e+06
         32×  320       10,240        Yes      1.243e+05      1.889e+07       2.36e+06
        128×   80       10,240         No      1.310e+05      1.889e+07       2.48e+06
        320×   32       10,240         No      1.318e+05      1.889e+07       2.50e+06
         32×  512       16,384        Yes      1.247e+05      1.889e+07       2.37e+06
         64×  256       16,384        Yes      1.310e+05      1.889e+07       2.48e+06
        128×  128       16,384         No      1.310e+05      1.889e+07       2.48e+06
        256×   64       16,384         No      1.318e+05      1.889e+07       2.50e+06
        512×   32       16,384         No      1.335e+05      1.889e+07       2.54e+06
      WReg=400: 36 feasible config(s)
        PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
         80×    4          320         No      1.412e+05      4.769e+07       6.75e+06
        128×    4          512         No      1.416e+05      2.760e+07       3.92e+06
         80×    8          640         No      1.412e+05      2.760e+07       3.91e+06
         80×   12          960         No      1.412e+05      2.421e+07       3.42e+06
        128×    8        1,024         No      1.416e+05      1.915e+07       2.72e+06
        256×    4        1,024         No      1.425e+05      1.944e+07       2.78e+06
         80×   16        1,280         No      1.412e+05      1.915e+07       2.71e+06
        320×    4        1,280         No      1.425e+05      1.944e+07       2.78e+06
        128×   12        1,536         No      1.416e+05      1.889e+07       2.68e+06
        128×   16        2,048         No      1.416e+05      1.889e+07       2.68e+06
        256×    8        2,048         No      1.425e+05      1.889e+07       2.70e+06
        512×    4        2,048         No      1.442e+05      1.889e+07       2.75e+06
         80×   32        2,560         No      1.412e+05      1.889e+07       2.67e+06
        320×    8        2,560         No      1.425e+05      1.889e+07       2.70e+06
        256×   12        3,072         No      1.425e+05      1.889e+07       2.70e+06
        320×   12        3,840         No      1.425e+05      1.889e+07       2.70e+06
        128×   32        4,096         No      1.416e+05      1.889e+07       2.68e+06
        256×   16        4,096         No      1.425e+05      1.889e+07       2.70e+06
        512×    8        4,096         No      1.442e+05      1.889e+07       2.75e+06
         80×   64        5,120         No      1.412e+05      1.889e+07       2.67e+06
        320×   16        5,120         No      1.425e+05      1.889e+07       2.70e+06
        512×   12        6,144         No      1.442e+05      1.889e+07       2.75e+06
         80×   80        6,400         No      1.412e+05      1.889e+07       2.67e+06
         32×  256        8,192        Yes      1.345e+05      1.889e+07       2.55e+06
        128×   64        8,192         No      1.416e+05      1.889e+07       2.68e+06
        256×   32        8,192         No      1.425e+05      1.889e+07       2.70e+06
        512×   16        8,192         No      1.442e+05      1.889e+07       2.75e+06
         32×  320       10,240        Yes      1.349e+05      1.889e+07       2.56e+06
         80×  128       10,240         No      1.412e+05      1.889e+07       2.67e+06
        128×   80       10,240         No      1.416e+05      1.889e+07       2.68e+06
        320×   32       10,240         No      1.425e+05      1.889e+07       2.70e+06
         32×  512       16,384        Yes      1.353e+05      1.889e+07       2.57e+06
         64×  256       16,384        Yes      1.416e+05      1.889e+07       2.68e+06
        128×  128       16,384         No      1.416e+05      1.889e+07       2.68e+06
        256×   64       16,384         No      1.425e+05      1.889e+07       2.70e+06
        512×   32       16,384         No      1.442e+05      1.889e+07       2.75e+06

    Out of 96 grid-search configs per WReg size (11×11=121 minus those exceeding max_total_pes=17000):
      WReg=200: 26 successes (pe_rows≥128, plus SACols Z at 32×512, 64×256)
      WReg=300: 27 successes (same + 32×320 SACols Z)
      WReg=400: 36 successes (adds pe_rows=80 configs)

    ANALYSIS — WReg footprint depends on pe_rows (uniform grid [4..512]):
      At pe_rows=128: WReg footprint ≈ 170 (no SACols Z) → fits WReg=200 ✓
      At pe_rows=80:  WReg footprint ≈ 300 (no SACols Z) → fits WReg=400 ✓, fails WReg=200/300
      At pe_rows=64:  WReg footprint > 400 → only SACols Z configs work (pe_cols≥160)
      At pe_rows=32:  WReg footprint ≈ 816 → only SACols Z with very high cols (256+)

      Key transitions in the uniform grid:
        pe_rows=128 is the threshold for WReg=200/300 without SACols Z.
        pe_rows=80 is the threshold for WReg=400 without SACols Z.
        Below pe_rows=80, only SACols Z configs (high pe_cols) can satisfy WReg.

    ANALYSIS — Latency behavior across pe_cols:
      At pe_rows=128, latency by pe_cols:
        pe_cols=4:   L=2.760e+07 cc
        pe_cols=8:   L=1.915e+07 cc
        pe_cols=12:  L=1.889e+07 cc  ← latency plateau starts
        pe_cols=16+: L=1.889e+07 cc  (no further improvement)
      Latency plateaus at pe_cols=12 because 12 columns provide enough spatial coverage
      for the tile_size=80 workload. Extra columns beyond 12 don't reduce GB iterations.
      




      
Eyeriss Case Study 2: IntermediateRegister Size Sweep (200, 300, 400 entries) find: A) Minimum PEs configuration that fits IntReg constraint, B) Minimum latency configuration
  FSRCNN 8-layer full fusion, Eyeriss, GB=128KB, InReg=34, WReg=500, OutReg=64, tile_size=120
  Grid search: 9 PE rows × 10 PE cols = 90 configs per IntReg size

    python3 experiment_runner.py --sweep-intreg-pe \
    --workload fsrcnn --fusion full --variant 8layer \
    --intermediate-reg-sizes 200 300 400 \
    --input-reg 234 --weight-reg 500 --output-reg 64 --tile-size 120

    PE grid: rows=[4,8,12,14,16,28,32,56,84], cols=[4,8,16,32,56,64,80,160,240,320]
    SACols Z active when pe_cols ≥ 160 (i.e. pe_cols ≥ 2 × tile_size)

    SUMMARY (A) — Minimum PEs Configuration:
      IntReg  PE Config  Total PEs  SACols Z  Energy(μJ)  Latency(cc)    EDP
       200    84×4          336      No       2.880e+04   3.525e+07   1.02e+06
       300    84×4          336      No       2.880e+04   3.525e+07   1.02e+06
       400    84×4          336      No       2.880e+04   3.525e+07   1.02e+06

    SUMMARY (B) — Minimum Latency Configuration:
      IntReg  PE Config  Total PEs  SACols Z  Energy(μJ)  Latency(cc)    EDP
       200    84×16       1,344      No       2.880e+04   1.889e+07   5.46e+05
       300    84×16       1,344      No       2.880e+04   1.889e+07   5.46e+05
       400    84×16       1,344      No       2.880e+04   1.889e+07   5.46e+05

    Out of 90 grid-search configs per IntReg size:
      IntReg=200: 11 successes, 79 failures → ALL IDENTICAL to IntReg=300 and 400
      IntReg=300: 11 successes, 79 failures
      IntReg=400: 11 successes, 79 failures
    Total: 33/270 successful. NO variation across IntReg values.

    The unsuccessful configs are caused by WReg violations at pe_rows < 84, which is independent of IntReg size.
    The successful configs are identical across all IntReg sizes, confirming that IntReg is not the binding constraint.

    minimum IntReg to cause a constraint:
      The footprint is 130 entries. An IntReg < 130 would start constraining configs.
      This is below the default of 320 entries.
      For FSRCNN, the intermediate Z dimensions are small (12-56 channels per layer).
      Deeper networks with wider intermediate layers (e.g., ResNet with Z=256+)
      would have much larger IntReg footprints and could become binding.

    CONCLUSION:
      IntReg is a non-binding constraint for FSRCNN 8-layer fusion at any tested size.
      The 130-entry footprint (fixed by the network's channel widths) fits comfortably
      in even a 200-entry register. The design bottleneck remains WReg (Case Study 1),
      not IntReg.
      This case study will lead to different conclusions for workloads with wider intermediate layers channels.


Eyeriss Case Study 3: OutRegister Size Sweep (200, 300, 400 entries) find: A) Minimum PEs configuration that fits OutReg constraint, B) Minimum latency configuration
  FSRCNN 8-layer full fusion, Eyeriss, GB=128KB, InReg=34, WReg=500, IntReg=32, tile_size=120

    python3 experiment_runner.py --sweep-outreg-pe \
    --workload fsrcnn --fusion full --variant 8layer \
    --output-reg-sizes 200 300 400 \
    --input-reg 34 --weight-reg 500 --intermediate-reg 32 --tile-size 120

    PE grid: rows=[4,8,12,14,16,28,32,56,84], cols=[4,8,16,32,56,64,80,160,240,320]
    SACols Z active when pe_cols ≥ 160

    SUMMARY (A) — Minimum PEs Configuration:
      OutReg  PE Config  Total PEs  SACols Z  Energy(μJ)  Latency(cc)    EDP
       200    84×4          336      No       2.880e+04   3.525e+07   1.02e+06
       300    84×4          336      No       2.880e+04   3.525e+07   1.02e+06
       400    84×4          336      No       2.880e+04   3.525e+07   1.02e+06

    SUMMARY (B) — Minimum Latency Configuration:
      OutReg  PE Config  Total PEs  SACols Z  Energy(μJ)  Latency(cc)    EDP
       200    84×16       1,344      No       2.880e+04   1.889e+07   5.46e+05
       300    84×16       1,344      No       2.880e+04   1.889e+07   5.46e+05
       400    84×16       1,344      No       2.880e+04   1.889e+07   5.46e+05

    Out of 90 grid-search configs per OutReg size:
      OutReg=200: 11 successes, 79 failures → ALL IDENTICAL to OutReg=300 and 400
      OutReg=300: 11 successes, 79 failures
      OutReg=400: 11 successes, 79 failures
    Total: 33/270 successful. NO variation across OutReg values.

    ANALYSIS — Why OutReg has ZERO effect:
      The OutRegister stores only the 'out' operand (bypasses in, w, int_in, int_out).
      out_coupling = [Z7][P][Q] — only the LAST layer's output dimensions.
        The maximum possible OutReg_Z7 = Z7_shape = 16
        when SARows_Z7=1, which is already the case and still only 16 entries.
        P and Q are not stored in OutReg (constrained to 1) — spatial output
        dimensions are handled in the GlobalBuffer and DRAM levels above.

Eyeriss Case Study 5: PE Aspect Ratio Sweep (fixed total PEs=2048)
  FSRCNN 8-layer full fusion, Eyeriss, GB=128KB, tile_size=120
  Registers: InReg=34, WReg=500, IntReg=32, OutReg=64

    python3 experiment_runner.py --sweep-pe-aspect \
    --workload fsrcnn --fusion full --variant 8layer \
    --total-pes 2048 \
    --input-reg 34 --weight-reg 500 --intermediate-reg 32 --output-reg 64 \
    --tile-size 120 --gb-size 128 --verbose 2>&1 | tee results/EY_CS5_FSRCNN.log

    Infeasible configs: 1x2048, 2x1024, 4x512, 8x256, 16x128, 32x64
    (column-heavy configs fail because FSRCNN's tiny channel counts
    Z_i ∈ {1,12,56} cannot fill SARows when pe_rows is small)

    Results:
      PE         Energy (μJ)   Latency (cc)    EDP
      64x32      1.517e+05     1.889e+07       2.87e+06    ← EDP minimum
      128x16     1.523e+05     1.889e+07       2.88e+06
      256x8      1.531e+05     1.889e+07       2.90e+06
      512x4      1.548e+05     1.889e+07       2.95e+06
      1024x2     1.562e+05     1.889e+07       2.98e+06
      2048x1     1.562e+05     1.918e+07       3.03e+06

    Energy:   monotonically increases 64x32→2048x1 (+3.0 %).
              Very gentle slope — all configs within 3 % of minimum.
              Minimum at 64x32 (1.517e+05 μJ).

    Latency:  identical at 1.889e+07 cc for all configs except 2048x1
              (1.918e+07 cc, +1.5 %). GlobalBuffer is the bottleneck
              level and its access pattern is invariant to PE aspect ratio.

    EDP:      minimum at 64x32 (2.87e+06), monotonically rising to
              2048x1 (3.03e+06, +5.6 %). Very flat landscape —
              aspect ratio has minimal impact on FSRCNN performance.

    Per-level access counts (reads):
      PE         GB Reads      InReg Reads     WReg Reads    IntReg R+W
      64x32      302,227,200   4,613,760,000   9,741,772,800   15,178,752,000
      128x16     302,227,200   5,271,091,200   9,741,772,800   15,178,752,000
      256x8      302,227,200   6,271,603,200   9,741,772,800   15,178,752,000
      512x4      302,227,200   8,304,768,000   9,741,772,800   15,178,752,000
      1024x2     302,227,200   9,741,772,800   9,741,772,800   15,178,752,000
      2048x1     302,227,200   9,741,772,800   9,741,772,800   15,178,752,000

    GlobalBuffer reads: constant (302M) — tile_size=80 produces identical
    spatial tiling for all row-heavy configs.
    WReg reads: constant (9.74B) — all weight iterations are register-level.
    InReg reads: increase 4.61B→9.74B (+111 %) with more rows. More rows
    means more C/S factors mapped to SARows, but since rows exceed actual
    channel dimensions, residual iterations shift to InRegister.
    IntReg R+W: constant (15.2B) — Z intermediates are invariant.

    Key insight: FSRCNN is so small (Z≤56, C≤56) that even 64 rows already
    exceed all channel dimensions, making additional rows wasted. The energy
    difference comes solely from InReg overhead. Aspect ratio is almost
    irrelevant for this workload.




        
        
Thesis Experiment Runner

A systematic framework for sweeping independent variables and collecting results
for layer fusion experiments.

Independent Variables:
- Workload: FSRCNN, MC-CNN, VGG16, ResNet18
- Fusion Level: single-layer, block-fused, fully-fused
- Architecture: Memory sizes, PE array dimensions
- Tile Size: For tile size sweep case study, with optional bandwidth scaling

Dependent Variables (collected):
- Energy (μJ)
- Latency (cycles)
- EDP (Energy-Delay Product)
- Memory Operations (MOPs) per level
- Utilization
- Mapping search time (s)

Usage:
    python experiment_runner.py --help
    python experiment_runner.py --workload fsrcnn --fusion-level block
    python experiment_runner.py --sweep-all --output results.csv

================================================================================
SAMPLE COMMANDS
================================================================================
0. Eyeriss run
    python3 experiment_runner.py --single \
    --workload fsrcnn --fusion single --variant L7_output \
    --arch-type eyeriss --gb-size 128 --pe-rows 16 --pe-cols 18 --verbose 

1. LIST AVAILABLE WORKLOADS
   --------------------------
   python3 experiment_runner.py --list


2. SINGLE WORKLOAD, FIXED ARCHITECTURE (one run, no sweep)
   -----------------------------------------------
   # FSRCNN 8-layer full fusion with fixed architecture
   python3 experiment_runner.py --single \
       --workload fsrcnn --fusion full --variant 8layer \
       --fmem-size 256 --wmem-size 512 --pe-rows 16 --pe-cols 128 \
       --verbose

   # ResNet18 block fusion (stage1 = 4 layers)
   python3 experiment_runner.py --single \
       --workload resnet18 --fusion block --variant stage1 \
       --fmem-size 1056 --wmem-size 524 --pe-rows 16 --pe-cols 128 \
       --verbose


3. SINGLE WORKLOAD, SWEEP ARCHITECTURE (memory sizes, PE config)
   ---------------------------------------------------------------
   # Sweep FMEM, WMEM sizes and PE configs: 3x2x3 = 18 runs
   python3 experiment_runner.py --sweep-arch \
       --workload fsrcnn --fusion full --variant 8layer \
       --fmem-sizes 128 256 512 --wmem-sizes 256 512 \
       --pe-configs 8x64 16x128 32x256 \
       --output results/fsrcnn_arch_sweep.csv --verbose

   # Fixed FMEM/WMEM, sweep PE configs only
   python3 experiment_runner.py --sweep-arch \
       --workload fsrcnn --fusion full --variant 8layer \
       --fmem-sizes 1056 --wmem-sizes 524 \
       --pe-configs 8x64 16x128 32x256 64x128 \
       --output results/fsrcnn_pe_sweep.csv --verbose


4. SINGLE WORKLOAD, SWEEP TILE SIZES (output tile size case study)
   ------------------------------------------------------------------
   # Auto-compute valid tile sizes, with bandwidth scaling (default)
   # NOTE: tile_size is the OUTPUT tile size. For stride-aware workloads,
   # input tile sizes are derived: input_tile = output_tile * cumulative_stride

   python3 experiment_runner.py --sweep-tile-sizes \
       --workload fsrcnn --fusion full --variant 8layer \
       --fmem-size 1056 --wmem-size 524 --pe-rows 16 --pe-cols 128 \
       --output results/fsrcnn_tile_sweep.csv --verbose

   # Specific tile sizes, no bandwidth scaling
   python3 experiment_runner.py --sweep-tile-sizes \
       --workload fsrcnn --fusion full --variant 8layer \
       --tile-sizes 128 64 32 16 8 \
       --no-scale-bandwidth \
       --output results/fsrcnn_tile_sweep_no_bw_scale.csv --verbose

   # ResNet18 block fusion tile sweep (all stride=1 within block)
   python3 experiment_runner.py --sweep-tile-sizes \
       --workload resnet18 --fusion block --variant stage3_b2 \
       --fmem-size 1056 --wmem-size 524 --pe-rows 16 --pe-cols 128 \
       --output results/resnet18_stage3_tile_sweep.csv --verbose

   # VGG16 full fusion (only valid output tiles: 7, 1 due to cumulative stride=16)
   python3 experiment_runner.py --sweep-tile-sizes \
       --workload vgg16 --fusion full --variant full \
       --fmem-size 1056 --wmem-size 524 --pe-rows 16 --pe-cols 128 \
       --output results/vgg16_full_tile_sweep.csv --verbose


5. COMBINED SWEEP: ARCHITECTURE + TILE SIZES (2D grid)
   -----------------------------------------------------
   # Sweep both arch params AND tile sizes simultaneously
   # This creates a grid: (FMEM × WMEM × PE configs × tile sizes)
   
   python3 experiment_runner.py --sweep-arch-tiles \
       --workload fsrcnn --fusion full --variant 8layer \
       --fmem-sizes 512 1024 --wmem-sizes 256 512 \
       --pe-configs 16x128 32x128 \
       --tile-sizes 128 64 32 \
       --output results/fsrcnn_arch_tile_grid.csv --verbose

   # With bandwidth scaling disabled
   python3 experiment_runner.py --sweep-arch-tiles \
       --workload resnet18 --fusion block --variant stage1 \
       --fmem-sizes 256 512 1024 --wmem-sizes 256 512 \
       --pe-configs 16x128 \
       --no-scale-bandwidth \
       --output results/resnet18_arch_tile_grid.csv --verbose


6. SWEEP ALL WORKLOADS (all networks, all fusion levels)
   -------------------------------------------------------
   # Default sweep over all registered workloads
   python3 experiment_runner.py --sweep-workloads \
       --output results/all_workloads.csv --verbose

   # Filter by network and fusion level
   python3 experiment_runner.py --sweep-workloads \
       --networks fsrcnn resnet18 \
       --fusion-levels full block \
       --output results/fsrcnn_resnet_fused.csv --verbose


7. COMBINING WITH LOGGING
   -----------------------
   # Capture stdout and stderr to log file
   python3 experiment_runner.py --sweep-tile-sizes \
       --workload fsrcnn --fusion full --variant 8layer \
       --verbose --output results/fsrcnn_tile_sweep.csv \
       2>&1 | tee results/fsrcnn_tile_sweep.log

   # With JSON export
   python3 experiment_runner.py --sweep-arch \
       --workload resnet18 --fusion block --variant stage1 \
       --json --output results/resnet18_stage1.csv --verbose


================================================================================
WORKLOAD NAMING CONVENTIONS
================================================================================

FSRCNN-TDC (activation-dominant, super-resolution):
  --workload fsrcnn --fusion single --variant L0  (or L1, L2, ..., L7)
  --workload fsrcnn --fusion 2layer --variant 2layer
  --workload fsrcnn --fusion 3layer --variant 3layer
  --workload fsrcnn --fusion full --variant 8layer

MC-CNN (activation-dominant, stereo matching):
  --workload mccnn --fusion single --variant L0  (or L1, L2, L3)
  --workload mccnn --fusion 2layer --variant 2layer
  --workload mccnn --fusion full --variant 4layer

VGG16 (weight-dominant, classification):
  --workload vgg16 --fusion single --variant L0  (L0-L12 for conv, L13-L15 for FC)
  --workload vgg16 --fusion block --variant block1  (block1-block5, fc)
  --workload vgg16 --fusion full --variant full  (13 conv layers, cumulative stride=16)

ResNet18 (weight-dominant, classification):
  --workload resnet18 --fusion single --variant L0  (L0-L19 for conv, L20 for FC)
  --workload resnet18 --fusion block --variant stage1  (stage1=4 layers, stage2_b2/3_b2/4_b2=2 layers each)
  --workload resnet18 --fusion 2layer --variant s1b1  (s1b1, s1b2, s2b2, s3b2, s4b2)
  --workload resnet18 --fusion full --variant full  (17 conv layers, cumulative stride=16)

================================================================================
TILE SIZE NOTES (for --sweep-tile-sizes)
================================================================================

The tile_size parameter refers to the OUTPUT tile size (Q dimension).

For multi-layer fusion with strides (ResNet full, VGG16 full), input tile sizes
are automatically derived using: input_tile = output_tile × cumulative_stride

Examples:
  - FSRCNN 8-layer (stride=1): Many valid tiles (120, 96, 80, 64, ...)
  - ResNet18 full (stride=16): Only tiles 7 and 1 are valid (7×16=112 input)
  - VGG16 full (stride=16): Only tiles 7 and 1 are valid (7×16=112 input)
  - ResNet18 block fusions (stride=1 within block): Many valid tiles

When --tile-sizes is not specified, valid tile sizes are auto-computed based
on the workload dimensions and stride constraints.

================================================================================
FUSION COMPARISON CASE STUDIES  (--compare-fusion)
================================================================================

  Two case studies comparing FULL fusion against non-fused / partially-fused
  executions, evaluated on both DepFiN and Eyeriss architectures across all
  four workloads (FSRCNN, MC-CNN, VGG16, ResNet18).

  ────────────────────────────────────────────────────────────────────────────
  METHODOLOGY
  ────────────────────────────────────────────────────────────────────────────

  CS1 — FULL FUSION  vs  SUM OF SINGLES (non-fused)
    Full fusion processes all N layers in a single tiled execution, keeping
    intermediate feature maps on-chip (FMEM / GlobalBuffer). The non-fused
    baseline runs each layer independently: every intermediate activation
    is written to DRAM after producing it and read back from DRAM before
    the next layer consumes it.

    Metric comparison:  Full_Fusion  /  Σ(single_layer_i)

  CS2 — FULL FUSION  vs  SUM OF INTERMEDIATE-LEVEL FUSED SEGMENTS
    Instead of single layers, the baseline fuses small groups of adjacent
    layers (2-layer or 3-layer or block segments). The intermediates between
    segments still pass through DRAM, but fewer of them do compared to CS1.

    Metric comparison:  Full_Fusion  /  Σ(intermediate_segment_j)

  Intermediate fusion levels per workload:
    FSRCNN   → 3-layer segments: L0_L1_L2, L3_L4_L5, L6_L7 (2-layer)
    MC-CNN   → 2-layer segments: L0_L1, L2_L3
    VGG16    → block segments:   block1(2L), block2(2L), block3(3L), block4(3L), block5(3L)
    ResNet18 → 2-layer segments: s1b1(2L), s1b2(3L), s2b1(2L), s2b2(2L),
                                 s3b1(2L), s3b2(2L), s4b1(2L), s4b2(2L)

  ────────────────────────────────────────────────────────────────────────────
  FAIR MEMORY SIZING (per-variant WMEM auto-sizing)
  ────────────────────────────────────────────────────────────────────────────

  IMPORTANT: For fair comparison, the Weight Memory (WMEM) is automatically
  sized per variant based on the actual weight parameters needed:
    
    WMEM_bytes = Σ_layers (Z_i × C_i × R_i × S_i) × bytes_per_weight

  This ensures that each fusion level is evaluated on an architecture whose
  WMEM is right-sized for its workload:
    - Intermediate segments: medium WMEM (block's weights)
    - Full fusion: large WMEM (all layers' weights)

  WHY THIS MATTERS:
    Accelergy computes energy-per-access as a function of memory depth.
    A larger WMEM has higher energy per read/write. Without per-variant
    sizing, single layers would be evaluated on the full-fusion WMEM
    (e.g., 14400 KB for VGG16), inflating their energy and unfairly
    making fusion look better.


  For Eyeriss: the GlobalBuffer energy is fixed (not size-dependent in the
  current energy model), so Eyeriss comparisons are less affected. However,
  register sizes still influence mapping quality.


  ────────────────────────────────────────────────────────────────────────────
  ARCHITECTURE CONFIGURATIONS USED
  ────────────────────────────────────────────────────────────────────────────

  DepFiN architecture (all workloads):
    PE array: 16×128 (2048 PEs)
    FMEM: 1056 KB (fixed across all variants)
    WMEM: AUTO-SIZED per variant (from weight parameters)
    FMEM BW scales with tile size: BW_scaled = BW_base × (tile_size / 128)
    Tile size: auto-detected per fusion level
      FSRCNN:    4x512 PEs (2048), FMEM=1056KB, WMEM=auto, tile=auto
      MC-CNN:    8x256 PEs (2048), FMEM=1056KB, WMEM=auto, tile=auto
      VGG16:     16x128 PEs (2048), FMEM=1056KB, WMEM=auto, tile=auto
      ResNet18:  16x128 PEs (2048), FMEM=1056KB, WMEM=auto, tile=auto
      

    
  Eyeriss architecture (min-latency / saturated PE configs per workload):
    FSRCNN:   84×16 PEs (1344),  GB=128KB, WReg=500, IntReg=32, OutReg=64, InReg=34,  tile=120
    MC-CNN:   56×64 PEs (3584),  GB=128KB, WReg=600, IntReg=32, OutReg=64, InReg=34,  tile=69
    VGG16:    196×64 PEs (12544), GB=128KB, WReg=5000, IntReg=300, OutReg=64, InReg=400, tile=1
    ResNet18: 256×32 PEs (8192),  GB=128KB, WReg=4000, IntReg=300, OutReg=64, InReg=400, tile=1

  ────────────────────────────────────────────────────────────────────────────
  HOW TO REPRODUCE
  ────────────────────────────────────────────────────────────────────────────

  # DepFiN (WMEM shared per case study — max(min_wmem) across all variants)
  python3 experiment_runner.py --compare-fusion-vs-partial --workload fsrcnn   --arch-type depfin --fmem-size 1056 --pe-rows 4  --pe-cols 512
  python3 experiment_runner.py --compare-partial-vs-single --workload fsrcnn    --arch-type depfin --fmem-size 1056 --pe-rows 8  --pe-cols 256
  
  # Eyeriss
  python3 experiment_runner.py --compare-full-vs-partial -w vgg16          --arch-type eyeriss --pe-rows 512 --pe-cols 32 --weight-reg 1200          --gb-size 128 --tile-size 1 --input-reg 400 --intermediate-reg 300          --output-reg 64
  python3 experiment_runner.py --compare-partial-vs-single -w vgg16          --arch-type eyeriss --pe-rows 512 --pe-cols 32 --weight-reg 1200          --gb-size 128 --tile-size 1 --input-reg 400 --intermediate-reg 300          --output-reg 64
  
"""

import os
import sys
import csv
import json
import time
import math
import argparse
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
from itertools import product

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factors import Shape, Coupling
from settings import Settings
from engine import run_engine
from cost_model import EDP, Energy, Latency, MOPs, DRAM_MOPs
from arch import Arch
from prints import factorsString

# Import workloads from case_studies_computations
from case_studies_computations import (
    # FSRCNN-TDC
    fsrcnn_tdc_single_layers,
    fsrcnn_tdc_2layer_fused,
    fsrcnn_tdc_3layer_fused,
    fsrcnn_tdc_8layer_fused,
    fsrcnn_tdc_8layer_coupling,
    # MC-CNN
    mccnn_single_layers,
    mccnn_2layer_fused,
    mccnn_4layer_fused,
    mccnn_4layer_coupling,
    # VGG16
    vgg16_single_layers,
    vgg16_block_fused,
    vgg16_block_couplings,
    vgg16_full_fused,
    vgg16_full_coupling,
    # ResNet18
    resnet18_single_layers,
    resnet18_block_fused,
    resnet18_block_couplings,
    resnet18_2layer_fused,
    resnet18_2layer_couplings,
    resnet18_full_fused,
    resnet18_full_coupling,
    # FSRCNN 3-layer per-variant couplings
    fsrcnn_tdc_3layer_couplings,
)

from computations import (
    conv_coupling,
    conv_2layers_coupling,
    conv_3layers_coupling,
    conv_4layers_coupling,
    conv_5layers_coupling,
    conv_8layers_coupling,
)

# Import architecture builder
# For now, we use pre-built architectures from architectures.py
# and the thesis_arch module for custom configurations
from architectures.architectures import arch_eyeriss_conv
try:
    from architectures.thesis_arch import (
        create_thesis_architecture,
        create_thesis_architecture_10layers,  # Legacy, for backward compatibility
        ThesisArchConfig,
        get_baseline_config,
        get_energy_values_from_accelergy,
        # Eyeriss architecture support
        create_eyeriss_architecture,
        EyerissArchConfig,
        get_eyeriss_energy_values,
        # Constrained Eyeriss (fully deterministic mapping)
        create_constrained_eyeriss_architecture,
    )
    THESIS_ARCH_AVAILABLE = True
except ImportError:
    THESIS_ARCH_AVAILABLE = False


# =============================================================================
# DEFAULT EYERISS REGISTER VALUES (from thesis_arch.EyerissArchConfig)
# =============================================================================
# These are used as defaults in ExperimentConfig, CLI parser, and sweep functions
# to ensure consistency with the architecture definition in thesis_arch.py.
if THESIS_ARCH_AVAILABLE:
    _EYERISS_DEFAULTS = EyerissArchConfig()  # Instantiate with all defaults
else:
    # Fallback if thesis_arch is not available
    class _EyerissFallback:
        input_reg_entries = 50
        weight_reg_entries = 15
        intermediate_out_reg_entries = 320
        output_reg_entries = 32
        global_buffer_size_B = 131072
        pe_cols = 14
        pe_rows = 12
    _EYERISS_DEFAULTS = _EyerissFallback()

# =============================================================================
# ORIGINAL arch_eyeriss_conv REGISTER/MEMORY SIZES (from architectures.py)
# =============================================================================
# These are the original Eyeriss architecture sizes, used as defaults for
# single-layer runs in --compare-fusion when no per-variant sizes are specified.
# They represent physically realistic register sizes for the original Eyeriss.
_ARCH_EYERISS_CONV_SIZES = {
    'global_buffer_kB': 128,          # 16384*8 entries = 131072 bytes = 128 KB
    'input_reg_entries': 24,          # 12*2 = 24 entries
    'weight_reg_entries': 384,        # 192*2 = 384 entries
    'output_reg_entries': 32,         # 16*2 = 32 entries
    'intermediate_reg_entries': 32,   # Same as output_reg for single layers
}


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

"""Configuration for a single experiment run."""
@dataclass
class ExperimentConfig:
    # Experiment identification
    experiment_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Workload specification
    workload_name: str = ""          # e.g., "fsrcnn", "vgg16"
    workload_variant: str = ""       # e.g., "L0", "block1", "full"
    fusion_level: str = ""           # "single", "block", "full"
    num_fused_layers: int = 1        # Actual number of layers fused
    
    # Architecture type selection
    arch_type: str = "depfin"        # "depfin" or "eyeriss"
    arch_name: str = "thesis_arch"
    
    # DepFiN-style architecture (separate FMEM/WMEM)
    fmem_size_kB: int = 1056
    wmem_size_kB: int = 524
    
    # Eyeriss-style architecture (unified GlobalBuffer)
    gb_size_kB: int = field(default_factory=lambda: _EYERISS_DEFAULTS.global_buffer_size_B // 1024)
    
    # Eyeriss register sizes (entries) — defaults from thesis_arch.EyerissArchConfig
    input_reg_entries: int = field(default_factory=lambda: _EYERISS_DEFAULTS.input_reg_entries)
    weight_reg_entries: int = field(default_factory=lambda: _EYERISS_DEFAULTS.weight_reg_entries)
    intermediate_reg_entries: int = field(default_factory=lambda: _EYERISS_DEFAULTS.intermediate_out_reg_entries)
    output_reg_entries: int = field(default_factory=lambda: _EYERISS_DEFAULTS.output_reg_entries)
    
    # PE array configuration (shared)
    pe_rows: int = 16
    pe_cols: int = 128
    
    # Tile size configuration (for tile size sweep case study)
    # NOTE: tile_size refers to the OUTPUT tile size (Q dimension).
    # For multi-layer fusion with strides (e.g., ResNet, VGG), input layer tile sizes
    # are automatically derived: input_tile = output_tile * cumulative_stride
    # Example: ResNet18 full fusion (cum_stride=16), tile_size=7 → input tile = 112
    tile_size: Optional[int] = None           # Output tile size override (None = auto GCD)
    scale_bandwidth: bool = True             # Scale FMEM bandwidth with tile size
    base_tile_size: int = 128                  # Reference tile size for scaling
    
    # Energy override
    dram_energy_override: Optional[float] = None  # If set, override DRAM energy (pJ/byte)
    
    # Settings
    bias_read: bool = False
    verbose: bool = False
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = f"{self.workload_name}_{self.workload_variant}_{self.timestamp}"
        # Eyeriss GlobalBuffer bandwidth is a fixed physical property of the buffer,
        # not dependent on workload tiling.  Bandwidth scaling is a DepFiN-only concept
        # (FMEM bus width shrinks with smaller tiles).  Force it off for Eyeriss to
        # avoid inflated latency from an artificial bandwidth reduction.
        if self.arch_type == "eyeriss":
            self.scale_bandwidth = False

"""Results from a single experiment run."""
@dataclass
class ExperimentResult:
    # Configuration reference
    config: ExperimentConfig = None
    
    # Success/failure
    success: bool = False
    error_message: str = ""
    
    # Performance metrics
    energy_uJ: float = 0.0
    latency_cycles: int = 0
    edp: float = 0.0
    mops: int = 0
    dram_reads: int = 0
    dram_writes: int = 0
    utilization: float = 0.0
    
    # Timing
    mapping_time_s: float = 0.0
    
    # Additional details (optional)
    mops_per_level: Dict[str, int] = field(default_factory=dict)
    mapping_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        result = {
            # From config
            "experiment_id": self.config.experiment_id if self.config else "",
            "workload_name": self.config.workload_name if self.config else "",
            "workload_variant": self.config.workload_variant if self.config else "",
            "fusion_level": self.config.fusion_level if self.config else "",
            "num_fused_layers": self.config.num_fused_layers if self.config else 0,
            "arch_type": self.config.arch_type if self.config else "depfin",
            "arch_name": self.config.arch_name if self.config else "",
            "fmem_size_kB": self.config.fmem_size_kB if self.config else 0,
            "wmem_size_kB": self.config.wmem_size_kB if self.config else 0,
            "gb_size_kB": self.config.gb_size_kB if self.config else 0,
            "pe_rows": self.config.pe_rows if self.config else 0,
            "pe_cols": self.config.pe_cols if self.config else 0,
            "tile_size": self.config.tile_size if self.config else None,
            "scale_bandwidth": self.config.scale_bandwidth if self.config else False,
            # Results
            "success": self.success,
            "error_message": self.error_message,
            "energy_uJ": f"{self.energy_uJ:.3e}",
            "latency_cycles": f"{self.latency_cycles:.3e}",
            "edp": f"{self.edp:.2e}",
            "mops": self.mops,
            "utilization": self.utilization,
            "mapping_time_s": self.mapping_time_s,
            "mapping": self.mapping_summary,
        }
        return result


# =============================================================================
# WORKLOAD REGISTRY
# =============================================================================
"""
    Registry of all available workloads organized by network and fusion level.
    
    Structure:
        workloads[network_name][fusion_level] = {
            variant_name: (shape, coupling, num_layers)
        }
"""    
class WorkloadRegistry:
    
    def __init__(self):
        self.workloads = self._build_registry()
    
    """Build the complete workload registry."""
    def _build_registry(self) -> Dict[str, Dict[str, Dict[str, Tuple[Shape, Coupling, int]]]]:
        registry = {
            # =========================================================
            # FSRCNN-TDC (Activation-Dominant)
            # =========================================================
            "fsrcnn": {
                "single": {
                    name: (shape, conv_coupling, 1)
                    for name, shape in fsrcnn_tdc_single_layers.items()
                },
                "2layer": {
                    name: (shape, conv_2layers_coupling, 2)
                    for name, shape in fsrcnn_tdc_2layer_fused.items()
                },
                "3layer": {
                    name: (shape, fsrcnn_tdc_3layer_couplings[name],
                           3 if name != 'L6_L7_expanding_output' else 2)
                    for name, shape in fsrcnn_tdc_3layer_fused.items()
                },
                "full": {
                    "8layer": (fsrcnn_tdc_8layer_fused, fsrcnn_tdc_8layer_coupling, 8)
                },
            },
            
            # =========================================================
            # MC-CNN (Activation-Dominant)
            # =========================================================
            "mccnn": {
                "single": {
                    name: (shape, conv_coupling, 1)
                    for name, shape in mccnn_single_layers.items()
                },
                "2layer": {
                    name: (shape, conv_2layers_coupling, 2)
                    for name, shape in mccnn_2layer_fused.items()
                },
                "full": {
                    "4layer": (mccnn_4layer_fused, mccnn_4layer_coupling, 4)
                },
            },
            
            # =========================================================
            # VGG16 (Weight-Dominant)
            # =========================================================
            "vgg16": {
                "single": {
                    name: (shape, conv_coupling, 1)
                    for name, shape in vgg16_single_layers.items()
                },
                "block": {
                    name: (shape, vgg16_block_couplings[name], 
                           2 if name in ['block1', 'block2', 'block3', 'block4', 'block5'] else 1)
                    for name, shape in vgg16_block_fused.items()
                },
                "full": {
                    "13layer": (vgg16_full_fused, vgg16_full_coupling, 13)
                },
            },
            
            # =========================================================
            # ResNet18 (Weight-Dominant)
            # =========================================================
            "resnet18": {
                "single": {
                    name: (shape, conv_coupling, 1)
                    for name, shape in resnet18_single_layers.items()
                },
                "2layer": {
                    name: (shape, resnet18_2layer_couplings[name],
                           3 if name == 's1b2' else 2)
                    for name, shape in resnet18_2layer_fused.items()
                },
                "block": {
                    name: (shape, resnet18_block_couplings[name],
                           5 if name == 'stage1' else 4)
                    for name, shape in resnet18_block_fused.items()
                },
                "full": {
                    "17layer": (resnet18_full_fused, resnet18_full_coupling, 17)
                },
            },
        }
        
        return registry
    
    """Get a specific workload by network, fusion level, and variant."""
    def get_workload(self, network: str, fusion_level: str, variant: str) -> Tuple[Shape, Coupling, int]:
        if network not in self.workloads:
            raise ValueError(f"Unknown network: {network}. Available: {list(self.workloads.keys())}")
        if fusion_level not in self.workloads[network]:
            raise ValueError(f"Unknown fusion level: {fusion_level} for {network}. "
                           f"Available: {list(self.workloads[network].keys())}")
        if variant not in self.workloads[network][fusion_level]:
            raise ValueError(f"Unknown variant: {variant} for {network}/{fusion_level}. "
                           f"Available: {list(self.workloads[network][fusion_level].keys())}")
        return self.workloads[network][fusion_level][variant]
    
    """List all workloads, optionally filtered by network and/or fusion level."""
    def list_workloads(self, network: Optional[str] = None, fusion_level: Optional[str] = None) -> List[Tuple[str, str, str]]:
        result = []
        for net_name, fusion_levels in self.workloads.items():
            if network and net_name != network:
                continue
            for fuse_name, variants in fusion_levels.items():
                if fusion_level and fuse_name != fusion_level:
                    continue
                for variant_name in variants.keys():
                    result.append((net_name, fuse_name, variant_name))
        return result
    
    def get_networks(self) -> List[str]:
        """Get list of available networks."""
        return list(self.workloads.keys())
    
    def get_fusion_levels(self, network: str) -> List[str]:
        """Get list of fusion levels for a network."""
        return list(self.workloads.get(network, {}).keys())


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

"""
    Main experiment runner that executes workloads and collects results.
    
    Features:
    - Run single experiments or sweep multiple variables
    - Collect and export results to CSV/JSON
    - Resume interrupted experiments
    - Progress tracking and logging
"""
class ExperimentRunner: 
    def __init__(self, output_dir: str = "results"):
        self.workload_registry = WorkloadRegistry()
        self.output_dir = output_dir
        self.results: List[ExperimentResult] = []
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
    
    """
        Create an architecture instance based on configuration.
        
        Supports two architecture types:
        - depfin: DepFiN-style with separate FMEM + WMEM (default)
        - eyeriss: Eyeriss-style with unified GlobalBuffer
        
        When tile_size is specified for eyeriss, uses the fully-constrained
        architecture (create_constrained_eyeriss_architecture) which produces
        a single deterministic mapping without search.
        
        Uses create_thesis_architecture for thesis experiments (supports any number
        of fused layers), or falls back to eyeriss_conv for basic single layer tests.
    """    
    def create_architecture(self, config: ExperimentConfig, coupling: Coupling, shape: Shape = None) -> Arch:
        # Use default architecture 
        arch = deepcopy(arch_eyeriss_conv)
        
        # If thesis architecture is available and we need custom config, use it
        if THESIS_ARCH_AVAILABLE and config.arch_name == "thesis_arch":
            try:
                if config.arch_type == "eyeriss":
                    # Create Eyeriss-like architecture with unified GlobalBuffer
                    arch_config = EyerissArchConfig(
                        global_buffer_size_B=config.gb_size_kB * 1024,  # Convert KB to Bytes
                        pe_cols=config.pe_cols,
                        pe_rows=config.pe_rows,
                        num_fused_layers=config.num_fused_layers,
                        # Register sizes
                        input_reg_entries=config.input_reg_entries,
                        weight_reg_entries=config.weight_reg_entries,
                        intermediate_out_reg_entries=config.intermediate_reg_entries,
                        output_reg_entries=config.output_reg_entries,
                        # Tile size override configuration
                        tile_size_override=config.tile_size,
                        scale_bandwidth_with_tile=config.scale_bandwidth,
                        base_tile_size_for_scaling=config.base_tile_size,
                    )
                    
                    # Build custom_energy dict if DRAM override is set
                    custom_energy = None
                    if config.dram_energy_override is not None:
                        # Get base energy from Accelergy, then override DRAM
                        base_energy = get_eyeriss_energy_values(arch_config)
                        base_energy['dram_energy'] = config.dram_energy_override
                        custom_energy = base_energy
                    
                    # If tile_size is specified, use constrained (deterministic) architecture
                    if config.tile_size is not None:
                        arch = create_constrained_eyeriss_architecture(
                            config=arch_config,
                            coupling=coupling,
                            shape=shape,
                            output_tile_size=config.tile_size,
                            num_layers=config.num_fused_layers,
                            custom_energy=custom_energy,
                        )
                    else:
                        arch = create_eyeriss_architecture(
                            config=arch_config,
                            coupling=coupling,
                            shape=shape,
                            num_layers=config.num_fused_layers,
                            custom_energy=custom_energy,
                        )
                else:
                    # Create DepFiN-style architecture with separate FMEM/WMEM (default)
                    arch_config = ThesisArchConfig(
                        feature_memory_size_B=config.fmem_size_kB * 1024,  # Convert KB to Bytes
                        weight_memory_size_B=config.wmem_size_kB * 1024,   # Convert KB to Bytes
                        pe_rows=config.pe_rows,
                        pe_cols=config.pe_cols,
                        num_fused_layers=config.num_fused_layers,
                        # Tile size override configuration
                        tile_size_override=config.tile_size,
                        scale_bandwidth_with_tile=config.scale_bandwidth,
                        base_tile_size_for_scaling=config.base_tile_size,
                    )
                    
                    # Build custom_energy dict if DRAM override is set
                    custom_energy = None
                    use_accelergy = True
                    if config.dram_energy_override is not None:
                        base_energy = get_energy_values_from_accelergy(arch_config)
                        custom_energy = {
                            'dram': config.dram_energy_override,
                            'fmem': base_energy['fmem_energy_per_byte'],
                            'wmem': base_energy['wmem_energy_per_byte'],
                            'accreg': base_energy['accreg_energy_per_byte'],
                            'compute': base_energy['compute_energy_per_mac'],
                        }
                        use_accelergy = False
                    
                    # Use generic architecture factory with proper layer count and shape
                    arch = create_thesis_architecture(
                        config=arch_config,
                        coupling=coupling,
                        shape=shape,
                        num_layers=config.num_fused_layers,
                        use_accelergy_energy=use_accelergy,
                        custom_energy=custom_energy,
                    )
            except Exception as e:
                print(f"Warning: Could not create thesis_arch ({e}), using eyeriss_conv")
                import traceback
                traceback.print_exc()
                arch = deepcopy(arch_eyeriss_conv)
        
        return arch
    

    """
        Run a single experiment with the given configuration.
        
        Steps:
        1. Get workload (shape + coupling) from registry
        2. Create architecture with specified parameters
        3. Run the mapping engine
        4. Collect and return results
    """    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        result = ExperimentResult(config=config)
        
        try:
            # Step 1: Get workload
            shape, coupling, num_layers = self.workload_registry.get_workload(
                config.workload_name,
                config.fusion_level,
                config.workload_variant
            )
            config.num_fused_layers = num_layers
            
            # Step 2: Create architecture (pass coupling and shape for thesis_arch compatibility)
            arch = self.create_architecture(config, coupling, shape)
            
            # Step 3: Check compatibility and fit constraints
            arch.checkCouplingCompatibility(coupling, shape, verbose=config.verbose)
            arch.fitConstraintsToComp(shape, enforce=True)
            
            # Step 4: Run mapping engine
            # Temporarily adjust verbosity
            original_verbose = Settings.VERBOSE
            Settings.VERBOSE = config.verbose
            
            edp, mops, energy, latency, utilization, mapping_time, final_arch = run_engine(
                arch, shape, coupling, config.bias_read, verbose=config.verbose
            )
            
            Settings.VERBOSE = original_verbose
            
            # Step 5: Collect results
            result.success = True
            result.energy_uJ = energy
            result.latency_cycles = latency
            result.edp = edp
            result.mops = mops
            dram_r, dram_w = DRAM_MOPs(final_arch)
            result.dram_reads = dram_r
            result.dram_writes = dram_w
            result.utilization = utilization
            result.mapping_time_s = mapping_time
            result.mapping_summary = factorsString(final_arch)
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            if config.verbose:
                import traceback
                traceback.print_exc()
        
        self.results.append(result)
        return result
    
    """
        Sweep over multiple workloads.
        
        Args:
            networks: List of networks to test (None = all)
            fusion_levels: List of fusion levels to test (None = all)
            arch_config: Base architecture configuration
            progress_callback: Optional callback(current, total, config) for progress
        
        Returns:
            List of ExperimentResult for all runs
    """
    def run_workload_sweep(
        self,
        networks: Optional[List[str]] = None,
        fusion_levels: Optional[List[str]] = None,
        arch_config: Optional[ExperimentConfig] = None,
        progress_callback=None
    ) -> List[ExperimentResult]:
        if arch_config is None:
            arch_config = ExperimentConfig()
        
        # Get all workloads to run
        workloads = []
        for net in (networks or self.workload_registry.get_networks()):
            for fuse in (fusion_levels or self.workload_registry.get_fusion_levels(net)):
                for (net_name, fuse_name, variant) in self.workload_registry.list_workloads(net, fuse):
                    workloads.append((net_name, fuse_name, variant))
        
        total = len(workloads)
        results = []
        
        print(f"\n{'='*60}")
        print(f"Starting workload sweep: {total} experiments")
        print(f"{'='*60}\n")
        
        for i, (network, fusion_level, variant) in enumerate(workloads):
            # Create config for this experiment
            config = ExperimentConfig(
                workload_name=network,
                workload_variant=variant,
                fusion_level=fusion_level,
                arch_name=arch_config.arch_name,
                fmem_size_kB=arch_config.fmem_size_kB,
                wmem_size_kB=arch_config.wmem_size_kB,
                pe_rows=arch_config.pe_rows,
                pe_cols=arch_config.pe_cols,
                bias_read=arch_config.bias_read,
                verbose=arch_config.verbose,
            )
            
            print(f"[{i+1}/{total}] Running: {network}/{fusion_level}/{variant}")
            
            if progress_callback:
                progress_callback(i, total, config)
            
            result = self.run_single_experiment(config)
            results.append(result)
            
            if result.success:
                print(f"  ✓ Energy: {result.energy_uJ:.2e} μJ, "
                      f"Latency: {result.latency_cycles:.2e} cycles, "
                      f"Time: {result.mapping_time_s:.2f}s")
            else:
                print(f"  ✗ Failed: {result.error_message[:50]}...")
        
        print(f"\n{'='*60}")
        print(f"Completed: {sum(1 for r in results if r.success)}/{total} successful")
        print(f"{'='*60}\n")
        
        return results
    
    """
        Sweep over architecture parameters for a fixed workload.
        
        Args:
            workload: (network, fusion_level, variant) tuple
            arch_type: Architecture type ("depfin" or "eyeriss")
            fmem_sizes_kb: List of FMEM sizes to try [DepFiN only]
            wmem_sizes_kb: List of WMEM sizes to try [DepFiN only]
            gb_sizes_kb: List of GlobalBuffer sizes to try [Eyeriss only]
            pe_configs: List of (rows, cols) PE configurations
            progress_callback: Optional callback for progress
        
        Returns:
            List of ExperimentResult for all runs
    """    
    def run_architecture_sweep(
        self,
        workload: Tuple[str, str, str],  # (network, fusion_level, variant)
        arch_type: str = "depfin",       # "depfin" or "eyeriss"
        fmem_sizes_kb: Optional[List[int]] = None,  # DepFiN only
        wmem_sizes_kb: Optional[List[int]] = None,  # DepFiN only
        gb_sizes_kb: Optional[List[int]] = None,    # Eyeriss only
        pe_configs: Optional[List[Tuple[int, int]]] = None,  # (rows, cols)
        tile_size: Optional[int] = None,  # Output tile size override (None = auto)
        input_reg: Optional[int] = None,   # Eyeriss register overrides
        weight_reg: Optional[int] = None,
        intermediate_reg: Optional[int] = None,
        output_reg: Optional[int] = None,
        progress_callback=None,
        verbose: bool = False
    ) -> List[ExperimentResult]:
        network, fusion_level, variant = workload
        
        # Default PE configs (shared)
        if pe_configs is None:
            pe_configs = [(8, 64), (16, 128), (32, 256)]
        
        results = []
        
        if arch_type == "eyeriss":
            # Eyeriss architecture: sweep GB sizes
            if gb_sizes_kb is None:
                gb_sizes_kb = [64, 128, 256]
            
            combinations = list(product(gb_sizes_kb, pe_configs))
            total = len(combinations)
            
            print(f"\n{'='*60}")
            print(f"Architecture sweep for {network}/{fusion_level}/{variant} (Eyeriss)")
            print(f"{total} configurations to test")
            print(f"GB sizes: {gb_sizes_kb} KB")
            print(f"PE configs: {[f'{r}x{c}' for r, c in pe_configs]}")
            print(f"{'='*60}\n")
            
            for i, (gb, (pe_r, pe_c)) in enumerate(combinations):
                config = ExperimentConfig(
                    workload_name=network,
                    workload_variant=variant,
                    fusion_level=fusion_level,
                    arch_type="eyeriss",
                    gb_size_kB=gb,
                    pe_rows=pe_r,
                    pe_cols=pe_c,
                    verbose=verbose,
                )
                # Apply register size overrides if provided
                if input_reg is not None:
                    config.input_reg_entries = input_reg
                if weight_reg is not None:
                    config.weight_reg_entries = weight_reg
                if intermediate_reg is not None:
                    config.intermediate_reg_entries = intermediate_reg
                if output_reg is not None:
                    config.output_reg_entries = output_reg
                if tile_size is not None:
                    config.tile_size = tile_size
                
                print(f"[{i+1}/{total}] GB={gb}KB, PE={pe_r}x{pe_c}")
                
                if progress_callback:
                    progress_callback(i, total, config)
                
                result = self.run_single_experiment(config)
                results.append(result)
                
                if result.success:
                    print(f"  ✓ EDP: {result.edp:.2e}")
                else:
                    print(f"  ✗ Failed")
        else:
            # DepFiN architecture: sweep FMEM/WMEM sizes
            if fmem_sizes_kb is None:
                fmem_sizes_kb = [128, 512, 1024]
            if wmem_sizes_kb is None:
                wmem_sizes_kb = [256, 512, 1024]
            
            combinations = list(product(fmem_sizes_kb, wmem_sizes_kb, pe_configs))
            total = len(combinations)
            
            print(f"\n{'='*60}")
            print(f"Architecture sweep for {network}/{fusion_level}/{variant} (DepFiN)")
            print(f"{total} configurations to test")
            print(f"FMEM sizes: {fmem_sizes_kb} KB, WMEM sizes: {wmem_sizes_kb} KB")
            print(f"PE configs: {[f'{r}x{c}' for r, c in pe_configs]}")
            if tile_size is not None:
                print(f"Tile size override: {tile_size}")
            print(f"{'='*60}\n")
            
            for i, (fmem, wmem, (pe_r, pe_c)) in enumerate(combinations):
                config = ExperimentConfig(
                    workload_name=network,
                    workload_variant=variant,
                    fusion_level=fusion_level,
                    arch_type="depfin",
                    fmem_size_kB=fmem,
                    wmem_size_kB=wmem,
                    pe_rows=pe_r,
                    pe_cols=pe_c,
                    tile_size=tile_size,
                    verbose=verbose,
                )
                
                print(f"[{i+1}/{total}] FMEM={fmem}KB, WMEM={wmem}KB, PE={pe_r}x{pe_c}")
                
                if progress_callback:
                    progress_callback(i, total, config)
                
                result = self.run_single_experiment(config)
                results.append(result)
                
                if result.success:
                    print(f"  ✓ EDP: {result.edp:.2e}")
                else:
                    print(f"  ✗ Failed")
        
        return results
    
    """
        Sweep over OUTPUT tile sizes for a fixed workload and architecture.
        
        This case study explores the trade-off between tile size and efficiency.

        Smaller tiles leave more PEs unused but may have different memory behavior.
        
        IMPORTANT: tile_sizes refers to the OUTPUT tile size (Q dimension).
        For multi-layer fusion with strides (e.g., ResNet, VGG with pooling),
        input layer tile sizes are automatically derived using:
            input_tile = output_tile * cumulative_stride
        
        Example: ResNet18 full fusion has cumulative_stride=16 (4 stride-2 layers).
        With output tile_size=7, the input tile becomes 7*16 = 112.
        
        In Depfin:
            With scale_bandwidth=True (default), FMEM bandwidth scales proportionally
            to tile size, isolating the tile size effect from bandwidth bottlenecks.
        
        Args:
            workload: (network, fusion_level, variant) tuple
            tile_sizes: List of OUTPUT tile sizes to try (None = auto-compute divisors)
            arch_type: Architecture type ("depfin" or "eyeriss")
            fmem_size_kb: Feature memory size in KB [DepFiN only]
            wmem_size_kb: Weight memory size in KB [DepFiN only]
            gb_size_kb: GlobalBuffer size in KB [Eyeriss only]
            pe_rows: PE array rows
            pe_cols: PE array columns (determines max tile size)
            scale_bandwidth: If True, scale FMEM bandwidth proportionally to tile size
            base_tile_size: Reference tile size when scale_bandwidth=True
            progress_callback: Optional callback for progress
            verbose: Enable verbose output
        
        Returns:
            List of ExperimentResult for all runs
    """    
    def run_tile_size_sweep(
        self,
        workload: Tuple[str, str, str],  # (network, fusion_level, variant)
        tile_sizes: Optional[List[int]] = None,  # If None, auto-compute divisors
        arch_type: str = "depfin",       # "depfin" or "eyeriss"
        fmem_size_kb: int = 1056,        # DepFiN only
        wmem_size_kb: int = 524,         # DepFiN only
        gb_size_kb: int = 128,           # Eyeriss only
        pe_rows: int = 16,
        pe_cols: int = 128,
        scale_bandwidth: bool = True,  # Scale FMEM bandwidth with tile size
        base_tile_size: int = 128,     # Reference tile size for scaling
        progress_callback=None,
        verbose: bool = False
    ) -> List[ExperimentResult]:
        from architectures.thesis_arch import (
            get_tile_size_divisors, 
            get_min_tile_size_for_multilayer,
            get_valid_output_tile_sizes,
            get_cumulative_stride
        )
        
        network, fusion_level, variant = workload
        
        # Get workload to determine valid tile sizes
        shape, coupling, num_layers = self.workload_registry.get_workload(
            network, fusion_level, variant
        )
        
        # Auto-compute tile sizes if not provided
        if tile_sizes is None:
            if num_layers > 1:
                # For multi-layer fusion, use stride-aware valid tile computation
                # This finds output tiles that result in valid input tiles for all layers
                cum_pstride, cum_qstride = get_cumulative_stride(shape, num_layers)
                
                if cum_qstride > 1:
                    # Has strides - use stride-aware computation
                    tile_sizes = get_valid_output_tile_sizes(
                        shape, num_layers, pe_cols,
                        max_input_tile=pe_cols * 2  # Allow some oversize for input
                    )
                    print(f"[Stride-Aware] Cumulative stride: {cum_pstride}x{cum_qstride}")
                    print(f"[Stride-Aware] Valid output tile sizes: {tile_sizes}")
                else:
                    # No strides (all stride=1) - use old GCD method
                    min_common_tile = get_min_tile_size_for_multilayer(shape, num_layers)
                    q_size = shape.get('Q', pe_cols)
                    all_divisors = get_tile_size_divisors(q_size, pe_cols)
                    tile_sizes = [d for d in all_divisors if min_common_tile % d == 0 or d <= min_common_tile]
                    tile_sizes = [d for d in tile_sizes if q_size % d == 0]
            else:
                q_size = shape.get('Q', pe_cols)
                tile_sizes = get_tile_size_divisors(q_size, pe_cols)
            
            # Limit to reasonable number of tile sizes for efficiency
            if len(tile_sizes) > 10:
                # Sample: keep largest, smallest, and evenly spaced in between
                step = max(1, len(tile_sizes) // 10)
                tile_sizes = sorted(set(tile_sizes[::step] + [tile_sizes[0], tile_sizes[-1]]), reverse=True)
        
        total = len(tile_sizes)
        results = []
        
        print(f"\n{'='*70}")
        print(f"TILE SIZE SWEEP CASE STUDY")
        print(f"{'='*70}")
        print(f"Workload: {network}/{fusion_level}/{variant} ({num_layers} layers)")
        if arch_type == "eyeriss":
            print(f"Architecture: Eyeriss, GB={gb_size_kb}KB, PE={pe_rows}x{pe_cols}")
        else:
            print(f"Architecture: DepFiN, FMEM={fmem_size_kb}KB, WMEM={wmem_size_kb}KB, PE={pe_rows}x{pe_cols}")
        print(f"Bandwidth scaling: {'ENABLED' if scale_bandwidth else 'DISABLED'} (base={base_tile_size})")
        print(f"Tile sizes to test: {tile_sizes}")
        print(f"Total: {total} configurations")
        print(f"{'='*70}\n")
        
        for i, tile_size in enumerate(tile_sizes):
            config = ExperimentConfig(
                workload_name=network,
                workload_variant=variant,
                fusion_level=fusion_level,
                arch_type=arch_type,
                fmem_size_kB=fmem_size_kb,
                wmem_size_kB=wmem_size_kb,
                gb_size_kB=gb_size_kb,
                pe_rows=pe_rows,
                pe_cols=pe_cols,
                tile_size=tile_size,
                scale_bandwidth=scale_bandwidth,
                base_tile_size=base_tile_size,
                verbose=verbose,
            )
            
            utilization_pct = (tile_size / pe_cols) * 100
            print(f"[{i+1}/{total}] Tile size: {tile_size} (PE utilization: {utilization_pct:.1f}%)")
            
            if progress_callback:
                progress_callback(i, total, config)
            
            result = self.run_single_experiment(config)
            results.append(result)
            
            if result.success:
                print(f"  ✓ Energy: {result.energy_uJ:.3e} μJ, "
                      f"Latency: {result.latency_cycles:.3e} cc, "
                      f"EDP: {result.edp:.2e}, "
                      f"Util: {result.utilization:.1%}, "
                      f"DRAM_Read: {result.dram_reads}, DRAM_Write: {result.dram_writes}")
            else:
                print(f"  ✗ Failed: {result.error_message[:50]}...")
        
        # Print summary table
        print(f"\n{'='*70}")
        print("TILE SIZE SWEEP SUMMARY")
        print(f"{'='*70}")
        print(f"{'Tile':>6} {'PE Util':>8} {'Energy(μJ)':>12} {'Latency(cc)':>14} {'EDP':>12} {'Util':>8} {'DRAM R':>8} {'DRAM W':>8}")
        print(f"{'-'*70}")
        
        successful = [r for r in results if r.success]
        for r in successful:
            tile = r.config.tile_size
            pe_util = (tile / pe_cols) * 100
            print(f"{tile:>6} {pe_util:>7.1f}% {r.energy_uJ:>12.3e} {r.latency_cycles:>14.3e} {r.edp:>12.2e} {r.utilization:>7.1%} {r.dram_reads:>8} {r.dram_writes:>8}")
        
        if successful:
            best_edp = min(successful, key=lambda r: r.edp)
            print(f"\nBest EDP: tile_size={best_edp.config.tile_size}, EDP={best_edp.edp:.2e}")
        
        return results
    
    """
        Combined sweep over architecture parameters AND tile sizes.
        
        Creates a 2D grid: (memory sizes × PE configs) × tile sizes
        
        Args:
            workload: (network, fusion_level, variant) tuple
            arch_type: Architecture type ("depfin" or "eyeriss")
            fmem_sizes_kb: List of FMEM sizes to try [DepFiN only]
            wmem_sizes_kb: List of WMEM sizes to try [DepFiN only]
            gb_sizes_kb: List of GlobalBuffer sizes to try [Eyeriss only]
            pe_configs: List of (rows, cols) PE configurations
            tile_sizes: List of tile sizes (OUTPUT dimension, None = auto-compute)
            scale_bandwidth: Whether to scale FMEM bandwidth with tile size
            base_tile_size: Reference tile size for bandwidth scaling
            progress_callback: Optional callback for progress
            verbose: Enable verbose output
        
        Returns:
            List of ExperimentResult for all runs
    """    
    def run_combined_arch_tile_sweep(
        self,
        workload: Tuple[str, str, str],  # (network, fusion_level, variant)
        arch_type: str = "depfin",       # "depfin" or "eyeriss"
        fmem_sizes_kb: Optional[List[int]] = None,  # DepFiN only
        wmem_sizes_kb: Optional[List[int]] = None,  # DepFiN only
        gb_sizes_kb: Optional[List[int]] = None,    # Eyeriss only
        pe_configs: Optional[List[Tuple[int, int]]] = None,
        tile_sizes: Optional[List[int]] = None,
        scale_bandwidth: bool = True,
        base_tile_size: int = 128,
        progress_callback=None,
        verbose: bool = False
    ) -> List[ExperimentResult]:
        from architectures.thesis_arch import (
            get_tile_size_divisors, 
            get_min_tile_size_for_multilayer,
            get_valid_output_tile_sizes,
            get_cumulative_stride
        )
        
        network, fusion_level, variant = workload
        
        # Default sweep values based on architecture type
        if pe_configs is None:
            pe_configs = [(16, 128)]  # Default to single PE config for combined sweep
        
        if arch_type == "eyeriss":
            if gb_sizes_kb is None:
                gb_sizes_kb = [64, 128, 256]
        else:
            if fmem_sizes_kb is None:
                fmem_sizes_kb = [128, 512, 1024]
            if wmem_sizes_kb is None:
                wmem_sizes_kb = [256, 512, 1024]
        
        # Get workload to determine valid tile sizes
        shape, coupling, num_layers = self.workload_registry.get_workload(
            network, fusion_level, variant
        )
        
        # For combined sweep, we need consistent pe_cols for tile size auto-compute
        # Use the first PE config's columns as reference
        ref_pe_cols = pe_configs[0][1]
        
        # Auto-compute tile sizes if not provided
        if tile_sizes is None:
            if num_layers > 1:
                cum_pstride, cum_qstride = get_cumulative_stride(shape, num_layers)
                
                if cum_qstride > 1:
                    tile_sizes = get_valid_output_tile_sizes(
                        shape, num_layers, ref_pe_cols,
                        max_input_tile=ref_pe_cols * 2
                    )
                    print(f"[Stride-Aware] Cumulative stride: {cum_pstride}x{cum_qstride}")
                    print(f"[Stride-Aware] Valid output tile sizes: {tile_sizes}")
                else:
                    min_common_tile = get_min_tile_size_for_multilayer(shape, num_layers)
                    q_size = shape.get('Q', ref_pe_cols)
                    all_divisors = get_tile_size_divisors(q_size, ref_pe_cols)
                    tile_sizes = [d for d in all_divisors if min_common_tile % d == 0 or d <= min_common_tile]
                    tile_sizes = [d for d in tile_sizes if q_size % d == 0]
            else:
                q_size = shape.get('Q', ref_pe_cols)
                tile_sizes = get_tile_size_divisors(q_size, ref_pe_cols)
            
            # Limit to reasonable number
            if len(tile_sizes) > 8:
                step = max(1, len(tile_sizes) // 8)
                tile_sizes = sorted(set(tile_sizes[::step] + [tile_sizes[0], tile_sizes[-1]]), reverse=True)
        
        # Generate all combinations based on architecture type
        if arch_type == "eyeriss":
            arch_combinations = list(product(gb_sizes_kb, pe_configs))
            all_combinations = list(product(arch_combinations, tile_sizes))
            total = len(all_combinations)
            results = []
            
            print(f"\n{'='*80}")
            print(f"COMBINED ARCHITECTURE + TILE SIZE SWEEP (Eyeriss)")
            print(f"{'='*80}")
            print(f"Workload: {network}/{fusion_level}/{variant} ({num_layers} layers)")
            print(f"Architecture sweep:")
            print(f"  GB sizes: {gb_sizes_kb} KB")
            print(f"  PE configs: {[f'{r}x{c}' for r, c in pe_configs]}")
            print(f"Tile sizes: {tile_sizes}")
            print(f"Bandwidth scaling: {'ENABLED' if scale_bandwidth else 'DISABLED'} (base={base_tile_size})")
            print(f"Total: {len(arch_combinations)} arch configs × {len(tile_sizes)} tiles = {total} experiments")
            print(f"{'='*80}\n")
            
            for i, ((gb, (pe_r, pe_c)), tile_size) in enumerate(all_combinations):
                config = ExperimentConfig(
                    workload_name=network,
                    workload_variant=variant,
                    fusion_level=fusion_level,
                    arch_type="eyeriss",
                    gb_size_kB=gb,
                    pe_rows=pe_r,
                    pe_cols=pe_c,
                    tile_size=tile_size,
                    scale_bandwidth=scale_bandwidth,
                    base_tile_size=base_tile_size,
                    verbose=verbose,
                )
                
                utilization_pct = (tile_size / pe_c) * 100
                print(f"[{i+1}/{total}] GB={gb}KB, PE={pe_r}x{pe_c}, Tile={tile_size} ({utilization_pct:.0f}% util)")
                
                if progress_callback:
                    progress_callback(i, total, config)
                
                result = self.run_single_experiment(config)
                results.append(result)
                
                if result.success:
                    print(f"  ✓ Energy: {result.energy_uJ:.3e} μJ, "
                          f"Latency: {result.latency_cycles:.3e} cc, "
                          f"EDP: {result.edp:.2e}")
                else:
                    print(f"  ✗ Failed: {result.error_message[:50]}...")
            
            # Print summary table
            print(f"\n{'='*80}")
            print("COMBINED SWEEP SUMMARY (Eyeriss)")
            print(f"{'='*80}")
            print(f"{'GB':>6} {'PE':>8} {'Tile':>6} {'Energy(μJ)':>12} {'Latency':>14} {'EDP':>12}")
            print(f"{'-'*80}")
            
            successful = [r for r in results if r.success]
            for r in successful:
                print(f"{r.config.gb_size_kB:>6} "
                      f"{r.config.pe_rows}x{r.config.pe_cols:>3} {r.config.tile_size:>6} "
                      f"{r.energy_uJ:>12.3e} {r.latency_cycles:>14.3e} {r.edp:>12.2e}")
            
            if successful:
                best_edp = min(successful, key=lambda r: r.edp)
                print(f"\nBest EDP: GB={best_edp.config.gb_size_kB}KB, "
                      f"PE={best_edp.config.pe_rows}x{best_edp.config.pe_cols}, "
                      f"Tile={best_edp.config.tile_size}, EDP={best_edp.edp:.2e}")
        else:
            # DepFiN architecture
            arch_combinations = list(product(fmem_sizes_kb, wmem_sizes_kb, pe_configs))
            all_combinations = list(product(arch_combinations, tile_sizes))
            total = len(all_combinations)
            results = []
            
            print(f"\n{'='*80}")
            print(f"COMBINED ARCHITECTURE + TILE SIZE SWEEP (DepFiN)")
            print(f"{'='*80}")
            print(f"Workload: {network}/{fusion_level}/{variant} ({num_layers} layers)")
            print(f"Architecture sweep:")
            print(f"  FMEM sizes: {fmem_sizes_kb} KB")
            print(f"  WMEM sizes: {wmem_sizes_kb} KB")
            print(f"  PE configs: {[f'{r}x{c}' for r, c in pe_configs]}")
            print(f"Tile sizes: {tile_sizes}")
            print(f"Bandwidth scaling: {'ENABLED' if scale_bandwidth else 'DISABLED'} (base={base_tile_size})")
            print(f"Total: {len(arch_combinations)} arch configs × {len(tile_sizes)} tiles = {total} experiments")
            print(f"{'='*80}\n")
            
            for i, ((fmem, wmem, (pe_r, pe_c)), tile_size) in enumerate(all_combinations):
                config = ExperimentConfig(
                    workload_name=network,
                    workload_variant=variant,
                    fusion_level=fusion_level,
                    arch_type="depfin",
                    fmem_size_kB=fmem,
                    wmem_size_kB=wmem,
                    pe_rows=pe_r,
                    pe_cols=pe_c,
                    tile_size=tile_size,
                    scale_bandwidth=scale_bandwidth,
                    base_tile_size=base_tile_size,
                    verbose=verbose,
                )
                
                utilization_pct = (tile_size / pe_c) * 100
                print(f"[{i+1}/{total}] FMEM={fmem}KB, WMEM={wmem}KB, PE={pe_r}x{pe_c}, Tile={tile_size} ({utilization_pct:.0f}% util)")
                
                if progress_callback:
                    progress_callback(i, total, config)
                
                result = self.run_single_experiment(config)
                results.append(result)
                
                if result.success:
                    print(f"  ✓ Energy: {result.energy_uJ:.3e} μJ, "
                          f"Latency: {result.latency_cycles:.3e} cc, "
                          f"EDP: {result.edp:.2e}")
                else:
                    print(f"  ✗ Failed: {result.error_message[:50]}...")
            
            # Print summary table grouped by architecture
            print(f"\n{'='*80}")
            print("COMBINED SWEEP SUMMARY (DepFiN)")
            print(f"{'='*80}")
            print(f"{'FMEM':>6} {'WMEM':>6} {'PE':>8} {'Tile':>6} {'Energy(μJ)':>12} {'Latency':>14} {'EDP':>12}")
            print(f"{'-'*80}")
            
            successful = [r for r in results if r.success]
            for r in successful:
                print(f"{r.config.fmem_size_kB:>6} {r.config.wmem_size_kB:>6} "
                      f"{r.config.pe_rows}x{r.config.pe_cols:>3} {r.config.tile_size:>6} "
                      f"{r.energy_uJ:>12.3e} {r.latency_cycles:>14.3e} {r.edp:>12.2e}")
            
            if successful:
                best_edp = min(successful, key=lambda r: r.edp)
                print(f"\nBest EDP: FMEM={best_edp.config.fmem_size_kB}KB, "
                      f"WMEM={best_edp.config.wmem_size_kB}KB, "
                      f"PE={best_edp.config.pe_rows}x{best_edp.config.pe_cols}, "
                      f"Tile={best_edp.config.tile_size}, EDP={best_edp.edp:.2e}")
        
        return results
    
    # =========================================================================
    # EYERISS CASE STUDY SWEEPS
    # =========================================================================
    
    def _find_min_pe_rows_for_feasibility(
        self,
        workload: Tuple[str, str, str],
        config_template: ExperimentConfig,
        max_pe_rows: int = 1024,
        verbose: bool = False
    ) -> Tuple[int, Optional['ExperimentResult']]:
        """
        Binary search to find minimum PE rows that makes mapping feasible.
        
        Returns:
            (min_pe_rows, result) where result is the successful experiment or None
        """
        network, fusion_level, variant = workload
        
        # Try progressively larger PE row counts
        pe_row_candidates = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 84, 96, 128, 168, 256, 336, 512, 1024]
        pe_row_candidates = [r for r in pe_row_candidates if r <= max_pe_rows]
        
        for pe_rows in pe_row_candidates:
            config = ExperimentConfig(
                workload_name=network,
                workload_variant=variant,
                fusion_level=fusion_level,
                arch_type="eyeriss",
                gb_size_kB=config_template.gb_size_kB,
                pe_rows=pe_rows,
                pe_cols=config_template.pe_cols,
                input_reg_entries=config_template.input_reg_entries,
                weight_reg_entries=config_template.weight_reg_entries,
                intermediate_reg_entries=config_template.intermediate_reg_entries,
                output_reg_entries=config_template.output_reg_entries,
                tile_size=config_template.tile_size,
                verbose=verbose,
            )
            
            result = self.run_single_experiment(config)
            
            if result.success:
                return pe_rows, result
        
        return -1, None  # No feasible configuration found
    
    """
        Find minimum PE rows (at fixed pe_cols) that makes mapping feasible.
        
        This is the "rows-first" strategy where we keep pe_cols fixed (typically = tile_size)
        and sweep pe_rows to find the minimum. SARows handles M/Z distribution.
        
        Returns:
            (min_pe_rows, total_pes, result)
    """    
    def _find_min_pe_via_rows_sweep(
        self,
        workload: Tuple[str, str, str],
        config_template: ExperimentConfig,
        pe_cols: int,
        tile_size: int = 80,
        verbose: bool = False
    ) -> Tuple[int, int, Optional['ExperimentResult']]:
        network, fusion_level, variant = workload
        
        # PE rows candidates
        pe_row_candidates = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 84, 96, 128, 168, 256, 336, 512, 1024]
        
        for pe_rows in pe_row_candidates:
            config = ExperimentConfig(
                workload_name=network,
                workload_variant=variant,
                fusion_level=fusion_level,
                arch_type="eyeriss",
                gb_size_kB=config_template.gb_size_kB,
                pe_rows=pe_rows,
                pe_cols=pe_cols,
                input_reg_entries=config_template.input_reg_entries,
                weight_reg_entries=config_template.weight_reg_entries,
                intermediate_reg_entries=config_template.intermediate_reg_entries,
                output_reg_entries=config_template.output_reg_entries,
                tile_size=tile_size,
                verbose=verbose,
            )
            
            result = self.run_single_experiment(config)
            
            if result.success:
                total_pes = pe_rows * pe_cols
                return pe_rows, total_pes, result
        
        return -1, -1, None  # No feasible configuration found
    
    """
        Find minimum PE cols (at fixed pe_rows) that makes mapping feasible.
        
        This is the "cols-first" strategy where we keep pe_rows fixed (typically small)
        and sweep pe_cols. When pe_cols >= 2×tile_size, SACols Z optimization activates.
        
        Returns:
            (min_pe_cols, total_pes, result)
    """    
    def _find_min_pe_via_cols_sweep(
        self,
        workload: Tuple[str, str, str],
        config_template: ExperimentConfig,
        pe_rows: int,
        tile_size: int = 80,
        verbose: bool = False
    ) -> Tuple[int, int, Optional['ExperimentResult']]:
        network, fusion_level, variant = workload
        
        # PE column candidates: multiples of tile_size
        # Start from 2×tile_size (where SACols Z activates) up to higher values
        pe_cols_candidates = [2*tile_size, 3*tile_size, 4*tile_size, 6*tile_size, 8*tile_size]
        
        for pe_cols in pe_cols_candidates:
            config = ExperimentConfig(
                workload_name=network,
                workload_variant=variant,
                fusion_level=fusion_level,
                arch_type="eyeriss",
                gb_size_kB=config_template.gb_size_kB,
                pe_rows=pe_rows,
                pe_cols=pe_cols,
                input_reg_entries=config_template.input_reg_entries,
                weight_reg_entries=config_template.weight_reg_entries,
                intermediate_reg_entries=config_template.intermediate_reg_entries,
                output_reg_entries=config_template.output_reg_entries,
                tile_size=tile_size,
                verbose=verbose,
            )
            
            result = self.run_single_experiment(config)
            
            if result.success:
                total_pes = pe_rows * pe_cols
                return pe_cols, total_pes, result
        
        return -1, -1, None  # No feasible configuration found
    
    @staticmethod
    def _default_pe_grid(tile_size: int = 80) -> Tuple[List[int], List[int]]:
        """
        Return the default (pe_rows_candidates, pe_cols_candidates) for grid search.
        
        The grid is expanded to support both small workloads (FSRCNN) and large ones
        (ResNet18 with Z up to 512). With tile_size=1, pe_cols tile multiples are
        trivially small, so having a wide range of base values is important.
        
        The grid search skips configs exceeding max_total_pes anyway, so large
        candidate values don't cause infeasible-config overhead.
        """
        # Uniform grid for fair cross-workload comparison (Eyeriss Case Study 1)
        pe_rows_candidates = [4, 16, 32, 64, 128, 256, 512]

        # PE cols: same uniform grid (tile-size multiples no longer added)
        pe_cols_candidates = [4, 16, 32, 64, 128, 256, 512]
        
        return pe_rows_candidates, pe_cols_candidates
    
    def _find_min_pe_grid_search(
        self,
        workload: Tuple[str, str, str],
        config_template: ExperimentConfig,
        tile_size: int = 80,
        verbose: bool = False,
        pe_rows_candidates: List[int] = None,
        pe_cols_candidates: List[int] = None,
    ) -> Tuple[dict, dict]:
        """
        Comprehensive grid search to find:
        1. Config with minimum total PEs (regardless of latency)
        2. Config with minimum latency, then minimum PEs among equal-latency
        
        Args:
            pe_rows_candidates: Custom row values. If None, uses _default_pe_grid.
            pe_cols_candidates: Custom col values. If None, uses _default_pe_grid.
        
        Returns:
            (min_pes_result, min_latency_result) - each is a dict with:
            {'pe_rows', 'pe_cols', 'total_pes', 'sacols_z', 'result'} or None if not found
        """
        network, fusion_level, variant = workload
        
        # Use default grid if not specified
        if pe_rows_candidates is None or pe_cols_candidates is None:
            default_rows, default_cols = self._default_pe_grid(tile_size)
            if pe_rows_candidates is None:
                pe_rows_candidates = default_rows
            if pe_cols_candidates is None:
                pe_cols_candidates = default_cols
        
        # Collect ALL successful configurations
        successful_configs = []
        
        for pe_rows in pe_rows_candidates:
            for pe_cols in pe_cols_candidates:
                total_pes = pe_rows * pe_cols
                
                if total_pes > 17000:  # Skip configs that exceed max_total_pes
                    continue
                
                config = ExperimentConfig(
                    workload_name=network,
                    workload_variant=variant,
                    fusion_level=fusion_level,
                    arch_type="eyeriss",
                    gb_size_kB=config_template.gb_size_kB,
                    pe_rows=pe_rows,
                    pe_cols=pe_cols,
                    input_reg_entries=config_template.input_reg_entries,
                    weight_reg_entries=config_template.weight_reg_entries,
                    intermediate_reg_entries=config_template.intermediate_reg_entries,
                    output_reg_entries=config_template.output_reg_entries,
                    tile_size=tile_size,
                    verbose=verbose,
                )
                
                result = self.run_single_experiment(config)
                
                if result.success:
                    sacols_z = pe_cols >= 2 * tile_size
                    successful_configs.append({
                        'pe_rows': pe_rows,
                        'pe_cols': pe_cols,
                        'total_pes': total_pes,
                        'sacols_z': sacols_z,
                        'result': result,
                        'latency': result.latency_cycles,
                        'energy': result.energy_uJ,
                        'edp': result.edp,
                    })
        
        if not successful_configs:
            return None, None, []
        
        # 1. Find config with minimum total PEs
        min_pes_config = min(successful_configs, key=lambda x: x['total_pes'])
        
        # 2. Find config with minimum latency, then minimum PEs
        min_latency = min(c['latency'] for c in successful_configs)
        min_latency_configs = [c for c in successful_configs if c['latency'] == min_latency]
        min_latency_config = min(min_latency_configs, key=lambda x: x['total_pes'])
        
        # Sort all successful by total_pes ascending, then latency ascending
        all_sorted = sorted(successful_configs, key=lambda x: (x['total_pes'], x['latency']))
        
        return min_pes_config, min_latency_config, all_sorted


    """
        Case Study 1: Sweep WRegister sizes and find minimum PE configuration.
        
        Uses comprehensive grid search across all (rows × cols) combinations.
        For each WRegister size, finds the configuration with minimum total PEs
        that makes the mapping feasible.
    """    
    def run_wreg_pe_sweep(
        self,
        workload: Tuple[str, str, str],
        weight_reg_sizes: List[int],
        gb_size_kb: int = 128,
        tile_size: int = 80,
        input_reg_entries: int = None,
        intermediate_reg_entries: int = None,
        output_reg_entries: int = None,
        pe_rows_grid: List[int] = None,
        pe_cols_grid: List[int] = None,
        verbose: bool = False
    ) -> List['ExperimentResult']:
        # Use thesis_arch defaults if not specified
        if input_reg_entries is None:
            input_reg_entries = _EYERISS_DEFAULTS.input_reg_entries
        if intermediate_reg_entries is None:
            intermediate_reg_entries = _EYERISS_DEFAULTS.intermediate_out_reg_entries
        if output_reg_entries is None:
            output_reg_entries = _EYERISS_DEFAULTS.output_reg_entries
        network, fusion_level, variant = workload
        
        # Use default grid if not specified
        default_rows, default_cols = self._default_pe_grid(tile_size)
        pe_rows_list = pe_rows_grid if pe_rows_grid else default_rows
        pe_cols_list = pe_cols_grid if pe_cols_grid else default_cols
        
        print(f"\n{'='*100}")
        print("CASE STUDY 1: WRegister Size vs Minimum PE Configuration")
        print(f"{'='*100}")
        print(f"Workload: {network}/{fusion_level}/{variant}")
        print(f"GlobalBuffer: {gb_size_kb} KB, Tile: {tile_size}")
        print(f"Other registers: InReg={input_reg_entries}, IntReg={intermediate_reg_entries}, OutReg={output_reg_entries}")
        print(f"WRegister sizes to sweep: {weight_reg_sizes}")
        print(f"\nGrid search space:")
        print(f"  - PE rows: {pe_rows_list}")
        print(f"  - PE cols: {pe_cols_list}")
        print(f"  - Total combinations: {len(pe_rows_list) * len(pe_cols_list)}")
        print(f"{'='*100}\n")
        
        results = []
        summary_min_pes = []
        summary_min_latency = []
        summary_all_configs = []
        
        for wreg_size in weight_reg_sizes:
            print(f"\n[WReg={wreg_size}] Searching for optimal PE configurations...")
            
            config_template = ExperimentConfig(
                arch_type="eyeriss",
                gb_size_kB=gb_size_kb,
                weight_reg_entries=wreg_size,
                input_reg_entries=input_reg_entries,
                intermediate_reg_entries=intermediate_reg_entries,
                output_reg_entries=output_reg_entries,
                tile_size=tile_size,
            )
            
            # Comprehensive grid search - returns min_pes, min_latency, and all feasible configs
            min_pes_config, min_latency_config, all_configs = self._find_min_pe_grid_search(
                workload, config_template, tile_size=tile_size, verbose=verbose,
                pe_rows_candidates=pe_rows_list, pe_cols_candidates=pe_cols_list,
            )
            
            # Collect all feasible configs for Summary (C)
            summary_all_configs.append({
                'wreg_size': wreg_size,
                'configs': all_configs,
            })
            
            if min_pes_config:
                results.append(min_pes_config['result'])
                summary_min_pes.append({
                    'wreg_size': wreg_size,
                    **min_pes_config
                })
                z_str = "Yes" if min_pes_config['sacols_z'] else "No"
                print(f"  ✓ Min PEs config: {min_pes_config['pe_rows']}×{min_pes_config['pe_cols']} = {min_pes_config['total_pes']} PEs (SACols Z: {z_str})")
                print(f"    Energy: {min_pes_config['energy']:.3e} μJ, Latency: {min_pes_config['latency']:.3e} cc")
            else:
                print(f"  ✗ No feasible configuration found")
                summary_min_pes.append({
                    'wreg_size': wreg_size,
                    'pe_rows': -1, 'pe_cols': -1, 'total_pes': -1, 'sacols_z': False,
                    'energy': None, 'latency': None, 'edp': None,
                })
            
            if min_latency_config:
                summary_min_latency.append({
                    'wreg_size': wreg_size,
                    **min_latency_config
                })
                z_str = "Yes" if min_latency_config['sacols_z'] else "No"
                print(f"  ✓ Min Latency config: {min_latency_config['pe_rows']}×{min_latency_config['pe_cols']} = {min_latency_config['total_pes']} PEs (SACols Z: {z_str})")
                print(f"    Energy: {min_latency_config['energy']:.3e} μJ, Latency: {min_latency_config['latency']:.3e} cc")
            else:
                summary_min_latency.append({
                    'wreg_size': wreg_size,
                    'pe_rows': -1, 'pe_cols': -1, 'total_pes': -1, 'sacols_z': False,
                    'energy': None, 'latency': None, 'edp': None,
                })
        
        # Print summary table - MIN PEs
        print(f"\n{'='*120}")
        print("CASE STUDY 1 SUMMARY (A): WRegister Size vs MINIMUM PEs Configuration")
        print("(Finds config with fewest total PEs, regardless of latency)")
        print(f"{'='*120}")
        print(f"{'WReg':>10} {'PE Rows':>10} {'PE Cols':>10} {'Total PEs':>12} {'SACols Z':>10} {'Energy(μJ)':>14} {'Latency(cc)':>14} {'EDP':>14}")
        print(f"{'-'*120}")
        
        for d in summary_min_pes:
            if d['total_pes'] > 0:
                z_str = "Yes" if d['sacols_z'] else "No"
                print(f"{d['wreg_size']:>10} {d['pe_rows']:>10} {d['pe_cols']:>10} {d['total_pes']:>12} "
                      f"{z_str:>10} {d['energy']:>14.3e} {d['latency']:>14.3e} {d['edp']:>14.2e}")
            else:
                print(f"{d['wreg_size']:>10} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>14} {'N/A':>14} {'N/A':>14}")
        
        # Print summary table - MIN LATENCY
        print(f"\n{'='*120}")
        print("CASE STUDY 1 SUMMARY (B): WRegister Size vs MINIMUM LATENCY Configuration")
        print("(Finds config with best latency, then minimum PEs among equal-latency configs)")
        print(f"{'='*120}")
        print(f"{'WReg':>10} {'PE Rows':>10} {'PE Cols':>10} {'Total PEs':>12} {'SACols Z':>10} {'Energy(μJ)':>14} {'Latency(cc)':>14} {'EDP':>14}")
        print(f"{'-'*120}")
        
        for d in summary_min_latency:
            if d['total_pes'] > 0:
                z_str = "Yes" if d['sacols_z'] else "No"
                print(f"{d['wreg_size']:>10} {d['pe_rows']:>10} {d['pe_cols']:>10} {d['total_pes']:>12} "
                      f"{z_str:>10} {d['energy']:>14.3e} {d['latency']:>14.3e} {d['edp']:>14.2e}")
            else:
                print(f"{d['wreg_size']:>10} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>14} {'N/A':>14} {'N/A':>14}")
        
        # Print summary table - ALL FEASIBLE CONFIGS
        print(f"\n{'='*120}")
        print("CASE STUDY 1 SUMMARY (C): ALL Feasible Configurations per WReg Size")
        print("(Every PE config that produced a valid mapping, sorted by total PEs)")
        print(f"{'='*120}")
        for entry in summary_all_configs:
            wreg_size = entry['wreg_size']
            configs = entry['configs']
            print(f"\n  WReg={wreg_size}: {len(configs)} feasible config(s)")
            if configs:
                print(f"  {'PE Config':>12} {'Total PEs':>12} {'SACols Z':>10} {'Energy(μJ)':>14} {'Latency(cc)':>14} {'EDP':>14}")
                print(f"  {'-'*80}")
                for c in configs:
                    z_str = "Yes" if c['sacols_z'] else "No"
                    print(f"  {c['pe_rows']}×{c['pe_cols']:>5} {c['total_pes']:>12,} {z_str:>10} "
                          f"{c['energy']:>14.3e} {c['latency']:>14.3e} {c['edp']:>14.2e}")
        
        return results
    
    def run_intreg_pe_sweep(
        self,
        workload: Tuple[str, str, str],
        intermediate_reg_sizes: List[int],
        gb_size_kb: int = 128,
        tile_size: int = 80,
        input_reg_entries: int = None,
        weight_reg_entries: int = None,
        output_reg_entries: int = None,
        pe_rows_grid: List[int] = None,
        pe_cols_grid: List[int] = None,
        verbose: bool = False
    ) -> List['ExperimentResult']:
        # Use thesis_arch defaults if not specified
        if input_reg_entries is None:
            input_reg_entries = _EYERISS_DEFAULTS.input_reg_entries
        if weight_reg_entries is None:
            weight_reg_entries = _EYERISS_DEFAULTS.weight_reg_entries
        if output_reg_entries is None:
            output_reg_entries = _EYERISS_DEFAULTS.output_reg_entries
        """
        Case Study 2: Sweep IntermediateRegister sizes and find minimum PE configuration.
        
        Uses comprehensive grid search across all (rows × cols) combinations.
        Finds the configuration with minimum total PEs that makes mapping feasible.
        """
        network, fusion_level, variant = workload
        
        # Use default grid if not specified
        default_rows, default_cols = self._default_pe_grid(tile_size)
        pe_rows_list = pe_rows_grid if pe_rows_grid else default_rows
        pe_cols_list = pe_cols_grid if pe_cols_grid else default_cols
        
        print(f"\n{'='*100}")
        print("CASE STUDY 2: IntermediateRegister Size vs Minimum PE Configuration")
        print(f"{'='*100}")
        print(f"Workload: {network}/{fusion_level}/{variant}")
        print(f"GlobalBuffer: {gb_size_kb} KB, Tile: {tile_size}")
        print(f"Other registers: InReg={input_reg_entries}, WReg={weight_reg_entries}, OutReg={output_reg_entries}")
        print(f"IntermediateRegister sizes to sweep: {intermediate_reg_sizes}")
        print(f"\nGrid search space:")
        print(f"  - PE rows: {pe_rows_list}")
        print(f"  - PE cols: {pe_cols_list}")
        print(f"  - Total combinations: {len(pe_rows_list) * len(pe_cols_list)}")
        print(f"{'='*100}\n")
        
        results = []
        summary_min_pes = []
        summary_min_latency = []
        summary_all_configs = []
        
        for intreg_size in intermediate_reg_sizes:
            print(f"\n[IntReg={intreg_size}] Searching for optimal PE configurations...")
            
            config_template = ExperimentConfig(
                arch_type="eyeriss",
                gb_size_kB=gb_size_kb,
                intermediate_reg_entries=intreg_size,
                input_reg_entries=input_reg_entries,
                weight_reg_entries=weight_reg_entries,
                output_reg_entries=output_reg_entries,
                tile_size=tile_size,
            )
            
            # Comprehensive grid search - returns min_pes, min_latency, and all configs
            min_pes_config, min_latency_config, all_configs = self._find_min_pe_grid_search(
                workload, config_template, tile_size=tile_size, verbose=verbose,
                pe_rows_candidates=pe_rows_list, pe_cols_candidates=pe_cols_list,
            )
            
            # Collect all feasible configs for Summary (C)
            summary_all_configs.append({
                'intreg_size': intreg_size,
                'configs': all_configs,
            })
            
            if min_pes_config:
                results.append(min_pes_config['result'])
                summary_min_pes.append({
                    'intreg_size': intreg_size,
                    **min_pes_config
                })
                z_str = "Yes" if min_pes_config['sacols_z'] else "No"
                print(f"  ✓ Min PEs config: {min_pes_config['pe_rows']}×{min_pes_config['pe_cols']} = {min_pes_config['total_pes']} PEs (SACols Z: {z_str})")
                print(f"    Energy: {min_pes_config['energy']:.3e} μJ, Latency: {min_pes_config['latency']:.3e} cc")
            else:
                print(f"  ✗ No feasible configuration found")
                summary_min_pes.append({
                    'intreg_size': intreg_size,
                    'pe_rows': -1, 'pe_cols': -1, 'total_pes': -1, 'sacols_z': False,
                    'energy': None, 'latency': None, 'edp': None,
                })
            
            if min_latency_config:
                summary_min_latency.append({
                    'intreg_size': intreg_size,
                    **min_latency_config
                })
                z_str = "Yes" if min_latency_config['sacols_z'] else "No"
                print(f"  ✓ Min Latency config: {min_latency_config['pe_rows']}×{min_latency_config['pe_cols']} = {min_latency_config['total_pes']} PEs (SACols Z: {z_str})")
                print(f"    Energy: {min_latency_config['energy']:.3e} μJ, Latency: {min_latency_config['latency']:.3e} cc")
            else:
                summary_min_latency.append({
                    'intreg_size': intreg_size,
                    'pe_rows': -1, 'pe_cols': -1, 'total_pes': -1, 'sacols_z': False,
                    'energy': None, 'latency': None, 'edp': None,
                })
        
        # Print summary table - MIN PEs
        print(f"\n{'='*120}")
        print("CASE STUDY 2 SUMMARY (A): IntermediateRegister Size vs MINIMUM PEs Configuration")
        print("(Finds config with fewest total PEs, regardless of latency)")
        print(f"{'='*120}")
        print(f"{'IntReg':>10} {'PE Rows':>10} {'PE Cols':>10} {'Total PEs':>12} {'SACols Z':>10} {'Energy(μJ)':>14} {'Latency(cc)':>14} {'EDP':>14}")
        print(f"{'-'*120}")
        
        for d in summary_min_pes:
            if d['total_pes'] > 0:
                z_str = "Yes" if d['sacols_z'] else "No"
                print(f"{d['intreg_size']:>10} {d['pe_rows']:>10} {d['pe_cols']:>10} {d['total_pes']:>12} "
                      f"{z_str:>10} {d['energy']:>14.3e} {d['latency']:>14.3e} {d['edp']:>14.2e}")
            else:
                print(f"{d['intreg_size']:>10} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>14} {'N/A':>14} {'N/A':>14}")
        
        # Print summary table - MIN LATENCY
        print(f"\n{'='*120}")
        print("CASE STUDY 2 SUMMARY (B): IntermediateRegister Size vs MINIMUM LATENCY Configuration")
        print("(Finds config with best latency, then minimum PEs among equal-latency configs)")
        print(f"{'='*120}")
        print(f"{'IntReg':>10} {'PE Rows':>10} {'PE Cols':>10} {'Total PEs':>12} {'SACols Z':>10} {'Energy(μJ)':>14} {'Latency(cc)':>14} {'EDP':>14}")
        print(f"{'-'*120}")
        
        for d in summary_min_latency:
            if d['total_pes'] > 0:
                z_str = "Yes" if d['sacols_z'] else "No"
                print(f"{d['intreg_size']:>10} {d['pe_rows']:>10} {d['pe_cols']:>10} {d['total_pes']:>12} "
                      f"{z_str:>10} {d['energy']:>14.3e} {d['latency']:>14.3e} {d['edp']:>14.2e}")
            else:
                print(f"{d['intreg_size']:>10} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>14} {'N/A':>14} {'N/A':>14}")
        
        # Print summary table - ALL FEASIBLE CONFIGS
        print(f"\n{'='*120}")
        print("CASE STUDY 2 SUMMARY (C): ALL Feasible Configurations per IntReg Size")
        print("(Every PE config that produced a valid mapping, sorted by total PEs)")
        print(f"{'='*120}")
        for entry in summary_all_configs:
            intreg_size = entry['intreg_size']
            configs = entry['configs']
            print(f"\n  IntReg={intreg_size}: {len(configs)} feasible config(s)")
            if configs:
                print(f"  {'PE Config':>12} {'Total PEs':>12} {'SACols Z':>10} {'Energy(μJ)':>14} {'Latency(cc)':>14} {'EDP':>14}")
                print(f"  {'-'*80}")
                for c in configs:
                    z_str = "Yes" if c['sacols_z'] else "No"
                    print(f"  {c['pe_rows']}×{c['pe_cols']:>5} {c['total_pes']:>12,} {z_str:>10} "
                          f"{c['energy']:>14.3e} {c['latency']:>14.3e} {c['edp']:>14.2e}")
        
        return results
    
    def run_outreg_pe_sweep(
        self,
        workload: Tuple[str, str, str],
        output_reg_sizes: List[int],
        gb_size_kb: int = 128,
        tile_size: int = 80,
        input_reg_entries: int = None,
        weight_reg_entries: int = None,
        intermediate_reg_entries: int = None,
        pe_rows_grid: List[int] = None,
        pe_cols_grid: List[int] = None,
        verbose: bool = False
    ) -> List['ExperimentResult']:
        # Use thesis_arch defaults if not specified
        if input_reg_entries is None:
            input_reg_entries = _EYERISS_DEFAULTS.input_reg_entries
        if weight_reg_entries is None:
            weight_reg_entries = _EYERISS_DEFAULTS.weight_reg_entries
        if intermediate_reg_entries is None:
            intermediate_reg_entries = _EYERISS_DEFAULTS.intermediate_out_reg_entries
        """
        Case Study 3: Sweep OutRegister sizes and find minimum PE configuration.
        
        Uses comprehensive grid search across all (rows × cols) combinations.
        For each OutRegister size, finds the configuration with minimum total PEs
        that makes the mapping feasible.
        """
        network, fusion_level, variant = workload
        
        # Use default grid if not specified
        default_rows, default_cols = self._default_pe_grid(tile_size)
        pe_rows_list = pe_rows_grid if pe_rows_grid else default_rows
        pe_cols_list = pe_cols_grid if pe_cols_grid else default_cols
        
        print(f"\n{'='*100}")
        print("CASE STUDY 3: OutRegister Size vs Minimum PE Configuration")
        print(f"{'='*100}")
        print(f"Workload: {network}/{fusion_level}/{variant}")
        print(f"GlobalBuffer: {gb_size_kb} KB, Tile: {tile_size}")
        print(f"Other registers: InReg={input_reg_entries}, WReg={weight_reg_entries}, IntReg={intermediate_reg_entries}")
        print(f"OutRegister sizes to sweep: {output_reg_sizes}")
        print(f"\nGrid search space:")
        print(f"  - PE rows: {pe_rows_list}")
        print(f"  - PE cols: {pe_cols_list}")
        print(f"  - Total combinations: {len(pe_rows_list) * len(pe_cols_list)}")
        print(f"{'='*100}\n")
        
        results = []
        summary_min_pes = []
        summary_min_latency = []
        summary_all_configs = []
        
        for outreg_size in output_reg_sizes:
            print(f"\n[OutReg={outreg_size}] Searching for optimal PE configurations...")
            
            config_template = ExperimentConfig(
                arch_type="eyeriss",
                gb_size_kB=gb_size_kb,
                output_reg_entries=outreg_size,
                input_reg_entries=input_reg_entries,
                weight_reg_entries=weight_reg_entries,
                intermediate_reg_entries=intermediate_reg_entries,
                tile_size=tile_size,
            )
            
            # Comprehensive grid search - returns min_pes, min_latency, and all configs
            min_pes_config, min_latency_config, all_configs = self._find_min_pe_grid_search(
                workload, config_template, tile_size=tile_size, verbose=verbose,
                pe_rows_candidates=pe_rows_list, pe_cols_candidates=pe_cols_list,
            )
            
            # Collect all feasible configs for Summary (C)
            summary_all_configs.append({
                'outreg_size': outreg_size,
                'configs': all_configs,
            })
            
            if min_pes_config:
                results.append(min_pes_config['result'])
                summary_min_pes.append({
                    'outreg_size': outreg_size,
                    **min_pes_config
                })
                z_str = "Yes" if min_pes_config['sacols_z'] else "No"
                print(f"  ✓ Min PEs config: {min_pes_config['pe_rows']}×{min_pes_config['pe_cols']} = {min_pes_config['total_pes']} PEs (SACols Z: {z_str})")
                print(f"    Energy: {min_pes_config['energy']:.3e} μJ, Latency: {min_pes_config['latency']:.3e} cc")
            else:
                print(f"  ✗ No feasible configuration found")
                summary_min_pes.append({
                    'outreg_size': outreg_size,
                    'pe_rows': -1, 'pe_cols': -1, 'total_pes': -1, 'sacols_z': False,
                    'energy': None, 'latency': None, 'edp': None,
                })
            
            if min_latency_config:
                summary_min_latency.append({
                    'outreg_size': outreg_size,
                    **min_latency_config
                })
                z_str = "Yes" if min_latency_config['sacols_z'] else "No"
                print(f"  ✓ Min Latency config: {min_latency_config['pe_rows']}×{min_latency_config['pe_cols']} = {min_latency_config['total_pes']} PEs (SACols Z: {z_str})")
                print(f"    Energy: {min_latency_config['energy']:.3e} μJ, Latency: {min_latency_config['latency']:.3e} cc")
            else:
                summary_min_latency.append({
                    'outreg_size': outreg_size,
                    'pe_rows': -1, 'pe_cols': -1, 'total_pes': -1, 'sacols_z': False,
                    'energy': None, 'latency': None, 'edp': None,
                })
        
        # Print summary table - MIN PEs
        print(f"\n{'='*120}")
        print("CASE STUDY 3 SUMMARY (A): OutRegister Size vs MINIMUM PEs Configuration")
        print("(Finds config with fewest total PEs, regardless of latency)")
        print(f"{'='*120}")
        print(f"{'OutReg':>10} {'PE Rows':>10} {'PE Cols':>10} {'Total PEs':>12} {'SACols Z':>10} {'Energy(μJ)':>14} {'Latency(cc)':>14} {'EDP':>14}")
        print(f"{'-'*120}")
        
        for d in summary_min_pes:
            if d['total_pes'] > 0:
                z_str = "Yes" if d['sacols_z'] else "No"
                print(f"{d['outreg_size']:>10} {d['pe_rows']:>10} {d['pe_cols']:>10} {d['total_pes']:>12} "
                      f"{z_str:>10} {d['energy']:>14.3e} {d['latency']:>14.3e} {d['edp']:>14.2e}")
            else:
                print(f"{d['outreg_size']:>10} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>14} {'N/A':>14} {'N/A':>14}")
        
        # Print summary table - MIN LATENCY
        print(f"\n{'='*120}")
        print("CASE STUDY 3 SUMMARY (B): OutRegister Size vs MINIMUM LATENCY Configuration")
        print("(Finds config with best latency, then minimum PEs among equal-latency configs)")
        print(f"{'='*120}")
        print(f"{'OutReg':>10} {'PE Rows':>10} {'PE Cols':>10} {'Total PEs':>12} {'SACols Z':>10} {'Energy(μJ)':>14} {'Latency(cc)':>14} {'EDP':>14}")
        print(f"{'-'*120}")
        
        for d in summary_min_latency:
            if d['total_pes'] > 0:
                z_str = "Yes" if d['sacols_z'] else "No"
                print(f"{d['outreg_size']:>10} {d['pe_rows']:>10} {d['pe_cols']:>10} {d['total_pes']:>12} "
                      f"{z_str:>10} {d['energy']:>14.3e} {d['latency']:>14.3e} {d['edp']:>14.2e}")
            else:
                print(f"{d['outreg_size']:>10} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>10} {'N/A':>14} {'N/A':>14} {'N/A':>14}")
        
        # Print summary table - ALL FEASIBLE CONFIGS
        print(f"\n{'='*120}")
        print("CASE STUDY 3 SUMMARY (C): ALL Feasible Configurations per OutReg Size")
        print("(Every PE config that produced a valid mapping, sorted by total PEs)")
        print(f"{'='*120}")
        for entry in summary_all_configs:
            outreg_size = entry['outreg_size']
            configs = entry['configs']
            print(f"\n  OutReg={outreg_size}: {len(configs)} feasible config(s)")
            if configs:
                print(f"  {'PE Config':>12} {'Total PEs':>12} {'SACols Z':>10} {'Energy(μJ)':>14} {'Latency(cc)':>14} {'EDP':>14}")
                print(f"  {'-'*80}")
                for c in configs:
                    z_str = "Yes" if c['sacols_z'] else "No"
                    print(f"  {c['pe_rows']}×{c['pe_cols']:>5} {c['total_pes']:>12,} {z_str:>10} "
                          f"{c['energy']:>14.3e} {c['latency']:>14.3e} {c['edp']:>14.2e}")
        
        return results
    
    def run_gb_tile_sweep(
        self,
        workload: Tuple[str, str, str],
        gb_sizes_kb: List[int],
        pe_rows: int = 84,
        pe_cols: int = 24,
        input_reg_entries: int = 256,
        weight_reg_entries: int = 4000,
        intermediate_reg_entries: int = 1000,
        output_reg_entries: int = 64,
        verbose: bool = False
    ) -> List['ExperimentResult']:
        """
        Case Study 4: GlobalBuffer size vs max feasible tile size.
        
        For each GlobalBuffer size, find the maximum tile size that fits,
        then compare performance.
        """
        network, fusion_level, variant = workload
        
        # Get workload shape to determine valid tile sizes
        shape, coupling, num_layers = self.workload_registry.get_workload(network, fusion_level, variant)
        q_size = shape.get('Q', 120)
        
        # Get all valid tile sizes (divisors of Q that are <= pe_cols for reasonable utilization)
        from architectures.thesis_arch import get_tile_size_divisors
        all_tile_sizes = get_tile_size_divisors(q_size, pe_cols * 4)  # Allow up to 4x pe_cols
        all_tile_sizes = sorted([t for t in all_tile_sizes if t <= q_size], reverse=True)
        
        print(f"\n{'='*80}")
        print("CASE STUDY 4: GlobalBuffer Size vs Max Feasible Tile Size")
        print(f"{'='*80}")
        print(f"Workload: {network}/{fusion_level}/{variant}")
        print(f"PE array: {pe_rows}x{pe_cols}, Q shape: {q_size}")
        print(f"Registers: InReg={input_reg_entries}, WReg={weight_reg_entries}, IntReg={intermediate_reg_entries}, OutReg={output_reg_entries}")
        print(f"GlobalBuffer sizes to sweep: {gb_sizes_kb} KB")
        print(f"Candidate tile sizes: {all_tile_sizes}")
        print(f"{'='*80}\n")
        
        results = []
        summary_data = []
        
        for gb_size in gb_sizes_kb:
            print(f"\n[GB={gb_size}KB] Finding max feasible tile size...")
            
            max_tile = None
            best_result = None
            
            # Try tile sizes from largest to smallest
            for tile_size in all_tile_sizes:
                config = ExperimentConfig(
                    workload_name=network,
                    workload_variant=variant,
                    fusion_level=fusion_level,
                    arch_type="eyeriss",
                    gb_size_kB=gb_size,
                    pe_rows=pe_rows,
                    pe_cols=pe_cols,
                    input_reg_entries=input_reg_entries,
                    weight_reg_entries=weight_reg_entries,
                    intermediate_reg_entries=intermediate_reg_entries,
                    output_reg_entries=output_reg_entries,
                    tile_size=tile_size,
                    verbose=verbose,
                )
                
                result = self.run_single_experiment(config)
                
                if result.success:
                    max_tile = tile_size
                    best_result = result
                    break  # Found the largest feasible tile
            
            if best_result and best_result.success:
                results.append(best_result)
                summary_data.append({
                    'gb_size': gb_size,
                    'max_tile': max_tile,
                    'energy': best_result.energy_uJ,
                    'latency': best_result.latency_cycles,
                    'edp': best_result.edp,
                })
                print(f"  ✓ Max tile: {max_tile}")
                print(f"    Energy: {best_result.energy_uJ:.3e} μJ, Latency: {best_result.latency_cycles:.3e} cc, EDP: {best_result.edp:.2e}")
            else:
                print(f"  ✗ No feasible tile size found")
        
        # Print summary table
        print(f"\n{'='*80}")
        print("CASE STUDY 4 SUMMARY: GlobalBuffer Size vs Max Feasible Tile Size")
        print(f"{'='*80}")
        print(f"{'GB Size(KB)':>12} {'Max Tile':>10} {'Energy(μJ)':>14} {'Latency':>14} {'EDP':>14}")
        print(f"{'-'*80}")
        
        for d in summary_data:
            print(f"{d['gb_size']:>12} {d['max_tile']:>10} "
                  f"{d['energy']:>14.3e} {d['latency']:>14.3e} {d['edp']:>14.2e}")
        
        return results
    
    def run_pe_aspect_sweep(
        self,
        workload: Tuple[str, str, str],
        total_pes: int = 168,
        pe_aspect_ratios: Optional[List[Tuple[int, int]]] = None,
        gb_size_kb: int = 128,
        tile_size: int = 80,
        input_reg_entries: int = 256,
        weight_reg_entries: int = 4000,
        intermediate_reg_entries: int = 1000,
        output_reg_entries: int = 64,
        verbose: bool = False
    ) -> List['ExperimentResult']:
        """
        Case Study 5: PE array aspect ratio with fixed total PEs.
        
        Keep total PEs constant, vary aspect ratio (rows x cols),
        and compare how it affects parallelization and energy.
        """
        network, fusion_level, variant = workload
        
        # Generate aspect ratios if not provided
        if pe_aspect_ratios is None:
            # Find all factor pairs of total_pes
            pe_aspect_ratios = []
            for rows in range(1, int(total_pes**0.5) + 1):
                if total_pes % rows == 0:
                    cols = total_pes // rows
                    pe_aspect_ratios.append((rows, cols))
                    if rows != cols:
                        pe_aspect_ratios.append((cols, rows))
            pe_aspect_ratios = sorted(pe_aspect_ratios, key=lambda x: x[0])
        
        # Validate all ratios have correct total
        for rows, cols in pe_aspect_ratios:
            if rows * cols != total_pes:
                print(f"Warning: {rows}x{cols}={rows*cols} != {total_pes}, skipping")
                pe_aspect_ratios.remove((rows, cols))
        
        print(f"\n{'='*80}")
        print("CASE STUDY 5: PE Aspect Ratio with Fixed Total PEs")
        print(f"{'='*80}")
        print(f"Workload: {network}/{fusion_level}/{variant}")
        print(f"Total PEs: {total_pes}, GlobalBuffer: {gb_size_kb} KB, Tile: {tile_size}")
        print(f"Registers: InReg={input_reg_entries}, WReg={weight_reg_entries}, IntReg={intermediate_reg_entries}, OutReg={output_reg_entries}")
        print(f"Aspect ratios to sweep: {[f'{r}x{c}' for r, c in pe_aspect_ratios]}")
        print(f"{'='*80}\n")
        
        results = []
        summary_data = []
        
        for pe_rows, pe_cols in pe_aspect_ratios:
            print(f"\n[PE={pe_rows}x{pe_cols}] Running experiment...")
            
            config = ExperimentConfig(
                workload_name=network,
                workload_variant=variant,
                fusion_level=fusion_level,
                arch_type="eyeriss",
                gb_size_kB=gb_size_kb,
                pe_rows=pe_rows,
                pe_cols=pe_cols,
                input_reg_entries=input_reg_entries,
                weight_reg_entries=weight_reg_entries,
                intermediate_reg_entries=intermediate_reg_entries,
                output_reg_entries=output_reg_entries,
                tile_size=tile_size,
                verbose=verbose,
            )
            
            result = self.run_single_experiment(config)
            
            if result.success:
                results.append(result)
                summary_data.append({
                    'pe_rows': pe_rows,
                    'pe_cols': pe_cols,
                    'aspect': f"{pe_rows}x{pe_cols}",
                    'energy': result.energy_uJ,
                    'latency': result.latency_cycles,
                    'edp': result.edp,
                    'utilization': result.utilization,
                })
                print(f"  ✓ Energy: {result.energy_uJ:.3e} μJ, Latency: {result.latency_cycles:.3e} cc, EDP: {result.edp:.2e}")
            else:
                print(f"  ✗ Failed: {result.error_message[:50]}...")
        
        # Print summary table
        print(f"\n{'='*80}")
        print("CASE STUDY 5 SUMMARY: PE Aspect Ratio (Fixed Total PEs)")
        print(f"{'='*80}")
        print(f"{'Aspect':>10} {'Rows':>6} {'Cols':>6} {'Energy(μJ)':>14} {'Latency':>14} {'EDP':>14} {'Util':>8}")
        print(f"{'-'*80}")
        
        for d in summary_data:
            print(f"{d['aspect']:>10} {d['pe_rows']:>6} {d['pe_cols']:>6} "
                  f"{d['energy']:>14.3e} {d['latency']:>14.3e} {d['edp']:>14.2e} "
                  f"{d['utilization']:>7.2%}")
        
        if summary_data:
            best_edp = min(summary_data, key=lambda d: d['edp'])
            print(f"\nBest EDP: {best_edp['aspect']}, EDP={best_edp['edp']:.2e}")
        
        return results

    # =========================================================================
    # FUSION COMPARISON CASE STUDIES
    # =========================================================================
    
    @staticmethod
    def compute_min_wmem_bytes(shape: Shape, num_layers: int, precision_bits: int = 8) -> int:
        """
        Compute minimum Weight Memory size needed to hold all weights for a workload.
        
        For single layer: M × C × R × S × bytes_per_param
        For N-layer fusion: Σ_i (Z_i × C_i × R_i × S_i) × bytes_per_param
        
        This is the total number of weight parameters across all fused layers,
        which must all fit simultaneously in WMEM during fused execution.
        
        Args:
            shape: Workload Shape (single-layer dims M/C/R/S, multi-layer Z_i/C_i/R_i/S_i)
            num_layers: Number of fused layers
            precision_bits: Weight precision in bits (default: 8)
        
        Returns:
            Minimum WMEM size in bytes
        """
        bytes_per_param = max(1, precision_bits // 8)
        total_params = 0
        
        if num_layers == 1:
            # Single layer: weight dims are M, C, R, S
            m = shape.get('M', 1)
            c = shape.get('C', 1)
            r = shape.get('R', 1)
            s = shape.get('S', 1)
            total_params = m * c * r * s
        else:
            # Multi-layer: weight dims for each layer are Z_i, C_i, R_i, S_i
            for layer in range(num_layers):
                z = shape.get(f'Z{layer}', 1)
                c = shape.get(f'C{layer}', 1)
                r = shape.get(f'R{layer}', 1)
                s = shape.get(f'S{layer}', 1)
                total_params += z * c * r * s
        
        return total_params * bytes_per_param
    
    @staticmethod
    def compute_min_wmem_kb(shape: Shape, num_layers: int, precision_bits: int = 8,
                            min_kb: int = 1) -> int:
        """
        Compute minimum WMEM in KB (rounded up), with a minimum floor.
        
        Args:
            shape: Workload Shape
            num_layers: Number of fused layers
            precision_bits: Weight precision in bits
            min_kb: Minimum WMEM size in KB (floor)
        
        Returns:
            Minimum WMEM size in KB (ceiling-rounded)
        """
        min_bytes = ExperimentRunner.compute_min_wmem_bytes(shape, num_layers, precision_bits)
        min_kb_computed = math.ceil(min_bytes / 1024)
        return max(min_kb_computed, min_kb)
    
    # Per-network configuration for fusion comparison:
    # - full: (fusion_level, variant) for full fusion
    # - intermediate: fusion_level whose variants cover all fused layers without overlap
    # - single_exclude: single-layer variants NOT part of full fusion (FC, projections)
    FUSION_COMPARISON_CONFIG = {
        'fsrcnn': {
            'full': ('full', '8layer'),
            'intermediate': '3layer',
            'single_exclude': set(),  # All 8 singles are in full fusion
        },
        'mccnn': {
            'full': ('full', '4layer'),
            'intermediate': '2layer',
            'single_exclude': set(),  # All 4 singles are in full fusion
        },
        'vgg16': {
            'full': ('full', '13layer'),
            'intermediate': 'block',
            'single_exclude': {'L13', 'L14', 'L15'},  # FC layers not in full fusion
        },
        'resnet18': {
            'full': ('full', '17layer'),
            'intermediate': '2layer',
            'single_exclude': {'L7_conv3_proj', 'L12_conv4_proj', 'L17_conv5_proj', 'L20_fc'},
        },
    }

    def find_min_feasible_wreg(
        self,
        network: str,
        fusion_level: str,
        variant: str,
        pe_rows: int,
        pe_cols: int,
        gb_size_kb: int = 128,
        input_reg_entries: int = 24,
        intermediate_reg_entries: int = 32,
        output_reg_entries: int = 32,
        tile_size: Optional[int] = None,
        wreg_min: int = 1,
        wreg_max: int = 50000,
    ) -> Optional[int]:
        """
        Binary search for the minimum WReg size that makes an Eyeriss mapping feasible.
        
        Returns the minimum feasible WReg, or None if even wreg_max fails.
        """
        shape, coupling, nlayers = self.workload_registry.get_workload(network, fusion_level, variant)
        
        # Save results length so we can discard binary-search probe results
        results_before = len(self.results)
        
        # First check if wreg_max works at all
        config = ExperimentConfig(
            workload_name=network, workload_variant=variant, fusion_level=fusion_level,
            num_fused_layers=nlayers, arch_type="eyeriss",
            gb_size_kB=gb_size_kb, pe_rows=pe_rows, pe_cols=pe_cols,
            input_reg_entries=input_reg_entries, weight_reg_entries=wreg_max,
            intermediate_reg_entries=intermediate_reg_entries,
            output_reg_entries=output_reg_entries, tile_size=tile_size,
        )
        result = self.run_single_experiment(config)
        if not result.success:
            # Discard probe results
            del self.results[results_before:]
            return None
        
        # Binary search: find minimum wreg where mapping succeeds
        lo, hi = wreg_min, wreg_max
        best = wreg_max
        while lo <= hi:
            mid = (lo + hi) // 2
            config = ExperimentConfig(
                workload_name=network, workload_variant=variant, fusion_level=fusion_level,
                num_fused_layers=nlayers, arch_type="eyeriss",
                gb_size_kB=gb_size_kb, pe_rows=pe_rows, pe_cols=pe_cols,
                input_reg_entries=input_reg_entries, weight_reg_entries=mid,
                intermediate_reg_entries=intermediate_reg_entries,
                output_reg_entries=output_reg_entries, tile_size=tile_size,
            )
            result = self.run_single_experiment(config)
            if result.success:
                best = mid
                hi = mid - 1
            else:
                lo = mid + 1
        # Discard all binary-search probe results from self.results
        del self.results[results_before:]
        return best

    def run_fusion_comparison(
        self,
        network: str,
        arch_type: str = "depfin",
        # DepFiN parameters
        fmem_size_kb: int = 1056,
        wmem_size_kb: int = 524,
        # Eyeriss parameters (for full fusion)
        gb_size_kb: int = 128,
        input_reg_entries: int = None,
        weight_reg_entries: int = None,
        intermediate_reg_entries: int = None,
        output_reg_entries: int = None,
        # Eyeriss parameters for SINGLES (CS1)
        # Default: same as full fusion (same chip, only WReg auto-sized)
        single_gb_size_kb: int = None,
        single_input_reg: int = None,
        single_weight_reg: int = None,
        single_output_reg: int = None,
        # Eyeriss parameters for INTERMEDIATE (CS2)
        # Default: same as full fusion sizes
        intermediate_gb_size_kb: int = None,
        intermediate_input_reg: int = None,
        intermediate_weight_reg: int = None,
        intermediate_intermediate_reg: int = None,
        intermediate_output_reg: int = None,
        # Shared
        pe_rows: int = 16,
        pe_cols: int = 128,
        tile_size: Optional[int] = None,
        scale_bandwidth: bool = False,
        base_tile_size: int = 128,
        dram_energy_override: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Fusion Comparison Case Studies for a given network.
        
        Runs three experiment groups and compares them:
          1. FULL FUSION: single experiment with all layers fused
          2. CS1 - NON-FUSED (singles): sum of all individual layer experiments
          3. CS2 - INTERMEDIATE FUSION: sum of block/2-3layer fused segments
        
        For CS1, only the single layers that are part of the full fusion are summed
        
        For CS2, the intermediate fusion level covers all fused layers without overlap:
          - FSRCNN: 3layer → L0_L1_L2(3) + L3_L4_L5(3) + L6_L7(2) = 8
          - MC-CNN: 2layer → L0_L1(2) + L2_L3(2) = 4
          - VGG16:  block  → blk1(2) + blk2(2) + blk3(3) + blk4(3) + blk5(3) = 13
          - ResNet18: 2layer → s1b1(2)+s1b2(3)+s2b1(2)+s2b2(2)+s3b1(2)+s3b2(2)+s4b1(2)+s4b2(2) = 17
        
        All experiments use the same PE config for fair comparison.
        Tile size is auto-detected per fusion level unless overridden.
        
        Args:
            network: Network name (fsrcnn, mccnn, vgg16, resnet18)
            arch_type: "depfin" or "eyeriss"
            ... architecture parameters ...
            tile_size: Output tile size override (None = auto per level)
        
        Returns:
            Dict with 'full', 'singles', 'intermediate' results and comparison metrics
        """
        if network not in self.FUSION_COMPARISON_CONFIG:
            raise ValueError(f"Unknown network: {network}. Available: {list(self.FUSION_COMPARISON_CONFIG.keys())}")
        
        cfg = self.FUSION_COMPARISON_CONFIG[network]
        full_fusion, full_variant = cfg['full']
        intermediate_level = cfg['intermediate']
        single_exclude = cfg['single_exclude']
        
        # === Resolve FULL FUSION Eyeriss register defaults ===
        if input_reg_entries is None:
            input_reg_entries = _EYERISS_DEFAULTS.input_reg_entries
        if weight_reg_entries is None:
            weight_reg_entries = _EYERISS_DEFAULTS.weight_reg_entries
        if intermediate_reg_entries is None:
            intermediate_reg_entries = _EYERISS_DEFAULTS.intermediate_out_reg_entries
        if output_reg_entries is None:
            output_reg_entries = _EYERISS_DEFAULTS.output_reg_entries
        
        # === Resolve SINGLE Eyeriss defaults (same chip as full fusion) ===
        # Only WReg differs (auto-sized via binary search); all other registers
        # match full fusion to model the same physical chip.
        if single_gb_size_kb is None:
            single_gb_size_kb = gb_size_kb
        if single_input_reg is None:
            single_input_reg = input_reg_entries
        if single_weight_reg is None:
            single_weight_reg = weight_reg_entries  # fallback; overridden by binary search
        if single_output_reg is None:
            single_output_reg = output_reg_entries
        
        # === Resolve INTERMEDIATE Eyeriss defaults (same as full fusion) ===
        if intermediate_gb_size_kb is None:
            intermediate_gb_size_kb = gb_size_kb
        if intermediate_input_reg is None:
            intermediate_input_reg = input_reg_entries
        if intermediate_weight_reg is None:
            intermediate_weight_reg = weight_reg_entries
        if intermediate_intermediate_reg is None:
            intermediate_intermediate_reg = intermediate_reg_entries
        if intermediate_output_reg is None:
            intermediate_output_reg = output_reg_entries
        
        print(f"\n{'='*90}")
        print(f"FUSION COMPARISON CASE STUDIES: {network.upper()}")
        print(f"{'='*90}")
        if arch_type == "eyeriss":
            print(f"Architecture: Eyeriss, PE={pe_rows}x{pe_cols}")
            print(f"  Full fusion:    GB={gb_size_kb}KB, WReg={weight_reg_entries}, "
                  f"InReg={input_reg_entries}, IntReg={intermediate_reg_entries}, "
                  f"OutReg={output_reg_entries}")
            print(f"  Singles (CS1):  GB={single_gb_size_kb}KB, WReg=auto (binary search min feasible, shared), "
                  f"InReg={single_input_reg}, OutReg={single_output_reg}")
            print(f"  Interm. (CS2):  GB={intermediate_gb_size_kb}KB, WReg=auto (binary search min feasible, shared), "
                  f"InReg={intermediate_input_reg}, IntReg={intermediate_intermediate_reg}, "
                  f"OutReg={intermediate_output_reg}")
        else:
            print(f"Architecture: DepFiN, FMEM={fmem_size_kb}KB, PE={pe_rows}x{pe_cols}")
            print(f"WMEM: shared per case study — max(min_wmem) across all variants")
            print(f"  Full fusion WMEM={wmem_size_kb}KB (from --wmem-size, used as fallback)")
        if tile_size is not None:
            print(f"Tile size override: {tile_size}")
        else:
            print(f"Tile size: auto-detected per fusion level")
        if dram_energy_override is not None:
            print(f"⚡ DRAM energy override: {dram_energy_override} pJ/byte (default: 32 pJ/byte)")
        print(f"{'='*90}\n")
        
        # Helper to create config with per-variant WMEM and Eyeriss sizing
        def make_config(wk_name, fusion_level, variant, num_layers, shape,
                        variant_category="full", wmem_override_kb=None,
                        tile_size_override=None):
            """Create ExperimentConfig with per-variant memory sizing.
            
            variant_category: "full" | "single" | "intermediate"
            wmem_override_kb: if set, use this WMEM instead of auto-computing.
              Used for CS1/CS2 where all variants share one architecture
              (WMEM = max across all variants in that case study).
            tile_size_override: if set, use this tile_size instead of the
              global one (used for stride-scaled partial segments).
            """
            # WMEM sizing for DepFiN
            if arch_type == "depfin":
                if wmem_override_kb is not None:
                    variant_wmem_kb = wmem_override_kb
                else:
                    variant_wmem_kb = self.compute_min_wmem_kb(shape, num_layers)
            else:
                variant_wmem_kb = wmem_size_kb  # Eyeriss doesn't use WMEM
            
            # Select per-variant Eyeriss register/GB sizes
            if variant_category == "single":
                v_gb     = single_gb_size_kb
                v_inreg  = single_input_reg
                v_wreg   = single_weight_reg
                v_intreg = intermediate_reg_entries  # singles don't use IntReg
                v_outreg = single_output_reg
            elif variant_category == "intermediate":
                v_gb     = intermediate_gb_size_kb
                v_inreg  = intermediate_input_reg
                v_wreg   = intermediate_weight_reg
                v_intreg = intermediate_intermediate_reg
                v_outreg = intermediate_output_reg
            else:  # "full"
                v_gb     = gb_size_kb
                v_inreg  = input_reg_entries
                v_wreg   = weight_reg_entries
                v_intreg = intermediate_reg_entries
                v_outreg = output_reg_entries
            
            effective_tile = (tile_size_override
                              if tile_size_override is not None
                              else tile_size)
            return ExperimentConfig(
                workload_name=wk_name,
                workload_variant=variant,
                fusion_level=fusion_level,
                num_fused_layers=num_layers,
                arch_type=arch_type,
                fmem_size_kB=fmem_size_kb,
                wmem_size_kB=variant_wmem_kb,
                gb_size_kB=v_gb,
                pe_rows=pe_rows,
                pe_cols=pe_cols,
                input_reg_entries=v_inreg,
                weight_reg_entries=v_wreg,
                intermediate_reg_entries=v_intreg,
                output_reg_entries=v_outreg,
                tile_size=effective_tile,
                scale_bandwidth=scale_bandwidth,
                base_tile_size=base_tile_size,
                dram_energy_override=dram_energy_override,
                verbose=verbose,
            ), variant_wmem_kb
        
        # -----------------------------------------------------------------
        # 1. FULL FUSION
        # -----------------------------------------------------------------
        print(f"--- FULL FUSION: {network}/{full_fusion}/{full_variant} ---")
        shape_full, coupling_full, nlayers_full = self.workload_registry.get_workload(
            network, full_fusion, full_variant
        )
        config_full, wmem_full = make_config(network, full_fusion, full_variant, nlayers_full, shape_full, "full")
        if arch_type == "depfin":
            print(f"  [WMEM auto-sized: {wmem_full} KB for {nlayers_full} fused layers]")
        result_full = self.run_single_experiment(config_full)
        
        # Energy, Latency, EDP, and MOPs for full fusion
        if result_full.success:
            print(f"  ✓ Energy: {result_full.energy_uJ:.3e} μJ, "
                  f"Latency: {result_full.latency_cycles:.3e} cc, "
                  f"EDP: {result_full.edp:.2e}, MOPs: {result_full.mops}")
        else:
            print(f"  ✗ FAILED: {result_full.error_message}")
            return {'full': result_full, 'error': 'Full fusion failed'}
        
        # -----------------------------------------------------------------
        # 2. CS1: NON-FUSED (sum of single layers)
        # -----------------------------------------------------------------
        print(f"\n--- CS1: NON-FUSED (single layers) ---")
        if arch_type == "eyeriss":
            print(f"  [Base config: GB={single_gb_size_kb}KB, WReg=auto, "
                  f"InReg={single_input_reg}, OutReg={single_output_reg}]")
        single_variants = self.workload_registry.list_workloads(network, 'single')
        single_results = []
        
        # Pre-compute WMEM for CS1: one architecture for ALL singles
        # → WMEM = max(min_wmem across all single layers)
        if arch_type == "depfin":
            wmem_per_single = {}
            for _, _, sv in single_variants:
                if sv in single_exclude:
                    continue
                sv_shape, _, sv_nl = self.workload_registry.get_workload(network, 'single', sv)
                wmem_per_single[sv] = self.compute_min_wmem_kb(sv_shape, sv_nl)
            cs1_wmem_kb = max(wmem_per_single.values()) if wmem_per_single else wmem_size_kb
            cs1_wmem_kb += 1
            print(f"  [CS1 shared WMEM = {cs1_wmem_kb} KB "
                  f"(max of {len(wmem_per_single)} layers: "
                  f"{', '.join(f'{v}={w}KB' for v, w in wmem_per_single.items())})]")
        elif arch_type == "eyeriss":
            # Binary search min feasible WReg per single, take max → shared CS1 WReg
            wreg_per_single = {}
            print(f"  [Binary search for min feasible WReg per single layer...]")
            for _, _, sv in single_variants:
                if sv in single_exclude:
                    continue
                min_wreg = self.find_min_feasible_wreg(
                    network, 'single', sv,
                    pe_rows=pe_rows, pe_cols=pe_cols,
                    gb_size_kb=single_gb_size_kb,
                    input_reg_entries=single_input_reg,
                    intermediate_reg_entries=intermediate_reg_entries,  # not used for singles
                    output_reg_entries=single_output_reg,
                    tile_size=tile_size,
                )
                if min_wreg is not None:
                    wreg_per_single[sv] = min_wreg
                else:
                    print(f"    ✗ {sv}: no feasible WReg found (even max=50000 fails)")
                    wreg_per_single[sv] = None
            feasible_wregs = [w for w in wreg_per_single.values() if w is not None]
            cs1_wreg = max(feasible_wregs) if feasible_wregs else single_weight_reg
            single_weight_reg = cs1_wreg  # Override for all singles
            print(f"  [CS1 shared WReg = {cs1_wreg} "
                  f"(max of {len(wreg_per_single)} layers: "
                  f"{', '.join(f'{v}={w}' for v, w in wreg_per_single.items())})]")
            cs1_wmem_kb = None
        else:
            cs1_wmem_kb = None  # not used for Eyeriss
        
        for _, _, variant in single_variants:
            if variant in single_exclude:
                print(f"  [skip] {variant} (not in full fusion)")
                continue
            
            shape_v, _, nlayers = self.workload_registry.get_workload(network, 'single', variant)
            config, variant_wmem = make_config(network, 'single', variant, nlayers, shape_v, "single",
                                               wmem_override_kb=cs1_wmem_kb)
            if arch_type == "depfin":
                print(f"  [WMEM={variant_wmem}KB for {variant}]")
            elif arch_type == "eyeriss":
                print(f"  [WReg={single_weight_reg} for {variant}]")
            result = self.run_single_experiment(config)
            single_results.append((variant, result))
            
            if result.success:
                print(f"  ✓ {variant}: E={result.energy_uJ:.3e} μJ, "
                      f"L={result.latency_cycles:.3e} cc, MOPs={result.mops}")
            else:
                print(f"  ✗ {variant}: FAILED - {result.error_message[:50]}")
        
        # -----------------------------------------------------------------
        # 3. CS2: INTERMEDIATE FUSION (sum of fused segments)
        # -----------------------------------------------------------------
        print(f"\n--- CS2: INTERMEDIATE FUSION ({intermediate_level}) ---")
        if arch_type == "eyeriss":
            print(f"  [Base config: GB={intermediate_gb_size_kb}KB, WReg=auto, "
                  f"InReg={intermediate_input_reg}, IntReg={intermediate_intermediate_reg}, "
                  f"OutReg={intermediate_output_reg}]")
        inter_variants = self.workload_registry.list_workloads(network, intermediate_level)
        inter_results = []
        
        # Pre-compute WMEM for CS2: one architecture for ALL intermediate segments
        # → WMEM = max(min_wmem across all segments)
        if arch_type == "depfin":
            wmem_per_inter = {}
            for _, _, iv in inter_variants:
                iv_shape, _, iv_nl = self.workload_registry.get_workload(network, intermediate_level, iv)
                wmem_per_inter[iv] = self.compute_min_wmem_kb(iv_shape, iv_nl)
            cs2_wmem_kb = max(wmem_per_inter.values()) if wmem_per_inter else wmem_size_kb
            cs2_wmem_kb += 1
            print(f"  [CS2 shared WMEM = {cs2_wmem_kb} KB "
                  f"(max of {len(wmem_per_inter)} segments: "
                  f"{', '.join(f'{v}={w}KB' for v, w in wmem_per_inter.items())})]")
        elif arch_type == "eyeriss":
            # Binary search min feasible WReg per segment, take max → shared CS2 WReg
            wreg_per_inter = {}
            print(f"  [Binary search for min feasible WReg per intermediate segment...]")
            for _, _, iv in inter_variants:
                min_wreg = self.find_min_feasible_wreg(
                    network, intermediate_level, iv,
                    pe_rows=pe_rows, pe_cols=pe_cols,
                    gb_size_kb=intermediate_gb_size_kb,
                    input_reg_entries=intermediate_input_reg,
                    intermediate_reg_entries=intermediate_intermediate_reg,
                    output_reg_entries=intermediate_output_reg,
                    tile_size=tile_size,
                )
                if min_wreg is not None:
                    wreg_per_inter[iv] = min_wreg
                else:
                    print(f"    ✗ {iv}: no feasible WReg found (even max=50000 fails)")
                    wreg_per_inter[iv] = None
            feasible_wregs = [w for w in wreg_per_inter.values() if w is not None]
            cs2_wreg = max(feasible_wregs) if feasible_wregs else intermediate_weight_reg
            intermediate_weight_reg = cs2_wreg  # Override for all intermediates
            print(f"  [CS2 shared WReg = {cs2_wreg} "
                  f"(max of {len(wreg_per_inter)} segments: "
                  f"{', '.join(f'{v}={w}' for v, w in wreg_per_inter.items())})]")
            cs2_wmem_kb = None
        else:
            cs2_wmem_kb = None  # not used for Eyeriss
        
        # Pre-compute per-block tile sizes from full fusion stride propagation
        _block_tiles_cs2 = {}
        if tile_size is not None:
            from architectures.thesis_arch import calculate_per_layer_tile_sizes
            full_tile_sizes = calculate_per_layer_tile_sizes(
                shape_full, nlayers_full, tile_size, tile_size)
            cumul = 0
            for _, _, iv in inter_variants:
                _, _, nl = self.workload_registry.get_workload(
                    network, intermediate_level, iv)
                cumul += nl
                _, block_tile_w = full_tile_sizes[cumul]
                # Only clamp to pe_cols for DepFiN (tile maps to SA columns);
                # Eyeriss tiles the Q dimension freely.
                if arch_type == "depfin":
                    _block_tiles_cs2[iv] = min(block_tile_w, pe_cols)
                else:
                    _block_tiles_cs2[iv] = block_tile_w
            print(f"  Per-block tile sizes (from full fusion stride propagation):")
            for iv, bt in _block_tiles_cs2.items():
                print(f"    {iv}: tile={bt}")

        for _, _, variant in inter_variants:
            shape_v, _, nlayers = self.workload_registry.get_workload(network, intermediate_level, variant)
            bt = _block_tiles_cs2.get(variant)  # stride-derived tile (None if no tile_size)
            config, variant_wmem = make_config(network, intermediate_level, variant, nlayers, shape_v, "intermediate",
                                               wmem_override_kb=cs2_wmem_kb,
                                               tile_size_override=bt)
            if arch_type == "depfin":
                print(f"  [WMEM={variant_wmem}KB for {variant} ({nlayers}L, tile={bt})]")
            elif arch_type == "eyeriss":
                print(f"  [WReg={intermediate_weight_reg} for {variant} ({nlayers}L, tile={bt})]")
            result = self.run_single_experiment(config)
            inter_results.append((variant, result))
            
            if result.success:
                print(f"  ✓ {variant} ({nlayers}L): E={result.energy_uJ:.3e} μJ, "
                      f"L={result.latency_cycles:.3e} cc, MOPs={result.mops}")
            else:
                print(f"  ✗ {variant}: FAILED - {result.error_message[:50]}")
        
        # -----------------------------------------------------------------
        # AGGREGATE & COMPARE
        # -----------------------------------------------------------------
        def aggregate(results_list):
            """Sum metrics across all successful results."""
            ok = [(v, r) for v, r in results_list if r.success]
            fail = [(v, r) for v, r in results_list if not r.success]
            if not ok:
                return None
            total_energy = sum(r.energy_uJ for _, r in ok)
            total_latency = sum(r.latency_cycles for _, r in ok)
            # EDP = E_total × L_total (in J·cc, matching engine units).
            # Summing individual EDPs would drop cross-terms (E_i × L_j, i≠j).
            total_edp = (total_energy * 1e-6) * total_latency
            total_dram_reads = sum(r.dram_reads for _, r in ok)
            total_dram_writes = sum(r.dram_writes for _, r in ok)
            return {
                'energy_uJ': total_energy,
                'latency_cycles': total_latency,
                'edp': total_edp,
                'dram_reads': total_dram_reads,
                'dram_writes': total_dram_writes,
                'num_experiments': len(ok),
                'num_failed': len(fail),
            }
        
        agg_singles = aggregate(single_results)
        agg_inter = aggregate(inter_results)
        
        full_data = {
            'energy_uJ': result_full.energy_uJ,
            'latency_cycles': result_full.latency_cycles,
            'edp': result_full.edp,
            'dram_reads': result_full.dram_reads,
            'dram_writes': result_full.dram_writes,
        }
        
        # Print comparison table
        print(f"\n{'='*100}")
        print(f"FUSION COMPARISON RESULTS: {network.upper()} ({arch_type.upper()})")
        print(f"{'='*100}")
        print(f"{'Level':<25} {'Energy (μJ)':>14} {'Latency (cc)':>14} {'EDP':>14} {'DRAM Reads':>16} {'DRAM Writes':>16}")
        print(f"{'-'*100}")
        
        print(f"{'Full Fusion':<25} {full_data['energy_uJ']:>14.3e} "
              f"{full_data['latency_cycles']:>14.3e} {full_data['edp']:>14.2e} "
              f"{full_data['dram_reads']:>16,} {full_data['dram_writes']:>16,}")
        
        if agg_singles:
            print(f"{'Σ Singles (CS1)':<25} {agg_singles['energy_uJ']:>14.3e} "
                  f"{agg_singles['latency_cycles']:>14.3e} {agg_singles['edp']:>14.2e} "
                  f"{agg_singles['dram_reads']:>16,} {agg_singles['dram_writes']:>16,}")
        
        if agg_inter:
            print(f"{'Σ Intermediate (CS2)':<25} {agg_inter['energy_uJ']:>14.3e} "
                  f"{agg_inter['latency_cycles']:>14.3e} {agg_inter['edp']:>14.2e} "
                  f"{agg_inter['dram_reads']:>16,} {agg_inter['dram_writes']:>16,}")
        
        # Ratios (full / non-fused) — values < 1 mean fusion is better
        print(f"\n{'─'*100}")
        print(f"RATIOS (Full Fusion / Baseline) — values < 1.0 mean fusion wins")
        print(f"{'─'*100}")
        print(f"{'Comparison':<35} {'Energy':>10} {'Latency':>10} {'EDP':>14} {'DRAM Rds':>10} {'DRAM Wrs':>10}")
        print(f"{'-'*100}")
        
        if agg_singles:
            e_ratio = full_data['energy_uJ'] / agg_singles['energy_uJ']
            l_ratio = full_data['latency_cycles'] / agg_singles['latency_cycles']
            edp_ratio = full_data['edp'] / agg_singles['edp']
            dr_ratio = full_data['dram_reads'] / agg_singles['dram_reads'] if agg_singles['dram_reads'] > 0 else float('inf')
            dw_ratio = full_data['dram_writes'] / agg_singles['dram_writes'] if agg_singles['dram_writes'] > 0 else float('inf')
            print(f"{'CS1: Full / Σ Singles':<35} {e_ratio:>10.4f} {l_ratio:>10.4f} "
                  f"{edp_ratio:>14.4f} {dr_ratio:>10.4f} {dw_ratio:>10.4f}")
            
            e_saving = (1 - e_ratio) * 100
            l_saving = (1 - l_ratio) * 100
            edp_saving = (1 - edp_ratio) * 100
            dr_saving = (1 - dr_ratio) * 100
            dw_saving = (1 - dw_ratio) * 100
            print(f"{'    → Savings (%)':<35} {e_saving:>+9.1f}% {l_saving:>+9.1f}% "
                  f"{edp_saving:>+13.1f}% {dr_saving:>+9.1f}% {dw_saving:>+9.1f}%")
        
        if agg_inter:
            e_ratio = full_data['energy_uJ'] / agg_inter['energy_uJ']
            l_ratio = full_data['latency_cycles'] / agg_inter['latency_cycles']
            edp_ratio = full_data['edp'] / agg_inter['edp']
            dr_ratio = full_data['dram_reads'] / agg_inter['dram_reads'] if agg_inter['dram_reads'] > 0 else float('inf')
            dw_ratio = full_data['dram_writes'] / agg_inter['dram_writes'] if agg_inter['dram_writes'] > 0 else float('inf')
            print(f"{'CS2: Full / Σ Intermediate':<35} {e_ratio:>10.4f} {l_ratio:>10.4f} "
                  f"{edp_ratio:>14.4f} {dr_ratio:>10.4f} {dw_ratio:>10.4f}")
            
            e_saving = (1 - e_ratio) * 100
            l_saving = (1 - l_ratio) * 100
            edp_saving = (1 - edp_ratio) * 100
            dr_saving = (1 - dr_ratio) * 100
            dw_saving = (1 - dw_ratio) * 100
            print(f"{'    → Savings (%)':<35} {e_saving:>+9.1f}% {l_saving:>+9.1f}% "
                  f"{edp_saving:>+13.1f}% {dr_saving:>+9.1f}% {dw_saving:>+9.1f}%")
        
        if agg_singles and agg_inter:
            print(f"\n{'─'*90}")
            print(f"CS1 vs CS2: Intermediate fusion captures how much of full fusion benefit?")
            e_cs1 = agg_singles['energy_uJ'] - full_data['energy_uJ']  # Savings from full vs singles
            e_cs2 = agg_singles['energy_uJ'] - agg_inter['energy_uJ']  # Savings from intermediate vs singles
            if e_cs1 > 0:
                e_captured = (e_cs2 / e_cs1) * 100
                print(f"  Energy: intermediate captures {e_captured:.1f}% of full fusion benefit")
            l_cs1 = agg_singles['latency_cycles'] - full_data['latency_cycles']
            l_cs2 = agg_singles['latency_cycles'] - agg_inter['latency_cycles']
            if l_cs1 > 0:
                l_captured = (l_cs2 / l_cs1) * 100
                print(f"  Latency: intermediate captures {l_captured:.1f}% of full fusion benefit")
        
        print(f"{'='*100}\n")
        
        return {
            'network': network,
            'arch_type': arch_type,
            'full': full_data,
            'singles': agg_singles,
            'intermediate': agg_inter,
            'single_results': single_results,
            'inter_results': inter_results,
            'full_result': result_full,
        }


    # ------------------------------------------------------------------
    # FIXED-CONFIG FUSION COMPARISON  (--compare-full-vs-single / --compare-partial-vs-single)
    # ------------------------------------------------------------------
    def run_fixed_fusion_comparison(
        self,
        network: str,
        mode: str = "full_vs_single",
        arch_type: str = "eyeriss",
        # DepFiN parameters
        fmem_size_kb: int = 1056,
        wmem_size_kb: int = 524,
        # Eyeriss parameters
        gb_size_kb: int = 128,
        pe_rows: int = 128,
        pe_cols: int = 128,
        input_reg_entries: int = None,
        weight_reg_entries: int = None,
        intermediate_reg_entries: int = None,
        output_reg_entries: int = None,
        tile_size: Optional[int] = None,
        intermediate_level_override: Optional[str] = None,
        dram_energy_override: Optional[float] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Fixed-config fusion comparison for Eyeriss or DepFiN.

        Uses the EXACT SAME architecture for both the fused and non-fused
        experiments.  No binary search / auto-sizing.

        Three modes:
          full_vs_single:    Full fusion vs sum of single layers
          partial_vs_single: Sum of partial (intermediate) fusion vs sum of single layers
          full_vs_partial:   Full fusion vs sum of partial (intermediate) fusion



        Example invocations (from CLI):
          python3 experiment_runner.py --compare-full-vs-single -w resnet18 \\
              --arch-type eyeriss --pe-rows 128 --pe-cols 128 --weight-reg 902 \\
              --gb-size 128 --tile-size 1 --input-reg 400 --intermediate-reg 300 --output-reg 64

          python3 experiment_runner.py --compare-partial-vs-single -w resnet18 \\
              --arch-type eyeriss --pe-rows 128 --pe-cols 128 --weight-reg 384 \\
              --gb-size 128 --tile-size 1 --input-reg 400 --intermediate-reg 300 --output-reg 64
              

          python3 experiment_runner.py --compare-full-vs-partial -w resnet18 \\
              --arch-type eyeriss --pe-rows 128 --pe-cols 128 --weight-reg 920 \\
              --gb-size 128 --tile-size 1 --input-reg 400 --intermediate-reg 300 --output-reg 64

              # Default (2layer for resnet18):
              python3 experiment_runner.py --compare-full-vs-partial -w resnet18 ...

              # Override to block:
              python3 experiment_runner.py --compare-full-vs-partial -w resnet18 --intermediate-level block ...

        """
        if network not in self.FUSION_COMPARISON_CONFIG:
            raise ValueError(f"Unknown network: {network}. "
                             f"Available: {list(self.FUSION_COMPARISON_CONFIG.keys())}")

        cfg = self.FUSION_COMPARISON_CONFIG[network]
        full_fusion, full_variant = cfg['full']
        intermediate_level = intermediate_level_override if intermediate_level_override else cfg['intermediate']
        single_exclude = cfg['single_exclude']

        if input_reg_entries is None:
            input_reg_entries = _EYERISS_DEFAULTS.input_reg_entries
        if weight_reg_entries is None:
            weight_reg_entries = _EYERISS_DEFAULTS.weight_reg_entries
        if intermediate_reg_entries is None:
            intermediate_reg_entries = _EYERISS_DEFAULTS.intermediate_out_reg_entries
        if output_reg_entries is None:
            output_reg_entries = _EYERISS_DEFAULTS.output_reg_entries

        if mode == "full_vs_single":
            mode_label = "FULL FUSION vs Sum SINGLES"
        elif mode == "partial_vs_single":
            mode_label = "Sum PARTIAL FUSION vs Sum SINGLES"
        else:  # full_vs_partial
            mode_label = "FULL FUSION vs Sum PARTIAL FUSION"

        print(f"\n{'='*100}")
        print(f"FIXED-CONFIG FUSION COMPARISON: {network.upper()} -- {mode_label}")
        print(f"{'='*100}")
        if arch_type == "eyeriss":
            print(f"Architecture: Eyeriss, PE={pe_rows}x{pe_cols} ({pe_rows*pe_cols:,} total)")
            print(f"  GB={gb_size_kb}KB, WReg={weight_reg_entries}, InReg={input_reg_entries}, "
                  f"IntReg={intermediate_reg_entries}, OutReg={output_reg_entries}")
        else:
            print(f"Architecture: DepFiN, PE={pe_rows}x{pe_cols} ({pe_rows*pe_cols:,} total)")
            print(f"  FMEM={fmem_size_kb}KB, WMEM={wmem_size_kb}KB")
        if tile_size is not None:
            print(f"  Tile size override: {tile_size}")
        if dram_energy_override is not None:
            print(f"  ⚡ DRAM energy override: {dram_energy_override} pJ/byte (default: 32 pJ/byte)")
        print(f"  ALL experiments use IDENTICAL architecture (fair comparison)")
        print(f"{'='*100}\n")

        def make_cfg(wk_name, fusion_level, variant, num_layers,
                    tile_size_override=None):
            effective_tile = (tile_size_override
                              if tile_size_override is not None
                              else tile_size)
            return ExperimentConfig(
                workload_name=wk_name, workload_variant=variant,
                fusion_level=fusion_level, num_fused_layers=num_layers,
                arch_type=arch_type,
                # DepFiN params
                fmem_size_kB=fmem_size_kb,
                wmem_size_kB=wmem_size_kb,
                # Eyeriss params
                gb_size_kB=gb_size_kb,
                pe_rows=pe_rows, pe_cols=pe_cols,
                input_reg_entries=input_reg_entries,
                weight_reg_entries=weight_reg_entries,
                intermediate_reg_entries=intermediate_reg_entries,
                output_reg_entries=output_reg_entries,
                tile_size=effective_tile,
                dram_energy_override=dram_energy_override,
                verbose=verbose,
            )

        # ── Pre-compute per-block tile sizes from full fusion stride info ──
        # When tile_size is set, derive what tile each block's output would
        # get inside full fusion, so partial blocks use the same tile as
        # the corresponding layers in full fusion.
        _block_tiles = {}  # variant_name → output tile for that block
        if tile_size is not None:
            from architectures.thesis_arch import calculate_per_layer_tile_sizes
            full_shape, _, full_nlayers = self.workload_registry.get_workload(
                network, full_fusion, full_variant)
            full_tile_sizes = calculate_per_layer_tile_sizes(
                full_shape, full_nlayers, tile_size, tile_size)
            # full_tile_sizes[i] = (h, w) at boundary i
            # full_tile_sizes[0] = input to layer 0
            # full_tile_sizes[k] = output of layer k-1 = input to layer k
            # full_tile_sizes[full_nlayers] = final output
            inter_variants = self.workload_registry.list_workloads(
                network, intermediate_level)
            cumul = 0
            for _, _, iv in inter_variants:
                _, _, nl = self.workload_registry.get_workload(
                    network, intermediate_level, iv)
                cumul += nl
                # Block output = output of its last layer = full_tile_sizes[cumul]
                _, block_tile_w = full_tile_sizes[cumul]
                # Only clamp to pe_cols for DepFiN (tile maps to SA columns);
                # Eyeriss tiles the Q dimension freely.
                if arch_type == "depfin":
                    _block_tiles[iv] = min(block_tile_w, pe_cols)
                else:
                    _block_tiles[iv] = block_tile_w
            print(f"  Per-block tile sizes (from full fusion stride propagation):")
            for iv, bt in _block_tiles.items():
                print(f"    {iv}: tile={bt}")

        # ── 1. FUSED side (the "better" fusion level) ──────────────────
        fused_results = []
        if mode in ("full_vs_single", "full_vs_partial"):
            # Full fusion is the "fused side"
            print(f"--- FULL FUSION: {network}/{full_fusion}/{full_variant} ---")
            _, _, nlayers_full = self.workload_registry.get_workload(
                network, full_fusion, full_variant)
            cfg_full = make_cfg(network, full_fusion, full_variant, nlayers_full)
            res_full = self.run_single_experiment(cfg_full)
            fused_results.append((f"{full_fusion}/{full_variant}", res_full))
            if res_full.success:
                print(f"  OK Energy: {res_full.energy_uJ:.3e} uJ, "
                      f"Latency: {res_full.latency_cycles:.3e} cc, "
                      f"EDP: {res_full.edp:.2e}")
            else:
                print(f"  FAILED: {res_full.error_message}")
        else:
            # partial_vs_single: partial fusion is the "fused side"
            print(f"--- PARTIAL FUSION ({intermediate_level}) ---")
            inter_variants = self.workload_registry.list_workloads(
                network, intermediate_level)
            for _, _, iv in inter_variants:
                _, _, nl = self.workload_registry.get_workload(
                    network, intermediate_level, iv)
                bt = _block_tiles.get(iv)  # stride-derived tile (None if no tile_size)
                cfg_v = make_cfg(network, intermediate_level, iv, nl,
                                 tile_size_override=bt)
                res_v = self.run_single_experiment(cfg_v)
                fused_results.append((f"{intermediate_level}/{iv}", res_v))
                if res_v.success:
                    print(f"  OK {iv} ({nl}L, tile={bt}): E={res_v.energy_uJ:.3e} uJ, "
                          f"L={res_v.latency_cycles:.3e} cc, EDP={res_v.edp:.2e}")
                else:
                    print(f"  FAIL {iv}: {res_v.error_message}")

        # ── 2. BASELINE side ───────────────────────────────────────────
        baseline_results = []
        if mode == "full_vs_partial":
            # Baseline = sum of partial (intermediate) segments
            print(f"\n--- BASELINE: PARTIAL FUSION ({intermediate_level}) ---")
            inter_variants = self.workload_registry.list_workloads(
                network, intermediate_level)
            for _, _, iv in inter_variants:
                _, _, nl = self.workload_registry.get_workload(
                    network, intermediate_level, iv)
                bt = _block_tiles.get(iv)  # stride-derived tile (None if no tile_size)
                cfg_v = make_cfg(network, intermediate_level, iv, nl,
                                 tile_size_override=bt)
                res_v = self.run_single_experiment(cfg_v)
                baseline_results.append((f"{intermediate_level}/{iv}", res_v))
                if res_v.success:
                    print(f"  OK {iv} ({nl}L, tile={bt}): E={res_v.energy_uJ:.3e} uJ, "
                          f"L={res_v.latency_cycles:.3e} cc, EDP={res_v.edp:.2e}")
                else:
                    print(f"  FAIL {iv}: {res_v.error_message}")
        else:
            # full_vs_single or partial_vs_single: baseline = single layers
            print(f"\n--- BASELINE: SINGLE LAYERS ---")
            single_variants = self.workload_registry.list_workloads(network, 'single')
            for _, _, sv in single_variants:
                if sv in single_exclude:
                    print(f"  [skip] {sv} (not in fusion)")
                    continue
                _, _, nl = self.workload_registry.get_workload(network, 'single', sv)
                cfg_v = make_cfg(network, 'single', sv, nl)
                res_v = self.run_single_experiment(cfg_v)
                baseline_results.append((sv, res_v))
                if res_v.success:
                    print(f"  OK {sv}: E={res_v.energy_uJ:.3e} uJ, "
                          f"L={res_v.latency_cycles:.3e} cc, EDP={res_v.edp:.2e}")
                else:
                    print(f"  FAIL {sv}: {res_v.error_message}")
        # backward compat alias
        single_results = baseline_results

        # ── AGGREGATE ──────────────────────────────────────────────────
        def aggregate(results_list):
            ok = [(v, r) for v, r in results_list if r.success]
            fail = [(v, r) for v, r in results_list if not r.success]
            if not ok:
                return None
            total_energy = sum(r.energy_uJ for _, r in ok)
            total_latency = sum(r.latency_cycles for _, r in ok)
            # EDP = E_total × L_total (in J·cc, matching engine units).
            # Summing individual EDPs would drop cross-terms (E_i × L_j, i≠j).
            total_edp = (total_energy * 1e-6) * total_latency
            return {
                'energy_uJ': total_energy,
                'latency_cycles': total_latency,
                'edp': total_edp,
                'dram_reads': sum(r.dram_reads for _, r in ok),
                'dram_writes': sum(r.dram_writes for _, r in ok),
                'num_ok': len(ok),
                'num_failed': len(fail),
                'failed_variants': [v for v, _ in fail],
            }

        agg_fused = aggregate(fused_results)
        agg_baseline = aggregate(baseline_results)
        # keep old name for return dict
        agg_singles = agg_baseline

        if mode == "full_vs_single":
            fused_label = "Full Fusion"
            baseline_label = "Sum Singles"
        elif mode == "partial_vs_single":
            fused_label = "Sum Partial Fusion"
            baseline_label = "Sum Singles"
        else:  # full_vs_partial
            fused_label = "Full Fusion"
            baseline_label = "Sum Partial Fusion"

        # ── Print comparison table ─────────────────────────────────────
        print(f"\n{'='*100}")
        print(f"COMPARISON RESULTS: {network.upper()} -- {mode_label}")
        print(f"{'='*100}")
        if arch_type == "eyeriss":
            print(f"  Architecture: Eyeriss {pe_rows}x{pe_cols}, GB={gb_size_kb}KB, "
                  f"WReg={weight_reg_entries}, InReg={input_reg_entries}, "
                  f"IntReg={intermediate_reg_entries}, OutReg={output_reg_entries}")
        else:
            print(f"  Architecture: DepFiN {pe_rows}x{pe_cols}, FMEM={fmem_size_kb}KB, "
                  f"WMEM={wmem_size_kb}KB")
        print(f"{'='*100}")
        print(f"{'Level':<25} {'Energy (uJ)':>14} {'Latency (cc)':>14} {'EDP':>14} "
              f"{'DRAM Reads':>16} {'DRAM Writes':>16}")
        print(f"{'-'*100}")

        if agg_fused:
            n_ok = agg_fused['num_ok']
            n_fail = agg_fused['num_failed']
            status = f" ({n_ok} ok" + (f", {n_fail} fail" if n_fail else "") + ")"
            print(f"{fused_label + status:<25} {agg_fused['energy_uJ']:>14.3e} "
                  f"{agg_fused['latency_cycles']:>14.3e} {agg_fused['edp']:>14.2e} "
                  f"{agg_fused['dram_reads']:>16,} {agg_fused['dram_writes']:>16,}")
            if agg_fused['num_failed'] > 0:
                print(f"  Warning: Failed: {', '.join(agg_fused['failed_variants'])}")
        else:
            print(f"{fused_label:<25} {'ALL FAILED':>14}")

        if agg_baseline:
            n_ok = agg_baseline['num_ok']
            n_fail = agg_baseline['num_failed']
            status = f" ({n_ok} ok" + (f", {n_fail} fail" if n_fail else "") + ")"
            print(f"{baseline_label + status:<25} {agg_baseline['energy_uJ']:>14.3e} "
                  f"{agg_baseline['latency_cycles']:>14.3e} {agg_baseline['edp']:>14.2e} "
                  f"{agg_baseline['dram_reads']:>16,} {agg_baseline['dram_writes']:>16,}")
            if agg_baseline['num_failed'] > 0:
                print(f"  Warning: Failed: {', '.join(agg_baseline['failed_variants'])}")
        else:
            print(f"{baseline_label:<25} {'ALL FAILED':>14}")

        if agg_fused and agg_baseline:
            print(f"\n{'-'*100}")
            print(f"RATIOS ({fused_label} / {baseline_label}) -- values < 1.0 mean fusion wins")
            print(f"{'-'*100}")

            e_ratio = agg_fused['energy_uJ'] / agg_baseline['energy_uJ']
            l_ratio = agg_fused['latency_cycles'] / agg_baseline['latency_cycles']
            edp_ratio = agg_fused['edp'] / agg_baseline['edp']
            dr_ratio = (agg_fused['dram_reads'] / agg_baseline['dram_reads']
                       if agg_baseline['dram_reads'] > 0 else float('inf'))
            dw_ratio = (agg_fused['dram_writes'] / agg_baseline['dram_writes']
                       if agg_baseline['dram_writes'] > 0 else float('inf'))

            print(f"{'Metric':<20} {'Ratio':>10} {'Savings':>12}")
            print(f"{'-'*45}")
            print(f"{'Energy':<20} {e_ratio:>10.4f} {(1-e_ratio)*100:>+11.1f}%")
            print(f"{'Latency':<20} {l_ratio:>10.4f} {(1-l_ratio)*100:>+11.1f}%")
            print(f"{'EDP':<20} {edp_ratio:>10.4f} {(1-edp_ratio)*100:>+11.1f}%")
            print(f"{'DRAM Reads':<20} {dr_ratio:>10.4f} {(1-dr_ratio)*100:>+11.1f}%")
            print(f"{'DRAM Writes':<20} {dw_ratio:>10.4f} {(1-dw_ratio)*100:>+11.1f}%")

            # ── Baseline detail ─────────────────────────────────────
            print(f"\n{'-'*100}")
            print(f"BASELINE DETAIL ({baseline_label}):")
            print(f"{'-'*100}")
            print(f"{'Layer/Segment':<30} {'Energy (uJ)':>14} {'Latency (cc)':>14} {'EDP':>14} {'Status':>10}")
            print(f"{'-'*85}")
            for v, r in baseline_results:
                if r.success:
                    print(f"{v:<30} {r.energy_uJ:>14.3e} {r.latency_cycles:>14.3e} "
                          f"{r.edp:>14.2e} {'OK':>10}")
                else:
                    print(f"{v:<30} {'--':>14} {'--':>14} {'--':>14} {'FAIL':>10}")

            # Show fused-side detail when it has multiple segments
            if len(fused_results) > 1:
                print(f"\n{'-'*100}")
                print(f"FUSED-SIDE SEGMENT DETAIL ({fused_label}):")
                print(f"{'-'*100}")
                print(f"{'Segment':<30} {'Energy (uJ)':>14} {'Latency (cc)':>14} {'EDP':>14} {'Status':>10}")
                print(f"{'-'*85}")
                for v, r in fused_results:
                    if r.success:
                        print(f"{v:<30} {r.energy_uJ:>14.3e} {r.latency_cycles:>14.3e} "
                              f"{r.edp:>14.2e} {'OK':>10}")
                    else:
                        print(f"{v:<30} {'--':>14} {'--':>14} {'--':>14} {'FAIL':>10}")

        print(f"\n{'='*100}\n")

        return {
            'network': network,
            'mode': mode,
            'fused_label': fused_label,
            'fused': agg_fused,
            'singles': agg_singles,
            'fused_results': fused_results,
            'single_results': single_results,
        }


    def export_results_csv(self, filename: Optional[str] = None) -> str:
        """Export results to CSV file."""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        if not self.results:
            print("No results to export")
            return filepath
        
        # Get field names from first result
        fieldnames = list(self.results[0].to_dict().keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:                
                writer.writerow(result.to_dict())
        
        print(f"Results exported to: {filepath}")
        return filepath
    
    def export_results_json(self, filename: Optional[str] = None) -> str:
        """Export results to JSON file."""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "num_experiments": len(self.results),
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results exported to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a summary of all results."""
        if not self.results:
            print("No results to summarize")
            return
        
        print(f"\n{'='*80}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*80}")
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        print(f"\nTotal experiments: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print(f"\n{'─'*80}")
            print("Results by workload:")
            print(f"{'─'*80}")
            
            # Group by network
            from collections import defaultdict
            by_network = defaultdict(list)
            for r in successful:
                by_network[r.config.workload_name].append(r)
            
            for network, results in by_network.items():
                print(f"\n{network.upper()}:")
                for r in results:
                    print(f"  {r.config.fusion_level}/{r.config.workload_variant}, fmem_size: {r.config.fmem_size_kB}kB, wmem_size: {r.config.wmem_size_kB}kB, pe_rows: {r.config.pe_rows}, pe_cols: {r.config.pe_cols} -> "
                          f"E={r.energy_uJ:.2e}μJ, L={r.latency_cycles:.2e}cc, "
                          f"EDP={r.edp:.2e}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Thesis Experiment Runner - Sweep workloads and collect results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available workloads
  python experiment_runner.py --list
  
  # Run a single experiment
  python experiment_runner.py --workload fsrcnn --fusion single --variant L0
  
  # Sweep all workloads with default architecture
  python experiment_runner.py --sweep-workloads
  
  # Sweep only VGG16 block fusions
  python experiment_runner.py --sweep-workloads --networks vgg16 --fusion-levels block
  
  # Sweep architecture parameters for a specific workload
  python experiment_runner.py --sweep-arch --workload fsrcnn --fusion full --variant 8layer
  
  # Sweep tile sizes for FSRCNN (case study with scaled bandwidth)
  python experiment_runner.py --sweep-tile-sizes --workload fsrcnn --fusion full --variant 8layer
  
  # Sweep tile sizes with specific tile values
  python experiment_runner.py --sweep-tile-sizes --workload fsrcnn --fusion full --variant 8layer --tile-sizes 240 120 60 30
  
  # Full sweep with custom output
  python experiment_runner.py --sweep-workloads --output my_results.csv
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--list", action="store_true",
                           help="List all available workloads")
    mode_group.add_argument("--sweep-workloads", action="store_true",
                           help="Sweep over all workloads")
    mode_group.add_argument("--sweep-arch", action="store_true",
                           help="Sweep architecture parameters for a workload")
    mode_group.add_argument("--sweep-tile-sizes", action="store_true",
                           help="Sweep tile sizes for Feature Memory width (case study)")
    mode_group.add_argument("--sweep-arch-tiles", action="store_true",
                           help="Combined sweep: architecture params AND tile sizes (2D grid)")
    mode_group.add_argument("--single", action="store_true",
                           help="Run a single experiment")
    # Eyeriss case study sweeps
    mode_group.add_argument("--sweep-wreg-pe", action="store_true",
                           help="Case Study 1: Sweep WRegister sizes vs minimum PE rows [Eyeriss]")
    mode_group.add_argument("--sweep-intreg-pe", action="store_true",
                           help="Case Study 2: Sweep IntermediateRegister sizes vs minimum PE rows [Eyeriss]")
    mode_group.add_argument("--sweep-outreg-pe", action="store_true",
                           help="Case Study 3: Sweep OutRegister sizes vs minimum PE rows [Eyeriss]")
    mode_group.add_argument("--sweep-gb-tile", action="store_true",
                           help="Case Study 4: Sweep GlobalBuffer size vs max feasible tile size [Eyeriss]")
    mode_group.add_argument("--sweep-pe-aspect", action="store_true",
                           help="Case Study 5: Sweep PE array aspect ratios with fixed total PEs [Eyeriss]")
    mode_group.add_argument("--compare-fusion", action="store_true",
                           help="Fusion Comparison: Full fusion vs non-fused vs intermediate fusion")
    mode_group.add_argument("--compare-full-vs-single", action="store_true",
                           help="Fixed-config: Full fusion vs sum singles (same WReg/PEs)")
    mode_group.add_argument("--compare-partial-vs-single", action="store_true",
                           help="Fixed-config: sum partial fusion vs sum singles (same WReg/PEs)")
    mode_group.add_argument("--compare-full-vs-partial", action="store_true",
                           help="Fixed-config: Full fusion vs sum partial fusion (same WReg/PEs)")

    # Intermediate level override for fixed-config comparisons
    parser.add_argument("--intermediate-level", type=str, default=None,
                       help="Override intermediate fusion level for --compare-*-vs-partial "
                            "(e.g. 'block' instead of default '2layer' for resnet18)")

    # Workload specification
    parser.add_argument("--workload", "-w", type=str,
                       help="Workload network name (fsrcnn, mccnn, vgg16, resnet18)")
    parser.add_argument("--fusion", "-f", type=str,
                       help="Fusion level (single, 2layer, 3layer, block, full)")
    parser.add_argument("--variant", "-v", type=str,
                       help="Workload variant name")
    
    # Sweep filters
    parser.add_argument("--networks", nargs="+", type=str,
                       help="Networks to include in sweep")
    parser.add_argument("--fusion-levels", nargs="+", type=str,
                       help="Fusion levels to include in sweep")
    
    # Architecture type selection
    parser.add_argument("--arch-type", type=str, default="depfin",
                       choices=["depfin", "eyeriss"],
                       help="Architecture type: depfin (separate FMEM/WMEM) or eyeriss (unified GlobalBuffer)")
    
    # Architecture parameters (single values)
    parser.add_argument("--fmem-size", type=int, default=1056,
                       help="Feature memory size in KB (default: 1056) [DepFiN only]")
    parser.add_argument("--wmem-size", type=int, default=524,
                       help="Weight memory size in KB (default: 524) [DepFiN only]")
    parser.add_argument("--gb-size", type=int, default=128,
                       help="Global buffer size in KB (default: 128) [Eyeriss only]")
    parser.add_argument("--pe-rows", type=int, default=16,
                       help="PE array rows (default: 16)")
    parser.add_argument("--pe-cols", type=int, default=128,
                       help="PE array columns (default: 128)")
    
    # Eyeriss register sizes (single values) — defaults from thesis_arch.EyerissArchConfig
    parser.add_argument("--input-reg", type=int, default=_EYERISS_DEFAULTS.input_reg_entries,
                       help=f"InRegister size in entries [Eyeriss only] (default: {_EYERISS_DEFAULTS.input_reg_entries})")
    parser.add_argument("--weight-reg", type=int, default=_EYERISS_DEFAULTS.weight_reg_entries,
                       help=f"WRegister size in entries [Eyeriss only] (default: {_EYERISS_DEFAULTS.weight_reg_entries})")
    parser.add_argument("--intermediate-reg", type=int, default=_EYERISS_DEFAULTS.intermediate_out_reg_entries,
                       help=f"IntermediateRegister size in entries [Eyeriss only] (default: {_EYERISS_DEFAULTS.intermediate_out_reg_entries})")
    parser.add_argument("--output-reg", type=int, default=_EYERISS_DEFAULTS.output_reg_entries,
                       help=f"OutRegister size in entries [Eyeriss only] (default: {_EYERISS_DEFAULTS.output_reg_entries})")
    
    # Per-variant Eyeriss register sizes for --compare-fusion
    # CS1 singles: defaults to arch_eyeriss_conv original sizes
    parser.add_argument("--single-gb-size", type=int, default=None,
                       help=f"GlobalBuffer size (KB) for CS1 singles [--compare-fusion] "
                            f"(default: {_ARCH_EYERISS_CONV_SIZES['global_buffer_kB']})")
    parser.add_argument("--single-input-reg", type=int, default=None,
                       help=f"InRegister entries for CS1 singles [--compare-fusion] "
                            f"(default: {_ARCH_EYERISS_CONV_SIZES['input_reg_entries']})")
    parser.add_argument("--single-weight-reg", type=int, default=None,
                       help=f"WRegister entries for CS1 singles [--compare-fusion] "
                            f"(default: {_ARCH_EYERISS_CONV_SIZES['weight_reg_entries']})")
    parser.add_argument("--single-output-reg", type=int, default=None,
                       help=f"OutRegister entries for CS1 singles [--compare-fusion] "
                            f"(default: {_ARCH_EYERISS_CONV_SIZES['output_reg_entries']})")
    # CS2 intermediate: defaults to full fusion sizes
    parser.add_argument("--int-gb-size", type=int, default=None,
                       help="GlobalBuffer size (KB) for CS2 intermediate [--compare-fusion] "
                            "(default: same as --gb-size)")
    parser.add_argument("--int-input-reg", type=int, default=None,
                       help="InRegister entries for CS2 intermediate [--compare-fusion] "
                            "(default: same as --input-reg)")
    parser.add_argument("--int-weight-reg", type=int, default=None,
                       help="WRegister entries for CS2 intermediate [--compare-fusion] "
                            "(default: same as --weight-reg)")
    parser.add_argument("--int-intermediate-reg", type=int, default=None,
                       help="IntermediateRegister entries for CS2 intermediate [--compare-fusion] "
                            "(default: same as --intermediate-reg)")
    parser.add_argument("--int-output-reg", type=int, default=None,
                       help="OutRegister entries for CS2 intermediate [--compare-fusion] "
                            "(default: same as --output-reg)")
    
    # Eyeriss register sweep ranges (for case studies 1-3)
    parser.add_argument("--weight-reg-sizes", nargs="+", type=int,
                       help="WRegister sizes to sweep [Eyeriss Case Study 1]")
    parser.add_argument("--intermediate-reg-sizes", nargs="+", type=int,
                       help="IntermediateRegister sizes to sweep [Eyeriss Case Study 2]")
    parser.add_argument("--output-reg-sizes", nargs="+", type=int,
                       help="OutRegister sizes to sweep [Eyeriss Case Study 3]")
    
    # PE grid search parameters (for Eyeriss case studies 1-3)
    parser.add_argument("--pe-rows-grid", nargs="+", type=int,
                       help="Custom PE row candidates for grid search [Eyeriss Case Studies 1-3]")
    parser.add_argument("--pe-cols-grid", nargs="+", type=int,
                       help="Custom PE col candidates for grid search [Eyeriss Case Studies 1-3]")
    # PE array sweep for case studies (for case study 5)
    parser.add_argument("--total-pes", type=int, default=168,
                       help="Total number of PEs for aspect ratio sweep [Eyeriss Case Study 5] (default: 168)")
    parser.add_argument("--pe-aspect-ratios", nargs="+", type=str,
                       help="PE aspect ratios to sweep as ROWSxCOLS (must equal --total-pes)")
    
    # Single tile size for constrained mapping (--single mode with Eyeriss)
    parser.add_argument("--tile-size", type=int, default=None,
                       help="Output tile size for constrained Eyeriss mapping. "
                            "When specified, creates a fully-constrained architecture "
                            "with deterministic mapping (no search). Must be a divisor of Q shape.")
    
    # Architecture sweep ranges (for --sweep-arch mode)
    parser.add_argument("--fmem-sizes", nargs="+", type=int,
                       help="FMEM sizes to sweep in KB [DepFiN only] (default: 512 1024 2048)")
    parser.add_argument("--wmem-sizes", nargs="+", type=int,
                       help="WMEM sizes to sweep in KB [DepFiN only] (default: 256 512 1024)")
    parser.add_argument("--gb-sizes", nargs="+", type=int,
                       help="GlobalBuffer sizes to sweep in KB [Eyeriss only] (default: 64 128 256)")
    parser.add_argument("--pe-configs", nargs="+", type=str,
                       help="PE configs to sweep as ROWSxCOLS (default: 8x64 16x128 32x256)")
    
    # Tile size sweep options (for --sweep-tile-sizes mode)
    parser.add_argument("--tile-sizes", nargs="+", type=int,
                       help="Specific tile sizes to sweep (default: auto-compute divisors)")
    parser.add_argument("--no-scale-bandwidth", action="store_true",
                       help="Disable bandwidth scaling with tile size (default: enabled)")
    parser.add_argument("--base-tile-size", type=int, default=128,
                       help="Reference tile size for bandwidth scaling (default: 128)")
    
    # Energy override
    parser.add_argument("--dram-energy", type=float, default=None,
                       help="Override DRAM access energy in pJ/byte (default: Accelergy-derived, ~32 pJ/byte)")
    
    # Output options
    parser.add_argument("--output", "-o", type=str,
                       help="Output filename for results (CSV)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory (default: results)")
    parser.add_argument("--json", action="store_true",
                       help="Also export results as JSON")
    
    # Verbosity
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output during experiments")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress progress output")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    # List mode
    if args.list:
        print("\nAvailable workloads:")
        print("=" * 60)
        for network in runner.workload_registry.get_networks():
            print(f"\n{network.upper()}:")
            for fusion in runner.workload_registry.get_fusion_levels(network):
                variants = runner.workload_registry.list_workloads(network, fusion)
                print(f"  {fusion}:")
                for _, _, variant in variants:
                    shape, coupling, num_layers = runner.workload_registry.get_workload(
                        network, fusion, variant
                    )
                    print(f"    - {variant} ({num_layers} layers)")
        return
    
    # Create base config from args
    base_config = ExperimentConfig(
        arch_type=args.arch_type,
        fmem_size_kB=args.fmem_size,
        wmem_size_kB=args.wmem_size,
        gb_size_kB=args.gb_size,
        pe_rows=args.pe_rows,
        pe_cols=args.pe_cols,
        input_reg_entries=args.input_reg,
        weight_reg_entries=args.weight_reg,
        intermediate_reg_entries=args.intermediate_reg,
        output_reg_entries=args.output_reg,
        tile_size=args.tile_size,
        scale_bandwidth=not args.no_scale_bandwidth,
        base_tile_size=args.base_tile_size,
        verbose=args.verbose,
    )
    
    # Run experiments based on mode
    if args.sweep_workloads:
        runner.run_workload_sweep(
            networks=args.networks,
            fusion_levels=args.fusion_levels,
            arch_config=base_config,
        )
    
    elif args.sweep_arch:
        if not all([args.workload, args.fusion, args.variant]):
            print("Error: --sweep-arch requires --workload, --fusion, and --variant")
            return
        
        # Parse PE configs if provided (format: "8x64 16x128")
        pe_configs = None
        if args.pe_configs:
            pe_configs = []
            for cfg in args.pe_configs:
                rows, cols = cfg.lower().split('x')
                pe_configs.append((int(rows), int(cols)))
        
        runner.run_architecture_sweep(
            workload=(args.workload, args.fusion, args.variant),
            arch_type=args.arch_type,
            fmem_sizes_kb=args.fmem_sizes,
            wmem_sizes_kb=args.wmem_sizes,
            gb_sizes_kb=args.gb_sizes,
            pe_configs=pe_configs,
            tile_size=args.tile_size,
            input_reg=args.input_reg,
            weight_reg=args.weight_reg,
            intermediate_reg=args.intermediate_reg,
            output_reg=args.output_reg,
            verbose=args.verbose,
        )
    
    elif args.sweep_tile_sizes:
        if not all([args.workload, args.fusion, args.variant]):
            print("Error: --sweep-tile-sizes requires --workload, --fusion, and --variant")
            return
        
        runner.run_tile_size_sweep(
            workload=(args.workload, args.fusion, args.variant),
            tile_sizes=args.tile_sizes,  # None = auto-compute
            arch_type=args.arch_type,
            fmem_size_kb=args.fmem_size,
            wmem_size_kb=args.wmem_size,
            gb_size_kb=args.gb_size,
            pe_rows=args.pe_rows,
            pe_cols=args.pe_cols,
            scale_bandwidth=not args.no_scale_bandwidth,
            base_tile_size=args.base_tile_size,
            verbose=args.verbose,
        )
    
    elif args.sweep_arch_tiles:
        if not all([args.workload, args.fusion, args.variant]):
            print("Error: --sweep-arch-tiles requires --workload, --fusion, and --variant")
            return
        
        # Parse PE configs if provided (format: "8x64 16x128")
        pe_configs = None
        if args.pe_configs:
            pe_configs = []
            for cfg in args.pe_configs:
                rows, cols = cfg.lower().split('x')
                pe_configs.append((int(rows), int(cols)))
        
        runner.run_combined_arch_tile_sweep(
            workload=(args.workload, args.fusion, args.variant),
            arch_type=args.arch_type,
            fmem_sizes_kb=args.fmem_sizes,
            wmem_sizes_kb=args.wmem_sizes,
            gb_sizes_kb=args.gb_sizes,
            pe_configs=pe_configs,
            tile_sizes=args.tile_sizes,
            scale_bandwidth=not args.no_scale_bandwidth,
            base_tile_size=args.base_tile_size,
            verbose=args.verbose,
        )
    
    # =========================================================================
    # EYERISS CASE STUDIES
    # =========================================================================
    
    elif args.sweep_wreg_pe:
        # Case Study 1: WRegister size vs minimum PE configuration
        if not all([args.workload, args.fusion, args.variant]):
            print("Error: --sweep-wreg-pe requires --workload, --fusion, and --variant")
            return
        if args.arch_type != "eyeriss":
            print("Warning: --sweep-wreg-pe is designed for Eyeriss architecture, setting arch_type=eyeriss")
        
        runner.run_wreg_pe_sweep(
            workload=(args.workload, args.fusion, args.variant),
            weight_reg_sizes=args.weight_reg_sizes or [96, 192, 384, 768],
            gb_size_kb=args.gb_size,
            tile_size=args.tile_size or 80,
            input_reg_entries=args.input_reg,
            intermediate_reg_entries=args.intermediate_reg,
            output_reg_entries=args.output_reg,
            pe_rows_grid=args.pe_rows_grid,
            pe_cols_grid=args.pe_cols_grid,
            verbose=args.verbose,
        )
    
    elif args.sweep_intreg_pe:
        # Case Study 2: IntermediateRegister size vs minimum PE configuration
        if not all([args.workload, args.fusion, args.variant]):
            print("Error: --sweep-intreg-pe requires --workload, --fusion, and --variant")
            return
        if args.arch_type != "eyeriss":
            print("Warning: --sweep-intreg-pe is designed for Eyeriss architecture, setting arch_type=eyeriss")
        
        runner.run_intreg_pe_sweep(
            workload=(args.workload, args.fusion, args.variant),
            intermediate_reg_sizes=args.intermediate_reg_sizes or [64, 128, 256, 512],
            gb_size_kb=args.gb_size,
            tile_size=args.tile_size or 80,
            input_reg_entries=args.input_reg,
            weight_reg_entries=args.weight_reg,
            output_reg_entries=args.output_reg,
            pe_rows_grid=args.pe_rows_grid,
            pe_cols_grid=args.pe_cols_grid,
            verbose=args.verbose,
        )
    
    elif args.sweep_outreg_pe:
        # Case Study 3: OutRegister size vs minimum PE configuration
        if not all([args.workload, args.fusion, args.variant]):
            print("Error: --sweep-outreg-pe requires --workload, --fusion, and --variant")
            return
        if args.arch_type != "eyeriss":
            print("Warning: --sweep-outreg-pe is designed for Eyeriss architecture, setting arch_type=eyeriss")
        
        runner.run_outreg_pe_sweep(
            workload=(args.workload, args.fusion, args.variant),
            output_reg_sizes=args.output_reg_sizes or [8, 16, 32, 64],
            gb_size_kb=args.gb_size,
            tile_size=args.tile_size or 80,
            input_reg_entries=args.input_reg,
            weight_reg_entries=args.weight_reg,
            intermediate_reg_entries=args.intermediate_reg,
            pe_rows_grid=args.pe_rows_grid,
            pe_cols_grid=args.pe_cols_grid,
            verbose=args.verbose,
        )
    
    elif args.sweep_gb_tile:
        # Case Study 4: GlobalBuffer size vs max feasible tile size
        if not all([args.workload, args.fusion, args.variant]):
            print("Error: --sweep-gb-tile requires --workload, --fusion, and --variant")
            return
        if args.arch_type != "eyeriss":
            print("Warning: --sweep-gb-tile is designed for Eyeriss architecture, setting arch_type=eyeriss")
        
        runner.run_gb_tile_sweep(
            workload=(args.workload, args.fusion, args.variant),
            gb_sizes_kb=args.gb_sizes or [64, 128, 256, 512],
            pe_rows=args.pe_rows,
            pe_cols=args.pe_cols,
            input_reg_entries=args.input_reg,
            weight_reg_entries=args.weight_reg,
            intermediate_reg_entries=args.intermediate_reg,
            output_reg_entries=args.output_reg,
            verbose=args.verbose,
        )
    
    elif args.sweep_pe_aspect:
        # Case Study 5: PE aspect ratio with fixed total PEs
        if not all([args.workload, args.fusion, args.variant]):
            print("Error: --sweep-pe-aspect requires --workload, --fusion, and --variant")
            return
        if args.arch_type != "eyeriss":
            print("Warning: --sweep-pe-aspect is designed for Eyeriss architecture, setting arch_type=eyeriss")
        
        # Parse PE aspect ratios if provided
        pe_aspect_ratios = None
        if args.pe_aspect_ratios:
            pe_aspect_ratios = []
            for cfg in args.pe_aspect_ratios:
                rows, cols = cfg.lower().split('x')
                pe_aspect_ratios.append((int(rows), int(cols)))
        
        runner.run_pe_aspect_sweep(
            workload=(args.workload, args.fusion, args.variant),
            total_pes=args.total_pes,
            pe_aspect_ratios=pe_aspect_ratios,
            gb_size_kb=args.gb_size,
            tile_size=args.tile_size or 80,
            input_reg_entries=args.input_reg,
            weight_reg_entries=args.weight_reg,
            intermediate_reg_entries=args.intermediate_reg,
            output_reg_entries=args.output_reg,
            verbose=args.verbose,
        )
    
    elif args.compare_fusion:
        # Fusion Comparison: full vs singles vs intermediate
        if not args.workload:
            print("Error: --compare-fusion requires --workload (fsrcnn, mccnn, vgg16, resnet18)")
            return
        
        runner.run_fusion_comparison(
            network=args.workload,
            arch_type=args.arch_type,
            fmem_size_kb=args.fmem_size,
            wmem_size_kb=args.wmem_size,
            gb_size_kb=args.gb_size,
            input_reg_entries=args.input_reg,
            weight_reg_entries=args.weight_reg,
            intermediate_reg_entries=args.intermediate_reg,
            output_reg_entries=args.output_reg,
            pe_rows=args.pe_rows,
            pe_cols=args.pe_cols,
            tile_size=args.tile_size,
            scale_bandwidth=not args.no_scale_bandwidth,
            base_tile_size=args.base_tile_size,
            dram_energy_override=args.dram_energy,
            verbose=args.verbose,
            # Per-variant Eyeriss sizes for fairness
            single_gb_size_kb=args.single_gb_size,
            single_input_reg=args.single_input_reg,
            single_weight_reg=args.single_weight_reg,
            single_output_reg=args.single_output_reg,
            intermediate_gb_size_kb=args.int_gb_size,
            intermediate_input_reg=args.int_input_reg,
            intermediate_weight_reg=args.int_weight_reg,
            intermediate_intermediate_reg=args.int_intermediate_reg,
            intermediate_output_reg=args.int_output_reg,
        )
    
    elif args.compare_full_vs_single or args.compare_partial_vs_single or args.compare_full_vs_partial:
        if not args.workload:
            print("Error: requires --workload (fsrcnn, mccnn, vgg16, resnet18)")
            return
        if args.compare_full_vs_single:
            compare_mode = "full_vs_single"
        elif args.compare_partial_vs_single:
            compare_mode = "partial_vs_single"
        else:
            compare_mode = "full_vs_partial"
        runner.run_fixed_fusion_comparison(
            network=args.workload,
            mode=compare_mode,
            arch_type=args.arch_type,
            fmem_size_kb=args.fmem_size,
            wmem_size_kb=args.wmem_size,
            gb_size_kb=args.gb_size,
            pe_rows=args.pe_rows,
            pe_cols=args.pe_cols,
            input_reg_entries=args.input_reg,
            weight_reg_entries=args.weight_reg,
            intermediate_reg_entries=args.intermediate_reg,
            output_reg_entries=args.output_reg,
            tile_size=args.tile_size,
            intermediate_level_override=args.intermediate_level,
            dram_energy_override=args.dram_energy,
            verbose=args.verbose,
        )


    elif args.single or (args.workload and args.fusion and args.variant):
        if not all([args.workload, args.fusion, args.variant]):
            print("Error: Single run requires --workload, --fusion, and --variant")
            return
        config = ExperimentConfig(
            workload_name=args.workload,
            workload_variant=args.variant,
            fusion_level=args.fusion,
            arch_type=args.arch_type,
            fmem_size_kB=args.fmem_size,
            wmem_size_kB=args.wmem_size,
            gb_size_kB=args.gb_size,
            pe_rows=args.pe_rows,
            pe_cols=args.pe_cols,
            input_reg_entries=args.input_reg,
            weight_reg_entries=args.weight_reg,
            intermediate_reg_entries=args.intermediate_reg,
            output_reg_entries=args.output_reg,
            tile_size=args.tile_size,  # For constrained Eyeriss mapping
            scale_bandwidth=not args.no_scale_bandwidth,
            base_tile_size=args.base_tile_size,
            verbose=True,  # Always verbose for single runs
        )
        result = runner.run_single_experiment(config)
        if result.success:
            print(f"\n{'='*60}")
            print("RESULT")
            print(f"{'='*60}")
            print(f"Energy: {result.energy_uJ:.4e} μJ")
            print(f"Latency: {result.latency_cycles:.4e} cycles")
            print(f"EDP: {result.edp:.4e}")
            print(f"Utilization: {result.utilization:.2%}")
            print(f"Mapping time: {result.mapping_time_s:.2f} s")
            if result.mapping_summary:
                print(f"\nFinal Mapping:")
                print(result.mapping_summary)
        else:
            print(f"\nExperiment failed: {result.error_message}")
    
    else:
        print("No mode specified. Use --help for usage information.")
        return
    
    # Export results
    if runner.results:
        runner.export_results_csv(args.output)
        if args.json:
            runner.export_results_json()
        runner.print_summary()


if __name__ == "__main__":
    main()
