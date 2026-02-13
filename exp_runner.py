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
    
    python3 experiment_runner.py --sweep-tile-sizes \
    --workload fsrcnn --fusion full --variant 8layer \
    --tile-sizes 120 64 32 16 8 2>&1

    Tried tile sizes: 120 - 64 - 32 - 16 - 8

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



    
Case Study: PE Array Sweep — Increasing PE Rows (fixed cols=64) ----- FSRCNN 8-layer full fusion
  FSRCNN 8-layer full fusion, DepFiN, FMEM=1056KB, WMEM=524KB

    python3 experiment_runner.py --sweep-arch \
    --workload fsrcnn --fusion full --variant 8layer \
    --fmem-sizes 1056 --wmem-sizes 524 \
    --pe-configs 8x64 16x64 32x64 64x64 2>&1

    Config   | PEs   | Energy(μJ) | Latency(cc) | EDP
    8×64     | 512   | 1.556e+04  | 2.125e+07   | 4.95e+05
    16×64    | 1024  | 1.500e+04  | 1.110e+07   | 2.51e+05
    32×64    | 2048  | 1.500e+04  | 1.070e+07   | 2.42e+05
    64×64    | 4096  | 1.490e+04  | 1.070e+07   | 2.41e+05

    What changes in the mapping: 
      SARows Z values increase (more output channels parallelized spatially).
      DRAM iterations unchanged (Q=15 for all, same tile_size=64).
      WMEM content unchanged (all weights fit in WMEM for all configs).
      Intermediate output register and output register iterations decrease (more spatial parallelism → fewer temporal iterations at WMEM/FMEM levels).

      8 rows:  SARows Z7=8,  Z6=8,  Z5=6,  Z4-Z2=6, Z1=6,  Z0=8
      16 rows: SARows Z7=16, Z6=14, Z5=12, Z4-Z2=12, Z1=12, Z0=14
      32 rows: SARows Z7=16, Z6=28, Z5=12, Z4-Z2=12, Z1=12, Z0=28
      64 rows: SARows Z7=16, Z6=56, Z5=12, Z4-Z2=12, Z1=12, Z0=56

    Energy: slight decrease (~4%, 8→16 rows), then near-constant.
      The total DRAM reads are identical (in=1,555,200, w=18,792, out_w=8,294,400).
      The slight reduction comes from fewer temporal iterations at WMEM/FMEM levels
      (more spatial parallelism → fewer reads at intermediate memory levels).

    Latency: two-stage saturation pattern.
      8→16:  ~1.9× improvement. The PRIMARY bottleneck layer L7 (Z7=16) goes from
        SARows Z7=8 to Z7=16, fully parallelizing its Z dimension.
        This halves the temporal weight iterations for L7, nearly halving latency.
      16→32: ~3.6% improvement (1.110e+07 → 1.070e+07). L7 is already saturated, but
        L6 and L0 (Z_shape=56) are SECONDARY bottlenecks. Z6 goes from 14→28
        (temporal iterations halved from 4→2), producing a small gain.

    KEY INSIGHT: Latency shows a two-stage saturation pattern.
      (1) PRIMARY bottleneck (L7, Z7=16): saturates at 16 rows → big 1.9× improvement.
      (2) SECONDARY bottleneck (L6, Z6=56): provides diminishing returns beyond 16 rows
          → small 3.6% improvement from 16→32 rows.


Case Study: PE Array Sweep — Increasing PE Cols (fixed rows=16) ----- FSRCNN 8-layer full fusion
  FSRCNN 8-layer full fusion, DepFiN, FMEM=1056KB, WMEM=524KB

  In the DepFiN architecture, the tile size is automatically set as the largest divisor of Q that fits in pe_cols

    python3 experiment_runner.py --sweep-arch \
    --workload fsrcnn --fusion full --variant 8layer \
    --fmem-sizes 1056 --wmem-sizes 524 \
    --pe-configs 16x64 16x120 16x128 16x256 2>&1

    Config   | PEs   | Energy(μJ) | Latency(cc) | EDP
    16×64    | 1024  | 1.500e+04  | 1.110e+07   | 2.51e+05
    16×120   | 1920  | 1.500e+04  | 6.120e+06   | 1.39e+05
    16×128   | 2048  | 1.500e+04  | 6.120e+06   | 1.39e+05
    16×256   | 4096  | 1.495e+04  | 3.349e+06   | 7.59e+04

    What changes in the mapping:
      SACols Q/X values increase (larger tile fits spatially).
      DRAM Q/X iterations DECREASE: Q=15 → 8 → 8 → 4.
      SARows Z unchanged (always Z7=16, Z6=14, Z5-Z2=12, Z0=14).
      WMEM content unchanged (all weights fit for all configs).

      64 cols:  SACols Q/X = 64,  DRAM Q = 15  (tile_size = 64)
      120 cols: SACols Q/X = 120, DRAM Q = 8   (tile_size = 120 = max divisor of 960 ≤ 120)
      128 cols: SACols Q/X = 120, DRAM Q = 8   (tile_size = 120, NOT 128: 960%128≠0)
      256 cols: SACols Q/X = 240, DRAM Q = 4   (tile_size = 240 = max divisor of 960 ≤ 256)

    Energy: near-constant across all configs (~1.50e+04 μJ).
      Total DRAM reads are identical (in=1,555,200, w=18,792). All weights always fit in WMEM.
      More cols only change HOW the spatial dimensions are distributed (larger tile), 
      not the total number of memory operations.

    Latency: consistent ~1.8× reduction each time tile size doubles.
      More cols → larger tile → fewer DRAM temporal iterations in Q/X dimensions.
      64→120: DRAM Q from 15→8 iterations. Latency halves (1.11e+07 → 6.12e+06).
      120→128: NO improvement. tile_size stays 120 (128 doesn't divide Q=960 evenly)
      128→256: DRAM Q from 8→4 iterations. Latency halves again (6.12e+06 → 3.35e+06).
    
    Note: 120→128 cols gives ZERO improvement because tile_size = max divisor of Q ≤ pe_cols.
      Q=960, and the max divisor of 960 ≤128 is 120 (≠128). So 8 extra PEs are unused.
      This shows that PE cols should be chosen to match valid tile sizes for the workload.




    
Case Study: PE Array Sweep — Changing the aspect ratio (fixed total PEs=2048)
  FSRCNN 8-layer full fusion, DepFiN, FMEM=1056KB, WMEM=524KB

    python3 experiment_runner.py --sweep-arch \
    --workload fsrcnn --fusion full --variant 8layer \
    --fmem-sizes 1056 --wmem-sizes 524 \
    --pe-configs 2x1024 4x512 8x256 16x128 32x64 64x32  2>&1
    
    Config   | Tile | DRAM Q | SARows Z7 | SARows Z6 | SARows Z1-5 | Energy(μJ) | Latency(cc) | EDP
    2×1024   | 960  | 1      | 2         | 2         | 2           | 1.83e+04   | 8.89e+06    | 2.31e+05
    4×512    | 480  | 2      | 4         | 4         | 4           | 1.63e+04   | 5.68e+06    | 1.37e+05  ← BEST EDP
    8×256    | 240  | 4      | 8         | 8         | 6           | 1.55e+04   | 5.98e+06    | 1.39e+05
    16×128   | 120  | 8      | 16        | 14        | 12          | 1.50e+04   | 6.12e+06    | 1.39e+05
    32×64    | 64   | 15     | 16        | 28        | 12          | 1.50e+04   | 1.07e+07    | 2.42e+05
    64×32    | 32   | 30     | 16        | 56        | 12          | 1.51e+04   | 2.11e+07    | 4.80e+05

    What changes in the mapping (trade-off between SACols Q/X and SARows Z):
      - SACols Q/X = tile_size = pe_cols  (960 → 480 → 240 → 120 → 64 → 32)
      - SARows Z = min(pe_rows, Z_dim)  — saturates when pe_rows ≥ Z dimension
      - DRAM Q iterations = ceil(960 / tile_size) — increases as tile shrinks
      - FSRCNN Z dimensions: Z7=16, Z6=56, Z0=14, Z1-Z5=12
      - Intermediate Output register iterations = ceil(Z_dim / SARows Z) — decreases as more Z parallelism

    Energy: monotonic decrease from 2×1024 to 16×128, then plateau (~1.50e+04 μJ).
      More rows → more Z parallelism → fewer temporal weight iterations at WMEM level
      → fewer intermediate reads. Saturates once Z dimensions are fully parallelized.
      Beyond 16 rows, only Z6 (=56) can use more rows, but it contributes
      little extra energy saving since the bottleneck layer L7 (Z7=16) is already saturated.

    Latency: Minimum at 4×512 (5.68e+06 cc).
      Too few rows → Z under-parallelized:
        2×1024: Tile=960 covers all of Q spatially (DRAM Q=1), but only 2 Z rows.
          Each layer needs many temporal Z iterations (Z7=16 needs 8 passes).
          Despite zero DRAM Q overhead, the WMEM weight-reload penalty dominates.

      Best SPOT (4×512 to 16×128): Latency roughly stable (~5.7M–6.1M cc).
        Increasing rows from 4→8→16 gives more Z parallelism, but the simultaneous
        tile shrink (480→240→120) adds more DRAM Q iterations (2→4→8).
        
      Too many rows → Q tile too small:
        32×64:  Z saturated at most layers (Z7=16 ✓, Z1-5=12 ✓), but tile=64 → 
          DRAM Q=15 iterations. The 15× Q overhead overwhelms the Z parallelism gain.
        64×32:  Same Z saturation, tile=32 → DRAM Q=30 iterations. Latency explodes ~2×.

      1. MORE ROWS (Z parallelism): Reduces temporal weight iterations across all layers.
         Benefit saturates when pe_rows ≥ max(Z_dims). For FSRCNN: only Z6=56 keeps benefiting beyond 32 rows.
      2. MORE COLS (Q tile size): Reduces DRAM temporal iterations for Q/X dimensions.
         Benefit scales continuously (no saturation until tile=Q=960).

      CONCLUSION: For activation-dominant workloads like FSRCNN (large Q, small Z),
      a column-heavy aspect ratio is preferred. The optimal ratio depends on when
      the dominant Z dimensions saturate vs. the Q iteration cost.


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
    --wmem-size 11264 --fmem-size 1056 --pe-rows 16 --pe-cols 128 \
    --tile-sizes 7 1 2>&1

    Only 2 valid tile sizes: Q=7 → divisors are 7 and 1.
    (Cumulative stride = 2⁴ = 16, so output_tile=7 → input_tile=112)

    Tile  | SACols Q | SACols X0 | DRAM Q iters | Energy(μJ) | Latency(cc) | EDP
    7     | Q=7      | X0=112    | 1            | 1.088e+04  | 8.998e+06   | 1.75e+05
    1     | Q=1      | X0=16     | 7            | 1.362e+04  | 6.299e+07   | 1.40e+06

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

    EDP: 8× worse (1.75e+05 → 1.40e+06).

    COMPARISON WITH FSRCNN TILE SWEEP:
      FSRCNN had many valid tile sizes (120, 64, 32, 16, 8) → gradual degradation.
      ResNet18 has only 2 valid tiles (7, 1) → binary choice, no middle ground.
      This is because Q=7 with cumulative stride=16 severely limits
      valid tiles. For weight-dominant workloads, tile=Q (full output width) is always best
      since there's no DRAM Q overhead and weight re-reads are minimized.





      
Case Study: PE Array Sweep — Increasing PE Rows (fixed cols=128) ----- ResNet18 17-layer full fusion
  DepFiN, FMEM=1056KB, WMEM=11264KB

    python3 experiment_runner.py --sweep-arch \
    --workload resnet18 --fusion full --variant 17layer \
    --fmem-sizes 1056 --wmem-sizes 11264 \
    --pe-configs 8x128 16x128 32x128 64x128 128x128 256x128 512x128 2>&1

    ResNet18 Z dimensions: Z0=64, Z1-4=64, Z5-8=128, Z9-12=256, Z13-16=512

    Config   | PEs    | SARows Z16 | SARows Z0 | FMem Reads | Energy(μJ) | Latency(cc) | Bottleneck | EDP
    8×128    | 1,024  | 8          | 8         | 292M       | 1.099e+04  | 1.56e+07    | DRAM       | 3.05e+05
    16×128   | 2,048  | 16         | 16        | 147M       | 1.088e+04  | 7.81e+06    | DRAM       | 1.52e+05
    32×128   | 4,096  | 32         | 32        | 75M        | 1.082e+04  | 7.81e+06    | WMem       | 1.51e+05
    64×128   | 8,192  | 64         | 64        | 39M        | 1.079e+04  | 7.81e+06    | WMem       | 1.51e+05
    128×128  | 16,384 | 128        | 64        | 25M        | 1.078e+04  | 7.81e+06    | WMem       | 1.51e+05
    256×128  | 32,768 | 256        | 64        | 21M        | 1.077e+04  | 7.81e+06    | WMem       | 1.51e+05
    512×128  | 65,536 | 512        | 64        | 20M        | 1.077e+04  | 7.81e+06    | WMem       | 1.51e+05

    What changes in the mapping:
      SARows Z increases with pe_rows: each layer gets Z_i = min(pe_rows, Z_i_shape).
      SACols unchanged.
      DRAM unchanged.

    Latency: only ONE step of improvement (8→16 rows: 2× reduction)
      8→16:  1.56e+07 → 7.81e+06 cc (2× improvement). Bottleneck is DRAM.
        At 8 rows, DRAM takes longer because fe memory feeds data more slowly
        (more temporal Z iterations upstream → more stalls/delay propagating to DRAM).
      16→512: Latency is CONSTANT at 7.81e+06 cc. No further improvement.
        The bottleneck shifts to WeightMemory (16 words/cc bandwidth) at 32+ rows.
        Adding rows doesn't help.

    Energy: tiny decrease (~2% total, from 10,992 → 10,773 μJ over 8→512 rows).
      What is CONSTANT across all configs (and why):
        - DRAM reads:  11,032,512 (all data fits in on-chip memories, loaded once)
        - WMem reads:  124,916,736 (all weights loaded once from DRAM→WMem→WReg)
        - WReg reads:  2,314,518,528 (total MAC operations, independent of rows)
        - IntReg R+W:  2,201,722,880 each (intermediate activations between layers)
        - OutReg R+W:  115,605,504 each (final output accumulation)
      What CHANGES (only FeatureMemory reads):
        8 rows:   FMem reads = 292,149,760
        16 rows:  FMem reads = 147,492,352  (halved)
        32 rows:  FMem reads = 75,163,648   (halved again)
        64 rows:  FMem reads = 38,999,296
        128 rows: FMem reads = 25,451,776
        256 rows: FMem reads = 20,935,936
        512 rows: FMem reads = 19,806,976   (near minimum)

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
      ResNet18: Latency shows a single step (2× at 16 rows), then HARD CEILING.
        Z dimensions are large (64-512), but the bottleneck is WeightMemory bandwidth,
        not DRAM Q. With DRAM Q=1 (tile=7 covers all of Q=7), the only bottleneck
        after DRAM is the WMem→WReg pipe. No amount of Z parallelism helps.


        
    
Case Study: PE Array Sweep — Increasing PE Cols (fixed rows=16) ----- ResNet18 17-layer full fusion
  ResNet18 17-layer full fusion, DepFiN, FMEM=1056KB, WMEM=11264KB

    python3 experiment_runner.py --sweep-arch \
      --workload resnet18 --fusion full --variant 17layer \
      --fmem-sizes 1056 --wmem-sizes 11264 \
      --pe-configs 16x7 16x14 16x28 16x56 16x112 16x128 \
      --verbose 2>&1 | tee results/DF_resnet_pe_cols_sweep.log

    SACols mapping (cols map X_i/Q spatial dims, clamped to actual dimension):
      ResNet18 input spatial dims per stage group (from cumulative strides):
        Q = 7  (output spatial)
        X13 = 14  (after stride-2 at L13, receptive field from Q)
        X9  = 28  (after stride-2 at L9)
        X5  = 56  (after stride-2 at L5)
        X0  = 112 (input image, after stride-2 at L0)

      SACols allocation = min(pe_cols, X_i) for each stage:
      Cols  |  Q  | X13 | X9  | X5  | X0
      ------+-----+-----+-----+-----+------
        7   |  7  |  7  |  7  |  7  |   7    
       14   |  7  | 14  | 14  | 14  |  14    <- X13 fully covered
       28   |  7  | 14  | 28  | 28  |  28    <- X9 fully covered
       56   |  7  | 14  | 28  | 56  |  56    <- X5 fully covered
      112   |  7  | 14  | 28  | 56  | 112    <- X0 fully covered (all stages full)
      128   |  7  | 14  | 28  | 56  | 112    <- same as 112 (X0 max = 112)

    Results:
      Cols | Energy (uJ) | Latency (cc) | Bottleneck | WMem Reads    | DRAM Reads  | FMem Reads
      -----+-------------+--------------+------------+---------------+-------------+------------
        7  | 11,630      | 2,067,968    | DRAM       | 330,645,504   | 11,032,512  | 147,492,352
       14  | 11,115      | 1,033,984    | DRAM       | 190,095,360   | 11,032,512  | 147,492,352
       28  | 10,933      | 1,032,640    | DRAM       | 140,464,128   | 11,032,512  | 147,492,352
       56  | 10,880      | 1,032,640    | DRAM       | 125,970,432   | 11,032,512  | 147,492,352
      112  | 10,876      | 1,032,640    | DRAM       | 124,916,736   | 11,032,512  | 147,492,352
      128  | 10,876      | 1,032,640    | DRAM       | 124,916,736   | 11,032,512  | 147,492,352

    Analysis:
      - DRAM reads and FMem reads are CONSTANT across all configs. 
        Only WMem reads change. Cols only affect how many temporal passes over
        weights are needed (more spatial coverage → fewer WMem reloads).

      - LATENCY: A single 2× step from cols=7 → cols=14, then completely flat.
        At 7 cols, DRAM latency is 2.07M cc; from 14+ cols it locks at ~1.03M cc.
        The bottleneck is always DRAM. Once cols ≥ 14 (enough to fully cover X13),
        the DRAM streaming schedule no longer requires extra passes.

      - ENERGY: Gradual decrease driven by falling WMem reads (energy cost per WMem
        access). From 7→128 cols: 11,630 → 10,876 uJ (−6.5% total).
        Breakdown by step:
          7→14:   −515 uJ (−4.4%)  ... WMem reads drop 43%
          14→28:  −182 uJ (−1.6%)  ... WMem reads drop 26%
          28→56:   −53 uJ (−0.5%)  ... WMem reads drop 10%
          56→112:   −4 uJ (~0%)    ... WMem reads drop 0.8% improvement for X0
          112→128:   0 uJ          ... identical (X0 already fully mapped at 112)

      - SATURATION AT 112 COLS: With 128 cols, the mapping is identical to 112.
        X0 = 112 is the largest spatial dimension in the fused workload, so
        any cols beyond 112 are wasted. The 16 extra cols in 128 sit idle.

    COMPARISON WITH FSRCNN PE COLS SWEEP:
      FSRCNN (activation-heavy, small weights): cols dramatically improve both
        latency and energy because X_7 = 52 spreads across cols, reducing
        DRAM Q iterations. Returns diminish above ~56 cols.
      ResNet18 (weight-heavy, fully cached weights): cols have minimal impact.
        Latency improves only once (2× at 14 cols), then hits the DRAM ceiling.
        Energy drops modestly via fewer WMem reads. The workload is fundamentally
        weight memory bandwidth-bound, not parallelism-bound.

    
Case Study: PE Array Sweep — Aspect Ratio (fixed total PEs=2048) ----- ResNet18 17-layer full fusion
  ResNet18 17-layer full fusion, DepFiN, FMEM=1056KB, WMEM=11264KB

    python3 experiment_runner.py --sweep-arch \
      --workload resnet18 --fusion full --variant 17layer \
      --fmem-sizes 1056 --wmem-sizes 11264 \
      --pe-configs 4x512 8x256 16x128 32x64 64x32 128x16 256x8 512x4 \
      --verbose 2>&1 | tee results/DF_resnet_pe_aspect_ratio_sweep.log

    Results:
      Config  | Energy (uJ) | Latency (cc) | Bottleneck |  WMem Reads   | FMem Reads  | DRAM Reads   | EDP
      --------+-------------+--------------+------------+---------------+-------------+--------------+----------
      4×512   |  11,226     | 31,247,104   | DRAM       |  124,916,736  | 581,464,576 | 11,032,512   | 3.51e+05
      8×256   |  10,993     | 15,623,552   | DRAM       |  124,916,736  | 292,149,760 | 11,032,512   | 1.72e+05
      16×128  |  10,876     |  7,811,776   | DRAM       |  124,916,736  | 147,492,352 | 11,032,512   | 8.50e+04
      32×64   |  10,822     |  7,872,768   | WMem       |  125,970,432  |  75,163,648 | 11,032,512   | 8.52e+04
      64×32   |  10,846     |  8,779,264   | WMem       |  140,464,128  |  38,999,296 | 11,032,512   | 9.52e+04
      128×16  |  11,013     | 11,823,616   | WMem       |  189,041,664  |  25,451,776 | 11,032,512   | 1.30e+05
      256×8   |  11,475     | 19,756,032   | DRAM       |  316,151,808  |  20,935,936 | 11,032,512   | 2.27e+05
      512×4   |  13,920     | 61,453,056   | WMem       |  983,248,896  |  19,806,976 | 11,032,512   | 8.55e+05

    Analysis:
      The tool picks the best tile per config, so
      per-tile latencies are now smaller and total latency reflects natural
      tiling across the spatial dims.

      LATENCY:
        - 4×512 → 8×256 → 16×128: Latency halves each step (31.2M → 15.6M → 7.81M).
          Bottleneck is DRAM. With few rows, Z parallelism is low, requiring more
          temporal passes to stream weights from DRAM.
        - 16×128 → 32×64: Bottleneck SHIFTS from DRAM to WMem (7.81M → 7.87M).
          Latency is essentially flat — the WMem bandwidth ceiling takes over.
        - 64×32 → 128×16: Latency INCREASES (8.78M → 11.8M) because with fewer cols,
          the X-dimension tiling becomes less efficient, requiring more WMem passes.
        - 256×8: Bottleneck shifts BACK to DRAM (19.8M) — with only 8 cols, the
          per-tile DRAM streaming cost dominates again.
        - 512×4: Worst (61.5M) — WMem bottleneck with massive temporal reuse.

      ENERGY:
        - Decreasing trend from 4×512 (11,226) to 32×64 (10,822): −3.6% total.
          Driven by decreasing FMem reads. More rows → more Z parallelism → fewer
          intermediate activation reloads from FMem.
          FMem reads: 581M → 292M → 147M → 75M (halving with each doubling of rows).
        - TURNAROUND from 64×32 onward: Energy INCREASES.
          WMem reads grow: 125M → 140M → 189M → 316M → 983M as fewer cols force
          extra WMem temporal passes. The WMem energy increase overwhelms the
          continued FMem decrease.
        - BEST ENERGY: 32×64 (10,822 uJ).

      DRAM READS: Constant 11,032,512 across all configs (weights + activations
        always read once from DRAM regardless of PE aspect ratio).

      WMem READS: Constant ~125M for 4×512 through 16×128 (identical weight
        streaming), then increases at 32×64 (126M), 64×32 (140M), 128×16 (189M),
        256×8 (316M), 512×4 (983M) due to more temporal weight reloads when cols
        are too few for the X-dimension.

      BEST EDP: 16×128 (8.50e+04) and 32×64 (8.52e+04) — essentially tied.
        This is the sweet spot where latency is near the WMem floor and energy
        is near minimum. The broad optimum spans 16×128 to 32×64.

    COMPARISON WITH FSRCNN PE ASPECT RATIO SWEEP:
      FSRCNN (activation-dominant): Best at 4×512 (col-heavy), then degrades with
        more rows. Q dimension (960) is huge, so more cols = bigger tile = fewer
        DRAM iterations. Z dims are small (12-56), saturated quickly.
      ResNet18 (weight-dominant, deep network): Best EDP at 16×128 to 32×64 (balanced).
        More rows help Z parallelism and reduce FMem reads, but too many rows (> 64)
        cause WMem reads to explode because cols drop below X_i requirements.
        Unlike FSRCNN, there is no strong preference for extreme aspect ratios —
        a balanced configuration works best.

    

    


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
  DepFiN 16×128 PEs, FMEM=1056KB, WMEM=524KB
  FMEM BW scales as: BW_scaled = BW_base × (tile_size / 128)
  Auto-selected tile = max divisor of 1242 ≤ 128 = 69

    python3 experiment_runner.py --sweep-tile-sizes \
    --workload mccnn --fusion full --variant 4layer \
    --tile-sizes 69 46 27 18 9 3 2>&1 | tee mccnn_4layer_tile_sweep.log

    Tried tile sizes: 69 - 46 - 27 - 18 - 9 - 3

    Tile  DRAM Q  Energy(μJ)  Latency(cc)       EDP    PE Util
    ----  ------  ----------  -----------  ---------   --------
    69      18    1.996e+04   1.218e+07    3.69e+05     53.9%
    46      27    2.004e+04   1.827e+07    5.55e+05     35.9%
    27      46    2.019e+04   3.114e+07    9.50e+05     21.1%
    18      69    2.039e+04   4.671e+07    1.43e+06     14.1%
     9     138    2.096e+04   9.343e+07    2.92e+06      7.0%
     3     414    2.325e+04   2.803e+08    9.41e+06      2.3%

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
    Energy: increase (+16.5%, from 19,960 → 23,250 μJ)
    Latency: increase 23× (12.2M → 280M cc), dominated by more DRAM temporal iterations
    EDP: increase 25.5× (dominated by latency)

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


      
Case Study: PE Array Sweep — Increasing PE Rows (fixed cols=64), MC-CNN 4-layer full fusion
  DepFiN, FMEM=1056KB, WMEM=524KB
  Auto tile = max divisor of 1242 ≤ 64 = 54

    python3 experiment_runner.py --sweep-arch \
    --workload mccnn --fusion full --variant 4layer \
    --fmem-sizes 1056 --wmem-sizes 524 \
    --pe-configs 8x64 16x64 32x64 64x64 2>&1

    MC-CNN Z dimensions: Z0=Z1=Z2=Z3=32 (uniform across all layers)

    Config  PEs    Energy(μJ)  Latency(cc)  Bottleneck  FMem Reads      WMem Reads      DRAM Reads
    8×64    512    2.066e+04   3.034e+07    DRAM        6,747,100,416     966,362,112   1,979,712
    16×64   1024   2.000e+04   1.522e+07    DRAM        3,500,572,032     966,362,112   1,979,712
    32×64   2048   1.967e+04   1.518e+07    WMem        1,869,835,968     966,362,112   1,979,712
    64×64   4096   1.967e+04   1.518e+07    WMem        1,869,835,968     966,362,112   1,979,712

    What changes in the mapping:
      SARows Z_i values increase (more output channels parallelized spatially).
      MC-CNN has ALL Z_i = 32, so:
        8 rows:  SARows Z_i = 8  → temporal Z = 32/8  = 4
        16 rows: SARows Z_i = 16 → temporal Z = 32/16 = 2
        32 rows: SARows Z_i = 32 → temporal Z = 32/32 = 1 (SATURATED)
        64 rows: SARows Z_i = 32 → wasted rows (Z already fully covered)

    DRAM reads and WMem reads: CONSTANT across all configs (1.98M and 966M).
    Only FMem reads change: 6.75B → 3.50B → 1.87B → 1.87B (halving with rows, saturates at 32).

    Energy: slight decrease (~4.8%, 20,659 → 19,675 μJ from 8→32 rows), then flat.
      Driven entirely by FMem reads reduction (fewer Z temporal iterations →
      fewer intermediate activation re-reads from FeatureMemory).

    Latency: single 2× step from 8→16 rows, then FLAT.
      8→16:  3.034e+07 → 1.522e+07 cc (2× improvement). Bottleneck is DRAM.
        At 8 rows, Z temporal iterations = 4, causing upstream stalls that
        propagate to DRAM streaming.
      16→32: 1.522e+07 → 1.518e+07 cc (<0.3% improvement). Bottleneck shifts to WMem.
        Z gets fully parallelized (SARows Z=32) but this doesn't help latency
        because the WeightMemory bandwidth ceiling to 16 locks the schedule.
      32→64: IDENTICAL to 32×64. The extra 32 rows are completely wasted.

    KEY INSIGHT: Z=32 saturates at 32 rows. This is even faster saturation than
      FSRCNN (Z_max=56, saturated at ~56 rows). For activation-dominant networks
      with small uniform channels, very few rows are needed.

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
    --fmem-sizes 1056 --wmem-sizes 524 \
    --pe-configs 16x46 16x54 16x69 16x138 16x207 16x414 2>&1

    SACols mapping: Since all strides=1, X_i = Q = 1242 for all layers.
      SACols = min(pe_cols, X_i) = pe_cols (always, since pe_cols ≤ 1242).
      All layers have IDENTICAL spatial mapping (no heterogeneous stages).

    Config   PEs    Tile  DRAM Q  Energy(μJ)  Latency(cc)  Bottleneck  WMem Reads
    16×46     736   46     27     2.004e+04   1.783e+07    DRAM        1,134,425,088
    16×54     864   54     23     2.000e+04   1.522e+07    DRAM          966,362,112
    16×69    1104   69     18     1.996e+04   1.200e+07    DRAM          756,283,392
    16×138   2208  138      9     1.989e+04   6.111e+06    DRAM          378,141,696
    16×207   3312  207      6     1.986e+04   4.150e+06    DRAM          252,094,464
    16×414   6624  414      3     1.984e+04   2.800e+06    DRAM          126,047,232

    What changes in the mapping:
      SACols X_i/Q values increase (larger tile mapped spatially).
      DRAM Q iterations = Q / tile_size = 1242 / pe_cols → decreases.
      FMem reads: CONSTANT at ~3.50B (activation traffic dominates, unchanged by tile).
      DRAM reads: CONSTANT at 1.98M.
      WMem reads: DECREASE proportionally with tile (fewer weight re-reads per DRAM pass).

    Energy: very slight decrease (−1.0% total, 20,036 → 19,837 μJ over 46→414 cols).
      Energy is overwhelmingly dominated by FMem reads (intermediates), which don't change.
      The WMem energy saving from fewer weight re-reads is marginal (~200 μJ over 9× range).

    Latency: monotonic decrease, proportional to 1/DRAM_Q.
      46→414: 1.78e+07 → 2.80e+06 cc (6.4× improvement). Bottleneck is ALWAYS DRAM.
      Each halving of DRAM Q iterations roughly halves latency.



Case Study: PE Array Sweep — Aspect Ratio (fixed total PEs=2048) ----- MC-CNN 4-layer full fusion
  DepFiN, FMEM=1056KB, WMEM=524KB

    python3 experiment_runner.py --sweep-arch \
    --workload mccnn --fusion full --variant 4layer \
    --fmem-sizes 1056 --wmem-sizes 524 \
    --pe-configs 2x1024 4x512 8x256 16x128 32x64 64x32 2>&1

    Auto-selected tile sizes and DRAM Q iterations:
      Config  Tile  DRAM Q  SARows Z_i  Temporal Z
      2×1024  621     2     2           16    ← severe Z under-parallelism
      4×512   414     3     4            8
      8×256   207     6     8            4
      16×128   69    18    16            2
      32×64    54    23    32            1    ← Z fully covered
      64×32    27    46    32            1    ← same Z, but tile too small

    Results:
      Config  PEs    Energy(μJ)  Latency(cc)  Bottleneck  FMem Reads       WMem Reads       DRAM Reads    EDP
      2×1024  2048   2.442e+04   1.081e+07    DRAM        26,315,933,184      84,031,488   1,979,712    3.76e+05
      4×512   2048   2.181e+04   8.100e+06    DRAM        13,284,988,416     126,047,232   1,979,712    2.60e+05
      8×256   2048   2.052e+04   8.075e+06    DRAM         6,762,044,160     252,094,464   1,979,712    2.49e+05  ← BEST EDP
      16×128  2048   1.996e+04   1.200e+07    DRAM         3,500,572,032     756,283,392   1,979,712    3.62e+05
      32×64   2048   1.967e+04   1.518e+07    WMem         1,869,835,968     966,362,112   1,979,712    4.56e+05
      64×32   2048   1.987e+04   3.021e+07    WMem         1,869,835,968   1,932,724,224   1,979,712    9.12e+05

    ANALYSIS — Trade-off between rows (Z parallelism) and cols (Q tile size):

      Energy: monotonic decrease from 2×1024 (24,423) to 32×64 (19,675), then
        TURNAROUND at 64×32 (19,866 μJ).
        The decrease is driven by FMem reads reduction:
          2×1024: FMem reads = 26.3B (Z has 16 temporal iterations → massive act re-reads)
          4×512:  FMem reads = 13.3B (halved)
          8×256:  FMem reads = 6.8B  (halved)
          16×128: FMem reads = 3.5B  (halved)
          32×64:  FMem reads = 1.87B (Z fully covered, minimum FMem reads)
          64×32:  FMem reads = 1.87B (same — extra rows wasted)
        The turnaround at 64×32 comes from WMem reads doubling (966M → 1,933M)
        due to smaller tile (27 vs 54) causing more weight re-reads.

      Latency: V-shaped with minimum at 4×512 and 8×256 (~8.1M cc).
        Too few rows (2×1024): Tile=621 (DRAM Q=2), but Z needs 16 temporal iterations.
          The massive Z loop at each PE causes upstream stalls → 10.8M cc.
        Sweet spot (4×512, 8×256): Good balance. DRAM Q stays small (3–6), and
          Z temporal iterations (8–4) are manageable. Bottleneck = DRAM.
        Too few cols (16×128+): Tile shrinks rapidly. DRAM Q = 18–46.
          At 32×64, Z is saturated but DRAM Q=23 → bottleneck shifts to WMem (15.2M cc).
          At 64×32, DRAM Q=46 → WMem latency doubles (30.2M cc).

      EDP: 
        Best at 8×256 (2.49e+05): good balance of Z parallelism and tile coverage.
        8×256 beats 4×512 on energy (less FMem) while matching on latency.
        The col-heavy configs (2×1024, 4×512) have higher energy from FMem reads.
        The row-heavy configs (32×64, 64×32) have exploding latency from small tiles.

    COMPARISON WITH FSRCNN PE ASPECT RATIO SWEEP:
      FSRCNN: Best EDP at 4×512. Q=960, Z_max=56. Column-heavy always preferred.
      MC-CNN: Best EDP at 8×256. Q=1242, Z=32. Also column-heavy preferred.
      Both activation-dominant workloads favor column-heavy aspect ratios but with
      slightly different optimal points due to Z dimension differences:
        - FSRCNN: Z_max=56, so even 4 rows leave 14× Z temporal iterations for L6.
          Best at 4×512 because tile=480 (DRAM Q=2) with acceptable Z overhead.
        - MC-CNN: Z=32 (uniform), so 8 rows give only 4× Z iterations (manageable).
          Best at 8×256 where tile=207 (DRAM Q=6) provides a better energy/latency trade-off.
      Conclusion: For activation-dominant workloads, column-heavy ratios win.
      The optimal #rows ≈ sqrt(Z_max) or Z_max/4, enough to limit Z temporal
      iterations without sacrificing too much tile coverage. Q dimension (1242 here,
      960 for FSRCNN) is the dominant parallelism axis for latency.


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
  DepFiN 16×128 PEs, FMEM=1056KB, WMEM=14400KB
  FMEM BW scales as: BW_scaled = BW_base × (tile_size / 128)
  NOTE: WMEM = 14400KB to fit all 13 layers' weights (14,710,464 total).

    python3 experiment_runner.py --sweep-tile-sizes \
    --workload vgg16 --fusion full --variant 13layer \
    --wmem-size 14400 --fmem-size 1056 --pe-rows 16 --pe-cols 128 \
    --tile-sizes 7 2 1 2>&1

    Q=14, divisors={1,2,7,14}

    Tile  | DRAM Q iters | Energy(μJ) | Latency(cc) | EDP      | WMem Reads
    14    | 1            | 7.766e+04  | 2.731e+07   | 3.84e+06 | 388,878,696
    7     | 2            | 7.758e+04  | 5.024e+07   | 7.11e+06 | 669,634,560
    2     | 7            | 8.380e+04  | 1.712e+08   | 2.52e+07 | 2,116,903,680   
    1     | 14           | 9.429e+04  | 3.516e+08   | 5.56e+07 | 4,687,441,920

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

    EDP: 14.5× worse (3.84e+06 → 5.56e+07).

    COMPARISON WITH RESNET18 TILE SWEEP:
      ResNet18: tile=7 vs tile=1, 7× latency increase, +25% energy.
      VGG16: tile=7 vs tile=1, 7× latency increase, +21.5% energy.
      Nearly identical pattern — both are weight-dominant, stride=16, fully cached.
      The slight energy difference comes from VGG16's larger weight volume (14.7M vs 11M).
      Key insight: For weight-dominant workloads, tile=Q (full width) always best.
      

Case Study: PE Array Sweep — Increasing PE Rows (fixed cols=128) ----- VGG16 13-layer full fusion
  DepFiN, FMEM=1056KB, WMEM=14400KB, Tile Size 14

    python3 experiment_runner.py --sweep-arch \
    --workload vgg16 --fusion full --variant 13layer \
    --fmem-sizes 1056 --wmem-sizes 14400 \
    --pe-configs 8x128 16x128 32x128 64x128 128x128 256x128 512x128 2>&1

    VGG16 Z dimensions: Z0-Z1=64, Z2-Z3=128, Z4-Z6=256, Z7-Z12=512

    Config   | PEs    | FMem Reads    | WMem Reads  | Energy(μJ) | Latency(cc) | EDP
    8×128    | 1,024  | 1,931,876,352 | 388,878,336 | 7.72e+04   | 4.86e+07    | 6.86e+06
    16×128   | 2,048  | 972,711,936   | 388,878,336 | 7.64e+04   | 2.43e+07    | 3.41e+06
    32×128   | 4,096  | 493,129,728   | 388,878,336 | 7.60e+04   | 2.43e+07    | 3.40e+06
    64×128   | 8,192  | 253,338,624   | 388,878,336 | 7.58e+04   | 2.43e+07    | 3.40e+06
    128×128  | 16,384 | 148,571,136   | 388,878,336 | 7.57e+04   | 2.43e+07    | 3.39e+06
    256×128  | 32,768 | 107,025,408   | 388,878,336 | 7.57e+04   | 2.43e+07    | 3.39e+06
    512×128  | 65,536 | 95,284,224    | 388,878,336 | 7.57e+04   | 2.43e+07    | 3.39e+06

    DRAM reads: CONSTANT at 14,860,992 across all configs.
    WMem reads: CONSTANT at 388,878,336 across all configs.
    FMem reads: Only FMem changes (halving pattern with doubling rows):
      8→16: 1,932M → 973M (halved)
      16→32: 973M → 493M (halved)
      32→64: 493M → 253M (halved)
      64→128: 253M → 149M
      128→256: 149M → 107M
      256→512: 107M → 95M (near minimum)

    Latency: Single 2× step (8→16 rows), then 2.43e+07 cc.
      4.86e+07 → 2.43e+07 (2× improvement at 8→16 rows).
      16→512: Latency CONSTANT. Bottleneck is WeightMemory bandwidth.
      Adding rows doesn't affect DRAM or WMem streaming schedules.

    Energy: Tiny decrease (~2% total, from 77,200 → 75,700 μJ over 8→512 rows).
      Driven by decreasing FMem reads. More rows → more Z parallelism → fewer
      temporal Z iterations → fewer re-reads of input activations from FMEM.

    IDENTICAL PATTERN TO RESNET18:
      ResNet18: single 2× latency step at 8→16, then flat. Same FMem halving.
      The weight-dominant characteristic dominates: once DRAM and WMem are the
      bottleneck (at ≥16 rows), row parallelism is useless for latency.
      Energy improvement is marginal because FMem is a tiny fraction of total.


Case Study: PE Array Sweep — Increasing PE Cols (fixed rows=16) ----- VGG16 13-layer full fusion
  VGG16 13-layer full fusion, DepFiN, FMEM=1056KB, WMEM=14400KB

    python3 experiment_runner.py --sweep-arch \
      --workload vgg16 --fusion full --variant 13layer \
      --fmem-sizes 1056 --wmem-sizes 14400 \
      --pe-configs 16x7 16x14 16x28 16x56 16x112 16x128 \
      --verbose 2>&1

    SACols mapping (cols map X_i/Q spatial dims):
      VGG16 spatial dims per stage group with tile=7 (output→input):
        Q   = 7   (output, tile of L12 Q=14)
        X10 = 7   (L10-L12, Q=14, tile=7)
        X7  = 14  (L7-L9, Q=28, tile=14 after pool4 stride=2)
        X4  = 28  (L4-L6, Q=56, tile=28 after pool3)
        X2  = 56  (L2-L3, Q=112, tile=56 after pool2)
        X0  = 112 (L0-L1, Q=224, tile=112 after pool1)

      SACols allocation = min(pe_cols, X_i) for each stage:
      Cols  |  Q  | X10 | X7  | X4  | X2  | X0
      ------+-----+-----+-----+-----+-----+------
        7   |  7  |  7  |  7  |  7  |   7 |   7
       14   |  7  |  7  | 14  | 14  |  14 |  14    ← X7 fully covered
       28   |  7  |  7  | 14  | 28  |  28 |  28    ← X4 fully covered
       56   |  7  |  7  | 14  | 28  |  56 |  56    ← X2 fully covered
      112   |  7  |  7  | 14  | 28  |  56 | 112    ← X0 fully covered (all stages full)
      128   |  7  |  7  | 14  | 28  |  56 | 112    ← same as 112 (X0 max=112)

    Results:
      Cols | Energy(μJ) | Latency(cc) | WMem Reads    | FMem Reads    | DRAM Reads
      -----+------------+-------------+---------------+---------------+------------
        7  | 83,900     | 1.37e+08    | 2,192,375,808 | 972,711,936   | 14,860,992
       14  | 79,400     | 6.86e+07    | 1,096,187,904 | 972,711,936   | 14,860,992
       28  | 77,300     | 3.74e+07    | 597,639,168   | 972,711,936   | 14,860,992
       56  | 76,600     | 2.70e+07    | 430,940,160   | 972,711,936   | 14,860,992
      112  | 76,400     | 2.43e+07    | 388,878,336   | 972,711,936   | 14,860,992
      128  | 76,400     | 2.43e+07    | 388,878,336   | 972,711,936   | 14,860,992

    Analysis:
      DRAM reads: CONSTANT (14.86M). FMem reads: CONSTANT (972.7M).
      Only WMem reads change — cols affect spatial coverage, reducing temporal
      weight re-reads as more of each layer's spatial dimension fits in SACols.

      LATENCY: Multi-step improvement, NOT a single jump like ResNet18.
        7→14:   2.00× improvement (137M → 68.6M cc)
        14→28:  1.83× improvement (68.6M → 37.4M cc)
        28→56:  1.39× improvement (37.4M → 27.0M cc)
        56→112: 1.11× improvement (27.0M → 24.3M cc)
        112→128: identical (X0=112 fully covered, 16 extra cols idle)

      WMem reads: Decreasing as more spatial coverage reduces weight re-reads.
        7→14: 2,192M → 1,096M (halved, X7 fully covered)
        14→28: 1,096M → 598M (halved, X4 fully covered)
        28→56: 598M → 431M (−28%, X2 fully covered)
        56→112: 431M → 389M (−10%, X0 fully covered)
        112→128: identical

      ENERGY: Gradual decrease driven by WMem reads.
        7→128: 83,900 → 76,400 μJ (−8.9%)
        Breakdown: 7→14: −5.4%, 14→28: −2.6%, 28→56: −0.9%, 56→128: −0.3%

      SATURATION AT 112 COLS: X0=112 is the largest spatial dim across all layers.
        Beyond 112 cols, extra PEs are idle. 128−112 = 16 wasted columns.

    KEY DIFFERENCE FROM RESNET18:
      ResNet18: Single 2× latency step at 7→14, then FLAT from 14+.
        ResNet18 has fewer layers with a more uniform weight distribution,
        so once X13=14 is fully covered, the bottleneck immediately shifts
        to DRAM, locking latency.
      VGG16: Multi-step improvement through 7→14→28→56→112.
        VGG16 has 13 layers across 5 spatial sizes (14, 28, 56, 112, 224),
        each contributing significant weight volume. Covering each stage
        progressively reduces WMem re-reads and therefore latency.
        The bottleneck remains DRAM throughout, but the per-layer DRAM
        scheduling improves as more spatial coverage reduces stalls.


Case Study: PE Array Sweep — Aspect Ratio (fixed total PEs=2048) ----- VGG16 13-layer full fusion
  VGG16 13-layer full fusion, DepFiN, FMEM=1056KB, WMEM=14400KB

    python3 experiment_runner.py --sweep-arch \
      --workload vgg16 --fusion full --variant 13layer \
      --fmem-sizes 1056 --wmem-sizes 14400 \
      --pe-configs 4x512 8x256 16x128 32x64 64x32 128x16 256x8 512x4 \
      --verbose 2>&1 | tee results/DF_vgg16_pe_aspect_ratio_sweep.log

    Results:
      Config  | Energy(μJ) | Latency(cc) | Bottleneck | FMem Reads    | WMem Reads    | DRAM Reads  | EDP
      --------+------------+-------------+------------+---------------+---------------+-------------+-----------
      4×512   |  78,700    | 95,107,328  | DRAM       | 3,850,205,184 |   380,233,728 | 14,860,992  | 1.36e+07
      8×256   |  77,100    | 47,553,664  | DRAM       | 1,931,876,352 |   380,233,728 | 14,860,992  | 6.71e+06
      16×128  |  76,400    | 24,338,432  | DRAM       |   972,711,936 |   388,878,336 | 14,860,992  | 3.41e+06
      32×64   |  76,200    | 26,926,080  | DRAM       |   493,129,728 |   430,940,160 | 14,860,992  | 3.77e+06
      64×32   |  76,700    | 36,831,232  | DRAM       |   253,338,624 |   588,994,560 | 14,860,992  | 5.17e+06
      128×16  |  78,500    | 65,917,952  | DRAM       |   148,571,136 | 1,054,126,080 | 14,860,992  | 9.38e+06
      256×8   |  82,500    |126,976,000  | DRAM       |   107,025,408 | 2,025,676,800 | 14,860,992  | 1.85e+07
      512×4   |  91,500    |261,005,312  | DRAM       |    95,284,224 | 4,183,474,176 | 14,860,992  | 4.06e+07

    Analysis:
      NOTE: Unlike the earlier tile_size=1 run, this uses unconstrained tiling.
      The tool auto-detects tile_size=14 (max divisor of Q=14 ≤ 128), so all
      spatial dims are processed in a single tile column. This dramatically
      reduces WMem reads compared to tile=1 (which forced 14× more iterations).

      BOTTLENECK: ALL configs are DRAM-bottlenecked (unlike tile=1 where WMem
      became the bottleneck from 16×128 onward). With tile=14, the weight
      temporal reuse is much better, so DRAM weight-loading dominates.

      LATENCY:
        - 4×512 → 8×256 → 16×128: Latency halves each step (95.1M → 47.6M → 24.3M).
          Fewer rows = fewer Z parallelism = more DRAM temporal weight passes.
        - 16×128 → 32×64: Latency INCREASES slightly (24.3M → 26.9M, +10.6%).
          With 64 cols, some X_i dimensions require extra temporal passes.
        - 32×64 → 64×32 → 128×16 → 256×8 → 512×4: Latency keeps increasing
          (26.9M → 36.8M → 65.9M → 127M → 261M) because with fewer cols, DRAM
          weight streaming requires more temporal passes per layer.
        - BEST LATENCY: 16×128 (24,338,432 cc).

      ENERGY:
        - Decreasing from 4×512 (78,700) to 32×64 (76,200 μJ): −3.2%.
          Driven by decreasing FMem reads (3,850M → 493M) as more rows provide
          more Z parallelism and fewer intermediate activation re-reads.
        - TURNAROUND from 64×32 onward: Energy INCREASES.
          WMem reads grow: 381M → 431M → 589M → 1,054M → 2,026M → 4,183M.
          The WMem energy increase overwhelms the continued FMem decrease.
        - BEST ENERGY: 32×64 (76,200 μJ).

      DRAM READS: CONSTANT at 14,860,992 across all configs (weights + activations
        always read once from DRAM regardless of PE aspect ratio).

      FMem READS: Halving pattern with rows (3,850M → 95M).
        More rows → more Z parallelism → fewer re-reads of intermediate activations.

      WMem READS: Nearly constant ~380M for 4×512 through 8×256, then increases
        steadily: 389M (16×128) → 431M (32×64) → 589M (64×32) → 1,054M (128×16)
        → 2,026M (256×8) → 4,183M (512×4). Fewer cols mean WMem can't fit enough
        of the X-dimension spatially, requiring temporal weight re-reads.

      BEST EDP: 16×128 (3.41e+06) — best latency with near-best energy.
        Close second: 32×64 (3.77e+06) — best energy but +10% latency.

    COMPARISON WITH RESNET18 PE ASPECT SWEEP:
      Both are weight-dominant networks with stride=16 and similar structure.
      ResNet18 (unconstrained tile): Best EDP at 16×128 / 32×64 (balanced).
        Bottleneck shifts from DRAM (few rows) to WMem (many rows).
      VGG16 (unconstrained tile=14): Best EDP at 16×128, all configs DRAM-bottlenecked.
        VGG16 has 33% more weights (14.7M vs 11M) and deeper Z channels (up to 512),
        so the DRAM weight-loading pressure is even stronger. The WMem reads
        pattern is similar but amplified (4,183M at 512×4 vs 983M for ResNet18).
        Both share the same optimal region: 16×128 to 32×64.

  COMBINED ANALYSIS — All four DepFiN sweeps for VGG16:

    VGG16 13-layer full fusion is archetypal WEIGHT-DOMINANT:
    Total weights: 14.7M params (33% more than ResNet18's 11M).
    Total input activations: 150K (tiny: 3×224×224 = 150,528).
    Weight-to-activation ratio: ~97.7× (ResNet18: ~92×).

    Key findings across all 4 case studies:
    1. TILE SIZE: Binary choice (7 vs 1), tile=7 always wins.
       No intermediate tiles available due to stride=16 constraint.
       Weight re-reads scale linearly with DRAM Q iterations.

    2. PE ROWS: Latency halves at each row doubling (4→8→16) while DRAM-bound.
       FMem reads halve with rows but contribute <2% of total energy.

    3. PE COLS: Multi-step latency improvement (unlike ResNet18's single step).
       This is VGG16's distinguishing feature: 13 layers across 5 spatial sizes
       (14, 28, 56, 112, 224) mean that each col increase progressively covers
       more stage groups, reducing weight re-reads across more layers.
       Saturates at 112 cols (X0 = 112).

    4. ASPECT RATIO: Best EDP at 16×128 (col-balanced), best energy at 32×64.
       Energy turnaround at 64×32 from WMem read explosion.
       All configs are DRAM-bottlenecked (unlike tile=1 where WMem dominated).

    COMPARISON WITH RESNET18:
      Nearly identical behavior in all sweeps except PE cols.
      Both are weight-dominant with stride=16 and 2 valid tiles.
      The PE cols difference (multi-step vs single-step latency) reflects
      VGG16's deeper 5-block structure vs ResNet18's fewer distinct spatial sizes.

    COMPARISON WITH FSRCNN/MC-CNN (activation-dominant):
      Completely opposite trade-offs:
        - FSRCNN/MC-CNN: Tile size sweep shows gradual degradation (many tiles).
          VGG16: Binary choice, 7× latency cliff.
        - FSRCNN/MC-CNN: PE cols are the dominant parallelism axis (large Q).
          VGG16: PE cols have limited impact (small Q=14 after stride=16).
        - FSRCNN/MC-CNN: Column-heavy aspect ratios win (4×512 best for FSRCNN).
          VGG16: Balanced to slightly col-heavy wins (16×128 best EDP).
        - FSRCNN/MC-CNN: FMem reads dominate energy (~billions per tile).
          VGG16: WMem reads dominate energy (weight-reload penalty).


================================================================================
EYERISS CASE STUDIES — ResNet18 17-layer full fusion
================================================================================

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



Eyeriss Case Study 1: WRegister Size Sweep (400–15000 entries)
  ResNet18 17-layer full fusion, Eyeriss, GB=128KB, tile_size=1
  Non-swept regs: InReg=700, IntReg=300, OutReg=64 (generous, non-binding)

    python3 experiment_runner.py --sweep-wreg-pe \
    -w resnet18 -f full -v 17layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 700 --intermediate-reg 300 --output-reg 64 \
    --weight-reg-sizes 400 1000 2000 4000 8000 15000

    SUMMARY (A) — Minimum PEs Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      400     512×128      65,536      Yes        7.320e+03    3.042e+06    2.28e+04
      1000    256×64       16,384      Yes        6.944e+03    3.042e+06    2.15e+04
      2000    256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04
      4000    256×16        4,096      Yes        6.697e+03    3.059e+06    2.06e+04  ← saturation
      8000    256×16        4,096      Yes        6.697e+03    3.059e+06    2.06e+04
      15000   256×16        4,096      Yes        6.697e+03    3.059e+06    2.06e+04

    SUMMARY (B) — Minimum Latency Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      400     512×128      65,536      Yes        7.320e+03    3.042e+06    2.28e+04
      1000    256×64       16,384      Yes        6.944e+03    3.042e+06    2.15e+04
      2000    256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04  ← saturation
      4000    256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04
      8000    256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04
      15000   256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04

    ANALYSIS — WReg is the dominant binding constraint:
      WReg footprint = Σ(17 layers) of residual C×RxSxZ factors not absorbed by SARows.
      At pe_rows=256: footprint = 3583 entries. At pe_rows=512: 1795 entries.
      SACols Z also helps: more cols → Z split across column groups → fewer
      weight iterations per PE. This is why small WReg needs more cols.

      WReg=400: Only 512×128 (65,536 PEs!) works. Even 512 rows leave footprint=1795 > 400,
        so SACols must take Z=128 to divide it further → needs 128 cols.
        Cost: 16× more PEs than the saturated point.

      WReg=1000: 256×64 (16,384 PEs). Footprint at 256 rows = 3583, but 64 cols
        split Z → effective footprint ≈ 3583/4 ≈ 896 < 1000. Only needs 4× over-provisioning.

      WReg=2000: 256×32 (8,192 PEs). 3583/2 ≈ 1792 < 2000. 2× cols suffice.

      WReg=4000: 256×16 (4,096 PEs). 3583 < 4000 fits with just pe_cols=16.
        This is the SATURATION POINT: beyond 4000, min PEs stays at 4096.

      Conclusion: Doubling WReg from 400→4000 (10×) reduces min PEs by 16× (65K→4K).
        Each doubling of WReg halves the required pe_cols (128→64→32→16).
        WReg ≥ 4000 entries is the design sweet spot for ResNet18 17-layer fusion.

    Latency behavior:
      All configs achieve similar latency (~3.04–3.06M cc). The 256×16 config is
      marginally slower (3.059M vs 3.042M cc, +0.6%) because fewer cols provide
      slightly less Z parallelism in SACols. But the difference is negligible.
      Min-latency saturates at 256×32 (8,192 PEs) for WReg ≥ 2000.

    Energy: Monotonic decrease 7320→6697 μJ as PEs decrease (−8.5% total).
      Fewer PEs → fewer idle PE energy → lower total. Energy saturates at WReg=4000.

    Contrast with FSRCNN: FSRCNN WReg footprint at pe_rows=84 was only 258 entries,
      so WReg=300 was sufficient. ResNet18 needs WReg ≥ 4000 — a 13× increase —
      reflecting the 17-layer depth (vs 8) and wider channels (Z up to 512 vs 56).


Eyeriss Case Study 2: IntermediateRegister Size Sweep (50–400 entries)
  ResNet18 17-layer full fusion, Eyeriss, GB=128KB, tile_size=1
  Non-swept regs: InReg=700, WReg=4000, OutReg=64

    python3 experiment_runner.py --sweep-intreg-pe \
    -w resnet18 -f full -v 17layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 700 --weight-reg 4000 --output-reg 64 \
    --intermediate-reg-sizes 50 100 200 400

    SUMMARY (A) — Minimum PEs Configuration:
      IntReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      50      256×128      32,768      Yes        7.113e+03    3.042e+06    2.22e+04
      100     256×64       16,384      Yes        6.944e+03    3.042e+06    2.15e+04
      200     256×32        8,192      Yes        6.779e+03    3.042e+06    2.08e+04
      400     256×16        4,096      Yes        6.697e+03    3.059e+06    2.06e+04  ← saturation
     1000     256×16        4,096      Yes        6.697e+03    3.059e+06    2.06e+04
      
    SUMMARY (B) — Minimum Latency Configuration:
      IntReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      50      256×128      32,768      Yes        7.113e+03    3.042e+06    2.22e+04
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
  Non-swept regs: InReg=700, WReg=4000, IntReg=300

    python3 experiment_runner.py --sweep-outreg-pe \
    -w resnet18 -f full -v 17layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 700 --weight-reg 4000 --intermediate-reg 300 \
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
    WReg alone determines the PE count when it's undersized (e.g. WReg=400 → 65K PEs),
    because its footprint is 17× larger than IntReg's.

    Comparison with FSRCNN:
      FSRCNN: WReg ≥ 300 sufficient, IntReg/OutReg non-binding. Min PEs = 336 (84×4).
      ResNet18: WReg ≥ 4000, IntReg ≥ 210, OutReg ≥ 32. Min PEs = 4,096 (256×16).
      The 12× PE increase reflects deeper fusion (17 vs 8 layers) and wider channels
      (Z up to 512 vs 56). Register requirements scale with network depth and width.


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

  PE grid searched: rows=[84,128,196,256,324,512], cols=[4,8,16,32,64,128]
    36 combos per register size.

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
  Non-swept regs: InReg=750, IntReg=300, OutReg=64 (generous, non-binding)

    python3 experiment_runner.py --sweep-wreg-pe \
    -w vgg16 -f full -v 13layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 750 --intermediate-reg 300 --output-reg 64 \
    --weight-reg-sizes 500 1000 2500 5000 10000 20000 \
    --pe-rows-grid 84 128 196 256 324 512 \
    --pe-cols-grid 4 8 16 32 64 128

    SUMMARY (A) — Minimum PEs Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      500     512×128      65,536      Yes        4.496e+04    5.455e+06    2.51e+05
      1000    196×128      25,088      Yes        4.357e+04    5.455e+06    2.43e+05
      2500    196×32        6,272      Yes        4.158e+04    5.681e+06    2.38e+05
      5000    196×16        3,136      Yes        4.114e+04    6.715e+06    2.77e+05
      10000   128×16        2,048      Yes        3.982e+04    1.103e+07    4.41e+05
      20000    84×16        1,344      Yes        3.916e+04    2.010e+07    7.90e+05

    SUMMARY (B) — Minimum Latency Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      500     512×128      65,536      Yes        4.496e+04    5.455e+06    2.51e+05
      1000    196×128      25,088      Yes        4.357e+04    5.455e+06    2.43e+05
      2500    196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05
      5000    196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05  ← sat.
      10000   196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05
      20000   196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05

    ANALYSIS — WReg is the dominant binding constraint:
      WReg footprint = Σ(13 layers) of residual C×R factors not absorbed by SARows.
      At pe_rows=196: footprint = 4,791 at pe_cols=16, or 2,397 at cols=32,
      1,200 at cols=64, 603 at cols=128.
      SACols Z helps: more cols → Z split across column groups → fewer
      weight iterations per PE.

      WReg=500: Only 512×128 (65,536 PEs!) works. Even 512 rows leave footprint=603
        at cols=128. Need maximum cols to divide further → needs 128 cols.
        Cost: 21× more PEs than the minimum-PEs saturation point.

      WReg=1000: 196×128 (25,088 PEs). Footprint at 196 rows, 128 cols = 603 < 1000. ✓
        Still needs 128 cols but rows drop from 512 to 196 (rows saturate there).

      WReg=2500: 196×32 (6,272 PEs). Footprint 2,397 < 2500 at 32 cols. ✓
        4× fewer PEs than WReg=1000. Each cols halving needs ~2× WReg to compensate.

      WReg=5000: 196×16 (3,136 PEs). Footprint 4,791 < 5000. ✓
        This is the min PEs saturation for 196+ rows: beyond 5000, the min-PEs
        config stays at 196×16 because IntReg/OutReg allow pe_cols=16.

      WReg=10000: 128×16 (2,048 PEs). Now pe_rows can drop below 196 because
        footprint at 128×16 = 9,579 < 10,000. Smaller rows → fewer PEs but
        latency degrades (1.103e+07 cc, 2× worse).

      WReg=20000: 84×16 (1,344 PEs). Footprint at 84×16 = 19,155 < 20,000. ✓
        Minimum achievable PEs, but latency = 2.01e+07 cc (3.7× worse).

    Latency behavior:
      Min-latency saturates at 196×64 = 12,544 PEs for WReg ≥ 2500.
      All 196×64 configs achieve identical 5.455e+06 cc.
      Below WReg=2500: need 128+ cols, constraining to 196×128 or 512×128.
      Min-PEs configs (128×16, 84×16) have 2–3.7× worse latency because
      fewer rows → more temporal Z iterations → slower DRAM streaming.

    Energy: Monotonic decrease 44,960 → 39,160 μJ as PEs decrease (−12.9%).
      Fewer PEs → fewer idle PE energy. But min-latency configs (196×64)
      use more PEs (12,544) and have higher energy (42,450 μJ) — the EDP
      trade-off favors the min-latency config (EDP 2.35e+05 vs 2.77e+05).

    KEY COMPARISON WITH RESNET18 WReg SWEEP:
      ResNet18: WReg=4000 saturates at 256×16 = 4,096 PEs.
      VGG16: WReg=5000 saturates at 196×16 = 3,136 PEs.
      VGG16 needs MORE WReg to saturate (5000 vs 4000, +25%) because its
      per-PE weight footprint is 33% larger. But the min PEs is actually
      FEWER (3,136 vs 4,096) because VGG16 saturates at 196 rows (not 256).
      This reflects VGG16's layer structure: fewer layers × larger weights per layer
      vs ResNet18's more layers × projection shortcuts that increase row demand.


Eyeriss Case Study 2: IntermediateRegister Size Sweep (50–500 entries)
  VGG16 13-layer full fusion, Eyeriss, GB=128KB, tile_size=1
  Non-swept regs: InReg=750, WReg=5000, OutReg=64

    python3 experiment_runner.py --sweep-intreg-pe \
    -w vgg16 -f full -v 13layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 750 --weight-reg 5000 --output-reg 64 \
    --intermediate-reg-sizes 50 100 230 500 \
    --pe-rows-grid 84 128 196 256 324 512 \
    --pe-cols-grid 4 8 16 32 64 128

    SUMMARY (A) — Minimum PEs Configuration:
      IntReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      50       84×128      10,752      Yes        4.159e+04    5.681e+06    2.42e+05
      100      84×64        5,376      Yes        4.048e+04    6.715e+06    2.76e+05
      230     196×16        3,136      Yes        4.114e+04    6.715e+06    2.77e+05  ← sat.
      500     196×16        3,136      Yes        4.114e+04    6.715e+06    2.77e+05

    SUMMARY (B) — Minimum Latency Configuration:
      IntReg  PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      50      128×128      16,384      Yes        4.225e+04    5.455e+06    2.36e+05
      100     196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05
      230     196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05  ← sat.
      500     196×64       12,544      Yes        4.245e+04    5.455e+06    2.35e+05

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
  Non-swept regs: InReg=750, WReg=5000, IntReg=300

    python3 experiment_runner.py --sweep-outreg-pe \
    -w vgg16 -f full -v 13layer \
    --tile-size 1 --gb-size 128 \
    --input-reg 750 --weight-reg 5000 --intermediate-reg 300 \
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

  PE grid searched: rows=[4,8,12,14,16,28,32,56,84], cols=[4,8,16,32,64,69,128,138]
    72 combos per register size.

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
  InReg=200, IntReg=100, OutReg=64 (generous, non-binding)

    python3 experiment_runner.py --sweep-wreg-pe \
    -w mccnn -f full -v 4layer \
    --tile-size 69 --gb-size 128 \
    --input-reg 200 --intermediate-reg 100 --output-reg 64 \
    --weight-reg-sizes 100 200 300 600 \
    --pe-rows-grid 4 8 12 14 16 28 32 56 84 128 \
    --pe-cols-grid 4 8 16 32 64 69 128 138

    SUMMARY (A) — Minimum PEs Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      100     128x16       2048        No         3.828e+04    1.223e+07    4.70e+05
      200     56×16        896         No         3.635e+04    2.344e+07    8.56e+05
      300     28×16        448         No         3.539e+04    4.585e+07    1.63e+06
      600     12×16        192         No         3.488e+04    9.068e+07    3.18e+06

    SUMMARY (B) — Minimum Latency Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      100     128x16       2048        No         3.828e+04    1.223e+07    4.70e+05
      200     56×16        896         No         3.635e+04    2.344e+07    8.56e+05
      300     56×64        3584        No         3.617e+04    1.223e+07    4.43e+05
      600     56×64        3584        No         3.617e+04    1.223e+07    4.43e+05

    ANALYSIS — WReg is the binding constraint (same as FSRCNN):
      WReg footprint = Σ(4 layers) of residual C×R factors not absorbed by SARows.
      At pe_rows=128: footprint=75 entries at the pe_cols 16.
      At pe_rows=84: footprint=147 at pe_cols=16, or 291 at pe_cols=8, or 582 at pe_cols=4.
      At pe_rows=8: footprint=1164 at pe_cols=16.

      WReg=100: Min PEs = 128×16 = 2048 PEs.
        At 128 rows, footprint=75 < 100 at pe_cols=16. ✓
        Smaller rows (84, 56) give larger footprints (147, 291) which don't fit. 
        The search finds 128×16 as the smallest feasible.

      WReg=200: Min PEs = 56×16 = 896 PEs.
        At 56 rows, footprint ≈ 147–291 range. pe_cols=16 gives footprint=147 < 200. ✓
        Smaller rows (28, 32) give larger footprints (291) which still fit, but more
        rows needed. The search finds 56×16 as the smallest feasible.

      WReg=300: Min PEs = 28×16 = 448 PEs.
        At 28 rows, pe_cols=16 gives footprint=291 (from interpolation) < 300. ✓
        This is 2× fewer PEs than WReg=200. Each halving of rows doubles footprint,
        but WReg=300 can absorb that.

      WReg=600: Min PEs = 12×16 = 192 PEs.
        At 12 rows, pe_cols=16 gives footprint ≈ 582 < 600. ✓
        This is a very lean configuration — only 192 PEs for a 4-layer fused network.

    Latency behavior:
      Min-latency saturates at 56×64 = 3584 PEs for WReg ≥ 300.
      At WReg=200, only 56×16 is feasible (not enough WReg for more cols at 56 rows
      because pe_cols=32/64 would change the SACols Z allocation and footprint).
      Latency = 1.223e+07 cc at 56×64 vs 2.344e+07 at 56×16 (1.9× improvement).

    Energy: Lower PEs → slightly lower energy (36,350→34,880 μJ, −4%).
      The bigger array (56×64) has more idle PEs contributing static energy.
      But the trade-off with latency makes 56×64 the better EDP choice.


Eyeriss Case Study 2: IntermediateRegister Size Sweep (15–100 entries)
  MC-CNN 4-layer full fusion, Eyeriss, GB=128KB, tile_size=69
  Non-swept regs: InReg=200, WReg=600, OutReg=64

    python3 experiment_runner.py --sweep-intreg-pe \
    -w mccnn -f full -v 4layer \
    --tile-size 69 --gb-size 128 \
    --input-reg 200 --weight-reg 600 --output-reg 64 \
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
  Non-swept regs: InReg=200, WReg=600, IntReg=100

    python3 experiment_runner.py --sweep-outreg-pe \
    -w mccnn -f full -v 4layer \
    --tile-size 69 --gb-size 128 \
    --input-reg 200 --weight-reg 600 --intermediate-reg 100 \
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


================================================================================
EYERISS CASE STUDIES — FSRCNN 8-layer full fusion
================================================================================

  Architecture: Eyeriss-like with unified GlobalBuffer (128KB), no FMEM/WMEM split.
  PE-level registers: WReg (weights), IntReg (intermediates), OutReg (outputs), InReg (inputs).
  Spatial mapping: SARows map C_i × S_i × Z_i factors; SACols map Q/X (tile) and Z.
  Constraint: pe_rows must satisfy ALL layers' SARows products simultaneously.
  SACols Z: Activates when pe_cols >= 2 × tile_size, splitting Z across column groups.


Eyeriss Case Study 1: WRegister Size Sweep (200, 300, 400 bytes) find: A) Minimum PEs configuration that fits WReg constraint, B) Minimum latency configuration
  FSRCNN 8-layer full fusion, Eyeriss, GB=128KB, InReg=64  IntReg=32*7, OutReg=32, tile_size=80
  PE grid: rows=[4,8,12,14,16,28,32,56,84], cols=[4,8,16,32,56,64,80,160,240,320]
    
    python3 experiment_runner.py --sweep-wreg-pe \
    --workload fsrcnn --fusion full --variant 8layer \
    --weight-reg-sizes 200 300 400 \
    --intermediate-reg 500 --output-reg 32 --tile-size 80

    SUMMARY (A) — Minimum PEs Configuration Feasible:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      200     84×160       13,440      Yes        2.904e+04    1.889e+07    5.53e+05
      300     84×4            336      No         2.880e+04    3.525e+07    1.02e+06
      400     84×4            336      No         2.880e+04    3.525e+07    1.02e+06

    SUMMARY (B) — Minimum Latency Configuration:
      WReg    PE Config    Total PEs   SACols Z   Energy(μJ)   Latency(cc)     EDP
      200     84×160       13,440      Yes        2.904e+04    1.889e+07    5.53e+05
      300     84×16         1,344      No         2.880e+04    1.889e+07    5.46e+05
      400     84×16         1,344      No         2.880e+04    1.889e+07    5.46e+05

    Out of 100 grid-search configs per WReg size:
      WReg=200: 1 success  (only 84×160), 99 failures  
      WReg=300: 10 successes (all at pe_rows=84), 90 failures
      WReg=400: 11 successes (all at pe_rows=84), 89 failures

      With pe_rows < 84, all configs fail:
        pe_rows=4:  Can't even distribute entirely the dimensions S, and none of the levels below can).
        pe_rows=8-56: Possible WReg violation (SARows distributes fewer C,S factors → weights can
                      overflow WReg)

    ANALYSIS — WReg footprint can depend entirely on pe_rows:
      If SACols don't distribute Z, what SARows does NOT distribute across rows stays at WReg level (per PE).
      More rows → more C_i, S_i distributed → smaller per-PE weight footprint.

      pe_rows   WReg footprint (no SACols Z)   WReg footprint (with SACols Z = 2 x tile_size)
      8         3244                            1622
      12        1664                            832
      14        1628                            814
      16        1488                            744
      20        1344                            672
      28        956                             478
      32        816                             408
      56        502                             251
      84        258                             ≤200 (fits!)

      The footprint drops dramatically with pe_rows because SARows absorbs more weight
      factors (C_i and S_i). SACols Z consistently halves the footprint by splitting
      Z7 across 2 column groups (pe_cols=160 ≥ 2×tile_size=160).

    ANALYSIS
      At pe_rows=84:
        - Without SACols Z: WReg footprint = 258 words
          → WReg=200: 258 > 200 → FAILS (29% overfit)
          → WReg=300: 258 < 300 → SUCCEEDS (86% utilization)
          → WReg=400: 258 < 400 → SUCCEEDS (65% utilization)
        - With SACols Z (pe_cols=160): WReg footprint ≤ 200 → SUCCEEDS even at WReg=200

      So WReg=200 REQUIRES SACols Z iterations → pe_cols=160 → 84×160 = 13,440 PEs.
      WReg=300 does NOT require SACols Z → pe_cols can be as small as 4 → 84×4 = 336 PEs.
      Adding just 100 bytes to WReg (200→300, +50%) enables 97.5% fewer PEs (13,440→336).

      Why WReg=300 and WReg=400 give identical results:
        The WReg footprint at pe_rows=84 is 258 words. Both 300 and 400 exceed 258.
        The excess capacity is unused.

    ANALYSIS — Latency behavior across pe_cols (for WReg≥300):
      With pe_rows=84, all 10 pe_col values succeed. Latency by pe_cols:
        pe_cols=4:   L=3.525e+07 cc  (DRAM Q = 960/80 × ... many iterations in GB)
        pe_cols=8:   L=2.140e+07 cc
        pe_cols=10:  L=2.020e+07 cc
        pe_cols=16:  L=1.889e+07 cc  ← latency plateau starts
        pe_cols=20-80: L=1.889e+07 cc  (no further improvement)
        pe_cols=160: L=1.889e+07 cc  (same, but slightly higher energy: 2.904e+04 vs 2.880e+04)
      Latency plateaus at pe_cols=16 because 16 columns provide enough spatial coverage
      for the tile_size=80 workload. Extra columns beyond 16 don't reduce GB iterations.
      




      
Eyeriss Case Study 2: IntermediateRegister Size Sweep (200, 300, 400 entries) find: A) Minimum PEs configuration that fits IntReg constraint, B) Minimum latency configuration
  FSRCNN 8-layer full fusion, Eyeriss, GB=128KB, InReg=50, WReg=500, OutReg=32, tile_size=80
  Grid search: 9 PE rows × 10 PE cols = 90 configs per IntReg size

    python3 experiment_runner.py --sweep-intreg-pe \
    --workload fsrcnn --fusion full --variant 8layer \
    --intermediate-reg-sizes 200 300 400 \
    --weight-reg 500 --output-reg 32 --tile-size 80

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
  FSRCNN 8-layer full fusion, Eyeriss, GB=128KB, InReg=50, WReg=500, IntReg=320, tile_size=80

    python3 experiment_runner.py --sweep-outreg-pe \
    --workload fsrcnn --fusion full --variant 8layer \
    --output-reg-sizes 200 300 400 \
    --weight-reg 500 --intermediate-reg 320 --tile-size 80

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
    - Single layers: small WMEM (just one layer's weights)
    - Intermediate segments: medium WMEM (block's weights)
    - Full fusion: large WMEM (all layers' weights)

  WHY THIS MATTERS:
    Accelergy computes energy-per-access as a function of memory depth.
    A larger WMEM has higher energy per read/write. Without per-variant
    sizing, single layers would be evaluated on the full-fusion WMEM
    (e.g., 14400 KB for VGG16), inflating their energy and unfairly
    making fusion look better.

    Example for VGG16 DepFiN:
      Full fusion (13 layers): WMEM ≈ 14,367 KB (all 13 layers' weights)
      Single conv1_1 layer:    WMEM ≈ 2 KB     (64×3×3×3 = 1,728 weights)
      → 7000× smaller WMEM → much lower energy per weight access

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
    FSRCNN:   84×16 PEs (1344),  GB=128KB, WReg=500, IntReg=500, OutReg=32, InReg=64,  tile=80
    MC-CNN:   56×64 PEs (3584),  GB=128KB, WReg=600, IntReg=100, OutReg=64, InReg=200, tile=69
    VGG16:    196×64 PEs (12544), GB=128KB, WReg=5000, IntReg=300, OutReg=64, InReg=750, tile=1
    ResNet18: 256×32 PEs (8192),  GB=128KB, WReg=4000, IntReg=300, OutReg=64, InReg=700, tile=1

  ────────────────────────────────────────────────────────────────────────────
  HOW TO REPRODUCE
  ────────────────────────────────────────────────────────────────────────────

  # DepFiN (WMEM shared per case study — max(min_wmem) across all variants)
  python3 experiment_runner.py --compare-fusion --workload fsrcnn   --arch-type depfin --fmem-size 1056 --pe-rows 4  --pe-cols 512
  python3 experiment_runner.py --compare-fusion --workload mccnn    --arch-type depfin --fmem-size 1056 --pe-rows 8  --pe-cols 256
  python3 experiment_runner.py --compare-fusion --workload vgg16    --arch-type depfin --fmem-size 1056 --pe-rows 16 --pe-cols 128
  python3 experiment_runner.py --compare-fusion --workload resnet18 --arch-type depfin --fmem-size 1056 --pe-rows 16 --pe-cols 128

  # Eyeriss
  python3 experiment_runner.py --compare-fusion --workload fsrcnn   --arch-type eyeriss --gb-size 128 --pe-rows 84  --pe-cols 16  --weight-reg 500  --intermediate-reg 500 --output-reg 32 --input-reg 64  --tile-size 80
  python3 experiment_runner.py --compare-fusion --workload mccnn    --arch-type eyeriss --gb-size 128 --pe-rows 56  --pe-cols 64  --weight-reg 600  --intermediate-reg 100 --output-reg 64 --input-reg 200 --tile-size 69
  python3 experiment_runner.py --compare-fusion --workload vgg16    --arch-type eyeriss --gb-size 128 --pe-rows 196 --pe-cols 64  --weight-reg 5000 --intermediate-reg 300 --output-reg 64 --input-reg 750 --tile-size 1
  python3 experiment_runner.py --compare-fusion --workload resnet18 --arch-type eyeriss --gb-size 128 --pe-rows 256 --pe-cols 32  --weight-reg 4000 --intermediate-reg 300 --output-reg 64 --input-reg 700 --tile-size 1


================================================================================
FUSION COMPARISON RESULTS — DEPFIN (shared WMEM = max(min_wmem) per case study)
================================================================================

  WMEM sizing: CS1 shared WMEM = max(min_wmem across all singles in CS1).
  CS2 shared WMEM = max(min_wmem across all intermediate segments in CS2).
  This models one physical chip per fusion strategy.

  ────────────────────────────────────────────────────────────────────────────
  FSRCNN — DepFiN (4×512, FMEM=1056KB, WMEM=auto)
  ────────────────────────────────────────────────────────────────────────────
  Full fusion (8L): WMEM=19KB | CS1 shared WMEM=8KB | CS2 shared WMEM=9KB

  Level                     Energy (μJ)  Latency (cc)        EDP           DRAM Reads    DRAM Writes
  ───────────────────────────────────────────────────────────────────────────────────────────────────
  Full Fusion               1.003e+04    5.680e+06           6.50e+04      1,573,992     8,294,400
  Σ Singles (CS1)           8.238e+03    2.244e+07           3.19e+04      90,738,792    97,459,200
  Σ Intermediate (CS2)      1.033e+04    6.224e+06           2.52e+04      14,015,592    20,736,000

  CS1 Ratios (Full / Σ Singles):
    Energy:  1.22× worse  (-21.7%)
    Latency: 0.25× better (+74.7%)
    DRAM Rd: 0.017× better (+98.3%)
    DRAM Wr: 0.085× better (+91.5%)

  CS2 Ratios (Full / Σ Intermediate):
    Energy:  0.97× better (+2.9%)
    Latency: 0.91× better (+8.8%)
    DRAM Rd: 0.11× better (+88.8%)
    DRAM Wr: 0.40× better (+60.0%)

  Intermediate fusion captures 96.8% of full fusion latency benefit.

  ────────────────────────────────────────────────────────────────────────────
  MC-CNN — DepFiN (8×256, FMEM=1056KB, WMEM=auto)
  ────────────────────────────────────────────────────────────────────────────
  Full fusion (4L): WMEM=28KB | CS1 shared WMEM=10KB | CS2 shared WMEM=19KB

  Level                      Energy (μJ)  Latency (cc)        EDP           DRAM Reads    DRAM Writes
  ───────────────────────────────────────────────────────────────────────────────────────────────────
  Full Fusion                1.227e+04    8.074e+06           1.16e+05      494,928       14,943,744
  Σ Singles (CS1)            4.833e+03    1.381e+07           1.87e+04      45,326,160    59,774,976
  Σ Intermediate (CS2)       1.301e+04    8.074e+06           6.58e+04      15,438,672    29,887,488

  CS1 Ratios (Full / Σ Singles):
    Energy:  2.54× worse  (-153.9%)
    Latency: 0.58× better (+41.5%)
    DRAM Rd: 0.011× better (+98.9%)
    DRAM Wr: 0.25× better (+75.0%)

  CS2 Ratios (Full / Σ Intermediate):
    Energy:  0.94× better (+5.7%)
    Latency: 1.00 (no change)
    DRAM Rd: 0.032× better (+96.8%)
    DRAM Wr: 0.50× better (+50.0%)

  Intermediate fusion captures 100.0% of full fusion latency benefit.

  ────────────────────────────────────────────────────────────────────────────
  VGG16 — DepFiN (16×128, FMEM=1056KB, WMEM=auto)
  ────────────────────────────────────────────────────────────────────────────
  Full fusion (13L): WMEM=14366KB | CS1 shared WMEM=2305KB | CS2 shared WMEM=6913KB

  Level                      Energy (μJ)  Latency (cc)        EDP          DRAM Reads       DRAM Writes
  ───────────────────────────────────────────────────────────────────────────────────────────────────────
  Full Fusion                7.641e+04    2.433e+07           3.41e+06     14,860,992       100,352
  Σ Singles (CS1)            2.706e+03    2.452e+07           7.49e+03     23,792,320       13,547,520
  Σ Intermediate (CS2)       5.799e+04    2.432e+07           5.87e+05     16,366,272       6,121,472

  CS1 Ratios (Full / Σ Singles):
    Energy:  28.24× worse  (-2723.5%)
    Latency: 0.99× better (+0.8%)
    DRAM Rd: 0.62× better (+37.5%)
    DRAM Wr: 0.007× better (+99.3%)

  CS2 Ratios (Full / Σ Intermediate):
    Energy:  1.32× worse  (-31.8%)
    Latency: 1.00 (no change)
    DRAM Rd: 0.91× better (+9.2%)
    DRAM Wr: 0.016× better (+98.4%)

  Intermediate fusion captures 100.7% of full fusion latency benefit.

  ────────────────────────────────────────────────────────────────────────────
  ResNet18 — DepFiN (16×128, FMEM=1056KB, WMEM=auto)
  ────────────────────────────────────────────────────────────────────────────
  Full fusion (17L): WMEM=10738KB | CS1 shared WMEM=2304KB | CS2 shared WMEM=4608KB
  Note: L16, L18, L19 singles FAILED (ill-posed) — CS1 sum = 14/17 layers.

  Level                   Energy (μJ)       Latency (cc)        EDP      DRAM Reads       DRAM Writes
  ────────────────────────────────────────────────────────────────────────────────────────────────────
  Full Fusion               1.090e+04       7.812e+06       1.52e+05     11,032,512       25,088
  Σ Singles (CS1)           4.294e+02       3.809e+06       1.78e+02     5,296,832        2,232,832
  Σ Intermediate (CS2)      8.085e+03       7.813e+06       1.41e+04     11,760,064       752,640

  CS1 Ratios (Full / Σ Singles):
    Energy:  25.38× worse  (-2438.5%)
    Latency: 2.05× worse   (-105.1%)
    DRAM Rd: 2.08× worse   (-108.3%)
    DRAM Wr: 0.011× better (+98.9%)

  CS2 Ratios (Full / Σ Intermediate):
    Energy:  1.35× worse  (-34.8%)
    Latency: 1.00 (no change)
    DRAM Rd: 0.94× better (+6.2%)
    DRAM Wr: 0.033× better (+96.7%)


================================================================================
FUSION COMPARISON RESULTS — EYERISS (fixed energy model, no WMEM auto-sizing)
================================================================================

  ────────────────────────────────────────────────────────────────────────────
  FSRCNN — Eyeriss (84×16, GB=128KB, tile=80)
  ────────────────────────────────────────────────────────────────────────────
  Level                   Energy (μJ)  Latency (cc)        EDP     DRAM Reads   DRAM Writes
  ───────────────────────────────────────────────────────────────────────────────────────────
  Full Fusion               2.880e+04    3.022e+07    8.74e+05      1,573,992     8,294,400
  Σ Singles (CS1)            8.482e+03    4.126e+07    5.43e+04     90,738,792    97,459,200
  Σ Intermediate (CS2)      2.942e+04    3.022e+07    3.18e+05     14,015,592    20,736,000

  CS1 Ratios (Full / Σ Singles):
    Energy:  3.40× worse  (-239.5%)
    Latency: 0.73× better (+26.8%)
    DRAM Rd: 0.017× better (+98.3%)
    DRAM Wr: 0.085× better (+91.5%)

  CS2 Ratios (Full / Σ Intermediate):
    Energy:  0.98× better (+2.1%)
    Latency: 1.00 (no change)
    DRAM Rd: 0.11× better (+88.8%)
    DRAM Wr: 0.40× better (+60.0%)

  Intermediate fusion captures 100.0% of full fusion latency benefit.

  ────────────────────────────────────────────────────────────────────────────
  MC-CNN — Eyeriss (56×64, GB=128KB, tile=69)
  ────────────────────────────────────────────────────────────────────────────
  Level                   Energy (μJ)  Latency (cc)        EDP     DRAM Reads   DRAM Writes
  ───────────────────────────────────────────────────────────────────────────────────────────
  Full Fusion               3.617e+04    2.302e+07    8.35e+05        494,928    14,943,744
  Σ Singles (CS1)            5.263e+03    2.483e+07    3.51e+04     45,326,160    59,774,976
  Σ Intermediate (CS2)      3.760e+04    2.302e+07    4.63e+05     15,438,672    29,887,488

  CS1 Ratios (Full / Σ Singles):
    Energy:  6.87× worse  (-587.3%)
    Latency: 0.93× better (+7.3%)
    DRAM Rd: 0.011× better (+98.9%)
    DRAM Wr: 0.25× better (+75.0%)

  CS2 Ratios (Full / Σ Intermediate):
    Energy:  0.96× better (+3.8%)
    Latency: 1.00 (no change)
    DRAM Rd: 0.032× better (+96.8%)
    DRAM Wr: 0.50× better (+50.0%)

  Intermediate fusion captures 100.0% of full fusion latency benefit.

  ────────────────────────────────────────────────────────────────────────────
  VGG16 — Eyeriss (196×64, GB=128KB, tile=1)
  ────────────────────────────────────────────────────────────────────────────
  Level                   Energy (μJ)  Latency (cc)        EDP     DRAM Reads   DRAM Writes
  ───────────────────────────────────────────────────────────────────────────────────────────
  Full Fusion               4.245e+04    8.159e+07    3.51e+06     14,860,992       100,352
  Σ Singles (CS1)            3.213e+03    8.159e+07    3.67e+04     23,792,320    13,547,520
  Σ Intermediate (CS2)      4.260e+04    8.159e+07    6.92e+05     16,366,272     6,121,472

  CS1 Ratios (Full / Σ Singles):
    Energy:  13.2× worse  (-1221.2%)
    Latency: 1.00 (no change)
    DRAM Rd: 0.625× better (+37.5%)
    DRAM Wr: 0.007× better (+99.3%)

  CS2 Ratios (Full / Σ Intermediate):
    Energy:  1.00 (no change)
    Latency: 1.00 (no change)
    DRAM Rd: 0.91× better (+9.2%)
    DRAM Wr: 0.016× better (+98.4%)

  ────────────────────────────────────────────────────────────────────────────
  ResNet18 — Eyeriss (256×32, GB=128KB, tile=1)
  ────────────────────────────────────────────────────────────────────────────
  Level                   Energy (μJ)  Latency (cc)        EDP     DRAM Reads   DRAM Writes
  ───────────────────────────────────────────────────────────────────────────────────────────
  Full Fusion               6.779e+03    1.745e+07    1.19e+05     11,032,512        25,088
  Σ Singles (CS1)            7.416e+02    1.501e+07    6.90e+02     12,449,984     2,308,096
  Σ Intermediate (CS2)      6.870e+03    1.746e+07    1.57e+04     11,760,064       752,640

  CS1 Ratios (Full / Σ Singles):
    Energy:  9.14× worse  (-814.1%)
    Latency: 1.16× worse  (-16.3%)
    DRAM Rd: 0.886× better (+11.4%)
    DRAM Wr: 0.011× better (+98.9%)

  CS2 Ratios (Full / Σ Intermediate):
    Energy:  0.99× better (+1.3%)
    Latency: 1.00 (no change)
    DRAM Rd: 0.94× better (+6.2%)
    DRAM Wr: 0.033× better (+96.7%)


================================================================================
KEY OBSERVATIONS AND ANALYSIS (shared WMEM per case study, workload-specific PEs)
================================================================================

  1. DRAM WRITE SAVINGS (CS1): Consistently massive (+75% to +99.2%)
     Full fusion eliminates ALL intermediate DRAM writes — only the final
     output is written. This remains the most consistent benefit of fusion.

  2. ENERGY: FUSION IS ALWAYS MORE EXPENSIVE (CS1):
     With shared WMEM per case study, fusion is 1.22× (FSRCNN) to 28.2×
     (VGG16) more expensive in energy. The fused iteration space creates
     massive on-chip data movement overhead.
     Eyeriss results are unchanged (fixed energy model).

  3. LATENCY: MIXED RESULTS WITH SHARED WMEM
     DepFiN: FSRCNN (+74.7%), MC-CNN (+41.5%), and VGG16 (+0.8%) benefit
     from fusion. ResNet18 (-13.1%) is slightly worse with fusion.
     Eyeriss: Modest to good latency savings (+7.3% to +26.8%) except
     VGG16 and ResNet18 where latency is similar.

  4. INTERMEDIATE FUSION (CS2) VS FULL FUSION:
     Energy ratios remain close to 1.0 (0.94-1.35×). The big energy and
     latency gaps are between singles and any level of fusion. Full vs
     intermediate fusion has marginal difference except DRAM writes.

  5. DRAM TRAFFIC IS ARCHITECTURE-INDEPENDENT:
     DRAM reads and writes are IDENTICAL between DepFiN and Eyeriss for
     each workload, confirming the DRAM level sees the same tiled pattern.

  ────────────────────────────────────────────────────────────────────────────
  SUMMARY TABLE — CS1: Full Fusion vs Σ Singles
  ────────────────────────────────────────────────────────────────────────────
  Workload   Arch     Energy      Latency    DRAM Rd    DRAM Wr     Note
  ─────────  ───────  ──────────  ─────────  ─────────  ─────────  ──────────
  FSRCNN     DepFiN    1.22×↑     +74.7%     +98.3%     +91.5%     8/8 ok
  FSRCNN     Eyeriss   3.40×↑     +26.8%     +98.3%     +91.5%
  MC-CNN     DepFiN    2.54×↑     +41.5%     +98.9%     +75.0%     4/4 ok
  MC-CNN     Eyeriss   6.87×↑      +7.3%     +98.9%     +75.0%
  VGG16      DepFiN   28.24×↑      +0.8%     +37.5%     +99.3%     13/13 ok
  VGG16      Eyeriss  13.2×↑       +0.0%     +37.5%     +99.3%
  ResNet18   DepFiN   14.18×↑     -13.1%     +11.4%     +98.9%     17/17 ok
  ResNet18   Eyeriss   9.14×↑     -16.3%     +11.4%     +98.9%

  (×↑ = times worse for energy; + = savings; - = worse)

  ────────────────────────────────────────────────────────────────────────────
  SUMMARY TABLE — CS2: Full Fusion vs Σ Intermediate Segments
  ────────────────────────────────────────────────────────────────────────────
  Workload   Arch     Energy    Latency   DRAM Rd    DRAM Wr
  ─────────  ───────  ────────  ────────  ─────────  ─────────
  FSRCNN     DepFiN    +2.9%     +8.8%    +88.8%     +60.0%
  FSRCNN     Eyeriss   +2.1%     +0.0%    +88.8%     +60.0%
  MC-CNN     DepFiN    +5.7%     +0.0%    +96.8%     +50.0%
  MC-CNN     Eyeriss   +3.8%     +0.0%    +96.8%     +50.0%
  VGG16      DepFiN   -31.8%     +0.0%     +9.2%     +98.4%
  VGG16      Eyeriss   +0.3%     +0.0%     +9.2%     +98.4%
  ResNet18   DepFiN   -34.6%     +0.0%     +6.2%     +96.7%
  ResNet18   Eyeriss   +1.3%     +0.0%     +6.2%     +96.7%

  Note: DRAM columns are identical across DepFiN/Eyeriss per workload.
  DepFiN CS2 energy is worse for VGG16/ResNet18 because full fusion's
  14366/10738 KB WMEM has much higher Accelergy energy-per-access than
  the intermediate segments' smaller WMEMs.

================================================================================
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
        # Eyeriss architecture support
        create_eyeriss_architecture,
        EyerissArchConfig,
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
    scale_bandwidth: bool = False              # Scale FMEM bandwidth with tile size
    base_tile_size: int = 128                  # Reference tile size for scaling
    
    # Settings
    bias_read: bool = False
    verbose: bool = False
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = f"{self.workload_name}_{self.workload_variant}_{self.timestamp}"

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
                           2 if name in ['block1', 'block2'] else 3)
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
                    
                    # If tile_size is specified, use constrained (deterministic) architecture
                    if config.tile_size is not None:
                        arch = create_constrained_eyeriss_architecture(
                            config=arch_config,
                            coupling=coupling,
                            shape=shape,
                            output_tile_size=config.tile_size,
                            num_layers=config.num_fused_layers
                        )
                    else:
                        arch = create_eyeriss_architecture(
                            config=arch_config,
                            coupling=coupling,
                            shape=shape,
                            num_layers=config.num_fused_layers
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
                    # Use generic architecture factory with proper layer count and shape
                    arch = create_thesis_architecture(
                        config=arch_config,
                        coupling=coupling,
                        shape=shape,
                        num_layers=config.num_fused_layers
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
        # pe_rows_candidates = [4, 8, 12, 14, 16, 28, 32, 56, 64, 84, 128, 256, 512]
        pe_rows_candidates = [256, 324, 512]

        # PE cols: include base values + tile_size multiples
        pe_cols_base = [1, 2, 4, 8, 16, 32, 56, 64, 128]
        pe_cols_tile_multiples = [tile_size, 2*tile_size, 3*tile_size, 4*tile_size]
        pe_cols_candidates = sorted(set(pe_cols_base + pe_cols_tile_multiples))
        
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
                
                if total_pes > 15512:  # Skip configs that exceed max_total_pes
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
            return None, None
        
        # 1. Find config with minimum total PEs
        min_pes_config = min(successful_configs, key=lambda x: x['total_pes'])
        
        # 2. Find config with minimum latency, then minimum PEs
        min_latency = min(c['latency'] for c in successful_configs)
        min_latency_configs = [c for c in successful_configs if c['latency'] == min_latency]
        min_latency_config = min(min_latency_configs, key=lambda x: x['total_pes'])
        
        return min_pes_config, min_latency_config


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
            
            # Comprehensive grid search - returns both min_pes and min_latency configs
            min_pes_config, min_latency_config = self._find_min_pe_grid_search(
                workload, config_template, tile_size=tile_size, verbose=verbose,
                pe_rows_candidates=pe_rows_list, pe_cols_candidates=pe_cols_list,
            )
            
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
            
            # Comprehensive grid search - returns both min_pes and min_latency configs
            min_pes_config, min_latency_config = self._find_min_pe_grid_search(
                workload, config_template, tile_size=tile_size, verbose=verbose,
                pe_rows_candidates=pe_rows_list, pe_cols_candidates=pe_cols_list,
            )
            
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
            
            # Comprehensive grid search - returns both min_pes and min_latency configs
            min_pes_config, min_latency_config = self._find_min_pe_grid_search(
                workload, config_template, tile_size=tile_size, verbose=verbose,
                pe_rows_candidates=pe_rows_list, pe_cols_candidates=pe_cols_list,
            )
            
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
        # Default: arch_eyeriss_conv original sizes (physically realistic baseline)
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
        
        # === Resolve SINGLE Eyeriss defaults (arch_eyeriss_conv original sizes) ===
        _aec = _ARCH_EYERISS_CONV_SIZES
        if single_gb_size_kb is None:
            single_gb_size_kb = _aec['global_buffer_kB']
        if single_input_reg is None:
            single_input_reg = _aec['input_reg_entries']
        if single_weight_reg is None:
            single_weight_reg = _aec['weight_reg_entries']
        if single_output_reg is None:
            single_output_reg = _aec['output_reg_entries']
        
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
        print(f"{'='*90}\n")
        
        # Helper to create config with per-variant WMEM and Eyeriss sizing
        def make_config(wk_name, fusion_level, variant, num_layers, shape,
                        variant_category="full", wmem_override_kb=None):
            """Create ExperimentConfig with per-variant memory sizing.
            
            variant_category: "full" | "single" | "intermediate"
            wmem_override_kb: if set, use this WMEM instead of auto-computing.
              Used for CS1/CS2 where all variants share one architecture
              (WMEM = max across all variants in that case study).
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
                tile_size=tile_size,
                scale_bandwidth=scale_bandwidth,
                base_tile_size=base_tile_size,
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
        
        for _, _, variant in inter_variants:
            shape_v, _, nlayers = self.workload_registry.get_workload(network, intermediate_level, variant)
            config, variant_wmem = make_config(network, intermediate_level, variant, nlayers, shape_v, "intermediate",
                                               wmem_override_kb=cs2_wmem_kb)
            if arch_type == "depfin":
                print(f"  [WMEM={variant_wmem}KB for {variant} ({nlayers}L)]")
            elif arch_type == "eyeriss":
                print(f"  [WReg={intermediate_weight_reg} for {variant} ({nlayers}L)]")
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
            total_edp = sum(r.edp for _, r in ok)  # Sum of individual EDPs (each includes leakage)
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
