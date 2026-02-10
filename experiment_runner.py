"""
Results:

================================================================================
DEPFIN CASE STUDIES — FSRCNN 8-layer full fusion
================================================================================
  Architecture: Depfin-like.
  Feature Memory Bandwidth scales with output tile size tried: BW_scaled = BW_base × (tile_size / 128)
  PE-level registers: WReg (weights), IntReg (intermediates), OutReg (outputs), InReg (inputs).
  Spatial mapping: SARows map Z_i factors; SACols map Q/X (tile).
  The output tile size is automatically set as the largest divisor of Q that fits in pe_cols



Case Study: Tile Size Width Sweep with Bandwidth Scaling, FSRCNN 8-layer full fusion
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

    
Case Study: Architecture Sweep: FMEM Size, FSRCNN 8-layer full fusion (THIS IS OK)
  FSRCNN 8-layer full fusion, DepFiN 16×128 PEs, FMEM= sweep, WMEM=524KB

    python3 experiment_runner.py --sweep-arch \
    --workload fsrcnn --fusion full --variant 8layer \
    --fmem-sizes 256 512 1024 2048 2>&1
    
    Mapping identical across all FMEM sizes

    256KB - 512KB - 1024KB
    Energy: increase : due to the size. Can be useful only when the FMEM is too small and doesn't allow fusion
    Latency: constant

    
Case Study: PE Array Sweep — Increasing PE Rows (fixed cols=64)
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


Case Study: PE Array Sweep — Increasing PE Cols (fixed rows=16)
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
EYERISS CASE STUDIES — FSRCNN 8-layer full fusion
================================================================================

  Architecture: Eyeriss-like with unified GlobalBuffer (128KB), no FMEM/WMEM split.
  PE-level registers: WReg (weights), IntReg (intermediates), OutReg (outputs), InReg (inputs).
  Spatial mapping: SARows map C_i × S_i × Z_i factors; SACols map Q/X (tile) and Z.
  Constraint: pe_rows must satisfy ALL layers' SARows products simultaneously.
  SACols Z: Activates when pe_cols >= 2 × tile_size, splitting Z across column groups.


Eyeriss Case Study 1: WRegister Size Sweep (200, 300, 400 bytes)
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
      

Eyeriss Case Study 2: IntermediateRegister Size Sweep (200, 300, 400 entries)
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


Eyeriss Case Study 3: OutRegister Size Sweep (200, 300, 400 entries)
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
"""

import os
import sys
import csv
import json
import time
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
from cost_model import EDP, Energy, Latency, MOPs
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
    resnet18_full_fused,
    resnet18_full_coupling,
)

from computations import (
    conv_coupling,
    conv_2layers_coupling,
    conv_3layers_coupling,
    conv_4layers_coupling,
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
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
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
    gb_size_kB: int = 128            # GlobalBuffer size for Eyeriss arch
    
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


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
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

class WorkloadRegistry:
    """
    Registry of all available workloads organized by network and fusion level.
    
    Structure:
        workloads[network_name][fusion_level] = {
            variant_name: (shape, coupling, num_layers)
        }
    """
    
    def __init__(self):
        self.workloads = self._build_registry()
    
    def _build_registry(self) -> Dict[str, Dict[str, Dict[str, Tuple[Shape, Coupling, int]]]]:
        """Build the complete workload registry."""
        
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
                    name: (shape, conv_3layers_coupling, 3)
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
                    name: (shape, conv_2layers_coupling, 2)
                    for name, shape in resnet18_2layer_fused.items()
                },
                "block": {
                    name: (shape, resnet18_block_couplings[name],
                           4 if name == 'stage1' else 2)
                    for name, shape in resnet18_block_fused.items()
                },
                "full": {
                    "17layer": (resnet18_full_fused, resnet18_full_coupling, 17)
                },
            },
        }
        
        return registry
    
    def get_workload(self, network: str, fusion_level: str, variant: str) -> Tuple[Shape, Coupling, int]:
        """Get a specific workload by network, fusion level, and variant."""
        if network not in self.workloads:
            raise ValueError(f"Unknown network: {network}. Available: {list(self.workloads.keys())}")
        if fusion_level not in self.workloads[network]:
            raise ValueError(f"Unknown fusion level: {fusion_level} for {network}. "
                           f"Available: {list(self.workloads[network].keys())}")
        if variant not in self.workloads[network][fusion_level]:
            raise ValueError(f"Unknown variant: {variant} for {network}/{fusion_level}. "
                           f"Available: {list(self.workloads[network][fusion_level].keys())}")
        return self.workloads[network][fusion_level][variant]
    
    def list_workloads(self, network: Optional[str] = None, fusion_level: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """List all workloads, optionally filtered by network and/or fusion level."""
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

class ExperimentRunner:
    """
    Main experiment runner that executes workloads and collects results.
    
    Features:
    - Run single experiments or sweep multiple variables
    - Collect and export results to CSV/JSON
    - Resume interrupted experiments
    - Progress tracking and logging
    """
    
    def __init__(self, output_dir: str = "results"):
        self.workload_registry = WorkloadRegistry()
        self.output_dir = output_dir
        self.results: List[ExperimentResult] = []
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
    
    def create_architecture(self, config: ExperimentConfig, coupling: Coupling, shape: Shape = None) -> Arch:
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
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.
        
        Steps:
        1. Get workload (shape + coupling) from registry
        2. Create architecture with specified parameters
        3. Run the mapping engine
        4. Collect and return results
        """
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
    
    def run_workload_sweep(
        self,
        networks: Optional[List[str]] = None,
        fusion_levels: Optional[List[str]] = None,
        arch_config: Optional[ExperimentConfig] = None,
        progress_callback=None
    ) -> List[ExperimentResult]:
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
    
    def run_architecture_sweep(
        self,
        workload: Tuple[str, str, str],  # (network, fusion_level, variant)
        arch_type: str = "depfin",       # "depfin" or "eyeriss"
        fmem_sizes_kb: Optional[List[int]] = None,  # DepFiN only
        wmem_sizes_kb: Optional[List[int]] = None,  # DepFiN only
        gb_sizes_kb: Optional[List[int]] = None,    # Eyeriss only
        pe_configs: Optional[List[Tuple[int, int]]] = None,  # (rows, cols)
        progress_callback=None,
        verbose: bool = False
    ) -> List[ExperimentResult]:
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
                      f"Util: {result.utilization:.1%}")
            else:
                print(f"  ✗ Failed: {result.error_message[:50]}...")
        
        # Print summary table
        print(f"\n{'='*70}")
        print("TILE SIZE SWEEP SUMMARY")
        print(f"{'='*70}")
        print(f"{'Tile':>6} {'PE Util':>8} {'Energy(μJ)':>12} {'Latency(cc)':>14} {'EDP':>12} {'Util':>8}")
        print(f"{'-'*70}")
        
        successful = [r for r in results if r.success]
        for r in successful:
            tile = r.config.tile_size
            pe_util = (tile / pe_cols) * 100
            print(f"{tile:>6} {pe_util:>7.1f}% {r.energy_uJ:>12.3e} {r.latency_cycles:>14.3e} {r.edp:>12.2e} {r.utilization:>7.1%}")
        
        if successful:
            best_edp = min(successful, key=lambda r: r.edp)
            print(f"\nBest EDP: tile_size={best_edp.config.tile_size}, EDP={best_edp.edp:.2e}")
        
        return results
    
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
    
    def _find_min_pe_via_rows_sweep(
        self,
        workload: Tuple[str, str, str],
        config_template: ExperimentConfig,
        pe_cols: int,
        tile_size: int = 80,
        verbose: bool = False
    ) -> Tuple[int, int, Optional['ExperimentResult']]:
        """
        Find minimum PE rows (at fixed pe_cols) that makes mapping feasible.
        
        This is the "rows-first" strategy where we keep pe_cols fixed (typically = tile_size)
        and sweep pe_rows to find the minimum. SARows handles M/Z distribution.
        
        Returns:
            (min_pe_rows, total_pes, result)
        """
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
    
    def _find_min_pe_via_cols_sweep(
        self,
        workload: Tuple[str, str, str],
        config_template: ExperimentConfig,
        pe_rows: int,
        tile_size: int = 80,
        verbose: bool = False
    ) -> Tuple[int, int, Optional['ExperimentResult']]:
        """
        Find minimum PE cols (at fixed pe_rows) that makes mapping feasible.
        
        This is the "cols-first" strategy where we keep pe_rows fixed (typically small)
        and sweep pe_cols. When pe_cols >= 2×tile_size, SACols Z optimization activates.
        
        Returns:
            (min_pe_cols, total_pes, result)
        """
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
    
    def _find_min_pe_grid_search(
        self,
        workload: Tuple[str, str, str],
        config_template: ExperimentConfig,
        tile_size: int = 80,
        verbose: bool = False
    ) -> Tuple[dict, dict]:
        """
        Comprehensive grid search to find:
        1. Config with minimum total PEs (regardless of latency)
        2. Config with minimum latency, then minimum PEs among equal-latency
        
        Returns:
            (min_pes_result, min_latency_result) - each is a dict with:
            {'pe_rows', 'pe_cols', 'total_pes', 'sacols_z', 'result'} or None if not found
        """
        network, fusion_level, variant = workload
        
        # REDUCED candidates for faster search - focused on likely solutions
        pe_rows_candidates = [4, 8, 12, 14, 16, 28, 32, 56, 84]
        
        # PE cols: include small values and tile_size multiples
        pe_cols_small = [4, 8, 16, 32, 56, 64]
        pe_cols_tile_multiples = [tile_size, 2*tile_size, 3*tile_size, 4*tile_size]
        pe_cols_candidates = sorted(set(pe_cols_small + pe_cols_tile_multiples))
        
        # Collect ALL successful configurations
        successful_configs = []
        
        for pe_rows in pe_rows_candidates:
            for pe_cols in pe_cols_candidates:
                total_pes = pe_rows * pe_cols
                
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

    def run_wreg_pe_sweep(
        self,
        workload: Tuple[str, str, str],
        weight_reg_sizes: List[int],
        gb_size_kb: int = 128,
        tile_size: int = 80,
        input_reg_entries: int = None,
        intermediate_reg_entries: int = None,
        output_reg_entries: int = None,
        verbose: bool = False
    ) -> List['ExperimentResult']:
        # Use thesis_arch defaults if not specified
        if input_reg_entries is None:
            input_reg_entries = _EYERISS_DEFAULTS.input_reg_entries
        if intermediate_reg_entries is None:
            intermediate_reg_entries = _EYERISS_DEFAULTS.intermediate_out_reg_entries
        if output_reg_entries is None:
            output_reg_entries = _EYERISS_DEFAULTS.output_reg_entries
        """
        Case Study 1: Sweep WRegister sizes and find minimum PE configuration.
        
        Uses comprehensive grid search across all (rows × cols) combinations.
        For each WRegister size, finds the configuration with minimum total PEs
        that makes the mapping feasible.
        """
        network, fusion_level, variant = workload
        
        # Define lists for display (matches _find_min_pe_grid_search)
        pe_rows_list = [4, 8, 12, 14, 16, 28, 32, 56, 84]
        
        # PE cols: include small values and tile_size multiples
        pe_cols_small = [4, 8, 16, 32, 56, 64]
        pe_cols_tile_multiples = [tile_size, 2*tile_size, 3*tile_size, 4*tile_size]
        pe_cols_list = sorted(set(pe_cols_small + pe_cols_tile_multiples))
        
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
                workload, config_template, tile_size=tile_size, verbose=verbose
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
        
        Uses comprehensive grid search across all (rows × cols) combinations:
        - pe_rows: [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 84, 96, 128, 168, 256, 336, 512, 1024]
        - pe_cols: [tile_size, 2×tile, 3×tile, 4×tile, 6×tile, 8×tile]
        
        Finds the configuration with minimum total PEs that makes mapping feasible.
        """
        network, fusion_level, variant = workload
        
        # Define candidate lists for display (matches _find_min_pe_grid_search)
        pe_rows_list = [4, 8, 12, 14, 16, 28, 32, 56, 84]
        
        # PE cols: include small values and tile_size multiples
        pe_cols_small = [4, 8, 16, 32, 56, 64]
        pe_cols_tile_multiples = [tile_size, 2*tile_size, 3*tile_size, 4*tile_size]
        pe_cols_list = sorted(set(pe_cols_small + pe_cols_tile_multiples))
        
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
        print(f"  - SACols Z active when pe_cols >= {2*tile_size}")
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
                workload, config_template, tile_size=tile_size, verbose=verbose
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
        
        # Define candidate lists for display (matches _find_min_pe_grid_search)
        pe_rows_list = [4, 8, 12, 14, 16, 28, 32, 56, 84]
        
        # PE cols: include small values and tile_size multiples
        pe_cols_small = [4, 8, 16, 32, 56, 64]
        pe_cols_tile_multiples = [tile_size, 2*tile_size, 3*tile_size, 4*tile_size]
        pe_cols_list = sorted(set(pe_cols_small + pe_cols_tile_multiples))
        
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
        print(f"  - SACols Z active when pe_cols >= {2*tile_size}")
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
                workload, config_template, tile_size=tile_size, verbose=verbose
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
    
    # Eyeriss register sweep ranges (for case studies 1-3)
    parser.add_argument("--weight-reg-sizes", nargs="+", type=int,
                       help="WRegister sizes to sweep [Eyeriss Case Study 1]")
    parser.add_argument("--intermediate-reg-sizes", nargs="+", type=int,
                       help="IntermediateRegister sizes to sweep [Eyeriss Case Study 2]")
    parser.add_argument("--output-reg-sizes", nargs="+", type=int,
                       help="OutRegister sizes to sweep [Eyeriss Case Study 3]")
    
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
