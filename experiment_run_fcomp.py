"""
────────────────────────────────────────────────────────────────────────────
RESULTS — ResNet18, Eyeriss 16384 PE: 512×32, WReg=902
────────────────────────────────────────────────────────────────────────────

  
  ── full_vs_single ─────────────────────────────────────────────────────────
 
  --compare-full-vs-single -w resnet18 --arch-type eyeriss
      --pe-rows 512 --pe-cols 32 --weight-reg 902
      --gb-size 128 --tile-size 1
      --input-reg 700 --intermediate-reg 300 --output-reg 64

  Level                    Energy (uJ)  Latency (cc)        EDP    DRAM Reads  DRAM Writes
  Full Fusion (1 ok)       5.062e+04    3.042e+06      1.60e+05    11,032,512       25,088
  Sum Singles (17 ok)      2.643e+03    3.301e+06      4.33e+02    12,449,984    2,308,096

  RATIOS (Full Fusion / Sum Singles)  — values < 1.0 mean fusion wins
    Energy:      19.1543  (-1815.4%)  ← full fusion MUCH MORE expensive
    Latency:      0.9215  (+7.8%)     ← full fusion slightly faster
    EDP:        370.5536  (-36955.4%) ← massively worse EDP
    DRAM Reads:   0.8861  (+11.4%)    ← full fusion reads less from DRAM
    DRAM Writes:  0.0109  (+98.9%)    ← full fusion writes almost nothing to DRAM

  Single-layer breakdown:
    L0_conv1:       E=1.738e+02 uJ, L=2.007e+05 cc
    L1_conv2_1_1:   E=3.448e+02 uJ, L=5.939e+04 cc
    L2_conv2_1_2:   E=3.448e+02 uJ, L=5.939e+04 cc
    L3_conv2_2_1:   E=3.448e+02 uJ, L=5.939e+04 cc
    L4_conv2_2_2:   E=3.448e+02 uJ, L=5.939e+04 cc
    L5_conv3_1_1:   E=9.773e+01 uJ, L=3.098e+04 cc
    L6_conv3_1_2:   E=1.174e+02 uJ, L=6.195e+04 cc
    L8_conv3_2_1:   E=1.174e+02 uJ, L=6.195e+04 cc
    L9_conv3_2_2:   E=1.174e+02 uJ, L=6.195e+04 cc
    L10_conv4_1_1:  E=4.628e+01 uJ, L=8.000e+04 cc
    L11_conv4_1_2:  E=7.533e+01 uJ, L=1.600e+05 cc
    L13_conv4_2_1:  E=7.533e+01 uJ, L=1.600e+05 cc
    L14_conv4_2_2:  E=7.533e+01 uJ, L=1.600e+05 cc
    L15_conv5_1_1:  E=5.622e+01 uJ, L=2.980e+05 cc
    L16_conv5_1_2:  E=1.038e+02 uJ, L=5.961e+05 cc
    L18_conv5_2_1:  E=1.038e+02 uJ, L=5.961e+05 cc
    L19_conv5_2_2:  E=1.038e+02 uJ, L=5.961e+05 cc


  Analysis:
      Full fusion is 19.2× MORE expensive in energy than sum of singles.
      This is expected for Eyeriss at this configuration: with WReg=902
      storing weight residuals for ALL 17 layers, the register-level
      WMOPs are enormous (each PE cycles through C×R factors for every
      layer on every MAC). Singles only store 1 layer's weights per run.

      Latency is 7.8 % LOWER for full fusion (3.04M vs 3.30M cc).
      Full fusion benefits from eliminating intermediate DRAM round-trips
      (DRAM Writes drop 98.9 %) — intermediates stay in GlobalBuffer.

      DRAM Reads also drop 11.4 % (DRAM read of intermediate activations
      eliminated). But the massive register energy overhead completely
      overwhelms these DRAM savings.

      EDP is 370× worse for full fusion — the energy penalty dominates.
      This shows that at WReg=902 (minimum feasible for 16,384 PEs),
      Eyeriss full fusion is not energy-competitive with layer-by-layer
      execution for ResNet18



  ── full_vs_partial: 2 layer ────────────────────────────────────────────────────────

  --compare-full-vs-partial -w resnet18 --arch-type eyeriss
      --pe-rows 512 --pe-cols 32 --weight-reg 902
      --gb-size 128 --tile-size 1
      --input-reg 700 --intermediate-reg 300 --output-reg 64

  Level                         Energy (uJ)  Latency (cc)        EDP         DRAM Reads       DRAM Writes
  Full Fusion (1 ok)            5.062e+04    3.042e+06           1.60e+05    11,032,512       25,088
  Sum Partial Fusion (8 ok)     4.972e+04    3.139e+06           1.83e+04    11,760,064      752,640

  RATIOS (Full Fusion / Sum Partial Fusion)  — values < 1.0 mean full fusion wins
    Energy:       1.0181  (-1.8%)     ← nearly identical energy
    Latency:      0.9692  (+3.1%)     ← full fusion slightly faster
    EDP:          8.7867  (-778.7%)   ← worse EDP for full fusion
    DRAM Reads:   0.9381  (+6.2%)     ← full fusion reads ~6% less
    DRAM Writes:  0.0333  (+96.7%)    ← full fusion writes 97% less to DRAM

  Notes:
  - This is the most interesting comparison: full vs partial on the SAME
    fixed architecture (WReg=902).
  - Energy is nearly identical (1.02×) — the extra weight storage cost of
    full fusion is offset by eliminating ALL intermediate DRAM writes.
  - Full fusion saves +96.7% DRAM writes vs partial fusion, because even
    partial fusion must write/read intermediates between non-adjacent pairs.
  - EDP is 8.8× worse for full fusion because its (slightly) higher energy
    multiplied by comparable latency amplifies the gap.
    EDP: full fusion is 8.8× worse despite nearly equal energy because
      the latency advantage is tiny while energy is slightly higher.
      The 2-layer blocks benefit from much lower per-segment EDP
      (each segment runs fast with few layers), and their sum is
      dominated by the fast early-stage blocks.
  - Latency difference is only +3.1% in favor of full fusion.

  



  ── full_vs_partial: 4 layer ────────────────────────────────────────────────────────


  Level                         Energy (uJ)  Latency (cc)        EDP         DRAM Reads       DRAM Writes
  Full Fusion (1 ok)            5.062e+04    3.042e+06           1.60e+05    11,032,512       25,088
  Sum Partial Fusion (4 ok)     5.007e+04    3.086e+06           3.64e+04    11,383,744       376,320

  RATIOS (Full Fusion / Sum Partial Fusion)  — values < 1.0 mean full fusion wins
    Energy:       1.0110  (-1.1%)     ← virtually identical energy
    Latency:      0.9859  (+1.4%)     ← full fusion marginally faster
    EDP:          4.4094  (-340.9%)   ← worse EDP for full fusion
    DRAM Reads:   0.9691  (+3.1%)     ← full fusion reads ~3% less
    DRAM Writes:  0.0667  (+93.3%)    ← full fusion writes 93% less to DRAM

  Partial-fusion block segment breakdown:
    block/stage1:    E=1.478e+04 uJ, L=2.885e+05 cc, EDP=4.61e+03  (5 layers)
    block/stage2_b2: E=1.281e+04 uJ, L=1.792e+05 cc, EDP=2.34e+03  (4 layers)
    block/stage3_b2: E=1.147e+04 uJ, L=5.412e+05 cc, EDP=6.28e+03  (4 layers)
    block/stage4_b2: E=1.101e+04 uJ, L=2.077e+06 cc, EDP=2.32e+04  (4 layers)

  Notes (2layer vs block comparison):
  - Block fusion (4 segments, 4-5L each) has slightly HIGHER energy than
    2layer (4 segments × ~1.25e+04 ≈ 5.007e+04 vs 8 segments × ~6.2e+03
    ≈ 4.972e+04). The larger per-segment weight volume increases
    per-access cost despite fewer segments.
  - Block fusion writes less to DRAM than 2layer (376K vs 753K) because
    fewer inter-segment boundaries means fewer intermediate write-backs.
  - Full fusion wins on DRAM writes in both cases (+96.7% vs +93.3%).
  - The energy ratio converges: full/2layer = 1.018, full/block = 1.011.
    As the partial segment size grows closer to full, the gap narrows.  

    
  ── partial_vs_single: 2 layer ────────────────────────────────────────────────────────

  
  
  python3 experiment_runner.py --compare-partial-vs-single -w resnet18 \
      --arch-type eyeriss --pe-rows 512 --pe-cols 32 --weight-reg 384 \
      --gb-size 128 --tile-size 1 \
      --input-reg 700 --intermediate-reg 300 --output-reg 64

    Partial: 8 runs (2-layer blocks).  Singles: 17 individual layer runs.

    Results:
      Level                        Energy (μJ)   Latency (cc)      EDP
      Sum Partial Fusion (8 ok)     3.662e+04     3.139e+06      1.32e+04
      Sum Singles (17 ok)           2.642e+03     3.301e+06      4.05e+02

    Ratios (Sum Partial / Sum Singles) — values < 1.0 mean fusion wins:
      Energy      13.86×     −1286.3 %
      Latency      0.95×        +4.9 %
      EDP         32.53×     −3117.2 %
      DRAM Reads   0.94×        +5.5 %
      DRAM Writes  0.33×       +67.4 %

    Baseline Detail (Sum Singles):
      Layer               Energy (μJ)   Latency (cc)      EDP
      L0_conv1              1.738e+02     2.007e+05     3.49e+01
      L1_conv2_1_1          3.447e+02     5.939e+04     2.05e+01
      L2_conv2_1_2          3.447e+02     5.939e+04     2.05e+01
      L3_conv2_2_1          3.447e+02     5.939e+04     2.05e+01
      L4_conv2_2_2          3.447e+02     5.939e+04     2.05e+01
      L5_conv3_1_1          9.773e+01     3.098e+04     3.05e+00
      L6_conv3_1_2          1.174e+02     6.195e+04     7.37e+00
      L8_conv3_2_1          1.174e+02     6.195e+04     7.37e+00
      L9_conv3_2_2          1.174e+02     6.195e+04     7.37e+00
      L10_conv4_1_1         4.628e+01     8.000e+04     3.94e+00
      L11_conv4_1_2         7.527e+01     1.600e+05     1.30e+01
      L13_conv4_2_1         7.527e+01     1.600e+05     1.30e+01
      L14_conv4_2_2         7.527e+01     1.600e+05     1.30e+01
      L15_conv5_1_1         5.622e+01     2.980e+05     2.03e+01
      L16_conv5_1_2         1.038e+02     5.961e+05     7.59e+01
      L18_conv5_2_1         1.038e+02     5.961e+05     7.59e+01
      L19_conv5_2_2         1.038e+02     5.961e+05     7.59e+01

    Partial Fusion Segment Detail:
      Segment         Energy (μJ)   Latency (cc)      EDP
      2layer/s1b1      4.129e+03     1.380e+05      7.90e+02
      2layer/s1b2      6.808e+03     1.597e+05      1.50e+03
      2layer/s2b1      5.911e+03     1.055e+05      8.43e+02
      2layer/s2b2      3.793e+03     9.882e+04      4.81e+02
      2layer/s3b1      5.030e+03     2.463e+05      1.74e+03
      2layer/s3b2      3.538e+03     3.075e+05      1.41e+03
      2layer/s4b1      4.702e+03     8.973e+05      4.26e+03
      2layer/s4b2      3.510e+03     1.186e+06      3.83e+03

    Analysis:
      2-layer fusion is 13× MORE expensive in energy than singles.
      Same fundamental issue as full fusion: even with only 2 layers
      fused, each PE must cycle through both layers' C×R weight factors
      at the WRegister level, inflating register WMOPs by ~19×.

      Latency improves 4.9 % (3.14M vs 3.30M cc) — inter-layer DRAM
      round-trips halved (DRAM Writes −67.4 %).

      DRAM Reads down 5.5 % (11.8M vs 12.4M) — modest savings from
      keeping one intermediate activation on-chip per 2-layer block.

      EDP: 32× worse for partial fusion, entirely driven by the energy
      penalty. The latency savings are negligible compared to the
      enormous register-level energy cost of multi-layer weight storage.





----------------------------------------------------------------------------------------------
      
    Cross-comparison (all three experiments at this config):
      Metric    Full(17L)    Partial(2L×8)   Singles(1L×17)
      Energy    5.062e+04    4.972e+04        2.643e+03  μJ
      Latency   3.042e+06    3.139e+06        3.301e+06  cc
      EDP       1.60e+05     1.83e+04         4.33e+02

      Singles are the clear winner in energy and EDP.
      Full fusion has the lowest latency (−7.8 % vs singles) thanks to
      complete DRAM write elimination, but the 19× energy penalty
      makes it impractical.
      Partial fusion sits in between — marginal latency gains, nearly
      the same energy penalty as full fusion.

      Conclusion: For Eyeriss with WReg=902 at 16,384 PEs, layer fusion
      is counterproductive. The register-level weight cycling overhead
      dominates all DRAM savings. Layer-by-layer execution (singles)
      is optimal in both energy and EDP.





────────────────────────────────────────────────────────────────────────────
RESULTS — ResNet18, Eyeriss 256×32, WReg=1800
────────────────────────────────────────────────────────────────────────────

  Architecture: 256×32 = 8192 PEs, WReg=1800, GB=128KB
  --arch-type eyeriss --pe-rows 256 --pe-cols 32 --weight-reg 1800
  --gb-size 128 --tile-size 1
  --input-reg 700 --intermediate-reg 300 --output-reg 64

  ── full_vs_single ─────────────────────────────────────────────────────────

  --compare-full-vs-single

  Level                    Energy (uJ)  Latency (cc)        EDP         DRAM Reads  DRAM Writes
  Full Fusion (1 ok)       7.005e+04    3.042e+06           2.18e+05    11,032,512       25,088
  Sum Singles (17 ok)      1.920e+03    3.301e+06           4.30e+02    12,449,984    2,308,096

  RATIOS (Full Fusion / Sum Singles)  — values < 1.0 mean fusion wins
    Energy:      36.4914  (-3549.1%)  ← full fusion MUCH MORE expensive
    Latency:      0.9215  (+7.8%)     ← full fusion slightly faster
    EDP:        505.6325  (-50463.3%) ← massively worse EDP
    DRAM Reads:   0.8861  (+11.4%)    ← full fusion reads less from DRAM
    DRAM Writes:  0.0109  (+98.9%)    ← full fusion writes almost nothing to DRAM

  Single-layer breakdown:
    L0_conv1:       E=1.746e+02 uJ, L=2.007e+05 cc
    L1_conv2_1_1:   E=1.956e+02 uJ, L=5.939e+04 cc
    L2_conv2_1_2:   E=1.956e+02 uJ, L=5.939e+04 cc
    L3_conv2_2_1:   E=1.956e+02 uJ, L=5.939e+04 cc
    L4_conv2_2_2:   E=1.956e+02 uJ, L=5.939e+04 cc
    L5_conv3_1_1:   E=6.053e+01 uJ, L=3.098e+04 cc
    L6_conv3_1_2:   E=1.022e+02 uJ, L=6.195e+04 cc
    L8_conv3_2_1:   E=1.022e+02 uJ, L=6.195e+04 cc
    L9_conv3_2_2:   E=1.022e+02 uJ, L=6.195e+04 cc
    L10_conv4_1_1:  E=3.866e+01 uJ, L=8.000e+04 cc
    L11_conv4_1_2:  E=6.789e+01 uJ, L=1.600e+05 cc
    L13_conv4_2_1:  E=6.789e+01 uJ, L=1.600e+05 cc
    L14_conv4_2_2:  E=6.789e+01 uJ, L=1.600e+05 cc
    L15_conv5_1_1:  E=5.250e+01 uJ, L=2.980e+05 cc
    L16_conv5_1_2:  E=1.003e+02 uJ, L=5.961e+05 cc
    L18_conv5_2_1:  E=1.003e+02 uJ, L=5.961e+05 cc
    L19_conv5_2_2:  E=1.003e+02 uJ, L=5.961e+05 cc

  Notes:
  - Energy ratio is 36.5× — much worse than 512×32/WReg=902
    (19.2×) because the larger WReg inflates per-access energy further.
  - DRAM traffic ratios (Reads=0.8861, Writes=0.0109) and latency ratio
    (0.9215) are IDENTICAL to 512×32 — workload-determined, not shape-dep.

    
  ── full_vs_partial (2layer) ───────────────────────────────────────────────

  --compare-full-vs-partial  (intermediate = 2layer, 8 segments)

  Level                         Energy (uJ)  Latency (cc)        EDP    DRAM Reads  DRAM Writes
  Full Fusion (1 ok)            7.005e+04    3.042e+06    2.18e+05    11,032,512       25,088
  Sum Partial Fusion (8 ok)     7.017e+04    3.139e+06    2.65e+04    11,760,064      752,640

  RATIOS (Full Fusion / Sum Partial Fusion)  — values < 1.0 mean full fusion wins
    Energy:       0.9983  (+0.2%)     ← FULL FUSION WINS on energy (barely)
    Latency:      0.9692  (+3.1%)     ← full fusion slightly faster
    EDP:          8.2036  (-720.4%)   ← worse EDP for full fusion
    DRAM Reads:   0.9381  (+6.2%)     ← full fusion reads ~6% less
    DRAM Writes:  0.0333  (+96.7%)    ← full fusion writes 97% less to DRAM

  Partial-fusion 2layer segment breakdown:
    2layer/s1b1:  E=7.722e+03 uJ, L=1.380e+05 cc, EDP=1.09e+03
    2layer/s1b2:  E=1.117e+04 uJ, L=1.597e+05 cc, EDP=1.83e+03
    2layer/s2b1:  E=1.072e+04 uJ, L=1.055e+05 cc, EDP=1.14e+03
    2layer/s2b2:  E=6.932e+03 uJ, L=9.882e+04 cc, EDP=6.93e+02
    2layer/s3b1:  E=1.017e+04 uJ, L=2.463e+05 cc, EDP=2.52e+03
    2layer/s3b2:  E=6.742e+03 uJ, L=3.075e+05 cc, EDP=2.09e+03
    2layer/s4b1:  E=9.975e+03 uJ, L=8.973e+05 cc, EDP=9.03e+03
    2layer/s4b2:  E=6.746e+03 uJ, L=1.186e+06 cc, EDP=8.13e+03

  Notes:
  - At WReg=1800, full fusion is CHEAPER on energy than 2layer sum by 0.2%.
    This is the first config where full fusion actually wins on energy!
    The oversized WReg penalises partial fusion proportionally more because
    8 separate segment runs each pay full per-access cost for unused
    register capacity, whereas full fusion pays it once but across more
    layers (amortised).
  - DRAM write savings remain +96.7% for full fusion.

  ── full_vs_partial (block) ────────────────────────────────────────────────

  --compare-full-vs-partial  (intermediate = block, 4 segments)


  Level                         Energy (uJ)  Latency (cc)        EDP    DRAM Reads  DRAM Writes
  Full Fusion (1 ok)            7.005e+04    3.042e+06    2.18e+05    11,032,512       25,088
  Sum Partial Fusion (4 ok)     7.000e+04    3.086e+06    5.27e+04    11,383,744      376,320

  RATIOS (Full Fusion / Sum Partial Fusion)  — values < 1.0 mean full fusion wins
    Energy:       1.0008  (-0.1%)     ← virtually identical energy
    Latency:      0.9859  (+1.4%)     ← full fusion marginally faster
    EDP:          4.1254  (-312.5%)   ← worse EDP for full fusion
    DRAM Reads:   0.9691  (+3.1%)     ← full fusion reads ~3% less
    DRAM Writes:  0.0667  (+93.3%)    ← full fusion writes 93% less to DRAM

  Partial-fusion block segment breakdown:
    block/stage1:    E=1.913e+04 uJ, L=2.885e+05 cc, EDP=5.69e+03  (5 layers)
    block/stage2_b2: E=1.762e+04 uJ, L=1.792e+05 cc, EDP=3.20e+03  (4 layers)
    block/stage3_b2: E=1.675e+04 uJ, L=5.412e+05 cc, EDP=9.15e+03  (4 layers)
    block/stage4_b2: E=1.649e+04 uJ, L=2.077e+06 cc, EDP=3.47e+04  (4 layers)

  Notes:
  - Block fusion sum energy (7.000e+04) is actually slightly LOWER than
    full fusion (7.005e+04), by just 0.08%. Both are effectively equal.
  - At WReg=1800 the per-access energy penalty is high enough that
    fewer segments (4 blocks vs 8 two-layer) gives the block fusion a
    marginal edge: fewer total register accesses across fewer runs.
  - DRAM writes: block writes 376K vs 2layer's 753K — half as many
    inter-segment boundaries means half the intermediate write-backs.

    
  ── partial_vs_single ────────────────────────────────────────────────


  
  python3 experiment_runner.py --compare-partial-vs-single -w resnet18  \
            --arch-type eyeriss --pe-rows 256 --pe-cols 32 --weight-reg 770  \
            --gb-size 128 --tile-size 1       --input-reg 700 --intermediate-reg 300 --output-reg 64
  
  Level                        Energy (uJ)   Latency (cc)            EDP       DRAM Reads      DRAM Writes
  ----------------------------------------------------------------------------------------------------
  Sum Partial Fusion (8 ok)      4.412e+04      3.139e+06       1.64e+04       11,760,064          752,640
  Sum Singles (17 ok)            1.916e+03      3.301e+06       3.74e+02       12,449,984        2,308,096

  ----------------------------------------------------------------------------------------------------
  RATIOS (Sum Partial Fusion / Sum Singles) -- values < 1.0 mean fusion wins
  ----------------------------------------------------------------------------------------------------
  Metric                    Ratio      Savings
  ---------------------------------------------
  Energy                  23.0295     -2202.9%
  Latency                  0.9508        +4.9%
  EDP                     43.8576     -4285.8%
  DRAM Reads               0.9446        +5.5%
  DRAM Writes              0.3261       +67.4%
            

  ----------------------------------------------------------------------------------------------------
  BASELINE DETAIL (Sum Singles):
  ----------------------------------------------------------------------------------------------------
  Layer/Segment                     Energy (uJ)   Latency (cc)            EDP     Status
  -------------------------------------------------------------------------------------
  L0_conv1                            1.736e+02      2.007e+05       3.49e+01         OK
  L1_conv2_1_1                        1.954e+02      5.939e+04       1.16e+01         OK
  L2_conv2_1_2                        1.954e+02      5.939e+04       1.16e+01         OK
  L3_conv2_2_1                        1.954e+02      5.939e+04       1.16e+01         OK
  L4_conv2_2_2                        1.954e+02      5.939e+04       1.16e+01         OK
  L5_conv3_1_1                        6.042e+01      3.098e+04       1.89e+00         OK
  L6_conv3_1_2                        1.020e+02      6.195e+04       6.40e+00         OK
  L8_conv3_2_1                        1.020e+02      6.195e+04       6.40e+00         OK
  L9_conv3_2_2                        1.020e+02      6.195e+04       6.40e+00         OK
  L10_conv4_1_1                       3.855e+01      8.000e+04       3.29e+00         OK
  L11_conv4_1_2                       6.768e+01      1.600e+05       1.16e+01         OK
  L13_conv4_2_1                       6.768e+01      1.600e+05       1.16e+01         OK
  L14_conv4_2_2                       6.768e+01      1.600e+05       1.16e+01         OK
  L15_conv5_1_1                       5.239e+01      2.980e+05       1.86e+01         OK
  L16_conv5_1_2                       1.001e+02      5.961e+05       7.16e+01         OK
  L18_conv5_2_1                       1.001e+02      5.961e+05       7.16e+01         OK
  L19_conv5_2_2                       1.001e+02      5.961e+05       7.16e+01         OK

  ----------------------------------------------------------------------------------------------------
  FUSED-SIDE SEGMENT DETAIL (Sum Partial Fusion):
  ----------------------------------------------------------------------------------------------------
  Segment                           Energy (uJ)   Latency (cc)            EDP     Status
  -------------------------------------------------------------------------------------
  2layer/s1b1                         5.091e+03      1.380e+05       7.23e+02         OK
  2layer/s1b2                         7.261e+03      1.597e+05       1.21e+03         OK
  2layer/s2b1                         6.812e+03      1.055e+05       7.27e+02         OK
  2layer/s2b2                         4.329e+03      9.882e+04       4.35e+02         OK
  2layer/s3b1                         6.268e+03      2.463e+05       1.55e+03         OK
  2layer/s3b2                         4.139e+03      3.075e+05       1.29e+03         OK
  2layer/s4b1                         6.070e+03      8.973e+05       5.49e+03         OK
  2layer/s4b2                         4.143e+03      1.186e+06       4.98e+03         OK



  ── Cross-comparison across all three modes (256×32, WReg=1800) ────────────

                              Energy (uJ)  Latency (cc)   DRAM Writes
  Full Fusion (17L)           7.005e+04    3.042e+06          25,088
  Sum 2layer (8 seg)          7.017e+04    3.139e+06         752,640
  Sum block  (4 seg)          7.000e+04    3.086e+06         376,320
  Sum Singles (17 ind)        1.920e+03    3.301e+06       2,308,096

  Key takeaway: At WReg=1800 on 256×32, all three fusion granularities
  converge to ~7.0e+04 uJ energy — the oversized WReg dominates the
  energy cost and the fusion level barely matters. The real differentiator
  is DRAM writes: full=25K, block=376K, 2layer=753K, singles=2.3M.
  Singles remain dramatically cheaper on energy (1.9e+03 uJ) because
  each layer does far fewer register accesses despite using the same
  oversized WReg.    


  
  ────────────────────────────────────────────────────────────────────────────
  RESULTS — ResNet18, Eyeriss 256×16, WReg=3600
  ────────────────────────────────────────────────────────────────────────────

  Architecture: 256×16 = 4096 PEs, WReg=3600, GB=128KB
  --arch-type eyeriss --pe-rows 256 --pe-cols 16 --weight-reg 3600
  --gb-size 128 --tile-size 1
  --input-reg 700 --intermediate-reg 300 --output-reg 64

  ── full_vs_single ─────────────────────────────────────────────────────────

  --compare-full-vs-single

  Level                    Energy (uJ)  Latency (cc)        EDP         DRAM Reads  DRAM Writes
  Full Fusion (1 ok)       1.137e+05    3.059e+06           3.51e+05    11,032,512       25,088
  Sum Singles (17 ok)      1.412e+03    3.301e+06           4.78e+02    12,449,984    2,308,096

  RATIOS (Full Fusion / Sum Singles)  — values < 1.0 mean fusion wins
    Energy:      80.5538  (-7955.4%)  ← full fusion MUCH MORE expensive
    Latency:      0.9265  (+7.4%)     ← full fusion slightly faster
    EDP:        734.5104  (-73351.0%) ← massively worse EDP
    DRAM Reads:   0.8861  (+11.4%)    ← full fusion reads less from DRAM
    DRAM Writes:  0.0109  (+98.9%)    ← full fusion writes almost nothing to DRAM

  Single-layer breakdown:
    L0_conv1:       E=1.763e+02 uJ, L=2.007e+05 cc
    L1_conv2_1_1:   E=1.221e+02 uJ, L=5.939e+04 cc
    L2_conv2_1_2:   E=1.221e+02 uJ, L=5.939e+04 cc
    L3_conv2_2_1:   E=1.221e+02 uJ, L=5.939e+04 cc
    L4_conv2_2_2:   E=1.221e+02 uJ, L=5.939e+04 cc
    L5_conv3_1_1:   E=4.247e+01 uJ, L=3.098e+04 cc
    L6_conv3_1_2:   E=6.609e+01 uJ, L=6.195e+04 cc
    L8_conv3_2_1:   E=6.609e+01 uJ, L=6.195e+04 cc
    L9_conv3_2_2:   E=6.609e+01 uJ, L=6.195e+04 cc
    L10_conv4_1_1:  E=2.996e+01 uJ, L=8.000e+04 cc
    L11_conv4_1_2:  E=5.048e+01 uJ, L=1.600e+05 cc
    L13_conv4_2_1:  E=5.048e+01 uJ, L=1.600e+05 cc
    L14_conv4_2_2:  E=5.048e+01 uJ, L=1.600e+05 cc
    L15_conv5_1_1:  E=4.847e+01 uJ, L=2.980e+05 cc
    L16_conv5_1_2:  E=9.222e+01 uJ, L=5.961e+05 cc
    L18_conv5_2_1:  E=9.222e+01 uJ, L=5.961e+05 cc
    L19_conv5_2_2:  E=9.222e+01 uJ, L=5.961e+05 cc

  Notes:
  - With only 4096 PEs (256×16) and WReg=3600, the energy ratio explodes
    to 80.6× — by far the worst across all tested configurations.
  - WReg=3600 is ~4× the minimum needed for full fusion, so per-access
    energy is extremely high and each PE pays it.
  - Singles sum drops to 1.412e+03 uJ (lowest yet) because individual
    layers barely touch the oversized WReg.
  - Latency ratio (0.9265, +7.4%) is slightly different from 16384-PE
    configs (0.9215, +7.8%) because the smaller PE array changes the
    tiling and scheduling.

  ── full_vs_partial (2layer) ───────────────────────────────────────────────

  --compare-full-vs-partial  (intermediate = 2layer, 8 segments)

  Level                         Energy (uJ)  Latency (cc)        EDP    DRAM Reads  DRAM Writes
  Full Fusion (1 ok)            1.137e+05    3.059e+06    3.51e+05    11,032,512       25,088
  Sum Partial Fusion (8 ok)     1.144e+05    3.147e+06    4.40e+04    11,760,064      752,640

  RATIOS (Full Fusion / Sum Partial Fusion)  — values < 1.0 mean full fusion wins
    Energy:       0.9942  (+0.6%)     ← FULL FUSION WINS on energy (slightly)
    Latency:      0.9719  (+2.8%)     ← full fusion slightly faster
    EDP:          7.9869  (-698.7%)   ← worse EDP for full fusion
    DRAM Reads:   0.9381  (+6.2%)     ← full fusion reads ~6% less
    DRAM Writes:  0.0333  (+96.7%)    ← full fusion writes 97% less to DRAM

  Partial-fusion 2layer segment breakdown:
    2layer/s1b1:  E=1.232e+04 uJ, L=1.380e+05 cc, EDP=1.71e+03
    2layer/s1b2:  E=1.747e+04 uJ, L=1.597e+05 cc, EDP=2.81e+03
    2layer/s2b1:  E=1.724e+04 uJ, L=1.129e+05 cc, EDP=1.95e+03
    2layer/s2b2:  E=1.133e+04 uJ, L=9.958e+04 cc, EDP=1.13e+03
    2layer/s3b1:  E=1.685e+04 uJ, L=2.463e+05 cc, EDP=4.16e+03
    2layer/s3b2:  E=1.122e+04 uJ, L=3.075e+05 cc, EDP=3.47e+03
    2layer/s4b1:  E=1.672e+04 uJ, L=8.973e+05 cc, EDP=1.51e+04
    2layer/s4b2:  E=1.126e+04 uJ, L=1.186e+06 cc, EDP=1.36e+04

  Notes:
  - Full fusion wins on energy by +0.6% — consistent with the trend that
    larger WReg makes full fusion's energy competitive because partial
    segments also pay the oversized-WReg penalty per-run.
  - Latency ratio (0.9719) is slightly different from 16384-PE configs
    (0.9692) due to the smaller 4096-PE array tiling differently.

  ── full_vs_partial (block) ────────────────────────────────────────────────

  --compare-full-vs-partial --intermediate-level block  (4 segments)

  Level                         Energy (uJ)  Latency (cc)        EDP    DRAM Reads  DRAM Writes
  Full Fusion (1 ok)            1.137e+05    3.059e+06    3.51e+05    11,032,512       25,088
  Sum Partial Fusion (4 ok)     1.139e+05    3.095e+06    8.76e+04    11,383,744      376,320

  RATIOS (Full Fusion / Sum Partial Fusion)  — values < 1.0 mean full fusion wins
    Energy:       0.9982  (+0.2%)     ← full fusion wins on energy (barely)
    Latency:      0.9883  (+1.2%)     ← full fusion marginally faster
    EDP:          4.0096  (-301.0%)   ← worse EDP for full fusion
    DRAM Reads:   0.9691  (+3.1%)     ← full fusion reads ~3% less
    DRAM Writes:  0.0667  (+93.3%)    ← full fusion writes 93% less to DRAM

  Partial-fusion block segment breakdown:
    block/stage1:    E=2.988e+04 uJ, L=2.885e+05 cc, EDP=8.71e+03  (5 layers)
    block/stage2_b2: E=2.847e+04 uJ, L=1.882e+05 cc, EDP=5.38e+03  (4 layers)
    block/stage3_b2: E=2.786e+04 uJ, L=5.412e+05 cc, EDP=1.52e+04  (4 layers)
    block/stage4_b2: E=2.773e+04 uJ, L=2.077e+06 cc, EDP=5.83e+04  (4 layers)

  Notes:
  - Block fusion sum (1.139e+05) is almost identical to full (1.137e+05).
  - The gap between full and partial narrows as segment size increases:
    full/2layer ratio = 0.9942, full/block ratio = 0.9982 — block fusion
    is closer to full fusion because 4-5 layer segments amortise the WReg
    penalty more effectively than 2-layer segments.


  ── partial_vs_single ──────────────────────────────────────────────────────

      --compare-partial-vs-single -w resnet18 --arch-type eyeriss
      --pe-rows 256 --pe-cols 16 --weight-reg 1600
      --gb-size 128 --tile-size 1
      --input-reg 700 --intermediate-reg 300 --output-reg 64

  Level                         Energy (uJ)  Latency (cc)        EDP    DRAM Reads  DRAM Writes
  Sum Partial Fusion (8 ok)     6.380e+04    3.147e+06    2.43e+04    11,760,064      752,640
  Sum Singles (17 ok)           1.398e+03    3.301e+06    3.68e+02    12,449,984    2,308,096

  RATIOS (Sum Partial Fusion / Sum Singles)  — values < 1.0 mean fusion wins
    Energy:      45.6366  (-4463.7%)  ← partial fusion MUCH WORSE on energy
    Latency:      0.9533  (+4.7%)     ← partial fusion slightly faster
    EDP:         65.9746  (-6497.5%)  ← much worse EDP
    DRAM Reads:   0.9446  (+5.5%)     ← slightly fewer DRAM reads
    DRAM Writes:  0.3261  (+67.4%)    ← 67% fewer DRAM writes

  Partial-fusion 2layer segment breakdown:
    2layer/s1b1:  E=7.211e+03 uJ, L=1.380e+05 cc, EDP=1.01e+03
    2layer/s1b2:  E=9.884e+03 uJ, L=1.597e+05 cc, EDP=1.60e+03
    2layer/s2b1:  E=9.660e+03 uJ, L=1.129e+05 cc, EDP=1.10e+03
    2layer/s2b2:  E=6.277e+03 uJ, L=9.958e+04 cc, EDP=6.29e+02
    2layer/s3b1:  E=9.265e+03 uJ, L=2.463e+05 cc, EDP=2.29e+03
    2layer/s3b2:  E=6.162e+03 uJ, L=3.075e+05 cc, EDP=1.91e+03
    2layer/s4b1:  E=9.142e+03 uJ, L=8.973e+05 cc, EDP=8.27e+03
    2layer/s4b2:  E=6.204e+03 uJ, L=1.186e+06 cc, EDP=7.47e+03

  Single-layer breakdown:
    L0_conv1:       E=1.744e+02 uJ, L=2.007e+05 cc
    L1_conv2_1_1:   E=1.213e+02 uJ, L=5.939e+04 cc
    L2_conv2_1_2:   E=1.213e+02 uJ, L=5.939e+04 cc
    L3_conv2_2_1:   E=1.213e+02 uJ, L=5.939e+04 cc
    L4_conv2_2_2:   E=1.213e+02 uJ, L=5.939e+04 cc
    L5_conv3_1_1:   E=4.206e+01 uJ, L=3.098e+04 cc
    L6_conv3_1_2:   E=6.527e+01 uJ, L=6.195e+04 cc
    L8_conv3_2_1:   E=6.527e+01 uJ, L=6.195e+04 cc
    L9_conv3_2_2:   E=6.527e+01 uJ, L=6.195e+04 cc
    L10_conv4_1_1:  E=2.954e+01 uJ, L=8.000e+04 cc
    L11_conv4_1_2:  E=4.966e+01 uJ, L=1.600e+05 cc
    L13_conv4_2_1:  E=4.966e+01 uJ, L=1.600e+05 cc
    L14_conv4_2_2:  E=4.966e+01 uJ, L=1.600e+05 cc
    L15_conv5_1_1:  E=4.806e+01 uJ, L=2.980e+05 cc
    L16_conv5_1_2:  E=9.140e+01 uJ, L=5.961e+05 cc
    L18_conv5_2_1:  E=9.140e+01 uJ, L=5.961e+05 cc
    L19_conv5_2_2:  E=9.140e+01 uJ, L=5.961e+05 cc

  Notes:
  - 256×16 = 4096 PEs with WReg=1600 
  - Energy ratio 45.6× is the WORST partial-vs-single ratio tested,
    due to the combination of small PE count and oversized WReg.
  - Singles sum (1.398e+03 uJ) is the lowest across partial-vs-single
    configs because fewer PEs means fewer parallel accesses, and the
    per-access cost is still relatively small for singles even at
    WReg=1600.
  - Partial sum (6.380e+04 uJ) is significantly higher than 512×32/
    WReg=384 (3.662e+04) — the oversized WReg inflates each segment's
    per-access energy.
  - Latency ratio (0.9533, +4.7%) is slightly different from the
    0.9508 (+4.9%) seen on all other configs — the 4096-PE array
    changes the tiling for some 2layer segments (e.g. s2b1 latency
    is 1.129e+05 vs 1.055e+05 on larger arrays).
  - DRAM traffic ratios remain identical (Reads=0.9446, Writes=0.3261)
    — purely workload-determined.
  - Comparison across PE counts at ~2× WReg overhead for 2layer:
      512×32 (16384 PEs), WReg=700:  ratio=16.9×
      512×16 (8192 PEs),  WReg=770:  ratio=24.6×
      256×32 (8192 PEs),  WReg=1350: ratio=30.7×
      256×16 (4096 PEs),  WReg=1600: ratio=45.6×
    → Energy ratio scales roughly with WReg / PE_count. Fewer PEs +
      larger WReg is the worst combination.



    
  ── Cross-comparison across all three modes (256×16, WReg=3600) ────────────

                              Energy (uJ)  Latency (cc)   DRAM Writes
  Full Fusion (17L)           1.137e+05    3.059e+06          25,088
  Sum 2layer (8 seg)          1.144e+05    3.147e+06         752,640
  Sum block  (4 seg)          1.139e+05    3.095e+06         376,320
  Sum Singles (17 ind)        1.412e+03    3.301e+06       2,308,096

  Key takeaway: With WReg=3600 on only 4096 PEs, the energy penalty is
  catastrophic for any fusion (80× vs singles). However, all fusion
  granularities converge to ~1.14e+05 uJ — the oversized WReg completely
  dominates. Full fusion is the best among fused options by a tiny margin
  (+0.2-0.6% energy savings), and provides by far the best DRAM write
  reduction (25K vs 376K-753K-2.3M). The latency advantage of fusion
  also decreases at smaller PE counts (+7.4% vs singles, +1.2-2.8% vs
  partial).


  ────────────────────────────────────────────────────────────────────────────
  RESULTS — ResNet18, Eyeriss 128×16, WReg=7200             SUPERFLUO, USELESS
  ────────────────────────────────────────────────────────────────────────────

  Architecture: 128×16 = 2048 PEs, WReg=7200, GB=128KB
  --arch-type eyeriss --pe-rows 128 --pe-cols 16 --weight-reg 7200
  --gb-size 128 --tile-size 1
  --input-reg 700 --intermediate-reg 300 --output-reg 64

  ── full_vs_single ─────────────────────────────────────────────────────────

  --compare-full-vs-single

  Level                    Energy (uJ)  Latency (cc)        EDP    DRAM Reads  DRAM Writes
  Full Fusion (1 ok)       2.035e+05    3.422e+06    7.02e+05    11,032,512       25,088
  Sum Singles (17 ok)      1.368e+03    3.412e+06    6.85e+02    12,449,984    2,308,096

  RATIOS (Full Fusion / Sum Singles)  — values < 1.0 mean fusion wins
    Energy:     148.7670  (-14776.7%) ← full fusion catastrophically expensive
    Latency:      1.0032  (-0.3%)     ← FULL FUSION IS SLOWER (first time!)
    EDP:       1024.1983  (-102319.8%)← worst EDP across all configs
    DRAM Reads:   0.8861  (+11.4%)    ← full fusion reads less from DRAM
    DRAM Writes:  0.0109  (+98.9%)    ← full fusion writes almost nothing to DRAM

  Single-layer breakdown:
    L0_conv1:       E=1.798e+02 uJ, L=2.007e+05 cc
    L1_conv2_1_1:   E=1.113e+02 uJ, L=7.526e+04 cc
    L2_conv2_1_2:   E=1.113e+02 uJ, L=7.526e+04 cc
    L3_conv2_2_1:   E=1.113e+02 uJ, L=7.526e+04 cc
    L4_conv2_2_2:   E=1.113e+02 uJ, L=7.526e+04 cc
    L5_conv3_1_1:   E=3.707e+01 uJ, L=3.763e+04 cc
    L6_conv3_1_2:   E=6.309e+01 uJ, L=7.526e+04 cc
    L8_conv3_2_1:   E=6.309e+01 uJ, L=7.526e+04 cc
    L9_conv3_2_2:   E=6.309e+01 uJ, L=7.526e+04 cc
    L10_conv4_1_1:  E=2.845e+01 uJ, L=8.000e+04 cc
    L11_conv4_1_2:  E=5.138e+01 uJ, L=1.600e+05 cc
    L13_conv4_2_1:  E=5.138e+01 uJ, L=1.600e+05 cc
    L14_conv4_2_2:  E=5.138e+01 uJ, L=1.600e+05 cc
    L15_conv5_1_1:  E=4.892e+01 uJ, L=2.980e+05 cc
    L16_conv5_1_2:  E=9.508e+01 uJ, L=5.961e+05 cc
    L18_conv5_2_1:  E=9.508e+01 uJ, L=5.961e+05 cc
    L19_conv5_2_2:  E=9.508e+01 uJ, L=5.961e+05 cc

  Notes:
  - FIRST CONFIG WHERE FULL FUSION IS SLOWER THAN SINGLES. With only
    2048 PEs, the PE array is too small: the tiling overhead for 17
    fused layers overwhelms the DRAM-bandwidth savings, causing latency
    to increase by 0.3% instead of decreasing.
  - Energy ratio is 148.8× — nearly double the 256×16 config (80.6×).
    WReg=7200 is ~8× the minimum needed for full fusion, so per-access
    energy is enormous.
  - Singles latency changes at this PE count: some layers now show
    L=7.526e+04 cc (vs 5.939e+04 on larger PE arrays), indicating
    different tiling with fewer PEs.

  ── full_vs_partial (2layer) ───────────────────────────────────────────────

  --compare-full-vs-partial  (intermediate = 2layer, 8 segments)

  Level                         Energy (uJ)  Latency (cc)        EDP    DRAM Reads  DRAM Writes
  Full Fusion (1 ok)            2.035e+05    3.422e+06    7.02e+05    11,032,512       25,088
  Sum Partial Fusion (8 ok)     2.046e+05    3.454e+06    8.79e+04    11,760,064      752,640

  RATIOS (Full Fusion / Sum Partial Fusion)  — values < 1.0 mean full fusion wins
    Energy:       0.9944  (+0.6%)     ← full fusion wins on energy (slightly)
    Latency:      0.9909  (+0.9%)     ← full fusion slightly faster
    EDP:          7.9824  (-698.2%)   ← worse EDP for full fusion
    DRAM Reads:   0.9381  (+6.2%)     ← full fusion reads ~6% less
    DRAM Writes:  0.0333  (+96.7%)    ← full fusion writes 97% less to DRAM

  Partial-fusion 2layer segment breakdown:
    2layer/s1b1:  E=2.151e+04 uJ, L=1.631e+05 cc, EDP=3.52e+03
    2layer/s1b2:  E=3.085e+04 uJ, L=2.258e+05 cc, EDP=7.00e+03
    2layer/s2b1:  E=3.063e+04 uJ, L=2.258e+05 cc, EDP=6.93e+03
    2layer/s2b2:  E=2.036e+04 uJ, L=1.505e+05 cc, EDP=3.07e+03
    2layer/s3b1:  E=3.036e+04 uJ, L=2.980e+05 cc, EDP=9.07e+03
    2layer/s3b2:  E=2.028e+04 uJ, L=3.075e+05 cc, EDP=6.27e+03
    2layer/s4b1:  E=3.031e+04 uJ, L=8.973e+05 cc, EDP=2.75e+04
    2layer/s4b2:  E=2.034e+04 uJ, L=1.186e+06 cc, EDP=2.46e+04

  Notes:
  - Full fusion wins on energy by +0.6%, consistent with 256×16/WReg=3600.
  - The latency advantage of full fusion over 2layer (+0.9%) is smaller
    than at higher PE counts (+2.8-3.1%) because the small PE array
    reduces the DRAM-bandwidth-related latency savings.

  ── full_vs_partial (block) ────────────────────────────────────────────────

  --compare-full-vs-partial --intermediate-level block  (4 segments)

  Level                         Energy (uJ)  Latency (cc)        EDP    DRAM Reads  DRAM Writes
  Full Fusion (1 ok)            2.035e+05    3.422e+06    7.02e+05    11,032,512       25,088
  Sum Partial Fusion (4 ok)     2.039e+05    3.435e+06    1.76e+05    11,383,744      376,320

  RATIOS (Full Fusion / Sum Partial Fusion)  — values < 1.0 mean full fusion wins
    Energy:       0.9979  (+0.2%)     ← full fusion wins on energy (barely)
    Latency:      0.9963  (+0.4%)     ← full fusion marginally faster
    EDP:          3.9929  (-299.3%)   ← worse EDP for full fusion
    DRAM Reads:   0.9691  (+3.1%)     ← full fusion reads ~3% less
    DRAM Writes:  0.0667  (+93.3%)    ← full fusion writes 93% less to DRAM

  Partial-fusion block segment breakdown:
    block/stage1:    E=5.233e+04 uJ, L=3.889e+05 cc, EDP=2.05e+04  (5 layers)
    block/stage2_b2: E=5.082e+04 uJ, L=3.763e+05 cc, EDP=1.92e+04  (4 layers)
    block/stage3_b2: E=5.041e+04 uJ, L=5.929e+05 cc, EDP=3.00e+04  (4 layers)
    block/stage4_b2: E=5.038e+04 uJ, L=2.077e+06 cc, EDP=1.06e+05  (4 layers)

  Notes:
  - Block segments are remarkably uniform in energy (~5.0-5.2e+04 uJ)
    despite very different latencies. The oversized WReg=7200 dominates
    per-segment cost equally regardless of layer sizes.
  - The latency gap between full and block is only +0.4% — at 2048 PEs
    the scheduling gains from full fusion are nearly zero.


  ── partial_vs_single ────────────────────────────────────────────────




  ── Cross-comparison across all three modes (128×16, WReg=7200) ────────────

                              Energy (uJ)  Latency (cc)   DRAM Writes
  Full Fusion (17L)           2.035e+05    3.422e+06          25,088
  Sum 2layer (8 seg)          2.046e+05    3.454e+06         752,640
  Sum block  (4 seg)          2.039e+05    3.435e+06         376,320
  Sum Singles (17 ind)        1.368e+03    3.412e+06       2,308,096

  Key takeaway: At 2048 PEs with WReg=7200, this is the most extreme
  config tested. Energy ratio vs singles hits 149× (worst ever). Full
  fusion is now SLOWER than singles for the first time (latency ratio
  1.003). All fusion levels converge to ~2.04e+05 uJ — the massive
  WReg completely dominates. The only benefit of fusion at this config
  is DRAM write reduction (25K vs 2.3M). If DRAM write energy were
  modelled at the system level, that 98.9% reduction could offset some
  of the register-file penalty, but within the PE-array energy model
  the penalty is catastrophic.








  



  
END OF EYERISS RESNET18 RESULTS  
___________________________________________________________________________________________________
       





    ────────────────────────────────────────────────────────────────────────────
     FULL vs SINGLE — Vgg16 Eyeriss 512×32 (16384 PEs), WReg=1200
    ────────────────────────────────────────────────────────────────────────────
     python3 experiment_runner.py --compare-full-vs-single -w vgg16 \
         --arch-type eyeriss --pe-rows 512 --pe-cols 32 --weight-reg 1200 \
         --gb-size 128 --tile-size 1 --input-reg 700 --intermediate-reg 300 \
         --output-reg 64
     ----------------------------------------------------------------------------------
     Architecture: Eyeriss 512x32, GB=128KB, WReg=1200, InReg=700, IntReg=300, OutReg=64
        All experiments use IDENTICAL architecture (fair comparison).
    Full fusion: 1 run (13 layers fused).  Singles: 13 individual layer runs.

    Results:
      Level                   Energy (μJ)   Latency (cc)      EDP
      Full Fusion (1 ok)       3.618e+05     5.455e+06      2.03e+06
      Sum Singles (13 ok)      1.360e+04     6.922e+06      8.23e+03

    Ratios (Full Fusion / Sum Singles) — values < 1.0 mean fusion wins:
      Energy      26.60×     −2560.4 %
      Latency      0.79×       +21.2 %
      EDP        246.28×    −24528.4 %
      DRAM Reads   0.62×       +37.5 %
      DRAM Writes  0.01×       +99.3 %

    Baseline Detail (Sum Singles):
      Layer        Energy (μJ)   Latency (cc)      EDP
      L0             3.600e+02     8.028e+05     2.89e+02
      L1             5.499e+03     8.120e+05     4.47e+03
      L2             1.529e+03     4.014e+05     6.14e+02
      L3             1.809e+03     4.383e+05     7.94e+02
      L4             5.992e+02     2.007e+05     1.21e+02
      L5             9.226e+02     3.482e+05     3.24e+02
      L6             9.226e+02     3.482e+05     3.24e+02
      L7             3.335e+02     3.451e+05     1.20e+02
      L8             5.291e+02     6.902e+05     3.87e+02
      L9             5.291e+02     6.902e+05     3.87e+02
      L10            1.889e+02     6.149e+05     1.35e+02
      L11            1.889e+02     6.149e+05     1.35e+02
      L12            1.889e+02     6.149e+05     1.35e+02

    Analysis:
      Full fusion is 26.6× MORE expensive in energy than sum of singles.
      This is the worst energy ratio across all workloads tested — VGG16's
      13 layers with large C dimensions (up to 512) create enormous
      register-level WMOPs as each PE cycles through all layers' C×R
      weight factors on every MAC operation.

      Latency is 21.2 % LOWER for full fusion (5.46M vs 6.92M cc).
      This is a significantly stronger latency win than ResNet18 (7.8 %),
      because VGG16 has larger intermediate activation volumes
      (up to 512×28×28) that benefit more from DRAM trip elimination.
      DRAM Reads drop 37.5 %, DRAM Writes drop 99.3 %.

      Despite the substantial latency win, the 26.6× energy penalty
      makes EDP 246× worse for full fusion.
    ==================================================================================

    ────────────────────────────────────────────────────────────────────────────
     FULL vs PARTIAL(2layer) — Vgg16 Eyeriss 512×32 (16384 PEs), WReg=1200
    ────────────────────────────────────────────────────────────────────────────
     python3 experiment_runner.py --compare-full-vs-partial -w vgg16 \
         --arch-type eyeriss --pe-rows 512 --pe-cols 32 --weight-reg 1200 \
         --gb-size 128 --tile-size 1 --input-reg 700 --intermediate-reg 300 \
         --output-reg 64
     ----------------------------------------------------------------------------------
     
     
    Full fusion: 1 run (13 layers fused).
    Partial: 8 runs (block-level: block1(2L), block2(2L), block3(3L),
    L6(1L), block4(3L), L9(1L), block5(3L), L12(1L)).
    Note: VGG16 blocks contain 2-3 consecutive conv layers; the remaining
    single layers (L6, L9, L12) are the last layer of blocks 3, 4, 5
    that don't pair evenly.

    Results:
      Level                        Energy (μJ)   Latency (cc)      EDP
      Full Fusion (1 ok)            3.618e+05     5.455e+06      2.03e+06
      Sum Partial Fusion (8 ok)     2.620e+05     5.852e+06      2.05e+05

    Ratios (Full Fusion / Sum Partial) — values < 1.0 mean fusion wins:
      Energy       1.38×       −38.1 %
      Latency      0.93×        +6.8 %
      EDP          9.87×      −886.5 %
      DRAM Reads   0.84×       +15.9 %
      DRAM Writes  0.01×       +98.6 %

    Partial Fusion Segment Detail:
      Segment         Energy (μJ)   Latency (cc)      EDP
      block/block1     4.512e+04     1.032e+06      5.15e+04
      block/block2     6.781e+04     6.523e+05      4.50e+04
      block/block3     6.420e+04     3.748e+05      2.43e+04
      block/L6         9.226e+02     3.482e+05      3.24e+02
      block/block4     6.262e+04     9.349e+05      5.89e+04
      block/L9         5.291e+02     6.902e+05      3.87e+02
      block/block5     2.060e+04     1.205e+06      2.50e+04
      block/L12        1.889e+02     6.149e+05      1.35e+02

    Analysis:
      Full fusion is 38.1 % more expensive in energy than block-level
      fusion (3.618e+05 vs 2.620e+05 μJ). This is more pronounced
      than ResNet18's 1.8 % gap because VGG16 blocks already contain
      2-3 layers each, so the per-block WReg overhead is smaller than
      the 13-layer full fusion overhead.

      Latency: full fusion is 6.8 % faster (5.46M vs 5.85M cc).
      DRAM Writes drop 98.6 % (100K vs 7.4M) — all intermediates
      stay on-chip. DRAM Reads drop 15.9 %.

      EDP: full fusion is 9.9× worse. The 38 % energy penalty and
      modest 6.8 % latency gain combine unfavourably.

      Block2 and block4 are the most energy-expensive segments
      (6.78e+04 and 6.26e+04 μJ) — these contain the mid-network
      layers with large C×Z products (128→256 and 256→512).
    ==================================================================================

    ────────────────────────────────────────────────────────────────────────────
    PARTIAL(2layer) vs SINGLE — Vgg16 Eyeriss 512×32 (16384 PEs), WReg=1200
    ────────────────────────────────────────────────────────────────────────────
    
          python3 experiment_runner.py --compare-partial-vs-single -w vgg16 \
      --arch-type eyeriss --pe-rows 512 --pe-cols 32 --weight-reg 384 \
      --gb-size 128 --tile-size 1 \
      --input-reg 700 --intermediate-reg 300 --output-reg 64

    Note: WReg=384 (lower than full fusion's 1200) — sized for at most
    3-layer block-level fusion, not full 13-layer fusion.

    Partial: 8 runs (block-level).  Singles: 13 individual layer runs.

    Results:
      Level                        Energy (μJ)   Latency (cc)      EDP
      Sum Partial Fusion (8 ok)     1.622e+05     5.852e+06      1.29e+05
      Sum Singles (13 ok)           1.359e+04     6.922e+06      8.15e+03

    Ratios (Sum Partial / Sum Singles) — values < 1.0 mean fusion wins:
      Energy      11.94×     −1094.0 %
      Latency      0.85×       +15.5 %
      EDP         15.84×     −1484.1 %
      DRAM Reads   0.74×       +25.7 %
      DRAM Writes  0.55×       +45.2 %

    Baseline Detail (Sum Singles):
      Layer        Energy (μJ)   Latency (cc)      EDP
      L0             3.586e+02     8.028e+05     2.88e+02
      L1             5.498e+03     8.120e+05     4.46e+03
      L2             1.528e+03     4.014e+05     6.13e+02
      L3             1.807e+03     4.383e+05     7.92e+02
      L4             5.986e+02     2.007e+05     1.20e+02
      L5             9.213e+02     3.482e+05     3.22e+02
      L6             9.213e+02     3.482e+05     3.22e+02
      L7             3.328e+02     3.451e+05     1.17e+02
      L8             5.277e+02     6.902e+05     3.71e+02
      L9             5.277e+02     6.902e+05     3.71e+02
      L10            1.886e+02     6.149e+05     1.22e+02
      L11            1.886e+02     6.149e+05     1.22e+02
      L12            1.886e+02     6.149e+05     1.22e+02

    Partial Fusion Segment Detail:
      Segment         Energy (μJ)   Latency (cc)      EDP
      block/block1     2.785e+04     1.032e+06      3.37e+04
      block/block2     4.306e+04     6.523e+05      2.89e+04
      block/block3     3.945e+04     3.748e+05      1.50e+04
      block/L6         9.213e+02     3.482e+05      3.22e+02
      block/block4     3.787e+04     9.349e+05      3.57e+04
      block/L9         5.277e+02     6.902e+05      3.71e+02
      block/block5     1.235e+04     1.205e+06      1.50e+04
      block/L12        1.886e+02     6.149e+05      1.22e+02

    Analysis:
      Block-level fusion is 11.9× MORE expensive in energy than singles.
      Even with WReg=384 (sized for 2-3 layer blocks, not full fusion),
      the register WMOPs penalty is substantial. Each block-level PE
      stores 2-3 layers' weight residuals and cycles through them all.

      Latency improves 15.5 % (5.85M vs 6.92M cc) — stronger than
      ResNet18's 4.9 % because VGG16 has larger intermediate volumes.
      DRAM Reads drop 25.7 %, DRAM Writes drop 45.2 %.

      EDP: 15.8× worse for partial fusion, driven by energy.
      Even with reduced WReg (384 vs 1200), block-level energy
      (1.622e+05 μJ) is still 12× above singles (1.359e+04 μJ).

    Effect of WReg=384 vs WReg=1200 (from full-vs-partial experiment):
      The block-level segments with WReg=384 consume less energy than
      with WReg=1200 (1.622e+05 vs 2.620e+05 μJ, −38 %). This is
      because smaller registers have lower energy-per-access (Accelergy
      scales energy with register size). The latency is identical
      (5.852e+06 cc) since WReg=384 is still sufficient for all blocks.

    Cross-comparison (all three experiments):
      Metric    Full(13L,WReg=1200)  Partial(block,WReg=1200)  Partial(block,WReg=384)  Singles(1L)
      Energy    3.618e+05            2.620e+05                 1.622e+05                 1.360e+04  μJ
      Latency   5.455e+06            5.852e+06                 5.852e+06                 6.922e+06  cc
      EDP       2.03e+06             2.05e+05                  1.29e+05                  8.23e+03

      Singles remain the clear EDP winner.
      Full fusion has the lowest latency (−21.2 % vs singles) but
      the worst energy (26.6× higher).
      Block-level fusion is a middle ground: 15.5 % latency gain
      at 11.9× energy cost (WReg=384).
      Reducing WReg from 1200→384 cuts block energy by 38 % without
      affecting latency — a pure energy-per-access saving.

      Conclusion: For VGG16 on Eyeriss, layer fusion trades enormous
      register energy for moderate DRAM savings. Even block-level
      fusion at minimal WReg (384) is 12× more energy-expensive than
      singles. The Eyeriss register hierarchy is fundamentally
      ill-suited for multi-layer weight storage — each fused layer
      adds multiplicative WMOPs overhead that overwhelms DRAM savings.


    
        
    ────────────────────────────────────────────────────────────────────────────
    FULL vs SINGLE — Vgg16 Eyeriss 256×32 (8192 PEs), WReg=2400
    ────────────────────────────────────────────────────────────────────────────
    python3 experiment_runner.py --compare-full-vs-single -w vgg16 \
        --arch-type eyeriss --pe-rows 256 --pe-cols 32 --weight-reg 2400 \
        --gb-size 128 --tile-size 1 --input-reg 700 --intermediate-reg 300 \
        --output-reg 64
    ----------------------------------------------------------------------------------
    Architecture: Eyeriss 256x32, GB=128KB, WReg=2400, InReg=700, IntReg=300, OutReg=64

    Summary:
    Level                        Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
    Full Fusion (1 ok)             5.471e+05      5.681e+06      3.15e+06       14,860,992       100,352
    Sum Singles (13 ok)            9.812e+03      6.922e+06      5.84e+03       23,792,320    13,547,520

    Ratios (Full / Singles) -- values < 1.0 mean fusion wins:
        Energy:      55.76x  (-5475.5%)   fusion LOSES massively
        Latency:      0.82x  (+17.9%)     fusion wins ~18%
        EDP:        539.39x  (-53838.9%)  fusion LOSES massively
        DRAM Reads:   0.62x  (+37.5%)     fusion wins ~38%
        DRAM Writes:  0.007x (+99.3%)     fusion wins ~99%

    Per-Layer Singles Detail:
        Layer     Energy (uJ)   Latency (cc)         EDP    Status
        L0          3.619e+02      8.028e+05    2.91e+02        OK
        L1          3.113e+03      8.120e+05    2.53e+03        OK
        L2          9.340e+02      4.014e+05    3.76e+02        OK
        L3          1.566e+03      4.383e+05    6.88e+02        OK
        L4          4.780e+02      2.007e+05    9.75e+01        OK
        L5          8.051e+02      3.482e+05    2.86e+02        OK
        L6          8.051e+02      3.482e+05    2.86e+02        OK
        L7          2.747e+02      3.451e+05    1.06e+02        OK
        L8          4.740e+02      6.902e+05    3.70e+02        OK
        L9          4.740e+02      6.902e+05    3.70e+02        OK
        L10         1.751e+02      6.149e+05    1.46e+02        OK
        L11         1.751e+02      6.149e+05    1.46e+02        OK
        L12         1.751e+02      6.149e+05    1.46e+02        OK

    Analysis:
        - Full fusion is catastrophic at 256x32: 55.76x energy penalty, 539x EDP penalty.
        - Fusion does save 18% latency and 37.5% DRAM reads, but register-level
          weight cycling across 13 layers with 2400-entry WReg overwhelms any DRAM benefit.
        - Compared to 512x32 config (55.76x vs 26.60x energy ratio), doubling PE rows
          from 256 to 512 actually improved things, suggesting register pressure is
          worse at this PE count despite the same WReg.

    ==================================================================================
    

    
    ────────────────────────────────────────────────────────────────────────────
    FULL vs PARTIAL(2layer) — Vgg16 Eyeriss 256×32 (8192 PEs), WReg=2400
    ────────────────────────────────────────────────────────────────────────────
    python3 experiment_runner.py --compare-full-vs-partial -w vgg16 \
        --arch-type eyeriss --pe-rows 256 --pe-cols 32 --weight-reg 2400 \
        --gb-size 128 --tile-size 1 --input-reg 700 --intermediate-reg 300 \
        --output-reg 64
    ----------------------------------------------------------------------------------
    Level                           Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
    Full Fusion (1 ok)                5.471e+05      5.681e+06      3.15e+06       14,860,992       100,352
    Sum Partial Fusion (8 ok)         4.044e+05      5.952e+06      3.23e+05       17,670,848     7,426,048

    Ratios (Full / Partial) -- values < 1.0 mean full fusion wins:
        Energy:       1.35x  (-35.3%)    full fusion LOSES ~35%
        Latency:      0.95x  (+4.6%)     full fusion wins ~5%
        EDP:          9.75x  (-875.2%)   full fusion LOSES massively
        DRAM Reads:   0.84x  (+15.9%)    full fusion wins ~16%
        DRAM Writes:  0.014x (+98.6%)    full fusion wins ~99%

    Partial Fusion Segment Detail:
        Segment      Energy (uJ)   Latency (cc)         EDP    Status
        block1         7.052e+04      1.032e+06    7.52e+04        OK
        block2         1.018e+05      6.523e+05    6.72e+04        OK
        block3         9.955e+04      4.751e+05    4.76e+04        OK
        L6             8.051e+02      3.482e+05    2.86e+02        OK
        block4         9.849e+04      9.349e+05    9.24e+04        OK
        L9             4.740e+02      6.902e+05    3.70e+02        OK
        block5         3.260e+04      1.205e+06    3.95e+04        OK
        L12            1.751e+02      6.149e+05    1.46e+02        OK

    Analysis:
        - Full fusion is 35% more expensive in energy and 9.75x worse in EDP than partial.
        - Full fusion only saves 5% latency vs partial -- marginal benefit.
        - Full fusion does win on DRAM reads (16%) and writes (99%), but the register
          cycling overhead of fusing all 13 layers dominates over the partial approach
          which breaks the network into manageable 2-3 layer fused blocks.
        - Partial fusion is the clear winner: it captures most DRAM savings with
          far less register pressure.
  ==================================================================================
    

    ────────────────────────────────────────────────────────────────────────────
    PARTIAL(2layer) vs SINGLE — Vgg16 Eyeriss 256×32 (8192 PEs), WReg=770
    ────────────────────────────────────────────────────────────────────────────
      python3 experiment_runner.py --compare-partial-vs-single -w vgg16 --arch-type eyeriss
      --pe-rows 256 --pe-cols 32 --weight-reg 770
      --gb-size 128 --tile-size 1
      --input-reg 700 --intermediate-reg 300 --output-reg 64 --verbose

      Configuration:
          Architecture: Eyeriss 256x32 (8192 PEs), GB=128KB
          WReg=770, InReg=700, IntReg=300, OutReg=64, tile_size=1
          Total experiments: 21 (21 OK, 0 failed)

      Summary:
          Level                           Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
          Sum Partial Fusion (8 ok)         2.051e+05      5.952e+06      1.65e+05       17,670,848     7,426,048
          Sum Singles (13 ok)               9.765e+03      6.922e+06      5.66e+03       23,792,320    13,547,520

      Ratios (Partial / Singles) -- values < 1.0 mean partial fusion wins:
          Energy:      21.01x  (-2000.7%)  partial fusion LOSES massively
          Latency:      0.86x  (+14.0%)    partial fusion wins ~14%
          EDP:         29.19x  (-2819.3%)  partial fusion LOSES massively
          DRAM Reads:   0.74x  (+25.7%)    partial fusion wins ~26%
          DRAM Writes:  0.55x  (+45.2%)    partial fusion wins ~45%

      Partial Fusion Segment Detail:
          Segment      Energy (uJ)   Latency (cc)         EDP    Status
          block1         3.602e+04      1.032e+06    3.96e+04        OK
          block2         5.237e+04      6.523e+05    3.49e+04        OK
          block3         5.011e+04      4.751e+05    2.41e+04        OK
          L6             7.997e+02      3.482e+05    2.80e+02        OK
          block4         4.906e+04      9.349e+05    4.62e+04        OK
          L9             4.686e+02      6.902e+05    3.37e+02        OK
          block5         1.612e+04      1.205e+06    1.96e+04        OK
          L12            1.738e+02      6.149e+05    1.19e+02        OK

      Per-Layer Singles Detail:
          Layer     Energy (uJ)   Latency (cc)         EDP    Status
          L0          3.593e+02      8.028e+05    2.88e+02        OK
          L1          3.108e+03      8.120e+05    2.52e+03        OK
          L2          9.313e+02      4.014e+05    3.74e+02        OK
          L3          1.561e+03      4.383e+05    6.85e+02        OK
          L4          4.753e+02      2.007e+05    9.59e+01        OK
          L5          7.997e+02      3.482e+05    2.80e+02        OK
          L6          7.997e+02      3.482e+05    2.80e+02        OK
          L7          2.720e+02      3.451e+05    9.73e+01        OK
          L8          4.686e+02      6.902e+05    3.37e+02        OK
          L9          4.686e+02      6.902e+05    3.37e+02        OK
          L10         1.738e+02      6.149e+05    1.19e+02        OK
          L11         1.738e+02      6.149e+05    1.19e+02        OK
          L12         1.738e+02      6.149e+05    1.19e+02        OK

      Analysis:
          - Even partial fusion (2-3 layer blocks) costs 21x more energy and 29x worse EDP
            than running individual layers, despite saving 14% latency and 26% DRAM reads.
          - With WReg=770 (smaller than the 2400 used in full/partial comparison), the
            register cycling overhead is reduced but still dominant.
          - Comparing to the 512x32 config (11.94x energy at WReg=384), the 256x32 config
            with WReg=770 is worse (21x), indicating that register provisioning does not
            scale linearly with PE count for VGG16 fusion on Eyeriss.

           
    ────────────────────────────────────────────────────────────────────────────
    Eyeriss VGG16 Fusion Cross-Comparison at PE=256x32 (8192 PEs)
    ==============================================================
        Comparison              WReg    Energy Ratio    Latency Ratio    EDP Ratio
        Full vs Single          2400       55.76x           0.82x         539.39x
        Full vs Partial         2400        1.35x           0.95x           9.75x
        Partial vs Single        770       21.01x           0.86x          29.19x

    Key takeaways:
        1. Full fusion is disastrous (56x energy, 539x EDP) -- far worse than at
          512x32 (27x energy, 246x EDP). More PE rows amplify register pressure.
        2. Partial fusion bridges the gap well: only 1.35x more energy than partial,
          but partial itself is still 21x worse than singles.
        3. At 256x32, all forms of fusion are counterproductive for VGG16 on Eyeriss.
          Singles always win on EDP by huge margins.
        4. Fusion's only benefit is latency reduction (14-18%) and DRAM traffic
          reduction (26-38% reads, 45-99% writes), but register cycling overhead
          in energy completely dominates.
        5. Compared to 512x32 results: the 256x32 config is uniformly worse for
          fusion, suggesting that PE aspect ratio matters -- fewer rows with more
          cols (512x32) distributes weight register load better than 256x32.

    
    
    
    
    
    
    ────────────────────────────────────────────────────────────────────────────
    FULL vs SINGLE — Vgg16 Eyeriss 256×16 (4096 PEs), WReg=4800
    ────────────────────────────────────────────────────────────────────────────

    python3 experiment_runner.py --compare-full-vs-single -w vgg16 \
        --arch-type eyeriss --pe-rows 256 --pe-cols 16 --weight-reg 4800 \
        --gb-size 128 --tile-size 1 --input-reg 700 --intermediate-reg 300 \
        --output-reg 64
    ----------------------------------------------------------------------------------
    Architecture: Eyeriss 256x16, GB=128KB, WReg=4800, InReg=700, IntReg=300, OutReg=64

    
    Level                        Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
    Full Fusion (1 ok)             9.398e+05      6.715e+06      6.34e+06       14,860,992       100,352
    Sum Singles (13 ok)            6.707e+03      7.694e+06      4.77e+03       23,792,320    13,547,520

      Ratios (Full / Singles) -- values < 1.0 mean fusion wins:
          Energy:     140.11x  (-13911.4%)  fusion LOSES catastrophically
          Latency:      0.87x  (+12.7%)     fusion wins ~13%
          EDP:       1328.50x  (-132750.3%) fusion LOSES catastrophically
          DRAM Reads:   0.62x  (+37.5%)     fusion wins ~38%
          DRAM Writes:  0.007x (+99.3%)     fusion wins ~99%

      Per-Layer Singles Detail:
          Layer     Energy (uJ)   Latency (cc)         EDP    Status
          L0          3.659e+02      8.028e+05    2.94e+02        OK
          L1          1.943e+03      8.120e+05    1.58e+03        OK
          L2          6.482e+02      4.014e+05    2.62e+02        OK
          L3          9.946e+02      6.021e+05    6.04e+02        OK
          L4          3.417e+02      3.011e+05    1.08e+02        OK
          L5          5.325e+02      6.021e+05    3.39e+02        OK
          L6          5.325e+02      6.021e+05    3.39e+02        OK
          L7          2.132e+02      3.451e+05    9.50e+01        OK
          L8          3.510e+02      6.902e+05    3.28e+02        OK
          L9          3.510e+02      6.902e+05    3.28e+02        OK
          L10         1.444e+02      6.149e+05    1.65e+02        OK
          L11         1.444e+02      6.149e+05    1.65e+02        OK
          L12         1.444e+02      6.149e+05    1.65e+02        OK

      Analysis:
          - Full fusion at 256x16 is catastrophic: 140x energy penalty, 1329x EDP penalty.
          - This is far worse than 256x32 (55.76x energy) and 512x32 (26.60x energy),
            confirming that fewer PE cols (narrower array) dramatically increases
            register cycling overhead for full 13-layer fusion.
          - Fusion saves 13% latency and 38% DRAM reads, but register-level energy
            dominates by orders of magnitude.
          - With only 16 cols, each PE must cycle through more weight tiles, amplifying
            the WReg read/write overhead per fused layer.

      ==================================================================================
    
    ────────────────────────────────────────────────────────────────────────────
    FULL vs PARTIAL(2layer) — Eyeriss 256×16 (4096 PEs), WReg=4800
    ────────────────────────────────────────────────────────────────────────────
    python3 experiment_runner.py --compare-full-vs-partial -w vgg16 \
        --arch-type eyeriss --pe-rows 256 --pe-cols 16 --weight-reg 4800 \
        --gb-size 128 --tile-size 1 --input-reg 700 --intermediate-reg 300 \
        --output-reg 64
    ----------------------------------------------------------------------------------
    
        Level                           Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
    Full Fusion (1 ok)                9.398e+05      6.715e+06      6.34e+06       14,860,992       100,352
    Sum Partial Fusion (8 ok)         6.952e+05      6.897e+06      6.71e+05       17,670,848     7,426,048

Ratios (Full / Partial) -- values < 1.0 mean full fusion wins:
    Energy:       1.35x  (-35.2%)    full fusion LOSES ~35%
    Latency:      0.97x  (+2.6%)     full fusion wins ~3%
    EDP:          9.45x  (-844.9%)   full fusion LOSES massively
    DRAM Reads:   0.84x  (+15.9%)    full fusion wins ~16%
    DRAM Writes:  0.014x (+98.6%)    full fusion wins ~99%

Partial Fusion Segment Detail:
    Segment      Energy (uJ)   Latency (cc)         EDP    Status
    block1         1.213e+05      1.032e+06    1.26e+05        OK
    block2         1.734e+05      9.032e+05    1.57e+05        OK
    block3         1.717e+05      9.032e+05    1.55e+05        OK
    L6             5.325e+02      6.021e+05    3.39e+02        OK
    block4         1.710e+05      9.472e+05    1.62e+05        OK
    L9             3.510e+02      6.902e+05    3.28e+02        OK
    block5         5.671e+04      1.205e+06    6.87e+04        OK
    L12            1.444e+02      6.149e+05    1.65e+02        OK

Analysis:
    - Full fusion is 35% more expensive in energy and 9.45x worse in EDP than partial.
    - Full fusion only saves 2.6% latency vs partial -- almost negligible.
    - Full fusion does win on DRAM reads (16%) and writes (99%), but the register
      cycling overhead of fusing all 13 layers dominates.
    - The full-vs-partial energy ratio (1.35x) is remarkably stable across PE configs:
      1.38x at 512x32, 1.35x at 256x32, 1.35x at 256x16, suggesting the overhead
      of going from partial to full is a fixed ~35% regardless of PE aspect ratio.
==================================================================================
    
    ────────────────────────────────────────────────────────────────────────────
    FULL vs PARTIAL(block) — Eyeriss 256×16 (4096 PEs), WReg=4800
    ────────────────────────────────────────────────────────────────────────────
    python3 experiment_runner.py --compare-full-vs-partial -w vgg16 \
        --arch-type eyeriss --pe-rows 256 --pe-cols 16 --weight-reg 4800 \
        --gb-size 128 --tile-size 1 --input-reg 700 --intermediate-reg 300 \
        --output-reg 64 --intermediate-level block
    ----------------------------------------------------------------------------------
        Level                           Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
    Full Fusion (1 ok)                9.398e+05      6.715e+06      6.34e+06       14,860,992       100,352
    Sum Partial Fusion (8 ok)         6.952e+05      6.897e+06      6.71e+05       17,670,848     7,426,048

    Ratios (Full / Partial) -- values < 1.0 mean full fusion wins:
        Energy:       1.35x  (-35.2%)    full fusion LOSES ~35%
        Latency:      0.97x  (+2.6%)     full fusion wins ~3%
        EDP:          9.45x  (-844.9%)   full fusion LOSES massively
        DRAM Reads:   0.84x  (+15.9%)    full fusion wins ~16%
        DRAM Writes:  0.014x (+98.6%)    full fusion wins ~99%

    Partial Fusion Segment Detail:
        Segment      Energy (uJ)   Latency (cc)         EDP    Status
        block1         1.213e+05      1.032e+06    1.26e+05        OK
        block2         1.734e+05      9.032e+05    1.57e+05        OK
        block3         1.717e+05      9.032e+05    1.55e+05        OK
        L6             5.325e+02      6.021e+05    3.39e+02        OK
        block4         1.710e+05      9.472e+05    1.62e+05        OK
        L9             3.510e+02      6.902e+05    3.28e+02        OK
        block5         5.671e+04      1.205e+06    6.87e+04        OK
        L12            1.444e+02      6.149e+05    1.65e+02        OK

    Analysis:
        - Full fusion is 35% more expensive in energy and 9.45x worse in EDP than partial.
        - Full fusion only saves 2.6% latency vs partial -- almost negligible.
        - Full fusion does win on DRAM reads (16%) and writes (99%), but the register
          cycling overhead of fusing all 13 layers dominates.
        - The full-vs-partial energy ratio (1.35x) is remarkably stable across PE configs:
          1.38x at 512x32, 1.35x at 256x32, 1.35x at 256x16, suggesting the overhead
          of going from partial to full is a fixed ~35% regardless of PE aspect ratio.

    ==================================================================================
    
    
    ────────────────────────────────────────────────────────────────────────────
    PARTIAL(2layer) vs SINGLE — VGG16 Eyeriss 256×16 (4096 PEs), WReg=1600
    ────────────────────────────────────────────────────────────────────────────
   
    python3 experiment_runner.py --compare-partial-vs-single -w vgg16 \
        --arch-type eyeriss --pe-rows 256 --pe-cols 16 --weight-reg 1600 \
        --gb-size 128 --tile-size 1 --input-reg 700 --intermediate-reg 300 \
        --output-reg 64
    ----------------------------------------------------------------------------------
    Architecture: Eyeriss 256x16, GB=128KB, WReg=1600, InReg=700, IntReg=300, OutReg=64

        Level                           Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
    Sum Partial Fusion (8 ok)         3.039e+05      6.897e+06      2.94e+05       17,670,848     7,426,048
    Sum Singles (13 ok)               6.528e+03      7.694e+06      4.35e+03       23,792,320    13,547,520

    Ratios (Partial / Singles) -- values < 1.0 mean partial fusion wins:
        Energy:      46.55x  (-4555.2%)  partial fusion LOSES massively
        Latency:      0.90x  (+10.4%)    partial fusion wins ~10%
        EDP:         67.65x  (-6665.4%)  partial fusion LOSES massively
        DRAM Reads:   0.74x  (+25.7%)    partial fusion wins ~26%
        DRAM Writes:  0.55x  (+45.2%)    partial fusion wins ~45%

    Partial Fusion Segment Detail:
        Segment      Energy (uJ)   Latency (cc)         EDP    Status
        block1         5.359e+04      1.032e+06    5.65e+04        OK
        block2         7.635e+04      9.032e+05    6.95e+04        OK
        block3         7.469e+04      9.032e+05    6.77e+04        OK
        L6             5.115e+02      6.021e+05    3.14e+02        OK
        block4         7.393e+04      9.472e+05    7.02e+04        OK
        L9             3.300e+02      6.902e+05    2.56e+02        OK
        block5         2.436e+04      1.205e+06    2.95e+04        OK
        L12            1.391e+02      6.149e+05    1.11e+02        OK

    Per-Layer Singles Detail:
        Layer     Energy (uJ)   Latency (cc)         EDP    Status
        L0          3.606e+02      8.028e+05    2.90e+02        OK
        L1          1.922e+03      8.120e+05    1.56e+03        OK
        L2          6.376e+02      4.014e+05    2.56e+02        OK
        L3          9.735e+02      6.021e+05    5.88e+02        OK
        L4          3.312e+02      3.011e+05    1.01e+02        OK
        L5          5.115e+02      6.021e+05    3.14e+02        OK
        L6          5.115e+02      6.021e+05    3.14e+02        OK
        L7          2.027e+02      3.451e+05    7.71e+01        OK
        L8          3.300e+02      6.902e+05    2.56e+02        OK
        L9          3.300e+02      6.902e+05    2.56e+02        OK
        L10         1.391e+02      6.149e+05    1.11e+02        OK
        L11         1.391e+02      6.149e+05    1.11e+02        OK
        L12         1.391e+02      6.149e+05    1.11e+02        OK

    Analysis:
        - Partial fusion at 256x16 costs 46.55x more energy and 67.65x worse EDP
          than singles, despite saving 10% latency and 26% DRAM reads.
        - This is significantly worse than 256x32 (21.01x energy at WReg=770) and
          512x32 (11.94x energy at WReg=384), showing the 16-col configuration
          amplifies register cycling overhead dramatically.
        - With WReg=1600 (much larger than needed for singles), the per-PE register
          file energy per access scales with capacity, making each weight cycling
          iteration more expensive.
    ==================================================================================
    
        
    ==============================================================
      Comparison              WReg    Energy Ratio    Latency Ratio    EDP Ratio
      Full vs Single          4800      140.11x           0.87x        1328.50x
      Full vs Partial         4800        1.35x           0.97x           9.45x
      Partial vs Single       1600       46.55x           0.90x          67.65x

        Comparison across PE aspect ratios (VGG16 Eyeriss):
            Config       Full/Single E    Full/Partial E    Partial/Single E    Full/Single EDP
            512x32          26.60x            1.38x             11.94x              246.28x
            256x32          55.76x            1.35x             21.01x              539.39x
            256x16         140.11x            1.35x             46.55x             1328.50x

        Key takeaways:
            1. 256x16 is by far the worst config for fusion: 140x energy penalty for full
              fusion (vs 56x at 256x32, 27x at 512x32). EDP penalty exceeds 1300x.
            2. The full-vs-partial ratio is remarkably stable at ~1.35x across all configs,
              meaning the incremental cost of going from partial to full fusion is
              geometry-independent (~35% extra energy).
            3. The partial-vs-single ratio degrades sharply with fewer cols: 12x (512x32)
              -> 21x (256x32) -> 47x (256x16). Fewer cols = more weight cycling per PE.
            4. Latency savings are modest across all configs (10-18%) and shrink with
              fewer cols (13% at 256x16 vs 18% at 256x32 for full fusion).
            5. At 256x16, all forms of fusion are catastrophically counterproductive.
              Singles always win on EDP by 2-3 orders of magnitude.


    

END OF EYERISS VGG16
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    
    


    ────────────────────────────────────────────────────────────────────────────
    FULL vs SINGLE — MCCNN on Eyeriss 256×8 (2048 PEs), WReg=384
    ────────────────────────────────────────────────────────────────────────────
    python3 experiment_runner.py --compare-full-vs-single -w mccnn \
        --arch-type eyeriss --pe-rows 256 --pe-cols 8 --weight-reg 384 \
        --gb-size 128 --tile-size 69 --input-reg 34 --intermediate-reg 32 \
        --output-reg 64
    ----------------------------------------------------------------------------------
    Architecture: Eyeriss 256x8, GB=128KB, WReg=384, InReg=34, IntReg=32, OutReg=64

        Level                        Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
    Full Fusion (1 ok)             8.955e+04      1.223e+07      1.10e+06          494,928    14,943,744
    Sum Singles (4 ok)             7.488e+03      1.495e+07      2.80e+04       45,326,160    59,774,976

    Ratios (Full / Singles) -- values < 1.0 mean fusion wins:
        Energy:      11.96x  (-1095.9%)  fusion LOSES ~12x
        Latency:      0.82x  (+18.2%)    fusion wins ~18%
        EDP:         39.31x  (-3831.4%)  fusion LOSES massively
        DRAM Reads:   0.011x (+98.9%)    fusion wins ~99%
        DRAM Writes:  0.25x  (+75.0%)    fusion wins ~75%

    Per-Layer Singles Detail:
        Layer     Energy (uJ)   Latency (cc)         EDP    Status
        L0          5.742e+02      3.736e+06    2.15e+03        OK
        L1          2.305e+03      3.738e+06    8.62e+03        OK
        L2          2.305e+03      3.738e+06    8.62e+03        OK
        L3          2.305e+03      3.738e+06    8.62e+03        OK

    Analysis:
        - Full fusion of all 4 MC-CNN layers costs 12x more energy and 39x worse EDP
          than running layers individually, despite 18% latency improvement.
        - Fusion achieves massive DRAM read reduction (99%) -- from 45.3M to 0.5M reads --
          and 75% DRAM write reduction, indicating excellent intermediate data reuse.
        - However, register-level weight cycling (WReg reads = 8.6B for fused vs 3.7M
          per single) dominates energy, overwhelming all DRAM savings.
        - MC-CNN's 4-layer structure with large spatial dimensions (376x414) makes each
          weight cycling iteration very expensive at 256x8 (only 8 cols per row).

    ==================================================================================
    
        
    ────────────────────────────────────────────────────────────────────────────
    FULL vs PARTIAL — MCCNN on Eyeriss 256×8 (2048 PEs), WReg=384
    ────────────────────────────────────────────────────────────────────────────
    python3 experiment_runner.py --compare-full-vs-partial -w mccnn --arch-type eyeriss
      --pe-rows 256 --pe-cols 8 --weight-reg 384
      --gb-size 128 --tile-size 69
      --input-reg 40 --intermediate-reg 40 --output-reg 64 --verbose


          Level                           Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
          Full Fusion (1 ok)                8.955e+04      1.223e+07      1.10e+06          494,928    14,943,744
          Sum Partial Fusion (2 ok)         9.138e+04      1.223e+07      6.00e+05       15,438,672    29,887,488

      Ratios (Full / Partial) -- values < 1.0 mean full fusion wins:
          Energy:       0.98x  (+2.0%)     full fusion wins ~2%
          Latency:      1.00x  (+0.0%)     essentially identical
          EDP:          1.84x  (-83.5%)    full fusion LOSES ~84%
          DRAM Reads:   0.032x (+96.8%)    full fusion wins ~97%
          DRAM Writes:  0.50x  (+50.0%)    full fusion wins ~50%

      Partial Fusion Segment Detail:
          Segment      Energy (uJ)   Latency (cc)         EDP    Status
          L0_L1          3.136e+04      4.757e+06    1.50e+05        OK
          L2_L3          6.002e+04      7.474e+06    4.50e+05        OK

      Analysis:
          - Full fusion is 2% cheaper in energy than partial -- a rare case where full
            fusion actually wins on energy, though by a tiny margin.
          - Latency is essentially identical (0.0% difference).
          - However, EDP favors partial by 1.84x because partial has lower absolute EDP
            (6.00e+05 vs 1.10e+06), likely due to different mapping efficiency.
          - Full fusion achieves 97% fewer DRAM reads and 50% fewer DRAM writes,
            showing superior intermediate data reuse across all 4 layers vs 2-layer blocks.
          - This is a unique MC-CNN result: with only 4 layers, the incremental cost of
            going from 2-layer blocks to full 4-layer fusion is negligible in energy (~2%),
            unlike VGG16 where it was a consistent 35% penalty            
    ==================================================================================
    
    ────────────────────────────────────────────────────────────────────────────
    PARTIAL vs SINGLE — MCCNN on Eyeriss 256×8 (2048 PEs), WReg=384
    ────────────────────────────────────────────────────────────────────────────
        python3 experiment_runner.py --compare-partial-vs-single -w mccnn --arch-type eyeriss
      --pe-rows 256 --pe-cols 8 --weight-reg 384
      --gb-size 128 --tile-size 69
      --input-reg 40 --intermediate-reg 40 --output-reg 64 --verbose

    Configuration:
        Architecture: Eyeriss 256x8 (2048 PEs), GB=128KB
        WReg=384, InReg=40, IntReg=40, OutReg=64, tile_size=69
        Total experiments: 6 (6 OK, 0 failed)

    Summary:
        Level                           Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
        Sum Partial Fusion (2 ok)         9.138e+04      1.223e+07      6.00e+05       15,438,672    29,887,488
        Sum Singles (4 ok)                7.488e+03      1.495e+07      2.80e+04       45,326,160    59,774,976

    Ratios (Partial / Singles) -- values < 1.0 mean partial fusion wins:
        Energy:      12.20x  (-1120.3%)  partial fusion LOSES ~12x
        Latency:      0.82x  (+18.2%)    partial fusion wins ~18%
        EDP:         21.42x  (-2042.0%)  partial fusion LOSES massively
        DRAM Reads:   0.34x  (+65.9%)    partial fusion wins ~66%
        DRAM Writes:  0.50x  (+50.0%)    partial fusion wins ~50%

    Partial Fusion Segment Detail:
        Segment      Energy (uJ)   Latency (cc)         EDP    Status
        L0_L1          3.136e+04      4.757e+06    1.50e+05        OK
        L2_L3          6.002e+04      7.474e+06    4.50e+05        OK

    Per-Layer Singles Detail:
        Layer     Energy (uJ)   Latency (cc)         EDP    Status
        L0          5.742e+02      3.736e+06    2.15e+03        OK
        L1          2.305e+03      3.738e+06    8.62e+03        OK
        L2          2.305e+03      3.738e+06    8.62e+03        OK
        L3          2.305e+03      3.738e+06    8.62e+03        OK

    Analysis:
        - Partial fusion (2-layer blocks) costs 12.2x more energy and 21.4x worse EDP
          than singles, despite 18% latency savings and 66% DRAM read reduction.
        - The energy ratio (12.2x) is very close to the full-vs-single ratio (12.0x),
          confirming that for MC-CNN almost all fusion overhead comes from the first
          level of fusion (2-layer blocks), not from extending to full 4-layer.
        - L2_L3 block (6.00e+04 uJ) is ~2x more expensive than L0_L1 (3.14e+04 uJ),
          likely due to larger channel counts in deeper layers requiring more weight cycling.

Eyeriss MC-CNN Fusion Cross-Comparison at PE=256x8 (2048 PEs)
==============================================================
    Comparison              WReg    Energy Ratio    Latency Ratio    EDP Ratio
    Full vs Single           384       11.96x           0.82x          39.31x
    Full vs Partial          384        0.98x           1.00x           1.84x
    Partial vs Single        384       12.20x           0.82x          21.42x

Key takeaways:
    1. MC-CNN fusion overhead is dominated by the 2-layer partial fusion step:
       going from partial to full adds only ~2% energy (0.98x ratio means full
       is actually 2% cheaper than partial in energy).
    2. Full fusion achieves remarkable DRAM savings: 99% fewer reads and 75% fewer
       writes vs singles. Even partial achieves 66% read and 50% write reduction.
    3. Despite massive DRAM traffic reduction, register cycling overhead (WReg
       reads explode from 3.7M to 8.6B per fused pair) overwhelms all savings,
       resulting in 12x energy penalty.
    4. Latency improvement is consistent at 18% for both full and partial fusion,
       meaning the latency benefit comes entirely from the first fusion step.
    5. Unlike VGG16 (where full-vs-partial was always ~1.35x extra energy), MC-CNN
       shows full fusion is actually slightly cheaper than partial -- the 4-layer
       network is small enough that full fusion mapping can be more efficient.
    6. At 256x8, all forms of MC-CNN fusion are counterproductive on EDP.
       Singles win by 21-39x despite inferior DRAM traffic patterns.


END OF EYERISS MCCNN
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    
    


    ────────────────────────────────────────────────────────────────────────────
    FULL vs SINGLE — FSRCNN on Eyeriss 128×16 (2048 PEs), WReg=384
    ────────────────────────────────────────────────────────────────────────────
    python3 experiment_runner.py --compare-full-vs-single -w fsrcnn \
        --arch-type eyeriss --pe-rows 128 --pe-cols 16 --weight-reg 384 \
        --gb-size 128 --tile-size 120 --input-reg 34 --intermediate-reg 34 \
        --output-reg 64
    ----------------------------------------------------------------------------------
    Architecture: Eyeriss 128x16, GB=128KB, WReg=384, InReg=34, IntReg=34, OutReg=64

    Level                        Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
        Full Fusion (1 ok)             6.757e+04      1.889e+07      1.28e+06        1,573,992     8,294,400
        Sum Singles (8 ok)             8.747e+03      3.525e+07      4.90e+04       90,738,792    97,459,200

    Ratios (Full / Singles) -- values < 1.0 mean fusion wins:
        Energy:       7.73x  (-672.5%)   fusion LOSES ~8x
        Latency:      0.54x  (+46.4%)    fusion wins ~46%
        EDP:         26.17x  (-2516.7%)  fusion LOSES massively
        DRAM Reads:   0.017x (+98.3%)    fusion wins ~98%
        DRAM Writes:  0.085x (+91.5%)    fusion wins ~92%

    Per-Layer Singles Detail:
        Layer                    Energy (uJ)   Latency (cc)         EDP    Status
        L0_feature_extraction      1.402e+03      7.258e+06    1.02e+04        OK
        L1_shrinking               1.470e+03      7.258e+06    1.07e+04        OK
        L2_mapping1                6.340e+02      1.556e+06    9.86e+02        OK
        L3_mapping2                6.340e+02      1.556e+06    9.86e+02        OK
        L4_mapping3                6.340e+02      1.556e+06    9.86e+02        OK
        L5_mapping4                6.340e+02      1.556e+06    9.86e+02        OK
        L6_expanding               1.464e+03      7.258e+06    1.06e+04        OK
        L7_output                  1.875e+03      7.260e+06    1.36e+04        OK

    Analysis:
        - Full fusion of all 8 FSRCNN layers costs 7.73x more energy but achieves
          a remarkable 46% latency reduction -- best latency savings seen across all
          workloads on Eyeriss.
        - DRAM traffic reduction is massive: 98% fewer reads and 92% fewer writes,
          reflecting excellent intermediate feature map reuse within the 8-layer pipeline.
        - Despite the best latency and DRAM savings, EDP is still 26x worse due to
          register-level weight cycling overhead dominating the energy budget.
        - The 46% latency win comes from FSRCNN's large spatial dimensions (1080x1920)
          with tile_size=120, allowing fusion to overlap computation across tiles
          and avoid repeated DRAM accesses for intermediate activations.
    ==================================================================================
    

    ────────────────────────────────────────────────────────────────────────────
    FULL vs PARTIAL — FSRCNN on Eyeriss 128×16 (2048 PEs), WReg=384
    ────────────────────────────────────────────────────────────────────────────

        python3 experiment_runner.py --compare-full-vs-partial -w fsrcnn --arch-type eyeriss
      --pe-rows 128 --pe-cols 16 --weight-reg 384
      --gb-size 128 --tile-size 120
      --input-reg 40 --intermediate-reg 40 --output-reg 64 --verbose

      Configuration:
          Architecture: Eyeriss 128x16 (2048 PEs), GB=128KB
          WReg=384, InReg=40, IntReg=40, OutReg=64, tile_size=120
          Total experiments: 4 (4 OK, 0 failed)

      Summary:
          Level                           Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
          Full Fusion (1 ok)                6.757e+04      1.889e+07      1.28e+06        1,573,992     8,294,400
          Sum Partial Fusion (3 ok)         6.855e+04      1.889e+07      4.65e+05       14,015,592    20,736,000

      Ratios (Full / Partial) -- values < 1.0 mean full fusion wins:
          Energy:       0.99x  (+1.4%)     full fusion wins ~1.4%
          Latency:      1.00x  (+0.0%)     identical latency
          EDP:          2.76x  (-176.0%)   full fusion LOSES ~2.8x
          DRAM Reads:   0.11x  (+88.8%)    full fusion wins ~89%
          DRAM Writes:  0.40x  (+60.0%)    full fusion wins ~60%

      Partial Fusion Segment Detail:
          Segment                      Energy (uJ)   Latency (cc)         EDP    Status
          L0_L1_L2                       2.199e+04      6.059e+06    1.34e+05        OK
          L3_L4_L5                       1.433e+04      4.666e+06    6.73e+04        OK
          L6_L7_expanding_output         3.222e+04      8.165e+06    2.64e+05        OK

      Analysis:
          - Full fusion is 1.4% cheaper in energy than partial -- virtually identical.
          - Latency is exactly the same (1.889e+07 cc for both), meaning extended fusion
            across 8 layers vs 3-layer blocks doesn't improve pipeline scheduling.
          - EDP favors partial by 2.76x because partial achieves the same latency with
            lower absolute EDP (4.65e+05 vs 1.28e+06).
          - Full fusion achieves 89% fewer DRAM reads and 60% fewer DRAM writes than
            partial, showing superior intermediate reuse across all 8 layers.
          - Like MC-CNN, the incremental cost of going from partial to full fusion is
            negligible in energy (~1%), unlike VGG16's consistent 35% penalty.
            This suggests that networks with fewer layers (4-8) can extend fusion depth
            almost for free in terms of energy.

            

    ────────────────────────────────────────────────────────────────────────────
    PARTIAL vs SINGLE — FSRCNN on Eyeriss 128×16 (2048 PEs), WReg=384
    ────────────────────────────────────────────────────────────────────────────

      --pe-rows 128 --pe-cols 16 --weight-reg 384
      --gb-size 128 --tile-size 120
      --input-reg 40 --intermediate-reg 40 --output-reg 64 --verbose

Configuration:
    Architecture: Eyeriss 128x16 (2048 PEs), GB=128KB
    WReg=384, InReg=40, IntReg=40, OutReg=64, tile_size=120
    Total experiments: 11 (11 OK, 0 failed)

    Summary:
        Level                           Energy (uJ)   Latency (cc)           EDP       DRAM Reads   DRAM Writes
        Sum Partial Fusion (3 ok)         6.855e+04      1.889e+07      4.65e+05       14,015,592    20,736,000
        Sum Singles (8 ok)                8.747e+03      3.525e+07      4.90e+04       90,738,792    97,459,200

    Ratios (Partial / Singles) -- values < 1.0 mean partial fusion wins:
        Energy:       7.84x  (-683.7%)   partial fusion LOSES ~8x
        Latency:      0.54x  (+46.4%)    partial fusion wins ~46%
        EDP:          9.48x  (-848.1%)   partial fusion LOSES ~9.5x
        DRAM Reads:   0.15x  (+84.6%)    partial fusion wins ~85%
        DRAM Writes:  0.21x  (+78.7%)    partial fusion wins ~79%

    Partial Fusion Segment Detail:
        Segment                      Energy (uJ)   Latency (cc)         EDP    Status
        L0_L1_L2                       2.199e+04      6.059e+06    1.34e+05        OK
        L3_L4_L5                       1.433e+04      4.666e+06    6.73e+04        OK
        L6_L7_expanding_output         3.222e+04      8.165e+06    2.64e+05        OK

    Per-Layer Singles Detail:
        Layer                    Energy (uJ)   Latency (cc)         EDP    Status
        L0_feature_extraction      1.402e+03      7.258e+06    1.02e+04        OK
        L1_shrinking               1.470e+03      7.258e+06    1.07e+04        OK
        L2_mapping1                6.340e+02      1.556e+06    9.86e+02        OK
        L3_mapping2                6.340e+02      1.556e+06    9.86e+02        OK
        L4_mapping3                6.340e+02      1.556e+06    9.86e+02        OK
        L5_mapping4                6.340e+02      1.556e+06    9.86e+02        OK
        L6_expanding               1.464e+03      7.258e+06    1.06e+04        OK
        L7_output                  1.875e+03      7.260e+06    1.36e+04        OK

    Analysis:
        - Partial fusion (3-layer blocks) costs 7.84x more energy but wins 46% on
          latency -- the same impressive latency reduction as full fusion.
        - DRAM savings are substantial: 85% fewer reads and 79% fewer writes.
        - EDP is 9.48x worse than singles, still dominated by register cycling overhead.
        - L6_L7 block is the most expensive segment (3.22e+04 uJ), likely due to
          the expanding layer's large output channel count (32->1) with deconvolution.
        - L3_L4_L5 (mapping layers) is cheapest (1.43e+04 uJ) because mapping layers
          have small 1x1 kernels with only 12 channels.



==================================================================          
    Eyeriss FSRCNN Fusion Cross-Comparison at PE=128x16 (2048 PEs)
==================================================================
    Comparison              WReg    Energy Ratio    Latency Ratio    EDP Ratio
    Full vs Single           384        7.73x           0.54x          26.17x
    Full vs Partial          384        0.99x           1.00x           2.76x
    Partial vs Single        384        7.84x           0.54x           9.48x

Key takeaways:
    1. FSRCNN achieves the best latency savings of any workload on Eyeriss:
       46% reduction for both full and partial fusion. This stems from FSRCNN's
       large spatial dimensions (1080x1920) with tile_size=120.
    2. Full and partial fusion are nearly identical in energy (0.99x ratio) and
       latency (1.00x), meaning 3-layer blocks capture essentially all the
       benefit of full 8-layer fusion.
    3. The energy penalty is moderate compared to VGG16 (7.7x vs 26-140x),
       reflecting FSRCNN's smaller layer count and simpler weight structure.
    4. EDP still favors singles (9.5-26x), but the gap is the smallest among
       all workloads tested, making FSRCNN the closest to fusion-viable.
    5. DRAM traffic reduction is exceptional: 98% read reduction for full fusion,
       85% for partial. FSRCNN's pipeline structure with large activations and
       small weights makes it an ideal candidate for fusion if register energy
       can be reduced.

       
END OF EYERISS FSRCNN
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    






END OF EYERISS all the workloads
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────



    ==================================================================================
    FULL vs SINGLE — ResNet18 on DepFiN 16×128 (2048 PEs), FMEM=266KB, WMEM=10738KB
    ----------------------------------------------------------------------------------
    python3 experiment_runner.py --compare-full-vs-single -w resnet18 \
        --arch-type depfin --pe-rows 16 --pe-cols 128 --fmem-size 266 \
        --wmem-size 10738 --tile-size 7
    ----------------------------------------------------------------------------------
    Architecture: DepFiN 16x128, FMEM=266KB, WMEM=10738KB

    RATIOS (Full Fusion / Singles) — values < 1.0 mean fusion wins
      Energy:      11.4025   (-1040.2%)   ← fusion ~11× more expensive
      Latency:      1.1314     (-13.1%)   ← fusion 13% SLOWER
      EDP:        163.2098  (-16221.0%)
      DRAM Reads:   0.8861     (+11.4%)
      DRAM Writes:  0.0109     (+98.9%)

    Full Fusion (1 segment, OK):
      Energy = 1.083e+04 uJ,  Latency = 7.812e+06 cc,  EDP = 1.51e+05
      DRAM Reads = 11,032,512   DRAM Writes = 25,088

    Sum Singles (17 layers, all OK):
      Energy = 9.499e+02 uJ,  Latency = 6.905e+06 cc,  EDP = 9.28e+02
      DRAM Reads = 12,449,984   DRAM Writes = 2,308,096

    BASELINE SINGLES:
      L0_conv1:      E=3.468e+01  L=6.690e+04
      L1_conv2_1_1:  E=2.495e+01  L=1.290e+05
      L2_conv2_1_2:  E=2.495e+01  L=1.290e+05
      L3_conv2_2_1:  E=2.495e+01  L=1.290e+05
      L4_conv2_2_2:  E=2.495e+01  L=1.290e+05
      L5_conv3_1_1:  E=1.669e+01  L=1.290e+05
      L6_conv3_1_2:  E=3.009e+01  L=2.580e+05
      L8_conv3_2_1:  E=3.009e+01  L=2.580e+05
      L9_conv3_2_2:  E=3.009e+01  L=2.580e+05
      L10_conv4_1_1: E=2.945e+01  L=2.580e+05
      L11_conv4_1_2: E=5.725e+01  L=5.161e+05
      L13_conv4_2_1: E=5.725e+01  L=5.161e+05
      L14_conv4_2_2: E=5.725e+01  L=5.161e+05
      L15_conv5_1_1: E=7.282e+01  L=5.161e+05
      L16_conv5_1_2: E=1.448e+02  L=1.032e+06
      L18_conv5_2_1: E=1.448e+02  L=1.032e+06
      L19_conv5_2_2: E=1.448e+02  L=1.032e+06
    ==================================================================================
 
    ==================================================================================
    FULL vs PARTIAL(2layer) — ResNet18 on DepFiN 16×128 (2048 PEs), FMEM=266KB, WMEM=10738KB
    ----------------------------------------------------------------------------------
    python3 experiment_runner.py --compare-full-vs-partial -w resnet18 \
        --arch-type depfin --pe-rows 16 --pe-cols 128 --fmem-size 266 \
        --wmem-size 10738 --tile-size 7
    ----------------------------------------------------------------------------------
    RATIOS (Full Fusion / Partial 2layer) — values < 1.0 mean full wins
      Energy:       0.9484      (+5.2%)   ← FULL WINS by 5.2%
      Latency:      0.4578     (+54.2%)   ← full 54% faster!
      EDP:          3.4922    (-249.2%)
      DRAM Reads:   0.9381      (+6.2%)
      DRAM Writes:  0.0333     (+96.7%)

    Full Fusion (1 segment, OK):
      Energy = 1.083e+04 uJ,  Latency = 7.812e+06 cc,  EDP = 1.51e+05
      DRAM Reads = 11,032,512   DRAM Writes = 25,088

    Sum Partial Fusion — 2layer (8 segments, all OK):
      Energy = 1.142e+04 uJ,  Latency = 1.706e+07 cc,  EDP = 4.33e+04
      DRAM Reads = 11,760,064   DRAM Writes = 752,640

    PARTIAL SEGMENTS (2layer):
      2layer/s1b1:  E=1.113e+03  L=1.566e+06  EDP=3.09e+03  OK
      2layer/s1b2:  E=1.700e+03  L=3.102e+06  EDP=9.23e+03  OK
      2layer/s2b1:  E=1.639e+03  L=2.068e+06  EDP=6.03e+03  OK
      2layer/s2b2:  E=1.138e+03  L=2.066e+06  EDP=4.11e+03  OK
      2layer/s3b1:  E=1.656e+03  L=2.066e+06  EDP=6.06e+03  OK
      2layer/s3b2:  E=1.163e+03  L=2.065e+06  EDP=4.17e+03  OK
      2layer/s4b1:  E=1.738e+03  L=2.065e+06  EDP=6.25e+03  OK
      2layer/s4b2:  E=1.275e+03  L=2.065e+06  EDP=4.42e+03  OK
    ==================================================================================
   
     ==================================================================================
    PARTIAL(2layer) vs SINGLE — ResNet18 on DepFiN 16×128 (2048 PEs), FMEM=266KB, WMEM=10738KB
    ----------------------------------------------------------------------------------
    python3 experiment_runner.py --compare-partial-vs-single -w resnet18 \
        --arch-type depfin --pe-rows 16 --pe-cols 128 --fmem-size 266 \
        --wmem-size 10738 --tile-size 7
    ----------------------------------------------------------------------------------
    RATIOS (Partial Fusion / Singles) — values < 1.0 mean fusion wins
      Energy:      12.0231   (-1102.3%)   ← fusion ~12× more expensive
      Latency:      2.4712    (-147.1%)   ← fusion 147% SLOWER
      EDP:         46.7359   (-4573.6%)
      DRAM Reads:   0.9446      (+5.5%)
      DRAM Writes:  0.3261     (+67.4%)

    Sum Partial Fusion (8 segments, all OK):
      Energy = 1.142e+04 uJ,  Latency = 1.706e+07 cc,  EDP = 4.33e+04
      DRAM Reads = 11,760,064   DRAM Writes = 752,640

    Sum Singles (17 layers, all OK):
      Energy = 9.499e+02 uJ,  Latency = 6.905e+06 cc,  EDP = 9.28e+02
      DRAM Reads = 12,449,984   DRAM Writes = 2,308,096

    FUSED-SIDE SEGMENTS:
      2layer/s1b1:  E=1.113e+03  L=1.566e+06  EDP=3.09e+03  OK
      2layer/s1b2:  E=1.700e+03  L=3.102e+06  EDP=9.23e+03  OK
      2layer/s2b1:  E=1.639e+03  L=2.068e+06  EDP=6.03e+03  OK
      2layer/s2b2:  E=1.138e+03  L=2.066e+06  EDP=4.11e+03  OK
      2layer/s3b1:  E=1.656e+03  L=2.066e+06  EDP=6.06e+03  OK
      2layer/s3b2:  E=1.163e+03  L=2.065e+06  EDP=4.17e+03  OK
      2layer/s4b1:  E=1.738e+03  L=2.065e+06  EDP=6.25e+03  OK
      2layer/s4b2:  E=1.275e+03  L=2.065e+06  EDP=4.42e+03  OK

    BASELINE SINGLES:
      L0_conv1:      E=3.468e+01  L=6.690e+04
      L1_conv2_1_1:  E=2.495e+01  L=1.290e+05
      L2_conv2_1_2:  E=2.495e+01  L=1.290e+05
      L3_conv2_2_1:  E=2.495e+01  L=1.290e+05
      L4_conv2_2_2:  E=2.495e+01  L=1.290e+05
      L5_conv3_1_1:  E=1.669e+01  L=1.290e+05
      L6_conv3_1_2:  E=3.009e+01  L=2.580e+05
      L8_conv3_2_1:  E=3.009e+01  L=2.580e+05
      L9_conv3_2_2:  E=3.009e+01  L=2.580e+05
      L10_conv4_1_1: E=2.945e+01  L=2.580e+05
      L11_conv4_1_2: E=5.725e+01  L=5.161e+05
      L13_conv4_2_1: E=5.725e+01  L=5.161e+05
      L14_conv4_2_2: E=5.725e+01  L=5.161e+05
      L15_conv5_1_1: E=7.282e+01  L=5.161e+05
      L16_conv5_1_2: E=1.448e+02  L=1.032e+06
      L18_conv5_2_1: E=1.448e+02  L=1.032e+06
      L19_conv5_2_2: E=1.448e+02  L=1.032e+06

    NOTES:
      - DepFiN vs Eyeriss key difference: latency penalty is much worse.
        Full fusion is 13% SLOWER than singles (vs 8% faster on Eyeriss).
        Partial 2layer is 147% SLOWER (vs 5% faster on Eyeriss).
      - Full vs Partial: full wins by 5.2% on energy AND 54% on latency —
        much stronger full-fusion advantage than on Eyeriss (~0-1.6% energy).
      - Energy ratio vs singles (~11-12×) is comparable to Eyeriss at
        small WReg configs (e.g. 512×32/WReg=384 → 13.9×).
      - DRAM savings identical to Eyeriss (same dataflow, same workload):
        Reads 11.4%, Writes 98.9% (full), 67.4% (partial).
    ==================================================================================

    ==================================================================================
    DepFiN 16x128 (2,048 PEs), VGG16, FMEM=568KB, WMEM=14366KB, tile=14
    FULL FUSION (13-layer) vs Sum of SINGLE layers
    ==================================================================================

    Full Fusion:   E=7.607e+04 uJ,  L=2.433e+07 cc,  EDP=3.41e+06
    Sum Singles:   E=3.395e+03 uJ,  L=2.452e+07 cc,  EDP=1.20e+04

    Energy ratio:   22.4065  (-2140.7%)  — single layers WIN by 22×
    Latency ratio:   0.9919  (+0.8%)    — fusion barely faster (<1%)
    EDP ratio:     282.5927             — single layers WIN massively
    DRAM Reads:      0.6246  (+37.5%)   — fusion saves 37.5% reads
    DRAM Writes:     0.0074  (+99.3%)   — fusion saves 99.3% writes

    Observation: Full fusion is catastrophically worse in energy (22×)
    despite near-identical latency and excellent DRAM write savings.
    The compute energy penalty of 13-layer fusion far outweighs
    the memory traffic savings on DepFiN.

    
    ==================================================================================

    ==================================================================================
    DepFiN 16x128 (2,048 PEs), VGG16, FMEM=528KB, WMEM=14366KB, tile=14
    FULL FUSION (13-layer) vs Sum of PARTIAL FUSION (block-level 2-layer)
    ==================================================================================

    Full Fusion:       E=7.607e+04 uJ,  L=2.433e+07 cc,  EDP=3.41e+06
    Sum Partial (8):   E=5.834e+04 uJ,  L=5.204e+07 cc,  EDP=9.94e+05

    Energy ratio:    1.3039  (-30.4%)   — partial fusion WINS by 30.4%
    Latency ratio:   0.4674  (+53.3%)   — full fusion 53% faster!
    EDP ratio:       3.4258             — partial fusion wins on EDP
    DRAM Reads:      0.8410  (+15.9%)   — full fusion saves 15.9% reads
    DRAM Writes:     0.0135  (+98.6%)   — full fusion saves 98.6% writes

    Partial baseline detail:
    block1 (2L): E=1.002e+04, L=8.673e+06  OK
    block2 (2L): E=1.415e+04, L=1.033e+07  OK
    block3 (2L): E=1.413e+04, L=1.033e+07  OK
    L6     (1L): E=2.727e+02, L=2.064e+06  OK
    block4 (2L): E=1.419e+04, L=1.032e+07  OK
    L9     (1L): E=4.502e+02, L=4.129e+06  OK
    block5 (2L): E=4.890e+03, L=4.130e+06  OK
    L12    (1L): E=2.428e+02, L=2.064e+06  OK

    Observation: Full fusion is 30% worse in energy but 53% faster
    in latency vs block-level partial. The latency advantage of full
    fusion is strong on VGG16 DepFiN due to eliminated inter-block
    DRAM round-trips, but energy penalty remains significant.
    =====================================================================

    =====================================================================
    DepFiN 16x128 (2,048 PEs), VGG16, FMEM=82KB, WMEM=4610KB, tile=14
    Sum of PARTIAL FUSION (block-level 2-layer) vs Sum of SINGLE layers
    =====================================================================

    Sum Partial (8 ok):  E=3.821e+04 uJ,  L=5.204e+07 cc,  EDP=6.27e+05
    Sum Singles (13 ok): E=2.443e+03 uJ,  L=2.452e+07 cc,  EDP=7.97e+03

    Energy ratio:   15.6380  (-1463.8%)  — single layers WIN by ~16×
    Latency ratio:   2.1219  (-112.2%)   — fusion 112% SLOWER
    EDP ratio:      78.6843              — single layers WIN massively
    DRAM Reads:      0.7427  (+25.7%)    — fusion saves 25.7% reads
    DRAM Writes:     0.5481  (+45.2%)    — fusion saves 45.2% writes

    All 8 partial segments completed successfully (no FMEM overflow).

    Partial fusion detail:
    block1 (2L): E=6.559e+03, L=8.673e+06  OK
    block2 (2L): E=9.241e+03, L=1.033e+07  OK
    block3 (2L): E=9.222e+03, L=1.033e+07  OK
    L6     (1L): E=1.828e+02, L=2.064e+06  OK
    block4 (2L): E=9.287e+03, L=1.032e+07  OK
    L9     (1L): E=3.027e+02, L=4.129e+06  OK
    block5 (2L): E=3.237e+03, L=4.130e+06  OK
    L12    (1L): E=1.769e+02, L=2.064e+06  OK

    Observation: With FMEM=82KB (vs 78KB prior), block4 now fits and
    all segments succeed. However, 2-layer partial fusion remains ~16×
    worse in energy and 2.1× slower than single-layer execution.
    DRAM traffic savings (25.7% reads, 45.2% writes) are insufficient
    to offset the massive compute energy overhead of fusion on DepFiN
    with VGG16's large channel counts.



    

    =================================================================
    DepFiN 8x256 (2,048 PEs), MCCNN, FMEM=522KB, WMEM=32KB, tile=207
    FULL FUSION (4-layer) vs Sum of SINGLE layers
    =================================================================

    --compare-full-vs-single -w mccnn \
        --arch-type depfin --pe-rows 8 --pe-cols 256 --fmem-size 522 \
        --wmem-size 32 --tile-size 207 | tee DF_MCCNN_522KB_32KB_full_vs_single.log

Level                        Energy (uJ)   Latency (cc)       EDP       DRAM Reads   DRAM Writes
  Full Fusion (1 ok)           1.177e+04     8.074e+06    1.13e+05          494,928    14,943,744
  Sum Singles (4 ok)           4.224e+03     1.381e+07    1.62e+04       45,326,160    59,774,976

  RATIOS (Full Fusion / Sum Singles) — values < 1.0 mean fusion wins
    Energy:       2.7871  (-178.7%)   ← single layers WIN by 2.8×
    Latency:      0.5846  (+41.5%)    ← fusion FASTER by 41.5%
    EDP:          6.9670  (-596.7%)   ← single layers WIN on EDP
    DRAM Reads:   0.0109  (+98.9%)    ← fusion saves 98.9% reads
    DRAM Writes:  0.2500  (+75.0%)    ← fusion saves 75.0% writes

    Observation: Fusion is 2.8× worse in energy but 41.5% faster.
    Near-total DRAM read elimination (98.9% savings) and 75% write
    savings. The latency advantage is significant — fusion halves
    execution time by keeping all intermediates on-chip. However,
    compute energy overhead still dominates, making singles win
    on EDP by ~7×. Compared to the 16×128 config (tile=69), the
    8×256 config with tile=207 achieves much better latency, thanks
    to the larger tile covering more of Q=1242.

    ====================================================================
    DepFiN 8x256 (2,048 PEs), MCCNN, FMEM=30KB, WMEM=30KB, tile=207
    FULL FUSION (4-layer) vs Sum of PARTIAL FUSION (2-layer)
    ====================================================================


  --compare-full-vs-partial -w mccnn \
        --arch-type depfin --pe-rows 8 --pe-cols 256 --fmem-size 522 \
        --wmem-size 32 --tile-size 69 | tee DF_MCCNN_522KB_32KB_full_vs_partial.log


    
    Level                        Energy (uJ)   Latency (cc)       EDP       DRAM Reads   DRAM Writes
    Full Fusion (1 ok)           1.177e+04     8.074e+06    1.13e+05          494,928    14,943,744
    Sum Partial Fusion (2 ok)    1.273e+04     8.074e+06    6.60e+04       15,438,672    29,887,488

    RATIOS (Full Fusion / Sum Partial) — values < 1.0 mean full fusion wins
      Energy:       0.9248  (+7.5%)     ← FULL FUSION WINS by 7.5%!
      Latency:      1.0000  (+0.0%)     ← identical latency
      EDP:          1.7126  (-71.3%)    ← partial wins on EDP
      DRAM Reads:   0.0321  (+96.8%)    ← full fusion saves 96.8% reads
      DRAM Writes:  0.5000  (+50.0%)    ← full fusion saves 50.0% writes

    Partial fusion detail:
    L0_L1 (2L): E=4.097e+03, L=4.145e+06  OK
    L2_L3 (2L): E=7.934e+03, L=7.824e+06  OK

  Observation: Full fusion WINS in energy by 7.5% over sum of
    2-layer partials, with identical latency. Massive DRAM read
    savings (96.8%). This confirms MCCNN for
    full fusion on DepFiN — its small 4-layer topology and low
    channel counts (Z=32) keep compute overhead manageable.
    The energy win comes from eliminating inter-segment DRAM
    traffic that partial fusion still incurs between L1→L2.




    =====================================================================    
    DepFiN 8x256 (2,048 PEs), MCCNN, FMEM=396KB, WMEM=22KB, tile=207
    Sum of PARTIAL FUSION (2-layer) vs Sum of SINGLE layers
    =====================================================================


  --compare-partial-vs-single -w mccnn --arch-type depfin --pe-rows 8 --pe-cols 256 --fmem-size 396 --wmem-size 22 --tile-size 207 | tee DF_MCCNN_396KB_22KB_partial_vs_single.log

    Level                        Energy (uJ)   Latency (cc)       EDP       DRAM Reads   DRAM Writes
    Sum Partial Fusion (2 ok)    1.238e+04     8.074e+06    6.33e+04       15,438,672    29,887,488
    Sum Singles (4 ok)           4.123e+03     1.381e+07    1.58e+04       45,326,160    59,774,976

    RATIOS (Sum Partial / Sum Singles) — values < 1.0 mean partial fusion wins
      Energy:       3.0032  (-200.3%)   ← single layers WIN by 3.0×
      Latency:      0.5846  (+41.5%)    ← fusion FASTER by 41.5%
      EDP:          4.0099  (-301.0%)   ← single layers WIN on EDP
      DRAM Reads:   0.3406  (+65.9%)    ← fusion saves 65.9% reads
      DRAM Writes:  0.5000  (+50.0%)    ← fusion saves 50.0% writes

    Partial fusion detail:
    L0_L1 (2L): E=4.220e+03, L=2.858e+06  OK
    L2_L3 (2L): E=8.160e+03, L=5.216e+06  OK

    Observation: 2-layer partial fusion is 3.0× worse in energy
    but 41.5% faster than single-layer execution. DRAM savings
    are substantial (65.9% reads, 50% writes) but cannot compensate
    for the compute energy overhead. The latency advantage matches
    full fusion (both 0.5846× ratio), confirming that on MCCNN
    the latency benefit comes from 2-layer fusion and full fusion
    adds no further latency improvement. Energy ratio (3.0×) is
    slightly worse than full fusion vs singles (2.8×) because
    partial fusion still has inter-segment DRAM overhead.
    





    ==================================================================================    
    DepFiN 16x128 (2,048 PEs), FSRCNN, FMEM=72KB, WMEM=19KB, tile=120
    FULL FUSION (8-layer) vs Sum of SINGLE layers
    ==============================================================================

        python3 experiment_runner.py  --compare-full-vs-single -w fsrcnn \
        --arch-type depfin --pe-rows 16 --pe-cols 128 --fmem-size 266 \
        --wmem-size 19 --tile-size 120 | tee DF_FSRCNN_72KB_19KB_full_vs_single.log



                            Energy (μJ)   Latency (cc)       EDP       DRAM Reads   DRAM Writes
    Full Fusion (1 ok)         8.231e+03      6.122e+06    5.91e+04      1,573,992     8,294,400
    Sum Singles (8 ok)         6.402e+03      1.175e+07    1.23e+04     90,738,792    97,459,200

    Energy ratio:    1.2853  (-28.5%)   — single layers WIN by 29.5%
    Latency ratio:   0.5210  (+47.9%)   — fusion 48% faster!
    EDP ratio:       4.8583             — single layers WIN on EDP
    DRAM Reads:      0.0173  (+98.3%)   — fusion saves 98.3% reads!
    DRAM Writes:     0.0851  (+91.5%)   — fusion saves 91.5% writes!

    Observation: Full fusion is only 1.3× worse in energy (much better
    than VGG16's 22×) while being 48% faster. DRAM savings are
    outstanding: 98.3% read and 91.5% write elimination. FSRCNN's
    small channel counts and 8-layer topology make fusion highly
    effective on DepFiN. Energy penalty is low enough that the
    latency and DRAM benefits may justify fusion in practice.

    ======================================================================
    DepFiN 16x128 (2,048 PEs), FSRCNN, FMEM=72KB, WMEM=19KB, tile=120
    FULL FUSION (8-layer) vs Sum of PARTIAL FUSION (3-layer segments)
    ======================================================================



    python3 experiment_runner.py --compare-full-vs-partial -w fsrcnn \
        --arch-type depfin --pe-rows 16 --pe-cols 128 --fmem-size 266 \
        --wmem-size 19 --tile-size 120 | tee DF_FSRCNN_72KB_19KB_full_vs_partial.log
    
                                Energy (μJ)   Latency (cc)       EDP       DRAM Reads   DRAM Writes
    Full Fusion (1 ok)              8.231e+03      6.122e+06    5.91e+04      1,573,992     8,294,400
    Sum Partial Fusion (3 ok)       9.027e+03      6.313e+06    2.34e+04     14,015,592    20,736,000

    Energy ratio:    0.9116  (+8.8%)    — FULL FUSION WINS by 8.8%!
    Latency ratio:   0.9697  (+3.0%)    — full fusion 3% faster
    EDP ratio:       2.5157             — partial wins on EDP
    DRAM Reads:      0.1123  (+88.8%)   — full fusion saves 88.8% reads
    DRAM Writes:     0.4000  (+60.0%)   — full fusion saves 60.0% writes

    Partial fusion detail:
    L0_L1_L2            (3L): E=2.808e+03, L=2.095e+06  OK
    L3_L4_L5            (3L): E=2.009e+03, L=1.508e+06  OK
    L6_L7_expand_output (2L): E=4.063e+03, L=2.710e+06  OK

    Observation: Full fusion WINS in energy by 8.9% and is 3% faster
    than sum of 3-layer partial segments. Full fusion also achieves
    far superior DRAM savings (88.8% reads, 60% writes). This
    confirms FSRCNN as a workload where full fusion dominates
    partial on DepFiN — similar to MCCNN. Small workloads with
    low channel counts consistently favor full fusion.


    ======================================================================    
    DepFiN 16x128 (2,048 PEs), FSRCNN, FMEM=248KB, WMEM=9KB, tile=120
    Sum of PARTIAL FUSION (3-layer segments) vs Sum of SINGLE layers
    ======================================================================

                                 Energy (μJ)   Latency (cc)       EDP       DRAM Reads   DRAM Writes
    Sum Partial Fusion (3 ok)       8.519e+03      6.313e+06    2.12e+04     14,015,592    20,736,000
    Sum Singles (8 ok)              6.390e+03      1.175e+07    1.22e+04     90,738,792    97,459,200

    Energy ratio:    1.3330  (-33.3%)   — single layers WIN by 34.5%
    Latency ratio:   0.5372  (+46.3%)   — fusion 46% faster!
    EDP ratio:       1.7480             — single layers WIN on EDP
    DRAM Reads:      0.1545  (+84.6%)   — fusion saves 84.6% reads
    DRAM Writes:     0.2128  (+78.7%)   — fusion saves 78.7% writes

    Partial fusion detail:
    L0_L1_L2            (3L): E=2.635e+03, L=2.095e+06  OK
    L3_L4_L5            (3L): E=1.899e+03, L=1.508e+06  OK
    L6_L7_expand_output (2L): E=3.820e+03, L=2.710e+06  OK

    Observation: Partial fusion is only 1.35× worse in energy but
    46% faster than single-layer execution — nearly identical to
    the full fusion result (1.30× energy, 48% faster). DRAM savings
    are excellent (84.6% reads, 78.7% writes). With FMEM=248KB and
    WMEM=9KB all segments fit (unlike the failed 19KB/5KB attempt).
    The modest energy overhead and strong latency/DRAM gains make
    partial fusion viable for FSRCNN on DepFiN at this memory budget.

    """