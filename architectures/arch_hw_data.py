from math import log2, ceil
import sys
from dataclasses import dataclass
import json
import csv 


try:
    from ..computations import *
    from ..engine import *
    from ..utils import *
except:
    sys.path.append("..")
    from computations import *
    from engine import *
    from utils import *

try:
    from ..architectures.accelergy_hw_data import accelergy_estimate_energy, accelergy_estimate_area
    from ..architectures.architectures import WS, OS, IS
    from ..prints import printEnergyPerAction, printAreaPerLevel
    from ..computations import conv_coupling_with_stride_and_batches
    from ..settings import *
    from ..levels import *
    from ..arch import *
except:
    sys.path.append("..")
    from architectures.accelergy_hw_data import accelergy_estimate_energy, accelergy_estimate_area
    from architectures.architectures import WS, OS, IS
    from prints import printEnergyPerAction, printAreaPerLevel
    from computations import conv_coupling_with_stride_and_batches
    from settings import *
    from levels import *
    from arch import *


"""
Class containing all the memory information
"""
@dataclass
class ParamInfo:

    precision : int = 8
    technology : str = "32nm"
    cycle_seconds : float = 1.075e-09 #1.2e-09

    # the list of valid attributes and values can be found in 
    # /home/mronzani/accelergy-timeloop-infrastructure/src/accelergy-cacti-plug-in/cacti_wrapper.py
    # Energy costs (pJ):
    #   # Public data
    #   if tech == 'LPDDR4':
    #       energy = 8 * width
    #   # Malladi et al., ISCA'12
    #   elif tech == 'LPDDR':
    #       energy = 40 * width
    #   elif tech == 'DDR3':
    #       energy = 70 * width
    #   # Chatterjee et al., MICRO'17
    #   elif tech == 'GDDR5':
    #       energy = 14 * width
    #   elif tech == 'HBM2':
    #       energy = 3.9 * width
    #
    # --- Provided by gemini ---
    # Bandwidth:
    #   LPDDR4: 
    #   Configuration       Bus Width   3200 MT/s (Standard)    4266 MT/s (High Speed/4X)
    #   Single Die (x16)    16-bit      6.4 GB/s                8.53 GB/s
    #   Dual Channel (x32)  32-bit      12.8 GB/s               17.06 GB/s
    #   Quad Channel (x64)  64-bit      25.6 GB/s               34.12 GB/s
    #
    #   --> considering 64-bit bus width at 3200 MT/s: 25.6 GB/s
    #       reporting everything at EyeRiss' frequency (200MHz), we have
    #       32*32*200M = 204.800 Gbit/s = 25.6GB/s --> bandwidth (bit/cc) = 32*32 = 1024
    #
    #       reporting everything at 1GHz
    #       25.6*8*1000/1000 = 204 bit/cc
    #
    #
    #   HBM2
    #   Generation  Data Rate per Pin   Bus Width   Bandwidth per Stack
    #   HBM1        1.0 Gbps            1024-bit    128 GB/s
    #   HBM2        2.0 Gbps            1024-bit    256 GB/s
    #   HBM2E       3.2 – 3.6 Gbps      1024-bit    410 – 460 GB/s
    #   
    #   --> considering 1024-bit bus width HBM2: 256 GB/s
    #       reporting everything at EyeRiss' frequency (200MHz), we have:
    #       bandwidth (bit/cc): 256*8*1000/200 = 10240
    #
    #       reporting everything at 1GHz
    #       256*8*1000/1000 = 2040 bit/cc
    #
    DRAM_type : str = "LPDDR4" # 'type': ['DDR3', 'HBM2', 'GDDR5', 'LPDDR', 'LPDDR4']}
    DRAM_word_bits : int = 64 # LPDDR4: 64, HBM2: 1024 
    DRAM_value_bits : int = precision
    DRAM_bandwidth : int = 96*2 #  LPDDR4: 1024 (204), HBM2: 10240 (2040) (bit/cc)
    DRAM_read_bandwidth_bytes : int = 12
    DRAM_write_bandwidth_bytes : int = 12
    DRAM_size : int = (2**64-1)*64 # bits

    # --- Provided by gemini ---
    # SRAM @65nm, 64-bit interface, SINGLE PORT
    # Low Power (0.7V)	~250 MHz	2.0 GB/s --> 2*1000*8/200 = 80 bit/cc
    # Standard Logic (1.0V)	~667 MHz	5.3 GB/s --> 5.3*1000*8/200 = 212 bit/cc
    # High Performance (1.2V)	~1.1 GHz	8.8 GB/s --> 352 bit/cc

    # Feature	        65nm	            32nm	            12nm
    # Typical Frequency	~1.0 GHz	        ~1.6 GHz	        ~2.0 GHz
    # Interface Width	8 bytes (64-bit)	8 bytes (64-bit)	128 bytes (1024-bit)
    # Peak Bandwidth	8.0 GB/s	        12.8 GB/s	        256.0 GB/s 
    # Access Latency	~1.0 ns	            ~0.6 ns	            ~0.4 ns
    #
    # reporting everything at 200Mz
    # 65nm  8*8*1000/200 = 320 bit/cc
    # 32nm  12.9*1000/200 = 516 bit/s
    #
    # reporting everything at 1GHz
    # 65nm  8*8*1000/1000 = 64 bit/cc
    # 32nm  12.9*1000/1000 = 103.2 bit/s

    # double ports (1R, 1W, 1RW) doubles the bandwidth

    SRAM_word_bits : int = 128*8 # 1024
    SRAM_value_bits : int = 8
    SRAM_bandwidth : int = 224 # 28 GB/s # 32*8 256 bits per cycle 
    SRAM_size : int = 16384*SRAM_word_bits # bits = 128 KB
    global_buffer_banks : int = 32
   
    '''
    # Depfin feature memory
    SRAM_word_bits : int = 128*8 #1024
    SRAM_value_bits : int = 8
    SRAM_bandwidth : int = 2*132*8 # NON VIENE CONSIDERATA QUI 
    SRAM_size : int = 1056*8*SRAM_word_bits # bits = 8.5 Mb
    global_buffer_banks : int = 64
   
    # Depfin weigth memory
    SRAM_word_bits : int = 16*8
    SRAM_value_bits : int = 8
    SRAM_bandwidth : int = 2*132*8 # NON VIENE CONSIDERATA QUI 
    SRAM_size : int = 4192*8*SRAM_word_bits # bits = 4 Mb 
    global_buffer_banks : int = 32
    '''

    arguments = { # in theory, this is not needed on anything but "leak"
        "global_cycle_seconds": cycle_seconds,
    }

    DRAM_attributes = {
        "type": DRAM_type,
        "width": DRAM_word_bits,
        "technology": technology,
        "cycle_seconds": cycle_seconds
    }
    
    SRAM_attributes = {
        "n_rd_ports": 1,
        "n_wr_ports": 1,
        "n_rdwr_ports": 1,
        "depth": math.ceil(SRAM_size/SRAM_word_bits),
        "width": SRAM_word_bits,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "global_cycle_seconds": cycle_seconds,
        "n_banks": global_buffer_banks
        }

# ...existing code...

#---------------------------------------------------------------
# >>> DEPFIN 10-Layer <<<

@dataclass
class DepFinParamInfo:
    """
    DepFiN-specific parameters from multi_layer_arch.py
    All sizes are in BYTES, bandwidths are in BYTES/cycle
    """
    precision : int = 8  # bits per element
    technology : str = "22nm"  # DepFiN uses 22nm
    cycle_seconds : float = 1.075e-09  # ~930 MHz (DepFiN frequency)

   
    # DRAM parameters
    DRAM_type : str = "LPDDR4"
    DRAM_word_bits : int = 64  # 8 bytes
    DRAM_size_bytes : int = 2**64-1
    DRAM_read_bandwidth_bytes : int = 12  # from multi_layer_arch: read_bandwidth = 12
    DRAM_write_bandwidth_bytes : int = 12

    

    # Feature Memory (FMEM) parameters - from multi_layer_arch
    FMEM_size : int = 1056 * 1024 * 8  # bits 
    FMEM_word_bits : int = 1056  # (1056 bits, 132 * 8)
    FMEM_bandwidth: int = 2 * 132 * 8
    FMEM_banks : int = 2  # multiple banks, mentioned in Figure 3

    # Weight Memory (WMEM) parameters - from multi_layer_arch
    WMEM_size : int = 524 * 1024 * 8  # 524 KB
    WMEM_word_bits : int = 16 * 8  # 16 bytes = 128 bits
    WMEM_bandwidth_bytes : int = 2 * 16 * 8  # bandwidth = 2*16, so 16 per direction
    WMEM_banks : int = 4 # multiple banks, mentioned in Figure 3

    # Accumulation Register parameters
    AccReg_size : int = 10 * 4 * 8  # 40 B
    AccReg_word_bits : int = 32
    AccReg_bandwidth : int = 1 * 8  # 1 element per cycle

    arguments = {
        "global_cycle_seconds": cycle_seconds,
    }


def get_depfin_energy_per_byte(param: DepFinParamInfo = None) -> dict:
    """
    Calculate energy per byte access for each memory level in DepFiN.
    Returns a dictionary with read/write energy per byte for each level.
    """
    if param is None:
        param = DepFinParamInfo()
    
    cycle_seconds = param.cycle_seconds
    technology = param.technology
    arguments = {"global_cycle_seconds": cycle_seconds}
    
    # --- DRAM Energy ---
    DRAM_attributes = {
        "type": param.DRAM_type,
        "width": param.DRAM_word_bits,
        "technology": technology,
        "cycle_seconds": cycle_seconds
    }
    
    dram_read_energy_per_access = aclg_energy_mem("DRAM", DRAM_attributes, "read", arguments)
    dram_write_energy_per_access = aclg_energy_mem("DRAM", DRAM_attributes, "write", arguments)
    dram_leakage = aclg_energy_mem("DRAM", DRAM_attributes, "leak", arguments)
    dram_bytes_per_access = param.DRAM_word_bits // 8
    dram_depth = param.DRAM_size_bytes * 8 // param.DRAM_word_bits
    dram_area = aclg_area_mem("DRAM", DRAM_attributes)
    dram_read_energy_per_byte = dram_read_energy_per_access / dram_bytes_per_access
    dram_write_energy_per_byte = dram_write_energy_per_access / dram_bytes_per_access

    # --- Feature Memory (FMEM) Energy ---
    FMEM_attributes = {
        "n_rd_ports": 1,
        "n_wr_ports": 1,
        "n_rdwr_ports": 0,
        "depth": math.ceil(param.FMEM_size / param.FMEM_word_bits),  # FMEM_size is already in bits
        "width": param.FMEM_word_bits,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "global_cycle_seconds": cycle_seconds,
        "n_banks": param.FMEM_banks
    }
    
    fmem_read_energy_per_access = aclg_energy_mem("SRAM", FMEM_attributes, "read", arguments)
    fmem_write_energy_per_access = aclg_energy_mem("SRAM", FMEM_attributes, "write", arguments)
    fmem_leakage = aclg_energy_mem("SRAM", FMEM_attributes, "leak", arguments)
    fmem_bytes_per_access = param.FMEM_word_bits // 8
    fmem_depth = math.ceil(param.FMEM_size / param.FMEM_word_bits)  # FMEM_size is already in bits
    fmem_area = aclg_area_mem("SRAM", FMEM_attributes)
    fmem_read_energy_per_byte = fmem_read_energy_per_access / fmem_bytes_per_access
    fmem_write_energy_per_byte = fmem_write_energy_per_access / fmem_bytes_per_access

    # --- Weight Memory (WMEM) Energy ---
    WMEM_attributes = {
        "n_rd_ports": 1,
        "n_wr_ports": 1,
        "n_rdwr_ports": 0,
        "depth": math.ceil(param.WMEM_size / param.WMEM_word_bits),  # WMEM_size is already in bits
        "width": param.WMEM_word_bits,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "global_cycle_seconds": cycle_seconds,
        "n_banks": param.WMEM_banks
    }
    
    wmem_read_energy_per_access = aclg_energy_mem("SRAM", WMEM_attributes, "read", arguments)
    wmem_write_energy_per_access = aclg_energy_mem("SRAM", WMEM_attributes, "write", arguments)
    wmem_leakage = aclg_energy_mem("SRAM", WMEM_attributes, "leak", arguments)
    wmem_bytes_per_access = param.WMEM_word_bits // 8
    wmem_depth = math.ceil(param.WMEM_size / param.WMEM_word_bits)  # WMEM_size is already in bits
    wmem_area = aclg_area_mem("SRAM", WMEM_attributes)
    wmem_read_energy_per_byte = wmem_read_energy_per_access / wmem_bytes_per_access
    wmem_write_energy_per_byte = wmem_write_energy_per_access / wmem_bytes_per_access

    # --- Accumulation Register Energy ---
    acc_depth = math.ceil(param.AccReg_size / param.AccReg_word_bits)  # AccReg_size is already in bits
    acc_read_energy = smartbuffer_registerfile(
        acc_depth, param.AccReg_word_bits, param.precision, 
        cycle_seconds, technology, "read"
    )
    acc_write_energy = smartbuffer_registerfile(
        acc_depth, param.AccReg_word_bits, param.precision,
        cycle_seconds, technology, "write"
    )
    acc_leakage = smartbuffer_registerfile(
        acc_depth, param.AccReg_word_bits, param.precision,
        cycle_seconds, technology, "leak"
    )
    acc_bytes_per_access = param.AccReg_word_bits // 8
    acc_area = smartbuffer_registerfile(acc_depth, param.AccReg_word_bits, param.precision, cycle_seconds, technology, energy=False)
    acc_read_energy_per_byte = acc_read_energy / acc_bytes_per_access
    acc_write_energy_per_byte = acc_write_energy / acc_bytes_per_access

    # --- Compute Energy ---
    type_multiplier = "aladdin_multiplier"
    width_multiplier = 2 * param.precision
    type_adder = "aladdin_adder"
    out_precision = 32  # accumulator precision
    
    multiplier_energy = aclg_energy_mul(type_multiplier, width_multiplier, param.precision, technology, "read", arguments)
    multiplier_leakage = aclg_energy_mul(type_multiplier, width_multiplier, param.precision, technology, "leak", arguments)
    multiplier_area = aclg_area_mul(type_multiplier, width_multiplier, param.precision, technology)
    
    adder_energy = aclg_energy_add(type_adder, out_precision, technology, "read", arguments)
    adder_leakage = aclg_energy_add(type_adder, out_precision, technology, "leak", arguments)
    adder_area = aclg_area_add(type_adder, out_precision, technology)
    
    fma_energy = multiplier_energy + adder_energy
    fma_leakage = multiplier_leakage + adder_leakage
    fma_area = multiplier_area + adder_area

    return {
        "DRAM": {
            "read_energy_per_byte": dram_read_energy_per_byte,
            "write_energy_per_byte": dram_write_energy_per_byte,
            "read_energy_per_access": dram_read_energy_per_access,
            "write_energy_per_access": dram_write_energy_per_access,
            "value_access_energy_per_wordline": dram_read_energy_per_access,  # for read, can add write if needed
            "leakage": dram_leakage,
            "bytes_per_access": dram_bytes_per_access,
            "word_bits": param.DRAM_word_bits,
            "depth": dram_depth,
            "area": dram_area,
        },
        "FeatureMemory": {
            "read_energy_per_byte": fmem_read_energy_per_byte,
            "write_energy_per_byte": fmem_write_energy_per_byte,
            "read_energy_per_access": fmem_read_energy_per_access,
            "write_energy_per_access": fmem_write_energy_per_access,
            "value_access_energy_per_wordline": fmem_read_energy_per_access,
            "leakage": fmem_leakage,
            "bytes_per_access": fmem_bytes_per_access,
            "word_bits": param.FMEM_word_bits,
            "depth": fmem_depth,
            "banks": param.FMEM_banks,
            "area": fmem_area,
        },
        "WeightMemory": {
            "read_energy_per_byte": wmem_read_energy_per_byte,
            "write_energy_per_byte": wmem_write_energy_per_byte,
            "read_energy_per_access": wmem_read_energy_per_access,
            "write_energy_per_access": wmem_write_energy_per_access,
            "value_access_energy_per_wordline": wmem_read_energy_per_access,
            "leakage": wmem_leakage,
            "bytes_per_access": wmem_bytes_per_access,
            "word_bits": param.WMEM_word_bits,
            "depth": wmem_depth,
            "banks": param.WMEM_banks,
            "area": wmem_area,
        },
        "AccumulationRegister": {
            "read_energy_per_byte": acc_read_energy_per_byte,
            "write_energy_per_byte": acc_write_energy_per_byte,
            "read_energy_per_access": acc_read_energy,
            "write_energy_per_access": acc_write_energy,
            "value_access_energy_per_wordline": acc_read_energy,
            "leakage": acc_leakage,
            "bytes_per_access": acc_bytes_per_access,
            "word_bits": param.AccReg_word_bits,
            "depth": acc_depth,
            "area": acc_area,
        },
        "Compute": {
            "multiplier_energy": multiplier_energy,
            "multiplier_leakage": multiplier_leakage,
            "multiplier_area": multiplier_area,
            "adder_energy": adder_energy,
            "adder_leakage": adder_leakage,
            "adder_area": adder_area,
            "fma_energy": fma_energy,
            "fma_leakage": fma_leakage,
            "fma_area": fma_area,
        }
    }


def print_depfin_energy_summary():
    """Print a summary of DepFiN energy values for use in multi_layer_arch.py"""
    energies = get_depfin_energy_per_byte()
    
    print("\n" + "="*80)
    print("DepFiN Energy, Area, and Configuration Values")
    print("="*80)
    
    for level, data in energies.items():
        print(f"\n{'─'*40}")
        print(f"{level}:")
        print(f"{'─'*40}")
        
        if level == "Compute":
            print(f"  Multiplier:")
            print(f"    energy  = {data['multiplier_energy']:.6f} pJ")
            print(f"    leakage = {data['multiplier_leakage']:.6f} pJ/cycle")
            print(f"    area    = {data['multiplier_area']:.4f} um²")
            print(f"  Adder:")
            print(f"    energy  = {data['adder_energy']:.6f} pJ")
            print(f"    leakage = {data['adder_leakage']:.6f} pJ/cycle")
            print(f"    area    = {data['adder_area']:.4f} um²")
            print(f"  FMA (Fused Multiply-Add):")
            print(f"    energy  = {data['fma_energy']:.6f} pJ/MAC")
            print(f"    leakage = {data['fma_leakage']:.6f} pJ/cycle")
            print(f"    area    = {data['fma_area']:.4f} um²")
        else:
            print(f"  Configuration:")
            print(f"    word_bits (wordline) = {data['word_bits']} bits ({data['bytes_per_access']} bytes)")
            print(f"    depth                = {data['depth']} entries")
            if 'banks' in data:
                print(f"    banks                = {data['banks']}")
            print(f"  Energy:")
            print(f"    read_energy_per_byte   = {data['read_energy_per_byte']:.6f} pJ/byte")
            print(f"    write_energy_per_byte  = {data['write_energy_per_byte']:.6f} pJ/byte")
            print(f"    value_access_energy_per_wordline (read) = {data['value_access_energy_per_wordline']:.6f} pJ (entire wordline)")
            print(f"    read_energy_per_access = {data['read_energy_per_access']:.6f} pJ (per wordline)")
            print(f"    write_energy_per_access= {data['write_energy_per_access']:.6f} pJ (per wordline)")
            print(f"    leakage                = {data['leakage']:.6f} pJ/cycle")
            print(f"  Area:")
            print(f"    area = {data['area']:.4f} um²")
    
    print("\n" + "="*80)
    print("Suggested values for multi_layer_arch.py:")
    print("="*80)
    
    # For value_access_energy (average of read/write per byte)
    dram_avg = (energies["DRAM"]["read_energy_per_byte"] + 
                energies["DRAM"]["write_energy_per_byte"]) / 2
    fmem_avg = (energies["FeatureMemory"]["read_energy_per_byte"] + 
                energies["FeatureMemory"]["write_energy_per_byte"]) / 2
    wmem_avg = (energies["WeightMemory"]["read_energy_per_byte"] + 
                energies["WeightMemory"]["write_energy_per_byte"]) / 2
    acc_avg = (energies["AccumulationRegister"]["read_energy_per_byte"] + 
               energies["AccumulationRegister"]["write_energy_per_byte"]) / 2
    
    print(f"""
# DRAM
read_value_access_energy = {energies["DRAM"]["read_energy_per_byte"]:.4f},  # pJ/byte
write_value_access_energy = {energies["DRAM"]["write_energy_per_byte"]:.4f},  # pJ/byte
value_access_energy = {dram_avg:.4f},  # pJ/byte (average)
leakage = {energies["DRAM"]["leakage"]:.6f},  # pJ/cycle
area = {energies["DRAM"]["area"]:.4f},  # um²

# FeatureMemory (FMEM) - depth={energies["FeatureMemory"]["depth"]}, word={energies["FeatureMemory"]["word_bits"]}bits
value_access_energy = {fmem_avg:.6f},  # pJ/byte
leakage = {energies["FeatureMemory"]["leakage"]:.6f},  # pJ/cycle
area = {energies["FeatureMemory"]["area"]:.4f},  # um²

# WeightMemory (WMEM) - depth={energies["WeightMemory"]["depth"]}, word={energies["WeightMemory"]["word_bits"]}bits
value_access_energy = {wmem_avg:.6f},  # pJ/byte
leakage = {energies["WeightMemory"]["leakage"]:.6f},  # pJ/cycle
area = {energies["WeightMemory"]["area"]:.4f},  # um²

# AccumulationRegister - depth={energies["AccumulationRegister"]["depth"]}, word={energies["AccumulationRegister"]["word_bits"]}bits
value_access_energy = {acc_avg:.6f},  # pJ/byte
leakage = {energies["AccumulationRegister"]["leakage"]:.6f},  # pJ/cycle
area = {energies["AccumulationRegister"]["area"]:.4f},  # um²

# Compute
multiplier_energy = {energies["Compute"]["multiplier_energy"]:.6f},  # pJ
multiplier_leakage = {energies["Compute"]["multiplier_leakage"]:.6f},  # pJ/cycle
adder_energy = {energies["Compute"]["adder_energy"]:.6f},  # pJ
adder_leakage = {energies["Compute"]["adder_leakage"]:.6f},  # pJ/cycle
fma_energy = {energies["Compute"]["fma_energy"]:.6f},  # pJ/MAC
fma_leakage = {energies["Compute"]["fma_leakage"]:.6f},  # pJ/cycle
""")


# Quick test function
if __name__ == "__main__":
    print_depfin_energy_summary()


"""
Helper functions to generate accelergy estimation and make code shorter
"""
def aclg_energy_mul(class_name, width, width_op, technology, action_name, arguments):

    return accelergy_estimate_energy({
        "class_name": class_name,
        "attributes": {
            "width": width,
            "width_a": width_op,
            "width_b": width_op,
            "technology": technology,
        },
        "action_name": action_name,
        "arguments": arguments
    })

def aclg_energy_add(class_name, width, technology, action_name, arguments):

    return accelergy_estimate_energy({
        "class_name": class_name,
        "attributes": {
            "width": width,
            "technology": technology,
        },
        "action_name": action_name,
        "arguments": arguments
    })


def aclg_energy_mem(class_name, attributes, action_name, arguments):

    return accelergy_estimate_energy({
        "class_name": class_name,
        "attributes": attributes,
        "action_name": action_name,
        "arguments": arguments
    })


def aclg_area_mul(class_name, width, width_op, technology):
    
    return accelergy_estimate_area({
        "class_name": class_name,
        "attributes": {
            "width": width,
            "width_a": width_op,
            "width_b": width_op,
            "technology": technology,
        }
    }) 

def aclg_area_add(class_name, width, technology):

    return accelergy_estimate_area({
        "class_name": class_name,
        "attributes": {
            "width": width,
            "technology": technology,
        }
    })

def aclg_area_mem(class_name, attributes):

    return accelergy_estimate_area({
        "class_name": class_name,
        "attributes": attributes,
    })


"""
Helper function to instantiate a smartbuffer register file (see Accelergy docs)

Set "energy" to True to estimate energy, set it to "False" for area.
"""
def smartbuffer_registerfile(depth, word_bits, value_bits, cycle_seconds, technology, action = None, energy = True):
    assert not energy or action in ["read", "write", "leak"], f"SmartBuffer SRAM Estimator: Action ({action}) must be one of 'read', 'write', or 'leak' when estimating energy!"
    std_width, std_depth = 32, 64
    # NOTE: currently Accelergy is bugged and does not apply those scales, for consistency we do not apply them too...
    dynamic_energy_scale = 1#(16/std_width)*((12/std_depth)**(1.56/2))
    static_energy_scale = 1#(16/std_width)*(12/std_depth) # = area_scale

    registers_attributes = {
        "global_cycle_seconds": cycle_seconds,
        "technology": technology,
    }
    registers_address_gen_attributes = {
        "n_bits": value_bits,
        "precision": value_bits,
        "datawidth": value_bits,
        "technology": technology,
        "global_cycle_seconds": cycle_seconds,
        "cycle_seconds": cycle_seconds,
    }
    if energy:
        return accelergy_estimate_energy(query={
                "class_name": "aladdin_register",
                "attributes": registers_attributes,
                "action_name": action,
                "arguments": {"global_cycle_seconds": cycle_seconds}
            })*max(std_width, word_bits)*(dynamic_energy_scale if action != "leak" else static_energy_scale*max(std_depth, depth)) + accelergy_estimate_energy(query={
                "class_name": "aladdin_comparator",
                "attributes": registers_attributes,
                "action_name": ("compare" if action != "leak" else action),
                "arguments": {"global_cycle_seconds": cycle_seconds}
            })*max(std_depth, depth)*(dynamic_energy_scale if action != "leak" else static_energy_scale) + accelergy_estimate_energy(query={
                "class_name": "intadder",
                "attributes": registers_address_gen_attributes,
                "action_name": ("add" if action != "leak" else action),
                "arguments": {"global_cycle_seconds": cycle_seconds}
            })*(1 if action != "leak" else 2)
    else:
        return accelergy_estimate_area(query={
                "class_name": "aladdin_register",
                "attributes": registers_attributes,
            })*max(std_width, word_bits)*static_energy_scale*max(std_depth, depth) + accelergy_estimate_area(query={
                "class_name": "aladdin_comparator",
                "attributes": registers_attributes,
            })*max(std_depth, depth)*static_energy_scale + accelergy_estimate_area(query={
                "class_name": "intadder",
                "attributes": registers_address_gen_attributes,
            })*2

"""
Helper function to instantiate a smartbuffer SRAM (see Accelergy docs)

Set "energy" to True to estimate energy, set it to "False" for area.
"""
def smartbuffer_SRAM(depth, word_bits, rw_ports, banks, cycle_seconds, technology, action = None, energy = True):
    assert not energy or action in ["read", "write", "leak"], f"SmartBuffer SRAM Estimator: Action ({action}) must be one of 'read', 'write', or 'leak' when estimating energy!"
    
    SRAM_attributes = {
        "n_rw_ports": rw_ports,
        "depth": depth,
        "width": word_bits,
        "technology": technology,
        "cycle_seconds": cycle_seconds,
        "n_banks": banks,
    }
    address_gen_attributes = {
        "n_bits": max(1, ceil(log2(depth))) if depth >= 1 else 1,
        "precision": max(1, ceil(log2(depth))) if depth >= 1 else 1,
        "datawidth": max(1, ceil(log2(depth))) if depth >= 1 else 1,
        "technology": technology,
        "global_cycle_seconds": cycle_seconds,
        "cycle_seconds": cycle_seconds,
    }
    
    if energy:
        return accelergy_estimate_energy(query={
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
                "action_name": action,
                "arguments": {"global_cycle_seconds": cycle_seconds}
            }) + accelergy_estimate_energy(query={
                "class_name": "intadder",
                "attributes": address_gen_attributes,
                "action_name": ("add" if action != "leak" else action),
                "arguments": {"global_cycle_seconds": cycle_seconds}
            })*(1 if action != "leak" else 2)
    else:
        return accelergy_estimate_area(query={
                "class_name": "SRAM",
                "attributes": SRAM_attributes,
            }) + accelergy_estimate_area(query={
                "class_name": "intadder",
                "attributes": address_gen_attributes,
            })*2

#---------------------------------------------------------------
# >>> EYERISS <<<

def get_arch_eyeriss_hw_data(
        name : str = "Eyeriss (Accelergy data)",
        #global_buffer_size : int = 16384*64, # bits 'depth' in timeloop-accelergy # 16384*8,
        #global_buffer_banks : int = 32, # 2,
        sa_cols : int = 14,
        sa_rows : int = 12,
        #in_reg_size : int = 12*16, # bits # 12*2,
        #w_reg_size : int = 192*16, # bits # 192*2, 
        #out_reg_size : int = 16*16, # bits #16*2,
        is_fp : bool = False,
        in_mac_precision : int = 8,
        out_mac_precision : int = 32,
        parallel_macs : int = 1,
        mac_latency : int = 1
        ) -> Arch:

    myParam = ParamInfo()
    number_of_PEs = sa_cols*sa_rows
    cycle_seconds = myParam.cycle_seconds
    technology = myParam.technology
    arguments = myParam.arguments
    DRAM_attributes = myParam.DRAM_attributes
    DRAM_size_elem = myParam.DRAM_size//in_mac_precision
    DRAM_bandwith_elem = myParam.DRAM_bandwidth//in_mac_precision
    SRAM_attributes = myParam.SRAM_attributes
    SRAM_size_elem = myParam.SRAM_size//in_mac_precision
    SRAM_bandwith_elem = myParam.SRAM_bandwidth//in_mac_precision
    
    InRegister_size_elem = myParam.InRegister_size//in_mac_precision
    InRegister_depth = myParam.InRegister_size//myParam.InRegister_word_bits
    InRegister_bandwith_elem = myParam.InRegister_bandwidth//in_mac_precision

    WRegister_size_elem = myParam.WRegister_size//in_mac_precision
    WRegister_depth = myParam.WRegister_size//myParam.WRegister_word_bits
    WRegister_bandwith_elem = myParam.WRegister_bandwidth//in_mac_precision

    # NOTE Modeling the writing of 1 value at out_mac_precision as out_mac_precision/in_mac_precision elements at in_mac_precision precision
    OutRegister_size_elem = myParam.OutRegister_size//in_mac_precision # NOTE This is a bit tricky... 
    OutRegister_depth = myParam.OutRegister_size//myParam.OutRegister_word_bits
    OutRegister_bandwith_elem = myParam.OutRegister_bandwidth//in_mac_precision 

    #----------------------------------------------------------
    full_precision = myParam.precision
    #----------------------------------------------------------
    # LF: modeling of the low precision funtional units, considering the cost of the
    # adder tree distributed among the multipliers (i.e., each functional
    # unit is "composed" of a multiplier and a piece of the adder tree
    #----------------------------------------------------------
    type_multiplier = "aladdin_multiplier"
    width_multiplier = 2*in_mac_precision
    type_adder = "aladdin_adder"
       
    ComputeLevelEnergy = aclg_energy_mul(type_multiplier, width_multiplier, in_mac_precision, technology, "read", arguments) + \
                         aclg_energy_add(type_adder, out_mac_precision, technology, "read", arguments)

    ComputeLevelLeakage = aclg_energy_mul(type_multiplier, width_multiplier, in_mac_precision, technology, "leak", arguments) + \
                          aclg_energy_add(type_adder, out_mac_precision, technology, "leak", arguments)


    ComputeLevelArea = aclg_area_mul(type_multiplier, width_multiplier, in_mac_precision, technology) + \
                       aclg_area_add(type_adder, out_mac_precision, technology)

    #----------------------------------------------------------

    arch = Arch([
        MemLevel(
            name = "DRAM",
            dataflow_constraints = [], # ['N', 'M', 'K'],
            size = DRAM_size_elem, #2**64-1, # number of entries
            read_wordline_access_energy = aclg_energy_mem("DRAM", DRAM_attributes, "read", arguments),
            write_wordline_access_energy = aclg_energy_mem("DRAM", DRAM_attributes, "write", arguments),
            leakage_energy = aclg_energy_mem("DRAM", DRAM_attributes, "leak", arguments),
            word_bits = myParam.DRAM_word_bits,
            value_bits = in_mac_precision,
            area = aclg_area_mem("DRAM", DRAM_attributes),
            bandwidth = DRAM_bandwith_elem, # operands per cycle (shared)
            factors_constraints = {},
            bypasses = []
        ),
        MemLevel(
            name = "GlobalBuffer",
            dataflow_constraints = [], #WS,
            size = SRAM_size_elem, #global_buffer_size, # number of entries of 'in_mac_precision' bits
            read_wordline_access_energy = aclg_energy_mem("SRAM", SRAM_attributes, "read", arguments),
            write_wordline_access_energy = aclg_energy_mem("SRAM", SRAM_attributes, "write", arguments),
            leakage_energy = aclg_energy_mem("SRAM", SRAM_attributes, "leak", arguments),
            word_bits = myParam.SRAM_word_bits,
            value_bits = in_mac_precision,
            area = aclg_area_mem("SRAM", SRAM_attributes),
            bandwidth = SRAM_bandwith_elem, # operands per cycle (shared): 8 @32 bits, 32 @8 bits, 64 @4 bits, 128 @2 bits 
            factors_constraints = {},
            bypasses = ['w']
        ),
        FanoutLevel(
            name = "SACols",
            mesh = sa_cols,
            dims = ['Q', 'M'],
            area = 0,
            factors_constraints = {}
        ),
        FanoutLevel(
            name = "SARows",
            mesh = sa_rows,
            dims = ['S', 'C', 'M'],
            area = 0,
            factors_constraints = {}
        ),
    
        MemLevel(
            name = "InRegister",
            dataflow_constraints = [], #WS,
            size = InRegister_size_elem, # number of entries: 
            read_wordline_access_energy = smartbuffer_registerfile(InRegister_depth, full_precision, full_precision, cycle_seconds, technology, "read"),
            write_wordline_access_energy = smartbuffer_registerfile(InRegister_depth, full_precision, full_precision, cycle_seconds, technology, "write"),
            leakage_energy = smartbuffer_registerfile(InRegister_depth, full_precision, full_precision, cycle_seconds, technology, "leak"),
            word_bits = myParam.InRegister_word_bits,
            value_bits = in_mac_precision,
            area = smartbuffer_registerfile(InRegister_depth, full_precision, full_precision, cycle_seconds, technology, energy = False),
            bandwidth = InRegister_bandwith_elem, #4 # operands per cycle (shared): as for SRAM
            factors_constraints = {'M': 1, 'C': 1, 'P': 1},
            bypasses = ['w', 'out']
        ),
        MemLevel(
            name = "WRegister",
            dataflow_constraints = [], #WS,
            size = WRegister_size_elem, # number of entries
            read_wordline_access_energy = smartbuffer_registerfile(WRegister_depth, full_precision, full_precision, cycle_seconds, technology, "read"),
            write_wordline_access_energy = smartbuffer_registerfile(WRegister_depth, full_precision, full_precision, cycle_seconds, technology, "write"),
            leakage_energy = smartbuffer_registerfile(WRegister_depth, full_precision, full_precision, cycle_seconds, technology, "leak"),
            word_bits = myParam.WRegister_word_bits,
            value_bits = in_mac_precision,
            area = smartbuffer_registerfile(WRegister_depth, full_precision, full_precision, cycle_seconds, technology, energy = False),
            bandwidth = WRegister_bandwith_elem, #4 # operands per cycle (shared): as for SRAM
            factors_constraints = {'M': 1, 'P': 1},
            bypasses = ['in', 'out']
        ),
        MemLevel(
            name = "OutRegister",
            dataflow_constraints = [], #WS,
            size = OutRegister_size_elem, # number of entries
            read_wordline_access_energy = smartbuffer_registerfile(OutRegister_depth, full_precision, full_precision, cycle_seconds, technology, "read"),
            write_wordline_access_energy = smartbuffer_registerfile(OutRegister_depth, full_precision, full_precision, cycle_seconds, technology, "write"),
            leakage_energy = smartbuffer_registerfile(OutRegister_depth, full_precision, full_precision, cycle_seconds, technology, "leak"),
            word_bits = myParam.OutRegister_word_bits,
            value_bits = in_mac_precision,
            area = smartbuffer_registerfile(OutRegister_depth, full_precision, full_precision, cycle_seconds, technology, energy = False),
            bandwidth = OutRegister_bandwith_elem, # 4 # operands per cycle (shared): as for SRAM
            factors_constraints = {'C': 1, 'N': 1},
            bypasses = ['in', 'w']
        ),
        ComputeLevel(
            name = "Compute",
            mesh = parallel_macs,
            dims = ['S', 'R', 'C'],
            compute_energy = ComputeLevelEnergy,
            leakage_energy = ComputeLevelLeakage,
            area = ComputeLevelArea,
            cycles = mac_latency,
        )
    ], coupling=conv_coupling_with_stride_and_batches, name = name)
    
    if Settings.VERBOSE:
        print(f"\nEnergy per action in {arch.name}:")
        printEnergyPerAction(arch)
        print(f"\nArea per level in {arch.name}:")
        printAreaPerLevel(arch)
        print(f"Total area of {arch.name}: {arch.totalArea(True):.3e} um^2\n")
    return arch
