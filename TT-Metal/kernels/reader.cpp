#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);
    uint32_t src0_dram_noc_x = get_arg_val<uint32_t>(2);
    uint32_t src0_dram_noc_y = get_arg_val<uint32_t>(3);
    uint32_t src1_dram_noc_x = get_arg_val<uint32_t>(4);
    uint32_t src1_dram_noc_y = get_arg_val<uint32_t>(5);

    uint64_t src0_noc_addr = get_noc_addr(src0_dram_noc_x, src0_dram_noc_y, src0_addr);
    uint64_t src1_noc_addr = get_noc_addr(src1_dram_noc_x, src1_dram_noc_y, src1_addr);

    uint32_t c_in0_size = get_tile_size(tt::CB::c_in0);
    uint32_t c_in1_size = get_tile_size(tt::CB::c_in1);
    uint32_t c_in0_write_address = get_write_ptr(tt::CB::c_in0);
    uint32_t c_in1_write_address = get_write_ptr(tt::CB::c_in1);

    cb_reserve_back(tt::CB::c_in0, 1);
    cb_reserve_back(tt::CB::c_in1, 1);

    noc_async_read(src0_noc_addr, c_in0_write_address, c_in0_size);
    noc_async_read(src1_noc_addr, c_in1_write_address, c_in1_size);

    noc_async_read_barrier();

    cb_push_back(tt::CB::c_in0, 1);
    cb_push_back(tt::CB::c_in1, 1);
}