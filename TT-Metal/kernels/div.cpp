#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/cb_api.h"
#include "debug/dprint_tensix.h"
#include "debug/dprint.h"

namespace NAMESPACE {
    void MAIN {
        init_sfpu(tt::CB::c_in0);

        acquire_dst(tt::DstMode::Half);
        cb_wait_front(tt::CB::c_in0, 1); // 2
        cb_wait_front(tt::CB::c_in1, 1); // 4
        cb_reserve_back(tt::CB::c_intermed0, 1); // Should be filled with zeros
        cb_reserve_back(tt::CB::c_out0, 1);

        copy_tile(tt::CB::c_in0, 0, 0);
        copy_tile(tt::CB::c_in1, 0, 1);
        copy_tile(tt::CB::c_intermed0, 0, 2);

        DPRINT_MATH(DPRINT << "c_in0:" << ENDL());
        dprint_tensix_dest_reg(0);

        DPRINT_MATH(DPRINT << "c_in1:" << ENDL());
        dprint_tensix_dest_reg(1);

        DPRINT_MATH(DPRINT << "c_intermed0:" << ENDL());
        dprint_tensix_dest_reg(2);

        // Reciprocal
        recip_tile_init();
        recip_tile(3);

        DPRINT_MATH(DPRINT << "recip_tile(3):" << ENDL());
        dprint_tensix_dest_reg(3);

        pack_tile(3, tt::CB::c_intermed0, 0);

        // Multiplication
        mul_tiles_init(tt::CB::c_in1, tt::CB::c_intermed0);
        mul_tiles(tt::CB::c_in1, tt::CB::c_intermed0, 0, 0, 4);

        DPRINT_MATH(DPRINT << "Multiplication result:" << ENDL());
        dprint_tensix_dest_reg(4);

        pack_tile(4, tt::CB::c_out0, 0);

        cb_pop_front(tt::CB::c_in0, 1);
        cb_pop_front(tt::CB::c_in1, 1);
        cb_push_back(tt::CB::c_intermed0, 1);
        cb_push_back(tt::CB::c_out0, 1);
        release_dst(tt::DstMode::Half);
    }
}