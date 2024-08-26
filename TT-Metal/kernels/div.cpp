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
        uint32_t do_mul = get_arg_val<uint32_t>(0);
        init_sfpu(tt::CB::c_in0);
        cb_wait_front(tt::CB::c_in0, 1); // 2
        cb_wait_front(tt::CB::c_in1, 1); // 4
        cb_reserve_back(tt::CB::c_out0, 1);
        tile_regs_acquire();

        copy_tile(tt::CB::c_in0, 0, 1);
        copy_tile(tt::CB::c_in1, 0, 2);

        DPRINT_MATH(DPRINT << "do_mul: " << do_mul << ENDL());
        DPRINT_MATH(DPRINT << "c_in0:" << ENDL());
        dprint_tensix_dest_reg(1);
        DPRINT_MATH(DPRINT << "c_in1:" << ENDL());
        dprint_tensix_dest_reg(2);

        if (do_mul == 0) {
            copy_tile(tt::CB::c_in0, 0, 0);
            recip_tile_init();
            recip_tile(0);
        }
        else {
            mul_tiles_init(tt::CB::c_in0, tt::CB::c_in1);
            mul_tiles(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0);
        }

        DPRINT_MATH(DPRINT << "c_out0:" << ENDL());
        dprint_tensix_dest_reg(0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, tt::CB::c_out0, 0);
        tile_regs_release();

        cb_pop_front(tt::CB::c_in0, 1);
        cb_pop_front(tt::CB::c_in1, 1);
        cb_push_back(tt::CB::c_out0, 1);
    }
}