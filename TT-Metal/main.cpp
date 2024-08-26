#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include <iostream>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    /* Silicon accelerator setup */
    Device *device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t single_tile = 32;
    tt_metal::InterleavedBufferConfig dram_config{
                .device= device,
                .size = single_tile,
                .page_size = single_tile,
                .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    auto src0_dram_noc_coord = src0_dram_buffer->noc_coordinates();
    auto src1_dram_noc_coord = src1_dram_buffer->noc_coordinates();
    auto dst_dram_noc_coord = dst_dram_buffer->noc_coordinates();

    uint32_t src0_dram_noc_x = src0_dram_noc_coord.x;
    uint32_t src0_dram_noc_y = src0_dram_noc_coord.y;

    uint32_t src1_dram_noc_x = src1_dram_noc_coord.x;
    uint32_t src1_dram_noc_y = src1_dram_noc_coord.y;

    uint32_t dst_dram_noc_x = dst_dram_noc_coord.x;
    uint32_t dst_dram_noc_y = dst_dram_noc_coord.y;

    // set L1 CBs
    constexpr uint32_t src0_cb_index = CB::c_in0;
    constexpr uint32_t src1_cb_index = CB::c_in1;
    constexpr uint32_t output_cb_index = CB::c_out0;

    CircularBufferConfig src0_cb_config = CircularBufferConfig(
        single_tile,
        {{src0_cb_index, tt::DataFormat::Float16_b}}
    ).set_page_size(src0_cb_index, single_tile);
    CircularBufferConfig src1_cb_config = CircularBufferConfig(
        single_tile,
        {{src1_cb_index, tt::DataFormat::Float16_b}}
    ).set_page_size(src1_cb_index, single_tile);   
    CircularBufferConfig output_cb_config = CircularBufferConfig(
        single_tile,
        {{output_cb_index, tt::DataFormat::Float16_b}}
    ).set_page_size(output_cb_index, single_tile);

    // create CBs for src0, src1, src2, and output
    tt_metal::CreateCircularBuffer(program, core, src0_cb_config);
    tt_metal::CreateCircularBuffer(program, core, src1_cb_config);
    tt_metal::CreateCircularBuffer(program, core, output_cb_config);

    KernelHandle reader_kernel = CreateKernel(
        program,
        "../kernels/reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default
        });

    KernelHandle writer_kernel = CreateKernel(
        program,
        "../kernels/writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, 
            .noc = NOC::RISCV_0_default
        });

    vector<uint32_t> compute_kernel_args = {};
    KernelHandle div_kernel = CreateKernel(
        program,
        "../kernels/div.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = (std::vector<uint32_t>) {},
        }
    );

    std::vector<uint32_t> src0_vec;
    std::vector<uint32_t> src1_vec;
    std::vector<uint32_t> output_vec;

    constexpr uint32_t src0_val = 2.0f;
    constexpr uint32_t src1_val = 4.0f;

    src0_vec = create_constant_vector_of_bfloat16(single_tile, src0_val);
    src1_vec = create_constant_vector_of_bfloat16(single_tile, src1_val);

    // write src0, src1, and src2 into their DRAM buffers
    EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, false);

    auto reader_args = {
        src0_dram_buffer->address(),
        src1_dram_buffer->address(),
        src0_dram_noc_x,
        src0_dram_noc_y,
        src1_dram_noc_x,
        src1_dram_noc_y,
    };
    auto writer_args = {
        dst_dram_buffer->address(),
        dst_dram_noc_x,
        dst_dram_noc_y
    };

    SetRuntimeArgs(program, reader_kernel, core, reader_args);
    SetRuntimeArgs(program, div_kernel, core, { 0 }); // Set do_mul to 0
    SetRuntimeArgs(program, writer_kernel, core, writer_args);
    
    // EXECUTE!!!
    EnqueueProgram(cq, program, false);
    Finish(cq);
    EnqueueReadBuffer(cq, dst_dram_buffer, output_vec, true);

    // Copy the output buffer to the input buffer
    EnqueueWriteBuffer(cq, src0_dram_buffer, output_vec, false);

    SetRuntimeArgs(program, div_kernel, core, { 1 }); // Set do_mul to 1

    // Run kernels again!!!!!
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Get final output
    EnqueueReadBuffer(cq, dst_dram_buffer, output_vec, true);

    bfloat16 *result_bf16 = reinterpret_cast<bfloat16*>(output_vec.data());
    float result = result_bf16[0].to_float();
    float expected = ((1.0/2.0)*4.0);

    std::cout << "Result: " << result << std::endl;
    std::cout << "Expected: " << expected << std::endl;

    if(is_close(result, expected)) {
        std::cout << "Test Passed" <<std::endl;
    } else {
        std::cout << "Test Failed" << std::endl;
    }
    CloseDevice(device);
}