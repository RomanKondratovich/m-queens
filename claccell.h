#ifndef CLACCELL_H
#define CLACCELL_H

#include <cstdint>
#include <vector>
#include <CL/cl.h>
#include <CL/cl.hpp>
#include "cpusolver.h"

class ClAccell
{
public:
    ClAccell();

    static ClAccell *makeClAccell(unsigned int platform, unsigned int device);
    bool init(size_t threads, size_t lut_size, size_t high_stride, size_t low_stride,
              const std::vector<uint32_t>& high_sizes, const std::vector<uint32_t>& low_sizes,
              const diags_packed_t* lut_high_prob, const diags_packed_t* lut_low_prob);
    uint64_t count(size_t thread, uint32_t lut_idx, cpuSolver::cand_lock_t *lck, const diags_packed_t *candidates, bool prob);
    uint64_t get_count();

private:
    uint64_t get_cl_count();
    cl::Context context;
    cl::Device device;
    cl::Program program;
    std::vector<cl::Kernel> clKernel;

    cl::Buffer clFlatHighProb;
    cl::Buffer clFlatLowProb;

    std::vector<cl::CommandQueue> cmdQueue;
    std::vector<cl::Buffer> clResultCnt;

    std::vector<cl::Buffer> clCanBuff;
    size_t first_free = 0;

    uint64_t cnt = 0;

    size_t threads = 0;

    size_t lut_high_prob_stride = 0;
    size_t lut_low_prob_stride = 0;
    std::vector<uint32_t> lut_high_prob_sizes;
    std::vector<uint32_t> lut_low_prob_sizes;
};

#endif // CLACCELL_H
