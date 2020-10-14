#include "claccell.h"

#include <ctime>
#include <ios>
#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <queue>
#include <list>
#include <thread>

ClAccell::ClAccell()
{
}

ClAccell* ClAccell::makeClAccell(unsigned int platform, unsigned int device)
{
    cl_int err = 0;
    ClAccell* solver = new ClAccell();
    if(!solver) {
        std::cout << "Failed to allocate memory" << std::endl;
        return nullptr;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty()) {
        std::cout << "No OpenCL platforms found" << std::endl;
        return nullptr;
    }

    if(!(platform < platforms.size())) {
        std::cout << "Invalid OpenCL platform" << std::endl;
        return nullptr;
    }

    const cl::Platform& used_platform = platforms[platform];

    std::cout << "Platform name: " << used_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "Platform version: " << used_platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

    std::vector<cl::Device> devices;

    err = used_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(err != CL_SUCCESS) {
        std::cout << "getDevices failed" << std::endl;
        return nullptr;
    }

    if(devices.empty()) {
        std::cout << "No devices found" << std::endl;
        return nullptr;
    }

    if(!(device < devices.size())) {
        std::cout << "Invalid OpenCL platform" << std::endl;
        return nullptr;
    }

    const cl::Device& used_device = devices[device];

    // check if device is available
    bool available = used_device.getInfo<CL_DEVICE_AVAILABLE>(&err);
    if(err != CL_SUCCESS) {
        std::cout << "getInfo<CL_DEVICE_AVAILABLE> failed" << std::endl;
        return nullptr;
    }

    if(!available) {
        std::cout << "OpenCL device not available" << std::endl;
        return nullptr;
    }

    solver->device = used_device;

    std::cout << "selected Device: " << used_device.getInfo<CL_DEVICE_NAME>(&err) << std::endl;
    if(err != CL_SUCCESS) {
        std::cout << "getInfo<CL_DEVICE_NAME> failed" << std::endl;
        return nullptr;
    }

    solver->context = cl::Context(used_device, nullptr, nullptr, nullptr, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Context failed" << std::endl;
        return nullptr;
    }

    // load source code
    std::ifstream sourcefile("claccell.cl");
    std::string sourceStr((std::istreambuf_iterator<char>(sourcefile)),
                     std::istreambuf_iterator<char>());

    // create OpenCL program
    solver->program = cl::Program(solver->context, sourceStr, false, &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Program failed" << std::endl;
        return nullptr;
    }

    return solver;
}

bool ClAccell::init(size_t threads, size_t lut_size, size_t high_stride, size_t low_stride,
                    const std::vector<uint32_t> &high_sizes, const std::vector<uint32_t> &low_sizes,
                    const diags_packed_t* lut_high_prob, const diags_packed_t* lut_low_prob)
{
    this->threads = threads;
    lut_low_prob_sizes = low_sizes;
    lut_high_prob_sizes = high_sizes;

    lut_low_prob_stride = low_stride;
    lut_high_prob_stride = high_stride;

    cl_int err = 0;

    // allocate OpenCL buffers for lut
    clFlatHighProb = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                lut_size * high_stride * sizeof(diags_packed_t), const_cast<diags_packed_t*>(lut_high_prob), &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clFlatHighProb failed: " << err << std::endl;
        return false;
    }

    clFlatLowProb = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                lut_size * low_stride * sizeof(diags_packed_t), const_cast<diags_packed_t*>(lut_low_prob), &err);
    if(err != CL_SUCCESS) {
        std::cout << "cl::Buffer clFlatLowProb failed: " << err << std::endl;
        return false;
    }

    // compile Program
    std::ostringstream optionsStream;
    optionsStream << "-D MAX_CANDIDATES=" << std::to_string(cpuSolver::max_candidates);
    std::string options = optionsStream.str();

    std::cout << "OPTIONS: " << options << std::endl;

    cl_int builderr = program.build(options.c_str());

    std::cout << "OpenCL build log:" << std::endl;
    auto buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &err);
    std::cout << buildlog << std::endl;
    if(err != CL_SUCCESS) {
        std::cout << "getBuildInfo<CL_PROGRAM_BUILD_LOG> failed" << std::endl;
    }
    if(builderr != CL_SUCCESS) {
        std::cout << "program.build failed: " << builderr << std::endl;
        return false;
    }

    for(size_t t = 0; t < threads; t++) {
        // Create command queue.
        cmdQueue.push_back(cl::CommandQueue(context, device, 0, &err));
        if(err != CL_SUCCESS) {
            std::cout << "failed to create command queue: " << err << std::endl;
            return false;
        }

        // allocate OpenCL buffer for result
        std::vector<uint32_t> resultCnt;
        resultCnt.resize(cpuSolver::max_candidates, 0);
        clResultCnt.push_back(cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_HOST_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                 resultCnt.size() * sizeof (uint32_t), resultCnt.data(), &err));
        if(err != CL_SUCCESS) {
            std::cout << "cl::Buffer clResultCnt failed: " << err << std::endl;
            return false;
        }

        // allocate OpenCL buffer for candidates

        clCanBuff.push_back(cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_HOST_WRITE_ONLY,
                                 cpuSolver::max_candidates * sizeof (diags_packed_t), nullptr, &err));
        if(err != CL_SUCCESS) {
            std::cout << "cl::Buffer clCanBuff failed: " << err << std::endl;
            return false;
        }

        // create device kernel
        clKernel.push_back(cl::Kernel(program, "count_solutions_trans", &err));
        if(err != CL_SUCCESS) {
            std::cout << "cl::Kernel failed: " << err << std::endl;
            return {};
        }

        // allocate fixed args
        err = clKernel[t].setArg(1, clCanBuff[t]);
        if(err != CL_SUCCESS) {
            std::cout << "setArg 1 failed: " << err << std::endl;
            return {};
        }
        err = clKernel[t].setArg(2, clResultCnt[t]);
        if(err != CL_SUCCESS) {
            std::cout << "setArg 2 failed: " << err << std::endl;
            return {};
        }

        // create device kernel for cleanup
        clKernelCleanup.push_back(cl::Kernel(program, "count_solutions_trans_cleanup", &err));
        if(err != CL_SUCCESS) {
            std::cout << "cl::Kernel failed: " << err << std::endl;
            return {};
        }

        // allocate fixed args
        err = clKernelCleanup[t].setArg(1, clCanBuff[t]);
        if(err != CL_SUCCESS) {
            std::cout << "setArg 1 failed: " << err << std::endl;
            return {};
        }
        err = clKernelCleanup[t].setArg(2, clResultCnt[t]);
        if(err != CL_SUCCESS) {
            std::cout << "setArg 2 failed: " << err << std::endl;
            return {};
        }
    }

    return true;
}

uint64_t ClAccell::count(size_t thread, uint32_t lut_idx, cpuSolver::cand_lock_t* lck, const diags_packed_t *candidates, bool prob)
{
    // check which lut to use
    const auto& lut_lens = prob ? lut_high_prob_sizes : lut_low_prob_sizes;
    cl_int err = 0;

    // can skip upload on high prob, already uploaded by prev low prob execution
    if(!prob) {
        err = cmdQueue[thread].enqueueWriteBuffer(clCanBuff[thread], CL_FALSE, 0, cpuSolver::max_candidates * sizeof (diags_packed_t), candidates);
        if(lck) {
            lck->store(0, std::memory_order_relaxed);
        }
    }

    if(lut_lens[lut_idx] == 0) {
        return 0;
    }

    // allocate dynamic args
    err = clKernel[thread].setArg(0, prob ? clFlatHighProb : clFlatLowProb);
    if(err != CL_SUCCESS) {
        std::cout << "setArg 0 failed: " << err << std::endl;
        return {};
    }

    uint32_t stride = prob ? lut_high_prob_stride : lut_low_prob_stride;

    err = clKernel[thread].setArg(3, stride*lut_idx);
    if(err != CL_SUCCESS) {
        std::cout << "setArg 3 failed: " << err << std::endl;
        return {};
    }

    uint32_t range = prob ? lut_high_prob_sizes[lut_idx] : lut_low_prob_sizes[lut_idx];
    err = clKernel[thread].setArg(4, range);
    if(err != CL_SUCCESS) {
        std::cout << "setArg 4 failed: " << err << std::endl;
        return {};
    }

    // Launch kernel on the compute device.
    err = cmdQueue[thread].enqueueNDRangeKernel(clKernel[thread], cl::NullRange,
                                        cl::NDRange{cpuSolver::max_candidates/32}, cl::NDRange{32});
    if(err != CL_SUCCESS) {
        std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
        return {};
    }

    return 0;
}

uint64_t ClAccell::count_cleanup(size_t thread, uint32_t lut_idx, size_t cand_cnt, const diags_packed_t *candidates, bool prob)
{
    const auto& lut_lens = prob ? lut_high_prob_sizes : lut_low_prob_sizes;
    cl_int err = 0;

    // can skip upload on high prob, already uploaded by prev low prob execution
    if(!prob) {
        err = cmdQueue[thread].enqueueWriteBuffer(clCanBuff[thread], CL_FALSE, 0, cpuSolver::max_candidates * sizeof (diags_packed_t), candidates);
    }

    if(lut_lens[lut_idx] == 0) {
        return 0;
    }

    // allocate dynamic args
    err = clKernelCleanup[thread].setArg(0, prob ? clFlatHighProb : clFlatLowProb);
    if(err != CL_SUCCESS) {
        std::cout << "setArg 0 failed: " << err << std::endl;
        return {};
    }

    uint32_t stride = prob ? lut_high_prob_stride : lut_low_prob_stride;

    err = clKernelCleanup[thread].setArg(3, stride*lut_idx);
    if(err != CL_SUCCESS) {
        std::cout << "setArg 3 failed: " << err << std::endl;
        return {};
    }

    uint32_t range = prob ? lut_high_prob_sizes[lut_idx] : lut_low_prob_sizes[lut_idx];
    err = clKernelCleanup[thread].setArg(4, range);
    if(err != CL_SUCCESS) {
        std::cout << "setArg 4 failed: " << err << std::endl;
        return {};
    }

    // Launch kernel on the compute device.
    err = cmdQueue[thread].enqueueNDRangeKernel(clKernelCleanup[thread], cl::NullRange,
                                        cl::NDRange{cand_cnt}, cl::NullRange);
    if(err != CL_SUCCESS) {
        std::cout << "enqueueNDRangeKernel failed: " << err << std::endl;
        return {};
    }

    return 0;
}

uint64_t ClAccell::get_count()
{
    return get_cl_count();
}

uint64_t ClAccell::get_cl_count()
{
    std::vector<uint32_t> resultCnt;
    resultCnt.resize(cpuSolver::max_candidates, 0);

    cl_int err = 0;
    uint64_t res = 0;

    for(size_t t = 0; t < threads; t++) {
        err = cmdQueue[t].enqueueReadBuffer(clResultCnt[t], CL_TRUE, 0, resultCnt.size() * sizeof (uint32_t), resultCnt.data());
        if(err != CL_SUCCESS) {
            std::cout << "enqueueReadBuffer failed: " << err << std::endl;
        }


        for(const auto& i: resultCnt) {
            res += i;
        }

        // allocate OpenCL buffer for result
#if 0
        std::vector<uint32_t> emptyBuf;
        emptyBuf.resize(cpuSolver::max_candidates/64, 0);
        clResultCnt[t] = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_HOST_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                 emptyBuf.size() * sizeof (uint32_t), emptyBuf.data(), &err);
        if(err != CL_SUCCESS) {
            std::cout << "cl::Buffer clResultCnt failed: " << err << std::endl;
            return {};
        }
#endif
    }

    return res;
}


