#ifndef CLSOLVER_H
#define CLSOLVER_H

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#include <CL/cl2.hpp>

#include <cstdlib>
#include <vector>
#include <mutex>
#include <memory>
#include <thread>
#include "solverstructs.h"
#include "presolver.h"
#include "isolver.h"


class ClSolver : public ISolver
{
public:
    bool init(uint8_t boardsize, uint8_t placed);
    uint64_t solve_subboard(const std::vector<start_condition>& start);
    static void enumerate_devices();

    static ClSolver* makeClSolver(unsigned int platform, unsigned int device);
    static ClSolver* makeClSolver(cl::Platform platform, cl::Device device);

private:
    struct ThreadData {
        cl::CommandQueue cmdQueue;
        cl::Kernel clRelaunchKernel;
        std::vector<cl::Kernel> clMainKernels;
        cl::Buffer clWorkspaceBuf;
        cl::Buffer clWorkspaceSizeBuf;
        cl::Buffer clOutputBuf;
        cl::Kernel sumKernel;
        cl::Buffer sumBuffer;
        std::unique_ptr<std::thread> thread;
        uint64_t result;
        std::vector<start_condition_t> hostStartBuf;
    };
    bool allocateThreads(size_t cnt);


private:
    enum class OCL_VERSION
    {
        OCL_1X,
        OCL_2X
    };

    ClSolver();
    void threadWorker(uint32_t id, std::mutex &pre_lock);
    PreSolver nextPre(std::mutex &pre_lock);

    OCL_VERSION ocl_version;

    std::vector<start_condition> start;
    size_t solved = 0;

    uint8_t gpu_depth = 0;
    uint8_t presolve_depth = 0;
    uint8_t placed = 0;
    uint8_t boardsize = 0;
    uint64_t workspace_size = 0;
    cl::Context context;
    cl::Device device;
    cl::DeviceCommandQueue devQueue;
    cl::Program program;
    std::vector<ThreadData> threads;
};

#endif // CLSOLVER_H
