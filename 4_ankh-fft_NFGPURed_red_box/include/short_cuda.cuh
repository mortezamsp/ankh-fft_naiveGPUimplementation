#pragma once
#include <cuda_runtime.h>
#include "grid.hpp"
#include "multipoles.hpp"
#include "kernel.hpp"
using namespace ankh;

class nfgpu
{
    public: nfgpu(int lvl);
    ~nfgpu();

    //leaves data needed for GPU
    double *d_reddata;
    int *d_redOffset; 
    double *d_res;

    private: int max_parts, max_qs;

    private: void initGPU();
    private: void freeGPUMemory();
    public: void short_range_naiiveGPU_dataCollection(ankh::leaf<ankh::charge_dipole_quadrupole_3<double>>* leaves, int lvl, int Nimages, double diam);
    public: double short_range_wrapper(int lvl, int Nimages, double diam);
    private: double short_range_singleThreadReduction(double* d_res, int length);
};


inline int idx_in_grid_(int i, int j, int k, int E);
inline int sgn(double x);


__global__ void short_range_cuda(
    //leaves data
    double* reddata, int* redOffset,
    //output
    double* d_res,
    //other parameters
    double diam, int length);
__global__ void short_range_reduction(double* d_res, int length);
__device__ double direct_interaction_0(
    //taget
    int trg_nptr, double* trg_prts, double* trg_Qs,
    //source
    int src_nprt, double* src_prts, double* src_Qs,
    //others
    int u0, int u1, int u2, double diam);
__device__ inline void idx_in_grid_gpu(int i, int j, int k, int E, int* res);
//the function which was used in analyze test
__device__ double energy_gpu(const double& dx, const double& dy, const double& dz, double* trg, double* src, double kernel_beta);
__device__ double direct_interaction_0p(bool print, int trg_nptr, double* trg_prts, double* trg_Qs, int src_nprt, double* src_prts, double* src_Qs, int u0, int u1, int u2, double diam);
