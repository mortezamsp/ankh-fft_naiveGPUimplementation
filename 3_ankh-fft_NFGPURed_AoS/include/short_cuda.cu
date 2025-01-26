#ifndef SHORT_CUDA_CUH
#define SHORT_CUDA_CUH

#include <cuda_runtime.h>
#include "short_cuda.cuh"
#include "grid.hpp"
#include "kernel.hpp"
using namespace std;
#include<iostream>

using namespace ankh;

nfgpu::nfgpu(int lvl)
{
    initGPU();
}
nfgpu::~nfgpu()
{
    freeGPUMemory();
}
void nfgpu::initGPU()
{
    //init gpu
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA-capable device found!\n");
    } else {
        // printf("working with device 0\n");
        cudaSetDevice(0); // Use the first available device
    }
}
void nfgpu::freeGPUMemory()
{
    cudaFree(d_aosdata);
    cudaFree(d_res);   
}

//template<typename charge_type, int KRNL> double short_range(leaf<charge_type>* leaves, int lvl, int Nimages, double diam){
void nfgpu::short_range_naiiveGPU_dataCollection(ankh::leaf<ankh::charge_dipole_quadrupole_3<double>>* leaves, int lvl, int Nimages, double diam)
{
    int numBoxes = (1<<lvl)*(1<<lvl)*(1<<lvl);

    //calculate the padding size
    int max_nptr = 0;
    max_parts = 0;
    max_qs = 0;
    for(int leaf=0; leaf < numBoxes; leaf++)
    {
        max_nptr = leaves[leaf].nprt > max_nptr ? leaves[leaf].nprt : max_nptr;
        max_parts = leaves[leaf].prts.size() > max_parts ? leaves[leaf].prts.size() : max_parts;
        max_qs = leaves[leaf].Qs.size() > max_qs ? leaves[leaf].Qs.size() : max_qs;
    }

    //allocate memory
    paddingSize = 1 + 3 * max_parts + max_qs;//numptr:1, parts:max_parts*3,  qs:max_qs*1
    double* aosdata = (double*)malloc(numBoxes * paddingSize * sizeof(double));
    int arraylength = numBoxes * paddingSize;
    int dataSize = arraylength * sizeof(double);
    std::cout
        <<"                         Near field - max_nptr=" <<max_nptr<< " max_parts="<<max_parts
        <<" max_qs="<<max_qs<<" numBoxes="<<numBoxes<<" E = " << (1<<lvl) <<std::endl;
    std::cout << "                         Near field - Memory Allocation (" 
        << round((double)dataSize/1024.0/1024.0) <<" MB), paddingSize = " << paddingSize << std::endl;
    //collect data
    // printf("data collection...");
    int E = (1<<lvl); 
    int idx_trg, offset, prtsSize, qsize;
    for(int idx_trg_x = 0; idx_trg_x < E; idx_trg_x++)//for all boxes in 3d grid
    {
    	for(int idx_trg_y = 0; idx_trg_y < E; idx_trg_y++)
        {
			for(int idx_trg_z = 0; idx_trg_z < E; idx_trg_z++)
            {
				idx_trg = idx_in_grid_(idx_trg_x,idx_trg_y,idx_trg_z,E); //sequential number of box in 3d grid
                // printf("idx_trg[%d], ", idx_trg);
                offset = idx_trg * paddingSize;
                aosdata[offset] = (double)leaves[idx_trg].nprt;
                offset++;

                prtsSize = leaves[idx_trg].prts.size();
                // printf("prtssize = %d ", prtsSize);
                for(int k=0; k<prtsSize; k++)
                {
                    aosdata[offset + k*3 + 0] = leaves[idx_trg].prts[k].x;
                    aosdata[offset + k*3 + 1] = leaves[idx_trg].prts[k].y;
                    aosdata[offset + k*3 + 2] = leaves[idx_trg].prts[k].z;
                }
                offset += 3 * max_parts;// prtsSize;

                qsize = leaves[idx_trg].Qs.size();
                // printf("qsize = %d ", qsize);
                for(int k=0; k<qsize; k++)
                {
                    // printf("%d,", k);
                    aosdata[offset + k] = leaves[idx_trg].Qs[k];
                }
                // printf("\n");
			}
		}
    }

    //allocate data in GPU
    // printf("mem allocation...\n");
    cudaMalloc((void**)&d_aosdata, dataSize); 
    cudaMalloc((void**)&d_res, numBoxes * sizeof(double)); 

    //copy to GPU
    // printf("data transfer...\n");
    cudaError_t er = cudaMemcpy(d_aosdata, aosdata, dataSize, cudaMemcpyHostToDevice);
    if(er != cudaSuccess){
        printf("error in copying d_aosdata : %s\n", cudaGetErrorString(er));
    }

    //free RAM
    delete aosdata;
}
double nfgpu::short_range_wrapper(int lvl, int Nimages, double diam)
{
    int E = (1<<lvl); 
    int nleaves = E*E*E;
    
    cudaError_t kernelLaunchError = cudaGetLastError();
    if (kernelLaunchError != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(kernelLaunchError));
    }//else{printf("no error at initial steps\n");}


    //calculate NF for each box, each device thread handles a single box
    dim3 block(max(1,min(1024, nleaves)));
    dim3 grid(max(1, nleaves / 1024));
    // printf("nleaves %d grid %d block %d\n", nleaves, grid.x, block.x);
    // printf("max_parts=%d, max_qs=%d\n", max_parts,max_qs);
    short_range_cuda<<<grid, block>>>(d_aosdata, paddingSize, max_parts, max_qs, d_res, Nimages, diam, E);
    kernelLaunchError = cudaGetLastError();
    if (kernelLaunchError != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(kernelLaunchError));
    }
    kernelLaunchError = cudaDeviceSynchronize();
    if(kernelLaunchError != cudaSuccess)
    {
        printf("Error calling kernel short_range : %s", cudaGetErrorString(kernelLaunchError));
    }


    //summarizing NFs of all boxes
    //double res = short_range_singleThreadReduction(d_res, E);
    //std::cout << "sequential reduction result = " << res << std::endl;

    //a bit faster reduction in gpu
    printf("                         Near field - reduction...\n");
    int chunk = max(1, nleaves / 1024);
    dim3 blockr(min(1024, chunk));
    dim3 gridr(max(1, chunk / blockr.x));
    short_range_reduction<<<gridr,blockr>>>(d_res, E);
    double d_final_res = 0;
    cudaMemcpy(&d_final_res, d_res, 1 * sizeof(double), cudaMemcpyDeviceToHost);
    kernelLaunchError = cudaGetLastError();
    if (kernelLaunchError != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(kernelLaunchError));
    }
    kernelLaunchError = cudaDeviceSynchronize();
    if(kernelLaunchError != cudaSuccess)
    {
        printf("Error calling kernel short_range_reduction : %s", cudaGetErrorString(kernelLaunchError));
    }

    return d_final_res;
}


__device__ inline void idx_in_grid_gpu(int i, int j, int k, int E, int* res){*res = i*E*E + j*E + k;}
__global__ void short_range_cuda(double* aosdata, int paddingSize, int max_parts, int max_qs, double* d_res, int Nimages, double diam, int length)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < length*length*length)
    {
        int idx_trg_x = tid / length / length;
        int idx_trg_y = (tid % (length * length)) / length;
        int idx_trg_z = tid % length;
        double tmp;
        int idx_trg = tid, idx_src;
        int offs_trg = idx_trg * paddingSize, offs_src;
        //eacth thread handles all interactions of a single leaf, box
        double res = 0;
        for(int sidx = idx_trg_x-1; sidx <= 1+idx_trg_x; sidx++){ //for all E2 neighbors in any dierction in 3d grid
            for(int sidy = idx_trg_y-1; sidy <= 1+idx_trg_y; sidy++){
                for(int sidz = idx_trg_z-1; sidz <= idx_trg_z+1; sidz++){
                    idx_in_grid_gpu(sidx,sidy,sidz,length, &idx_src); //sequential number of box in 3d grid
                    if(Nimages == 0){
                        if(sidx > -1 && sidx < length && sidy > -1 && sidy < length && sidz > -1 && sidz < length)
                        {
                            offs_src = idx_src * paddingSize;
                            tmp = direct_interaction_0(
                                aosdata[offs_trg], &aosdata[offs_trg + 1], &aosdata[offs_trg + 1 + 3 * max_parts],
                                aosdata[offs_src], &aosdata[offs_src + 1], &aosdata[offs_src + 1 + 3 * max_parts],
                                0,0,0,diam); //direct kenel call
                            res += tmp;
                        }
                        if(tid == 0) printf("N%f,",res);
                    }
                    else{ //handling periodicity
                        int u0 = sgn((sidx < 0 ? length : 0) + (sidx >= length ? -length : 0));
                        int u1 = sgn((sidy < 0 ? length : 0) + (sidy >= length ? -length : 0));
                        int u2 = sgn((sidz < 0 ? length : 0) + (sidz >= length ? -length : 0));
                        idx_in_grid_gpu(sidx+u0,sidy+u1,sidz+u2,length, &idx_src);
                        offs_src = idx_src * paddingSize;
                        tmp = direct_interaction_0( 
                            aosdata[offs_trg], &aosdata[offs_trg + 1], &aosdata[offs_trg + 1 + 3 * max_parts],
                            aosdata[offs_src], &aosdata[offs_src + 1], &aosdata[offs_src + 1 + 3 * max_parts],
                            u0,u1,u2,diam);
                        res += tmp;
                    }
                }
            }
        }
        d_res[tid] = res;
    }
}
double nfgpu::short_range_singleThreadReduction(double* d_res, int length)
{
    int nleaves = length*length*length;
    double res = 0;
    double *h_res = (double*)malloc(nleaves * sizeof(double));
    cudaMemcpy(h_res, d_res, nleaves * sizeof(double), cudaMemcpyDeviceToHost);
    for(int i = 0; i < nleaves; i++)
    {
            res += h_res[i];
            //printf("%f,", h_res[i]);
    }
    delete h_res;
    return res;
}
__global__ void short_range_reduction(double* d_res, int length)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //each thread summarizes 1024 elements from d_res array
    int nleaves = length*length*length;
    int numtherads = nleaves / 1024;
    if(tid < numtherads)
    {
        double res_ = 0;
        int idx0 = tid * 1024;
        for(int i = 0; i < 1024; i++)
        {
            if(idx0 + i < length*length*length)
                res_ += d_res[idx0 + i];
            else
                printf("thread[%d], step[%d] localElement[%d]\n", tid, i, idx0+i);
        }
        if(tid>length*length*length) printf("writing error");
        d_res[tid] = res_;
    }

    __syncthreads();

    if(tid == 0)
    {
        double res_ = 0;
        for(int i = 0; i < numtherads; i++)
        {
            if(i > length*length*length)printf("reading error");
            res_ += d_res[i];
        }
        d_res[0] = res_;
    }

    __syncthreads();
}
//  double direct_interaction_0<charge_dipole_quadrupole_3<double>,1>(leaf<charge_dipole_quadrupole_3<double> >& trg, leaf<charge_dipole_quadrupole_3<double> >& src, int u0, int u1, int u2, double diam)
__device__ double direct_interaction_0(int trg_nptr, double* trg_prts, double* trg_Qs, int src_nprt, double* src_prts, double* src_Qs, int u0, int u1, int u2, double diam)
{
    double res = 0.;
    double dx, dy, dz;
    for(int i = 0; i < trg_nptr; i++){
		for(int j = 0; j < src_nprt; j++){
			dx = trg_prts[i*3+0] - src_prts[j*3+0] + u0*diam;
			dy = trg_prts[i*3+1] - src_prts[j*3+1] + u1*diam;
			dz = trg_prts[i*3+2] - src_prts[j*3+2] + u2*diam;
			res += energy_gpu(dx,dy,dz,&(trg_Qs[10*i]),&(src_Qs[10*j]),0.01); ////0.01 = Ewald parameter used in analyze
      }
    }
    return res;
 }
//the function which was used in analyze test
// __device__ double energy_gpu(const double& dx, const double& dy, const double& dz, double* trg, double* src, double kernel_beta)
// {
//     double R     = dx*dx+dy*dy+dz*dz;
//     double K     = 0.;
//     if(R > 1.e-16){K = interact0(R);}		
//     double K2    = K*K;
//     double ci    = trg[0];
//     double dix   = trg[1];
//     double diy   = trg[2];
//     double diz   = trg[3];
//     double qixx  = trg[4];
//     double qixy  = trg[5];
//     double qixz  = trg[6];
//     double qiyy  = trg[7];
//     double qiyz  = trg[8];
//     double qizz  = trg[9];
//     double ck    = src[0];
//     double dkx   = src[1];
//     double dky   = src[2];
//     double dkz   = src[3];
//     double qkxx  = src[4];
//     double qkxy  = src[5];
//     double qkxz  = src[6];
//     double qkyy  = src[7];
//     double qkyz  = src[8];
//     double qkzz  = src[9];
//     double dri   = - dix*dx - diy*dy - diz*dz;     // - <di ,r  >
//     double drk   = - dkx*dx - dky*dy - dkz*dz;     // - <dk ,r  >
//     double dik   =   dix*dkx + diy*dky + diz*dkz;  //   <di ,dk >
//     double qrix  = - qixx*dx - qixy*dy - qixz*dz;  // - <qix,r  > = qrix
//     double qriy  = - qixy*dx - qiyy*dy - qiyz*dz;  // - <qiy,r  > = qriy
//     double qriz  = - qixz*dx - qiyz*dy - qizz*dz;  // - <qiz,r  > = qriz
//     double qrkx  = - qkxx*dx - qkxy*dy - qkxz*dz;  // - <qkx,r  > = qrkx
//     double qrky  = - qkxy*dx - qkyy*dy - qkyz*dz;  // - <qky,r  > = qrky
//     double qrkz  = - qkxz*dx - qkyz*dy - qkzz*dz;  // - <qkz,r  > = qrkz
//     double qrri  = - qrix*dx - qriy*dy - qriz*dz;  // - <qri,r  >
//     double qrrk  = - qrkx*dx - qrky*dy - qrkz*dz;  // - <qrk,r  >
//     double qrrik = qrix*qrkx + qriy*qrky + qriz*qrkz; // <qri,qrk>
//     double qik   = 2.0*(qixy*qkxy+qixz*qkxz+qiyz*qkyz) + qixx*qkxx + qiyy*qkyy + qizz*qkzz;
//     double diqrk = dix*qrkx + diy*qrky + diz*qrkz;
//     double dkqri = dkx*qrix + dky*qriy + dkz*qriz;
//     double term1 = ci*ck;
//     double term2 = ck*dri - ci*drk + dik;
//     double term3 = ci*qrrk + ck*qrri - dri*drk + 2.0*(dkqri-diqrk+qik);
//     double term4 = dri*qrrk - drk*qrri - 4.0*qrrik;
//     double term5 = qrri*qrrk;
//     double bl    = 0.;
//     if(R > 1.e-16){bl = interact1(R, kernel_beta);}
//     double res   = bl * term1;
//     double p2a   = exp(-R*kernel_beta*kernel_beta)/(kernel_beta*sqrt(M_PI));
//     double bt22  = 2.*kernel_beta*kernel_beta;
//     p2a *= bt22;
//     bl   = K2 * (   bl + p2a);
//     res += bl * term2;
//     p2a *= bt22;
//     bl   = K2 * (3.*bl + p2a);
//     res += bl * term3;
//     p2a *= bt22;
//     bl   = K2 * (5.*bl + p2a);
//     res += bl * term4;
//     p2a *= bt22;
//     bl   = K2 * (7.*bl + p2a);
//     res += bl * term5;
//     return res;
// }
__device__  double energy_gpu(const double& dx, const double& dy, const double& dz, double* __restrict__ trg, double* __restrict__ src, double kernel_beta) {
	__shared__ double inm[21]; //interrmediate variables
    // Load trg and src values into arrays
    __shared__ double trg_vals[10];
	__shared__ double src_vals[10];
    for (int i = 0; i < 10; i++) {
        trg_vals[i] = trg[i];
        src_vals[i] = src[i];
    }
    // Compute intermediate values
    inm[0] = -(trg_vals[1] * dx); // -<di, r>
    inm[0] += -(trg_vals[2] * dy); // -<di, r>
    inm[0] += -(trg_vals[3] * dz); // -<di, r>
    inm[1] = -(src_vals[1] * dx); // -<dk, r>
    inm[1] += -(src_vals[2] * dy); // -<dk, r>
    inm[1] += -(src_vals[3] * dz); // -<dk, r>
    inm[2] = trg_vals[1] * src_vals[1];
    inm[2] += trg_vals[2] * src_vals[2];
    inm[2] += trg_vals[3] * src_vals[3]; // <di, dk>
    inm[3] = -(trg_vals[4] * dx); // -<qix, r>
    inm[3] += -(trg_vals[5] * dy);
    inm[3] += -(trg_vals[6] * dz);
    inm[4] = -(trg_vals[5] * dx); // -<qiy, r>
    inm[4] += -(trg_vals[7] * dy); // -<qiy, r>
    inm[4] += -(trg_vals[8] * dz); // -<qiy, r>
    inm[5] = -(trg_vals[6] * dx); // -<qiz, r>
    inm[5] += -(trg_vals[8] * dy); // -<qiz, r>
    inm[5] += -(trg_vals[9] * dz); // -<qiz, r>

    inm[6] = -(src_vals[4] * dx + src_vals[5] * dy + src_vals[6] * dz); // -<qkx, r>
    inm[7] = -(src_vals[5] * dx + src_vals[7] * dy + src_vals[8] * dz); // -<qky, r>
    inm[8] = -(src_vals[6] * dx + src_vals[8] * dy + src_vals[9] * dz); // -<qkz, r>

    inm[10] = -(inm[3] * dx + inm[4] * dy + inm[5] * dz); // -<qri, r>
    inm[11] = -(inm[6] * dx + inm[7] * dy + inm[8] * dz); // -<qrk, r>
    inm[9] = inm[3] * inm[6] + inm[4] * inm[7] + inm[5] * inm[8]; // <qri, qrk>

    inm[12] = 2.0 * (trg_vals[5] * src_vals[5] + trg_vals[6] * src_vals[6] + trg_vals[8] * src_vals[8]) +
                 trg_vals[4] * src_vals[4] + trg_vals[7] * src_vals[7] + trg_vals[9] * src_vals[9];

    inm[13] = trg_vals[1] * inm[6] + trg_vals[2] * inm[7] + trg_vals[3] * inm[8];
    inm[14] = src_vals[1] * inm[3] + src_vals[2] * inm[4] + src_vals[3] * inm[5];

    // Compute terms
    inm[15] = trg_vals[0] * src_vals[0];
    inm[16] = src_vals[0] * inm[0] - trg_vals[0] * inm[1] + inm[2];
    inm[17] = trg_vals[0] * inm[11] + src_vals[0] * inm[10] - inm[0] * inm[1] + 2.0 * (inm[14] - inm[13] + inm[12]);
    inm[18] = inm[0] * inm[11] - inm[1] * inm[10] - 4.0 * inm[9];
    inm[19] = inm[10] * inm[11];


	
	double R = dx * dx + dy * dy + dz * dz;
    double K = (R > 1.e-16) ? 1. / sqrt(R) : 0.0;
    inm[20] = K * K;



    // Compute final result
    double r = sqrt(R);
    r = std::erfc(kernel_beta*r) / r; 
    double bl = (R > 1.e-16) ? r : 0.0;
    double res = bl * inm[15];

    double p2a = exp(-R * kernel_beta * kernel_beta) / (kernel_beta * sqrt(M_PI));
    double bt22 = 2.0 * kernel_beta * kernel_beta;
    p2a *= bt22;

    bl = inm[20] * (bl + p2a);
    res += bl * inm[16];

    p2a *= bt22;
    bl = inm[20] * (3.0 * bl + p2a);
    res += bl * inm[17];

    p2a *= bt22;
    bl = inm[20] * (5.0 * bl + p2a);
    res += bl * inm[18];

    p2a *= bt22;
    bl = inm[20] * (7.0 * bl + p2a);
    res += bl * inm[19];

    return res;
}
inline int idx_in_grid_(int i, int j, int k, int E)
{
    return i*E*E + j*E + k;
}
__device__ int sgn(int x) 
{
    if (x < 0) return -1;
    if (x > 0) return 1;
    return 0;
}
#endif // SHORT_CUDA_CUH
