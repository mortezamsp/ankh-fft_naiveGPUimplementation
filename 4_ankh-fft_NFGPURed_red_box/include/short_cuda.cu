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
    cudaFree(d_reddata);
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
    vector<double> reddata;
    vector<int> redOffset;
    redOffset.push_back(0);
    
    //collect data
    //printf("data collection...");
    int E = (1<<lvl); 
    int idx_trg, idx_src, localDataCounter, globalCcounter = 0, prtsSize, qsize;
    //for all last level leafs
    for(int idx_trg_x = 0; idx_trg_x < E; idx_trg_x++)
    {
    	for(int idx_trg_y = 0; idx_trg_y < E; idx_trg_y++)
        {
			for(int idx_trg_z = 0; idx_trg_z < E; idx_trg_z++)
            {
                localDataCounter = 0;
                
                //copy data of central box
				idx_trg = idx_in_grid_(idx_trg_x,idx_trg_y,idx_trg_z,E); //sequential number of box in 3d grid
                prtsSize = leaves[idx_trg].prts.size();
                qsize = leaves[idx_trg].Qs.size();

                if(prtsSize == 0)
                {    
                    //printf("idx_src[%d] prtsSize = 0\n", idx_src);
                    reddata.push_back(0);
                    reddata.push_back(0);
                    localDataCounter += 2;
                    globalCcounter += localDataCounter;
                    redOffset.push_back(globalCcounter);
                    continue;
                }

                reddata.push_back((double)prtsSize);
                reddata.push_back((double)qsize);
                localDataCounter += 2;
                
                for(int k=0; k<prtsSize; k++)
                {
                    reddata.push_back(leaves[idx_trg].prts[k].x);
                    reddata.push_back(leaves[idx_trg].prts[k].y);
                    reddata.push_back(leaves[idx_trg].prts[k].z);
                }
                localDataCounter += 3 * prtsSize;

                for(int k=0; k<qsize; k++)
                {
                    reddata.push_back(leaves[idx_trg].Qs[k]);
                }
                localDataCounter += qsize;
                
                
                //for all 27 E2 neighbours
                for(int sidx = idx_trg_x-1; sidx <= 1+idx_trg_x; sidx++) 
                {
                    for(int sidy = idx_trg_y-1; sidy <= 1+idx_trg_y; sidy++)
                    {
                        for(int sidz = idx_trg_z-1; sidz <= idx_trg_z+1; sidz++)
                        {
                            if(sidx == idx_trg_x && sidy==idx_trg_y && sidz==idx_trg_z)
                                continue;

                            //idx_in_grid_gpu(sidx,sidy,sidz,E, &idx_src); //sequential number of box in 3d grid
                            int u0 = sgn((sidx < 0 ? E : 0) + (sidx >= E ? -E : 0));
                            int u1 = sgn((sidy < 0 ? E : 0) + (sidy >= E ? -E : 0));
                            int u2 = sgn((sidz < 0 ? E : 0) + (sidz >= E ? -E : 0));
                            idx_src = idx_in_grid_(sidx+u0,sidy+u1,sidz+u2,E);
                            
                            prtsSize = leaves[idx_src].prts.size();
                            qsize = leaves[idx_src].Qs.size();

                            if(prtsSize == 0)
                            {    
                                //printf("idx_src[%d] prtsSize = 0\n", idx_src);
                                reddata.push_back(0);
                                reddata.push_back(0);
                                localDataCounter += 2;
                                continue;
                            }
                            
                            reddata.push_back((double)prtsSize);
                            reddata.push_back((double)qsize);
                            localDataCounter += 2;

                            for(int k=0; k<prtsSize; k++)
                            {
                                reddata.push_back(leaves[idx_src].prts[k].x);
                                reddata.push_back(leaves[idx_src].prts[k].y);
                                reddata.push_back(leaves[idx_src].prts[k].z);
                            }
                            localDataCounter += 3 * prtsSize;

                            for(int k=0; k<qsize; k++)
                            {
                                reddata.push_back(leaves[idx_src].Qs[k]);
                            }
                            localDataCounter += qsize;
                        }
                    }
                }

                globalCcounter += localDataCounter;
                redOffset.push_back(globalCcounter);
            }
		}
    }

    
    //allocate data in GPU
    //printf("mem allocation...\n");
    int dataSize = reddata.size() * sizeof(double);
    std::cout
        <<"                          Near field - max_nptr=" <<max_nptr<< " max_parts="<<max_parts
        <<" max_qs="<<max_qs<<" numBoxes="<<numBoxes<<" E = " << (1<<lvl) <<std::endl;
    std::cout 
        << "                         Near field - Memory Allocation (" 
        << round((double)(dataSize + redOffset.size() * sizeof(int))/1024.0/1024.0) <<" MB)," << std::endl;
    cudaMalloc((void**)&d_reddata, dataSize); 
    cudaMalloc((void**)&d_redOffset, redOffset.size() * sizeof(int)); 
    cudaMalloc((void**)&d_res, numBoxes * sizeof(double)); 

    //copy to GPU
    //printf("data transfer...\n");
    cudaError_t er = cudaMemcpy(d_reddata, reddata.data(), dataSize, cudaMemcpyHostToDevice);
    if(er != cudaSuccess){
        printf("error in copying d_redData : %s\n", cudaGetErrorString(er));
    }
    er = cudaMemcpy(d_redOffset, redOffset.data(), redOffset.size() * sizeof(int), cudaMemcpyHostToDevice);
    if(er != cudaSuccess){
        printf("error in copying d_redOffset : %s\n", cudaGetErrorString(er));
    }

    //free RAM
    reddata.clear();
    reddata.shrink_to_fit();
    redOffset.clear();
    redOffset.shrink_to_fit();
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
    //printf("nleaves %d grid %d block %d\n", nleaves, grid.x, block.x);
    short_range_cuda<<<grid, block>>>(d_reddata, d_redOffset, d_res, diam, E);
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
__global__ void short_range_cuda(double* reddata, int* redOffset, double* d_res, double diam, int length)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < length*length*length)
    {
        double tmp;
        double res = 0;
        int errcnt = 0;
        
        //self interaction
        int offset = redOffset[tid];
        int cprts = (int)reddata[offset + 0]; //number of particles in center box
        int cqs = (int)reddata[offset + 1]; //number of qs of center box
        //printf("tid[%d], offset = %d, cprts=%d, cqs=%d\n", tid, offset, cprts, cqs);
        if(cprts > 0)
        {
            tmp = direct_interaction_0(
                cprts, &reddata[offset + 2], &reddata[offset + 2 + cprts * 3],
                cprts, &reddata[offset + 2], &reddata[offset + 2 + cprts * 3],
                0,0,0,diam); 
            res += tmp;
            //printf("tid[%d] self interaction done");


            //for all E2 neighbours, except itself
            int localofs = offset + 2 + cprts * 3 + cqs;  //offset of first child
            int iprts;//number of particles in nei box i
            int iqs; //number of qs of nei box i
            for(int i = 0; i<27; i++)
            {
                iprts = reddata[localofs + 0];
                iqs = reddata[localofs + 1];
                
                // printf("tid[%d: prts:%d qs:%d] child[%d: offset:%d prts:%d qs:%d]\n",
                //  tid, cprts, cqs, 
                //  i, localofs, iprts, iqs);
                
                if(iprts > 0)
                {
                    tmp = direct_interaction_0(
                        cprts, &reddata[offset + 2], &reddata[offset + 2 + cprts * 3],
                        iprts, &reddata[localofs + 2], &reddata[localofs + 2 + iprts * 3],
                        0,0,0,diam); 
                    res += tmp;
                    localofs += (2 + iprts * 3 + iqs);
                }
                else
                {
                    localofs += 2;
                    errcnt++;
                }
            }
            //printf("tid[%d] E2 interaction done");
        }
        //else
            //printf("tid[%d] has no E2 nei\n", tid);
        
        // printf("tid[%d: crsp:%d cqs:%d] has %d empty E2 nei\n", tid, cprts, cqs, errcnt);

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
int sgn(double x) 
{
    return (x < 0) ? -1 : (x == 0) ? 0 : 1;
}
#endif // SHORT_CUDA_CUH
