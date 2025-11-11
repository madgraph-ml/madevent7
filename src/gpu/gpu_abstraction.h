#pragma once

#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#define gpuMalloc cudaMalloc
#define gpuMallocAsync cudaMallocAsync
#define gpuFree cudaFree
#define gpuFreeAsync cudaFreeAsync
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDefault cudaMemcpyDefault
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuStreamPerThread cudaStreamPerThread
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStream_t cudaStream_t
#define gpuEvent_t cudaEvent_t
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuStreamWaitEvent cudaStreamWaitEvent
#define gpuEventRecord cudaEventRecord
#define gpuDeviceSynchronize cudaDeviceSynchronize

#define gpublasStatus_t cublasStatus_t
#define gpublasHandle_t cublasHandle_t
#define gpublasGetStatusString cublasGetStatusString
#define GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define gpublasCreate cublasCreate
#define gpublasDestroy cublasDestroy
#define gpublasSetStream cublasSetStream
#define gpublasDgemm cublasDgemm
#define gpublasDgemv cublasDgemv
#define GPUBLAS_OP_N CUBLAS_OP_N
#define GPUBLAS_OP_T CUBLAS_OP_T

#define gpurandStatus_t curandStatus_t
#define gpurandGenerator_t curandGenerator_t
#define gpurandCreateGenerator curandCreateGenerator
#define gpurandDestroyGenerator curandDestroyGenerator
#define gpurandSetPseudoRandomGeneratorSeed curandSetPseudoRandomGeneratorSeed
#define gpurandSetStream curandSetStream
#define gpurandGenerateUniformDouble curandGenerateUniformDouble
#define GPURAND_STATUS_SUCCESS CURAND_STATUS_SUCCESS
#define GPURAND_RNG_PSEUDO_DEFAULT CURAND_RNG_PSEUDO_DEFAULT

#define thrust_par thrust::cuda::par

#elif defined __HIPCC__

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <rocrand/rocrand.h>

#define gpuMalloc hipMalloc
#define gpuMallocAsync hipMallocAsync
#define gpuFree hipFree
#define gpuFreeAsync hipFreeAsync
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDefault hipMemcpyDefault
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuStreamPerThread hipStreamPerThread
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStream_t hipStream_t
#define gpuEvent_t hipEvent_t
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuStreamWaitEvent hipStreamWaitEvent
#define gpuEventRecord hipEventRecord
#define gpuDeviceSynchronize hipDeviceSynchronize

#define gpublasStatus_t rocblasStatus_t
#define gpublasHandle_t rocblasHandle_t
#define gpublasGetStatusString rocblasGetStatusString
#define GPUBLAS_STATUS_SUCCESS ROCBLAS_STATUS_SUCCESS
#define gpublasCreate rocblasCreate
#define gpublasDestroy rocblasDestroy
#define gpublasSetStream rocblasSetStream
#define gpublasDgemm rocblasDgemm
#define gpublasDgemv rocblasDgemv
#define GPUBLAS_OP_N ROCBLAS_OP_N
#define GPUBLAS_OP_T ROCBLAS_OP_T

#define gpurandStatus_t rocrandStatus_t
#define gpurandGenerator_t rocrandGenerator_t
#define gpurandCreateGenerator rocrandCreateGenerator
#define gpurandDestroyGenerator rocrandDestroyGenerator
#define gpurandSetPseudoRandomGeneratorSeed rocrandSetPseudoRandomGeneratorSeed
#define gpurandSetStream rocrandSetStream
#define gpurandGenerateUniformDouble rocrandGenerateUniformDouble
#define GPURAND_STATUS_SUCCESS ROCRAND_STATUS_SUCCESS
#define GPURAND_RNG_PSEUDO_DEFAULT ROCRAND_RNG_PSEUDO_DEFAULT

#define thrust_par thrust::hip_rocprim::par;

#endif
