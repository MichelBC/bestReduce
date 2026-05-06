/***********************************************************************************/
/*                                                                                 */
/*   IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.            */
/*   By downloading, copying, installing or using the software you agree           */
/*   to this license.  If you do not agree to this license, do not download,       */
/*   install, copy or use the software.                                            */
/*                                                                                 */
/*  BSD 3-Clause License                                                           */
/*                                                                                 */
/*  Copyright (c) 2024-2026, Michel Brasil Cordeiro and Wagner M. Nunan Zola       */
/*  All rights reserved.                                                           */
/*                                                                                 */
/*  Redistribution and use in source and binary forms, with or without             */
/*  modification, are permitted provided that the following conditions are met:    */
/*                                                                                 */
/*  1. Redistributions of source code must retain the above copyright notice,      */
/*     this list of conditions and the following disclaimer.                       */
/*                                                                                 */
/*  2. Redistributions in binary form must reproduce the above copyright notice,   */
/*     this list of conditions and the following disclaimer in the documentation   */
/*     and/or other materials provided with the distribution.                      */
/*                                                                                 */
/*  3. Neither the name of the copyright holder nor the names of its               */
/*     contributors may be used to endorse or promote products derived from        */
/*     this software without specific prior written permission.                    */
/*                                                                                 */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"    */
/*  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE      */
/*  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE */
/*  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE   */
/*  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL     */
/*  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR     */
/*  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER     */
/*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,  */
/*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE  */
/*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.           */
/*                                                                                 */
/*  If you use this software or a modified version of it,                          */
/*  please cite the most relevant among the following papers:                      */
/*                                                                                 */
/*  - Optimized Parallel Reduction for Regular and Irregular Segments on GPU       */
/*    Cordeiro, Michel B. and Nunan Zola, Wagner M.                                */
/*    In: Concurrency and Computation: Practice and Experience, 2025,              */
/*    Special Issue SSCAD24                                                        */
/*                                                                                 */
/***********************************************************************************/

#include "cuda_functors.cuh"

//void uint32_reduceMax(uint32_t *In, long numElements, int *Out )   // will generate the sum of n_random numbers PER THREAD
// __device__ __forceinline__
// void reduce_impl( 
template<
  typename Op=Max<DATA_TYPE>,
  typename T=DATA_TYPE>
__global__
void uint32_reduceMax(
	T * __restrict__ In,
	int numElements, 
	T * __restrict__ Out )   // will generate the sum of n_random numbers PER THREAD
{   
		Op op;
		register int i;
		register T myVal = op.identity();
		__shared__ T blockVal;

		if( threadIdx.x == 0 )
			 blockVal = op.identity();
		__syncthreads();

//    for (long i = ((long)blockDim.x * blockIdx.x + threadIdx.x); 
//              i < numElements; 
//              i += (long)gridDim.x * blockDim.x) {
		for (i = blockDim.x * blockIdx.x + threadIdx.x; 
				 i < numElements; 
				 i += gridDim.x * blockDim.x) {

				myVal = op( In[i], myVal );         
		}

		op.atomic( &blockVal, myVal );
		__syncthreads();
		if( threadIdx.x == 0 )
				op.atomic( Out, blockVal );       
}

// __global__
// void uint32_reduceMax(uint32_t * __restrict__ In, 
// 	int numElements, 
// 	uint32_t * __restrict__ Out )   // will generate the sum of n_random numbers PER THREAD
// {
// 		reduce_impl<Max<uint32_t>,uint32_t>(In, numElements, Out);
// }

template<
  typename Op=Max<DATA_TYPE>,
  typename T=DATA_TYPE, 
  typename T4=DATA_TYPEx4> 
// __device__ __forceinline__
// void reducex4_impl(T * __restrict__ In,
__global__
void uint32x4_reduceMax(T * __restrict__ In, 
	int numElements, 
	T * __restrict__ Out )   // will generate the sum of n_random numbers PER THREAD
{
		Op op;
		register int i;
		register T myVal = op.identity();
		T4 *In4 = (T4 *) In;
		
		
		__shared__ T blockVal;

		if( threadIdx.x == 0 )
			 blockVal = op.identity();
		__syncthreads();

//    for (long i = ((long)blockDim.x * blockIdx.x + threadIdx.x); 
//              i < numElements; 
//              i += (long)gridDim.x * blockDim.x) {
		for (i = (blockDim.x * blockIdx.x + threadIdx.x); 
				 i < numElements/4; 
				 i += (gridDim.x * blockDim.x) ) {
		//for (i = blockDim.x * blockIdx.x + threadIdx.x; 
		//     i < numElements; 
		//     i += (gridDim.x * blockDim.x)*4 ) {

				//T4 element = reinterpret_cast<T4*>(In)[i];
				T4 element = In4[i];
				//myVal = max( element, myVal );

				myVal = op( element.x, myVal );
				myVal = op( element.y, myVal );         
				myVal = op( element.z, myVal );         
				myVal = op( element.w, myVal );      
					 
		}   


		// in only one thread, process final elements (if there are any)
		// int remainder = (numElements%4);
		// if( (blockIdx.x * blockDim.x + threadIdx.x)==(numElements/4) && remainder!=0) {
		// 	while(remainder) {
		// 		int idx = numElements - remainder--;
		// 		myVal = op( In[idx], myVal );
		// 	}
		// }

		int remainder = (numElements%4);
		int idx = numElements - remainder + (blockDim.x * blockIdx.x + threadIdx.x);
		if (idx < numElements) 
			myVal = op( In[idx], myVal );


		op.atomic( &blockVal, myVal );
		__syncthreads();
		if( threadIdx.x == 0 )
				op.atomic( Out, blockVal );       
}

// __global__
// void uint32x4_reduceMax(T * __restrict__ In, 
// 	int numElements, 
// 	T * __restrict__ Out )   // will generate the sum of n_random numbers PER THREAD
// {
// 		reducex4_impl<Max<T>,T,uint4>(In, numElements, Out);
// }
// --- uint32x4_reduceMax

// ui32_shfl_reduceMax
template<
  typename Op=Max<DATA_TYPE>,
  typename T=DATA_TYPE>
// __device__ __forceinline__
// void shfl_reduce_impl(T * __restrict__ In, int numElements, T * __restrict__ Out )
__global__
void ui32_shfl_reduceMax(T * __restrict__ In, int numElements, 
	T * __restrict__ Out )
{   
		Op op;
		register int i;
		register T myVal = op.identity();
		__shared__ T blockVal;

		if( threadIdx.x == 0 )
			 blockVal = op.identity();
		__syncthreads();

//    for (long i = ((long)blockDim.x * blockIdx.x + threadIdx.x); 
//              i < numElements; 
//              i += (long)gridDim.x * blockDim.x) {
		for (i = blockDim.x * blockIdx.x + threadIdx.x; 
				 i < numElements; 
				 i += gridDim.x * blockDim.x) {

				myVal = op( In[i], myVal );         
		}

		// shfl_down warp reduce
		myVal = op( myVal, __shfl_down_sync( 0xffffffff, myVal, 16 ) ); // assuming warpSize=32
		myVal = op( myVal, __shfl_down_sync( 0xffffffff, myVal,  8 ) ); // assuming warpSize=32
		myVal = op( myVal, __shfl_down_sync( 0xffffffff, myVal,  4 ) ); // assuming warpSize=32
		myVal = op( myVal, __shfl_down_sync( 0xffffffff, myVal,  2 ) ); // assuming warpSize=32
		myVal = op( myVal, __shfl_down_sync( 0xffffffff, myVal,  1 ) ); // assuming warpSize=32
		if( threadIdx.x % 32 == 0 )
				op.atomic( &blockVal, myVal );
		__syncthreads();
		if( threadIdx.x == 0 )
				op.atomic( Out, blockVal );       
}

// __global__
// void ui32_shfl_reduceMax(T * __restrict__ In, int numElements, T * __restrict__ Out )   // will generate the sum of n_random numbers PER THREAD
// {
// 		shfl_reduce_impl<Max<T>,T>(In, numElements, Out);
// }

template <typename T>
__global__
void printReduceResult(T *Out, int position )   
{
	 if( threadIdx.x == 0 ) {
		 printf( "Vector[%d]: ", position );
		 printf( "%llu ", (unsigned long long)Out[position] );
		 printf( "\n" );
	 }
}

// ui32_sh_mem
template<
  typename Op=Max<DATA_TYPE>,
  typename T=DATA_TYPE>
// __device__ __forceinline__
// void sh_mem_reduce_impl(
__global__
void ui32_sh_mem_reduceMax(
	T* data, 
	uint size,
	T* out
){
	Op op;
	const int tid = threadIdx.x+blockDim.x*blockIdx.x;

	extern __shared__ T sh_values[]; // size should be N_THREADS*sizeof(T)

	T myValue = op.identity();
	for(int i=tid;i<size;i+=blockDim.x*gridDim.x){
		myValue = op( myValue, data[i] );
	}
	sh_values[threadIdx.x] = myValue;

	for (int stride=N_THREADS/2; stride > 0; stride /= 2){
		__syncthreads();
		if (threadIdx.x < stride){
			sh_values[threadIdx.x] = op( sh_values[threadIdx.x], sh_values[threadIdx.x+stride] );
		}
	}
	if(threadIdx.x == 0){
		op.atomic(out,sh_values[0]);
	}
}


// __global__
// void ui32_sh_mem_reduceMax(
// 	uint32_t* data, 
// 	uint size,
// 	uint32_t* out
// ){
// 	sh_mem_reduce_impl<Max<uint32_t>,uint32_t>(data, size, out);
// }
// --- ui32_sh_mem