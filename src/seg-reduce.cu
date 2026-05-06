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

#ifndef SEGREDUCE
#define SEGREDUCE

#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <stdlib.h>
// #include <curand.h>
// #include <curand_kernel.h>
#include <cuda_runtime.h>

#include "defines-bestreduce.h"
// #include "exp-histo.c"
#include "cuda_functors.cuh"

template<
  typename Op=Max<DATA_TYPE>,
  typename T=DATA_TYPE>
__global__
void warpReduce(T * __restrict__ In, 
  int numElements, 
  uint* segStart,
  int numSegs,
  T * __restrict__ Out
)   // will generate the sum of n_random numbers PER THREAD
{   
  Op op;
  const int blockSegSize = ceil(numSegs/(float)gridDim.x);
  const int blockSegStart = blockSegSize*blockIdx.x;
  const int blockSegEnd = min(blockSegSize*(blockIdx.x+1),numSegs+1);
  const int nWarps = blockDim.x / WARP_SIZE;
  const int wid = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  for(int seg = blockSegStart + wid; seg < blockSegEnd; seg+=nWarps){
    T myVal = op.identity();
    for (int i = segStart[seg]+lane; i < segStart[seg+1]; i+=WARP_SIZE){
      myVal = op(myVal, In[i]);
    }
    for (int i = 1; i < WARP_SIZE; i*=2){
      myVal = op(myVal, __shfl_xor_sync( 0xffffffff, myVal,  i));
    }
    Out[seg] = myVal;
  }
}

template<
  typename Op=Max<DATA_TYPE>,
  typename T=DATA_TYPE,
  typename T4=DATA_TYPEx4>
__global__
void  blockReduce(T * __restrict__ In, 
                                 int numElements, 
                                 uint* segStart,
                                 int numSegs,
                                 T * __restrict__ Out )   // will generate the sum of n_random numbers PER THREAD
{   
  __shared__ T sh_val;
  Op op;
  
  int seg = blockIdx.x;

  // T* aligned_data = (T*) (((size_t)In + 15) & ~0xF);
  T4 *In4 = (T4 *) In;
  T myVal = op.identity();
  uint start = (segStart[seg]+3)/4;
  uint end = (segStart[seg+1])/4;
  
  int idxStart = segStart[seg]+threadIdx.x;
  if(idxStart < start*4 && idxStart < segStart[seg+1]){ 
    //deal with unaligned start
    myVal = op(myVal, In[idxStart]);
  }
  if(threadIdx.x == 0){
    sh_val = op.identity();   
  }
  __syncthreads();

  for( int i=start+threadIdx.x; i < end; i+=blockDim.x){
    T4 element = In4[i];
    myVal = op( element.x, myVal );
    myVal = op( element.y, myVal );         
    myVal = op( element.z, myVal );         
    myVal = op( element.w, myVal );   
  }

  int idxEnd = end*4 + threadIdx.x;
  if (idxEnd < segStart[seg+1] && idxEnd >= segStart[seg]) 
    myVal = op( In[idxEnd], myVal );

  op.atomic(&sh_val, myVal);
  __syncthreads(); // to ensures that all threads completed atomic op
    if(threadIdx.x == 0){
      Out[seg] = sh_val;
    }
}

template<
  typename Op=Max<DATA_TYPE>,
  typename T=DATA_TYPE,
  typename T4=DATA_TYPEx4>
__global__
void manyBlocksGr(T * __restrict__ In, 
                                 int numElements, 
                                 uint* segStart,
                                 int numSegs,
                                 T * __restrict__ Out,
                                 uint grain )   // will generate the sum of n_random numbers PER THREAD
{   
  Op op;
  int seg = blockIdx.x/grain;
  // int segSize = (segStart[seg+1] - segStart[seg])/4;
  int grainId = blockIdx.x%grain;

  uint start = threadIdx.x+(segStart[seg]+3)/4+
               grainId*blockDim.x;
  uint end = (segStart[seg+1])/4;

  T4 *In4 = (T4 *) In;

  T myVal = op.identity();


  if(grainId == 0){ 
    //first block will take care of the unaligned start
    int idxStart = segStart[seg]+threadIdx.x;
    int segStartCeil = (segStart[seg]+3)/4;
    if(idxStart < segStartCeil*4 && idxStart < segStart[seg+1]){ 
      myVal = op(myVal, In[idxStart]);
    }
  }
  
  __shared__ T sh_max;
  if(threadIdx.x == 0){ 
    sh_max = op.identity();
  }
  __syncthreads();

  for( int i=start; i < end; i+=blockDim.x*grain){
    T4 element = In4[i];

    myVal = op( element.x, myVal );
    myVal = op( element.y, myVal );         
    myVal = op( element.z, myVal );         
    myVal = op( element.w, myVal );   
  }

  if(grainId == grain-1){
    int idxEnd = end*4 + threadIdx.x;
    if (idxEnd < segStart[seg+1] && idxEnd >= segStart[seg]) 
      myVal = op( In[idxEnd], myVal );
  }


  op.atomic(&sh_max, myVal);
  __syncthreads(); // to ensures that all threads completed atomic op
    if(threadIdx.x == 0){
      op.atomic(&Out[seg], sh_max);
    }
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

template<
  typename Op=Max<DATA_TYPE>,
  typename T=DATA_TYPE,
  typename T4=DATA_TYPEx4>
__global__
void threadReduce(T * __restrict__ In, 
                                 int numElements, 
                                 uint* segStart,
                                 int numSegs,
                                 T * __restrict__ Out,
                                int bufferSize )   // will generate the sum of n_random numbers PER THREAD
{   
  if(blockIdx.x*blockDim.x >= numSegs) return; // no segments for this block to process
  Op op;
  extern __shared__ T4 buffer[];
  T4 *In4 = (T4 *) In;

  int lastSeg = (blockIdx.x+1)*blockDim.x;
  if(lastSeg > numSegs){
    lastSeg = numSegs;
  }

  int firstElementx4 = segStart[blockIdx.x*blockDim.x]/4; //floor
  int lastElementx4 = segStart[lastSeg]/4; //floor bc we will take care of the unaligned end later 

  int mysegStart = -1; 
  int mysegEndFloor = -1; //to avoid segfault
  int mysegEndCeil = -1; //to avoid segfault
  int unalignedStart = 0;
  int unalignedEnd = 0;
  int seg = threadIdx.x + blockIdx.x*blockDim.x;
  T myVal = op.identity();

  if(seg < numSegs){
    mysegStart = segStart[seg]/4;
    mysegEndFloor = segStart[seg+1]/4; 
    mysegEndCeil = (segStart[seg+1]+3)/4; //ceil
    
    unalignedStart = segStart[seg] % 4;
    unalignedEnd = segStart[seg+1] % 4;
  }


  for(int i=firstElementx4;i<lastElementx4;i+=bufferSize){

    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //   printf("i=%d, firstElementx4=%d, lastElementx4=%d, myVal=%d\n", i, firstElementx4, lastElementx4, myVal);
    // }

    //step 1: fill the buffer
    __syncthreads(); // wait for all threads to start to fill the buffer
    for(int j=threadIdx.x; j < bufferSize && j+i < lastElementx4; j+=blockDim.x){
      int idx = j+i; //index of In that thread will read
      buffer[j] = In4[idx];
    }
    __syncthreads(); // wait for all threads to finish to fill the buffer

    //step 2: check if the buffer contains elements of the thread's segment
    // and find out where each thread starts to read in the buffer
    int startIdx = mysegStart-i;
    if(startIdx >= bufferSize || mysegEndCeil-i <= 0)
     continue;
    int endIdx = mysegEndFloor-i;
    if(startIdx < 0)
      startIdx = 0;
    if(endIdx > bufferSize){
      endIdx = bufferSize;
    }
      
    //step 3: each thread reduce an element of its segment
    T4 element = buffer[startIdx];

    //the start of segment may not be aligned with the start of a T4 element
    if(unalignedStart){
      T* vals = (T*)&element;
      int tempEnd = 4;
      if(mysegStart == mysegEndCeil){
        tempEnd = unalignedEnd;
        unalignedEnd=0; //to avoid doing the unaligned end reduction twice in case the segment is smaller than 4 elements
      }
      for(int j=unalignedStart; j < tempEnd; j++){
        myVal = op(myVal, vals[j]);
      }
      startIdx++;
    }
    unalignedStart=0;

    //step 4: find out where each thread ends to read in the buffer
    for (int s = startIdx; s < endIdx; s++){
      element = buffer[s];
      myVal = op(myVal, element.x);
      myVal = op(myVal, element.y);
      myVal = op(myVal, element.z);
      myVal = op(myVal, element.w);
    }

    if(
      unalignedEnd != 0
      && endIdx < bufferSize 
    ){
      element = buffer[endIdx];
      myVal = op(myVal, element.x);
      if(unalignedEnd == 2){
        myVal = op(myVal, element.y);
      }    
      else if(unalignedEnd == 3){
        myVal = op(myVal, element.y);
        myVal = op(myVal, element.z);
      }
    }
  }
  if(seg < numSegs){
    if(seg == lastSeg-1){ 
      //last segment takes care of the remaining elements that are not read by the buffer
      for(int i=lastElementx4*4; i<segStart[seg+1]; i++)
        myVal = op(myVal, In[i]);
    }
    Out[seg] = myVal; //there is no atomic operation here, because each thread writes in its own segment
  }

    // if(threadIdx.x == 0)
    //   printf("bid=%d Out[%d]=%d\n",blockIdx.x,seg,Out[seg]);
}












template<
  typename Op=Max<DATA_TYPE>,
  typename T=DATA_TYPE>
__global__
void kernelReduce(DATA_TYPE * __restrict__ In, 
                                 int numElements, 
                                 uint* segStart,
                                 int nSegs,
                                 DATA_TYPE * __restrict__ Out )   // will generate the sum of n_random numbers PER THREAD

{   
  Op op;
  register int i, seg;
  register DATA_TYPE myVal;
  DATA_TYPEx4 *In4 = (DATA_TYPEx4 *) In;
  
  // if((uintptr_t)In % 16 != 0){
  //   if(threadIdx.x == 0 && blockIdx.x == 0){
  //     printf("Warning: input data is not aligned to 16 bytes, performance may be degraded\n");
  //     printf("In pointer mod 16 = %ld\n", ((uintptr_t)In) % 16);
  //   }
  //   return;
  // }

  __shared__ DATA_TYPE blockVal;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int blockStart = blockDim.x * blockIdx.x;
  for(seg = 0; seg < nSegs; seg++){
    if(segStart[seg]/4+blockStart > (segStart[seg+1]+3)/4)
      continue; //this block has no work for this segment
    int start = (segStart[seg]+3)/4+blockStart;
    int end = segStart[seg+1]/4;
    myVal = op.identity();

    int idxStart = segStart[seg]+tid;
    if(idxStart < start*4 && idxStart < segStart[seg+1]){ 
      myVal = op(myVal, In[idxStart]);
    }

    if( threadIdx.x == 0 )
      blockVal = op.identity();
    __syncthreads();

    for (i = start + threadIdx.x; //not tid bc start has already taken into account the block offset
        i < end; 
        i += (gridDim.x * blockDim.x) ) {

        DATA_TYPEx4 element = In4[i];

        myVal = op( element.x, myVal );
        myVal = op( element.y, myVal );         
        myVal = op( element.z, myVal );         
        myVal = op( element.w, myVal );      
          
    }

    // take care of the remaining elements that are not read by the loop above
    int idxEnd = end*4 + tid;
    if (idxEnd < segStart[seg+1] && idxEnd >= segStart[seg]) 
      myVal = op( In[idxEnd], myVal );

    op.atomic( &blockVal, myVal );
    __syncthreads();
    if( threadIdx.x == 0 )
        op.atomic( &Out[seg], blockVal );       
  } 
}

void getGpuSettings(int &nblocks, int &nthreads){
  if(nblocks != -1 && nthreads != -1){
    return;
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  if(nthreads < 0)
    nthreads = prop.maxThreadsPerBlock/2;
  nblocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / nthreads);
}

// WZ 2025
// entry for best kernel regular segmented reduce (FIXED size segments)
// This version will will run on the host 
// AND activate the proper kernel based on the UNIQUE segment size
// v2: incluido THRESHOLD1os manyBlocks2 e 16
template<
  typename Op=Max<DATA_TYPE>,
  typename T=DATA_TYPE>
void bestReduce(T * __restrict__ In, 
                                 int numElements, 
                                 uint* segStart,
                                 int numSegs,
                                 T * __restrict__ Out,
                                 unsigned int* threshold,
                                 int nthreads = -1,
                                 int nblocks = -1
                                )
{   
  if(nblocks < 0 || nthreads < 0) 
    getGpuSettings(nblocks, nthreads);

    int avarageSegSize = numElements / numSegs;
    if( avarageSegSize < threshold[0] ) {
        int nt = 512;
        int nb = (numSegs+nt-1) / nt;
        int bufferSize = nt*2;

        threadReduce<Op,T>
          <<<nb, nt, bufferSize*sizeof(DATA_TYPEx4)>>>
          (In, numElements, segStart, numSegs, Out, bufferSize);

    //antes do tunning era:
    // } else if( avarageSegSize < 500 )
    } else if( avarageSegSize < threshold[1] ) {
        warpReduce<Op,T>
          <<<nblocks, nthreads>>>( In, numElements, segStart, numSegs, Out);
    //antes do tunning era:
    // else if( avarageSegSize < 8000 ){ 
    } else if( avarageSegSize < threshold[2] ) {
    		// int maxthreads = nthreads*nblocks;
        uint averageSegSize = numElements/numSegs/4;
        int nt = nthreads;
        if( averageSegSize < 512*8 ) {
          if( averageSegSize < 256*8 ) {
            if( averageSegSize < 128*8 ) {
              if( averageSegSize < 64*8 ) {
                nt = 64;
              } else {
                nt = 128;
              }
            } else {
              nt = 256;
            }
          } else {
            nt = 512;
          }
        }
        // int nblocks = maxthreads/nt;

         blockReduce<Op,T>
          <<<numSegs, nt>>>( In, numElements, segStart, numSegs, Out
                        );
    //antes do tunning era:
    // } else if( avarageSegSize < 64000 ){ 
    } else if( avarageSegSize < threshold[3] ) {
        int grain =2;
        if(avarageSegSize < nthreads*grain){
          grain = ceil((float)avarageSegSize/nthreads);
          if(grain < 1) grain = 1;
          printf("Adjusting grain size to %d to better fit segment size\n", grain);
        }
				manyBlocksGr<Op,T>
          <<<numSegs*grain, nthreads>>>(
					  In, numElements, segStart, numSegs, Out, grain);
    //antes do tunning era:
    // } else if( avarageSegSize < 2500000 ){ 
    } else if( avarageSegSize < threshold[4] ) {
        int grain =16;
        if(avarageSegSize < nthreads*grain){
          grain = ceil((float)avarageSegSize/nthreads);
          if(grain < 1) grain = 1;
          printf("Adjusting grain size to %d to better fit segment size\n", grain);
        }
				manyBlocksGr<Op,T>
          <<<numSegs*grain, nthreads>>>(
					  In, numElements, segStart, numSegs, Out, grain);
    } else {
        kernelReduce<Op,T>
          <<<nblocks, nthreads>>>( In, numElements, segStart, numSegs, Out);
    }                         
}

#endif
