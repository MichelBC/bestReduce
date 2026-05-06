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

#ifndef DEFINES_BEST_REDUCE
#define DEFINES_BEST_REDUCE

//===========================
// BESTREDUCE DEFINES
//===========================
	// #define DEBUG 
	#define WARP_SIZE 32
    // #define RTX3050
	#define RTXA4500

	//DEFINES DATATYPE
	#ifndef UINT64
		#define UINT32
	#endif
	#ifdef UINT32
		#define DATA_TYPE uint32_t
		#define DATA_TYPEx2 uint2
		#define DATA_TYPEx4 uint4
		#define ATOMIC_OP(addr,val) atomicMax(addr,val)
	// #elif defined(UINT64)
	#else
		#define DATA_TYPE unsigned long long
		#define DATA_TYPEx2 ulonglong2
		#define DATA_TYPEx4 ulonglong4
		#define ATOMIC_OP(addr,val) atomicMax(addr,val)
	#endif

	//DEFINES TAMANHO BLOCO E WARP
	#define N_BLOCKS ((THREADS_PER_MP/N_THREADS)*MP)
	#define N_WARPS (N_THREADS / WARP_SIZE)
	#define TOTAL_THREADS (THREADS_PER_MP * MP)
//END

#ifdef RTXA4500
	#define GPU_NAME "RTXA4500"
	#define MP 56
	#define THREADS_PER_BLOCK 768   // can be defined in the compilation line with -D
	#define THREADS_PER_MP 1536
#elif defined(RTX3050)
 #define GPU_NAME "RTX3050-------------"
 #define MP 16
 #define THREADS_PER_BLOCK 768   // can be defined in the compilation line with -D
 #define THREADS_PER_MP 1536
#elif defined(RTX3060)
 #define GPU_NAME "RTX3060-------------"
 #define MP 28
 #define THREADS_PER_BLOCK 768   // can be defined in the compilation line with -D
 #define THREADS_PER_MP 1536
#elif defined(RTX2070)
	#define GPU_NAME "RTX2070"
	#define MP 36
	#define THREADS_PER_BLOCK 1024   // can be defined in the compilation line with -D
	#define THREADS_PER_MP 2048
#elif defined(GTX1050)
	#define GPU_NAME "GTX1050"
	#define MP 3
	#define THREADS_PER_BLOCK 1024   // can be defined in the compilation line with -D
//  #define THREADS_PER_BLOCK 512
	#define THREADS_PER_MP 2048
#elif defined(GTX1080ti)
	#define GPU_NAME "GTX1080ti"
	#define MP 28
	#define THREADS_PER_BLOCK 1024   // can be defined in the compilation line with -D
	#define THREADS_PER_MP 2048
#else
	#define GPU_NAME "??"
	#define MP 5
	#define THREADS_PER_BLOCK 1024   // can be defined in the compilation line with -D
	#define THREADS_PER_MP 2048
#endif
#define N_THREADS THREADS_PER_BLOCK



#define RESIDENT_BLOCKS_PER_MP THREADS_PER_MP/THREADS_PER_BLOCK
#define NTA \
	(MP * RESIDENT_BLOCKS_PER_MP * THREADS_PER_BLOCK)

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

#define SEED 75439

// #define NTIMES 30
#define NTIMES 100

#endif