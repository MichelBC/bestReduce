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

/*
 * cub-uInt-device-reduceMax-test
 *  by WZ (2021)
 *
 * will provide 2 kernels to generate nElements random numbers in a GPU array of DATA_TYPE type
 *  - curand_kernel_uint32_manyT is a many threads version
 *  - curand_kernel_uint32_persT is a persistent threads version
 *
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>

// #include <helper_cuda.h>
#include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>

#include <string.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <iostream>
#include <fstream>

#include <string>
#include <algorithm>
#include <sstream>

#include "src/chrono.c"
#include "src/defines-bestreduce.h"
#include "src/getOpt.cpp"
#include "src/seg-reduce.cu"
#include "src/nonseg-reduce.cu"
#include "src/init-dist.cu"

// #include "thresholds.h"

struct bestReduce_tuning_s {
   const char * gpu_name;
//    unsigned int left[5];    // valores a esquerda no intervalo
//    unsigned int right[5];   // valores a direita no intervalo
   unsigned int threshold[5];
   unsigned int tuning_for;
};

std::vector<bestReduce_tuning_s> static_tuning = {
	#include "tuning.data"
    {"DEFAULT VALUES",{63,503,8124,65037,2562728},32000000}
};

void print_usage(char* exec_name){
	printf("Usage: TEST=\"function\" %s <nElements> [<nSegments>] [-d <dist_name>]\n", exec_name);
	printf("   function for non segmented can be: ui32, ui32_shfl, ui32x4, ui32_sh_mem, cub, thrust\n");
	printf("   function for segmented can be: segCub, segThrust, bestReduce\n");
	printf("   or you can test an individual kernel for segmented reduction: segThread, segWarp, segKernel, manyBlocks\n");

	printf("Note: the number of segments is optional for non-segmented reduction, but required for segmented reduction.\n");
	printf("   -d : distribution (can be regular, normal, or exp)\n");
	printf("\n");
	printf("If \"function\" == manyBlocks, the argument '-grain' can be used to set how many blocks per segment.\n");
}
/*
 * Host main program
 */
DATA_TYPE next_power_of_two(DATA_TYPE n) {
	if (n == 0)
		return 1;

	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n++;

	return n;
}

DATA_TYPE last_power_of_two(DATA_TYPE n) {
    if (n == 0)
        return 0;   // no power of two <= 0

    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;

    return n - (n >> 1);
}










#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

template <typename T>
void run_segmented_reduce_max_benchmark(int num_segments, size_t target_avg_size) {
    auto seg_lengths = generate_random_segment_lengths(num_segments, target_avg_size);
    size_t total_elements = std::accumulate(seg_lengths.begin(), seg_lengths.end(), 0LL);

    std::cout << "Running MAX reduction for " << sizeof(T)*8 << "-bit, segments: " << num_segments 
              << ", target_avg_size: " << target_avg_size 
              << ", actual_total_elements: " << total_elements 
              << ", actual_avg: " << (total_elements / static_cast<double>(num_segments)) << std::endl;

    T* d_in = nullptr;
    T* d_out = nullptr;
    uint* d_begin_offsets = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Allocate and fill input with random values in a reasonable range
    CHECK_CUDA(cudaMalloc(&d_in, total_elements * sizeof(T) * 2));

    std::vector<T> h_in(total_elements);
    std::mt19937 gen(42);  // fixed seed for reproducibility
    if constexpr (std::is_same_v<T, uint32_t>) {
        std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFu);  // avoid full range to reduce overflow chance in visualization
        for (auto& v : h_in) v = dist(gen);
    } else {
        std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFULL);
        for (auto& v : h_in) v = dist(gen);
    }
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), total_elements * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(&d_in[total_elements], h_in.data(), total_elements * sizeof(T), cudaMemcpyHostToDevice));

    // Allocate output
    CHECK_CUDA(cudaMalloc(&d_out, num_segments * sizeof(T)));

    // Prepare offsets
    std::vector<int> h_begin_offsets(num_segments + 1, 0);
    for (int i = 1; i <= num_segments; ++i) {
        h_begin_offsets[i] = h_begin_offsets[i-1] + seg_lengths[i-1];
    }
    h_begin_offsets[num_segments] = total_elements; // end offset
    CHECK_CUDA(cudaMalloc(&d_begin_offsets, (num_segments + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_begin_offsets, h_begin_offsets.data(), (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Query temp storage size for Max
    CHECK_CUDA(cub::DeviceSegmentedReduce::Max(
        nullptr, temp_storage_bytes,
        d_in, d_out,
        num_segments,
        d_begin_offsets, d_begin_offsets + 1));  // end_offsets = begin_offsets + 1
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Warm-up run
    CHECK_CUDA(cub::DeviceSegmentedReduce::Max(
        d_temp_storage, temp_storage_bytes,
        d_in, d_out,
        num_segments,
        d_begin_offsets, d_begin_offsets + 1));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    const int num_iterations = 100;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < num_iterations; ++iter) {
        CHECK_CUDA(cub::DeviceSegmentedReduce::Max(
            d_temp_storage, temp_storage_bytes,
            &d_in[iter % 2 * total_elements],
            d_out,
            num_segments,
            d_begin_offsets, d_begin_offsets + 1));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double time_sec = ms / 1000.0 / num_iterations;
    double geps = static_cast<double>(total_elements) / time_sec / 1e9;

    std::cout << "Throughput (max): " << geps << " GigaElements/s" << std::endl;

    //print 
    // last output for verification
    T h_out;
    CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost));
    std::cout << "Sample output (segment 0 max): " << h_out << std::endl;

    //write in a file d_begin_offsets to host for verification
    // std::vector<int> h_begin_offsets_check(num_segments + 1);
    // CHECK_CUDA(cudaMemcpy(h_begin_offsets_check.data(), d_begin_offsets, (num_segments + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    // std::ofstream f("begin_offsets.txt");
    // for (int i = 0; i < num_segments + 1; ++i) {
    //     f << h_begin_offsets_check[i] << "\n";
    // }
    // f.close();

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_begin_offsets));
    CHECK_CUDA(cudaFree(d_temp_storage));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

}








int
main(int argc, char *argv[]){
	cudaError_t err = cudaSuccess;
		
	for( int i=0; i<argc; i++ ) {
		printf( "argv[%d] is %s\n", i, argv[i] );
	}
		

	if( argc < 2 ) {
		print_usage(argv[0]);
		exit( 0 ); 
	}

	// int usePersistentKernel = 0, useManyThreadsKernel = 1;

	printf( "Argc = %d ", argc );

	char *testName;
	if (( testName = getenv( "TEST" )) != NULL )
		printf( "TEST is [%s]\n", testName );
	else{ 
		//ERROR
		printf("ERROR: TEST environment variable not set\n");
		print_usage(argv[0]);
		exit(EXIT_FAILURE);
	}					

	long nElements = atol(argv[1]);
	printf( "Number of elements requested in command line: %ld \n", nElements );

	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0 );
	printf( "cudaDeviceName: %s\n", prop.name );
	printf( "l2CacheSize: %d\n", prop.l2CacheSize );
	printf( "persistingL2CacheMaxSize: %d\n", prop.persistingL2CacheMaxSize );



    int nthreads = prop.maxThreadsPerMultiProcessor / 2;
    if(nthreads > prop.maxThreadsPerBlock)
        nthreads = prop.maxThreadsPerBlock;
    int nblocks = prop.multiProcessorCount*
        (prop.maxThreadsPerMultiProcessor/nthreads);

    printf("=== Device Configuration ===\n");
    printf("nblocks: %d\n", nblocks);
    printf("nthreads: %d\n", nthreads);
    printf("============================\n");






	int use_gnuplot = 0;
	if(cmdOptionExists(argv, argc+argv, "-gnuplot")){
		use_gnuplot = 1;
		printf("Using gnuplot to visualize exp distribution\n");
	} else {
		printf("Not using gnuplot to visualize exp distribution\n");
	}

		
	std::string distname;
	if(cmdOptionExists(argv, argc+argv, "-d")){
		distname = getStrFromCmdOption(argv, argv+argc, "-d");
	} else {
		distname = "regular"; // default distribution
		printf("Using default distribution: regular\n");
	}


	uint* d_segStart;
	uint nSegments = 1;
	int segreduce = 0;
	if(argc > 2){
		nSegments = atoi(argv[2]);
		segreduce = 1;
		err = cudaMalloc(&d_segStart, (nSegments+1)*sizeof(uint));
		if (err != cudaSuccess){
				fprintf(stderr, "Failed to allocate device vector d_segStart (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
		}

		
		if(distname == "regular"){
			printf("Using regular distribution for segment sizes\n");
			printf("Segment size: %d\n", (int)(nElements / nSegments));
			if(nElements % nSegments != 0){
				printf("Last segment will have size %d\n", (int)nElements % nSegments);
			}
			initSegStarts<<<nblocks, nthreads>>>(d_segStart,nSegments,nElements);
		} else if(distname == "normal"){
			printf("Using normal distribution for segment sizes\n");
			double mean = nElements/nSegments;
			double stddev = mean/2;
			segStartInitNormalDist(d_segStart,nSegments,nElements, mean, stddev);
		} else if(distname == "exp"){
			printf("Using exp distribution for segment sizes\n");
			double lambda = 4.0;
			if(cmdOptionExists(argv, argc+argv, "-lambda")){
				lambda = atof(getStrFromCmdOption(argv, argv+argc, "-lambda").c_str());
				printf("Using lambda = %f for exp distribution\n", lambda);
			} else {
				printf("Using default lambda = 4.0 for exp distribution\n");
			}
			initExpDist(d_segStart,nSegments,nElements, lambda, use_gnuplot);
		} else if(distname == "normalv1"){
				// void generate_normal_random_segments(uint32_t* d_segStart, int num_segments, size_t target_avg_size, int& nElements){
				size_t target_avg_size = nElements/nSegments;
				generate_normal_random_segments(d_segStart,(int)nSegments, target_avg_size, nElements);
		} else {
			printf("ERROR: Invalid distribution name '%s'\n", distname.c_str());
			exit(EXIT_FAILURE);
		}

	}

	size_t size = nElements * sizeof(DATA_TYPE);
	printf("[Initialize %ld numbers, input vector size %ld Bytes]\n", 
	        nElements, size);
	// Declare, allocate, and initialize device-accessible 
	//                       pointers for input and output
	DATA_TYPE  *d_in_dtype;
	// DATA_TYPE  *d_in_dtype_not_aligned;
	DATA_TYPE  *d_out_dtype;
	int workingSetSize = nElements*sizeof(DATA_TYPE);
	int numBuffers;

	int ntimes = NTIMES;

	/////////////////////////////
	//DATA TYPE VARIABLE
	//TODO: ADJUST ALL KERNELS TO USE DATA_TYPE



	// test if working set will be larger that 2*l2CacheSize
	int nItemsToAllocate;
	int alignedSize = ((nElements+3)/4)*4; // align to 16 bytes
	if( workingSetSize > 2*prop.l2CacheSize ) {

		nItemsToAllocate = alignedSize;
		numBuffers = 1;
				// Allocate device vectors
		err = cudaMalloc(&d_in_dtype, alignedSize*sizeof(DATA_TYPE));
		if (err != cudaSuccess)
		{
				fprintf(stderr, "Failed to allocate device vector d_in_dtype (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
		}

	} else {  // wss = 200 l2s= 100 i,e. wss = 2*l2s => nb = ceil( 200/200.0 ) => 1 ok
					// wss = 100 l2s= 200 i,e. wss < 2*l2s => nb = ceil( 400/100.0 ) => 4 ok
					// wss = 100 l2s= 70 i,e. wss < 2*l2s => nb = ceil( 140 /100.0 ) => 2 ok
		numBuffers = ceil( (double)(2*prop.l2CacheSize) / workingSetSize );
		nItemsToAllocate = numBuffers*alignedSize;
				// Allocate device vectors
		err = cudaMalloc(&d_in_dtype, nItemsToAllocate*sizeof(DATA_TYPE));
		if (err != cudaSuccess)
		{
				fprintf(stderr, ".Failed to allocate device vector d_in_dtype (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
		}
	}

	// align to 16 bytes
	// d_in_dtype = (DATA_TYPE*)(
	// 	((uintptr_t)d_in_dtype_not_aligned + 15u) & ~(uintptr_t)0x0F
	// );

	thrust::device_vector<DATA_TYPE> d_in2(nItemsToAllocate);    
	thrust::device_vector<DATA_TYPE> d_out2(1);

	chronometer_t c1;
	chrono_reset(&c1);
				
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop); 
	//int nEvents = 2;   
					
/////////////////////////////////////////////

	err = cudaMalloc(&d_out_dtype, (nSegments+1)*sizeof(DATA_TYPE));
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector d_out_dtype (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	cudaMemset(d_out_dtype, 0, (nSegments+1)*sizeof(DATA_TYPE));
	if(cmdOptionExists(argv, argc+argv, "-initSeq"))
	initializeSequential<<<nblocks, nthreads>>>( d_in_dtype, alignedSize, numBuffers );
	else if(cmdOptionExists(argv, argc+argv, "-initRevSeq"))
	initializeInputVector<<<nblocks, nthreads>>>( d_in_dtype, alignedSize, numBuffers );
	else if(distname == "-initRandom"){
		std::vector<DATA_TYPE> h_in(alignedSize);
		std::mt19937 gen(42);  // fixed seed for reproducibility
		if constexpr (std::is_same_v<DATA_TYPE, uint32_t>) {
			std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFu);  // avoid full range to reduce overflow chance in visualization
			for (auto& v : h_in) v = dist(gen);
		} else {
			std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFULL);
			for (auto& v : h_in) v = dist(gen);
		}
		cudaMemcpy(d_in_dtype, h_in.data(), alignedSize * numBuffers * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
		}
	else
	curand_kernel_uint32_persT<<<nblocks, nthreads>>>( d_in_dtype, alignedSize*numBuffers,1000000000, 0 );

	initializeOutputVariable<DATA_TYPE><<<1, 1>>>( d_out_dtype, 0 );
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to initialize input vector (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	const char* funcname;

	printf("\n-------------------------------------------------------------");
	printf("\n---------- Launch ---------- Kernels ------------------------");

	printf( "\nCUDA persistent kernel for *vector inicialization* " );
	printf( "\n    launch with %d blocks of %d threads", nblocks, nthreads);
	printf( "\n    Total of threads in kernel is %ld ", (long)nblocks*nthreads);
	printf("\nGPU: %s Persistent kernel: nblocks= %d nthreads= %d\n", 
					prop.name, nblocks, nthreads );

	if(!segreduce){
		/////////////////////////////////////////////////////
		if( strcmp( testName, "ui32" ) == 0 || strcmp( testName, "ui32_atomic_op" ) == 0 ) {
			funcname = "uint32_reduceMax";
			printf("Lauching: uint32_reduceMax kernel\n" );
			chrono_start(&c1);
				for (int i = 0; i < ntimes; i++) {
					DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];
					uint32_reduceMax<<<nblocks, nthreads>>>(d_inBuff, nElements, d_out_dtype);
				}
				cudaDeviceSynchronize();
			chrono_stop(&c1);

		/////////////////////////////////////////////////////
		} else if( strcmp( testName, "ui32_shfl" ) == 0  ) {
			funcname = "ui32_shfl_reduceMax";
			printf("Lauching: ui32_shfl_reduceMax kernel\n");
			chrono_start(&c1);
				for (int i = 0; i < ntimes; i++) {
					DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];
					ui32_shfl_reduceMax<<<nblocks, nthreads>>>(d_inBuff, nElements, d_out_dtype);
				}
				cudaDeviceSynchronize();
			chrono_stop(&c1);

		/////////////////////////////////////////////
		/////////////////////////////////////////////////////
		} else if( strcmp( testName, "ui32x4" ) == 0 ) {
			funcname = "uint32x4_reduceMax";
			printf("Lauching: uint32x4_reduceMax kernel\n");
			chrono_start(&c1);
				for (int i = 0; i < ntimes; i++) {
					DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];
					uint32x4_reduceMax<<<nblocks, nthreads>>>(d_inBuff, nElements, d_out_dtype);
				}
				cudaDeviceSynchronize();
			chrono_stop(&c1);
		/////////////////////////////////////////////////////
		} else if( strcmp( testName, "ui32_sh_mem" ) == 0 ) {
			funcname = "ui32_sh_mem";
			printf("Lauching: ui32_sh_mem kernel\n");

			chrono_start(&c1);
				for (int i = 0; i < ntimes; i++) {

					DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];
					ui32_sh_mem_reduceMax<<<nblocks, nthreads,nthreads*sizeof(DATA_TYPE)>>>
					(d_inBuff, nElements, d_out_dtype
					);
				}
				cudaDeviceSynchronize();
			chrono_stop(&c1);

		/////////////////////////////////////////////
		} else if( strcmp( testName, "cub" ) == 0  ) {
			funcname = "cub::DeviceReduce::Max";
			printf("Lauching: cub::DeviceReduce::Max kernel\n");

			// Determine temporary device storage requirements
			void     *d_temp_storage = NULL;
			size_t   temp_storage_bytes = 0;
			cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in_dtype, d_out_dtype, nElements);
			// Allocate temporary storage
			// printf("bytes required: %lu\n", temp_storage_bytes);
			err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
			if (err != cudaSuccess){
					fprintf(stderr, "Failed to allocate device vector d_temp_storage (error code %s)!\n", cudaGetErrorString(err));
					exit(EXIT_FAILURE);
			}
			
			cudaDeviceSynchronize();
			chrono_start(&c1);
				for (int i = 0; i < ntimes; i++) {

					DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];
					cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_inBuff, d_out_dtype, nElements);
				}
				cudaDeviceSynchronize();
			chrono_stop(&c1);

		/////////////////////////////////////////////////////
		} else if( strcmp( testName, "thrust" ) == 0  ) {
			funcname = "thrust::reduce with thrust::maximum<DATA_TYPE>()";
			printf("Lauching: thrust::reduce with thrust::maximum<DATA_TYPE>() kernel\n");

			initializeInputVector<<<nblocks, nthreads>>>( thrust::raw_pointer_cast(d_in2.data()) , nElements, numBuffers );
			initializeOutputVariable<DATA_TYPE><<<1, 1>>>( thrust::raw_pointer_cast(d_out2.data()), 0 );

			cudaDeviceSynchronize();
			chrono_start(&c1);
				for (int i = 0; i < ntimes; i++) {

					thrust::device_ptr<DATA_TYPE> d_inBuff2 = d_in2.data() + (alignedSize*(i%numBuffers)); 
					//thrust::device_ptr<DATA_TYPE> dev_ptr = thrust::device_pointer_cast(raw_ptr);
					auto reduced_thrust = thrust::reduce(d_inBuff2,
														d_inBuff2 + nElements,
														0,
														thrust::maximum<DATA_TYPE>());
				}
				cudaDeviceSynchronize();
			chrono_stop(&c1);
		}
	} 
	else{

		/////////////////////////////////////////////////////
		/////////////////////////////////////////////////////
		/////////////////////////////////////////////////////
		//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		// INIT SEG REDUCE
		// INIT SEG REDUCE
		// INIT SEG REDUCE
		// INIT SEG REDUCE
		// INIT SEG REDUCE
		// INIT SEG REDUCE

	cudaMemset(d_out_dtype,0,nSegments*sizeof(uint));
		// initSegStartsUniformRandomSize(d_segStart,nSegments,nElements, d_out_dtype);

		cudaDeviceSynchronize();
		err = cudaGetLastError();
		if (err != cudaSuccess){
				fprintf(stderr, "Failed to initialize segment starts (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
		}


		/////////////////////////////////////////////////////
		/////////////////////////////////////////////////////
		/////////////////////////////////////////////////////
		/////////////////////////////////////////////////////
		/////////////////////////////////////////////////////

	/////////////////////////////////////////////////////
	/////////////////////////////////////////////////////
	if( strcmp( testName, "bestReduce" ) == 0 ) {
		funcname = "bestRegReduce2";
		printf("Lauching: bestRegReduce2\n");
		// int threshold[5] = {THRESHOLD1, THRESHOLD2, THRESHOLD3, THRESHOLD4, THRESHOLD5};

		// TUNING: GETTING BEST PARAMETERS FOR THIS EXECUTION
		unsigned int *threshold = nullptr;
		int minDiff = std::numeric_limits<int>::max();
		for (auto& gpu : static_tuning) {
			if (std::string(gpu.gpu_name) == std::string(prop.name)) {
				int diff = std::abs(gpu.tuning_for - nElements);

				if (diff <= minDiff) {
					minDiff = diff;
					threshold = gpu.threshold;
				}
			}
		}
		if(!threshold){
			threshold = static_tuning.back().threshold;
			printf("No specific tuning found for this GPU, using default values\n");
			printf("Using thresholds: %u, %u, %u, %u, %u\n", threshold[0], threshold[1], threshold[2], threshold[3], threshold[4]);
			printf("Note: these thresholds may not be optimal for this GPU, use tuning-bestReduce to find better thresholds.\n");
		} else {
			printf("Using thresholds: %u, %u, %u, %u, %u\n", threshold[0], threshold[1], threshold[2], threshold[3], threshold[4]);
			printf("Tuned for GPU: %s, using nElements = %ld\n", prop.name, nElements);
		}
		// END

		chrono_start(&c1);
			for (int i = 0; i < ntimes; i++) {
				DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];
				bestReduce(d_inBuff, nElements, d_segStart, nSegments, d_out_dtype, threshold);
			}
			cudaDeviceSynchronize();
		chrono_stop(&c1);

	/////////////////////////////////////////////////////
	} else if( strcmp( testName, "segThread" ) == 0 ) {
		funcname = "threadReduce";
		printf("Lauching: threadReduce kernel\n");

		int averageSegSize = ceil((float)nElements/nSegments);
		if(averageSegSize > 32000){
			printf("WARNING: Average segment size is very large (%d). Execution may take a long time. To reduce runtime, consider decreasing the size of segments or using another method.\n", averageSegSize);
		}


		int nb = (nSegments+nthreads-1)/nthreads;
		int bufferSize = nthreads*2;
			chrono_start(&c1);
				for (int i = 0; i < ntimes; i++) {
					DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];
					threadReduce<<<nb, nthreads, bufferSize*sizeof(DATA_TYPEx4)>>>
					(d_inBuff, nElements, d_segStart, nSegments, d_out_dtype, bufferSize
					);
				}
				cudaDeviceSynchronize();
			chrono_stop(&c1);
	/////////////////////////////////////////////////////
	} else if( strcmp( testName, "segWarp" ) == 0 ) {
		funcname = "warpSegReduce";
		printf("Lauching: warpSegReduce kernel\n");

				chrono_start(&c1);
					for (int i = 0; i < ntimes; i++) {
						DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];
						warpReduce<<<nblocks, nthreads>>>(d_inBuff, nElements, d_segStart, nSegments, d_out_dtype
						);
					}
					cudaDeviceSynchronize();
				chrono_stop(&c1);

	/////////////////////////////////////////////////////
	/////////////////////////////////////////////////////
	/////////////////////////////////////////////////////
	} else if( strcmp( testName, "manyBlocks" ) == 0 ) {
		funcname = "manyBlocks";
		int nt = nthreads;
		uint grain = 16;



		if(nSegments > 4000000){
			printf("WARNING: Too many segments. Execution may take a long time. To reduce runtime, consider lowering the number of segments or using another method.\n");
			printf("%s Throughput: %lf uint/ns (or giga elements/s)\n", 
				funcname, 0.0);
			return 0;
		}


		if (cmdOptionExists(argv, argc+argv, "-grain")) {
			grain = getIntFromCmdOption(argv, argv+argc, "-grain");
			printf("Using grain size: %d\n", grain);
		} else {
			printf("Using default grain size: %d\n", grain);
		}

		chrono_start(&c1);
			for (int i = 0; i < ntimes; i++) {
				DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];
				manyBlocksGr<<<nSegments*grain, nt>>>(
					d_inBuff, nElements, d_segStart, nSegments, d_out_dtype, grain);
			}
			cudaDeviceSynchronize();
		chrono_stop(&c1);
	/////////////////////////////////////////////////////
	} else if( strcmp( testName, "segKernel" ) == 0 ) {
		funcname = "kernelReduce";
		printf("Lauching: kernelReduce kernel\n");

		int nt = nthreads;
		int nb = nblocks;
		// int averageSegSize = ceil((float)nElements/nSegments);
		// if(averageSegSize <= last_power_of_two(nt)){
		// 	nt = next_power_of_two(averageSegSize);
		// 	if( nt < 128 ) nt = 128; // at least one warp
		// 	nb = prop.multiProcessorCount*
		// 		(prop.maxThreadsPerMultiProcessor/nt);
		// 	printf("Adjusting number of threads to %d and blocks to %d to better fit segment size\n", nt, nb);
		// }

		if(nSegments > 128000){
			printf("WARNING: Too many segments. Execution may take a long time. To reduce runtime, consider lowering the number of segments or using another method\n");
		}

		chrono_start(&c1);
			for (int i = 0; i < ntimes; i++) {
				DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];

				kernelReduce<<<nb, nt>>>(d_inBuff, nElements, 
										d_segStart, nSegments, d_out_dtype
										);
				err = cudaGetLastError();

				if (err != cudaSuccess)
				{
						fprintf(stderr, "Failed to launch kernelReduce kernel (error code %s)!\n", 
															cudaGetErrorString(err));
						exit(EXIT_FAILURE);
				} 
				
			}
			cudaDeviceSynchronize();
		chrono_stop(&c1);

	/////////////////////////////////////////////////////
	} else if( strcmp( testName, "segBlock" ) == 0 ) {
		funcname = "blockReduce";
		printf("Lauching: blockReduce kernel\n");

        uint averageSegSize = nElements/nSegments/4;
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

		chrono_start(&c1);
			for (int i = 0; i < ntimes; i++) {
				DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];

				blockReduce<<<nSegments, nt>>>(d_inBuff, nElements, 
										d_segStart, nSegments, d_out_dtype
				);			
			}
			cudaDeviceSynchronize();
		chrono_stop(&c1);
		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
				fprintf(stderr, "Failed to launch blockReduce kernel (error code %s)!\n", 
													cudaGetErrorString(err));
				exit(EXIT_FAILURE);
		} 
									
	/////////////////////////////////////////////////////
	/////////////////////////////////////////////////////
	} else if( strcmp( testName, "segCub" ) == 0 ) {
		funcname = "cub::DeviceSegmentedReduce::Max";
		printf("Lauching: cub::DeviceSegmentedReduce::Max kernel\n");
			
		// Determine temporary device storage requirements
		void     *d_temp_storage = nullptr;
		size_t   temp_storage_bytes = 0;

		cub::DeviceSegmentedReduce::Max(
											d_temp_storage, temp_storage_bytes, 
											d_in_dtype, d_out_dtype, 
											nSegments,
											d_segStart,
											d_segStart+1);
		// Allocate temporary storage
		printf("bytes required: %lu\n", temp_storage_bytes);
		err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
		if (err != cudaSuccess){
			fprintf(stderr, "Failed to allocate device vector d_temp_storage (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// Warm-up run
		DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(numBuffers-1) ];
		cub::DeviceSegmentedReduce::Max(
			d_temp_storage, temp_storage_bytes,
			d_inBuff, d_out_dtype,
			nSegments,
			d_segStart, d_segStart + 1);
		cudaDeviceSynchronize();

		// DATA_TYPE *d_inBuff = d_in_dtype;
		cudaEvent_t start, stop;

    	cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		chrono_start(&c1);
			for (int i = 0; i < ntimes; i++) {

				// DATA_TYPE *d_inBuff = &d_in_dtype[ alignedSize*(i%numBuffers) ];
				err = cub::DeviceSegmentedReduce::Max(
								d_temp_storage, temp_storage_bytes, 
								d_inBuff, d_out_dtype, 
								nSegments,
								d_segStart,
								d_segStart + 1);
			}
			cudaDeviceSynchronize();
		chrono_stop(&c1);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		double time_sec = milliseconds / 1000.0 / ntimes;
		// printf("Time measured with cuda events: %f ms\n", milliseconds);
		printf("cuda event: %lf giga elements/s\n", 
		// ((double)nElements*ntimes) / (milliseconds * 1e6));
		((double)nElements) / (time_sec * 1e9));
	
	/////////////////////////////////////////////
	/////////////////////////////////////////////////////
	} else if( strcmp( testName, "segThrust" ) == 0 ) {
		funcname = "thrust::reduce_by_key with thrust::maximum<DATA_TYPE>()";
		printf("Lauching: thrust::reduce_by_key with thrust::maximum<DATA_TYPE>() kernel\n");

		// Create a thrust::device_ptr from the raw device pointer
		// thrust::device_ptr<int> d_in_thrust_ptr(d_in_dtype);

		// Create a thrust::device_vector from the thrust::device_ptr
		thrust::device_vector<DATA_TYPE> d_in_thrust_vector(d_in_dtype, d_in_dtype + alignedSize);
		thrust::device_vector<DATA_TYPE> d_out_thrust_vector(nElements);

		// Criar iteradores transformados para as chaves
		// auto start_iter = thrust::make_transform_iterator(
		//     thrust::counting_iterator<uint>(0),
		//     [&] __device__ (uint x) { return x / nSegments; }  // Função lambda para transformar o índice
		// );
		// auto end_iter = thrust::make_transform_iterator(
		//     thrust::counting_iterator<uint>(nElements * nSegments),
		//     [&] (uint x) { return x / nSegments; }  // Função lambda para transformar o índice
		// );

		chrono_start(&c1);
			for (int i = 0; i < ntimes; i++) {
				//  DATA_TYPE *d_inBuff = &d_in_dtype[ nElements*(i%numBuffers) ];

				// reduce_by_key(InputIterator1 keys_first, 
				//               InputIterator1 keys_last,
				//               InputIterator2 values_first,
				//               OutputIterator1 keys_output,
				//               OutputIterator2 values_output,
				//               BinaryPredicate binary_pred,
				//               BinaryFunction binary_op)

				thrust::device_vector<DATA_TYPE> d_vector(nElements);
				thrust::sequence(d_vector.begin(), d_vector.end());
				// thrust::fill(d_vector.begin(), d_vector.end(), 0);
				// Initialize the device_vector with segmented values
				// thrust::transform(
				//     thrust::counting_iterator<int>(0), // Start iterator
				//     thrust::counting_iterator<int>(nElements), // End iterator
				//     d_vector.begin(),                   // Output iterator
				//     [&] (int x) { 
				//         return x / nSegments; 
				//     }  // Lambda function to determine the value based on the segment
				// );

				thrust::reduce_by_key(
						thrust::device,
						// thrust::counting_iterator<uint>(0),
						// thrust::counting_iterator<uint>(nElements),
						d_vector.begin(),
						d_vector.end(),
						d_in_thrust_vector.begin(),
						thrust::discard_iterator<DATA_TYPE>(),
						d_out_thrust_vector.begin(),
						thrust::equal_to<DATA_TYPE>(),
						thrust::maximum<DATA_TYPE>()
				);
				
			}
			cudaDeviceSynchronize();
		chrono_stop(&c1);

	/////////////////////////////////////////////////////
	} else {
		printf("ERROR: Invalid test name '%s'\n", testName);
		exit(EXIT_FAILURE);
	}
	err = cudaGetLastError();
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to launch %s kernel (error code %s)!\n", 
				funcname, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	// #define DEBUG 1
	#if DEBUG 
		printf( "Output vector on device after reduce\n" );
		printReduceResult<<<1, 1>>>( d_out_dtype, 0 );
		printReduceResult<<<1, 1>>>( d_out_dtype, 1 );
		if(nSegments > 1){
			printReduceResult<<<1, 1>>>( d_out_dtype, 2 );
			printReduceResult<<<1, 1>>>( d_out_dtype, nSegments-1 );
		}
		cudaDeviceSynchronize();

		printf( "input vector\n" );
		printReduceResult<<<1, 1>>>( d_in_dtype, 0 );
		printReduceResult<<<1, 1>>>( d_in_dtype, 1 );
		if(nSegments > 1){
			printReduceResult<<<1, 1>>>( d_in_dtype, 2 );
			printReduceResult<<<1, 1>>>( d_in_dtype, nSegments-1 );
		}
		cudaDeviceSynchronize();

		printf( "seg vector\n" );
		printReduceResult<<<1, 1>>>( d_segStart, 0 );
		printReduceResult<<<1, 1>>>( d_segStart, 1 );
		if(nSegments > 1){
			printReduceResult<<<1, 1>>>( d_segStart, 2 );
			printReduceResult<<<1, 1>>>( d_segStart, nSegments-1 );
		}
		cudaDeviceSynchronize();

		// printf( "Output vector on device after reduce (Vector has only position[0])\n" );
		// printReduceResult<<<1, 1>>>( d_out_dtype, 0 );
		// cudaDeviceSynchronize();

	#endif

	}
		// printf( "Output vector on device after reduce (Vector has only position[0])\n" );
		// printReduceResult<<<1, 1>>>( d_out_dtype, 0 );
		// cudaDeviceSynchronize();

	err = cudaGetLastError();
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to to print results (error code %s)!\n", 
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("\nGPU: %s %s kernel\n", 
					prop.name, funcname );
	chrono_report_TimeInLoop(&c1, "CUDA kernel launch", ntimes);
	printf("%s Throughput: %lf uint/ns (or giga elements/s)\n", 
						funcname, ((double)nElements*ntimes)/((double)chrono_gettotal(&c1)));
	printf("Global Memory Throughput: %lf GiB/s (Giga Bytes/s)\n", 
						((double)sizeof(DATA_TYPE)*nElements*ntimes) / ((double)chrono_gettotal(&c1)) );
		
/////////////////////////////////////////
///////////////////////// DEALLOC MEMORY
		err = cudaFree(d_in_dtype);
		if (err != cudaSuccess)
		{
				fprintf(stderr, "Failed to free device vector d_in_dtype (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
		}
		err = cudaFree(d_out_dtype);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to free device vector d_out_dtype (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		if(segreduce)
		{
			err = cudaFree(d_segStart);
			if (err != cudaSuccess)
			{
				fprintf(stderr, "Failed to free device vector d_segStart (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		}
		

		printf("Done\n");

















	// std::vector<std::pair<int, size_t>> configs = {
    //     {2,   16ULL * 1024 * 1024},
    //     {4,    8ULL * 1024 * 1024},
    //     {8,    4ULL * 1024 * 1024},
    //     {16,   2ULL * 1024 * 1024},
    //     {32,   1ULL * 1024 * 1024},
    //     {64, 512ULL * 1024}
    //     // { 16ULL * 1024 * 1024, 2},
    //     // {  8ULL * 1024 * 1024, 4},
    //     // {  4ULL * 1024 * 1024, 8},
    //     // {  2ULL * 1024 * 1024, 16},
    //     // {  1ULL * 1024 * 1024, 32},
    //     // {512ULL * 1024, 64}
    // };

    // std::cout << "\n=== Benchmarking uint32_t  (MAX reduction) ===\n";
    // for (const auto& cfg : configs) {
    //     run_segmented_reduce_max_benchmark<uint32_t>(cfg.first, cfg.second);
    //     std::cout << std::string(70, '-') << "\n";
    // }

    // std::cout << "\n=== Benchmarking uint64_t  (MAX reduction) ===\n";
    // for (const auto& cfg : configs) {
    //     run_segmented_reduce_max_benchmark<uint64_t>(cfg.first, cfg.second);
    //     std::cout << std::string(70, '-') << "\n";
    // }























		return 0;
}

