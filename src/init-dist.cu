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

#ifndef INITDISTCU
#define INITDISTCU

#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <stdlib.h>
#include <curand_kernel.h>

#include "defines-bestreduce.h"
#include "exp-histo.c"

// curand_kernel_uint32_manyT  
//   a manyThreads kernel
//   will generate numElements random numbers in a GPU array of uint32_t type
template<
  typename T=DATA_TYPE>
__global__
void curand_kernel_uint32_manyT(T *Out, long numElements, int seed )   
{
		curandState_t state; 
		curand_init((seed+blockDim.x)+seed+threadIdx.x, // the seed controls the sequence of random values that are produced
						blockIdx.x,  // the sequence number is only important with multiple cores 
						threadIdx.x,  // the offset is how much extra we advance in the sequence for each call, can be 0 
						&state);
		
		if( ((long)blockDim.x * blockIdx.x + threadIdx.x) < numElements )
					Out[ ((long)blockDim.x * blockIdx.x + threadIdx.x) ] = curand(&state);
}


// curand_kernel_uint32_persT  
//   a persistent kernel
//   will generate numElements random numbers in a GPU array of uint32_t type
template<
  typename T=DATA_TYPE>
__global__
void curand_kernel_uint32_persT(T *Out, long numElements, long maxValue, int seed )   // will generate the sum of n_random numbers PER THREAD
{
		curandState_t state; 
		curand_init((seed+blockDim.x)+seed+threadIdx.x, // the seed controls the sequence of random values that are produced
						blockIdx.x,  // the sequence number is only important with multiple cores 
						threadIdx.x,  // the offset is how much extra we advance in the sequence for each call, can be 0 
						&state);
		
		for (long i = ((long)blockDim.x * blockIdx.x + threadIdx.x); 
							i < numElements; 
							i += (long)gridDim.x * blockDim.x) {

				Out[i] = curand(&state) % maxValue;
		}
}

template <typename T=DATA_TYPE>
__global__
void initializeInputVector(T *In, long numElements, int numBuffs) {
	for (long i = ((long)blockDim.x * blockIdx.x + threadIdx.x); 
						i < numElements; 
						i += (long)gridDim.x * blockDim.x) {

		for (int nb = 0; nb < numBuffs; nb++) {
			In[i + nb * numElements] = static_cast<T>(numElements-i);
		}
	}
}

template <typename T=DATA_TYPE>
__global__
void initializeSequential(T *In, long numElements, int numBuffs) {
	for (long i = ((long)blockDim.x * blockIdx.x + threadIdx.x); 
						i < numElements; 
						i += (long)gridDim.x * blockDim.x) {

		for (int nb = 0; nb < numBuffs; nb++) {
			In[i + nb * numElements] = static_cast<T>(i);
		}
	}
}

template <typename T=DATA_TYPE>
__global__
void initializeOutputVariable(T *Out, T value )   
{

	 if( blockIdx.x == 0 && threadIdx.x == 0 ) 
			*Out = value;
}

template <typename T=DATA_TYPE>
__global__
void initSegStarts(T *segStart,
      int nSegments,
      int num_items){
  
  int seg_size = num_items / nSegments;
  for (int i = threadIdx.x+blockDim.x*blockIdx.x; 
              i < nSegments; 
              i+=blockDim.x*gridDim.x){
    segStart[i] = i*seg_size;
  }
  segStart[nSegments] = num_items;
}


// Função para gerar um número aleatório seguindo a distribuição normal
double normal_rand(double mean, double stddev) {
    // Gera números aleatórios usando a distribuição normal (Box-Muller transform)
    double u1 = ((double) rand() / RAND_MAX);
    double u2 = ((double) rand() / RAND_MAX);
    double z1 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z1 * stddev;
}

// Função para criar o vetor com a soma desejada
void generate_normal_vector(int *vector, int nSegmentos, int totalElementos, double mean, double stddev) {
    // Gera valores normais e armazena em um vetor temporário
    double* temp_vector = (double *) malloc(nSegmentos*sizeof(double));
    double sum = 0.0;

    for (int i = 0; i < nSegmentos; i++) {
        temp_vector[i] = normal_rand(mean, stddev);
        if(temp_vector[i] < 0)
          temp_vector[i] = 0;
        sum += temp_vector[i];
    }

    // Ajusta os valores para que a soma seja igual a totalElementos
    double adjustment = floor((totalElementos - sum) / (double)nSegmentos);

    for (int i = 0; i < nSegmentos; i++) {
        vector[i] = (int) round(temp_vector[i] + adjustment);
        if (vector[i] < 0){
          vector[i] = 0;
        }
    }

    free(temp_vector);
}



void generate_normal_vector(uint32_t* vector, int n, uint32_t target_sum,
    double mean,
    double stddev) {
    double* temp = (double*) malloc(n * sizeof(double));
    if (temp == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return;
    }

    double sum = 0.0;

    // Generate n normal distributed numbers (Box-Muller transform)
    for (int i = 0; i < n; i += 2) {
        double u1 = ((double) rand() + 1) / ((double) RAND_MAX + 2);  // avoid log(0)
        double u2 = ((double) rand() + 1) / ((double) RAND_MAX + 2);

        double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

        if (i < n) temp[i] = mean + stddev * z0;
        if (i + 1 < n) temp[i + 1] = mean + stddev * z1;
    }

    // Shift all values to positive
    double min = temp[0];
    for (int i = 1; i < n; i++) {
        if (temp[i] < min) min = temp[i];
    }
    if (min < 0) {
        for (int i = 0; i < n; i++) {
            temp[i] -= min;
        }
    }

    // Sum all values
    sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += temp[i];
    }

    // Scale and convert to uint32_t
    if (sum != 0) {
        double scale = (double)target_sum / sum;
        // double temp_sum = 0.0;
        uint32_t int_sum = 0;

        for (int i = 0; i < n; i++) {
            temp[i] *= scale;
            vector[i] = (uint32_t)(temp[i]);
            int_sum += vector[i];
        }

        // Adjust to make sure total sum exactly matches target_sum
        int i = 0;
        while (int_sum < target_sum && i < n) {
            vector[i]++;
            int_sum++;
            i++;
            if (i == n) i = 0;
        }
    } else {
        // If sum was zero, distribute uniformly
        uint32_t base = target_sum / n;
        uint32_t remainder = target_sum % n;
        for (int i = 0; i < n; i++) {
            vector[i] = base + (i < remainder ? 1 : 0);
        }
    }

	// printf("vector[%d]=%u\n", 0, vector[0]);
  uint sum_vector = vector[0];
  uint ant = vector[0];
  vector[0] = 0;
	for (int i = 1; i < n+1; i++) {
		ant = vector[i];
    vector[i] = sum_vector;
    sum_vector+=ant;
		// printf("vector[%d]=%u\n", i, vector[i]);
	}


    free(temp);
}

void segStartInitNormalDist(uint* d_segStart,
							uint nSegments,
							uint num_items, 
              double mean,
              double stddev
){

	uint* h_segStart = (uint*)malloc((nSegments+2)*sizeof(uint));

	generate_normal_vector(h_segStart,nSegments, num_items, mean, stddev);

	cudaError_t err = cudaMemcpy(d_segStart,h_segStart,(nSegments+1)*sizeof(uint),cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy data from host to device: %s\n", cudaGetErrorString(err));
		return;
	}

  #if WRITE_SEGMENT_SIZES_TO_FILE
    int* h_segSize = (int*)malloc((nSegments)*sizeof(int));
    for(int i = 0; i < nSegments; i++){
      h_segSize[i] = h_segStart[i+1] - h_segStart[i];
      if(h_segSize[i] < 0){
        printf("ERROR: h_segSize[%d]=%d < 0\n",i,h_segSize[i]);
        exit(-1);
      }
    }
    FILE* file0 = fopen("distribuicaoNormal.txt","w");
    if(file0 == NULL){
      printf("ERROR: can't create file.\n");
      return;
    }
    for(int i = 0; i < nSegments; i++){
      fprintf(file0, "%d ",h_segSize[i]);
    }
    fclose(file0);
    free(h_segSize);
  #endif


	// uint min = h_segStart[0];
	// uint max = h_segStart[nSegments];

	// #define HISTOGRAM_BINS 100

  //   // Generate histogram data
  //   size_t histogram[HISTOGRAM_BINS] = {0};
  //   double bin_width = (max - min) / HISTOGRAM_BINS;
  //   printf("***********************\n");
  //   for (size_t i = 1; i < nSegments+1; i++) {
  //       size_t size = h_segStart[i] - h_segStart[i-1];
  //       size_t bin = (size_t)((size) / bin_width);
  //       if (bin >= HISTOGRAM_BINS) bin = HISTOGRAM_BINS - 1;
  //       histogram[bin]++;
  //       // printf("h_segStart[%zu]=%u, size=%zu bin=%zu histogram[bin]=%zu\n", 
  //       //        i, h_segStart[i], size, bin, histogram[bin]);
  //       printf("%zu ", size);
  //   }
  //   printf("\n");
  //   printf("+++++++++++++++++++++++++++\n");



  //   // Save histogram data to a file
  //   FILE *data_file = fopen("histogram_data.txt", "w");
  //   if (!data_file) {
  //       perror("Failed to open data file");
  //       free(h_segStart);
  //       // return -1;
  //   }
  //   for (size_t i = 0; i < HISTOGRAM_BINS; i++) {
  //       fprintf(data_file, "%lf %lu\n", min + i * bin_width, histogram[i]);
  //   }
  //   fclose(data_file);

  //   // Save metadata for Python script
  //   FILE *meta_file = fopen("histogram_meta.txt", "w");
  //   if (!meta_file) {
  //       perror("Failed to open metadata file");
  //       free(h_segStart);
  //       // return EXIT_FAILURE;
  //   }
  //   fprintf(meta_file, "mean: %.2f\n", mean);
  //   fprintf(meta_file, "variance: %.2f\n", stddev * stddev);
  //   fprintf(meta_file, "stddev: %.2f\n", stddev);
  //   fprintf(meta_file, "nelements: %ld\n", nSegments);
  //   fclose(meta_file);

  //   // Execute Python script to plot histogram
  //   system("python3 plot_histogram.py");

	free(h_segStart);

}




void initNormalDist(uint *d_segStart,
      int nSegments,
      int num_items,
      DATA_TYPE* d_out){
  srand(0);

  int* h_segSize = (int*)malloc((nSegments)*sizeof(int));
  int* h_segStart = (int*)malloc((nSegments+1)*sizeof(int));

  double t = (double)num_items/(double)nSegments; // t como definido no artigo
  double mean = t*2;
  double stddev = t*4;
  generate_normal_vector(h_segSize, nSegments, num_items, mean, stddev );


  for(int i = 0; i < nSegments; i++){
    int p0 = rand() % nSegments;
    int p1 = rand() % nSegments;
    
    int aux = h_segSize[p0];
    h_segSize[p0] = h_segSize[p1];
    h_segSize[p1] = aux;
  }

  #if WRITE_SEGMENT_SIZES_TO_FILE
    FILE* file0 = fopen("distribuicaoNormal.txt","w");
    if(file0 == NULL){
      printf("ERROR: can't create file.\n");
      return;
    }
    for(int i = 0; i < nSegments; i++){
      fprintf(file0, "%d ",h_segSize[i]);
    }
    fclose(file0);
  #endif

  //shuffle
  for(int i = 0; i < nSegments; i++){
    int p0 = rand() % nSegments;
    int p1 = rand() % nSegments;
    
    int aux = h_segSize[p0];
    h_segSize[p0] = h_segSize[p1];
    h_segSize[p1] = aux;
  }

  //prefix sum
  h_segStart[0] = 0;
  int accumulator = 0;
  for(int i = 0; i < nSegments; i++){
    accumulator += h_segSize[i];
    h_segStart[i+1] = accumulator;

    if(accumulator > num_items){
      if (i < nSegments-1){
        printf("ERROR: accumulator > num_items !!!!!!!!!!!!\n");
        printf("accumulator=%d i=%d h_segStart[i]=%d h_segSize[i]=%d\n",accumulator, i, h_segStart[i], h_segSize[i]);
      }
      h_segStart[i+1] = num_items;
    }
  }

  cudaError_t err = cudaMemcpy(d_segStart,h_segStart,(nSegments+1)*sizeof(uint),cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to copy data from host to device: %s\n", cudaGetErrorString(err));
      return;
  }
  cudaMemset(d_out,0,nSegments*sizeof(uint));

  free(h_segStart);
  free(h_segSize);
}

double generate_exponential(double lambda) {
    double u = (rand() + 1.0) / (RAND_MAX + 2.0); // evitar log(0)
    return -log(u) / lambda;
}

// Function to generate an exponential random variable with mean 1
double generate_exponential_v0(double lambda) {
    double u = rand() / (RAND_MAX + 1.0);
    return -log(1 - u) / lambda;
}

// Function to generate a vector with exponential distribution and sum equal to X
void generate_exponential_vector(int nSegs, int nElements, int *vector, double lambda) {
    double sum = 0.0;
    double *exp_values = (double *)malloc(nSegs * sizeof(double));


    // Step 1: Generate nSegs exponential random values
    for (int i = 0; i < nSegs; ++i) {
        exp_values[i] = generate_exponential(lambda);
        sum += exp_values[i];
    }
    
    // Step 2: Normalize values to make their sum equal to nElements
    double scale_factor = floor((nElements) / sum);
    int int_sum = 0;
    
    for (int i = 0; i < nSegs; ++i) {
        vector[i] = (int)(exp_values[i] * scale_factor);
        int_sum += vector[i];
    }
    
    // Step 3: Adjust the values to ensure the sum is exactly nElements
    int i = 0;
    while (int_sum < nElements) {
        vector[i]++;
        int_sum++;
        i = (i + 1) % nSegs;
    }
    while (int_sum > nElements) {
        if (vector[i] > 1) {
            vector[i]--;
            int_sum--;
        }
        i = (i + 1) % nSegs;
    }
    
    free(exp_values);
}

// Function to generate a vector with exponential distribution and sum equal to X
void generate_exponential_vector_v0(int nSegs, int nElements, int *vector, double lambda) {
    double sum = 0.0;
    double *exp_values = (double *)malloc(nSegs * sizeof(double));


    // Step 1: Generate nSegs exponential random values
    for (int i = 0; i < nSegs; ++i) {
        exp_values[i] = generate_exponential_v0(lambda);
        sum += exp_values[i];
    }
    
    // Step 2: Normalize values to make their sum equal to nElements
    double scale_factor = floor((nElements) / sum);
    int int_sum = 0;
    
    for (int i = 0; i < nSegs; ++i) {
        vector[i] = (int)(exp_values[i] * scale_factor);
        int_sum += vector[i];
    }
    
    // Step 3: Adjust the values to ensure the sum is exactly nElements
    int i = 0;
    while (int_sum < nElements) {
        vector[i]++;
        int_sum++;
        i = (i + 1) % nSegs;
    }
    while (int_sum > nElements) {
        if (vector[i] > 1) {
            vector[i]--;
            int_sum--;
        }
        i = (i + 1) % nSegs;
    }
    
    free(exp_values);
}

void initExpDist(uint *d_segStart,
      int nSegments,
      int num_items,
      double lambda,
      int use_gnuplot
      ){
  srand(0);

  int* h_segSize = (int*)malloc((nSegments)*sizeof(int));
  int* h_segStart = (int*)malloc((nSegments+1)*sizeof(int));

  generate_exponential_vector(nSegments, num_items, h_segSize,lambda );

  if(use_gnuplot)
    geraGrafico(lambda, num_items, nSegments, num_items/nSegments, 500, h_segSize);

  for(int i = 0; i < nSegments; i++){
    int p0 = rand() % nSegments;
    int p1 = rand() % nSegments;
    
    int aux = h_segSize[p0];
    h_segSize[p0] = h_segSize[p1];
    h_segSize[p1] = aux;
  }

  //prefix sum
  h_segStart[0] = 0;
  int accumulator = 0;
  for(int i = 0; i < nSegments; i++){
    accumulator += h_segSize[i];
    h_segStart[i+1] = accumulator;

    if(accumulator > num_items){
      if (i < nSegments-1){
        printf("ERROR: accumulator > num_items !!!!!!!!!!!!\n");
        printf("accumulator=%d i=%d h_segStart[i]=%d h_segSize[i]=%d\n",accumulator, i, h_segStart[i], h_segSize[i]);
      }
      h_segStart[i+1] = num_items;
    }
  }




  for(int i = 0; i < nSegments; i++){
    h_segSize[i] = h_segStart[i+1] - h_segStart[i];
    if(h_segSize[i] < 0){
      printf("ERROR: h_segSize[%d]=%d < 0\n",i,h_segSize[i]);
      exit(-1);
    }
  }
  #if WRITE_SEGMENT_SIZES_TO_FILE
    FILE* file0 = fopen("distribuicaoExp.txt","w");
    if(file0 == NULL){
      printf("ERROR: can't create file.\n");
      return;
    }
    for(int i = 0; i < nSegments; i++){
      fprintf(file0, "%d ",h_segSize[i]);
    }
    fclose(file0);
  #endif







  cudaError_t err = cudaMemcpy(d_segStart,h_segStart,(nSegments+1)*sizeof(uint),cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to copy data from host to device: %s\n", cudaGetErrorString(err));
      return;
  }

  free(h_segStart);
  free(h_segSize);
}









std::vector<int> generate_random_segment_lengths(int num_segments, size_t target_avg_size) {
    std::random_device rd;
    std::mt19937 gen(rd());

    double mean   = static_cast<double>(target_avg_size);
    double stddev = mean * 0.5; // estava 0,28
    std::normal_distribution<double> dist(mean, stddev);

    std::vector<int> lengths(num_segments);
    long long total = 0;

    for (int i = 0; i < num_segments; ++i) {
        double val = dist(gen);
        lengths[i] = std::max(1, static_cast<int>(std::round(val)));
        total += lengths[i];
    }

    // Adjust last segment to approximate target total
    long long target_total = static_cast<long long>(num_segments) * target_avg_size;
    long long diff = target_total - total;
    if (diff != 0 && num_segments > 0) {
        lengths.back() += static_cast<int>(diff);
        if (lengths.back() < 1) lengths.back() = 1;
    }

    return lengths;
}

template <typename T=DATA_TYPE>
void generate_normal_random_segments(uint32_t* d_segStart, int num_segments, size_t target_avg_size, long& num_items){
    auto seg_lengths = generate_random_segment_lengths(num_segments, target_avg_size);
    size_t total_elements = std::accumulate(seg_lengths.begin(), seg_lengths.end(), 0LL);
    num_items = static_cast<long>(total_elements);

    // Prepare offsets
    std::vector<int> h_begin_offsets(num_segments + 1, 0);
    for (int i = 1; i <= num_segments; ++i) {
        h_begin_offsets[i] = h_begin_offsets[i-1] + seg_lengths[i-1];
    }
    cudaMemcpy(d_segStart, h_begin_offsets.data(), (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice);

}

#endif
