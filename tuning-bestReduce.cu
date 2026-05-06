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

#include <iostream>
#include <vector>
#include <cmath>
#include <boost/math/tools/roots.hpp>
#include <boost/cstdint.hpp>

#include <functional>
#include <fstream>
#include <sstream>


#include "src/gpu-utils.h"
#include "src/seg-reduce.cu"
#include "src/init-dist.cu"
#include "src/chrono.c"

// #define A4500 1
// #define rtx4080 2
// #define GPU A4500
//#define GPU rtx4080

// Definir os limites do intervalo
#define XMIN 2
#define XMAX (1ULL << 24) // 2^24 = 16,777,216

// resultados da RTX4080 na maquina do Bruno
// isso foi gerado pelo script ./do-seg-tests-30-times-FIG1.sh
#if GPU == rtx4080
double  segSizes[]={-1, 2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216};
double threadReducex4[]={-1,80.67,107.29,129.29,142.63,150.72,156.54,154.47,136.93,160.95,88.07,27.79,8.70,7.96,6.78,5.28,3.64,4.49,5.07,5.42,5.62,5.72,5.77,5.80,5.82};
double warpReduceResults[]={-1,13.06,25.94,42.56,81.85,137.13,153.18,158.01,161.20,162.23,162.73,164.54,163.74,163.63,165.59,164.15,154.67,109.88,59.19,30.53,16.15,8.20,4.12,2.07,1.07};
double blockReduceResults[]={-1,4.60,9.08,17.61,28.81,56.99,107.03,152.08,160.82,161.84,162.45,163.61,163.51,165.34,165.10,165.39,165.25,164.99,163.85,163.25,162.75,142.63,83.45,43.16,23.47};
double mBlocks2[]={-1,0.29,0.51,1.02,2.03,4.07,8.12,16.15,31.91,62.01,109.06,152.67,162.69,164.60,164.53,164.87,165.53,163.88,164.24,165.32,164.43,161.91,141.75,84.38,43.85};
double manyBlocks16[]={-1,0.05,0.10,0.20,0.40,0.80,1.59,3.18,6.35,12.68,25.23,49.62,90.15,117.09,135.53,154.39,165.73,165.82,165.70,165.79,165.67,165.67,164.91,165.30,163.22};
double kernelReduceResults[]={-1,0.00,0.00,0.00,0.00,0.00,0.14,0.28,0.54,1.06,2.05,3.95,7.33,14.51,28.54,56.43,107.90,162.37,165.32,159.62,165.50,165.83,165.66,165.69,165.42};

#elif GPU == A4500
// resultados da A4500 na t101
// isso foi gerado pelo script ./do-seg-tests-30-times-FIG1.sh

double  segSizes[]={-1, 2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216};
double threadReducex4[]={-1,69.57,96.35,115.51,124.11,129.66,133.05,128.72,135.63,137.79,100.22,36.88,11.17,3.08,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00};
double warpReduceResults[]={-1,7.40,14.03,24.11,47.80,93.98,135.34,140.24,143.19,144.06,144.19,144.72,143.15,144.97,145.51,145.13,140.10,97.76,51.16,26.00,13.10,6.59,3.30,1.65,0.83};
double blockReduceResults[]={-1,3.15,3.68,7.25,11.53,23.03,45.77,86.46,125.31,144.76,145.76,146.05,146.02,146.13,145.87,145.64,146.12,145.96,142.45,146.21,144.69,125.82,71.57,36.77,18.56};
//double mBlocks2[]={-1,3.00,3.00,3.00,3.00,3.00,3.00,3.00,3.00,3.00,3.00,3.00,128.52,146.43,146.63,146.53,146.12,146.46,146.22,143.16,146.15,144.43,128.11,72.66,37.26};
double mBlocks2[]={-1,0.16, 0.28, 0.55, 1.10, 2.21, 4.42, 8.80, 17.45, 34.42, 66.13, 100.17, 128.36, 146.43, 146.62, 146.52, 146.12, 146.45, 146.22, 143.18, 146.15, 144.44, 127.99, 72.54, 37.19};
//double manyBlocks16[]={-1,2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.00,2.80,50.32,92.56,126.70,130.95,146.37,146.78,146.80,146.50,146.55,146.26,142.61,146.01,144.41};
double manyBlocks16[]={-1,0.03, 0.05, 0.10, 0.20, 0.41, 0.82, 1.63, 3.26, 6.51, 12.94, 25.43, 50.30, 92.28, 126.67, 131.03, 146.36, 146.79, 146.80, 146.49, 146.56, 146.26, 142.57, 146.00, 144.38};
double kernelReduceResults[]={-1,0.00,0.01,0.01,0.02,0.05,0.09,0.18,0.36,0.71,1.41,2.62,5.11,10.21,20.64,44.60,88.04,143.89,146.37,145.60,140.79,145.13,146.53,146.63,146.75};

#endif

int log2(int x) {
//int xx = x;
    int result = 0;
    while (x > 1) {
        x >>= 1;
        result++;
    }
//    std::cout << "log2 " << xx << "=" << result << "\n";
    return result;
}

int funcao = 0;

uint* d_segStart = nullptr;
uint* d_in = nullptr;
int ncopies = 0;
uint* d_out = nullptr;
int nElements = 0;
int nblocks = 0;
int nthreads = 0;
int ntimes = 0;
double measureThroughput(int segSize, int funcao_id){
    int nSegments = nElements / segSize;
    initSegStarts<<<nblocks, nthreads>>>(d_segStart,nSegments,nElements);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    std::function<void(uint*, int)> f;
    switch (funcao_id) {
    case 0: {

		int bufferSize = nthreads*2;
        int nb = (nSegments+nthreads-1) / nthreads;
        // printf("Calling threadReduce\n");
        f = [nb, bufferSize] (uint* in, int ns) {
            threadReduce<<<nb,nthreads, bufferSize*sizeof(DATA_TYPEx4)>>>(
                in, nElements, 
                d_segStart, ns, 
                d_out, bufferSize
            );
        };
        break;
    }
    case 1: 
        // printf("Calling warpReduce\n");
        f = [] (uint* in, int ns) {
            warpReduce<<<nblocks,nthreads>>>(
                in, nElements, 
                d_segStart, ns, 
                d_out
            );
        };
        break;
    case 2: {
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
        // printf("Calling blockReduce\n");
        f = [nt] (uint* in, int ns) {
            blockReduce<<<ns, nt>>>(
                in, nElements, 
                d_segStart, ns, 
                d_out
            );
        };
        break;
    }
    case 3: 
        // printf("Calling manyBlocksGr with grain=2\n");
        f = [grain=2] (uint* in, int ns) {
            manyBlocksGr<<<ns*grain, nthreads>>>(
                in, nElements, d_segStart, ns, d_out, grain);
        };
        break;
    case 4: 
        // printf("Calling manyBlocksGr with grain=16\n");
        f = [grain=16] (uint* in, int ns) {
            manyBlocksGr<<<ns*grain, nthreads>>>(
                in, nElements, d_segStart, ns, d_out, grain);
        };
        break;
    case 5:
        // printf("Calling kernelReduce\n");
        f = [] (uint* in, int ns) {
            kernelReduce<<<nblocks,nthreads>>>(
                in, nElements, 
                d_segStart, ns, 
                d_out
            );
        };
        break;
    default:
        printf("ERROR: FUNCTION ID DOES NOT EXIST! File: %s, Line: %d\n", __FILE__, __LINE__);
        exit(1);
   }  

	chronometer_t c1;
	chrono_reset(&c1);
    chrono_start(&c1);
        for (int i = 0; i < ntimes; i++) {
            DATA_TYPE *d_inBuff = &d_in[ nElements*(i%ncopies) ];
            f(d_inBuff, nSegments);
        }
        cudaDeviceSynchronize();
    chrono_stop(&c1);

    double throughput = ((double)nElements*ntimes)/((double)chrono_gettotal(&c1));
    // printf("func_id = %d nsegs = %d throughput = %lf\n",funcao_id,nSegments,throughput);
    return throughput;
}

// Definir as funções (para simular medições de tempo)
double f1(int x) {
    return measureThroughput(x,funcao);
}

double f2(int x) {
   double EPSILON = 0.002L ;
   double fator = 1.0L + EPSILON;
   
    return measureThroughput(x,funcao+1) * fator;    
}


// Função para interpolação linear
double interpolate(const std::vector<double>& x, const std::vector<double>& y, double x_val) {
    for (size_t i = 0; i < x.size() - 1; ++i) {
        if (x[i] <= x_val && x_val <= x[i + 1]) {
            double t = (x_val - x[i]) / (x[i + 1] - x[i]);
            return y[i] + t * (y[i + 1] - y[i]);
        }
    }
    // Se x_val está fora do intervalo, retorna o valor mais próximo
    if (x_val < x[0]) return y[0];
    return y.back();
}

#include <stdbool.h>

#define TYPE double
//#define TYPE float

// Função para verificar se um valor está entre 0 e 1 (inclusive)
bool isBetweenZeroAndOne( TYPE t) {
    return t >= 0.0f && t <= 1.0f;
}



// Função para calcular o ponto de interseção entre dois segmentos de reta
bool findIntersection( TYPE sa_x1,  TYPE sa_y1,  TYPE sa_x2,  TYPE sa_y2,
                       TYPE sb_x1,  TYPE sb_y1,  TYPE sb_x2,  TYPE sb_y2,
                       TYPE *px,  TYPE *py) {
    // Vetores direção dos segmentos
     TYPE dx1 = sa_x2 - sa_x1;
     TYPE dy1 = sa_y2 - sa_y1;
     TYPE dx2 = sb_x2 - sb_x1;
     TYPE dy2 = sb_y2 - sb_y1;

    // Calcula o determinante
     TYPE det = dx1 * dy2 - dy1 * dx2;

    // Se o determinante for zero, as retas são paralelas
    if (det == 0.0f) {
        return false;
    }

    // Calcula os parâmetros t e u
     TYPE t = ((sb_x1 - sa_x1) * dy2 - (sb_y1 - sa_y1) * dx2) / det;
     TYPE u = ((sb_x1 - sa_x1) * dy1 - (sb_y1 - sa_y1) * dx1) / det;

    // Verifica se a interseção está dentro dos segmentos
    if (isBetweenZeroAndOne(t) && isBetweenZeroAndOne(u)) {
        // Calcula o ponto de interseção
        *px = sa_x1 + t * dx1;
        *py = sa_y1 + t * dy1;
        return true;
    }

    return false;
}

int main(int argc, char* argv[]){
	cudaError_t err = cudaSuccess;

    nElements = 1 << 25;
    if (argc > 1)
        nElements = atol(argv[1]);
    ncopies = 10;
    ntimes = 100;
    if (argc > 2)
        ntimes = atol(argv[2]);


        
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0 );
    nthreads = prop.maxThreadsPerMultiProcessor / 2;
    if(nthreads > prop.maxThreadsPerBlock)
        nthreads = prop.maxThreadsPerBlock;
    nblocks = prop.multiProcessorCount*
        (prop.maxThreadsPerMultiProcessor/nthreads);
    printf("=========== Device Properties ===========\n");
    printf( "cudaDeviceName: %s\n", prop.name );
    printf( "l2CacheSize: %d\n", prop.l2CacheSize );
    printf( "persistingL2CacheMaxSize: %d\n", prop.persistingL2CacheMaxSize );
    printf( "maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock );
    printf( "multiProcessorCount: %d\n", prop.multiProcessorCount );
    printf( "maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor );
    printf( "warpSize: %d\n", prop.warpSize );
    printf( "regsPerBlock: %d\n", prop.regsPerBlock );
    printf( "sharedMemPerBlock: %lu\n", prop.sharedMemPerBlock );
    printf( "regsPerBlock: %d\n", prop.regsPerBlock );
    printf( "sharedMemPerBlock: %lu\n", prop.sharedMemPerBlock );

    printf("=========== Device Configuration ===========\n");
    printf("nblocks: %d\n", nblocks);
    printf("nthreads: %d\n", nthreads);
    printf("ntimes: %d\n", ntimes);
    printf("ncopies: %d\n", ncopies);
    printf("==========================================\n");
    printf("nElements: %d\n", nElements);

    err = cudaMalloc(&d_segStart, (nElements+1)*sizeof(int));
    if (err != cudaSuccess){
            fprintf(stderr, "Failed to allocate device vector d_segStart (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&d_in, nElements*ncopies*sizeof(int));
    if (err != cudaSuccess){
            fprintf(stderr, "Failed to allocate device vector d_in (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
    }    err = cudaMalloc(&d_out, nElements*sizeof(int));
    if (err != cudaSuccess){
            fprintf(stderr, "Failed to allocate device vector d_out (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
    }
	curand_kernel_uint32_persT<<<nblocks, nthreads>>>( d_in, nElements*ncopies,1000000000, 0 );
    
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    // std::ofstream header("thresholds.h", std::ios::app);
    // header << 
    // "//----------------------------------------------\n"
    // "#ifdef DEFAULT_GPU_NAME\n"
    // "    #undef DEFAULT_GPU_NAME\n"
    // "    #undef THRESHOLD1\n"
    // "    #undef THRESHOLD2\n"
    // "    #undef THRESHOLD3\n"
    // "    #undef THRESHOLD4\n"
    // "    #undef THRESHOLD5\n"
    // "#endif\n"
    // "#define DEFAULT_GPU_NAME \""
    // << prop.name << "\"\n"
    // "// this threshold was tuned for " << nElements << " elements. \n";

    // Gerar medições de tempo para x = 2, 4, 8, ..., até <= XMAX
    std::vector<double> x_values, t1_values, t2_values;

    // unsigned int left[5];    // valores a esquerda no intervalo
    // unsigned int right[5];   // valores a direita no intervalo
    unsigned int threshold[5] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
    int last_x = XMIN;    
    int xmax = nElements;
    for(int f=0; f<5; f++ ) 
    //int f = 2;
    {
        funcao = f;
        std::cout << "funcao: " << f << "\n";

        int x;        
        // for (x = last_x; x <= XMAX; x *= 2) {
        for (x = last_x; x < xmax; x *= 2) {
            std::cout << "x:" << x << " f1:" << f1(x) << " f2:" << f2(x) << "\n";
            if( f2(x) >= f1(x) )   // f2 ultrapassa f1 (ou iguala)
            break;
        }       
        last_x = x;
        if( f2(x) < f1(x) ) {
            // printf("Nenhuma mudança de sinal no intervalo\n");
            printf("ERROR: Something went wrong during the tuning process. Please check whether there are any background processes consuming GPU resources, or consider increasing the number of elements in the tests (currently %d elements).\n", nElements);
            exit( 0 );
            break;
        }
    
        // chegando aqui f2 ultrapassa f1 (ou iguala) no ponto x
        std::cout << "// Ponto de transição no intervalo [" << x/2 << "," << x << "]\n";
        // header << "// Ponto de transição no intervalo [" << x/2 << "," << x << "]\n";

            TYPE px, py;
            // Função para calcular o ponto de interseção entre dois segmentos de reta
            bool instersects = findIntersection( x/2, f1(x/2), x, f1(x), 
                                                x/2, f2(x/2), x, f2(x),
                                                &px, &py );
            if( instersects ) {
            std::cout << "Intersecao em x=" << (int)ceil(px) << ", y=" << py << "\n";
            std::cout << "#define THRESHOLD" << f+1 << " " << (int)ceil(px) << "\n";
            // header << "#define THRESHOLD" << f+1 << " " << (int)ceil(px) << "\n";
            // left[found] = x/2;
            // right[found] = x;
            threshold[f] = (int)ceil(px);

            } else{
                    // std::cout << "Intersecao entre segmentos nao encontrada\n";
                printf("ERROR: Something went wrong during the tuning process. Please check whether there are any background processes consuming GPU resources, or consider increasing the number of elements in the tests (currently %d elements).\n", nElements);
                exit( 0 );
                break;

            }
            
        
        
    }    

    // std::ofstream file("thresholds.txt", std::ios::app);
    // file << prop.name << " "
    //     << threshold[0] << " "
    //     << threshold[1] << " "
    //     << threshold[2] << " "
    //     << threshold[3] << " "
    //     << threshold[4] << " "
    //     << nElements << "\n";

    std::ofstream tfile("tuning.data", std::ios::app);
        tfile << "{\""
        << prop.name << "\",{"
        << threshold[0] << ","
        << threshold[1] << ","
        << threshold[2] << ","
        << threshold[3] << ","
        << threshold[4] << "},"
        << nElements << "},\n";

    return 0;
}
