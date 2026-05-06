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

#pragma once
#include <cuda_runtime.h>
#include <limits>


//
// =====================================================
// Regular Operators
// =====================================================
//
#include <climits>

template<typename T>
struct Max;

template<>
struct Max<int> {
    __device__ __forceinline__
    int operator()(int a, int b) const { return a > b ? a : b; }

    __device__ __forceinline__
    static int identity() { return INT_MIN; }

    __device__ __forceinline__
    int atomic(int* addr, int val) const { return atomicMax(addr, val); }
};

template<>
struct Max<unsigned int> {
    __device__ __forceinline__
    unsigned int operator()(unsigned int a, unsigned int b) const { return a > b ? a : b; }

    __device__ __forceinline__
    static unsigned int identity() { return 0; }

    __device__ __forceinline__
    unsigned int atomic(unsigned int* addr, unsigned int val) const { return atomicMax(addr, val); }
};

template<>
struct Max<unsigned long long> {
    __device__ __forceinline__
    unsigned long long operator()(unsigned long long a, unsigned long long b) const { return a > b ? a : b; }

    __device__ __forceinline__
    static unsigned long long identity() { return 0; }

    __device__ __forceinline__
    unsigned long long atomic(unsigned long long* addr, unsigned long long val) const { return atomicMax(addr, val); }
};


template<typename T>
struct Add {
    __device__ __forceinline__
    T operator()(T a, T b) const {
        return a + b;
    }

    __device__ __forceinline__
    static constexpr T identity() {
        return T(0);
    }

    // Atomic version
    __device__ __forceinline__
    T atomic(T* addr, T val) const {
        return atomicAdd(addr, val);
    }
};

template<typename T>
struct Min {
    __device__ __forceinline__
    T operator()(T a, T b) const {
        return a < b ? a : b;
    }

    __device__ __forceinline__
    static constexpr T identity() {
        return std::numeric_limits<T>::max();
    }

    // Atomic version
    __device__ __forceinline__
    T atomic(T* addr, T val) const {
        return atomicMin(addr, val);
    }
};
