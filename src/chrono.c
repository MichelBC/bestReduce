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

// chrono.h
//
// A small library to measure time in programs
//
// by W.Zola (2017)

#ifndef CHRNONOC
#define CHRNONOC

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <sys/time.h>     /* struct timeval definition           */
#include <unistd.h>       /* declaration of gettimeofday()       */

#include <time.h>

typedef struct {
    struct timespec xadd_time1, xadd_time2;
    long long xtotal_ns;
    long xn_events;
} chronometer_t;

void chrono_reset(chronometer_t* chrono) {
    chrono->xtotal_ns = 0;
    chrono->xn_events = 0;
}

inline void chrono_start(chronometer_t* chrono) {
    clock_gettime(CLOCK_MONOTONIC_RAW, &(chrono->xadd_time1));
}

inline long long chrono_gettotal(chronometer_t* chrono) {
    return chrono->xtotal_ns;
}

inline void chrono_decrement(chronometer_t* chrono, chronometer_t* chrono2) {
    chrono->xtotal_ns -= chrono2->xtotal_ns;
}

inline long long  chrono_getcount(chronometer_t* chrono) {
    return chrono->xn_events;
}

inline void chrono_stop(chronometer_t* chrono) {
    clock_gettime(CLOCK_MONOTONIC_RAW, &(chrono->xadd_time2) );

    long long ns1 = chrono->xadd_time1.tv_sec*1000*1000*1000 +
        chrono->xadd_time1.tv_nsec;
    long long ns2 = chrono->xadd_time2.tv_sec*1000*1000*1000 +
        chrono->xadd_time2.tv_nsec;
    long long deltat_ns = ns2 - ns1;

    chrono->xtotal_ns += deltat_ns;
    chrono->xn_events++;
}

void chrono_reportTime(chronometer_t* chrono, const char* s) {
    printf("%s deltaT(ns): %lld ns for %ld ops \n"
            "        ==> each op takes %lld ns\n",
            s, chrono->xtotal_ns, chrono->xn_events,
            chrono->xtotal_ns/chrono->xn_events );
}

void chrono_report_TimeInLoop(chronometer_t* chrono, const char* s, int loop_count) {
    printf("%s deltaT(ns): %lld ns for %ld ops \n"
            "        ==> each op takes %lld ns\n",
            s, chrono->xtotal_ns, chrono->xn_events*loop_count,
            chrono->xtotal_ns/(chrono->xn_events*loop_count));
}

#endif