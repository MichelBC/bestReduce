#!/usr/bin/env bash

# /***********************************************************************************/
# /*                                                                                 */
# /*   IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.            */
# /*   By downloading, copying, installing or using the software you agree           */
# /*   to this license.  If you do not agree to this license, do not download,       */
# /*   install, copy or use the software.                                            */
# /*                                                                                 */
# /*  BSD 3-Clause License                                                           */
# /*                                                                                 */
# /*  Copyright (c) 2024-2026, Michel Brasil Cordeiro and Wagner M. Nunan Zola       */
# /*  All rights reserved.                                                           */
# /*                                                                                 */
# /*  Redistribution and use in source and binary forms, with or without             */
# /*  modification, are permitted provided that the following conditions are met:    */
# /*                                                                                 */
# /*  1. Redistributions of source code must retain the above copyright notice,      */
# /*     this list of conditions and the following disclaimer.                       */
# /*                                                                                 */
# /*  2. Redistributions in binary form must reproduce the above copyright notice,   */
# /*     this list of conditions and the following disclaimer in the documentation   */
# /*     and/or other materials provided with the distribution.                      */
# /*                                                                                 */
# /*  3. Neither the name of the copyright holder nor the names of its               */
# /*     contributors may be used to endorse or promote products derived from        */
# /*     this software without specific prior written permission.                    */
# /*                                                                                 */
# /*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"    */
# /*  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE      */
# /*  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE */
# /*  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE   */
# /*  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL     */
# /*  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR     */
# /*  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER     */
# /*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,  */
# /*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE  */
# /*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.           */
# /*                                                                                 */
# /*  If you use this software or a modified version of it,                          */
# /*  please cite the most relevant among the following papers:                      */
# /*                                                                                 */
# /*  - Optimized Parallel Reduction for Regular and Irregular Segments on GPU       */
# /*    Cordeiro, Michel B. and Nunan Zola, Wagner M.                                */
# /*    In: Concurrency and Computation: Practice and Experience, 2025,              */
# /*    Special Issue SSCAD24                                                        */
# /*                                                                                 */
# /***********************************************************************************/

sizes=()
for exp in {17..26}; do    # 2^17 = 128Ki, 2^26 = 64Mi
    sizes+=($((2 ** exp)))
done

tests=(thrust cub ui32_sh_mem ui32_atomic_op ui32_shfl ui32x4)
mytests=(ui32_sh_mem ui32_atomic_op ui32_shfl ui32x4)

runs=30
exec=./bestReduce


run_test() {
  local test_name=$1

  echo -n "------ $test_name ------ "

  for ((i=1; i<=runs; i++)); do
    if [ $i -ne 1 ]; then
      printf " $test_name  "
    fi
    for size in "${sizes[@]}"; do
      TEST="$test_name" "$exec" "$size" | grep "giga elements/s" | awk -F 'Throughput: ' '{print $2}' | awk '{printf "%s ", $1}'
    done
    printf "\n"
  done
}

# -----------------------------
# Main logic
# -----------------------------

alg=$1
# check if second parameter exists
if [ ! -z "$2" ]; then
    runs=$2
fi

if [[ -z "$alg" ]]; then
  echo "Usage: $0 <alg>"
  echo "Available alg: ${tests[*]} or 'all'"
  exit 1
fi

if [[ "$alg" == "all" ]]; then
  for t in "${tests[@]}"; do
    run_test "$t"
  done
elif [[ "$alg" == "myall" ]]; then
  for t in "${mytests[@]}"; do
    run_test "$t"
  done
else
  # Check if alg is valid
  valid=false
  for t in "${tests[@]}"; do
    [[ "$t" == "$alg" ]] && valid=true
  done

  if [[ "$valid" == true ]]; then
    run_test "$alg"
  else
    echo "Invalid algorithm: $alg"
    echo "Available alg: ${tests[*]} or 'all'"
    exit 1
  fi
fi