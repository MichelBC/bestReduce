#!/bin/bash
exec=./bestReduce

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

alg=$1
#distribution

alg_options=(segCub segWarp segBlock segKernel bestReduce segThread segThrust manyBlocks)

dist=$2    
dist_options=(regular normal exp normalv1)
runs=30

# Function to display menu
display_menu() {
    echo "Select an option:"
    for i in "${!choices[@]}"; do
        echo "$((i+1)). ${choices[i]}"
    done
}


nElements=$((2 ** 25)) 

do_test() {     #essa eh a do_test TRANSPOSTA

#  echo ------ 
#  echo TEST=$alg 
#  echo -d:$dist
#  echo ------ 
  for ((i=1; i<=runs; i++));  
 do 
    # if [ $i -ne 1 ]; then
    #   printf " $alg "
    # fi 
    if [ $i -eq 1 ]; then
      printf "\-\-\-\- TEST=$alg \-\-\-\- "
    elif [ $i -eq 2 ]; then
      printf " -d:$dist  "
    elif [ $i -eq 3 ]; then
      printf "\-\-\-\- \-\-\-\- \-\-\-\- "
    else
      printf " $alg  "
    fi
   for j in {24..1}  
   do 
     nseg=$((2 ** j)) 
     segSize=$((nElements / nseg))
  #  echo [$nseg][$segSize] [$j]

         TEST=$alg "$exec" "$nElements" "$nseg" -d $dist | grep "giga elements/s" | awk -F 'Throughput: ' '{print $2}' | awk '{printf "%s ", $1}' 
   done 
   echo "" 
 done

}


do_manyBlocks_test() {     #essa eh a do_test TRANSPOSTA

#  echo ------ 
#  echo TEST=$alg 
#  echo -d:$dist
#  echo -grain:$1
#  echo ------ 
  for ((i=1; i<=runs; i++));  
 do  
    if [ $i -eq 1 ]; then
      printf "\-\-\-\- TEST=$alg \-\-\-\- "
    elif [ $i -eq 2 ]; then
      printf " -d:$dist  "
    elif [ $i -eq 3 ]; then
      printf " -grain:$1  "
    elif [ $i -eq 4 ]; then
      printf "\-\-\-\- \-\-\-\- \-\-\-\- "
    else
      printf " $alg  "
    fi
   for j in {24..1}  
   do 
     nseg=$((2 ** j)) 
     segSize=$((nElements / nseg))
#    echo [$nseg][$segSize]

         TEST=$alg "$exec" "$nElements" "$nseg" -d $dist -grain $1 | grep "giga elements/s" | awk -F 'Throughput: ' '{print $2}' | awk '{printf "%s ", $1}' 
   done 
   echo "" 
 done

}

valid=false   # IMPORTANTE: inicializar


if [ ! -z "$3" ]; then
    runs=$3
fi

for d in "${dist_options[@]}"; do
  if [[ "$d" == "$dist" ]]; then
    valid=true
    break
  fi
done
if [[ "$valid" == false ]]; then
  echo "Invalid distribution name: '$dist'"
  echo "Available distributions are: ${dist_options[*]}"
  exit 1
fi

if [[ "$alg" == "allkernels" ]]; then
  alg="segThread"
  do_test
  alg="segWarp"
  do_test
  alg="segBlock"
  do_test
  alg="manyBlocks"
  do_manyBlocks_test 2
  do_manyBlocks_test 16
  alg="segKernel"
  do_test
elif [[ "$alg" == "all" ]]; then
  alg="segCub"
  do_test
  alg="segThrust"
  do_test 
  alg="bestReduce"
  do_test 
elif [[ "$alg" == "manyBlocks" ]]; then
  if [ ! -z "$4" ]; then
      grain=2
  fi
  do_manyBlocks_test $4
else
  # Verifica se a palavra está na lista (como palavra inteira)valid=false

for item in "${alg_options[@]}"; do
  if [[ "$item" == "$alg" ]]; then
    do_test
    exit 0
  fi
  done
  echo "Invalid alg name: $alg"
  echo "Available alg are:"
  echo "${alg_options[*]}"
  exit 1
fi
