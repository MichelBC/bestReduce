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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define GNUPLOT_CMD "gnuplot -persistent"

double calcular_media(int* dados, int n) {
    long long soma = 0;
    for (int i = 0; i < n; ++i) soma += dados[i];
    return (double)soma / n;
}

double calcular_desvio(int* dados, int n, double media) {
    double soma_quadrados = 0;
    for (int i = 0; i < n; ++i)
        soma_quadrados += (dados[i] - media) * (dados[i] - media);
    return sqrt(soma_quadrados / n);
}

void salvar_histograma(int* dados, int n, int n_bins, const char* filename) {
    int max = dados[0], min = dados[0];
    for (int i = 1; i < n; ++i) {
        if (dados[i] > max) max = dados[i];
        if (dados[i] < min) min = dados[i];
    }

    double bin_width = (double)(max - min + 1) / n_bins;
    int* bins = (int*)calloc(n_bins, sizeof(int));

    for (int i = 0; i < n; ++i) {
        int idx = (int)((dados[i] - min) / bin_width);
        if (idx >= n_bins) idx = n_bins - 1;
        bins[idx]++;
    }

    FILE* f = fopen(filename, "w");
    for (int i = 0; i < n_bins; ++i) {
        double centro = min + (i + 0.5) * bin_width;
        fprintf(f, "%.2f %d\n", centro, bins[i]);
    }
    fclose(f);
    free(bins);
}

//        plotar_grafico(nome_arquivo, lambda, lamda_real, erro_lambda, soma_desejada, soma_real, erro_soma, n, media, media_desejada, erro_media, desvio);

//void plotar_grafico(const char* filename, double lambda, long long soma, int n, double media, double desvio) {
void plotar_grafico(const char* filename, double lambda, double lambda_real, double erro_lambda, long long soma_desejada, long long  soma_real, double erro_soma, int n, double media, double media_desejada, double erro_media, double desvio ) {
    FILE* gp = popen(GNUPLOT_CMD, "w");

    fprintf(gp, "set terminal qt font 'Arial,10'\n");
    fprintf(gp, "set title 'Distribuição Exponencial Normalizada'\n");
    fprintf(gp, "set xlabel 'Valor'\n");
    fprintf(gp, "set ylabel 'Frequência'\n");
    fprintf(gp, "set format y '%%.0f'\n");
    fprintf(gp, "set format x '%%.0f'\n");
    fprintf(gp, "set xtics rotate by 45 right\n");

    // Legendas verticais
/*    fprintf(gp, "set label 1 at graph 0.7, 0.9  sprintf('λ = %.2f') font ',10'\n", lambda);
    fprintf(gp, "set label 2 at graph 0.7, 0.85 sprintf('soma = %.0f') font ',10'\n", (double)soma);
    fprintf(gp, "set label 3 at graph 0.7, 0.80 sprintf('n = %d') font ',10'\n", n);
    fprintf(gp, "set label 4 at graph 0.7, 0.75 sprintf('média = %.2f') font ',10'\n", media);
    fprintf(gp, "set label 5 at graph 0.7, 0.70 sprintf('desvio = %.2f') font ',10'\n", desvio);
*/    
    /////////////
    fprintf(gp, "set label 1 at graph 0.7, 0.95 sprintf('λ = %.2f (erro = %.2f%%)') font ',10'\n", lambda_real, erro_lambda);
fprintf(gp, "set label 2 at graph 0.7, 0.90 sprintf('soma = %.0f (erro = %.2f%%%%)') font ',10'\n", (double)soma_real, erro_soma);
fprintf(gp, "set label 3 at graph 0.7, 0.85 sprintf('n = %d') font ',10'\n", n);
fprintf(gp, "set label 4 at graph 0.7, 0.80 sprintf('média = %.2f (erro = %.2f%%%%)') font ',10'\n", media, erro_media);
fprintf(gp, "set label 5 at graph 0.7, 0.75 sprintf('desvio = %.2f') font ',10'\n", desvio);

    ////////////

    fprintf(gp, "plot '%s' using 1:2 with boxes lc rgb 'blue' notitle\n", filename);
    fflush(gp);
}

void geraGrafico(double lambda,
long long soma_desejada,
int n,
double media_desejada,
int n_bins,
int* dados) {

    if (fabs((double)soma_desejada - media_desejada * n) > 1e-6) {
        fprintf(stderr, "Aviso: soma_desejada e media_desejada * n são inconsistentes!\n");
    }

    double media = calcular_media(dados, n);
    double desvio = calcular_desvio(dados, n, media);
    
    ////////
    long long soma_real = 0;
for (int i = 0; i < n; ++i)
    soma_real += dados[i];

double erro_soma = 100.0 * ((double)soma_real - soma_desejada) / soma_desejada;
double erro_media = 100.0 * (media - media_desejada) / media_desejada;
double lambda_real = lambda; //1.0 / media;
double erro_lambda = 100.0 * (lambda_real - lambda) / lambda;
    /////////

    const char* nome_arquivo = "histograma.dat";
    salvar_histograma(dados, n, n_bins, nome_arquivo);
//    plotar_grafico(nome_arquivo, lambda, soma_desejada, n, media, desvio);
        plotar_grafico(nome_arquivo, lambda, lambda_real, erro_lambda, soma_desejada, soma_real, erro_soma, n, media, media_desejada, erro_media, desvio);

}

