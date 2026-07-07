/* Copyright (C) 2015 Atsushi Togo */
/* All rights reserved. */

/* This file is part of phonopy. */

/* Redistribution and use in source and binary forms, with or without */
/* modification, are permitted provided that the following conditions */
/* are met: */

/* * Redistributions of source code must retain the above copyright */
/*   notice, this list of conditions and the following disclaimer. */

/* * Redistributions in binary form must reproduce the above copyright */
/*   notice, this list of conditions and the following disclaimer in */
/*   the documentation and/or other materials provided with the */
/*   distribution. */

/* * Neither the name of the phonopy project nor the names of its */
/*   contributors may be used to endorse or promote products derived */
/*   from this software without specific prior written permission. */

/* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
/* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
/* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS */
/* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE */
/* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, */
/* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, */
/* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; */
/* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER */
/* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT */
/* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
/* POSSIBILITY OF SUCH DAMAGE. */

#include "reciprocal_to_normal.h"

#include <stdint.h>

#if defined(MKL_BLAS) || defined(SCIPY_MKL_H)
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lapack_wrapper.h"

static void get_fc3_triplets_optimized(
    double *fc3_normal_squared, const int64_t (*g_pos)[4],
    const int64_t num_g_pos, const lapack_complex_double *fc3_reciprocal,
    const lapack_complex_double *e0, const lapack_complex_double *e1,
    const lapack_complex_double *e2, const int64_t *band_indices,
    const int64_t num_band0, const int64_t num_band,
    const double cutoff_frequency, const int64_t openmp_per_triplets);

static double get_fc3_triplet_blas(
    const lapack_complex_double *fc3_reciprocal,
    const lapack_complex_double *e0, const lapack_complex_double *e1,
    const lapack_complex_double *e2, const int64_t band_index_0,
    const int64_t band_index_1, const int64_t band_index_2,
    const int64_t num_band, lapack_complex_double *workspace);

void reciprocal_to_normal_squared(
    double *fc3_normal_squared, const int64_t (*g_pos)[4],
    const int64_t num_g_pos, const lapack_complex_double *fc3_reciprocal,
    const double *freqs0, const double *freqs1, const double *freqs2,
    const lapack_complex_double *eigvecs0,
    const lapack_complex_double *eigvecs1,
    const lapack_complex_double *eigvecs2, const double *masses,
    const int64_t *band_indices, const int64_t num_band0,
    const int64_t num_band, const double cutoff_frequency,
    const int64_t openmp_per_triplets) {
    int64_t i, j, ij, num_atom;
    double *inv_sqrt_masses_freqs0, *inv_sqrt_masses_freqs1, *inv_sqrt_masses_freqs2;
    lapack_complex_double *e0, *e1, *e2;

    /* Pre-compute inverse sqrt of masses and frequencies for efficiency */
    num_atom = num_band / 3;
    
    /* Pre-scale with both masses and frequencies */
    inv_sqrt_masses_freqs0 = (double *)malloc(sizeof(double) * num_band);
    inv_sqrt_masses_freqs1 = (double *)malloc(sizeof(double) * num_band);
    inv_sqrt_masses_freqs2 = (double *)malloc(sizeof(double) * num_band);
    
    for (i = 0; i < num_atom; i++) {
        for (j = 0; j < 3; j++) {
            inv_sqrt_masses_freqs0[i * 3 + j] = 1.0 / sqrt(masses[i]);
            inv_sqrt_masses_freqs1[i * 3 + j] = 1.0 / sqrt(masses[i]);
            inv_sqrt_masses_freqs2[i * 3 + j] = 1.0 / sqrt(masses[i]);
        }
    }
    
    /* Apply frequency scaling for non-zero frequencies */
    for (i = 0; i < num_band; i++) {
        if (freqs0[i] > cutoff_frequency) {
            inv_sqrt_masses_freqs0[i] /= sqrt(freqs0[i]);
        }
        if (freqs1[i] > cutoff_frequency) {
            inv_sqrt_masses_freqs1[i] /= sqrt(freqs1[i]);
        }
        if (freqs2[i] > cutoff_frequency) {
            inv_sqrt_masses_freqs2[i] /= sqrt(freqs2[i]);
        }
    }

    /* Transpose eigenvectors and apply mass+frequency weighting */
    /* Memory space for three eigenvector matrices is allocated contiguously */
    e0 = (lapack_complex_double *)malloc(sizeof(lapack_complex_double) * 3 *
                                         num_band * num_band);
    e1 = e0 + num_band * num_band;
    e2 = e1 + num_band * num_band;

#ifdef _OPENMP
#pragma omp parallel for private(i, j) if (!openmp_per_triplets)
#endif
    for (ij = 0; ij < num_band * num_band; ij++) {
        i = ij / num_band;
        j = ij % num_band;
        e0[i * num_band + j] = lapack_make_complex_double(
            lapack_complex_double_real(eigvecs0[j * num_band + i]) *
                inv_sqrt_masses_freqs0[j],
            lapack_complex_double_imag(eigvecs0[j * num_band + i]) *
                inv_sqrt_masses_freqs0[j]);
        e1[i * num_band + j] = lapack_make_complex_double(
            lapack_complex_double_real(eigvecs1[j * num_band + i]) *
                inv_sqrt_masses_freqs1[j],
            lapack_complex_double_imag(eigvecs1[j * num_band + i]) *
                inv_sqrt_masses_freqs1[j]);
        e2[i * num_band + j] = lapack_make_complex_double(
            lapack_complex_double_real(eigvecs2[j * num_band + i]) *
                inv_sqrt_masses_freqs2[j],
            lapack_complex_double_imag(eigvecs2[j * num_band + i]) *
                inv_sqrt_masses_freqs2[j]);
    }

    free(inv_sqrt_masses_freqs0);
    free(inv_sqrt_masses_freqs1);
    free(inv_sqrt_masses_freqs2);

    /* Process triplets directly without intermediate storage */
    get_fc3_triplets_optimized(fc3_normal_squared, g_pos, num_g_pos, fc3_reciprocal,
                              e0, e1, e2, band_indices, num_band0, num_band,
                              cutoff_frequency, openmp_per_triplets);

    free(e0);
    e0 = NULL;
    e1 = NULL;
    e2 = NULL;
}

static void get_fc3_triplets_optimized(
    double *fc3_normal_squared, const int64_t (*g_pos)[4],
    const int64_t num_g_pos, const lapack_complex_double *fc3_reciprocal,
    const lapack_complex_double *e0, const lapack_complex_double *e1,
    const lapack_complex_double *e2, const int64_t *band_indices,
    const int64_t num_band0, const int64_t num_band,
    const double cutoff_frequency, const int64_t openmp_per_triplets) {
    
    int64_t i;
    
#ifdef _OPENMP
    if (openmp_per_triplets) {
#pragma omp parallel
        {
            /* Pre-allocate workspace per thread to avoid repeated allocations */
            lapack_complex_double *workspace = (lapack_complex_double *)malloc(
                sizeof(lapack_complex_double) * (num_band * num_band + num_band));
            
#pragma omp for
            for (i = 0; i < num_g_pos; i++) {
                int64_t band_0 = band_indices[g_pos[i][0]];
                int64_t band_1 = g_pos[i][1];
                int64_t band_2 = g_pos[i][2];
                
                /* Note: frequencies are already incorporated in the eigenvectors */
                /* so we don't need the frequency check here */
                
                fc3_normal_squared[g_pos[i][3]] = get_fc3_triplet_blas(
                    fc3_reciprocal, e0, e1, e2, 
                    g_pos[i][0], band_1, band_2, num_band, workspace);
            }
            
            free(workspace);
        }
    } else 
#endif
    {
        /* Serial execution or OpenMP disabled */
        lapack_complex_double *workspace = (lapack_complex_double *)malloc(
            sizeof(lapack_complex_double) * (num_band * num_band + num_band));
        
        for (i = 0; i < num_g_pos; i++) {
            int64_t band_0 = band_indices[g_pos[i][0]];
            int64_t band_1 = g_pos[i][1];
            int64_t band_2 = g_pos[i][2];
            
            fc3_normal_squared[g_pos[i][3]] = get_fc3_triplet_blas(
                fc3_reciprocal, e0, e1, e2, 
                g_pos[i][0], band_1, band_2, num_band, workspace);
        }
        
        free(workspace);
    }
}

static double get_fc3_triplet_blas(
    const lapack_complex_double *fc3_reciprocal,
    const lapack_complex_double *e0, const lapack_complex_double *e1,
    const lapack_complex_double *e2, const int64_t band_index_0,
    const int64_t band_index_1, const int64_t band_index_2,
    const int64_t num_band, lapack_complex_double *workspace) {
    
    int64_t i, j;
    lapack_complex_double zero, one, result;
    lapack_complex_double *fc3_slice = workspace;
    lapack_complex_double *temp_vector = workspace + num_band * num_band;
    
    zero = lapack_make_complex_double(0.0, 0.0);
    one = lapack_make_complex_double(1.0, 0.0);
    
    /* Step 1: Contract fc3_reciprocal with e0 */
    /* fc3_slice[k,l] = sum_j fc3_reciprocal[j,k,l] * e0[band_index_0, j] */
    const lapack_complex_double *e0_row = e0 + band_index_0 * num_band;
    
    for (i = 0; i < num_band * num_band; i++) {
        fc3_slice[i] = zero;
    }
    
    for (j = 0; j < num_band; j++) {
        cblas_zaxpy(num_band * num_band, e0_row + j,
                   fc3_reciprocal + j * num_band * num_band, 1,
                   fc3_slice, 1);
    }
    
    /* Step 2: Contract with e1 */
    /* temp_vector[l] = sum_k fc3_slice[k,l] * e1[band_index_1, k] */
    const lapack_complex_double *e1_row = e1 + band_index_1 * num_band;
    
    cblas_zgemv(CblasRowMajor, CblasTrans, num_band, num_band,
                &one, fc3_slice, num_band, e1_row, 1, &zero,
                temp_vector, 1);
    
    /* Step 3: Final contraction with e2 */
    /* result = sum_l temp_vector[l] * e2[band_index_2, l] */
    const lapack_complex_double *e2_row = e2 + band_index_2 * num_band;
    
    cblas_zdotu_sub(num_band, temp_vector, 1, e2_row, 1, &result);
    
    return lapack_complex_double_real(result) * lapack_complex_double_real(result) +
           lapack_complex_double_imag(result) * lapack_complex_double_imag(result);
}

 