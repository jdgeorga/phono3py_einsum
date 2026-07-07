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

#ifdef MULTITHREADED_BLAS
#if defined(MKL_BLAS) || defined(SCIPY_MKL_H)
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lapack_wrapper.h"

/* Optimized implementation following einsum path:
 * ilI,jmJ,knK,ijklmn,i,j,k,I,J,K -> IJK
 * 
 * Optimization path:
 * 1. I,ilI->Iil (contract frequency I with eigenvector ilI)
 * 2. J,jmJ->Jjm (contract frequency J with eigenvector jmJ)  
 * 3. K,knK->Kkn (contract frequency K with eigenvector knK)
 * 4. Iil,i->Iil (contract mass i with Iil)
 * 5. Jjm,j->Jjm (contract mass j with Jjm)
 * 6. Kkn,k->Kkn (contract mass k with Kkn)
 * 7. Iil,ijklmn->Ijkmn (contract first index with fc3, BLAS)
 * 8. Ijkmn,Jjm->IknJ (contract second index, BLAS)
 * 9. IknJ,Kkn->IJK (final contraction, BLAS)
 */

static void get_fc3_einsum_optimized(
    double *fc3_normal_squared, const int64_t (*g_pos)[4],
    const int64_t num_g_pos, const lapack_complex_double *fc3_reciprocal,
    const double *freqs0, const double *freqs1, const double *freqs2,
    const lapack_complex_double *e0, const lapack_complex_double *e1,
    const lapack_complex_double *e2, const int64_t *band_indices,
    const int64_t num_band0, const int64_t num_band,
    const double cutoff_frequency, const int64_t openmp_per_triplets);

static void create_frequency_weighted_eigenvectors(
    lapack_complex_double *e_weighted, const lapack_complex_double *eigvecs,
    const double *freqs, const double *inv_sqrt_masses,
    const int64_t *band_indices, const int64_t num_bands_selected,
    const int64_t num_band, const double cutoff_frequency);

static void contract_fc3_optimized(
    lapack_complex_double *result, const lapack_complex_double *fc3_reciprocal,
    const lapack_complex_double *e0_weighted, const lapack_complex_double *e1_weighted,
    const lapack_complex_double *e2_weighted, const int64_t num_band0,
    const int64_t num_band1, const int64_t num_band2, const int64_t num_band);

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
    double *inv_sqrt_masses;
    lapack_complex_double *e0, *e1, *e2;

    /* Prepare inverse sqrt masses */
    num_atom = num_band / 3;
    inv_sqrt_masses = (double *)malloc(sizeof(double) * num_band);
    for (i = 0; i < num_atom; i++) {
        for (j = 0; j < 3; j++) {
            inv_sqrt_masses[i * 3 + j] = 1.0 / sqrt(masses[i]);
        }
    }

    /* Transpose eigenvectors for better memory access pattern */
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
                inv_sqrt_masses[j],
            lapack_complex_double_imag(eigvecs0[j * num_band + i]) *
                inv_sqrt_masses[j]);
        e1[i * num_band + j] = lapack_make_complex_double(
            lapack_complex_double_real(eigvecs1[j * num_band + i]) *
                inv_sqrt_masses[j],
            lapack_complex_double_imag(eigvecs1[j * num_band + i]) *
                inv_sqrt_masses[j]);
        e2[i * num_band + j] = lapack_make_complex_double(
            lapack_complex_double_real(eigvecs2[j * num_band + i]) *
                inv_sqrt_masses[j],
            lapack_complex_double_imag(eigvecs2[j * num_band + i]) *
                inv_sqrt_masses[j]);
    }

    free(inv_sqrt_masses);
    inv_sqrt_masses = NULL;

    get_fc3_einsum_optimized(fc3_normal_squared, g_pos, num_g_pos, fc3_reciprocal,
                           freqs0, freqs1, freqs2, e0, e1, e2, band_indices,
                           num_band0, num_band, cutoff_frequency,
                           openmp_per_triplets);

    free(e0);
    e0 = NULL;
    e1 = NULL;
    e2 = NULL;
}

static void get_fc3_einsum_optimized(
    double *fc3_normal_squared, const int64_t (*g_pos)[4],
    const int64_t num_g_pos, const lapack_complex_double *fc3_reciprocal,
    const double *freqs0, const double *freqs1, const double *freqs2,
    const lapack_complex_double *e0, const lapack_complex_double *e1,
    const lapack_complex_double *e2, const int64_t *band_indices,
    const int64_t num_band0, const int64_t num_band,
    const double cutoff_frequency, const int64_t openmp_per_triplets) {
    
    int64_t i, j, k;
    lapack_complex_double *temp_ijkmn, *temp_iknj;
    lapack_complex_double zero, one;
    
    zero = lapack_make_complex_double(0.0, 0.0);
    one = lapack_make_complex_double(1.0, 0.0);
    
    /* Allocate temporary arrays for intermediate results following einsum path */
    temp_ijkmn = (lapack_complex_double *)malloc(
        sizeof(lapack_complex_double) * num_band0 * num_band * num_band * num_band);
    temp_iknj = (lapack_complex_double *)malloc(
        sizeof(lapack_complex_double) * num_band0 * num_band * num_band);
    
    /* Initialize arrays */
    for (i = 0; i < num_band0 * num_band * num_band * num_band; i++) {
        temp_ijkmn[i] = zero;
    }
    for (i = 0; i < num_band0 * num_band * num_band; i++) {
        temp_iknj[i] = zero;
    }

    /* Step 7: Contract e0 (with freq weights) with fc3: Iil,ijklmn->Ijkmn 
     * The einsum optimizer determined this is the most efficient first step
     * This reduces from 6D tensor to 5D tensor
     */
#ifdef _OPENMP
#pragma omp parallel for if (!openmp_per_triplets)
#endif
    for (i = 0; i < num_band0; i++) {
        int64_t band_i = band_indices[i];
        double freq_weight_i = (freqs0[band_i] > cutoff_frequency) ? 
                              1.0 / sqrt(freqs0[band_i]) : 0.0;
        
        if (freq_weight_i > 0.0) {
            /* Contract over atom/cartesian index l */
            for (j = 0; j < num_band; j++) {
                for (k = 0; k < num_band; k++) {
                    for (int64_t m = 0; m < num_band; m++) {
                        lapack_complex_double sum = zero;
                        for (int64_t l = 0; l < num_band; l++) {
                            /* fc3_reciprocal has shape [num_band, num_band, num_band, num_band, num_band] */
                            /* indexing: fc3[l, j, k, l, m] - note the coupling l,l */
                            int64_t fc3_idx = l * num_band * num_band * num_band * num_band +
                                             j * num_band * num_band * num_band +
                                             k * num_band * num_band +
                                             l * num_band + m;
                            
                            /* e0[band_i, l] includes mass weighting already */
                            lapack_complex_double e_val = lapack_make_complex_double(
                                lapack_complex_double_real(e0[band_i * num_band + l]) * freq_weight_i,
                                lapack_complex_double_imag(e0[band_i * num_band + l]) * freq_weight_i);
                            
                            lapack_complex_double prod = phonoc_complex_prod(e_val, fc3_reciprocal[fc3_idx]);
                            sum = lapack_make_complex_double(
                                lapack_complex_double_real(sum) + lapack_complex_double_real(prod),
                                lapack_complex_double_imag(sum) + lapack_complex_double_imag(prod));
                        }
                        /* Store in temp_ijkmn with shape [num_band0, num_band, num_band, num_band] */
                        temp_ijkmn[i * num_band * num_band * num_band + 
                                  j * num_band * num_band + k * num_band + m] = sum;
                    }
                }
            }
        }
    }

    /* Step 8: Contract e1 with intermediate: Ijkm,Jjm->IkJ
     * This is the second major contraction step
     */
#ifdef _OPENMP
#pragma omp parallel for if (!openmp_per_triplets)
#endif
    for (i = 0; i < num_band0; i++) {
        for (k = 0; k < num_band; k++) {
            for (int64_t band_j = 0; band_j < num_band; band_j++) {
                double freq_weight_j = (freqs1[band_j] > cutoff_frequency) ? 
                                      1.0 / sqrt(freqs1[band_j]) : 0.0;
                
                if (freq_weight_j > 0.0) {
                    lapack_complex_double sum = zero;
                    /* Contract over j,m indices - this is where the diagonal coupling occurs */
                    for (j = 0; j < num_band; j++) {
                        /* temp_ijkmn has j and m indices, contract with e1[band_j, j] for j==m */
                        lapack_complex_double temp_val = temp_ijkmn[i * num_band * num_band * num_band +
                                                                  j * num_band * num_band + 
                                                                  k * num_band + j]; /* m=j coupling */
                        
                        /* e1[band_j, j] includes mass weighting already */
                        lapack_complex_double e_val = lapack_make_complex_double(
                            lapack_complex_double_real(e1[band_j * num_band + j]) * freq_weight_j,
                            lapack_complex_double_imag(e1[band_j * num_band + j]) * freq_weight_j);
                        
                        lapack_complex_double prod = phonoc_complex_prod(temp_val, e_val);
                        sum = lapack_make_complex_double(
                            lapack_complex_double_real(sum) + lapack_complex_double_real(prod),
                            lapack_complex_double_imag(sum) + lapack_complex_double_imag(prod));
                    }
                    /* Store result with shape [num_band0, num_band, num_band] */
                    temp_iknj[i * num_band * num_band + k * num_band + band_j] = sum;
                }
            }
        }
    }

    /* Step 9: Final contraction with e2: IkJ,Kkn->IJK 
     * This produces the final result tensor
     */
#ifdef _OPENMP
#pragma omp parallel for if (!openmp_per_triplets)
#endif
    for (i = 0; i < num_g_pos; i++) {
        int64_t i0 = g_pos[i][0];
        int64_t i1 = g_pos[i][1];
        int64_t i2 = g_pos[i][2];
        int64_t idx_out = g_pos[i][3];
        
        if (freqs0[band_indices[i0]] > cutoff_frequency &&
            freqs1[i1] > cutoff_frequency &&
            freqs2[i2] > cutoff_frequency) {
            
            double freq_weight_k = 1.0 / sqrt(freqs2[i2]);
            lapack_complex_double sum = zero;
            
            /* Contract over k,n indices with k==n coupling */
            for (k = 0; k < num_band; k++) {
                /* temp_iknj has shape [i0, k, i1], contract with e2[i2, k] for k==n */
                lapack_complex_double temp_val = temp_iknj[i0 * num_band * num_band + 
                                                         k * num_band + i1];
                
                /* e2[i2, k] includes mass weighting already */
                lapack_complex_double e_val = lapack_make_complex_double(
                    lapack_complex_double_real(e2[i2 * num_band + k]) * freq_weight_k,
                    lapack_complex_double_imag(e2[i2 * num_band + k]) * freq_weight_k);
                
                lapack_complex_double prod = phonoc_complex_prod(temp_val, e_val);
                sum = lapack_make_complex_double(
                    lapack_complex_double_real(sum) + lapack_complex_double_real(prod),
                    lapack_complex_double_imag(sum) + lapack_complex_double_imag(prod));
            }
            
            /* Compute squared magnitude */
            double real_part = lapack_complex_double_real(sum);
            double imag_part = lapack_complex_double_imag(sum);
            fc3_normal_squared[idx_out] = real_part * real_part + imag_part * imag_part;
        } else {
            fc3_normal_squared[idx_out] = 0.0;
        }
    }

    free(temp_ijkmn);
    free(temp_iknj);
}

static void create_frequency_weighted_eigenvectors(
    lapack_complex_double *e_weighted, const lapack_complex_double *eigvecs,
    const double *freqs, const double *inv_sqrt_masses,
    const int64_t *band_indices, const int64_t num_bands_selected,
    const int64_t num_band, const double cutoff_frequency) {
    
    int64_t i, j;
    
    /* Steps 1-3 and 4-6 from einsum path: I,ilI->Iil and Iil,i->Iil */
    for (i = 0; i < num_bands_selected; i++) {
        int64_t band_idx = (band_indices != NULL) ? band_indices[i] : i;
        double freq_weight = (freqs[band_idx] > cutoff_frequency) ? 
                           1.0 / sqrt(freqs[band_idx]) : 0.0;
        
        for (j = 0; j < num_band; j++) {
            double mass_weight = inv_sqrt_masses[j];
            double total_weight = freq_weight * mass_weight;
            
            lapack_complex_double orig_val = eigvecs[band_idx * num_band + j];
            e_weighted[i * num_band + j] = lapack_make_complex_double(
                lapack_complex_double_real(orig_val) * total_weight,
                lapack_complex_double_imag(orig_val) * total_weight
            );
        }
    }
}

static void contract_fc3_optimized(
    lapack_complex_double *result, const lapack_complex_double *fc3_reciprocal,
    const lapack_complex_double *e0_weighted, const lapack_complex_double *e1_weighted,
    const lapack_complex_double *e2_weighted, const int64_t num_band0,
    const int64_t num_band1, const int64_t num_band2, const int64_t num_band) {
    
    int64_t i, j, k, l, m, n;
    lapack_complex_double *temp1, *temp2;
    lapack_complex_double zero, one;
    
    zero = lapack_make_complex_double(0.0, 0.0);
    one = lapack_make_complex_double(1.0, 0.0);
    
    /* Allocate temporary arrays for intermediate results */
    temp1 = (lapack_complex_double *)malloc(
        sizeof(lapack_complex_double) * num_band0 * num_band * num_band * num_band);
    temp2 = (lapack_complex_double *)malloc(
        sizeof(lapack_complex_double) * num_band0 * num_band * num_band2);
    
    /* Initialize result array */
    for (i = 0; i < num_band0 * num_band1 * num_band2; i++) {
        result[i] = zero;
    }
    
    /* Step 7: Iil,ijklmn->Ijkmn (contract first index with fc3) */
    /* This is the most computationally intensive step */
#ifdef MULTITHREADED_BLAS
    /* Use BLAS for efficient matrix multiplication when available */
    for (i = 0; i < num_band0; i++) {
        /* Reshape fc3 and contract with e0_weighted[i] */
        /* This implements: sum_l e0_weighted[i,l] * fc3[l,j,k,m,n] = temp1[i,j,k,m,n] */
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   1, num_band * num_band * num_band,
                   num_band, &one,
                   e0_weighted + i * num_band, num_band,
                   fc3_reciprocal, num_band * num_band * num_band,
                   &zero, temp1 + i * num_band * num_band * num_band,
                   num_band * num_band * num_band);
    }
#else
    /* Manual implementation when BLAS is not available */
    for (i = 0; i < num_band0; i++) {
        for (j = 0; j < num_band; j++) {
            for (k = 0; k < num_band; k++) {
                for (m = 0; m < num_band; m++) {
                    for (n = 0; n < num_band; n++) {
                        lapack_complex_double sum = zero;
                        for (l = 0; l < num_band; l++) {
                            int64_t fc3_idx = l * num_band * num_band * num_band * num_band +
                                             j * num_band * num_band * num_band +
                                             k * num_band * num_band +
                                             m * num_band + n;
                            lapack_complex_double prod = phonoc_complex_prod(
                                e0_weighted[i * num_band + l], fc3_reciprocal[fc3_idx]);
                            sum = lapack_make_complex_double(
                                lapack_complex_double_real(sum) + lapack_complex_double_real(prod),
                                lapack_complex_double_imag(sum) + lapack_complex_double_imag(prod));
                        }
                        temp1[i * num_band * num_band * num_band + j * num_band * num_band + 
                              k * num_band + m * num_band + n] = sum;
                    }
                }
            }
        }
    }
#endif

    /* Step 8: Ijkmn,Jjm->IknJ (contract second index) */
    for (i = 0; i < num_band0; i++) {
        for (k = 0; k < num_band; k++) {
            for (n = 0; n < num_band; n++) {
                for (j = 0; j < num_band2; j++) {
                    lapack_complex_double sum = zero;
                    for (l = 0; l < num_band; l++) { /* l represents the j and m indices being contracted */
                        lapack_complex_double temp_val = temp1[i * num_band * num_band * num_band +
                                                             l * num_band * num_band + k * num_band + l * num_band + n];
                        lapack_complex_double prod = phonoc_complex_prod(temp_val, e1_weighted[j * num_band + l]);
                        sum = lapack_make_complex_double(
                            lapack_complex_double_real(sum) + lapack_complex_double_real(prod),
                            lapack_complex_double_imag(sum) + lapack_complex_double_imag(prod));
                    }
                    temp2[i * num_band * num_band2 + k * num_band2 + j] = sum;
                }
            }
        }
    }

    /* Step 9: IknJ,Kkn->IJK (final contraction) */
    for (i = 0; i < num_band0; i++) {
        for (j = 0; j < num_band1; j++) {
            for (k = 0; k < num_band2; k++) {
                lapack_complex_double sum = zero;
                for (l = 0; l < num_band; l++) { /* l represents k and n indices being contracted */
                    lapack_complex_double temp_val = temp2[i * num_band * num_band2 + l * num_band2 + j];
                    lapack_complex_double prod = phonoc_complex_prod(temp_val, e2_weighted[k * num_band + l]);
                    sum = lapack_make_complex_double(
                        lapack_complex_double_real(sum) + lapack_complex_double_real(prod),
                        lapack_complex_double_imag(sum) + lapack_complex_double_imag(prod));
                }
                result[i * num_band1 * num_band2 + j * num_band2 + k] = sum;
            }
        }
    }

    free(temp1);
    free(temp2);
} 