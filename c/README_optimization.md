# Optimized C Implementation of Phonon Reciprocal to Normal Transformation

## Overview

This directory contains an optimized C implementation (`reciprocal_to_normal_fast.c`) that follows the einsum optimization path discovered by `opt_einsum` for the tensor contraction:

```
ilI,jmJ,knK,ijklmn,i,j,k,I,J,K -> IJK
```

## Optimization Strategy

The original C code uses a naive nested-loop approach with scaling order 9. The optimized version follows the einsum path that reduces this to scaling order 7, providing a theoretical speedup of ~25,820x.

### Einsum Optimization Path

```
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   3              0             I,ilI->Iil     jmJ,knK,ijklmn,i,j,k,J,K,Iil->IJK
   3              0             J,jmJ->Jjm       knK,ijklmn,i,j,k,K,Iil,Jjm->IJK
   3              0             K,knK->Kkn         ijklmn,i,j,k,Iil,Jjm,Kkn->IJK
   3              0             Iil,i->Iil           ijklmn,j,k,Jjm,Kkn,Iil->IJK
   3              0             Jjm,j->Jjm             ijklmn,k,Kkn,Iil,Jjm->IJK
   3              0             Kkn,k->Kkn               ijklmn,Iil,Jjm,Kkn->IJK
   7           TDOT      Iil,ijklmn->Ijkmn                    Jjm,Kkn,Ijkmn->IJK
   6           TDOT        Ijkmn,Jjm->IknJ                         Kkn,IknJ->IJK
   5           TDOT          IknJ,Kkn->IJK                              IJK->IJK
```

### Implementation Steps

1. **Steps 1-6**: Create frequency and mass weighted eigenvectors
   - Contract frequencies with eigenvectors: `I,ilI->Iil`, `J,jmJ->Jjm`, `K,knK->Kkn`
   - Apply mass weightings: `Iil,i->Iil`, `Jjm,j->Jjm`, `Kkn,k->Kkn`

2. **Step 7**: First major tensor contraction (`Iil,ijklmn->Ijkmn`)
   - Contract first eigenvector index with fc3 tensor
   - Reduces from 6D to 5D tensor
   - Uses BLAS TDOT operations when available

3. **Step 8**: Second major contraction (`Ijkmn,Jjm->IknJ`)
   - Contract second eigenvector index
   - Reduces from 5D to 4D tensor

4. **Step 9**: Final contraction (`IknJ,Kkn->IJK`)
   - Contract third eigenvector index
   - Produces final 3D result tensor

## Performance Comparison

- **Original C code**: ~O(N^9) scaling with manual nested loops
- **Optimized C code**: ~O(N^7) scaling following einsum path
- **Theoretical speedup**: 25,820x for the test case (42x42x42 system)
- **Memory efficiency**: Largest intermediate only 1.95M elements

## Key Features

- **OpenMP parallelization**: Parallel loops where beneficial
- **BLAS integration**: Ready for BLAS TDOT operations when MULTITHREADED_BLAS is available
- **Memory efficient**: Minimal intermediate storage following optimal path
- **Frequency cutoff handling**: Proper handling of bands below cutoff frequency

## Usage

```c
#include "reciprocal_to_normal_fast.h"

void reciprocal_to_normal_squared_fast(
    double *fc3_normal_squared, const int64_t (*g_pos)[4],
    const int64_t num_g_pos, const lapack_complex_double *fc3_reciprocal,
    const double *freqs0, const double *freqs1, const double *freqs2,
    const lapack_complex_double *eigvecs0,
    const lapack_complex_double *eigvecs1,
    const lapack_complex_double *eigvecs2, const double *masses,
    const int64_t *band_indices, const int64_t num_band0,
    const int64_t num_band, const double cutoff_frequency,
    const int64_t openmp_per_triplets);
```

This optimized implementation should provide significantly better performance than the original, especially for larger phonon systems where the O(N^7) vs O(N^9) scaling difference becomes substantial. 