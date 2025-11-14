#ifndef EMMIX_BLAS_LAPACK_H
#define EMMIX_BLAS_LAPACK_H

#include "rshim.h" /* defines F77_NAME/F77_SUB */
#ifdef __cplusplus
extern "C" {
#endif

/* BLAS Level-1/2 used by the C code */
void F77_NAME(dcopy)(const int *N, const double *X, const int *incX,
                           double *Y, const int *incY);
void F77_NAME(daxpy)(const int *N, const double *alpha, const double *X,
                     const int *incX, double *Y, const int *incY);
double F77_NAME(ddot)(const int *N, const double *X, const int *incX,
                      const double *Y, const int *incY);
void F77_NAME(dgemv)(const char *trans, const int *M, const int *N,
                     const double *alpha, const double *A, const int *lda,
                     const double *X, const int *incX,
                     const double *beta, double *Y, const int *incY);

/* BLAS Level-2/3 + LAPACK used by module.c */
void F77_NAME(dtrsv)(const char *uplo, const char *trans, const char *diag,
                     const int *N, const double *A, const int *lda,
                     double *X, const int *incX);

void F77_NAME(dscal)(const int *N, const double *alpha, double *X, const int *incX);

/* LAPACK cholesky */
void F77_NAME(dpotrf)(const char *uplo, const int *N, double *A, const int *lda, int *info);

#ifdef __cplusplus
}
#endif
#endif /* EMMIX_BLAS_LAPACK_H */
