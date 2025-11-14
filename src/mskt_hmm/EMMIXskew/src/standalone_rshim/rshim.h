/* rshim.h - minimal replacements for R headers to build standalone
   This header provides small macro/type shims so that sources that used
   <R.h> / R allocation routines can compile without linking to R. */

#ifndef EMMIX_STANDALONE_RSHIM_H
#define EMMIX_STANDALONE_RSHIM_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <errno.h>

/* R-like typedefs used occasionally */
typedef int R_len_t;

/* Memory allocation shims (R's Calloc/Free semantics) */
#define Calloc(n, type)    ((type*)calloc((n), sizeof(type)))
#define Realloc(ptr, n, type) ((type*)realloc((ptr), (n) * sizeof(type)))
#define Free(p)            free((p))

/* Printing & error shims */
#define Rprintf(...)       do { printf(__VA_ARGS__); } while(0)
/* Map Rf_error() or error() to a hard abort; adjust if you prefer errno returns */
#define error(...)         do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); abort(); } while(0)

/* Some math constants if missing */
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

/* BLAS expects Fortran naming; not needed here but harmless */
#ifndef F77_CALL
#  define F77_CALL(x) x##_
#endif
#ifndef F77_NAME
#  define F77_NAME(x) x##_
#endif
#ifndef F77_SUB
#  define F77_SUB(x)  x##_
#endif

#endif /* EMMIX_STANDALONE_RSHIM_H */
