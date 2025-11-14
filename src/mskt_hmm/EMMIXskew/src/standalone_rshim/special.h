/* special.h - prototypes for math special functions & Fortran bridges */
#ifndef EMMIX_STANDALONE_SPECIAL_H
#define EMMIX_STANDALONE_SPECIAL_H

/* Fortran-callable symbols (trailing underscore) */
double mygammln_(const double *x);             /* = lgamma(*x) */
double mydigamma_(const double *x);            /* digamma(*x)  */
double mvphin_(const double *z);               /* N(0,1) CDF   */
double mvphit_(const double *t, const double *nu); /* t_nu CDF */
double mydnorm_(const double *z);              /* N(0,1) pdf   */
void nonzeromax_(const double* v, const int *p, double *vmax);

#endif
