/* special.c - standalone implementations of special math used by EMMIXskew
   All functions expose Fortran-friendly symbols with trailing underscore. */

#include "rshim.h"
#include "special.h"

/* ---------- Normal PDF ---------- */
double mydnorm_(const double *z) {
    const double x = *z;
    return 0.39894228040143267794 /* 1/sqrt(2*pi) */ * exp(-0.5 * x * x);
}

/* ---------- Normal CDF via erf ---------- */
double mvphin_(const double *z) {
    const double x = *z;
    return 0.5 * (1.0 + erf(x / M_SQRT2));
}

/* ---------- lgamma wrapper ---------- */
double mygammln_(const double *x) {
    return lgamma(*x);
}

/* ---------- digamma (psi) approximation ----------
   Recurrence to shift x>7, then asymptotic expansion. */
static double digamma_core(double x) {
    double r = 0.0;
    while (x < 7.0) { r -= 1.0/x; x += 1.0; }
    double x2 = 1.0/(x*x);
    /* Asymptotic series: psi(x) ~ log(x) - 1/(2x) - sum(B2k/(2k x^{2k})) */
    r += log(x) - 0.5/x
       - x2*(1.0/12.0 - x2*(1.0/120.0 - x2*(1.0/252.0 - x2*(1.0/240.0))));
    return r;
}
double mydigamma_(const double *xptr) {
    double x = *xptr;
    if (isnan(x)) return NAN;
    /* Simple reflection not needed for x>0 in our use; guard small values */
    if (x <= 0.0) return NAN;
    return digamma_core(x);
}

/* ---------- regularized incomplete beta I_x(a,b) ----------
   Continued fraction (NR/Cephes-like) implementation. */
static double betaln(double a, double b) {
    return lgamma(a) + lgamma(b) - lgamma(a + b);
}
static double betacf(double a, double b, double x) {
    const int maxit = 200;
    const double eps = 1e-12;
    const double fpmin = DBL_MIN/eps;

    double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0, d = 1.0 - qab * x / qap;
    if (fabs(d) < fpmin) d = fpmin;
    d = 1.0/d;
    double h = d;

    for (int m=1; m<=maxit; ++m) {
        int m2 = 2*m;
        double aa = (m*(b - m) * x)/((qam + m2)*(a + m2));
        d = 1.0 + aa * d; if (fabs(d) < fpmin) d = fpmin; d = 1.0/d;
        c = 1.0 + aa / c; if (fabs(c) < fpmin) c = fpmin;
        h *= d * c;

        aa = -((a + m)*(qab + m) * x)/((a + m2)*(qap + m2));
        d = 1.0 + aa * d; if (fabs(d) < fpmin) d = fpmin; d = 1.0/d;
        c = 1.0 + aa / c; if (fabs(c) < fpmin) c = fpmin;
        double del = d * c;
        h *= del;
        if (fabs(del - 1.0) < eps) break;
    }
    return h;
}
static double ibeta_reg(double a, double b, double x) {
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    double bt = exp(a*log(x) + b*log(1.0 - x) - betaln(a,b));
    if (x < (a+1.0)/(a+b+2.0))
        return bt * betacf(a,b,x) / a;
    else
        return 1.0 - bt * betacf(b,a,1.0 - x) / b;
}

/* ---------- Student-t CDF ----------
   For x>=0: F_t(x;nu) = 1 - 0.5 * I_{nu/(nu+x^2)}(nu/2, 1/2)
   For x<0:  symmetry. */
double mvphit_(const double *t, const double *nu) {
    const double x  = *t;
    const double df = *nu;
    if (!(df > 0.0)) return NAN;
    if (x == 0.0) return 0.5;
    double z = df / (df + x*x);
    double ib = ibeta_reg(0.5*df, 0.5, z);
    double c  = 0.5 * ib;
    return (x > 0.0) ? (1.0 - c) : c;
}

/* ---------- simple max over an array (used by gettau) ---------- */
void nonzeromax_(const double *x, const int *n, double *out) {
    int m = *n;
    double v = x[0];
    for (int i=1; i<m; ++i) if (x[i] > v) v = x[i];
    *out = v;
}
