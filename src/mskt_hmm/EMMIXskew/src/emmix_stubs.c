/* emmix_stubs.c
   Standalone replacements for a few R API routines. 
   IMPORTANT: these must be non-static so the linker sees the symbols. */

#include <math.h>
#include <float.h>
#include <stddef.h>

/* --- already present in your file (keep them) ---
   double digamma(double x) { ... }
   double pnorm5(double x, double mu, double sd, int lower_tail, int log_p) { ... }
   double pt(double x, double nu, int lower_tail, int log_p) { ... }
   double dnorm4(double x, double mu, double sd, int give_log) { ... }
   void   revsort(double *a, int *ib, int n) { ... }
   void   nonzeromax_(const double *v, const int *p, double *vmax) { ... }
   (etc.)
*/

/* ---- FOR LINKER: provide R_zeroin2 with NON-STATIC linkage ----
   Signature must match what module.c calls:
   R_zeroin2(ax, bx, fa, fb, f, info, Tol, Maxit)
   returns the root as a double.
*/
double R_zeroin2(double ax, double bx,
                 double fa, double fb,
                 double (*f)(double, void*),
                 void *info,
                 double Tol,
                 int *Maxit)
{
    /* Robust Brent’s method fallback (simplified):
       - requires fa*f(ax) and fb=f(bx) of opposite signs
       - stops if interval width < Tol or iterations exhausted
    */
    double a = ax, b = bx, c = bx;
    double fa_ = fa;
    double fb_ = fb;
    double fc = fb_;
    double d = 0.0, e = 0.0;

    if (fa_ == 0.0) return a;
    if (fb_ == 0.0) return b;
    if (fa_ * fb_ > 0.0) {
        /* bracket not valid: fall back to bisection until we get opposite sign or iterations end */
        for (int it = 0; it < (*Maxit > 0 ? *Maxit : 50); ++it) {
            double m = 0.5*(a + b);
            double fm = f(m, info);
            if (fm == 0.0) return m;
            if (fa_ * fm < 0.0) { b = m; fb_ = fm; }
            else                { a = m; fa_ = fm; }
            if (fabs(b - a) <= Tol) return 0.5*(a + b);
        }
        return 0.5*(a + b);
    }

    for (int it = 0; it < (*Maxit > 0 ? *Maxit : 100); ++it) {
        if (fb_ * fc > 0.0) { c = a; fc = fa_; e = d = b - a; }
        if (fabs(fc) < fabs(fb_)) { a = b; b = c; c = a; fa_ = fb_; fb_ = fc; fc = fa_; }

        double tol1 = 2.0*DBL_EPSILON*fabs(b) + 0.5*Tol;
        double xm = 0.5*(c - b);
        if (fabs(xm) <= tol1 || fb_ == 0.0) {
            *Maxit = it;
            return b;
        }

        if (fabs(e) >= tol1 && fabs(fa_) > fabs(fb_)) {
            /* Inverse quadratic interpolation / secant */
            double s = fb_/fa_;
            double p, q;
            if (a == c) { p = 2.0*xm*s; q = 1.0 - s; }
            else {
                double q1 = fa_/fc, r = fb_/fc;
                p = s*(2.0*xm*q1*(q1 - r) - (b - a)*(r - 1.0));
                q = (q1 - 1.0)*(r - 1.0)*(s - 1.0);
            }
            if (p > 0) q = -q; else p = -p;

            if (2.0*p < fmin(3.0*xm*q - fabs(tol1*q), fabs(e*q))) {
                e = d; d = p/q;
            } else {
                d = xm; e = d;
            }
        } else {
            d = xm; e = d;
        }

        a = b; fa_ = fb_;
        if (fabs(d) > tol1) b += d;
        else                b += (xm > 0 ? tol1 : -tol1);

        fb_ = f(b, info);
    }

    /* If we get here, iterations exhausted: return best current estimate */
    return b;
}
