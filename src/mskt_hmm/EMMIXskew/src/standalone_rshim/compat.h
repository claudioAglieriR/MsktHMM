#ifndef EMMIX_COMPAT_H
#define EMMIX_COMPAT_H

#include <math.h>

/* R’s lgammafn → libc’s lgamma */
#ifndef lgammafn
#  define lgammafn lgamma
#endif

/* R’s min/max helpers */
#ifndef imin2
#  define imin2(a,b) (( (a) < (b) ) ? (a) : (b))
#endif

#ifndef fmin2
#  define fmin2(a,b) fmin((a),(b))
#endif

#ifndef fmax2
#  define fmax2(a,b) fmax((a),(b))
#endif

/* Constants R exposes, define if missing */
#ifndef M_LN_SQRT_2PI
#  define M_LN_SQRT_2PI 0.918938533204672741780329736406 /* log(sqrt(2*pi)) */
#endif

#ifndef M_LN_SQRT_PI
#  define M_LN_SQRT_PI  0.572364942924700087071713675677 /* log(sqrt(pi))   */
#endif

#endif /* EMMIX_COMPAT_H */
