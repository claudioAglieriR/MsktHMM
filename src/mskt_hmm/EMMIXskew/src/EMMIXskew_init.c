#include <R_ext/RS.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME:
   Check these declarations against the C/Fortran source code.
*/

/* .C calls */
extern void ddmix2(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void emskewfit1(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void emskewfit2(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void initfit_(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void interdist_(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void intradist_(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void mahalonobis_(void *, void *, void *, void *, void *, void *);

/* .Fortran calls */
extern void F77_NAME(inverse3)(void *, void *, void *, void *, void *, void *, void *);

static const R_CMethodDef CEntries[] = {
    {"ddmix2",       (DL_FUNC) &ddmix2,       11},
    {"emskewfit1",   (DL_FUNC) &emskewfit1,   32},
    {"emskewfit2",   (DL_FUNC) &emskewfit2,   28},
    {"initfit_",     (DL_FUNC) &initfit_,     27},
    {"interdist_",   (DL_FUNC) &interdist_,   10},
    {"intradist_",   (DL_FUNC) &intradist_,   10},
    {"mahalonobis_", (DL_FUNC) &mahalonobis_,  6},
    {NULL, NULL, 0}
};

static const R_FortranMethodDef FortranEntries[] = {
    {"inverse3", (DL_FUNC) &F77_NAME(inverse3), 7},
    {NULL, NULL, 0}
};

void R_init_EMMIXskew(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, NULL, FortranEntries, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
