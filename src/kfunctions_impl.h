// src/kfunctions_impl.h
#ifndef KFUNCTIONS_IMPL_H
#define KFUNCTIONS_IMPL_H

#if defined(KFUNC_KIC) && defined(KFUNC_LT)
#error "Build with exactly one objective macro: KFUNC_KIC or KFUNC_LT."
#endif

#if !defined(KFUNC_KIC) && !defined(KFUNC_LT)
#error "Build with exactly one objective macro: KFUNC_KIC or KFUNC_LT."
#endif

#if defined(KFUNC_REVENUE) || defined(KFUNC_SENSOR_ENTROPY)
#error "Revenue and sensor objectives are disabled in this draft. Use KIC or LT."
#endif

#if defined(KFUNC_KIC)
#include "objectvalue/kic.h"
#elif defined(KFUNC_LT)
#include "objectvalue/lt.h"
#endif

#endif // KFUNCTIONS_IMPL_H
