#ifndef RSORT_H
#define RSORT_H

#include "global.h"

#ifdef USE_OMP
#include <omp.h>
#endif

#define RADIX_BITS 8
#define RADIX_BUCKETS (1 << RADIX_BITS)
#define RADIX_MASK (RADIX_BUCKETS - 1)

void parallel_radix_sort(SortItem *src, SortItem *dst, uint N);

#endif // RSORT_H
