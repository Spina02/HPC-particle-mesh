#include "rsort.h"
#include <string.h>

void parallel_radix_sort(SortItem *src, SortItem *dst, uint N) {
    int max_threads = 1;
    #ifdef USE_OMP
    max_threads = omp_get_max_threads();
    #endif

    // allocate histograms for each thread
    uint **histograms = (uint**) malloc(max_threads * sizeof(uint*));
    for(int i=0; i<max_threads; i++) {
        // zero-initialization with calloc
        histograms[i] = (uint*) calloc(RADIX_BUCKETS, sizeof(uint)); 
    }

    SortItem *curr_src = src;
    SortItem *curr_dst = dst;

    // Loop over the bits of the cell index (32 bits)
    for (int shift = 0; shift < 32; shift += RADIX_BITS) {
        
        // 1: compute histograms
        #if defined(USE_OMP)
        #pragma omp parallel
        #endif
        {
            #if defined(USE_OMP)
            int tid = omp_get_thread_num();
            #else
            int tid = 0;
            #endif
            memset(histograms[tid], 0, RADIX_BUCKETS * sizeof(uint));

            #if defined(USE_OMP)
            #pragma omp for schedule(static)
            #endif
            for(uint i = 0; i < N; i++) {
                uint key = curr_src[i].cell_index;
                uint bucket = (key >> shift) & RADIX_MASK;
                histograms[tid][bucket]++;
            }
        }

        // 2: compute prefix sums
        uint total_count = 0;

        for (int b = 0; b < RADIX_BUCKETS; b++) {
            for (int t = 0; t < max_threads; t++) {
                uint count = histograms[t][b];
                histograms[t][b] = total_count; // Write offset for thread t in bucket b
                total_count += count;
            }
        }

        // 3. reorder particles
        #if defined(USE_OMP)
        #pragma omp parallel
        #endif
        {
            #if defined(USE_OMP)
            int tid = omp_get_thread_num();
            #else
            int tid = 0;
            #endif
           
            #if defined(USE_OMP)
            #pragma omp for schedule(static)
            #endif
            for(uint i = 0; i < N; i++) {
                uint key = curr_src[i].cell_index;
                uint bucket = (key >> shift) & RADIX_MASK;
                
                // get the offset and increment it
                uint dst_idx = histograms[tid][bucket]++;
                curr_dst[dst_idx] = curr_src[i];
            }
        }
        
        // 4. swap src and dst
        SortItem *tmp = curr_src;
        curr_src = curr_dst;
        curr_dst = tmp;
    }

    // free histograms
    for(int i=0; i<max_threads; i++) {
        free(histograms[i]);
    }
    free(histograms);
}