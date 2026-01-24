#include "rsort.h"

void parallel_radix_sort(SortItem *src, SortItem *dst, uint N) {
    int max_threads = 1;
    #ifdef OMP
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
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            memset(histograms[tid], 0, RADIX_BUCKETS * sizeof(uint));

            #pragma omp for schedule(static)
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
                histograms[t][b] = total_count; // L'offset di scrittura per il thread t nel bucket b
                total_count += count;
            }
        }

        // 3. reorder particles
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
           
            #pragma omp for schedule(static)
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
}