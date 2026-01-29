#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "global.h"

// Thrust needs a comparator to sort the SortItem
struct SortItemComparator {
    __host__ __device__
    bool operator()(const SortItem& a, const SortItem& b) {
        return a.cell_index < b.cell_index;
    }
};

extern "C" void gpu_sort(SortItem *d_data, unsigned int N) {
    // 1. "Cast" the C raw pointer to a Thrust pointer for device
    thrust::device_ptr<SortItem> d_ptr(d_data);

    // 2. Call the sort.
    thrust::sort(thrust::device, d_ptr, d_ptr + N, SortItemComparator());
}