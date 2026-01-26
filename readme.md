# Particle Mesh code optimization

## Vectorization

- simplified operations in loops 
    - precalculate all constants
    - fast_fmod, fast_mod
    - preferred multiplications to divisions
- Aligned memory
    - use of posix_memalign(...) in init
    - use of assume_aligned in pm.c (interpolate_forces, estimate_density, ...)
- unrolled short loops
    - inner loops of interpolate forces
    - inner loops of estimate density
- reduced memory access

- used directives to hint the compiler
    - #pragma GCC unroll
    - #pragma omp simd

-  cache locality
    - sorting particles every 10-20 iterations

## Using Threads

- easily parallelized independent loops:
    #pragma omp parallel for

- managed atomic update in estimate density

- increased data locality with touch first policy

## Using GPUs

- #pragma kernels as baseline

- 