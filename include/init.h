#ifndef INIT_H
#define INIT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "global.h"

// function to read the parameters from a file
Params* read_params(int argc, char** argv);

real_t compute_Gprime(NormalizationParams* norm);

Particles* init_particles(Params* params);
void place_particles(Particles* particles, vec2d_t BoxSize, real_t A_deltaPar);
int destroy_particles(Particles* particles);

Mesh* init_mesh(vec2_t Ngrid);
int destroy_mesh(Mesh* mesh);

static inline void* allocate_aligned(size_t size) {
    #ifdef ALIGNED
    void* ptr;
    if (posix_memalign(&ptr, ALIGNMENT, size) != 0) {
            return NULL;
        }
        return ptr;
    #else
    return malloc(size);
    #endif
}


#endif // INIT_H
