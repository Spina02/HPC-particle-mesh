#ifndef TIMING_H
#define TIMING_H

#include <time.h>

/* ·········································································
 *
 *  CPU TIME for process
 */

// return process cpu time
static inline double pcpu_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// return process cpu time with a long double
static inline long double pcpu_time_l(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (long double)ts.tv_sec + (long double)ts.tv_nsec * 1e-9;
}

/* ·········································································
 *
 *  CPU TIME for thread
 */

// return thread cpu time
static inline double tcpu_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// return thread cpu time with a long double
static inline long double tcpu_time_l(void) {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (long double)ts.tv_sec + (long double)ts.tv_nsec * 1e-9;
}

/* ·········································································
 *
 *  return the number of nanoseconds between two different points,
 *  processing two timespec structures
 *  returns a single unsigned long long
 */

// both tstart and tstop are struct timespec
// for instance returned by clock_gettime
static inline unsigned long long get_deltat(struct timespec tstart, struct timespec tstop) {
    unsigned long long sec = (tstop.tv_sec - tstart.tv_sec) * 1000000000ULL;
    unsigned long long nsec = 1000000000ULL - tstart.tv_nsec + tstop.tv_nsec;
    return sec + nsec;
}

/* Backwards compatibility macros (now call inline functions) */
#define PCPU_TIME pcpu_time()
#define PCPU_TIME_L pcpu_time_l()
#define TCPU_TIME tcpu_time()
#define TCPU_TIME_L tcpu_time_l()
#define GET_DELTAT(TSTART, TSTOP) get_deltat((TSTART), (TSTOP))

#endif /* TIMING_H */
