#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

// uncomment to start with n=2 and compare to known results
//#define TESTSUITE

#ifndef N
#define N 15
#define MAXN N
#else
#define MAXN 29
#endif


// parallel factor
#ifndef P_FACT
#define P_FACT 1
#endif

#if N > 29
#warning "N too big, overflow may occur"
#endif

#if N < 2
#error "N too small"
#endif

// get the current wall clock time in seconds
double get_time() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec + tp.tv_usec / 1000000.0;
}


uint_fast32_t diagl_shifted[P_FACT];
uint_fast32_t diagr_shifted[P_FACT];
uint_fast32_t old_cols[P_FACT];
uint_fast32_t new_cols[P_FACT];
uint_fast32_t new_diagl[P_FACT];
uint_fast32_t new_diagr[P_FACT];
uint_fast32_t new_posib[P_FACT];
//  The variable posib contains the bitmask of possibilities we still have
//  to try in a given row ...
uint_fast32_t posib[P_FACT];

uint_fast32_t cols[MAXN][P_FACT], posibs[MAXN][P_FACT]; // Our backtracking 'stack'
uint_fast32_t diagl[MAXN][P_FACT], diagr[MAXN][P_FACT];
int_fast8_t d[P_FACT] = {0}; // d is our depth in the backtrack stack
int_fast32_t old_posib[P_FACT];

uint64_t nqueens(uint_fast8_t n) {

  // counter for the number of solutions
  // sufficient until n=29
  uint_fast64_t num = 0;
  //
  // The top level is two fors, to save one bit of symmetry in the enumeration
  // by forcing second queen to be AFTER the first queen.
  //
  uint_fast16_t num_starts = ((n - 2) * (n - 2) * (n - 1)) / 2;

  // ensure start_queens is a multiple of P_FACT
  uint_fast16_t array_size = num_starts + P_FACT - (num_starts % P_FACT);
  uint_fast16_t start_cnt = 0;
  uint_fast32_t start_queens[array_size][2];
  for (uint_fast8_t q0 = 0; q0 < n - 2; q0++) {
    for (uint_fast8_t q1 = q0 + 2; q1 < n; q1++) {
      start_queens[start_cnt][0] = 1 << q0;
      start_queens[start_cnt][1] = 1 << q1;
      start_cnt++;
    }
  }


//#pragma omp parallel for reduction(+ : num) schedule(dynamic)
  for (uint_fast16_t cnt = 0; cnt < start_cnt; ) {
    // initialization loop
    // should be 100% vectorised
    for(uint_fast8_t p = 0; p < P_FACT; p++) {
        d[p] = 0;
        uint_fast32_t bit0 = start_queens[cnt + p][0]; // The first queen placed
        uint_fast32_t bit1 = start_queens[cnt + p][1]; // The second queen placed
        cols[d[p]][p] = bit0 | bit1 | (UINT_FAST32_MAX << n);
        // The next two lines are done with different algorithms, this somehow
        // improves performance a bit...
        diagl[d[p]][p] = (bit0 << 2) | (bit1 << 1);
        diagr[d[p]][p] = (bit0 >> 2) | (bit1 >> 1);
        diagl_shifted[p] = diagl[d[p]][p] << 1;
        diagr_shifted[p] = diagr[d[p]][p] >> 1;
        old_cols[p] = cols[d[p]][p];

        if(cnt + p < start_cnt)
        {
            posib[p] = ~(cols[d[p]][p] | diagl[d[p]][p] | diagr[d[p]][p]);
        }
        else
        {
            posib[p] = 0;
            d[p] = 0;
        }
    }
    cnt += P_FACT;

    int_fast32_t work = P_FACT;
    while(work) {
        #pragma omp simd reduction(+:num)
        for(uint_fast8_t p = 0; p < P_FACT; p++) {
            old_posib[p] = !posib[p]; //? 0 : UINT_FAST8_MAX;
            uint_fast32_t bit = posib[p] & (~posib[p] + (uint_fast32_t)1);
            new_cols[p] = old_cols[p] | bit;
            new_diagl[p] = (bit << 1) | diagl_shifted[p];
            new_diagr[p] = (bit >> 1) | diagr_shifted[p];
            new_posib[p] = ~(new_cols[p] | new_diagl[p] | new_diagr[p]);
            posib[p] ^= bit; // Eliminate the tried possibility.
        }

        for(uint_fast8_t p = 0; p < P_FACT; p++) {
            if (old_posib[p] && (d[p] > 0)) {
                posib[p] = posibs[d[p]][p];
                d[p]--;
                diagl_shifted[p] = diagl[d[p]][p] << 1;
                diagr_shifted[p] = diagr[d[p]][p] >> 1;
                old_cols[p] = cols[d[p]][p];
            } else if (old_posib[p]) {
                if(cnt < start_cnt) {
                    uint_fast32_t bit0 = start_queens[cnt][0]; // The first queen placed
                    uint_fast32_t bit1 = start_queens[cnt][1]; // The second queen placed
                    cols[d[p]][p] = bit0 | bit1 | (UINT_FAST32_MAX << n);
                    // The next two lines are done with different algorithms, this somehow
                    // improves performance a bit...
                    diagl[d[p]][p] = (bit0 << 2) | (bit1 << 1);
                    diagr[d[p]][p] = (bit0 >> 2) | (bit1 >> 1);
                    posib[p] = ~(cols[d[p]][p] | diagl[d[p]][p] | diagr[d[p]][p]);
                    diagl_shifted[p] = diagl[d[p]][p] << 1;
                    diagr_shifted[p] = diagr[d[p]][p] >> 1;
                    old_cols[p] = cols[d[p]][p];
                    cnt++;
                } else if(d[p] < 0){
                } else {
                    posib[p] = 0;
                    d[p] = -1;
                    work--;
                }
            } else if (new_posib[p]) {
              // Go lower in the stack, avoid branching by writing above the current
              // position
              posibs[d[p] + 1][p] = posib[p];

              // The next two lines save stack depth + backtrack operations
              // when we passed the last possibility in a row.
              d[p] += posib[p] != 0; // avoid branching with this trick

              // make values current
              posib[p] = new_posib[p];
              cols[d[p]][p] = new_cols[p];
              diagl[d[p]][p] = new_diagl[p];
              diagr[d[p]][p] = new_diagr[p];

              diagl_shifted[p] = new_diagl[p] << 1;
              diagr_shifted[p] = new_diagr[p] >> 1;
              old_cols[p] = cols[d[p]][p];
            } else {
                // when all columns are used, we found a solution
                num += new_cols[p] == UINT_FAST32_MAX;
            }
        }
    }
  }
  return num * 2;
}

// expected results from https://oeis.org/A000170
uint64_t results[27] = {1ULL,
                        0ULL,
                        0ULL,
                        2ULL,
                        10ULL,
                        4ULL,
                        40ULL,
                        92ULL,
                        352ULL,
                        724ULL,
                        2680ULL,
                        14200ULL,
                        73712ULL,
                        365596ULL,
                        2279184ULL,
                        14772512ULL,
                        95815104ULL,
                        666090624ULL,
                        4968057848ULL,
                        39029188884ULL,
                        314666222712ULL,
                        2691008701644ULL,
                        24233937684440ULL,
                        227514171973736ULL,
                        2207893435808352ULL,
                        22317699616364044ULL,
                        234907967154122528ULL};

int main(int argc, char **argv) {

#ifdef TESTSUITE
  int i = 2;
#else
  int i = N;
#endif
  if (argc == 2) {
    i = atoi(argv[1]);
    if (i < 1 || i > MAXN) {
      printf("n must be between 2 and %d!\n", MAXN);
    }
  }

  for (; i <= N; i++) {
    double time_diff, time_start; // for measuring calculation time
    time_start = get_time();
    uint64_t result = nqueens((uint8_t)i);
    time_diff = (get_time() - time_start); // calculating time difference
    result == results[i - 1] ? printf("PASS ") : printf("FAIL ");
    printf("N %2d, Solutions %18" PRIu64 ", Expected %18" PRIu64
           ", Time %fs, Solutions/s %f\n",
           i, result, results[i - 1], time_diff, result / time_diff);
  }
  return 0;
}
