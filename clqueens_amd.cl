typedef struct __attribute__ ((packed)) {
    uint cols; // bitfield with all the used columns
    uint diagl;// bitfield with all the used diagonals down left
    uint diagr;// bitfield with all the used diagonals down right
    //uint dummy;// dummy to align at 128bit
} start_condition;

#define MAXN 29

typedef char int_fast8_t;
typedef uchar uint_fast8_t;
typedef short int_fast16_t;
typedef ushort uint_fast16_t;
typedef int int_fast32_t;
typedef uint uint_fast32_t;
#define UINT_FAST32_MAX UINT_MAX

typedef ulong uint_fast64_t;

#define MAXD (N - PLACED)

#if (MAXD > 13)
#error "Depth to high, risk of overflow in result counter"
#endif

kernel void solve_subboard(__global const start_condition* in_starts, __global ulong* out_cnt) {
    size_t id = get_global_id(0);

    // counter for the number of solutions
    // sufficient until n=29

    // 32 bits are sufficient if the rest depth is <= 13
    uint_fast32_t num = 0;
    __private uint4 stack[MAXD+1]; // cols -> s0, posibs -> s1, diagl -> s2, diagr -> s3
    //__private uint_fast32_t cols[MAXD], posibs[MAXD]; // Our backtracking 'stack'
    //__private uint_fast32_t diagl[MAXD], diagr[MAXD];
    __local int_fast8_t rest[MAXD]; // number of rows left
    int_fast16_t d = 0; // d is our depth in the backtrack stack
    // The UINT_FAST32_MAX here is used to fill all 'coloumn' bits after n ...
    stack[d].s0 = in_starts[id].cols | (UINT_FAST32_MAX << N);
    // This places the first two queens
    stack[d].s2 = in_starts[id].diagl;
    stack[d].s3 = in_starts[id].diagr;
    #define LOOKAHEAD 3
    // we're allready two rows into the field here
    rest[d] = N - LOOKAHEAD - PLACED;//in_starts[id].placed;

    //  The variable posib contains the bitmask of possibilities we still have
    //  to try in a given row ...
    uint_fast32_t posib = (stack[d].s0 | stack[d].s2 | stack[d].s3);

    while (d >= 0) {
      // moving the two shifts out of the inner loop slightly improves
      // performance
      uint_fast32_t diagl_shifted = stack[d].s2 << 1;
      uint_fast32_t diagr_shifted = stack[d].s3 >> 1;
      int_fast8_t l_rest = rest[d];
      uint_fast32_t l_cols = stack[d].s0;


      while (posib != UINT_FAST32_MAX) {
        // The standard trick for getting the rightmost bit in the mask
        uint_fast32_t bit = ~posib & (posib + 1);
        uint_fast32_t new_diagl = (bit << 1) | diagl_shifted;
        uint_fast32_t new_diagr = (bit >> 1) | diagr_shifted;
        uint_fast32_t new_posib = (l_cols | bit | new_diagl | new_diagr);
        posib ^= bit; // Eliminate the tried possibility.
        bit |= l_cols;

        if (new_posib != UINT_FAST32_MAX) {
            uint_fast32_t lookahead1 = (bit | (new_diagl << (LOOKAHEAD - 2)) | (new_diagr >> (LOOKAHEAD - 2)));
            uint_fast32_t lookahead2 = (bit | (new_diagl << (LOOKAHEAD - 1)) | (new_diagr >> (LOOKAHEAD - 1)));
            uint_fast8_t allowed1 = l_rest >= 0;
            uint_fast8_t allowed2 = l_rest > 0;


            if(allowed1 && (lookahead1 == UINT_FAST32_MAX)) {
                continue;
            }

            if(allowed2 && (lookahead2 == UINT_FAST32_MAX)) {
                continue;
            }

            // The next two lines save stack depth + backtrack operations
            // when we passed the last possibility in a row.
            // Go lower in the stack, avoid branching by writing above the current
            // position
            stack[d + 1].s1 = posib;
            d++;//= posib != UINT_FAST32_MAX; // avoid branching with this trick
            posib = new_posib;

            l_rest--;
            //allowed1 = l_rest >= 0;
            //allowed2 = l_rest > 0;

            // make values current
            l_cols = bit;
            stack[d].s0 = bit;
            stack[d].s2 = new_diagl;
            stack[d].s3 = new_diagr;
            rest[d] = l_rest;
            diagl_shifted = new_diagl << 1;
            diagr_shifted = new_diagr >> 1;
        } else {
            // when all columns are used, we found a solution
            num += bit == UINT_FAST32_MAX;
        }
      }
      posib = stack[d].s1; // backtrack ...
      d--;
    }

    out_cnt[id] += num;
}
