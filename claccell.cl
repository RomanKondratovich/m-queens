/*
typedef struct __attribute__ ((packed)) {
    uint diagr;// bitfield with all the used diagonals down left
    uint diagl;// bitfield with all the used diagonals down right
} diags_packed_t;
*/

typedef uint2 diags_packed_t;

#define MAXN 29

typedef char int8_t;
typedef char int_fast8_t;
typedef uchar uint_fast8_t;
typedef short int_fast16_t;
typedef ushort uint_fast16_t;
typedef int int_fast32_t;
typedef uint uint_fast32_t;
#define UINT_FAST32_MAX UINT_MAX

// for convenience
#define L (get_local_id(0))
#define G (get_global_id(0))

#if 0
kernel void count_solutions(__global const diags_packed_t* lut, __global const diags_packed_t* candidates, __global uint* out_cnt, uint lut_offset) {
    uint cnt = 0;
    uint lut_diagr = lut[G + lut_offset].diagr;
    uint lut_diagl = lut[G + lut_offset].diagl;

    for(int i = 0; i < MAX_CANDIDATES; i++) {
        cnt += ((lut_diagr & candidates[i].diagr) == 0) && ((lut_diagl & candidates[i].diagl) == 0);
    }

    //printf("G: %d, L: %d, cnt: %d, lut_diagr: %x, lut_diagl: %x\n", G, L, cnt, lut_diagr, lut_diagl);

    out_cnt[G] += cnt;
}
#endif

#define LOCAL_FACTOR 32
// Depth 5
//#define LOCAL_MEM 128
// Depth 6
#define LOCAL_MEM 256
// Depth 7
//#define LOCAL_MEM 896
// Depth 8
//#define LOCAL_MEM 2944

kernel void count_solutions_trans(__global const diags_packed_t* restrict lut,
                                  __global const diags_packed_t* restrict candidates,
                                  __global uint* restrict out_cnt, uint lut_offset, uint lut_count) {
    uint cnt = 0;

    __local diags_packed_t l_lut[LOCAL_MEM];

    //initialize  shared local memory
    int lidx = get_local_id(0);
    int size_x = get_local_size(0);
    for(uint index = lidx; index < lut_count; index += size_x ) {
        l_lut[index] = lut[lut_offset + index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    uint gsize_x = get_global_size(0);
    uint gstart_x = (get_global_id(0) - lidx)*LOCAL_FACTOR;

    uint limit = gstart_x + size_x*LOCAL_FACTOR;
    //#pragma unroll 2
    for(int j = (gstart_x + lidx*16); j < limit; j += (size_x*16)) {
        diags_packed_t can_diag0 = candidates[j+0];
        diags_packed_t can_diag1 = candidates[j+1];
        diags_packed_t can_diag2 = candidates[j+2];
        diags_packed_t can_diag3 = candidates[j+3];
        diags_packed_t can_diag4 = candidates[j+4];
        diags_packed_t can_diag5 = candidates[j+5];
        diags_packed_t can_diag6 = candidates[j+6];
        diags_packed_t can_diag7 = candidates[j+7];
        diags_packed_t can_diag8 = candidates[j+8];
        diags_packed_t can_diag9 = candidates[j+9];
        diags_packed_t can_diagA = candidates[j+10];
        diags_packed_t can_diagB = candidates[j+11];
        diags_packed_t can_diagC = candidates[j+12];
        diags_packed_t can_diagD = candidates[j+13];
        diags_packed_t can_diagE = candidates[j+14];
        diags_packed_t can_diagF = candidates[j+15];


        for(int i = 0; i < lut_count; i++) {
            diags_packed_t lut_diag = l_lut[i];
            diags_packed_t tmp0 = lut_diag & can_diag0;
            diags_packed_t tmp1 = lut_diag & can_diag1;
            diags_packed_t tmp2 = lut_diag & can_diag2;
            diags_packed_t tmp3 = lut_diag & can_diag3;
            diags_packed_t tmp4 = lut_diag & can_diag4;
            diags_packed_t tmp5 = lut_diag & can_diag5;
            diags_packed_t tmp6 = lut_diag & can_diag6;
            diags_packed_t tmp7 = lut_diag & can_diag7;
            diags_packed_t tmp8 = lut_diag & can_diag8;
            diags_packed_t tmp9 = lut_diag & can_diag9;
            diags_packed_t tmpA = lut_diag & can_diagA;
            diags_packed_t tmpB = lut_diag & can_diagB;
            diags_packed_t tmpC = lut_diag & can_diagC;
            diags_packed_t tmpD = lut_diag & can_diagD;
            diags_packed_t tmpE = lut_diag & can_diagE;
            diags_packed_t tmpF = lut_diag & can_diagF;
            cnt += (tmp0.x == 0) && (tmp0.y == 0);
            cnt += (tmp1.x == 0) && (tmp1.y == 0);
            cnt += (tmp2.x == 0) && (tmp2.y == 0);
            cnt += (tmp3.x == 0) && (tmp3.y == 0);
            cnt += (tmp4.x == 0) && (tmp4.y == 0);
            cnt += (tmp5.x == 0) && (tmp5.y == 0);
            cnt += (tmp6.x == 0) && (tmp6.y == 0);
            cnt += (tmp7.x == 0) && (tmp7.y == 0);
            cnt += (tmp8.x == 0) && (tmp8.y == 0);
            cnt += (tmp9.x == 0) && (tmp9.y == 0);
            cnt += (tmpA.x == 0) && (tmpA.y == 0);
            cnt += (tmpB.x == 0) && (tmpB.y == 0);
            cnt += (tmpC.x == 0) && (tmpC.y == 0);
            cnt += (tmpD.x == 0) && (tmpD.y == 0);
            cnt += (tmpE.x == 0) && (tmpE.y == 0);
            cnt += (tmpF.x == 0) && (tmpF.y == 0);
        }
    }

    //printf("G: %d, L: %d, cnt: %d, lut_diagr: %x, lut_diagl: %x\n", G, L, cnt, lut_diagr, lut_diagl);

    out_cnt[G] += cnt;
}

kernel void count_solutions_trans_cleanup(__global const diags_packed_t* restrict lut,
                                  __global const diags_packed_t* restrict candidates,
                                  __global uint* restrict out_cnt, uint lut_offset, uint lut_count) {
    uint cnt = 0;

    __local diags_packed_t l_lut[LOCAL_MEM];

    //initialize  shared local memory
    int lidx = get_local_id(0);
    int size_x = get_local_size(0);
    for(uint index = lidx; index < lut_count; index += size_x ) {
        l_lut[index] = lut[lut_offset + index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    uint gsize_x = get_global_size(0);
    uint gstart_x = (get_global_id(0) - lidx);

    uint limit = gstart_x + size_x;
    //#pragma unroll 2
    for(int j = (gstart_x + lidx); j < limit; j += (size_x)) {
        diags_packed_t can_diag0 = candidates[j+0];

        for(int i = 0; i < lut_count; i++) {
            diags_packed_t lut_diag = l_lut[i];
            diags_packed_t tmp0 = lut_diag & can_diag0;
            cnt += (tmp0.x == 0) && (tmp0.y == 0);
        }
    }

    //printf("G: %d, L: %d, cnt: %d, lut_diagr: %x, lut_diagl: %x\n", G, L, cnt, lut_diagr, lut_diagl);

    out_cnt[G] += cnt;
}
