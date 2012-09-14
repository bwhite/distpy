void hamdist_cmap_popcount(uint8_t *xs, uint8_t *ys, uint32_t *out, const int size, const int num_xs, const int num_ys) {
    int i, j, k;
    uint8_t *ys_orig = ys;
    const int size_by_8 = size / 8;
    assert(sizeof(unsigned long long) == 8);
    assert(size % 8 == 0);
    for (i = 0; i < num_xs; ++i, xs += size)
        for (j = 0, ys = ys_orig; j < num_ys; ++j, ++out, ys += size) {
            for (k = 0; k < size_by_8; ++k) {
                *out += __builtin_popcountll(((uint64_t *) xs)[k] ^ ((uint64_t *) ys)[k]);
            }
        }
}


// Older implementations kept for future use or reference
int hamdist8(uint8_t *x, uint8_t *y, const int num_ints) {
    uint8_t val;
    int dist = 0, i;
    for (i = 0; i < num_ints; ++i) {
        val = x[i] ^ y[i];
        while(val) {
            ++dist; 
            val &= val - 1;
        }
    }
    return dist;
}

int hamdist16(uint16_t *x, uint16_t *y, const int num_ints) {
    uint16_t val;
    int dist = 0, i;
    for (i = 0; i < num_ints; ++i) {
        val = x[i] ^ y[i];
        while(val) {
            ++dist; 
            val &= val - 1;
        }
    }
    return dist;
}


int hamdist32(uint32_t *x, uint32_t *y, const int num_ints) {
    uint32_t val;
    int dist = 0, i;
    for (i = 0; i < num_ints; ++i) {
        val = x[i] ^ y[i];
        while(val) {
            ++dist; 
            val &= val - 1;
        }
    }
    return dist;
}


int hamdist64(uint64_t *x, uint64_t *y, const int num_ints) {
    uint64_t val;
    int dist = 0, i;
    for (i = 0; i < num_ints; ++i) {
        val = x[i] ^ y[i];
        while(val) {
            ++dist; 
            val &= val - 1;
        }
    }
    return dist;
}

void hamdist_batch(uint8_t *xs, void *y, int *out, const int num_ints, const int num_xs, const int num_bytes, int (*hamdist)(void *, void *, int)) {
    int i;
    for (i = 0; i < num_xs; ++i, xs += num_bytes)
        out[i] = hamdist((void *)xs, (void *)y, num_ints);
}


// From https://github.com/norouzi/mih/blob/master/include/bitops.h: Modified for comparison, slower when tested compared to our method
#define popcntll __builtin_popcountll
#define popcnt __builtin_popcount

#include <stdio.h>
#include <math.h>


const int lookup [] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8};

int match(uint8_t *P, uint8_t *Q, int codelb) {
    int output = 0;
    int i;
    switch(codelb) {
    case 4: // 32 bit
        return popcnt(*(uint32_t*)P ^ *(uint32_t*)Q);
        break;
    case 8: // 64 bit
        return popcntll(((uint64_t*)P)[0] ^ ((uint64_t*)Q)[0]);
        break;
    case 16: // 128 bit
        return popcntll(((uint64_t*)P)[0] ^ ((uint64_t*)Q)[0]) \
            + popcntll(((uint64_t*)P)[1] ^ ((uint64_t*)Q)[1]);
        break;
    case 32: // 256 bit
        return popcntll(((uint64_t*)P)[0] ^ ((uint64_t*)Q)[0]) \
            + popcntll(((uint64_t*)P)[1] ^ ((uint64_t*)Q)[1]) \
            + popcntll(((uint64_t*)P)[2] ^ ((uint64_t*)Q)[2]) \
            + popcntll(((uint64_t*)P)[3] ^ ((uint64_t*)Q)[3]);
        break;
    case 64: // 512 bit
        return popcntll(((uint64_t*)P)[0] ^ ((uint64_t*)Q)[0]) \
            + popcntll(((uint64_t*)P)[1] ^ ((uint64_t*)Q)[1]) \
            + popcntll(((uint64_t*)P)[2] ^ ((uint64_t*)Q)[2]) \
            + popcntll(((uint64_t*)P)[3] ^ ((uint64_t*)Q)[3]) \
            + popcntll(((uint64_t*)P)[4] ^ ((uint64_t*)Q)[4]) \
            + popcntll(((uint64_t*)P)[5] ^ ((uint64_t*)Q)[5]) \
            + popcntll(((uint64_t*)P)[6] ^ ((uint64_t*)Q)[6]) \
            + popcntll(((uint64_t*)P)[7] ^ ((uint64_t*)Q)[7]);
        break;
    default:
        for (i=0; i<codelb; i++)
            output+= lookup[P[i] ^ Q[i]];
        return output;
        break;
    }

    return -1;
}


void hamdist_cmap(uint8_t *xs, uint8_t *ys, uint32_t *out, const int size, const int num_xs, const int num_ys) {
    int i, j;
    for (i = 0; i < num_xs; ++i)
        for (j = 0; j < num_ys; ++j, out++) {
            *out = match(xs + size * i, ys + size * j, size);
        }
}
