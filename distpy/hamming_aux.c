#include "hamming_aux.h"

void make_lut16bit(uint8_t *lut16bit) {
    int i, val;
    for (i = 0; i < 65536; ++i) {
        val = i;
        while(val) {
            ++lut16bit[i]; 
            val &= val - 1;
        }
    }  
}

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

void hamdist_cmap_lut16(uint16_t *xs, uint16_t *ys, int32_t *out, const int size_by_2, const int num_xs, const int num_ys, const uint8_t *lut16bit) {
    int i, j, k;
    uint16_t *ys_orig = ys;
    for (i = 0; i < num_xs; ++i, xs += size_by_2)
        for (j = 0, ys = ys_orig; j < num_ys; ++j, ++out, ys += size_by_2) {
            for (k = 0; k < size_by_2; ++k) {
                *out += lut16bit[xs[k] ^ ys[k]];
            }
        }
}
