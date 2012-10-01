#include "jaccard_weighted_aux.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>


void make_lut16_chunk(double *weights, double *chunk) {
    /*
      Args:
          weights: 16 values of weights
          chunk: 2**16 values zeroed out
      Bits are numbered (w.r.t. weights), we keep the byte order but flip the bits
      Order:   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
      Original:[7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]
      
     */
    uint32_t i;
    uint16_t j;
    for (i = 0; i < 65536; ++i) {
        for (j = 0; j < 16; ++j)
            if ((i >> j) & 1)
                chunk[i] += weights[j / 8 * 8 + (7 - j % 8)];
    }
}

void jaccard_weighted_cmap_lut16_even(uint16_t *xs, uint16_t *ys, double *out, const int size_by_2, const int num_xs, const int num_ys, const double *chunks, const int num_chunks) {
    int i, j, k, l;
    uint16_t *ys_orig = ys;
    double *chunk;
    for (i = 0; i < num_xs; ++i, xs += size_by_2)
        for (j = 0, ys = ys_orig; j < num_ys; ++j, ++out, ys += size_by_2) {
            for (k = 0, chunk = chunks; k < size_by_2; ++k, chunk += 65536) {
                *out += chunk[xs[k] & ys[k]];
            }
        }
}

void jaccard_weighted_cmap_lut16(uint8_t *xs, uint8_t *ys, double *out, const int size, const int num_xs, const int num_ys, const double *chunks, const int num_chunks) {
    assert(size % 2 == 0);
    jaccard_weighted_cmap_lut16_even((uint16_t *)xs, (uint16_t *)ys, out, size / 2, num_xs, num_ys, chunks, num_chunks);
}
