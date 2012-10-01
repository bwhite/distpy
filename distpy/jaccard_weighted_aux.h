#ifndef JACCARD_WEIGHTED_AUX_H
#define JACCARD_WEIGHTED_AUX_H
#include <stdint.h>
void make_lut16_chunk(double *weights, double *chunk);
void jaccard_weighted_cmap_lut16(uint8_t *xs, uint8_t *ys, double *out, const int size, const int num_xs, const int num_ys, const double *chunks, const int num_chunks);
#endif
