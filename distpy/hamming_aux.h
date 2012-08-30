#ifndef HAMMING_AUX_H
#define HAMMING_AUX_H
#include <stdint.h>
void hamdist_cmap_lut16(uint8_t *xs, uint8_t *ys, uint32_t *out, const int size, const int num_xs, const int num_ys);
#endif
