#ifndef HAMMING_AUX_H
#define HAMMING_AUX_H
#include <stdint.h>
void make_lut16bit(uint8_t *lut16bit);
void hamdist_cmap_lut16(uint16_t *xs, uint16_t *ys, int32_t *out, const int size_by_2, const int num_xs, const int num_ys, const uint8_t *lut16bit);
#endif
