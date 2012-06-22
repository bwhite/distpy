#ifndef HAMMING_AUX_H
#define HAMMING_AUX_H
#include <stdint.h>
void make_lut16bit(uint8_t *lut16bit);
int hamdist8(uint8_t *x, uint8_t *y, const int num_ints);
int hamdist16(uint16_t *x, uint16_t *y, const int num_ints);
int hamdist32(uint32_t *x, uint32_t *y, const int num_ints);
int hamdist64(uint64_t *x, uint64_t *y, const int num_ints);
void hamdist_batch(uint8_t *xs, void *y, int32_t *out, const int num_ints, const int num_xs, const int num_bytes, int (*hamdist)(void *, void *, int));
void hamdist_cmap_lut16(uint16_t *xs, uint16_t *ys, int32_t *out, const int size_by_2, const int num_xs, const int num_ys, const uint8_t *lut16bit);
void hamdist_cmap_lut16_odd(uint8_t *xs, uint8_t *ys, int32_t *out, const int size_by_2, const int num_xs, const int num_ys, const uint8_t *lut16bit);
#endif
