#ifndef HAMMING_AUX_H
#define HAMMING_AUX_H
#include <stdint.h>
int hamdist8(uint8_t *x, uint8_t *y, const int num_ints);
int hamdist16(uint16_t *x, uint16_t *y, const int num_ints);
int hamdist32(uint32_t *x, uint32_t *y, const int num_ints);
int hamdist64(uint64_t *x, uint64_t *y, const int num_ints);
void hamdist_batch(uint8_t *xs, void *y, int32_t *out, const int num_ints, const int num_xs, const int num_bytes, int (*hamdist)(void *, void *, int));
#endif
