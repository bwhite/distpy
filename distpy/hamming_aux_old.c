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
