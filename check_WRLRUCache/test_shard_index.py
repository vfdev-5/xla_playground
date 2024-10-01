

## Nanobind:
"""
inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= (uint64_t) 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= (uint64_t) 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return k;
}


inline nb_shard &shard(void *p) {
    uintptr_t highbits = ((uintptr_t) p) >> 20;
    size_t index = ((size_t) fmix64((uint64_t) highbits)) & shard_mask;
    return shards[index];
}

"""
def fmix64(k):
    k ^= k >> 33
    k *= int(0xff51afd7ed558ccd)
    k ^= k >> 33
    k *= int(0xc4ceb9fe1a85ec53)
    k ^= k >> 33
    return k


def nanobind_shard_index(ptr, shard_mask):
    highbits = ptr >> 20
    index = fmix64(highbits) & shard_mask
    return ptr, highbits, index


## PyBind11
"""
"""

def run_checks():
    # shard: p: 0x7b2800029ff0, index: 23, shard_mask: 31 | thread: 139938756261440
    # shard: p: 0x7b280002ff90, index: 23, shard_mask: 31 | thread: 139938746070592

    shard_mask = 31
    ptr1 = 0x7b2800029ff0
    ptr2 = 0x7b280002ff90

    print(nanobind_shard_index(ptr1, shard_mask))
    print(nanobind_shard_index(ptr2, shard_mask))



run_checks()
