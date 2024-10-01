#include <mutex>
#include <absl/synchronization/mutex.h>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <iostream>


void check1() {

    std::mutex mu;
    std::cout << "Size of std::mutex: " << (int) sizeof(mu) << std::endl;

    absl::Mutex mu2;
    std::cout << "Size of absl::Mutex: " << (int) sizeof(mu2) << std::endl;

}

// ------ shard index by Nanobind

inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= (uint64_t) 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= (uint64_t) 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return k;
}

inline size_t shard_index_nanobind(uintptr_t p, size_t shard_mask) {
    uintptr_t highbits = ((uintptr_t) p) >> 20;
    size_t index = ((size_t) fmix64((uint64_t) highbits)) & shard_mask;
    return index;
}

// ------- shard index by Pybind11

inline std::uint64_t mix64(std::uint64_t z) {
    // David Stafford's variant 13 of the MurmurHash3 finalizer popularized
    // by the SplitMix PRNG.
    // https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}


inline size_t shard_index_pybind11(uintptr_t addr, size_t instance_shards_mask) {
    auto hash = mix64(static_cast<std::uint64_t>(addr >> 20));
    auto idx = static_cast<size_t>(hash & instance_shards_mask);
    return idx;
}


void check2() {

    uintptr_t p1 = 0x7b2800029ff0;
    uintptr_t p2 = 0x7b280002ff90;

    size_t shard_mask = 31;

    std::cout << "Nanobind: thread 1, pointer:" << (int) p1 << " -> index: " << (int) shard_index_nanobind(p1, shard_mask) << std::endl;
    std::cout << "Nanobind: thread 2, pointer:" << (int) p2 << " -> index: " << (int) shard_index_nanobind(p2, shard_mask) << std::endl;

    std::cout << "PyBind11: thread 1, pointer:" << (int) p1 << " -> index: " << (int) shard_index_pybind11(p1, shard_mask) << std::endl;
    std::cout << "PyBind11: thread 2, pointer:" << (int) p2 << " -> index: " << (int) shard_index_pybind11(p2, shard_mask) << std::endl;


}



int main() {

    // check1();
    check2();


    return 0;
}