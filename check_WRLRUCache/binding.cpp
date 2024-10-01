#include <nanobind/nanobind.h>

#include "weakref_lru_cache.h"


NB_MODULE(wrlru_cache_ext, m) {
    jax::BuildWeakrefLRUCacheAPI(m);
}
