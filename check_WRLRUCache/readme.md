# Multi-threading tests and TSAN for XLA weakref_lru_cache

## Copy files here

```bash
XLA=/tmp/jax/xla
cp $XLA/xla/python/weakref_lru_cache.h .
cp $XLA/xla/python/weakref_lru_cache.cc .
cp $XLA/xla/pjrt/lru_cache.h .
```


Make code updates:
```diff
- #include "xla/python/weakref_lru_cache.h"
+ #include "weakref_lru_cache.h"

- #include "xla/pjrt/lru_cache.h"
+ #include "lru_cache.h"


- #include "tsl/platform/logging.h"
+ // #include "tsl/platform/logging.h"
```


## Install deps

- absl-cpp: https://abseil.io/docs/cpp/quickstart-cmake
```bash
git clone https://github.com/abseil/abseil-cpp.git
cd abseil-cpp
mkdir build && cd build
# Without TSAN
# cmake -DABSL_BUILD_TESTING=OFF -DCMAKE_CXX_STANDARD=14 -DABSL_BUILD_MONOLITHIC_SHARED_LIBS=ON -DBUILD_SHARED_LIBS=ON ..
# With TSAN
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-15 \
    -DCMAKE_CXX_COMPILER=clang++-15 \
    -DCMAKE_CXX_FLAGS="-fsanitize=thread" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" \
    -DCMAKE_MODULE_LINKER_FLAGS="-fsanitize=thread" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=thread" \
    -DABSL_BUILD_TESTING=OFF -DCMAKE_CXX_STANDARD=14 \
    -DABSL_BUILD_MONOLITHIC_SHARED_LIBS=ON -DBUILD_SHARED_LIBS=ON \
    -DABSL_ENABLE_INSTALL=ON \
    -DCMAKE_INSTALL_PREFIX=/tmp/abseil ..

cmake --build . --target install
```

### Nanobind

```bash
cd /tmp/jax/nanobind

python3.13t -mpip install -vvv -e .
```


## Build the WRLRUCache extension

```bash
mkdir build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Debug -DPython_EXECUTABLE=$(which python3.13t) -DUSE_TSAN=ON \
    -DCMAKE_C_COMPILER=clang-15 \
    -DCMAKE_CXX_COMPILER=clang++-15 \
    -DCMAKE_PREFIX_PATH=/tmp/abseil \
    ..
cmake --build .


export TSAN_SYMBOLIZER_PATH=$(which llvm-symbolizer)

# If built with GCC/G++
# LD_PRELOAD=/lib/x86_64-linux-gnu/libtsan.so.0 python3.13t -c "import wrlru_cache_ext"
# LD_PRELOAD=/lib/x86_64-linux-gnu/libtsan.so.0 python3.13t -c "from wrlru_cache_ext import weakref_lru_cache, WeakrefLRUCache"

# If built with Clang
# LD_PRELOAD=$(clang++-15 -print-file-name=libtsan.so) python3.13t -c "import wrlru_cache_ext"

python3.13t -c "import wrlru_cache_ext"
```

## Tests

```bash
# PYTHONPATH=build LD_PRELOAD=/lib/x86_64-linux-gnu/libtsan.so.0 python -m pytest tests.py -vvv
# PYTHON_GIL=1 PYTHONPATH=build LD_PRELOAD=/lib/x86_64-linux-gnu/libtsan.so.0 python tests.py

# PYTHON_GIL=0 PYTHONPATH=build LD_PRELOAD=/lib/x86_64-linux-gnu/libtsan.so.0 python tests.py

PYTHON_GIL=0 PYTHONPATH=build python3.13t tests.py
```

