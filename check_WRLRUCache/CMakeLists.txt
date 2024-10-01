cmake_minimum_required(VERSION 3.16)

project(weakref_lru_cache_checks)

# https://nanobind.readthedocs.io/en/latest/building.html


if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(USE_TSAN "Compile with TSAN" OFF)

function(append value)
  foreach(variable ${ARGN})
    set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
  endforeach(variable)
endfunction()

if (USE_TSAN)
  append("-fsanitize=thread -g -O2" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  append("-fsanitize=thread" CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

find_package(nanobind CONFIG REQUIRED)

# Abseil requires C++14
set(CMAKE_CXX_STANDARD 14)

# Process Abseil's CMake build system
find_package(absl REQUIRED)

# Compile extension module with size optimization and add 'wrlru_cache_lib'
nanobind_add_module(
    wrlru_cache_ext
    FREE_THREADED
    binding_all.cpp
)
#    binding.cpp
#    weakref_lru_cache.cc weakref_lru_cache.h lru_cache.h

set_target_properties(wrlru_cache_ext PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(wrlru_cache_ext PRIVATE absl::strings absl::node_hash_map absl::base absl::synchronization absl::cleanup)


add_executable(side_main side_main.cpp)
target_link_libraries(side_main absl::synchronization)
