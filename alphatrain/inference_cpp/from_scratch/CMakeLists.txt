cmake_minimum_required(VERSION 3.16)
project(clines_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)  # -O2; matters for the conv loops
endif()

# Prefer a prebuilt abseil (e.g. `brew install abseil`) — links in seconds.
# Fall back to fetching+compiling it from source if not installed (~minutes once).
find_package(absl QUIET)
if(NOT absl_FOUND)
  message(STATUS "abseil not found via find_package; fetching from source")
  include(FetchContent)
  FetchContent_Declare(
    absl
    URL https://github.com/abseil/abseil-cpp/releases/download/20240722.0/abseil-cpp-20240722.0.tar.gz
  )
  set(ABSL_PROPAGATE_CXX_STD ON)
  set(ABSL_ENABLE_INSTALL OFF)
  FetchContent_MakeAvailable(absl)
endif()

add_executable(infer src/main.cc src/net.cc)
target_include_directories(infer PRIVATE src)
target_link_libraries(infer PRIVATE
  absl::flat_hash_map
  absl::status
  absl::statusor
  absl::strings
  absl::str_format
)
