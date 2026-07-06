cmake_minimum_required(VERSION 3.18)
project(clines_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# LibTorch ships *inside* your Python venv's torch package — no separate
# download. Point CMake at it on the command line (see README):
#   cmake -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/../../.venv/bin/python")
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
        "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

find_package(Torch REQUIRED)

add_executable(infer src/main.cc)
target_link_libraries(infer PRIVATE "${TORCH_LIBRARIES}")

# Pure game-engine golden test — no LibTorch needed, builds in a second.
add_executable(game_test src/game_test.cc src/game.cc src/obs.cc)

# Feature-value evaluator golden test — no LibTorch needed.
add_executable(feature_test src/feature_test.cc src/feature_value.cc src/game.cc)

# Full policy eval: C++ game loop + LibTorch batched inference.
add_executable(eval src/eval.cc src/game.cc src/obs.cc)
target_link_libraries(eval PRIVATE "${TORCH_LIBRARIES}")
