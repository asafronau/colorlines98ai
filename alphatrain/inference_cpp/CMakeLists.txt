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
# (Includes mcts.cc for the LegalPriors cross-check against the golden masks.)
add_executable(game_test src/game_test.cc src/game.cc src/obs.cc src/mcts.cc
                         src/feature_value.cc)

# Feature-value evaluator golden test — no LibTorch needed.
add_executable(feature_test src/feature_test.cc src/feature_value.cc src/game.cc)

add_executable(mcts_controls_test src/mcts_controls_test.cc src/mcts.cc
                                 src/game.cc src/obs.cc src/feature_value.cc)

# Full policy eval: C++ game loop + LibTorch batched inference.
add_executable(eval src/eval.cc src/game.cc src/obs.cc)
target_link_libraries(eval PRIVATE "${TORCH_LIBRARIES}")

# MCTS eval: feature-value MCTS over a seed range (game threads + one shared
# inference-server thread).
add_executable(mcts_eval src/mcts_eval.cc src/mcts.cc src/game.cc src/obs.cc
                         src/feature_value.cc)
target_link_libraries(mcts_eval PRIVATE "${TORCH_LIBRARIES}")

# MCTS selfplay: records visit distributions in the moves-schema JSON that
# build_expert_v2_tensor consumes.
add_executable(mcts_selfplay src/mcts_selfplay.cc src/mcts.cc src/game.cc
                             src/obs.cc src/feature_value.cc)
target_link_libraries(mcts_selfplay PRIVATE "${TORCH_LIBRARIES}")

# Crisis mining: greedy probe to death -> rewind anchors -> deep-MCTS replay
# (the decisive escape-or-die games; port of crisis_mining.py).
add_executable(mcts_crisis src/mcts_crisis.cc src/mcts.cc src/game.cc
                           src/obs.cc src/feature_value.cc)
target_link_libraries(mcts_crisis PRIVATE "${TORCH_LIBRARIES}")

# Teacher-strength probe: search the first W moves from saved anchors, then greedy.
add_executable(anchor_search src/anchor_search.cc src/mcts.cc src/game.cc
                             src/obs.cc src/feature_value.cc)
target_link_libraries(anchor_search PRIVATE "${TORCH_LIBRARIES}")

# Rollout judge: teacher-move vs base-move on correction states (died-within-H).
add_executable(rollout_judge src/rollout_judge.cc src/mcts.cc src/game.cc
                             src/obs.cc src/feature_value.cc)
target_link_libraries(rollout_judge PRIVATE "${TORCH_LIBRARIES}")

add_executable(mcts_relabel src/mcts_relabel.cc src/mcts.cc src/game.cc
               src/obs.cc src/feature_value.cc)
target_link_libraries(mcts_relabel PRIVATE "${TORCH_LIBRARIES}")

add_executable(obs_dump src/obs_dump.cc src/game.cc src/obs.cc)

# CPU cost + bit-exactness of eval's per-step game work, old vs fast path (HISTORY 257).
add_executable(eval_cpu_bench src/eval_cpu_bench.cc src/game.cc src/obs.cc)
target_link_libraries(eval_cpu_bench PRIVATE "${TORCH_LIBRARIES}")

# Inference round-trip timing (forward vs logits readback) for the InferenceServer.
add_executable(infer_bench src/infer_bench.cc)
target_link_libraries(infer_bench PRIVATE "${TORCH_LIBRARIES}")
