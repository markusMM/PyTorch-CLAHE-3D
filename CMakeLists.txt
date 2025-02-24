cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(torch_clahe)

# lib dirs
set(Torch_DIR ${Torch_PATH}/share/cmake/Torch)

# requirements
find_package(Torch REQUIRED)

# Define our library target
add_library(compute_clahe SHARED torch_clahe.cpp)

# Enable C++14
target_compile_features(torch_clahe PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(torch_clahe ${LIBTORCH})
