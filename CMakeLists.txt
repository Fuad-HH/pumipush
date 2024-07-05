cmake_minimum_required(VERSION 3.20)
project(pumipush LANGUAGES CXX)

set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
# set the pumipic root directory
#set(pumipic /lore/hasanm4/wsources/pumirelated/build-engpar-life-cuda/install)
#list(APPEND CMAKE_PREFIX_PATH $pumipic)
#list(APPEND CMAKE_PREFIX_PATH $oh)
find_package(pumipic REQUIRED)
find_package(Omega_h REQUIRED)
find_package(redev REQUIRED)
find_package(MPI REQUIRED)

add_executable(pumipush pumipush.cpp)
add_executable(pseudoMC pseudoMC.cpp)
#target_link_libraries(pseudoMC PUBLIC pumipic::pumipic Omega_h::omega_h redev::redev MPI::MPI_CXX)
target_link_libraries(pseudoMC PUBLIC pumipic::pumipic Omega_h::omega_h MPI::MPI_CXX)
target_link_libraries(pumipush PUBLIC pumipic::pumipic)
target_link_libraries(pumipush PUBLIC Omega_h::omega_h)
