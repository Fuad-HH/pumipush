 cmake -B build-cuda -S . \
    -Dpumipic_ROOT=$pumipic \
    -DOmega_h_ROOT=$oh \
    -DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
    -DCMAKE_BUILD_TYPE=Release \
    -DPP_USE_GPU=ON \
    -Dperfstubs_DIR=/lore/hasanm4/otherTools/perfstubs/lib64/cmake \
    -DADIOS2_ROOT=$ADIOS2_RHEL9_ROOT \
    -Dredev_ROOT=/lore/hasanm4/otherTools/redev \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1
 #-DCMAKE_CXX_COMPILER=$root/kokkos/bin/nvcc_wrapper \
 # -DCMAKE_CXX_COMPILER=CC \
