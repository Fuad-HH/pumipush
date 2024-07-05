 cmake -B build -S . \
    -Dpumipic_ROOT=/lore/hasanm4/wsources/pumirelated/build-pumipic-life-cuda/install \
    -DOmega_h_ROOT=/lore/hasanm4/wsources/pumirelated/build-omegah-life-cuda/install \
    -DCMAKE_BUILD_TYPE=Debug \
    -Dperfstubs_DIR=/lore/hasanm4/otherTools/perfstubs/lib64/cmake \
    -DADIOS2_ROOT=$ADIOS2_RHEL9_ROOT \
    -Dredev_ROOT=/lore/hasanm4/otherTools/redev \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1
