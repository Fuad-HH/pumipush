 cmake -B build -S . \
    -Dpumipic_ROOT=/lore/hasanm4/wsources/pumirelated/build-pumipic-life-cuda/install \
    -DOmega_h_ROOT=/lore/hasanm4/wsources/pumirelated/build-omegah-life-cuda/install \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1