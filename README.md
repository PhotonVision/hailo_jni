# Building

Building the main project:
```
cmake -Bcmake_build -S . -DCMAKE_INSTALL_PREFIX=cmake_build -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
cmake --build cmake_build
```
You will also need to add `-DOPENCV_ARCH=platform` for whatevere platform you're building for.

Building hailort:
```
cmake . -Bbuild -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
sudo cmake --build build --target install
```

To cross compile to install `gcc-aarch64-linux-gnu` and `gfortran-aarch64-linux-gnu` and add `-DCMAKE_TOOLCHAIN_FILE=path/to/hailo_jni/arm64-toolchain.cmake` to both builds. Along with changing the install directory of the hailort to `/usr/aarch64-linux-gnu`.
