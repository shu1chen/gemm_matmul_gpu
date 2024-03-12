# gemm_matmul bench on GPU
Inspired by [gemmBench on CPU](https://github.com/shu1chen/gemmBench). Dependencies: [oneDNN](https://github.com/oneapi-src/oneDNN) and [XeTLA](https://https://github.com/intel/xetla)

## Update submodule
```
git submodule update --init --recursive
```

## Use MKL as the BLAS vendor in oneDNN
```
source /opt/intel/oneapi/mkl/latest/env/vars.sh
```
Then add `set(DNNL_BLAS_VENDOR MKL CACHE INTERNAL "" FORCE)` in gemm_matmul_gpu/CMakeList.txt and build.


## Compilation
```
mkdir build
cd build
cmake -DWITH_MKL=OFF ..
make -j
```

If you have [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) installed on your system, you can set `-DWITH_MKL=ON` during the CMake configuration.

## Usage
```
./gemm_matmul_gpu [iterations=1000]
```
