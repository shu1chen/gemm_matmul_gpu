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
./gemm_matmul_gpu [iterations=1000] [arch=any] [align=64]
```
align - specifies the alignment (bytes). Must be a valid alignment (valid for aligned_alloc, align > 32)
