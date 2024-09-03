# TT-Division

Small demo of doing division in TT-NN and TT-Metal

In python TTNN:
```python
z_ttnn = ttnn.div(x_ttnn, y_ttnn) # >0.50.0 TT-Metal versions only
```

Result:
```python
# CPU
tensor([2.], dtype=torch.bfloat16)

# TTNN
ttnn.Tensor([[ 1.99219,  0.00000,  ...,  0.00000,  0.00000],
             [ 0.00000,  0.00000,  ...,  0.00000,  0.00000],
             ...,
             [ 0.00000,  0.00000,  ...,  0.00000,  0.00000],
             [ 0.00000,  0.00000,  ...,  0.00000,  0.00000]], shape=Shape([1[32], 1[32]]), dtype=DataType::BFLOAT16, layout=Layout::TILE)
```

Result (TT-Metal):
```
Result: 2
Expected: 2
Test Passed
```

\* There is a bug where it says `Result: 7.96875`, not sure why but when you run it again it will go back to 2. For some reason c_intermed0 switches between 1.99 and 0.50 when it should be empty.

## How to run TT-Metal
```sh
# 1. Make sure you have set the right environment variables and built TT-Metal
export ARCH_NAME=<your device name (e.g "grayskull")>
export TT_METAL_HOME=/your/path/to/tt-metal

# If you haven't built TT-Metal, follow this:
# https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md

# 2. Build project
cd ./TT-Metal
mkdir build

cd build
cmake ..

make
./tt-division

# Or if you want to run with debug
TT_METAL_DPRINT_CORES=0,0 TT_METAL_DPRINT_FILE=log.txt ./tt-division
cat ./log.txt
```

![Osaka Thousand Year Stare](https://i.redd.it/b064yxmkl0zb1.jpg)
