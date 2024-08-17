# TT-Division

```python
  File "/host_data/projects/TT-Division/simple.py", line 15, in <module>
    z_3 = x_ttnn / y_ttnn
TypeError: unsupported operand type(s) for /: 'tt_lib.tensor.Tensor' and 'tt_lib.tensor.Tensor'
```

Ever seen this before? Can't figure out why Tenstorrent doesn't have a division operator? Look no further! I got a solution for you.

Simply multiply just multiply `x` by the reciprocol of `y`
$$
z = x * \frac{1}{y} = \frac{x}{y}
$$

In python TTNN:
```python
def ttnn_div(x: ttnn.Tensor, y: ttnn.Tensor) -> ttnn.Tensor:
    # x * (1 / y) = x / y
    # return ttnn.multiply(x, ttnn.reciprocal(y))
    z_0 = ttnn.reciprocal(y)
    z_1 = ttnn.multiply(x, z_0)
    return z_1
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

Reciprocal: https://docs.tenstorrent.com/ttnn/latest/ttnn/ttnn/reciprocal.html

\*TT-Metal Implementation coming soon

![Osaka Thousand Year Stare](https://i.redd.it/b064yxmkl0zb1.jpg)