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
Result: 1.99219
Expected: 2
Test Passed
```

![Osaka Thousand Year Stare](https://i.redd.it/b064yxmkl0zb1.jpg)
