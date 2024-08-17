import ttnn
import torch

device = ttnn.open_device(0)
try:
    x = torch.as_tensor([4.0], dtype=torch.bfloat16)
    y = torch.as_tensor([2.0], dtype=torch.bfloat16)
    z_1 = x / y
    print(f"z_1: {z_1}")
    y_ttnn = ttnn.from_torch(y, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    x_ttnn = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    y_1 = ttnn.reciprocal(y_ttnn)
    z_2 = ttnn.multiply(x_ttnn, y_1)
    print(f"z_2: {z_2}")
    z_3 = x_ttnn / y_ttnn
    print(f"z_3: {z_3}")
finally:
    ttnn.close_device(device)
