import ttnn
import torch

if __name__ == "__main__":
    device = ttnn.open_device(0)
    try:
        # Create normal torch tensors
        x = torch.as_tensor([4.0], dtype=torch.bfloat16, device="cpu")
        y = torch.as_tensor([2.0], dtype=torch.bfloat16, device="cpu")
        z_cpu = x / y # CPU division
        
        # Convert torch tensors to TTNN tensors
        y_ttnn = ttnn.from_torch(y, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        x_ttnn = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        z_ttnn = ttnn.div(x_ttnn, y_ttnn) # TTNN division

        print(f"z_cpu: {z_cpu}")
        print(f"z_ttnn: {z_ttnn}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ttnn.close_device(device)
