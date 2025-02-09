import torch
import time

# Check if a GPU is available and select the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    raise SystemError("GPU not available! Please run this on a machine with a CUDA-capable GPU.")

print("Using device:", device)

# Create two large random matrices on the GPU.
# Adjust matrix_size based on your GPU's memory and compute power.
matrix_size = 5000  
a = torch.randn(matrix_size, matrix_size, device=device)
b = torch.randn(matrix_size, matrix_size, device=device)

# Run a heavy computation loop for a few seconds (here, 5 seconds).
end_time = time.time() + 5  # Duration in seconds
while time.time() < end_time:
    # Perform a heavy matrix multiplication which should saturate the GPU.
    c = torch.matmul(a, b)
    # Synchronize to ensure that each operation is completed on the GPU.
    torch.cuda.synchronize()

print("Finished heavy GPU computation.")
