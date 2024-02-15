import torch

# Check PyTorch version
print("PyTorch Version:", torch.__version__)

# Check CUDA availability and version
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
else:
    print("CUDA is not available.")