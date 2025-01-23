import torch


# Check if CUDA is available
print(torch.version.cuda)
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")
print ( torch.cuda.is_available() )
if cuda_available:
    # Print CUDA version
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Print GPU details
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")
    
    # Perform a simple tensor operation on GPU
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    print(f"Tensor on GPU: {x}")
else:
    print("CUDA is not available. Please check your installation.")