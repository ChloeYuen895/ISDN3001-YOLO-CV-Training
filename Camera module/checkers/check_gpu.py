import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Prints GPU name
print(torch.__version__)  # Ensure CUDA-enabled (e.g., 2.4.0+cu121)