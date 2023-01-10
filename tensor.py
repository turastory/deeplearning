import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)

print(tensor)
print(tensor.dtype)
print(tensor.device)
print(tensor.shape)
print(tensor.requires_grad)

# Various ways to initialize tensors

print(torch.empty(size=(3, 3)))
print(torch.zeros(size=(3, 3)))
print(torch.rand(size=(3, 3)))
print(torch.ones(size=(3, 3)))
print(torch.eye(5))
print(torch.arange(start=0, end=5, step=1))
print(torch.linspace(start=0, end=1, steps=9))
print(torch.empty((1,5)).normal_(mean=0, std=1))
print(torch.empty((1,5)).uniform_(0, 1))
print(torch.diag(torch.ones(4)))

tensor = torch.arange(4)
print(tensor)
print(tensor.bool())
print(tensor.long()) # int64
# Note: float16 is only supported on GPU, so without a GPU this will throw an error or may be slow?
print(tensor.half()) # float16
print(tensor.float()) # float32
print(tensor.double()) # float64

import numpy as np

numpy_array = np.zeros((5, 5))
tensor = torch.from_numpy(numpy_array)
numpy_array_back = tensor.numpy()