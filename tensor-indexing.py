import torch


batch = 10
features = 25
x = torch.rand((batch, features))

print(x[0])  # First batch - all features
print(x[:, 0])  # All batches - first feature
print(x[2:4, 0:10])  # 2nd and 3rd batch - first 10 features

# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols]) # Pick (1,4), (0,0)

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])

# Useful operations
print(x.where(x > 5, x * 2)) # Replace all values less than 5 with 2 * x
print(torch.tensor([0, 1, 1, 2, 2, 3]).unique())
print(x.ndimension()) # Number of dimensions, 5x5x5 -> 3
print(x.numel()) # Number of elements, 5x5x5 -> 125