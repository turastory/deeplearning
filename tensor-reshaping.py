import torch

x = torch.arange(9)

x_3x3 = x.view(3, 3)
print(x_3x3)
x_3x3 = x.reshape(3, 3)

# Transpose
y = x_3x3.t()
print(y.contiguous().view(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1, x2), dim=0).shape)
print(torch.cat((x1, x2), dim=1).shape)

# Flatten the rest
z = x1.view(-1)

batch = 64
x = torch.rand((batch, 3, 4))
z = x.view(batch, -1)
print(z.shape)

# Permute
print(x.permute(0, 2, 1).shape) # batch, 4, 3

# Add dimension at the index
print(torch.arange(10).unsqueeze(0).shape) # [1, 10]