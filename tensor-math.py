import torch

x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])

z = x + y
print(z)

# Element-wise division
z = torch.true_divide(x, y)
print(z)

# In-place operations
t = torch.zeros(3)
t += x
print(t)

# Exponentiation
z = x ** 2
print(z)

# Simple comparison
z = x > 0
print(z)


# Matrix multiplication
x1 = torch.rand((2, 4))
x2 = torch.rand((4, 3))
# x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)
print(x3)

# Matrix exponentiation
matrix_exp = torch.rand((3, 3)) # It should be a square matrix
print(matrix_exp.matrix_power(3))

# Element-wise multiplication
z = x * y
print(z)

# Dot product
print(x.dot(y))


# Batch matrix multiplication
# Note: In many cases, you may calculate matrix multiplication for multiple matrices at once.
#       bmm() is used for this purpose.
batch = 30
n = 20
m = 10
h = 5

x1 = torch.rand((batch, n, m))
x2 = torch.rand((batch, m, h))
x3 = torch.bmm(x1, x2) # (batch, n, h)


# Broadcasting
x1 = torch.rand((5,5))
x2 = torch.ones(5)
print(x1 - x2)

sum_x = torch.sum(x1, dim=0)
print(sum_x)
values, indices = x1.max(dim=0)
print(values)
print(indices)
print(x1.abs())
print(x1.argmax()) # Returns only the index of the maximum value
print(x1.float().mean(dim=0))

x1 = torch.ones(5)
x2 = torch.tensor([1, 0, 1, 0, 1])

# Element-wise comparison
print(x1 == x2)

print(torch.rand(5, 3).clamp(min=0.2, max=0.5))