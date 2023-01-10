import torch

x = torch.arange(15).view(3, 5)
print(x)

y = torch.arange(15).view(3, 5) + 15

cat = torch.cat((x.unsqueeze(0), y.unsqueeze(0)), dim=0)
print(cat)

print(cat.permute(0, 2, 1))