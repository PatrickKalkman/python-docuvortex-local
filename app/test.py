import platform
import torch
x = torch.rand(5, 3)
print(x)

print(torch.has_mps)

print(platform.platform())
