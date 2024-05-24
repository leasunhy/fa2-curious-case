import torch
from flash_attn import flash_attn_func

q = torch.load('q.pt')
k = torch.load('k.pt')
v = torch.load('v.pt')

out = flash_attn_func(q, k, v)
out.sum().backward()
print(q.grad)
