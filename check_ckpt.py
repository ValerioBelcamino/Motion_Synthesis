import torch

path = '/home/belca/Downloads/best_model.pth'

checkpoint = torch.load(path)

print(checkpoint['epoch'])
print(checkpoint['loss'])