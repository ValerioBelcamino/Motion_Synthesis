import os 
from sklearn.model_selection import train_test_split
from tensor_text_dataset import TensorTextDataset
from torch.utils.data import DataLoader

from motion_encoder import MotionEncoder
from text_encoder import TextEncoder
from motion_decoder import MotionDecoder

from losses.losses import CrossModalLosses
from utils.collate_function import collate_fn
import torch 

import shutil


####### PARAMETERS #######

batch_size = 1

n_features = 135

_max_len = 205

print(f'{batch_size=}')
print(f'{n_features=}')
print(f'{_max_len=}')

##########################

print('Hello We Are Using the Last Version!!!')

basepath = '/home/belca/Desktop/Motion_Synthesis_Dataset'
motionpath = os.path.join(basepath, 'motion')

# Checkpoint directory
checkpoint_dir = "/home/belca/Desktop/Motion_Synthesis"
os.makedirs(checkpoint_dir, exist_ok=True)




fnames = [f.split('.')[0] for f in os.listdir(motionpath)]

# print(fnames)
print(len(fnames))

newfnames = []

for f in fnames:
    leng = int(f.split('_')[0])
    if leng < _max_len and leng > 15: # 1 minute at 30 fps
        newfnames.append(f)

fnames = newfnames
print(len(fnames))

# fnames = fnames[-20:]

# Let's split in train and test

# First, split into train (80%) and temp (20%) (test + validation)
# train_set, val_set = train_test_split(fnames, test_size=0.05, random_state=42)
train_set = fnames

# Then split temp into validation (10%) and test (10%)
# val_set, test_set = train_test_split(temp_set, test_size=0.6, random_state=42)

print("Train:", len(train_set))
# print("Validation:", len(val_set))
# print("Test:", len(test_set))


train_dataset = TensorTextDataset(train_set, basepath, _max_len)
# val_dataset = TensorTextDataset(val_set, basepath, _max_len)
# test_dataset = TensorTextDataset(test_set, basepath, 6361)

print("Train dataset:", len(train_dataset))
# print("Validation dataset:", len(val_dataset))
# print("Test dataset:", len(test_dataset))

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'{device=}')

# motion_encoder = MotionEncoder(nfeats=n_features, max_len=_max_len, num_heads=8, num_layers=8, latent_dim=512).to(device)
# print('Created motion encoder')
# text_encoder = TextEncoder(num_heads=8, num_layers=8, latent_dim=512).to(device)
# print('Created text encoder')
# motion_decoder = MotionDecoder(n_features, max_len=_max_len, num_heads=8, num_layers=8, latent_dim=512).to(device)
# print('Created motion decoder\n')


total_sum = torch.zeros((135), dtype=torch.float64).to('cuda')
total_sum_sq = torch.zeros((135), dtype=torch.float64).to('cuda')

total_count = 0

for i, (motions, lengths, texts) in enumerate(train_dataloader):
    texts = texts
    motions = motions.to(device)
    lengths = torch.tensor(lengths)
    
    B, T, F = motions.shape

    # Create mask to ignore padding
    mask = torch.arange(T, device=motions.device).unsqueeze(0) < lengths[0]

    # Masked data
    valid_data = motions[mask]
    summ = valid_data.sum(dim=0)
    
    valid_data_sq = valid_data ** 2
    summ_sq = valid_data_sq.sum(dim=0)
    
    total_sum += summ
    total_sum_sq +=summ_sq
    total_count += valid_data.shape[0]
    
    # print(total_sum)
    # print(total_sum_sq)
    # print(total_count)
    # print()

mean = total_sum / total_count
std = (total_sum_sq / total_count - mean ** 2).sqrt()


# Save as float32 for future use
torch.save(mean.float(), 'mean.pt')
torch.save(std.float(), 'std.pt')

print(f'Mean: {mean}')
print(f'Std: {std}')