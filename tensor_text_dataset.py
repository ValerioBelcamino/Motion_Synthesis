import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random

class TensorTextDataset(Dataset):
    def __init__(self, paths, base_path, max_length_motion):
        self.paths = paths
        self.max_length_motion = max_length_motion
        self.base_path = base_path
        self.mean = torch.load('mean.pt').to('cpu')
        self.std = torch.load('std.pt').to('cpu')
        print(f'loaded with {self.mean.shape}')
        print(f'loaded with {self.std.shape}')

    def __len__(self):
        return len(self.paths)
    

    def load_and_pad_torch(self, path):
        path = os.path.join(self.base_path, 'motion', path)
        path = path + '.pt'

        motion = torch.load(path)

        # not needed with the new version of the dataset
        # motion = motion.squeeze(0)  # Remove the first dimension

        # discard fingers since they dont move in this bvh dataset
        motion_split1 = motion[:, :18*6+3]
        motion_split2 = motion[:, -19*6:-15*6]
        motion = torch.cat((motion_split1, motion_split2), dim=1)

        # scale root location to be in a smaller range
        motion[:,:3] = motion[:,:3] / 1000.0

        motion =(motion - self.mean) / self.std

        # downsample to save training time
        # motion = motion[::2,:]

        current_size = motion.shape[0]
        # pad_amount = self.max_length_motion - current_size  # Only pad along dimension 1

        # Apply padding (pad last dimension first)
        # padded_motion = F.pad(motion, (0, 0, 0, pad_amount))  # (left, right, top, bottom)
        return motion, current_size  # Convert motion data to tensor

    def load_text(self, path):
        path = os.path.join(self.base_path, 'text', path)
        path = path + '.txt'

        with open(path, "r", encoding="utf-8") as f:
            text = f.readlines()

        # Pick a random line from the text
        # line = random.choice(text)
        line = text[0]

        line = line.split('#')[0]

        return line#[:-1]  # Remove the newline character
    
    def __getitem__(self, idx):
        motion, length = self.load_and_pad_torch(self.paths[idx])
        text = self.load_text(self.paths[idx])
        return motion, length, text

