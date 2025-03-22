import torch
from torch.utils.data import Dataset
from pymotion.io.bvh import BVH
import pymotion.rotations.ortho6d as sixd
import random

class BVHTextDataset(Dataset):
    def __init__(self, bvh_paths, txt_paths, max_length=128):
        self.bvh_paths = bvh_paths
        self.txt_paths = txt_paths
        self.max_length = max_length

    def __len__(self):
        return len(self.bvh_paths)
    
    def convert_line(self, line):
        line = line.strip().split(' ')
        print(line)
        print(len(line))
        exit()
    

    def load_bvh(self, path):
        print(path)
        bvh = BVH()
        bvh.load(path)
        local_rotations, world_positions, _, _, _, _ = bvh.get_data()
        continuous_rotations = sixd.from_quat(torch.from_numpy(local_rotations))
        print(continuous_rotations.shape)
        print(continuous_rotations.dtype)
        continuous_rotations = sixd.to_quat(continuous_rotations)
        print(continuous_rotations.shape)
        print(continuous_rotations.dtype)
        exit()
        dim1, dim2, _, _ = continuous_rotations.shape
        continuous_rotations = continuous_rotations.reshape(dim1, dim2, 6)
        continuous_rotations = continuous_rotations.reshape(dim1, dim2*6)

        world_positions = world_positions[:,0,:]

        continuous_rotations_tensor = torch.from_numpy(continuous_rotations)
        world_positions_tensor = torch.from_numpy(world_positions)

        continuous_rotations_tensor = continuous_rotations_tensor.float()
        world_positions_tensor = world_positions_tensor.float()

        motion_data = torch.cat((world_positions_tensor, continuous_rotations_tensor), dim=1)  # Concatenate along dim 1
        return motion_data, path  # Convert motion data to tensor

    def load_text(self, path):
        print(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.readlines()
        # Pick a random line from the text
        out_str = ''
        for t in text:
            t = t.split('.')[0]
            t = t.split('#')[0]
            out_str += t
            out_str += '\n'

        return out_str
    
    def __getitem__(self, idx):
        motion, path = self.load_bvh(self.bvh_paths[idx])
        text = self.load_text(self.txt_paths[idx])
        return motion, text, path

