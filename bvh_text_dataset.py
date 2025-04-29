import torch
from torch.utils.data import Dataset
from pymotion.io.bvh import BVH
import pymotion.rotations.ortho6d as sixd
import random

class BVHTextDataset(Dataset):
    def __init__(self, bvh_paths, txt_paths, cutting_len, max_length=128):
        self.bvh_paths = bvh_paths
        self.txt_paths = txt_paths
        self.max_length = max_length
        self.cutting_len = cutting_len

        self.fps_dict = {
            'ACCAD':120,
            'BMLmovi':120,
            'DFaust':60,
            'EKUT':100,
            'CMU':60,
            'HDM05':120,
            'EyesJapanDataset':120,
            'HumanEva':120,
            'Human4D':120,
            'SFU':120,
            'SSM':120,
            'Transitions':120,
            'TotalCapture':60,
            'KIT':100,
            }

    def __len__(self):
        return len(self.bvh_paths)
    
    def convert_line(self, line):
        line = line.strip().split(' ')
        print(line)
        print(len(line))
        exit()
    

    def load_bvh(self, path, cl):
        print(path)
        bvh = BVH()
        bvh.load(path)
        local_rotations, world_positions, _, _, _, _ = bvh.get_data()
        # continuous_rotations = sixd.from_quat(torch.from_numpy(local_rotations))
        continuous_rotations = local_rotations
        # print(continuous_rotations.shape)
        # print(continuous_rotations.dtype)
        # continuous_rotations = sixd.to_quat(continuous_rotations)
        # print(continuous_rotations.shape)
        # print(continuous_rotations.dtype)
        # exit()
        # print(continuous_rotations.shape)
        dim1, dim2, _ = continuous_rotations.shape
        continuous_rotations = continuous_rotations.reshape(dim1, dim2*4)
        
        # continuous_rotations = continuous_rotations.reshape(dim1, dim2, 6)
        # continuous_rotations = continuous_rotations.reshape(dim1, dim2*6)

        world_positions = world_positions[:,0,:]

        continuous_rotations_tensor = torch.from_numpy(continuous_rotations)
        world_positions_tensor = torch.from_numpy(world_positions)

        continuous_rotations_tensor = continuous_rotations_tensor.float()
        world_positions_tensor = world_positions_tensor.float()

        motion_data = torch.cat((world_positions_tensor, continuous_rotations_tensor), dim=1)  # Concatenate along dim 1

        dataset_name = path.split('/')[5]

        dataset_fps = self.fps_dict[dataset_name]
        print(dataset_name)
        print(dataset_fps)

        downsampling_rate = dataset_fps / 20.0
        print(f'{downsampling_rate=}, {int(downsampling_rate)=}')

        downsampling_rate = int(downsampling_rate)

        multiplier = dataset_fps / 20.0
        print(f'{multiplier=}, {int(multiplier)=}')


        if dataset_name == 'HDM05':
            print(f'removing {int(3*20*multiplier)} first samples')
            motion_data = motion_data[int(3*20*multiplier):]

        elif dataset_name == 'TotalCapture':
            print(f'removing {int(1*20*multiplier)} first samples')
            motion_data = motion_data[int(1*20*multiplier):]

        elif dataset_name == 'EyesJapanDataset':
            print(f'removing {int(3*20*multiplier)} first samples')
            motion_data = motion_data[int(3*20*multiplier):]

        elif dataset_name == 'Transitions':
            print(f'removing {int(0.5*20*multiplier)} first samples')
            motion_data = motion_data[int(0.5*20*multiplier):]

        lung = int(cl[1]) - int(cl[0])
        # print(cl)
        start = int(int(cl[0]) * multiplier)

        end = int(int(cl[1]) * multiplier)


        print(start)
        print(end)
        print(motion_data.shape)

        if end == -1 * multiplier:
            print('ciao')
            end = motion_data.shape[0]
            print(end)
            input('khggiy')

        # print(motion_data.shape)

        if int(cl[1]) == -1:
            exit()

        motion_data = motion_data[start:end, :]

        motion_data = motion_data[::downsampling_rate]
        print(motion_data.shape)

        if abs(motion_data.shape[0] - lung) > 5:
            print(abs(motion_data.shape[0] - lung))
            print(lung)
            # input('khggiy')
        # if motion_data.shape[0]<10:
        #     input('khggiy')
        return motion_data, path  # Convert motion data to tensor

    def load_text(self, path):
        print(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.readlines()
        # Pick a random line from the text
        out_str = ''
        for t in text:
            end = float(t.split('#')[-1])
            start = float(t.split('#')[-2])

            t = t.split('.')[0]
            t = t.split('#')[0]

            out_str += t + f'#{start}#{end}'
            out_str += '\n'
        # print(out_str)
        # exit()
        return out_str
    
    def __getitem__(self, idx):
        motion, path = self.load_bvh(self.bvh_paths[idx], self.cutting_len[idx])
        text = self.load_text(self.txt_paths[idx])
        return motion, text, path

