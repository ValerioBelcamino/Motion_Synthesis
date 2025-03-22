import torch 
import pymotion.rotations.ortho6d as sixd
import pymotion.rotations.quat as quat
import numpy as np
import torch.nn as nn

def insert_n_frames_in_header(n_frames):
    with open('bvh_header.txt', 'r') as file:
        header = file.readlines()
    header[-2] = f'Frames: {n_frames}'
    header = '\n'.join(header)
    # print(header)
    return header + '\n'

# insert_n_frames_in_header(100)


def reconstruct_bvh(out_file_path):

    action_tensor = torch.load('./test_bvh/action.pt')
    action_tensor2 = torch.load('./test_bvh/action2.pt')

    action_tensor = action_tensor.squeeze(0)
    action_tensor_split1 = action_tensor[:, :18*6+3]
    action_tensor_split2 = action_tensor[:, -19*6:-15*6]
    action_tensor = torch.cat((action_tensor_split1, action_tensor_split2), dim=1)
    print(action_tensor_split1)
    print(action_tensor_split2)
    print(action_tensor_split1.shape)
    print(action_tensor_split2.shape)
    print(action_tensor[0])
    print(action_tensor.shape)
    exit()
    print(action_tensor[:,-16*6:])
    print(action_tensor[:,18*6 + 3 + 1: 33*6 +3])
    print(action_tensor[:,-16*6:].shape)
    print(action_tensor[:,18*6 + 3 + 1: 33*6 +3].shape)
    action_tensor2 = action_tensor2.squeeze(0)[:206,:]

    smooth_l1_loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1)
    loss = smooth_l1_loss_fn(action_tensor, action_tensor2)
    print(f"{loss.item():.10f}") 
    exit()
    positions_tensor = action_tensor[:, :3]
    print(positions_tensor.shape)

    # (465, 52, 3, 2)
    orientations_6d_tensor = action_tensor[:, 3:]
    print(orientations_6d_tensor.shape)
    print(orientations_6d_tensor[0])
    print(positions_tensor[0])

    # orientations_6d_tensor = orientations_6d_tensor.to(torch.float16)
    # print(orientations_6d_tensor[0])
    anim_length = action_tensor.shape[0]
    print(f'Animation length: {anim_length}')

    # orientations_6d_tensor = orientations_6d_tensor.reshape(anim_length, 52, 6)
    # print(orientations_6d_tensor.shape)
    orientations_6d_np = orientations_6d_tensor.reshape(anim_length, 52, 3, 2).to(torch.float64).numpy()
    print(orientations_6d_tensor.shape)

    orientations_quat_np = sixd.to_quat(orientations_6d_np)
    print(orientations_quat_np.shape) 

    rotation_convention = np.tile(np.array(["x", "y", "z"]), (anim_length, 52, 1))
    print(rotation_convention.shape)

    orientations_euler_np = quat.to_euler(orientations_quat_np, order=rotation_convention)
    print(orientations_euler_np.shape)

    # orientations_quat_np = orientations_quat_np.reshape(anim_length, 52 * 4)
    orientations_euler_np = orientations_euler_np.reshape(anim_length, 52 * 3) * 180 / np.pi
    print(orientations_euler_np.shape) 

    positions_tensor = positions_tensor.numpy()
    print(positions_tensor.shape)

    motion_np = np.concatenate((positions_tensor, orientations_euler_np), axis=1)
    print(motion_np.shape)

    header = insert_n_frames_in_header(anim_length)

    with open(out_file_path, 'w') as file:
        file.write(header)

        for i in range(motion_np.shape[0]):
            for j in range(motion_np.shape[1]):
                file.write(str(motion_np[i][j]))
                file.write(' ')
            file.write('\n')

reconstruct_bvh('test_bvh/test.bvh')