# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions
from fairmotion.utils import utils

"""
Structure of npz file in AMASS dataset is as follows.
- trans (num_frames, 3):  translation (x, y, z) of root joint
- gender str: Gender of actor
- mocap_framerate int: Framerate in Hz
- betas (16): Shape parameters of body. See https://smpl.is.tue.mpg.de/
- dmpls (num_frames, 8): DMPL parameters
- poses (num_frames, 156): Pose data. Each pose is represented as 156-sized
    array. The mapping of indices encoding data is as follows:
    0-2 Root orientation
    3-65 Body joint orientations
    66-155 Finger articulations
"""

# Custom names for 22 joints in AMASS data
joint_names = [
    "root",
    "lhip",
    "rhip",
    "lowerback",
    "lknee",
    "rknee",
    "upperback",
    "lankle",
    "rankle",
    "chest",
    "ltoe",
    "rtoe",
    "lowerneck",
    "lclavicle",
    "rclavicle",
    "upperneck",
    "lshoulder",
    "rshoulder",
    "lelbow",
    "relbow",
    "lwrist",
    "rwrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3"
]

def create_skeleton_from_amass_bodymodel(bm, betas, joint_names):
    num_body_joints = 22
    num_hand_joints = 31
    pose_body_zeros = torch.zeros((1, 3 * (num_body_joints - 1)))
    pose_hand_zeros = torch.zeros((1, 3 * (num_hand_joints - 1)))
    num_joints = len(joint_names)
    body = bm(pose_body=pose_body_zeros, pose_hand=pose_hand_zeros, betas=betas)
    base_position = body.Jtr.detach().numpy()[0, :]

    parents = bm.kintree_table[0].long()[:]
    # parents_hand = bm.kintree_table[0].long()[num_body_joints+3:num_body_joints + num_hand_joints+3]
    # parents = np.concatenate([parents_body, parents_hand])
    # print(len(joint_names))
    # print(parents.shape)
    # print(base_position.shape)

    discard_joint_idx = []

    joints = []
    for i in range(len(joint_names)):
        if joint_names[i] in ["jaw", "left_eye_smplhf", "right_eye_smplhf"]:
            discard_joint_idx.append(i)
        joint = motion_class.Joint(name=joint_names[i])
        if i == 0:
            joint.info["dof"] = 6
            joint.xform_from_parent_joint = conversions.p2T(np.zeros(3))
        else:
            joint.info["dof"] = 3
            # print(base_position[i])
            # print(parents[i])
            # print(base_position[parents[i]])
            
            joint.xform_from_parent_joint = conversions.p2T(
                base_position[i] - base_position[parents[i]]
            )
        joints.append(joint)


    parent_joints = []
    for i in range(num_joints):
        parent_joint = None if parents[i] < 0 else joints[parents[i]]
        parent_joints.append(parent_joint)

    # print(f'discard_joint_idx = {discard_joint_idx}')

    skel = motion_class.Skeleton()
    for i in range(num_joints):
        # if joints[i] is not None and parent_joints[i] is not None:
        #     print(f'{joints[i].name} -- {parent_joints[i].name}')
        if i not in discard_joint_idx:
            skel.add_joint(joints[i], parent_joints[i])

    return skel


def create_motion_from_amass_data(filename, bm, override_betas=None):
    bdata = np.load(filename)
    lst = bdata.files
    # for item in lst:
        # print(item)



    # print(bdata["labels"])
    # print(bdata["labels"].shape)
    # print(bdata["pose_body"].shape)
    # print(bdata["pose_hand"].shape)
    # print(len(joint_names))


    # exit()



    if override_betas is not None:
        betas = torch.Tensor(override_betas[:10][np.newaxis]).to("cpu")
    else:
        betas = torch.Tensor(bdata["betas"][:10][np.newaxis]).to("cpu")
    
    # print(betas.shape)
    skel = create_skeleton_from_amass_bodymodel(
        bm, betas, joint_names,
    )


    try:
        fps = float(bdata["mocap_framerate"])
    except Exception as e:
        fps = 60
    root_orient = bdata["root_orient"]  # controls the global root orientation
    pose_body = bdata["pose_body"]  # controls body joint angles
    pose_hands = bdata["pose_hand"]  # controls body joint angles
    trans = bdata["trans"]  # controls the finger articulation

    # print(pose_body)
    # print(pose_body.shape)
    # print(bdata["poses"])
    # print(trans.shape)
    # exit()

    motion = motion_class.Motion(skel=skel, fps=fps)

    num_joints = skel.num_joints()
    parents = bm.kintree_table[0].long()[:num_joints]
    # print(f'numjoints: {num_joints}')

    for frame in range(pose_body.shape[0]):
        pose_body_frame = np.concatenate([pose_body[frame], pose_hands[frame]])
        # print(pose_body_frame)
        # print(pose_body_frame.shape)
        # print(pose_body[frame].shape)
        # print(pose_hands[frame].shape)
        # pose_hand_frame = pose_hands[frame]
        root_orient_frame = root_orient[frame]
        root_trans_frame = trans[frame]
        pose_data = []
        for j in range(num_joints):
            if j == 0:
                T = conversions.Rp2T(
                    conversions.A2R(root_orient_frame), root_trans_frame
                )
            else:
                T = conversions.R2T(
                    conversions.A2R(
                        pose_body_frame[(j - 1) * 3 : (j - 1) * 3 + 3]
                    )
                )
            pose_data.append(T)
        motion.add_one_frame(pose_data)

    return motion


def load_body_model(bm_path, num_betas=10, model_type="smplx"):
    comp_device = torch.device("cpu")
    bm = BodyModel(
        bm_path=bm_path, 
        num_betas=num_betas, 
        model_type=model_type
    ).to(comp_device)
    return bm


def load(file, bm=None, bm_path=None, num_betas=10, model_type="smplx", override_betas=None):
    if bm is None:
        # Download the required body model. For SMPL-H download it from
        # http://mano.is.tue.mpg.de/.
        print("creating model")
        assert bm_path is not None, "Please provide SMPL body model path"
        bm = load_body_model(bm_path, num_betas, model_type)
    return create_motion_from_amass_data(
        filename=file, bm=bm, override_betas=override_betas)


def save():
    raise NotImplementedError("Using bvh.save() is recommended")


def load_parallel(files, cpus=20, **kwargs):
    return utils.run_parallel(load, files, num_cpus=cpus, **kwargs)
