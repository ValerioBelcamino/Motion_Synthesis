import torch
import torch.nn.functional as F

def collate_fn(batch):
    # Separate motion and text data
    motions, lengths, texts = zip(*batch)
    
    # Find the maximum length of the motion sequences in the batch
    max_motion_length = max(lengths)

    # Pad motion sequences to the same length (max_motion_length)
    padded_motions = []
    for motion in motions:
        pad_amount = max_motion_length - motion.shape[0]
        padded_motion = F.pad(motion, (0, 0, 0, pad_amount))  # Pad along the second dimension
        padded_motions.append(padded_motion)
    
    # Stack motions into a tensor (after padding)
    padded_motions = torch.stack(padded_motions)

    return padded_motions, lengths, texts
