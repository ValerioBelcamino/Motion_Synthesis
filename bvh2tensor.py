import torch
from bvh_text_dataset import BVHTextDataset
import os 
import time

batch_size = 1


# Dictionary of filenames: idx to retrieve the descriptions

filename_to_idx = {}

with open('index.csv', 'r') as f:
    # Read and skip the header
    lines = f.readlines()[1:]

    for line in lines:
        path, _, _, idx = line.split(',')
        # path = os.path.basename(path)
        path = path.split('.npy')[0] + '.bvh'
        path = str.replace(path, './pose_data', '/home/belca/Desktop/AMASS')
        # path = str.replace(path, '  ', '_')
        path = str.replace(path, ' ', '_')
        idx = idx.split('.')[0]
        filename_to_idx[path] = idx

# print(len(filename_to_idx.keys()))
# print(filename_to_idx)
# exit()

# Let's start scanning the AMASS dir for the bvh files

AMASS_path = '/home/belca/Desktop/AMASS'
subdirs = os.listdir(AMASS_path)

bvh_files = []
bvh_real_names = {}

for sd in subdirs:
    print(sd)
    sd_path = os.path.join(AMASS_path, sd)
    if os.path.isdir(sd_path):
        subsubdirs = os.listdir(sd_path)
        for ssd in subsubdirs:
            print('\t',ssd)
            ssd_path = os.path.join(AMASS_path, sd, ssd)
            if os.path.isdir(ssd_path):
                files = os.listdir(ssd_path)
                for f in files:
                    if f.endswith('.bvh'):
                        print('\t\t', f)
                        f_path = os.path.join(AMASS_path, sd, ssd, f)
                        f_path_new = str.replace(f_path, 'stageii', 'poses')
                        # bvh_files[f_path] = f
                        bvh_files.append(f_path_new)
                        bvh_real_names[f_path_new] = f_path
                        # if f in filename_to_idx:
                        #     print('\t\t\t', filename_to_idx[f])
                        #     break

print(f'\nLoaded {len(bvh_files)} bvh files')
bvh_with_idx = []
descriptions = []
missed = []
for bvh in bvh_files:
    if bvh in filename_to_idx:
        bvh_with_idx.append(bvh_real_names[bvh])
        descriptions.append('texts/' + filename_to_idx[bvh]+'.txt')
        # print(bvh, filename_to_idx[bvh])
        continue
    else:
        missed.append(bvh)
        # print(bvh, 'not found')

print(f'\nFound {len(bvh_with_idx)} bvh files with idx')
print(f'Missed {len(missed)} bvh files with idx')


dataset = BVHTextDataset(bvh_with_idx, descriptions, max_length=128)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

motion_length = 0
text_length = 0

start_time = time.time()
for i, (motion, text, path) in enumerate(dataloader):
    new_path = f'{i}_{os.path.basename(path[0]).split(".")[0]}'
    text = text[0]
    print(i, motion.shape, text, path)
    if motion.shape[1] > motion_length:
        motion_length = motion.shape[1]
    if len(text) > text_length:
        text_length = len(text)
    # print(motion.dtype)
    # save the motion and text
    torch.save(motion, f'dataset/motion/{new_path}.pt')
    with open(f'dataset/text/{new_path}.txt', 'w') as f:
        f.write(text)

print(f'\nElapsed time: {time.time() - start_time} seconds') # Elapsed time: 479.1507658958435 seconds
print(f'Max motion length: {motion_length}') # Max motion length: 6361
print(f'Max text length: {text_length}') # Max text length: 324
