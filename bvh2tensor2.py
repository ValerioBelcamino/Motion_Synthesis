import torch
from bvh_text_dataset import BVHTextDataset
import os 
import time

batch_size = 1


# Dictionary of filenames: idx to retrieve the descriptions

filename_to_idx = {}
filename_cut = {}

with open('index.csv', 'r') as f:
    # Read and skip the header
    lines = f.readlines()[1:]

    for line in lines:
        path, start, finish, idx = line.split(',')
        # path = os.path.basename(path)
        path = path.split('.npy')[0] + '.bvh'
        path = str.replace(path, './pose_data', '/home/belca/Desktop/AMASS')
        # path = str.replace(path, '  ', '_')
        path = str.replace(path, ' ', '_')
        idx = idx.split('.')[0]

        if path in filename_to_idx:
            filename_to_idx[path].extend([idx])
        else:
            filename_to_idx[path] = [idx]


        if path in filename_cut:
            filename_cut[path].extend([(start, finish)])
        else:
            filename_cut[path] = [(start, finish)]


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
cutting_len = []
missed = []
for bvh in bvh_files:
    if bvh in filename_to_idx:
        descriptions_multiple = filename_to_idx[bvh]
        cutting_len_multiple = filename_cut[bvh]

        # bvh_with_idx.append(bvh_real_names[bvh])
        # descriptions.append('texts/' + filename_to_idx[bvh]+'.txt')
        # cutting_len.append(filename_cut[bvh])

        for des, cul in zip(descriptions_multiple, cutting_len_multiple):
            bvh_with_idx.append(bvh_real_names[bvh])
            descriptions.append('texts/' + des+'.txt')
            cutting_len.append(cul)

        # print(bvh, filename_to_idx[bvh])
        continue
    else:
        missed.append(bvh)
        # print(bvh, 'not found')

print(f'\nFound {len(bvh_with_idx)} bvh files with idx')
print(f'Missed {len(missed)} bvh files with idx')

# print(missed)

new_bvh_with_idx = []
for fff in bvh_with_idx:
    # if 'KIT' not in fff:
    #     continue
    new_bvh_with_idx.append(fff)
# print(missed)

print(f'\nFound {len(new_bvh_with_idx)} bvh files with idx')

dataset = BVHTextDataset(new_bvh_with_idx, descriptions, cutting_len, max_length=128)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

motion_length = 0
text_length = 0

start_time = time.time()
for i, (motion, text, path) in enumerate(dataloader):
    if i < 0:
        continue
    motion = motion.squeeze(0)
    new_path = f'{motion.shape[0]}_{i}_{os.path.basename(path[0]).split(".")[0]}'
    text = text[0]
    print('\n', i, i/len(dataloader), motion.shape, text, path)
    if motion.shape[0] > motion_length:
        motion_length = motion.shape[0]
    if len(text) > text_length:
        text_length = len(text)
    # print(motion.dtype)
    # save the motion and text
    torch.save(motion, f'dataset/motion/{new_path}.pt')
    with open(f'dataset/text/{new_path}.txt', 'w+') as f:
        f.write(text)
    print('\n')

print(f'\nElapsed time: {time.time() - start_time} seconds') # Elapsed time: 479.1507658958435 seconds
print(f'Max motion length: {motion_length}') # Max motion length: 6361
print(f'Max text length: {text_length}') # Max text length: 324
