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

learning_rate = 1e-4
n_epochs = 1000
batch_size = 32

n_features = 135

_max_len = 300

print(f'{learning_rate=}')
print(f'{n_epochs=}')
print(f'{batch_size=}')
print(f'{n_features=}')
print(f'{_max_len=}')

##########################

print('Hello We Are Using the Last Version!!!')

basepath = '/kaggle/input/motion/dataset/dataset'
motionpath = os.path.join(basepath, 'motion')

# Checkpoint directory
checkpoint_dir = "/kaggle/working/"
os.makedirs(checkpoint_dir, exist_ok=True)



# Checkpoint file path
checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

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

# Let's split in train and test

# First, split into train (80%) and temp (20%) (test + validation)
train_set, val_set = train_test_split(fnames, test_size=0.05, random_state=42)

# Then split temp into validation (10%) and test (10%)
# val_set, test_set = train_test_split(temp_set, test_size=0.6, random_state=42)

print("Train:", len(train_set))
print("Validation:", len(val_set))
# print("Test:", len(test_set))


train_dataset = TensorTextDataset(train_set, basepath, _max_len)
val_dataset = TensorTextDataset(val_set, basepath, _max_len)
# test_dataset = TensorTextDataset(test_set, basepath, 6361)

print("Train dataset:", len(train_dataset))
print("Validation dataset:", len(val_dataset))
# print("Test dataset:", len(test_dataset))

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

motion_encoder = MotionEncoder(nfeats=n_features, max_len=_max_len).to(device)
print('Created motion encoder')
text_encoder = TextEncoder().to(device)
print('Created text encoder')
motion_decoder = MotionDecoder(n_features, max_len=_max_len).to(device)
print('Created motion decoder\n')

optimizer = torch.optim.AdamW(list(motion_encoder.parameters()) +
                             list(text_encoder.parameters()) +
                             list(motion_decoder.parameters()), lr=learning_rate)

loss_function = CrossModalLosses()

best_loss_val = 10.0
best_loss_train = 10.0


# checkpoint_path = os.path.join(checkpoint_dir, )

# Load checkpoint if it exists
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    motion_encoder.load_state_dict(checkpoint['motion_encoder_state_dict'])
    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    motion_decoder.load_state_dict(checkpoint['motion_decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_loss_val = checkpoint['loss']  # Restore best validation loss
    
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch} with best validation loss {best_loss_val:.6f}")
else:
    start_epoch = 0  # Train from scratch
    print("No checkpoint found, starting from epoch 0")

# exit()

for e in range(start_epoch, n_epochs):
    total_train_loss = 0
    for i, (motions, lengths, texts) in enumerate(train_dataloader):
        texts = texts
        motions = motions.to(device)
        lengths = lengths

        # print('ciao')
        # print(len(texts))
        # print(motions.size())
        # print(motions.dtype)
        # print(len(lengths))
        # print(lengths)
        # print(texts)
        # exit()

        optimizer.zero_grad()

        dist_T = text_encoder(texts)
        dist_M = motion_encoder(motions, lengths)

        z_T = dist_T.rsample()
        z_M = dist_M.rsample()

        H_hat_T = motion_decoder(z_T, lengths)
        H_hat_M = motion_decoder(z_M, lengths)

        loss, kl_loss, embedding_similarity_loss, reconstruction_loss = loss_function(dist_T, dist_M, z_T, z_M, motions, H_hat_T, H_hat_M, lengths)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        # del motions, texts, lengths, dist_T, dist_M, epsilon_T, epsilon_M, mu_T, mu_M, std_T, std_M, z_T, z_M, H_hat_T, H_hat_M
        del motions, texts, lengths, dist_T, dist_M, z_T, z_M, H_hat_T, H_hat_M

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {e} training loss: {avg_train_loss}")
    print(f'{kl_loss=}, {embedding_similarity_loss=}, {reconstruction_loss=}')

    # Save the model checkpoint if the validation loss improves
    if avg_train_loss < best_loss_train:
        best_loss_train = avg_train_loss
        print('Training loss improved, saving training checkpoint...')
        checkpoint = {
            'epoch': e,
            'motion_encoder_state_dict': motion_encoder.state_dict(),
            'text_encoder_state_dict': text_encoder.state_dict(),
            'motion_decoder_state_dict': motion_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved at epoch {e}')
        shutil.move(checkpoint_path, "/kaggle/working/best_model_train.pth")

     # Validation
    with torch.no_grad():
        total_val_loss = 0
        for i, (motions, lengths, texts) in enumerate(val_dataloader):
            # Move validation data to device
            texts = texts
            motions = motions.to(device)
            lengths = lengths

            # Forward pass for validation
            dist_T = text_encoder(texts)
            dist_M = motion_encoder(motions, lengths)

            z_T = dist_T.rsample()
            z_M = dist_M.rsample()

            H_hat_T = motion_decoder(z_T, lengths)
            H_hat_M = motion_decoder(z_M, lengths)

            # Calculate validation loss
            loss, kl_loss, embedding_similarity_loss, reconstruction_loss = loss_function(dist_T, dist_M, z_T, z_M, motions, H_hat_T, H_hat_M, lengths)
            total_val_loss += loss.item()

            del motions, texts, lengths, dist_T, dist_M, z_T, z_M, H_hat_T, H_hat_M

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {e}, validation loss: {avg_val_loss}")
        

        # Save the model checkpoint if the validation loss improves
        if avg_val_loss < best_loss_val:
            best_loss_val = avg_val_loss
            print('Validation loss improved, saving validation checkpoint...')
            checkpoint = {
                'epoch': e,
                'motion_encoder_state_dict': motion_encoder.state_dict(),
                'text_encoder_state_dict': text_encoder.state_dict(),
                'motion_decoder_state_dict': motion_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved at epoch {e}')
            shutil.move(checkpoint_path, "/kaggle/working/best_model.pth")

    print('\n')
