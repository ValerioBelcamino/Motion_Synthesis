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

####### PARAMETERS #######

learning_rate = 1e-4
n_epochs = 100
batch_size = 4

n_features = 315

##########################



basepath = '/content/drive/My Drive/Motion_Synthesis_Dataset'
motionpath = os.path.join(basepath, 'motion')

# Checkpoint directory
checkpoint_dir = "/content/drive/My Drive/Motion_Synthesis/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Checkpoint file path
checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')

fnames = [f.split('.')[0] for f in os.listdir(motionpath)]

# Let's split in train and test

# First, split into train (80%) and temp (20%) (test + validation)
train_set, val_set = train_test_split(fnames, test_size=0.3, random_state=42)

# Then split temp into validation (10%) and test (10%)
# val_set, test_set = train_test_split(temp_set, test_size=0.6, random_state=42)

print("Train:", len(train_set))
print("Validation:", len(val_set))
# print("Test:", len(test_set))


train_dataset = TensorTextDataset(train_set, basepath, 6361)
val_dataset = TensorTextDataset(val_set, basepath, 6361)
# test_dataset = TensorTextDataset(test_set, basepath, 6361)

print("Train dataset:", len(train_dataset))
print("Validation dataset:", len(val_dataset))
# print("Test dataset:", len(test_dataset))

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

motion_encoder = MotionEncoder(nfeats=n_features, max_len=6363).to(device)
print('Created motion encoder')
text_encoder = TextEncoder().to(device)
print('Created text encoder')
motion_decoder = MotionDecoder(n_features, max_len=6363).to(device)
print('Created motion decoder\n')

optimizer = torch.optim.AdamW(list(motion_encoder.parameters()) +
                             list(text_encoder.parameters()) +
                             list(motion_decoder.parameters()), lr=learning_rate)

loss_function = CrossModalLosses()

best_loss = 10.0


for e in range(n_epochs):
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

        # Reparameterization trick
        epsilon_T = torch.randn_like(dist_T.mean)
        epsilon_M = torch.randn_like(dist_M.mean)

        mu_T = dist_T.mean
        mu_M = dist_M.mean

        std_T = dist_T.stddev
        std_M = dist_M.stddev

        z_T = mu_T + std_T * epsilon_T
        z_M = mu_M + std_M * epsilon_M

        H_hat_T = motion_decoder(z_T, lengths)
        H_hat_M = motion_decoder(z_M, lengths)

        loss = loss_function(mu_T, std_T, mu_M, std_M, z_T, z_M, motions, H_hat_M, H_hat_T)
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

        del motions, texts, lengths, dist_T, dist_M, epsilon_T, epsilon_M, mu_T, mu_M, std_T, std_M, z_T, z_M, H_hat_T, H_hat_M

    print(f"Epoch {e} loss: {total_train_loss / len(train_dataloader)}")

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

            # Reparameterization trick for validation
            epsilon_T = torch.randn_like(dist_T.mean)
            epsilon_M = torch.randn_like(dist_M.mean)

            mu_T = dist_T.mean
            mu_M = dist_M.mean

            std_T = dist_T.stddev
            std_M = dist_M.stddev

            z_T = mu_T + std_T * epsilon_T
            z_M = mu_M + std_M * epsilon_M

            H_hat_T = motion_decoder(z_T, lengths)
            H_hat_M = motion_decoder(z_M, lengths)

            # Calculate validation loss
            loss = loss_function(mu_T, std_T, mu_M, std_M, z_T, z_M, motions, H_hat_M, H_hat_T)
            total_val_loss += loss.item()

            del motions, texts, lengths, dist_T, dist_M, epsilon_T, epsilon_M, mu_T, mu_M, std_T, std_M, z_T, z_M, H_hat_T, H_hat_M

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {e}, validation loss: {avg_val_loss}")

        # Save the model checkpoint if the validation loss improves
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print('Validation loss improved, saving checkpoint...')
            checkpoint = {
                'epoch': e,
                'model_state_dict': motion_encoder.state_dict(),
                'text_encoder_state_dict': text_encoder.state_dict(),
                'motion_decoder_state_dict': motion_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved at epoch {e}')
