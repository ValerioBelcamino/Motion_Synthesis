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

learning_rate = 1e-5
n_epochs = 100
batch_size = 8

n_features = 315

##########################



basepath = '/home/belca/Desktop/Motion_Synthesis_Dataset'
motionpath = os.path.join(basepath, 'motion')

fnames = [f.split('.')[0] for f in os.listdir(motionpath)]

# Let's split in train and test

# First, split into train (80%) and temp (20%) (test + validation)
train_set, temp_set = train_test_split(fnames, test_size=0.4, random_state=42)

# Then split temp into validation (10%) and test (10%)
val_set, test_set = train_test_split(temp_set, test_size=0.6, random_state=42)

print("Train:", len(train_set))
print("Validation:", len(val_set))
print("Test:", len(test_set))


train_dataset = TensorTextDataset(train_set, basepath, 6361)
val_dataset = TensorTextDataset(val_set, basepath, 6361)
test_dataset = TensorTextDataset(test_set, basepath, 6361)

print("Train dataset:", len(train_dataset))
print("Validation dataset:", len(val_dataset))
print("Test dataset:", len(test_dataset))

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

motion_encoder = MotionEncoder(nfeats=n_features, max_len=6363).to(device)
print('Created motion encoder')
text_encoder = TextEncoder().to(device)
print('Created text encoder')
motion_decoder = MotionDecoder(n_features, max_len=6363).to(device)
print('Created motion decoder\n')

optimizer = torch.optim.Adam(list(motion_encoder.parameters()) +
                             list(text_encoder.parameters()) +
                             list(motion_decoder.parameters()), lr=learning_rate)

loss_function = CrossModalLosses()

for e in range(n_epochs):
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

        loss = loss_function(mu_T, std_M, mu_M, std_M, z_T, z_M, motions, H_hat_M, H_hat_T)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {e}, iteration {i}, loss: {loss.item()}")

        del motions, texts, lengths, dist_T, dist_M, epsilon_T, epsilon_M, mu_T, mu_M, std_T, std_M, z_T, z_M, H_hat_T, H_hat_M

    # Validation
    with torch.no_grad():
        total_loss = 0
        for i, (texts, motions) in enumerate(val_dataloader):
            texts = texts.to(device)
            motions = motions.to(device)
            print(texts.size())
            print(motions.size())
            exit()
            z = motion_encoder(motions)
            z_text = text_encoder(texts)

            output = motion_decoder(z, z_text)

            loss = torch.nn.functional.mse_loss(output, motions)
            total_loss += loss.item()

        print(f"Epoch {e}, validation loss: {total_loss / len(val_dataloader)}")
