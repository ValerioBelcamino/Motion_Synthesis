import os 
from text_encoder import TextEncoder
from motion_decoder import MotionDecoder
import torch 
import time

####### PARAMETERS #######

learning_rate = 1e-4
n_epochs = 100
batch_size = 8

n_features = 135

##########################



# Checkpoint directory
checkpoint_dir = "/home/belca/Desktop/Motion_Synthesis"
os.makedirs(checkpoint_dir, exist_ok=True)



# Checkpoint file path
checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

text_encoder = TextEncoder().to(device)
print('Created text encoder')
motion_decoder = MotionDecoder(n_features, max_len=1800).to(device)
print('Created motion decoder\n')

checkpoint = torch.load(checkpoint_path, map_location=device)

text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
motion_decoder.load_state_dict(checkpoint['motion_decoder_state_dict'])

print('loaded checkpoint')

# text = input('Describe the desired action:')
text = 'a person walks in circle'
print(text)

start_time = time.time()
dist_T = text_encoder([text])

z_T = dist_T.rsample()
print(f'{z_T.shape=}')

motion = motion_decoder(z_T, [15*40]).squeeze(0)
torch.save(motion, 'synthetic.pt')

print(f'{motion.shape=}')

print(f'time spend generating: {time.time() - start_time}')