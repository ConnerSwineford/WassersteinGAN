import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm

import utils
from generator import Generator
from critic import Critic

parser = ArgumentParser(description="Train a VAGAN.")
parser.add_argument('--data_path', type=str, help='Path to the csv of the training data')
parser.add_argument('--epochs', default=100, type=int, help='Number of iterations through the entire data set')
parser.add_argument('--batch_size', default=16, type=int, help='Number of subjects in one model iteration')
parser.add_argument('--workers', type=int, default=1, help='Number of processes for multiprocessing')
args = parser.parse_args()

class VAGAN(nn.Module):
    def __init__(self, input_shape):
        super(VAGAN, self).__init__()
        self.critic = Critic(input_shape[1])
        self.generator = Generator(input_shape[1])
        
    def forward(self, x):
        generated_maps = self.generator(x)
        fake_images = x + generated_maps
        return fake_images

def train(rank, vagan, epochs, gen_optimizer, critic_optimizer, world_size, device):

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    print(f'[{rank}] Importing Dataset...')

    df = utils.import_raw_data(args.data_path)
    data = utils.NiiDataset(df['NiiPath'], df['Irr'], df['SubjID'])
    sampler = torch.utils.data.distributed.DistributedSampler(data)
    dataloader = DataLoader(data, batch_size=args.batch_size, num_workers=world_size, sampler=sampler)

    print(f'[{rank}] Distributing Model...')

    vagan.to(device)
    vagan = DDP(vagan)

    for epoch in tqdm(range(epochs), leave=False, desc=f'Training'):
        for real_images, _, _ in dataloader:
            real_images = real_images.to(device)
            
            # Update critic
            for _ in range(5):
                critic_optimizer.zero_grad()
                
                fake_images = vagan(real_images).detach()
                critic_real = vagan.module.critic(real_images)
                critic_fake = vagan.module.critic(fake_images)
                
                critic_loss_real = F.binary_cross_entropy_with_logits(critic_real, torch.ones_like(critic_real))
                critic_loss_fake = F.binary_cross_entropy_with_logits(critic_fake, torch.zeros_like(critic_fake))
                critic_loss = critic_loss_real + critic_loss_fake
                
                critic_loss.backward()
                critic_optimizer.step()
            
            # Update generator
            gen_optimizer.zero_grad()
            
            fake_images = vagan(real_images)
            critic_fake = vagan.module.critic(fake_images)
            gen_loss = F.binary_cross_entropy_with_logits(critic_fake, torch.ones_like(critic_fake))
            
            gen_loss.backward()
            gen_optimizer.step()
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Gen Loss: {gen_loss.item()}, Critic Loss: {critic_loss.item()}")

    dist.destroy_process_group()

if __name__ == "__main__":

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_shape = (args.batch_size, 1, 91, 108, 91)
    vagan = VAGAN(input_shape)
    gen_optimizer = torch.optim.Adam(vagan.generator.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(vagan.critic.parameters(), lr=1e-4)

    print('Model Initialized...')

    mp.spawn(train,
             args=(vagan, args.epochs, gen_optimizer, critic_optimizer, args.workers, device),
             nprocs=args.workers,
             join=True)
