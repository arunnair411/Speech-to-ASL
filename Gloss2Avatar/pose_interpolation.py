import torch
import argparse
import os
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

# Input: (B, 2, 137, 3)
# Output: (B, 10, 137, 3) 
class PoseInterpolation(nn.Module):
  def __init__(self, hidden_dim):
    super(PoseInterpolation, self).__init__()
    self.input_layer = nn.Linear(2*137*3, hidden_dim)
    self.output_layer = nn.Linear(hidden_dim, 20*137*3)
    self.act = F.elu

  def forward(self, x):
    batch_size = x.size(0)
    x = x.view(batch_size, 2*137*3)
    x = self.input_layer(x)
    x = self.act(x)
    x = self.output_layer(x)
    x = x.view(batch_size, 20, 137, 3)
    return x

class InterpolationDataset(Dataset):
  def __init__(self, dataset_path='interpolation.dataset'):
    self.data = torch.load(dataset_path)
    self.start_end_frames = self.data['x']
    self.intermediate_frames = self.data['y']

  def __len__(self):
    return len(self.start_end_frames)

  def __getitem__(self, idx):
    return self.start_end_frames[idx], self.intermediate_frames[idx]

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.mse_loss(output, target)
        loss = torch.mean(torch.abs(output - target))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += torch.sum(torch.abs(output - target)).item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, choices=['experiment', 'train', 'inference'], required=True)
  parser.add_argument('--dataset_path', type=str, default='interpolation.dataset')
  parser.add_argument('--pose_path', type=str, default='data/RawData/pose_npy_dir_avi')
  parser.add_argument('--pose_numbers', type=str, default='207,15602,15316', help="comma-separated pose numbers") # Car She Drives
  parser.add_argument('--output_path', type=str, default='output.pose_sequence')
  parser.add_argument('--train_test_split', type=float, default=0.9)
  parser.add_argument('--epochs', type=int, default=100)
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--log_interval', type=int, default=500)
  parser.add_argument('--seed', type=int, default=1)
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  model_savefile = "pose_interpolator.model"
  model = PoseInterpolation(args.hidden_dim).to(args.gpu)

  if args.mode == 'inference':
    # Load model
    model.load_state_dict(torch.load(model_savefile))

    # Load poses
    pose_numbers = args.pose_numbers.split(',')
    assert len(pose_numbers) > 0, "Fill out pose numbers"
    pose_files = list(map(lambda x: np.load(os.path.join(args.pose_path,'Liz_%s.npy'%x)), pose_numbers))

    # Stitch poses together
    pose_list = []
    end_frame = None
    for pose in pose_files:
      pose = torch.from_numpy(pose).float().to(args.gpu)
      print(pose.size(2))
      start_frame = pose[:,:,0]
      if end_frame is not None:
        x = torch.cat([end_frame.unsqueeze(0), start_frame.unsqueeze(0)], 0)
        y = model(x.unsqueeze(0)) # we do it sequentially but can be done in parallel if need to optimize
        pose_list.append(y.squeeze(0))
      pose_list.append(pose.permute(2,0,1))
      end_frame = pose[:,:,-1]
    pose_sequence = torch.cat(pose_list, 0)
    torch.save(pose_sequence, args.output_path)
    
  else:
    # Load data
    data = InterpolationDataset(args.dataset_path)
    if args.mode == 'experiment':
      # Split data
      train_test_split = int(len(data) * args.train_test_split)
      indices = list(range(len(data)))
      train_idx, test_idx = indices[:train_test_split], indices[train_test_split:]
      train_sampler = SubsetRandomSampler(train_idx)
      test_sampler = SubsetRandomSampler(test_idx)
      train_loader = DataLoader(data, batch_size=args.batch_size, sampler=train_sampler,
          num_workers=args.num_workers, pin_memory=True)
      test_loader = DataLoader(data, batch_size=args.batch_size, sampler=test_sampler,
          num_workers=args.num_workers, pin_memory=True)
    elif args.mode == 'train':
      # Use all the data
      train_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
          num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters())    
    print("Num params: %d" % sum([p.numel() for p in model.parameters()]))
    print("Num datapoints: %d" % len(data))

    for epoch in range(args.epochs):
      train(args, model, args.gpu, train_loader, optimizer, epoch)
      if args.mode == 'experiment':
        test(model, args.gpu, test_loader)

    if args.mode == 'train':
      torch.save(model.state_dict(), model_savefile)