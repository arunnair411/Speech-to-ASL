import torch
from torch import nn

# Input: (B, 2, 137, 3)
# Output: (B, 10, 137, 3) 
class PoseInterpolation(nn.Module):
  def __init__(self, num_hidden_layers=8, hidden_dim=512):
    super(PoseInterpolation, self).__init__()
    self.num_hidden_layers = num_hidden_layers
    self.input_layer = nn.Linear(2*137*3, hidden_dim)
    self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)])
    self.output_layer = nn.Linear(hidden_dim, 10*137*3)
    self.act = torch.selu

  def forward(self, x):
    batch_size = x.size(0)
    x = x.view(batch_size, 2*137*3)
    x = self.input_layer(x)
    x = self.act(x)
    for layer in self.hidden_layers:
      x = layer(x)
      x = self.act(x)
    x = self.output_layer(x)
    x = x.view(batch_size, 10, 137, 3)
    return x

if __name__ == "__main__":
  model = PoseInterpolation()
  x = torch.randn(64, 2, 137, 3)
  x = model(x)
  print(x.size())