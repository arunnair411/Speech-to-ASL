import torch
import torch.nn
from torch.autograd import Function, Variable
import pdb
import math

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

class LSDLoss(torch.nn.Module):
    def __init__(self):
        super(LSDLoss,self).__init__()

    def forward(self,x,y):
        # x and y are LogPowerSpectrograms (LPS) of size B*1*F*T
        # See https://ieeexplore-ieee-org.proxy1.library.jhu.edu/stamp/stamp.jsp?tp=&arnumber=9054563 for details
        F = x.shape[2] # = #FreqBins
        T = x.shape[3] # = #TimeBins
        loss = torch.pow(x-y, 2)    # First, take the different between the LPS and square it
        loss = torch.sum(loss, 2)/F # Second, take the mean over the frequence dimension
        loss = torch.sqrt(loss)     # Third, take the elementwise square root - power -> amplitude Ahmet commented
        loss = torch.mean(loss)     # Fourth, average over B,T and channel(=1) dimension
        return loss

# The function below came with the library - it calculates the dice coefficient between the predicted mask and that of the ground truth
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input.view(-1)) + torch.sum(target.view(-1)) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

class DiceCoeffLoss(torch.nn.Module):
    def __init__(self):
        super(DiceCoeffLoss,self).__init__()
        
    def forward(self,input,target):
        smooth = 0.01
        self.intersection = torch.dot(input.view(-1), target.view(-1))
        self.mask_sum = torch.sum(input.view(-1)) + torch.sum(target.view(-1))

        t = 1- ((2 * self.intersection + smooth) / (self.mask_sum + smooth))
        return t

# The function below came with the library - it calculates the dice coefficient for the entire batch by averaging across the batch. 
# Keeping it to use it for evaluating the predictions.
def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
        # s_2 = s_2 + DiceCoeffLoss().forward(c[0], c[1]) # I did a quick check if the values returned by DiceCoeff() 
        # (which came with the original code) and DiceCoeffLoss() (which I wrote based on the former, to understand it better) 
        # are consistent. They are! s_2+s = 1 (as s_2 should be 1 - s, by definition) This verifies my code and function are right!
    return s / (i + 1)

# Original psnr formulation defined for the CPU
def psnr(img1, img2):
    """Dice coeff for individual examples"""
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Defining a class to handle PSNR loss evaluation for a batch of data
class PSNRCoeff(Function):
    def __init__(self):
        super(PSNRCoeff,self).__init__()        
    def forward(self, input, target):
        self.mse = torch.mean( (input.view(-1)-target.view(-1))**2 )
        if self.mse == 0:
            return 100
        PIXEL_MAX = 1.0
        t = 20 * math.log10(PIXEL_MAX / math.sqrt(self.mse))
        return t

# Calculating PSNR for a batch of data
def psnr_loss(input, target):
    """PSNR loss for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(input, target)):
        s = s + PSNRCoeff().forward(c[0], c[1])
    return s / (i + 1)
