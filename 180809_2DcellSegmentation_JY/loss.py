import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeWeightedLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        weight=((target<0.5)*1+(0.5<target)*self.alpha+(1.5<target)*self.beta).to(torch.float)
        #isMerged = ((target > 1.5).sum() > 0).to(torch.float)
        target=(target>0.5).to(torch.float)
        loss = (F.binary_cross_entropy(input,target)*weight).to(torch.float)

        return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha, torch_device):
        super().__init__()
        self.alpha = alpha
        self.beta  = 1 - alpha 
        self.smooth = 1.0

    def forward(self, target_, output_):
        output_ = F.sigmoid(output_)

        target_f = target_.contiguous().view(-1)
        output_f = output_.contiguous().view(-1)

        """
        P : set of predicted, G : ground truth label
        Tversky Index S is
        S(P, G; a, b) = PG / (PG + aP\G + bG\P)

        Tversky Loss T is
        PG = sum of P * G
        G\P = sum of G not P
        P\G = sum of P not G
        T(a, b) = PG / (PG + aG\P + bP\G)
        """

        PG = (target_f * output_f).sum()
        G_P = ((1 - target_f) * output_f).sum()
        P_G = ((1 - output_f) * target_f).sum()

        loss = (PG + self.smooth) / (PG + (self.alpha * G_P) + (self.beta * P_G) + self.smooth)
        return loss

if __name__ == "__main__":
    target = torch.tensor([[0,1,0],[1,1,1],[0,1,0]], dtype=torch.float)
    output = torch.tensor([[1,1,0],[0,0,0],[1,0,0]], dtype=torch.float)

    loss = TverskyLoss(0.3, torch.device("cpu"))
    print("Loss : ", loss(target, output))
