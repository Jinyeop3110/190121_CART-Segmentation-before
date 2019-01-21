import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss3d_ver3(nn.Module):
    def __init__(self, gamma=2, pw=30):
        super().__init__()
        self.gamma = gamma
        self.pw=pw

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        #target=target.to(torch.float)

        #b=2*((torch.clamp(target,min=2,max=3)-2)+(torch.clamp(target,min=3)-3))
        #print(b[0,0,0,0,:])
        a=((target<0.5)).to(torch.float)
        c=((target>=3)).to(torch.float)
        c=(c-a+((target-c)<2).to(torch.float)).to(torch.float)
        b=target-c
        d=(1.0*(b>0)).to(torch.float)
        b=torch.pow(((b-2.0*d)*2.0).to(torch.float),1)
        target=torch.cat((a,c,b),1)
        #weight=(weight.to(torch.float)/100+1.0).to(torch.float)
        #weight=torch.tensor([[1,1,100]]).to(torch.float)

        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))


        #max_val = (-input).clamp(min=0)
        #loss = target*(max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
        loss=-target*F.log_softmax(input,dim=1)
        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        loss = torch.pow(1-F.softmax(input,dim=1), self.gamma) * loss
        #loss[:,2]=loss[:,2]*self.pw
        loss=loss.sum(dim=1,keepdim=True)*(1+self.pw*d)

        return loss.mean()


class FocalLoss3d_ver2(nn.Module):
    def __init__(self, gamma=2, pw=10):
        super().__init__()
        self.gamma = gamma
        self.pw=pw

    def forward(self, input, target, weight):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        target=target.to(torch.float)
        a=(target<0.5).to(torch.float)
        c=(target>1.5).to(torch.float)
        b=target-2*c
        target=torch.cat((a,b,c),1)
        #weight=(weight.to(torch.float)/100+1.0).to(torch.float)
        #weight=torch.tensor([[1,1,100]]).to(torch.float)
        weight=weight

        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        #max_val = (-input).clamp(min=0)
        #loss = target*(max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
        loss=-target*(input.log())

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1

        loss = torch.pow(1-input, self.gamma) * loss
        print(loss.size())
        print(weight.size())
        print(weight)
        loss=torch.sum(loss,dim=1,keepdim=True)*weight

        return loss.mean()

class FocalLoss3d_ver1(nn.Module):
    def __init__(self, gamma=2, pw=30):
        super().__init__()
        self.gamma = gamma
        self.pw=pw

    def forward(self, input, target, weight):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        target=target.to(torch.float)
        a=(target<0.5).to(torch.float)
        c=(target>1.5).to(torch.float)
        b=target-2*c
        target=torch.cat((a,b,c),1)
        #weight=(weight.to(torch.float)/100+1.0).to(torch.float)
        #weight=torch.tensor([[1,1,100]]).to(torch.float)

        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        #max_val = (-input).clamp(min=0)
        #loss = target*(max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
        loss=-target*(input.log())

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1

        loss = torch.pow(1-input, self.gamma) * loss
        loss[:,2]=loss[:,2]*self.pw
        #loss = loss*weight

        return loss.mean()


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
