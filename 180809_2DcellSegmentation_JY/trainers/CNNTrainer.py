import os
from multiprocessing import Pool, Queue, Process

import scipy
import utils

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from .BaseTrainer import BaseTrainer

# from sklearn.metrics import f1_score, confusion_matrix, recall_score, jaccard_similarity_score, roc_curve, precision_recall_curve

class CNNTrainer(BaseTrainer):
    def __init__(self, arg, G, torch_device, recon_loss, logger):
        super(CNNTrainer, self).__init__(arg, torch_device, logger)
        self.recon_loss = recon_loss
        
        self.G = G
        self.optim = torch.optim.Adam(self.G.parameters(), lr=arg.lrG, betas=arg.beta)
            
        self.best_metric = 100000
        self.sigmoid = nn.Sigmoid().to(self.torch_device)

        self.load()
        self.prev_epoch_loss = 0


    def save(self, epoch, filename="models"):

        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)
        torch.save({"model_type" : self.model_type,
                    "start_epoch" : epoch + 1,
                    "network" : self.G.state_dict(),
                    "optimizer" : self.optim.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_path + "/%s.pth.tar"%(filename))
        print("Model saved %d epoch"%(epoch))


    def load(self, filename="models.pth.tar"):
        if os.path.exists(self.save_path +"/"+ filename) is True:
            print("Load %s File"%(self.save_path))            
            ckpoint = torch.load(self.save_path + "/"+ filename)
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s"%(ckpoint["model_type"]))

            self.G.load_state_dict(ckpoint['network'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d"%(ckpoint["model_type"], self.start_epoch))
        else:
            print("Load Failed, not exists file")


    def train(self, train_loader, val_loader=None):
        print("\nStart Train")

        for epoch in range(self.start_epoch, self.epoch):
            for i, (input_, target_, _) in enumerate(train_loader):    
                self.G.train()
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                print(input_.device)
                output_ = self.G(input_)
                recon_loss = self.recon_loss(output_, target_)
                
                self.optim.zero_grad()
                recon_loss.backward()
                self.optim.step()
            
                if (i % 20) == 0:
                    self.logger.will_write("[Train] epoch:%d loss:%f"%(epoch, recon_loss))

            if val_loader is not None:            
                self.valid(epoch, val_loader)
            else:
                self.save(epoch)
        print("End Train\n")

    def _test_foward(self, input_, target_):
        input_  = input_.to(self.torch_device)
        output_ = self.G(input_).type(torch.FloatTensor)
        target_ = target_.type(torch.FloatTensor)
        input_  = input_.type(torch.FloatTensor)

        return input_, output_, target_

    # TODO : Metric 정하기 
    def valid(self, epoch, val_loader):
        self.G.eval()
        with torch.no_grad():
            losssum=0
            count=0;
            for i, (input_, target_, _) in enumerate(val_loader):
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                _, output_, target_ = self._test_foward(input_, target_)
                loss=self.recon_loss(output_,target_)
                losssum=losssum+loss
                count=count+1

            if losssum/count < self.best_metric:
                self.best_metric = losssum/count
                self.save(epoch,"epoch[%04d]_losssum[%f]"%(epoch, losssum/count))

            self.logger.write("[Val] epoch:%d losssum:%f "%(epoch, losssum/count))
                    
    # TODO: Metric, save file 정하기
    def test(self, test_loader):
        print("\nStart Test")
        self.G.eval()

        if os.path.exists(self.save_path+'/result') is False:
            os.mkdir(self.save_path+'/result')

        with torch.no_grad():
            for i, (input_, target_, fname) in enumerate(test_loader):

                if(i>500):
                    break

                input_, output_, target_ = self._test_foward(input_, target_)
                data={}
                data['input']=input_.numpy()
                data['output']=output_.numpy()
                data['target']=target_.numpy()
                scipy.io.savemat(self.save_path + "/result/%d.mat"%(i), data)
                self.logger.will_write("[Save] fname:%s "%(fname[0][:-4]))
        print("End Test\n")
