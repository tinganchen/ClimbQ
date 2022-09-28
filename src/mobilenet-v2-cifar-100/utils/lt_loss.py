import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import Counter
from utils.options import args
import scipy.stats

device = torch.device(f"cuda:{args.gpus[0]}")

class HomoVar_loss(nn.Module):

    def __init__(self, sample_num_per_cls, class_num, alpha = 0.05, beta_factor = 0.999, loss_type = 'softmax'):
        super(HomoVar_loss, self).__init__()
        self.sample_num_per_cls = torch.Tensor(sample_num_per_cls).to(device)
        self.class_num = class_num
        self.alpha = alpha
        self.loss_type = loss_type
        self.beta_factor = beta_factor

    def forward(self, logits, labels, features):
        # one-hot labels
        labels_one_hot = F.one_hot(labels, self.class_num).float().to(device)
        
        # class sample scale factor
        ## class-level feature variances
        n_batch = len(features)
        
        Xij = torch.zeros([features.shape[0], self.class_num, features.shape[1]]).to(device)
        for n in range(n_batch):
            feat = features[n]
            label = labels[n]
            Xij[n][label] = feat
            
        Xij = Xij.transpose(1, 2)
        
        Xi_mean = torch.sum(Xij, 0) / self.sample_num_per_cls.reshape([1, -1])
        
        zij = torch.sum(torch.abs((Xij - Xi_mean)*(Xij != 0.)), 1)
        
        zi_mean = torch.sum(zij, 0) / self.sample_num_per_cls.reshape([1, -1])
        
        z_mean = torch.mean(zi_mean)
        
        ## hypothesis statistic
        N = sum(self.sample_num_per_cls.cpu())
        k = self.class_num
        f_score = scipy.stats.f.ppf(q = 1-self.alpha, 
                                    dfn = k-1, 
                                    dfd = N-k)
        
        ssw = torch.sum((zij - zi_mean)**2 * (zij != 0)) / (N-k)
        sb = (zi_mean - z_mean)**2 * (self.sample_num_per_cls.reshape([1, -1]))
        ssb = torch.sum(sb)  / (k-1)
        
        C = f_score * ssw * (k-1) - (ssb * (k-1) - sb)
        
        ## coefficients
        ### a*n**2 + b*n + c <= 0
        a = z_mean**2
        b = - (2 * z_mean * torch.sum(zij, 0) + C)
        c = torch.sum(zij, 0)**2
        
        n_lb = torch.abs((-b - torch.sqrt(b**2 - 4*a*c)) / (2*a)).cpu()
        n_ub = torch.abs((-b + torch.sqrt(b**2 - 4*a*c)) / (2*a)).cpu()
        
        
        ## scale factor
        beta = torch.zeros([k])
                              
        for i in range(k):
            n_cls = self.sample_num_per_cls[i].cpu()
            if n_cls < n_lb[0][i]:
                beta[i] = self.beta_factor**(1/(n_lb[0][i]-n_cls))
            elif n_cls > n_ub[0][i]:
                beta[i] = self.beta_factor**(1/(n_cls-n_ub[0][i]))
            else:
                beta[i] = self.beta_factor
        

        effective_num = 1.0 - beta**self.sample_num_per_cls.cpu()
        weights = (1.0 - beta) / effective_num
        weights = weights / torch.sum(weights) * self.class_num

        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1).to(device) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.class_num)
        
        # weighted cross entropy loss
        if self.loss_type == 'sigmoid':
            hv_loss = F.binary_cross_entropy_with_logits(input = logits, target = labels_one_hot, weights = weights)
        elif self.loss_type == 'softmax':
            pred = logits.softmax(dim = 1)
            hv_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return hv_loss


