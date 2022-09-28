import os
import numpy as np
import utils.common as utils
from utils.options import args
from tensorboardX import SummaryWriter
from importlib import import_module
import utils.lt_loss as lt_loss

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR
import json

from data import cifar100_lt

import warnings

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')


def main():

    start_epoch = 0
    best_prec1 = 0.0
    best_prec5 = 0.0

    # Data loading
    print('=> Preparing data..')
    loader = cifar100_lt.Data(args)

    # Create model
    print('=> Building model...')

    
    if args.method == 'ours':
        ARCH = 'resnet'
    else: 
        ARCH = f'resnet_{args.method}' 
    
    model_t = import_module(f'model.{ARCH}').__dict__[args.target_model](bitW = args.bitW, abitW = args.abitW, stage = args.stage).to(device)
    
    # Load pretrained weights
    if args.pretrained == 'True':
        ckpt = torch.load(args.source_dir + args.source_file, map_location = device)
        state_dict = ckpt['state_dict_t']
    
        model_dict_t = model_t.state_dict()
        
        for name, param in state_dict.items():
            if name in list(model_dict_t.keys()):
                model_dict_t[name] = param
        
        model_t.load_state_dict(model_dict_t)
        model_t = model_t.to(device)
        
        del ckpt, state_dict, model_dict_t
        
       
    models = [model_t]
    
    
    param_t = [param for name, param in model_t.named_parameters()]
    
    optimizer_t = optim.SGD(param_t, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    
    scheduler_t = MultiStepLR(optimizer_t, args.lr_decay_steps, gamma = args.lr_gamma)


    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location=device)
        best_prec1 = ckpt['best_prec1']
        start_epoch = ckpt['epoch']

        model_t.load_state_dict(ckpt['state_dict_t'])

        optimizer_t.load_state_dict(ckpt['optimizer_t'])

        scheduler_t.load_state_dict(ckpt['scheduler_t'])
        
        print('=> Continue from epoch {}...'.format(start_epoch))

    
    if args.test_only:
        test_prec1, test_prec5 = test(args, loader.loader_test, model_t)
        print('=> Test Prec@1: {:.2f}'.format(test_prec1))
        return
   

    optimizers = [optimizer_t]
    schedulers = [scheduler_t]

    
    block_bits = None
    
    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)
        
        train(args, loader.loader_train, models, optimizers, epoch, block_bits)
        test_prec1, test_prec5 = test(args, loader.loader_test, model_t, epoch)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)
        
        
        state = {
            'state_dict_t': model_t.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            
            #'optimizer': optimizer.state_dict(),
            'optimizer_t': optimizer_t.state_dict(),
            
            #'scheduler': scheduler.state_dict(),
            'scheduler_t': scheduler_t.state_dict(),
            
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        
    print_logger.info(f"Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")



       
def train(args, loader_train, models, optimizers, epoch, block_bits = None):
    losses_t = utils.AverageMeter()

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model_t = models[0]
     
    param_t = []
    
    for name, param in model_t.named_parameters():
        param.requires_grad = True
        param_t.append((name, param))
            
    #cross_entropy = nn.CrossEntropyLoss()
    
    class_num = 100
    sample_num_per_cls = [500]
    r = (1/args.lt_gamma)**(1/(class_num-1))
    
    for i in range(1, class_num):
        sample_num_per_cls.append(int(sample_num_per_cls[-1]*r))
            
    loss = lt_loss.HomoVar_loss(sample_num_per_cls, class_num, args.alpha, args.beta_factor) 
    
    optimizer_t = optimizers[0]
    
    # switch to train mode
    model_t.train()
        
    num_iterations = len(loader_train)

    
    for i, (inputs, targets) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        count_mat = torch.zeros([inputs.size(0), 100])
        for j in range(inputs.size(0)):
            count_mat[j][int(targets[j])] = 1.
        
        mean_mat = torch.mean(count_mat, 0)
        p = [mean_mat[int(target)] for target in targets]
        p = torch.Tensor(p).to(device).reshape([inputs.size(0), 1, 1, 1])
        
        optimizer_t.zero_grad()
    
        ## train weights
        output_t, features = model_t(inputs, p)
        
        if args.lt_loss != '':
            error_t = loss(output_t, targets)
        else:
            error_t = loss(output_t, targets, features) #cross_entropy(output_t, targets)

        error_t.backward() 
        
        losses_t.update(error_t.item(), inputs.size(0))
        
        writer_train.add_scalar('Performance_loss', error_t.item(), num_iters)

   
        optimizer_t.step()

        ## evaluate
        prec1, prec5 = utils.accuracy(output_t, targets, topk = (1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        
        writer_train.add_scalar('Train-top-1', top1.avg, num_iters)
        writer_train.add_scalar('Train-top-5', top5.avg, num_iters)
        
        if i % args.print_freq == 0:
            print_logger.info(
                    'Epoch[{0}]({1}/{2}): \n'
                    'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, num_iterations, 
                    train_loss = losses_t,
                    top1 = top1, 
                    top5 = top5))
                
        
 
            
def test(args, loader_test, model_t, epoch = 0):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model_t.eval()
    
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model_t(inputs.to(device), [0])
            loss = cross_entropy(logits, targets)
            
            writer_test.add_scalar('Test_loss', loss.item(), num_iters)
        
            prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
            writer_test.add_scalar('Test-top-1', top1.avg, num_iters)
            writer_test.add_scalar('Test-top-5', top5.avg, num_iters)
        
    print_logger.info('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
                      '===============================================\n'
                      .format(top1 = top1, top5 = top5))

    return top1.avg, top5.avg
    


if __name__ == '__main__':
    main()

