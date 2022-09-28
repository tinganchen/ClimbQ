import argparse
import os 
# import pdb

parser = argparse.ArgumentParser(description = 'Quantization')


ACT_RANGE = 2
LAMBDA = 1 # step fn. slope modification
LAMBDA2 = 4 # sigmoid scale

PRETRAINED = 'True'
TEST_ONLY = False
STAGE = ''

METHOD = 'ours' 
LOSS = ''
GAMMA = 50
BIT = 4

INDEX = f'q{BIT}_pre32_lt{GAMMA}_0'


## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')
parser.add_argument('--dataset', type = str, default = 'cifar100', help = 'Dataset to train')
parser.add_argument('--lt_gamma', type = int, default = GAMMA)
parser.add_argument('--lt_loss', type = str, default = LOSS)

parser.add_argument('--data_dir', type = str, default = '/home/tingan123/dataset/cifar100/', help = 'The directory where the input data is stored.')
parser.add_argument('--csv_dir', type = str, default = 'data/cifar100/', help = 'The directory where the input data is stored.')
parser.add_argument('--job_dir', type = str, default = f'experiment/{METHOD}_{LOSS}/{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'
# 1. train: default = f'experiment/resnet/t_{T}_sigma_{SIGMA}_lambda_{LAMBDA}_{INDEX}/'
# 2. fine_tuned: default = f'experiment/resnet/ft_thres_{THRES}_t_{T}_sigma_{SIGMA}_lambda_{LAMBDA}_{INDEX}/'

parser.add_argument('--pretrained',  type = str, default = PRETRAINED, help = 'Load pruned model')
parser.add_argument('--test_only', action = 'store_true', default = TEST_ONLY, help = 'Load pruned model')
parser.add_argument('--stage', type = str, default = STAGE, help = 'Load pruned model')
parser.add_argument('--method', type = str, default = METHOD, help = 'Load pruned model')

parser.add_argument('--source_dir', type = str, default = 'pretrained/', help = 'The directory where the teacher model saved.')
parser.add_argument('--source_file', type = str, default = f't_32_pre32_lt_{GAMMA}_0/checkpoint/model_best.pt', help = 'The file the teacher model weights saved as.')
#parser.add_argument('--source_dir', type = str, default = f'experiment/{METHOD}_{LOSS}/', help = 'The directory where the teacher model saved.')
#parser.add_argument('--source_file', type = str, default = f'q4_pre32_lt{GAMMA}_0/checkpoint/model_best.pt', help = 'The file the teacher model weights saved as.')

parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')

parser.add_argument('--refine', type = str, default = None, help = 'Path to the model to be fine tuned.') # None
#parser.add_argument('--refine', type = str, default = f'experiment/resnet/t_{T}_mask_{MASK}_sigma_{SIGMA}_lambda_{LAMBDA}_{LAMBDA2}_kd_{KD}_{INDEX}/checkpoint/model_best.pt', help = 'Path to the model to be fine tuned.') # None


## Training
parser.add_argument('--bitW', type = int, default = BIT, help = 'Quantized bitwidth.') # None
parser.add_argument('--abitW', type = int, default = BIT, help = 'Quantized bitwidth.') # None

parser.add_argument('--target_model', type = str, default = 'resnet56_quant', help = 'The target model.')

parser.add_argument('--source_model', type = str, default = 'resnet_56', help = 'The baseline model.') 
parser.add_argument('--num_epochs', type = int, default = 200, help = 'The num of epochs to train.') # 100
# 1. train: default = 100
# 2. fine_tuned: default = 50

parser.add_argument('--train_batch_size', type = int, default = 128, help = 'Batch size for training.')
parser.add_argument('--eval_batch_size', type = int, default = 100, help = 'Batch size for validation.')

parser.add_argument('--momentum', type = float, default = 0.9, help = 'Momentum for MomentumOptimizer.')
parser.add_argument('--lr', type = float, default = 0.04)
parser.add_argument('--lr_gamma', type = float, default = 0.1)

parser.add_argument('--lr_decay_step', type = int, default = 30)
parser.add_argument('--lr_decay_steps', type = list, default = [80, 150])
# 1. train: default = 30
# 2. fine_tuned: default = 30

parser.add_argument('--alpha', type = float, default = 0.05, help='learning rate (default: 0.001)')# 1. train: default = 0.01
parser.add_argument('--beta_factor', type = float, default = 0.999, help='learning rate (default: 0.001)')# 1. train: default = 0.01

parser.add_argument('--mask_step', type = int, default = 200, help = 'The frequency of mask to update') # 800
parser.add_argument('--weight_decay', type = float, default = 1e-4, help = 'The weight decay of loss.')

parser.add_argument('--random', action = 'store_true', help = 'Random weight initialize for finetune')

parser.add_argument('--keep_grad', action = 'store_true', help = 'Keep gradients of mask for finetune')
parser.add_argument('--lam', type = float, default = LAMBDA, help = 'Modify the approximated slope of step function.')
parser.add_argument('--lam2', type = float, default = LAMBDA2, help = 'Scale the sigmoid function.')
parser.add_argument('--act_range', type = float, default = ACT_RANGE, help = 'Scale the sigmoid function.')

## Status
parser.add_argument('--print_freq', type = int, default = 50, help = 'The frequency to print loss.')

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

