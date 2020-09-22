from __future__ import print_function
import os.path
import time
import argparse # ~ for comments on argparse see "distributed_nn.py"
from datetime import datetime
import copy

from mpi4py import MPI
import numpy as np

from nn_ops import NN_Trainer

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN

from model_ops.lenet import LeNet, LeNetSplit
from model_ops.resnet import *
from model_ops.resnet_split import *
from model_ops.densenet import *

from util import build_model

import logging # ~ for comments on logging see "distributed_nn.py"

from signal import signal, SIGINT
from sys import exit

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ~ see "nn_ops\__init__.py" for comments
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Validation settings
    parser.add_argument('--eval-batch-size', type=int, default=10000, metavar='N',
                        help='the batch size when doing model validation, complete at once on default')
    parser.add_argument('--eval-freq', type=int, default=50, metavar='N',
                        help='it determines per how many step the model should be evaluated')
    parser.add_argument('--model-dir', type=str, default='output/models/', metavar='N',
                        help='directory to save the temp model during the training process for evaluation')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='which dataset used in training, MNIST and Cifar10 supported currently')
    parser.add_argument('--network', type=str, default='LeNet', metavar='N',
                        help='which kind of network we are going to use, support LeNet and ResNet currently')
    parser.add_argument('--cur-step', type=int, default=0, metavar='N',
                        help='initial step to fetch')
    parser.add_argument('--last-step', type=int, default=0, metavar='N',
                        help='last step to fetch, must exist')
    args = parser.parse_args()
    return args

class DistributedEvaluator(NN_Trainer):
    '''
    The DistributedEvaluator aims at providing a seperate node in the distributed cluster to evaluate
    the model on validation/test set and return the results
    In this version, the DistributedEvaluator will only load the model from the dir where the master
    save the model and do the evaluation task based on a user defined frequency 
    '''
    def __init__(self, **kwargs):
        self._last_step = kwargs['last_step']
        self._cur_step = kwargs['cur_step']
        self._model_dir = kwargs['model_dir']
        self._eval_freq = int(kwargs['eval_freq'])
        self._eval_batch_size = kwargs['eval_batch_size']
        self.network_config = kwargs['network']
        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []
        
        # ~ for file to write loss, top-1 and top-5 accuracies
        self.outputF = open(self._model_dir + self._model_dir.split('/')[-2] + ".txt", "w+")
        self.test_losses = "test_loss = [" # ~ will be Matlab-style
        self.prec1s = "prec1 = ["
        self.prec5s = "prec5 = ["
        
        # ~ Handler, cleanup here after Ctrl+C is detected
        def handler(signal_received, frame):
            logger.info("Ctrl+C detected, finalizing...")
            self.outputF.write(self.test_losses+"]\n")
            self.outputF.write(self.prec1s+"]\n")
            self.outputF.write(self.prec5s+"]\n")
            self.outputF.close()
            exit(0)
        
        # ~ Tell Python to run the handler() function when SIGINT is received
        signal(SIGINT, handler)
        
        # ~ just to count the number of points evaluated
        self.point_ctr = 0
        

    def evaluate(self, validation_loader):
        # init objective to fetch at the begining
        self._next_step_to_fetch = self._cur_step + self._eval_freq
        self.point_ctr = self.point_ctr + 1
        self._num_batch_per_epoch = len(validation_loader) / self._eval_batch_size
        # check if next temp model exsits, if not we wait here else we continue to do the model evaluation
        while True:
        
            # ~ stop if user has specified last step to evaluate and this has been reached, do not wait for Ctrl+C
            if self._last_step != 0 and self._next_step_to_fetch > self._last_step:
                break
                
            model_dir_=self._model_dir_generator(self._next_step_to_fetch)
            if os.path.isfile(model_dir_):
                self._load_model(model_dir_)
                logger.info("Evaluator evaluating results on step {},{}".format(self.point_ctr, self._next_step_to_fetch))
                self._evaluate_model(validation_loader)
                self._next_step_to_fetch += self._eval_freq
                self.point_ctr = self.point_ctr + 1
            else:
                time.sleep(10)
                
                # ~ this assumes that all models to test are available, it won't wait
                # break
        
        # ~ the following will wrap up and close the file if a "self._max_step" has been specified by the user, we don't need SIGINT in this case
        self.outputF.write(self.test_losses+"]\n")
        self.outputF.write(self.prec1s+"]\n")
        self.outputF.write(self.prec5s+"]\n")
        self.outputF.close()

    def _evaluate_model(self, test_loader):
        self.network.eval()
        test_loss = 0
        correct = 0
        prec1_counter_ = prec5_counter_ = batch_counter_ = 0
        for data, y_batch in test_loader:
            data, target = Variable(data), Variable(y_batch)
            output = self.network(data)
            # test_loss += F.nll_loss(F.log_softmax(output), target, size_average=False).item() 
            test_loss += F.nll_loss(F.log_softmax(output, dim = 1), target, reduction='sum').item() # ~ this is to avoid warnings of previous line, may not be correct
            prec1_tmp, prec5_tmp = accuracy(output.detach(), y_batch, topk=(1, 5))
            prec1_counter_ += prec1_tmp.numpy()[0]
            prec5_counter_ += prec5_tmp.numpy()[0]
            batch_counter_ += 1
        prec1 = prec1_counter_ / batch_counter_
        prec5 = prec5_counter_ / batch_counter_
        test_loss /= len(test_loader.dataset)
        logger.info('Test set: Average loss: {:.4f}, Prec@1: {} Prec@5: {}'.format(test_loss, prec1, prec5))
        self.test_losses += str(test_loss) + ","
        self.prec1s += str(prec1) + ","
        self.prec5s += str(prec5) + ","

    def _load_model(self, file_path):
        self.network = build_model(self.network_config, num_classes=10)
        with open(file_path, "rb") as f_:
            self.network.load_state_dict(torch.load(f_))

    def _model_dir_generator(self, next_step_to_fetch):
        return self._model_dir+"model_step_"+str(next_step_to_fetch)

if __name__ == "__main__":    
    # this is only a simple test case
    args = add_fit_args(argparse.ArgumentParser(description='PyTorch Distributed Evaluator'))

    # load training and test set here:
    if args.dataset == "MNIST":
        test_loader = torch.utils.data.DataLoader(
            # datasets.MNIST('../data', train=False, download=True, # ~ changed to make it same as in util.py
            datasets.MNIST('./mnist_data', train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=args.eval_batch_size, shuffle=True)
    elif args.dataset == "Cifar10":
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        testset = datasets.CIFAR10(root='./cifar10_data', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size,
                                                 shuffle=False)
    elif args.dataset == "Cifar100":
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        testset = datasets.CIFAR100(root='./cifar100_data', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size,
                                                 shuffle=False)
    elif args.dataset == 'SVHN':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        testset = SVHN(root='./svhn_data', split='test',
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size,
                                                 shuffle=False)
    
    kwargs_evaluator={
                    'network':args.network,
                    'model_dir':args.model_dir, 
                    'eval_freq':args.eval_freq, 
                    'eval_batch_size':args.eval_batch_size,
                    'cur_step':args.cur_step,
                    'last_step':args.last_step}
    evaluator_nn = DistributedEvaluator(**kwargs_evaluator)
    evaluator_nn.evaluate(validation_loader=test_loader)
    # print("I am worker: {} in all {} workers".format(worker_fc_nn.rank, worker_fc_nn.world_size)) # ~ invalid