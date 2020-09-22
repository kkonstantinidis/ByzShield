from __future__ import print_function
import time
import copy
import math
from sys import getsizeof

from mpi4py import MPI
import numpy as np
import hdmedians as hd
from scipy import linalg as LA
from scipy import fftpack as FT
from scipy.optimize import lsq_linear
import torch

import sys
sys.path.append("..")
from nn_ops import NN_Trainer
from optim.sgd_modified import SGDModified
from compress_gradient import decompress # ~ for gradient decompression after receiving it
from compression import g_decompress, w_compress
#import c_coding
from util import * # ~ will import models from "model_ops", ...
import warnings

STEP_START_ = 1

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

class GradientAccumulator(object):
    '''a simple class to implement gradient aggregator like the `Conditional Accumulators` in tensorflow'''
    def __init__(self, module, num_worker, mode='None'):
        # we will update this counter dynamically during the training process
        # the length of this counter should be number of fc layers in the network
        # we used list to contain gradients of layers
        self.gradient_aggregate_counter = []
        self.model_index_range = []
        self.gradient_aggregator = [] # ~ list (one element per model parameter) of lists (one element per worker)
        self._mode = mode
        
        # ~ for each torch.nn.parameter.Parameter (one per layer)
        for param_idx, param in enumerate(module.parameters()):
            tmp_aggregator = []
            for worker_idx in range(num_worker):
                if self._mode == 'None':
                    tmp_aggregator.append(np.zeros((param.size())))
                elif self._mode == 'compress':
                    _shape = param.size()
                    # ~ the size used here is for Python bytearray() after decompression, the *2 factor here is to make sure the MPI buffer can hold the received data 
                    # but it's not exact value.
                    # see https://bitbucket.org/mpi4py/mpi4py/issues/65/mpi_err_truncate-message-truncated-when
                    # see https://stackoverflow.com/questions/59559597/mpi4py-irecv-causes-segmentation-f
                    if len(_shape) == 1:
                        tmp_aggregator.append(bytearray(getsizeof(np.zeros((_shape[0],)))*2))
                    else:
                        tmp_aggregator.append(bytearray(getsizeof(np.zeros(_shape))*2))
                        
                    # test
                    # ~ multiplies buffer size by 4 to see if anything changes, not really, see above
                    # if len(_shape) == 1:
                        # tmp_aggregator.append(bytearray(getsizeof(np.zeros((_shape[0],)))*4))
                    # else:
                        # tmp_aggregator.append(bytearray(getsizeof(np.zeros(_shape))*4))
                    
            # initialize the gradient aggragator
            self.gradient_aggregator.append(tmp_aggregator)
            self.gradient_aggregate_counter.append(0)
            self.model_index_range.append(param_idx)

    def meset_everything(self):
        self._meset_grad_counter()
        self._meset_grad_aggregator()

    def _meset_grad_counter(self):
        self.gradient_aggregate_counter = [0 for _ in self.gradient_aggregate_counter]

    def _meset_grad_aggregator(self):
        '''
        reset the buffers in grad accumulator, not sure if this is necessary
        '''
        if self._mode == 'compress':
            pass
        else:
            for i, tmp_aggregator in enumerate(self.gradient_aggregator):
                for j, buf in enumerate(tmp_aggregator):
                    self.gradient_aggregator[i][j] = np.zeros(self.gradient_aggregator[i][j].shape)


# ~ custom accumulator since we need to store multiple files (and their corresponding model layers) per receive buffer at the PS
class ByzShieldGradientAccumulator(GradientAccumulator):
    '''a simple class to implement gradient aggregator like the `Conditional Accumulators` in tensorflow'''
    # ~ ell (computation load per worker) of the ByzShield schemes need to be provided too to allocate enough space
    def __init__(self, module, num_worker, ell, mode='None'):
        # we will update this counter dynamically during the training process
        # the length of this counter should be number of fc layers in the network
        # we used list to contain gradients of layers
        self.gradient_aggregate_counter = []
        self.model_index_range = []
        self.gradient_aggregator = []
        self._mode = mode
        
        for param_idx, param in enumerate(module.parameters()):
            tmp_aggregator = []
            for worker_idx in range(num_worker):
                _shape = param.size()
                # ~ ell factor for first dimension and the rest are the same
                if self._mode == 'None':
                    tmp_aggregator.append(np.zeros((_shape[0]*ell,) + _shape[1:]))
                elif self._mode == 'compress':                   
                    if len(_shape) == 1:
                        tmp_aggregator.append(bytearray(getsizeof(np.zeros((_shape[0]*ell,)))*2))
                    else:
                        tmp_aggregator.append(bytearray(getsizeof(np.zeros((_shape[0]*ell,) + _shape[1:]))*2))
            # initialize the gradient aggragator
            self.gradient_aggregator.append(tmp_aggregator)
            self.gradient_aggregate_counter.append(0)
            self.model_index_range.append(param_idx)

    # ~ probably can be removed, inherited
    def meset_everything(self):
        self._meset_grad_counter()
        self._meset_grad_aggregator()

    # ~ probably can be removed, inherited
    def _meset_grad_counter(self):
        self.gradient_aggregate_counter = [0 for _ in self.gradient_aggregate_counter]

    # ~ probably can be removed, inherited
    def _meset_grad_aggregator(self):
        '''
        reset the buffers in grad accumulator, not sure if this is necessary
        '''
        if self._mode == 'compress':
            pass
        else:
            for i, tmp_aggregator in enumerate(self.gradient_aggregator):
                for j, buf in enumerate(tmp_aggregator):
                    self.gradient_aggregator[i][j] = np.zeros(self.gradient_aggregator[i][j].shape)