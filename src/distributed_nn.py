from __future__ import print_function

import sys
import math
import threading
import argparse
import time

from mpi4py import MPI

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

from nn_ops import NN_Trainer, accuracy
from data_loader_ops.my_data_loader import DataLoader

# from coding import search_w # ~ never used
from util import * # ~ will import a lot of stuff from other files like master, worker and models

# ~ logging is a means of tracking events that happen when some software runs
import logging

logging.basicConfig() # ~ does basic configuration for the logging system by creating a StreamHandler with a default Formatter and adding it to the root logger.
logger = logging.getLogger() # ~ return a logger with the specified name

# ~ sets the threshold for this logger to level. Logging messages which are less severe than level will be ignored; 
# logging messages which have severity level or higher will be emitted by whichever handler or handlers service this logger, unless a handler’s level has been set to a higher severity level than level.
logger.setLevel(logging.INFO)


# ~ util.prepare() is storing all these arguments in variables 
# parser.add_argument() converts dashes to underscoress
def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)') # ~ default value is used
    parser.add_argument('--max-steps', type=int, default=10000, metavar='N',
                        help='the maximum number of iterations')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training') # ~ default value is used, never used
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)') # ~ default value is used, used only by cyclic
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status') # ~ default value is used
    parser.add_argument('--network', type=str, default='LeNet', metavar='N',
                        help='which kind of network we are going to use, support LeNet and ResNet currently')
    parser.add_argument('--mode', type=str, default='normal', metavar='N',
                        help='determine if we use normal averaged gradients or geometric median (in normal mode)\
                         or whether we use normal/majority vote in coded mode to udpate the model')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='which dataset used in training, MNIST and Cifar10 supported currently')
    parser.add_argument('--comm-type', type=str, default='Bcast', metavar='N',
                        help='which kind of method we use during the mode fetching stage')
    parser.add_argument('--err-mode', type=str, default='rev_grad', metavar='N',
                        help='which type of byzantine err we are going to simulate rev_grad/constant/random are supported') # ~ default value is used
    parser.add_argument('--approach', type=str, default='maj_vote', metavar='N',
                        help='method used to achieve byzantine tolerence, currently majority vote is supported set to normal will return to normal mode') # ~ CHECK THIS maj_vote
    parser.add_argument('--num-aggregate', type=int, default=5, metavar='N',
                        help='how many number of gradients we wish to gather at each iteration') # ~ default value is used, never used after
    parser.add_argument('--eval-freq', type=int, default=50, metavar='N',
                        help='it determines per how many step the model should be evaluated')
    parser.add_argument('--train-dir', type=str, default='output/models/', metavar='N',
                        help='directory to save the temp model during the training process for evaluation') # ~ default value is used
    parser.add_argument('--adversarial', type=int, default=1, metavar='N',
                        help='how much adversary we want to add to a certain worker')
    parser.add_argument('--worker-fail', type=int, default=2, metavar='N',
                        help='how many number of worker nodes we want to simulate byzantine error on')
    parser.add_argument('--group-size', type=int, default=5, metavar='N',
                        help='in majority vote how many worker nodes are in a certain group')
    parser.add_argument('--bucket-size', type=int, default=0, metavar='N',
                        help='bucket size only for draco lite') # ~ default value is used
    parser.add_argument('--compress-grad', type=str, default='compress', metavar='N',
                        help='compress/none indicate if we compress the gradient matrix before communication')
    parser.add_argument('--checkpoint-step', type=int, default=0, metavar='N',
                        help='which step to proceed the training process')
    parser.add_argument('--lis-simulation', type=str, default="None", metavar='N',
                        help='To simulate or not to simulate the A Little Is Enough paper')
    parser.add_argument('--local-remote', type=str, default="remote", metavar='N',
                        help='To run the algorithm on a single machine or distributively')
    parser.add_argument('--rama-m', type=int, default=-1, metavar='N',
                        help='Ramanujan Case 2 parameter m')
    parser.add_argument('--detox-attack', type=str, default="worst", metavar='N',
                        help='Type of attack on DETOX (worst, benign or whole_group)')
    parser.add_argument('--byzantine-gen', type=str, default="random", metavar='N',
                        help='Type of byzantine set generation (random or hard_coded)')
    parser.add_argument('--gamma', type=float, default=1, metavar='N',
                        help='Learning rate decay (linear)')
    parser.add_argument('--lr-step', type=int, default=100000000000000000, metavar='N',
                        help='Frequency of learning rate decay')
    args = parser.parse_args()
    return args

# ~ test
# writes class variables of "obj" to logger
def writeVarsLog(obj, obj_name):
    logger.info("vars("+obj_name+")")
    objVars = vars(obj) # ~ dictionary with class variables
    for name in objVars:
        var = objVars[name]
        logger.info("\n{}".format(name))
        writeVarSpecs(var, "")


# ~ test 
# goes deep into each dictionary, list, np.ndarray and prints only necessary information in an indented fashion 
def writeVarSpecs(var, indent: str):
    if isinstance(var, dict):
        logger.info(indent+"type(var):{}, len(var): {}".format(type(var), len(var)))
        for key in var:
            logger.info(indent+"\t"+"key: {}".format(key))
            writeVarSpecs(var[key], indent+"\t")
    elif isinstance(var, list):
        logger.info(indent+"type(var):{}, len(var): {}".format(type(var), len(var)))
        for elem in var:
            writeVarSpecs(elem, indent+"\t")
    elif isinstance(var, np.ndarray):
        logger.info(indent+"{} {} {}".format(type(var), var.dtype, var.shape))
    else:
        logger.info(indent+"{}".format(var))


if __name__ == "__main__": # ~ PS and workers will call this
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    # ~ test
    # outputs log to file, "w+" is to overwrite existing file
    setLogFh = False
    if setLogFh:
        fh = logging.FileHandler('logRank'+str(rank)+'.log', 'w+')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    

    # ~ The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv. 
    # The argparse module also automatically generates help and usage messages and issues errors when users give the program invalid arguments.
    # argparse.ArgumentParser() creates a new ArgumentParser object.
    # description: Text to display before the argument help (default: none)
    args = add_fit_args(argparse.ArgumentParser(description='Draco'))

    if rank == 0:
        logger.info("Arguments: {}".format(args))

    datum, kwargs_master, kwargs_worker = prepare(args, rank, world_size) # from util.py
    
    # ~ test
    if setLogFh:
        logger.info("kwargs_master: {}".format(kwargs_master))
        logger.info("kwargs_worker: {}".format(kwargs_worker))
            
    # test
    if rank == 0:        
        logger.info("Attack: Byzantines for 1st iteration are ranks: {}".format(' '.join(map(str, kwargs_worker['adversaries'][0]))))
    
    # test
    # if rank > 0:
        # logger.info("DEBUG: group_seeds from rank {}: {}".format(rank, ' '.join(map(str, kwargs_worker['group_seeds']))))
                
    if args.approach == "baseline": # ~ the baseline approach (see "baseline_master.py" and "baseline_worker.py") is doing all kinds of aggregation except majority, it also has signSGD
        train_loader, _, test_loader = datum
        if rank == 0:
            master_fc_nn = baseline_master.SyncReplicasMaster_NN(comm=comm, **kwargs_master)
            master_fc_nn.build_model()
            # writeVarsLog(master_fc_nn, "master_fc_nn") # ~ test
            logger.info("I am the master: the world size is {}, cur step: {}".format(master_fc_nn.world_size, master_fc_nn.cur_step))
            master_fc_nn.start()
            logger.info("Done sending messages to workers!")
        else:
            worker_fc_nn = baseline_worker.DistributedWorker(comm=comm, **kwargs_worker)
            worker_fc_nn.build_model()
            # writeVarsLog(worker_fc_nn, "worker_fc_nn") # ~ test
            logger.info("I am worker: {} in all {} workers, next step: {}".format(worker_fc_nn.rank, worker_fc_nn.world_size-1, worker_fc_nn.next_step))
            worker_fc_nn.train(train_loader=train_loader, test_loader=test_loader)
            logger.info("Now the next step is: {}".format(worker_fc_nn.next_step))
    # Repitition Code in Draco
    elif args.approach == "maj_vote":
        train_loader, _, test_loader = datum
        if rank == 0:
            coded_master = rep_master.CodedMaster(comm=comm, **kwargs_master)
            coded_master.build_model()
            # writeVarsLog(coded_master, "coded_master") # ~ test
            logger.info("I am the master: the world size is {}, cur step: {}".format(coded_master.world_size, coded_master.cur_step))
            coded_master.start()
            logger.info("Done sending messages to workers!")
        else:
            coded_worker = rep_worker.CodedWorker(comm=comm, **kwargs_worker)
            coded_worker.build_model()
            # writeVarsLog(coded_worker, "coded_worker") # ~ test
            logger.info("I am worker: {} in all {} workers, next step: {}".format(coded_worker.rank, coded_worker.world_size-1, coded_worker.next_step))
            coded_worker.train(train_loader=train_loader, test_loader=test_loader)
            logger.info("Now the next step is: {}".format(coded_worker.next_step))
    elif args.approach == "draco_lite" or args.approach == "draco_lite_attack": # ~ DETOX method, Kostas DETOX attack method
        train_loader, _, test_loader = datum
        if rank == 0:
            coded_master = draco_lite_master.DracoLiteMaster(comm=comm, **kwargs_master) # ~ util.py is importing this
            coded_master.build_model()
            # writeVarsLog(coded_master, "coded_master") # ~ test
            logger.info("I am the master: the world size is {}, cur step: {}".format(coded_master.world_size, coded_master.cur_step))
            coded_master.start()
            logger.info("Done sending messages to workers!")
        else:   
            coded_worker = draco_lite_worker.DracoLiteWorker(comm=comm, **kwargs_worker)
            coded_worker.build_model()
            # writeVarsLog(coded_worker, "coded_worker") # ~ test
            logger.info("I am worker: {} in all {} workers, next step: {}".format(coded_worker.rank, coded_worker.world_size-1, coded_worker.next_step))
            coded_worker.train(train_loader=train_loader, test_loader=test_loader)
            logger.info("Now the next step is: {}".format(coded_worker.next_step))  
    elif args.approach == "mols" or args.approach == "rama_one" or args.approach == "rama_two": # ~ MOLS, Ramanujan Case 1, Ramanujan Case 2
        train_loader, _, test_loader = datum
        if rank == 0:
            coded_master = byzshield_master.ByzshieldMaster(comm=comm, **kwargs_master) # ~ util.py is importing this
            coded_master.build_model()
            # writeVarsLog(coded_master, "coded_master") # ~ test
            logger.info("ByzShield: I am the master: the world size is {}, cur step: {}".format(coded_master.world_size, coded_master.cur_step))
            coded_master.start()
            logger.info("ByzShield: Done sending messages to workers!")
        else:   
            coded_worker = byzshield_worker.ByzshieldWorker(comm=comm, **kwargs_worker)
            coded_worker.build_model()
            # writeVarsLog(coded_worker, "coded_worker") # ~ test
            logger.info("ByzShield: I am worker: {} in all {} workers, next step: {}".format(coded_worker.rank, coded_worker.world_size-1, coded_worker.next_step))
            coded_worker.train(train_loader=train_loader, test_loader=test_loader)
            logger.info("ByzShield: Now the next step is: {}".format(coded_worker.next_step))
        