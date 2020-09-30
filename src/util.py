import random

import numpy as np
from torchvision import datasets, transforms

from model_ops.lenet import LeNet, LeNetSplit
from model_ops.resnet import *
from model_ops.resnet_split import *
from model_ops.vgg import *
from model_ops.densenet import *
from model_ops.fc_nn import FC_NN, FC_NN_Split
from model_ops.utils import err_simulation
from model_ops.utils import float_type

from torchvision.datasets import SVHN
from coding import search_w
from master import baseline_master, rep_master, cyclic_master, draco_lite_master, byzshield_master
from worker import baseline_worker, rep_worker, cyclic_worker, draco_lite_worker, byzshield_worker

from itertools import chain # ~ for the attack

SEED_ = 428

# ~ hard-coded Byzantines for a single iteration (only for debugging, discontinued)
# hard_byz_set = np.array([1,8]) # ~ test
# hard_byz_set = np.array([40,41]) # ~ test
# hard_byz_set = np.array([]) # q=0
# hard_byz_set = np.array([1,2,3,6,7,11,25]) # K=25, l=5, r=5, Rama Case 2, q=7
# hard_byz_set = np.array([1,2,6,7,10,15,17,25]) # K=25, l=5, r=5, Rama Case 2, q=8
# hard_byz_set = np.array([1,2,3,6,7,13,14,16,22]) # K=25, l=5, r=5, Rama Case 2, q=9

# ~ hard_byz_set: dictionary, key: K, value: dictionary (key: q, value: list with worst-case Byzantine workers)
# Caution: this supports only one particular scheme for each value of K (hard-coded)
hard_byz_set = {
                # K=15, l=5, r=3, MOLS/Rama Case 1, worst-case Byzantine sets for some values of q
                15: {0:np.array([]), 
                1:np.array([1]),
                2:np.array([1,6]),
                3:np.array([1,6,12]),
                4:np.array([1,2,6,12]),
                5:np.array([1,2,6,7,14]),
                6:np.array([1,2,6,8,12,13]),
                7:np.array([1,2,3,6,8,11,12]),
                8:np.array([1,2,3,4,5,6,7,8])}, # ~ use q >= 8 just for debugging
            
                # K=35, l=7, r=5, MOLS/Rama Case 1, worst-case Byzantine sets for some values of q
                35: {0:np.array([]), 
                1:np.array([1]),
                2:np.array([1,2]),
                3:np.array([1,8,15]),
                4:np.array([1,2,8,15]),
                5:np.array([1,2,8,9,33]),
                6:np.array([1,2,8,11,21,22]),
                7:np.array([1,2,3,8,9,15,25]),
                8:np.array([1,2,8,10,19,21,29,34]),
                9:np.array([1,8,9,11,20,22,24,25,32]),
                10:np.array([1,2,3,4,8,15,18,25,33,34]),
                11:np.array([1,2,3,8,10,12,15,20,28,31,32]),
                12:np.array([1,2,3,4,8,9,10,17,21,22,30,34]),
                13:np.array([1,2,3,8,9,10,16,22,26,28,31,32,33])},
            
                # K=25, l=5, r=5, Rama Case 2, worst-case Byzantine sets for some values of q
                25: {0:np.array([]), 
                1:np.array([1]),
                2:np.array([1,2]),
                3:np.array([1,6,11]),
                4:np.array([1,2,6,11]),
                5:np.array([1,2,6,7,19]),
                6:np.array([1,2,6,8,15,21]),
                7:np.array([1,2,3,6,7,11,25]),
                8:np.array([1,2,6,7,11,15,17,25]),
                9:np.array([1,2,3,6,7,13,14,16,22]),
                10:np.array([1,2,3,6,7,12,14,18,21,23]),
                11:np.array([1,2,3,6,7,8,12,14,16,19,24]),
                12:np.array([1,2,3,6,7,8,12,14,16,19,22,24])}
}

# ~ c_q_max: dictionary, key: K, value: dictionary (key: q, value: maximum no. of distorted files after majority voting)
# ~ needed for "bulyan" and "multi-krum"
# Caution: this supports only one particular scheme, for each value of K (hard-coded)
c_q_max = {
            # K=15, l=5, r=3, MOLS/Rama Case 1, worst-case Byzantine sets for some values of q
            15: {0:0, 1:0, 2:1, 3:3, 4:5, 5:8, 6:12, 7:14, 8:17}, # ~ use q >= 8 just for debugging

            # K=35, l=7, r=5, MOLS/Rama Case 1, worst-case Byzantine sets for some values of q
            35: {0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:4, 7:5, 8:8, 9:10, 10:11, 11:14, 12:16, 13:20},

            # K=25, l=5, r=5, Rama Case 2, worst-case Byzantine sets for some values of q
            25: {0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:4, 7:5, 8:7, 9:9, 10:12, 11:14, 12:17}
}

def build_model(model_name, num_classes):
    # build network
    if model_name == "LeNet":
        return LeNet()
    elif model_name == "ResNet18":
        return ResNet18()
    elif model_name == "ResNet34":
        return ResNet34()
    elif model_name == "ResNet50":
        return ResNet50()
    elif model_name == "VGG11":
        return vgg11_bn(num_classes=num_classes)
    elif model_name == "VGG13":
        return vgg13_bn(num_classes=num_classes)
    elif model_name == "DenseNet":
        return DenseNet121()


# ~ returned value of torch data loader is an iterable over the batches
def load_data(dataset, seed, args, rank):
    if seed:
        # in normal method we do not implement random seed here
        # same group should share the same shuffling result
        torch.manual_seed(seed) # ~ sets the PyTorch seed for the caller and fixes it
        random.seed(seed)
    
    # ~ decide whether all MPI processes should download the dataset depending on whether it runs locally or remotely, works only for MNIST
    toShuffle = True
    # toDownload = False # ~ assumes that data sets are downloaded by prior data_prepare.sh
    if args.local_remote == "remote":
        toDownload = True
    else:
        # ~ only the PS will download the MNIST data set
        if rank == 0:
            toDownload = True
        else:
            toDownload = False

    if dataset == "MNIST":
    
        training_set = datasets.MNIST('./mnist_data', train=True, download=toDownload,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=toShuffle, drop_last=True)
        test_loader = None # ~ never used, only for cyclic_worker
    elif dataset == "Cifar10":
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        # data prep for training set
        # note that the key point to reach convergence performance reported in this paper (https://arxiv.org/abs/1512.03385)
        # is to implement data augmentation
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = datasets.CIFAR10(root='./cifar10_data', train=True,
                                                download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                                  shuffle=True, drop_last=True)
        testset = datasets.CIFAR10(root='./cifar10_data', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                 shuffle=False) # ~ never used, only for cyclic_worker
    elif args.dataset == 'SVHN':
        training_set = SVHN('./svhn_data', split='train', transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ]))
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                                  shuffle=True, drop_last=True)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        testset = SVHN(root='./svhn_data', split='test',
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                 shuffle=False) # ~ never used, only for cyclic_worker
    elif args.dataset == "Cifar100":
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = datasets.CIFAR100(root='./cifar100_data', train=True,
                                                download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                                  shuffle=True, drop_last=True)
        testset = datasets.CIFAR100(root='./cifar100_data', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                 shuffle=False) # ~ never used, only for cyclic_worker
    return train_loader, training_set, test_loader

# ~ Returns:
# ret_group_dict: see function...
# group_num: see function...
# group_seeds: see function...
def group_assign(world_size, group_size, rank): # ~ world_size == no. of workers here
    if world_size % group_size == 0: # ~ equal-sized groups
        ret_group_dict, group_list = _assign(world_size, group_size, rank)  
    else: # ~ extra worker goes to last group, dictionary does not reflect on this but it may have issues since it violates assumptions
        ret_group_dict, group_list = _assign(world_size-1, group_size, rank)
        group_list[-1].append(world_size)
    group_num, group_seeds = _group_identify(group_list, rank)
    return ret_group_dict, group_num, group_seeds
    
# ~ Returns:
# group_num: which group ID from {0,...,K/r-1} caller worker belongs to
# group_seeds: random seeds for all groups (list)
def _group_identify(group_list, rank):
    # test
    # print("group list I see",rank,group_list)
    group_seeds = [0]*len(group_list) # ~ == K/r, i.e., the number of groups
    if rank == 0: # ~ PS doesn't have a file & doesn't care about seeds
        return -1, group_seeds
    for i,group in enumerate(group_list): # ~ a random seed is set for each group
        group_seeds[i] = np.random.randint(0, 20000) # ~ the numpy random used here is set by a previously-called function to "SEED_" so this is persistent across workers
        if rank in group:
            # test
            # print('Hi, I am in group', group, rank)
            group_num = i # ~ do not return here, since master needs to know the seeds of all groups, rank does not play a role for him
    return group_num, group_seeds

# ~ assign workers to groups and return the groups in form of dictionary & list of lists
# Returns:
# ret_group_dict: dictionary, key: group ID from {0,...,K/r-1}, value: workers (ranks) in that group (list)
# group_list: list (one element for each group) of lists (workers (ranks) in that group)
def _assign(world_size, group_size, rank): # ~ world_size == no. of workers here as long as K%r == 0
    np.random.seed(SEED_) # ~ this will make the assignment consistent for all machines but may not need in this function
    ret_group_dict={}
    k = int(world_size/group_size) # ~ no. of groups
    group_list=[[j+i*group_size+1 for j in range(group_size)] for i in range(k)] # ~ ranks from {1,2,...} that are in each group
    for i, l in enumerate(group_list): # ~ key: group counter, value: list of workers in group
        ret_group_dict[i]=l
    return ret_group_dict, group_list # ~ both have the same information, the dictionary is indexed with group counter


def _generate_adversarial_nodes(args, world_size): # ~ world_size == no. of workers + 1 here
    # ~ generate indices of adversarial compute nodes randomly for all iterations at the beginning
    np.random.seed(SEED_) # ~ this will make the assignment consistent for all machines
    if args.byzantine_gen == "random":
        return [np.random.choice(np.arange(1, world_size), size=args.worker_fail, replace=False) for _ in range(args.max_steps+1)]
    elif args.byzantine_gen == "hard_coded":
        assert hard_byz_set[world_size-1][args.worker_fail].shape[0] == args.worker_fail, "Error: Hard-code Byzantine set size mismatch!"
        return [hard_byz_set[world_size-1][args.worker_fail] for _ in range(args.max_steps+1)]
    else: # ~ invalid
        assert 0 == 1, "Error: Unknown Byzantine generation mode!"


# ~ Attack 1 (worst attack)
# def _attack_detox(args, world_size): # world_size == no. of workers + 1 here
    # K = world_size-1
    # q = args.worker_fail
    # r = args.group_size
    # r_prime = (args.group_size + 1)//2
    # if args.worker_fail % r_prime == 0:
         # byzantines = [np.asarray(list(chain.from_iterable([list(range(i, i+r_prime)) for i in range(1, (q//r_prime)*r, r)])), dtype=np.int32) for _ in range(args.max_steps+1)]
    # else:
        # byzset = list(chain.from_iterable([list(range(i, i+r_prime)) for i in range(1, (q//r_prime)*r, r)]))
        # rest = q % r_prime
        # byzset.extend(list(range((q//r_prime)*r+1, (q//r_prime)*r+1 + rest)))
        # byzantines = [np.asarray(byzset, dtype=np.int32) for _ in range(args.max_steps+1)]
    # return byzantines


# ~ Attack 2 (benign attack)
# benign attack (picks 1 from each group until it runs out of Byzantines)
# def _attack_detox(args, world_size):
    # K = world_size-1
    # q = args.worker_fail
    # r = args.group_size
    # byzset = []
    # ctr = 0
    # for start in range(1, r+1):
        # for x in range(start,K+1,r):
            # byzset.append(x)
            # ctr += 1
            # if ctr == q: return [np.asarray(byzset, dtype=np.int32) for _ in range(args.max_steps+1)]


# ~ Attack 3 (whole-group attack)
# picks up all workers in 1st group, then all in 2nd group...
# def _attack_detox(args, world_size):
    # q = args.worker_fail
    # return [np.asarray(list(range(1,q+1)), dtype=np.int32) for _ in range(args.max_steps+1)]


# ~ This includes all attacks 1-3 above
# see descriptions above
def _attack_detox(args, world_size):
    K = world_size-1
    q = args.worker_fail
    r = args.group_size
    if args.detox_attack == "worst":
        r_prime = (args.group_size + 1)//2
        if args.worker_fail % r_prime == 0:
            byzantines = [np.asarray(list(chain.from_iterable([list(range(i, i+r_prime)) for i in range(1, (q//r_prime)*r, r)])), dtype=np.int32) for _ in range(args.max_steps+1)]
        else:
            byzset = list(chain.from_iterable([list(range(i, i+r_prime)) for i in range(1, (q//r_prime)*r, r)]))
            rest = q % r_prime
            byzset.extend(list(range((q//r_prime)*r+1, (q//r_prime)*r+1 + rest)))
            byzantines = [np.asarray(byzset, dtype=np.int32) for _ in range(args.max_steps+1)]
        return byzantines
    elif args.detox_attack == "benign":
        byzset = []
        ctr = 0
        for start in range(1, r+1):
            for x in range(start,K+1,r):
                byzset.append(x)
                ctr += 1
                if ctr == q: return [np.asarray(byzset, dtype=np.int32) for _ in range(args.max_steps+1)]
    elif args.detox_attack == "whole_group":
        return [np.asarray(list(range(1,q+1)), dtype=np.int32) for _ in range(args.max_steps+1)]
    else: # ~ invalid attack
        assert 0 == 1, "Error: Unknown DETOX attack!"


# ~
def mols(n):
    ''' Generate a set of mutually orthogonal latin squares 
        n must be prime
    ''' 
    r = range(n)
    # r = range(1,n+1)

    #Generate each Latin square
    allgrids = []
    for k in range(1, n):
        grid = []
        for i in r:
            row = []
            for j in r:
                a = (k*i + j) % n
                row.append(a)
            grid.append(row)
        allgrids.append(grid)

    return allgrids


# ~ Decides file assignment for MOLS scheme
# Arguments:
# args: arguments to pull r from
# K: no. of workers
# rank: rank of caller worker in {1,...,K}
# Returns:
# ret_group_dict: dictionary from file in {0,...,f-1} to list of workers (ranks) that have it
# seeds_dict[rank] (returned to worker) OR seeds_dict (returned to PS): 
#     seeds_dict[rank]: list of files for caller worker (rank)
#     seeds_dict: dictionary from worker (ranks) to list of files in {0,...,f-1} that it has
# ret_group_dict.keys(): list of distinct files 0...f-1
def mols_groups(args, K, rank): # K == no. of workers
    l, r = K//args.group_size, args.group_size # MOLS parameters
    allMOLS = mols(l)
    ls = r
    ret_group_dict = {x:[] for x in range(0,l**2)} # list of workers (ranks) for each file
    seeds_dict = {x:[] for x in range(1,K+1)} # list of files for each worker (rank)
    for lsInd in range(ls):
        for symbol in range(l):
            curWorker = lsInd*l+symbol # this starts from zero i.e. it is rank-1
            for i in range(l):
                for j in range(l):
                    if allMOLS[lsInd][i][j] == symbol:
                        seeds_dict[curWorker+1].append(i*l+j)
                        ret_group_dict[i*l+j].append(curWorker+1)
                        
    if rank == 0: # PS needs to know both file -> worker and worker -> file assignment to collect and aggregate the gradients
        return ret_group_dict, seeds_dict, [0]*len(ret_group_dict)
    else: # worker
        return ret_group_dict, seeds_dict[rank], list(ret_group_dict.keys())


# ~ Checks if positive integer n is prime or not
# https://stackoverflow.com/a/17377939/1467434
def is_prime(n):
    if n == 2:
        return True
    if n % 2 == 0 or n <= 1:
        return False

    sqr = int(n**0.5) + 1

    for divisor in range(3, sqr, 2):
        if n % divisor == 0:
            return False
    return True


# ~ Decides file assignment for Ramanujan schemes based on paper "Deterministic Completion of Rectangular Matrices Using Ramanujan Bigraphs -- II"
# Arguments
# K: no. of workers
# case: 1 or 2, Ramanujan Case
# rank: MPI rank of caller worker
# Returns:
# ret_group_dict: dictionary from file to list of workers (ranks) that have it
# seeds_dict[rank]: list of files for caller worker (rank)
#    OR seeds_dict (returned to PS): dictionary from worker (ranks) to list of files that it has
# ret_group_dict.keys(): list of distinct files
def rama_groups(args, K, case, rank):
    # group_size = 5 # not for AWS code, group_size == args.group_size to be used in AWS code
    
    # m: parameter m of Ramanujan (l in original paper)
    # s: parameter s of Ramanujan (q in original paper), prime
    # f: no. of files
    
    if case == 1: # Ramanujan Case 1, f == s**2
        # m, s = group_size, K//group_size # not for AWS code
        m, s = args.group_size, K//args.group_size
        assert m < s # Case 1 requirement
    elif case == 2: # Ramanujan Case 2, f == m*s
        # rama_m = 10 # not for AWS code
        # m, s = rama_m, group_size # not for AWS code
        m, s = args.rama_m, args.group_size
        assert m >= s and m%s == 0 # Case 2 requirement
        
    assert is_prime(s) # Ramanujan requirement
        
    # cyclic shift permutation matrix
    P = np.zeros((s,s))
    for i in range(1,s+1):
        for j in range(1,s+1):
            P[i-1][j-1] = 1 if j%s == (i-1)%s else 0 

    # biadjacency matrix of Ramanujan biregular bipartite graph
    B = np.tile(np.identity(s), m) # first block row of B
    step = 1
    for r in range(s-1):
        curRow = np.identity(s)
        expo = r+1
        for i in range(m-1):
            curRow = np.concatenate((curRow, np.linalg.matrix_power(P, expo)), axis=1)
            expo += step
            
        B = np.concatenate((B, curRow), axis=0)
        step += 1
        
    # also compute the left and right degrees of the bipartite graph, i.e.,
    # the number of files per worker (computation load) and the number of workers per file
    if m < s: # we work with B^T, Ramanujan Case 1
        adj = np.transpose(B)
    else: # we work with B, Ramanujan Case 2
        adj = B
        
    # no. of files
    f = np.ma.size(adj, axis=1)

    # populate workers
    ret_group_dict = {x:[] for x in range(0,f)} # list of workers (ranks) for each file
    seeds_dict = {x:[] for x in range(1,K+1)} # list of files for each worker (rank)
    for i in range(K): # for each worker
        for j in range(f): # for each file
            if adj[i][j] == 1:
                seeds_dict[i+1].append(j)
                ret_group_dict[j].append(i+1)

    if rank == 0: # PS needs to know both file -> worker and worker -> file assignment to collect and aggregate the gradients
        return ret_group_dict, seeds_dict, [0]*len(ret_group_dict)
    else: # worker
        return ret_group_dict, seeds_dict[rank], list(ret_group_dict.keys())
        

# ~ Receives the number of epochs and returns a numpy vector of random seeds to be used by all workers to retrieve batches
def epoch_seeds(num_epochs):
    np.random.seed(SEED_)
    return np.random.randint(0, 20000, size = num_epochs)


def prepare(args, rank, world_size):
    device = torch.device("cpu")
    if args.approach == "baseline":
        # randomly select adversarial nodes
        adversaries = _generate_adversarial_nodes(args, world_size)
        train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=None, args=args, rank=rank)
        kwargs_master = {
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps, 
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'worker_fail':args.worker_fail,
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir, 
                    'update_mode':args.mode, 
                    'compress_grad':args.compress_grad, 
                    'checkpoint_step':args.checkpoint_step,
                    'lis_simulation':args.lis_simulation,
                    'device':device,
                    'adversaries':adversaries,
                    'gamma':args.gamma,
                    'lr_step':args.lr_step,
                    'err_mode':args.err_mode
                    }
        kwargs_worker = {
                    'update_mode':args.mode,  # for implementing signSGD
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps,
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'adversery':args.adversarial, 
                    'worker_fail':args.worker_fail,
                    'err_mode':args.err_mode, 
                    'compress_grad':args.compress_grad, 
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir, 
                    'checkpoint_step':args.checkpoint_step,
                    'lis_simulation':args.lis_simulation,
                    'adversaries':adversaries,
                    'device':device
                    }
    # majority vote
    elif args.approach == "maj_vote":
        adversaries = _generate_adversarial_nodes(args, world_size)
        group_list, group_num, group_seeds=group_assign(world_size-1, args.group_size, rank) # ~ exclude the PS from grouping assignments
        train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=group_seeds[group_num], args=args, rank=rank)
        kwargs_master = {
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps, 
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir, 
                    'group_list':group_list, 
                    'update_mode':args.mode, 
                    'compress_grad':args.compress_grad, 
                    'checkpoint_step':args.checkpoint_step,
                    'device':device
                    } # ~ group_list is stored in PS dict
        kwargs_worker = {
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps,
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'adversery':args.adversarial, 
                    'worker_fail':args.worker_fail,
                    'err_mode':args.err_mode, 
                    'group_list':group_list, 
                    'group_seeds':group_seeds, 
                    'group_num':group_num,
                    'compress_grad':args.compress_grad, 
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir,
                    'adversaries':adversaries,
                    'device':device
                    } # ~ group_list, group_seeds, group_num & adversaries are stored in worker's dict
    # ~ draco lite & all other Kostas cases
    elif args.approach == "draco_lite" or args.approach == "draco_lite_attack" or args.approach == "mols" or args.approach == "rama_one" or args.approach == "rama_two":
        seeds = epoch_seeds(args.epochs) # ~ won't be used by DETOX, only by ByzShield
        if args.approach == "draco_lite":
            adversaries = _generate_adversarial_nodes(args, world_size) # ~ randomly chooses the adversaries at the beginning of training
            group_list, group_num, group_seeds=group_assign(world_size-1, args.group_size, rank)
            train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=group_seeds[group_num], args=args, rank=rank) # loader seeds for torch are set here but are they persistent ??? they are set later too
        elif args.approach == "draco_lite_attack": # ~ draco lite Kostas attack
            adversaries = _attack_detox(args, world_size) # ~ chooses the adversaries such that the majority of each group is distorted (Kostas attack)
            group_list, group_num, group_seeds=group_assign(world_size-1, args.group_size, rank)
            train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=group_seeds[group_num], args=args, rank=rank)
        elif args.approach == "mols": # ~ MOLS
            adversaries = _generate_adversarial_nodes(args, world_size)
            group_list, group_num, group_seeds=mols_groups(args, world_size-1, rank)
            
            # ~ arbitrary 1st seed (file) to torch (won't be used)
            train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=42, args=args, rank=rank)
            
        elif args.approach == "rama_one": # ~ Ramanujan Case 1
            adversaries = _generate_adversarial_nodes(args, world_size)
            group_list, group_num, group_seeds=rama_groups(args, world_size-1, 1, rank)
            
            # ~ arbitrary 1st seed (file) to torch (won't be used)
            train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=42, args=args, rank=rank)
            
        elif args.approach == "rama_two": # ~ Ramanujan Case 2
            adversaries = _generate_adversarial_nodes(args, world_size)
            assert isinstance(args.rama_m, int) and args.rama_m > 0
            group_list, group_num, group_seeds=rama_groups(args, world_size-1, 2, rank)
            
            # ~ arbitrary 1st seed (file) to torch (won't be used)
            train_loader, training_set, test_loader = load_data(dataset=args.dataset, seed=42, args=args, rank=rank)
        
        # ~ this is the same for all of the above cases
        kwargs_master = {
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps, 
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir, 
                    'group_list':group_list, 
                    'group_num':group_num,
                    'update_mode':args.mode,
                    'bucket_size':args.bucket_size,
                    'compress_grad':args.compress_grad, 
                    'checkpoint_step':args.checkpoint_step,
                    'lis_simulation':args.lis_simulation,
                    'worker_fail':args.worker_fail,
                    'device':device,
                    'adversaries':adversaries,
                    'c_q_max':c_q_max[world_size-1][args.worker_fail], # ~ won't be used by DETOX, only by ByzShield
                    'gamma':args.gamma,
                    'lr_step':args.lr_step,
                    'err_mode':args.err_mode
                    }
        kwargs_worker = {
                    'update_mode':args.mode,  # for implementing signSGD
                    'batch_size':args.batch_size, 
                    'learning_rate':args.lr, 
                    'max_epochs':args.epochs, 
                    'max_steps':args.max_steps,
                    'momentum':args.momentum, 
                    'network':args.network,
                    'comm_method':args.comm_type, 
                    'adversery':args.adversarial, 
                    'worker_fail':args.worker_fail,
                    'err_mode':args.err_mode, 
                    'group_list':group_list, 
                    'group_seeds':group_seeds, 
                    'group_num':group_num,
                    'compress_grad':args.compress_grad, 
                    'eval_freq':args.eval_freq, 
                    'train_dir':args.train_dir,
                    'checkpoint_step':args.checkpoint_step,
                    'adversaries':adversaries,
                    'lis_simulation':args.lis_simulation,
                    'device':device,
                    'seeds':seeds # ~ won't be used by DETOX, only by ByzShield
                    }
    datum = (train_loader, training_set, test_loader)
    return datum, kwargs_master, kwargs_worker