from .utils import * # ~ will import gradient compression compress(), err_simulation(), accuracy(), models from "model_ops", ...
from .baseline_worker import DistributedWorker

import logging # ~ for comments on logging see "distributed_nn.py"

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class DracoLiteWorker(DistributedWorker):
    def __init__(self, comm, **kwargs):
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.rank = comm.Get_rank() # rank of this Worker
        #self.status = MPI.Status()
        self.cur_step = 0 # ~ current step index (across all epochs), updated globally at the master
        self.next_step = 0 # we will fetch this one from parameter server

        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.momentum = kwargs['momentum']
        self.lr = kwargs['learning_rate']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']
        self._adversery = kwargs['adversery'] # ~ never used
        self._err_mode = kwargs['err_mode']
        self._group_list = kwargs['group_list'] # ~ DETOX: dictionary, key: group ID from {0,...,K/r-1}, value: workers (ranks) in that group (list)
        self._train_dir = kwargs['train_dir']
        self._checkpoint_step = kwargs['checkpoint_step']
        self._eval_freq = kwargs['eval_freq']
        self._max_steps = kwargs['max_steps']
        self._lis_simulation = kwargs['lis_simulation']

        self._fail_workers = kwargs['adversaries']

        self._group_seeds = kwargs['group_seeds'] 
        self._group_num = kwargs['group_num'] # ~ which group this worker belongs to
        self._group_size = len(self._group_list[0]) # ~ == r
        self._compress_grad = kwargs['compress_grad']
        self._device = kwargs['device']
        # this one is going to be used to avoid fetch the weights for multiple times
        self._layer_cur_step = []
        
        self._update_mode = kwargs['update_mode']
        
        # ~ test
        # logger.info("DEBUG_W_DETOX {}: adversaries: {}, group_list: {}, group_num: {}, group_seeds: {}".format(self.rank, self._fail_workers, self._group_list, self._group_num, self._group_seeds))

    def build_model(self):
        # build network
        if self.network_config == "LeNet":
            self.network=LeNet()
        elif self.network_config == "ResNet18":
            self.network=ResNet18()
        elif self.network_config == "ResNet34":
            self.network=ResNet34()
        elif self.network_config == "ResNet50":
            self.network=ResNet50()
        elif self.network_config == "FC":
            self.network=FC_NN()
        elif self.network_config == "DenseNet":
            self.network=DenseNet121()
        elif self.network_config == "VGG11":
            self.network=vgg11_bn(num_classes=100)
        elif self.network_config == "VGG13":
            self.network=vgg13_bn(num_classes=100)
        elif self.network_config == "VGG19":
            self.network=vgg19_bn(num_classes=100)

        # set up optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        self.criterion = nn.CrossEntropyLoss()
        # assign a buffer for receiving models from parameter server
        self.init_recv_buf() # ~ creates a buffer equal to the model size & shape

        self.network.to(self._device)

    def train(self, train_loader, test_loader):
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure the value we fetched from parameter server is 1

        self.sync_fetch_step() # ~ updates self.next_step to 1
        # do some sync check here
        assert(self.update_step()) # ~ verifies that self.cur_step (== 0) is not equal to self.next_step (== 1) and sets self.cur_step = 1
        # assert(self.cur_step == STEP_START_) # ~ verifies above change
        if self._checkpoint_step == 0:
            assert(self.cur_step == STEP_START_)
        else:
            assert(self.cur_step == int(self._checkpoint_step)+1)

        # number of batches in one epoch
        num_batch_per_epoch = len(train_loader.dataset) / self.batch_size # ~ never used
        batch_idx = -1
        epoch_idx = 0
        epoch_avg_loss = 0
        iteration_last_step = 0
        iter_start_time = 0
        first = True
        iter_avg_prec1 = 0 # ~ never used
        iter_avg_prec5 = 0 # ~ never used
        # use following flags to achieve letting each worker compute more batches
        should_enter_next = False # ~ never used

        logger.info("W_DETOX: Worker {}: starting training".format(self.rank))
        # start the training process
        for num_epoch in range(self.max_epochs):
            # after each epoch we need to make sure workers in the same group re-shuffling using the same seed
            
            # ~ test
            # torch.manual_seed(428)
            # torch.manual_seed(self._group_num) # ~ ONLY TEMPORARY, USE THE ONE BELOW
            
            torch.manual_seed(self._group_seeds[self._group_num]+num_epoch) # ~ the +num_epoch is not necessary it's just so that they don't get the exact same data at each epoch
            
            # ~ test
            # logger.info("DEBUG_W_DETOX: torch.manual_seed: {}".format(self._group_seeds[self._group_num]+num_epoch))
            
            for batch_idx, (train_image_batch, train_label_batch) in enumerate(train_loader):
                # worker exit task
                if self.cur_step == self._max_steps:
                    break
                X_batch, y_batch = train_image_batch.to(self._device), train_label_batch.to(self._device) # ~ tensors
                
                # ~ test
                # if self.rank == 1:
                    # logger.info("DEBUG_W_DETOX: X_batch: {}, {}".format(type(X_batch), X_batch.shape))
                    # logger.info("DEBUG_W_DETOX: y_batch: {}, {}".format(type(y_batch), y_batch.shape))
                    # logger.info("DEBUG_W_DETOX: train_image_batch: {}, {}".format(type(train_image_batch), train_image_batch.shape))
                    # logger.info("DEBUG_W_DETOX: train_label_batch: {}, {}".format(type(train_label_batch), train_label_batch.shape))
                    # logger.info("DEBUG_W_DETOX: y_batch values: {}".format(y_batch))
                    
                while True:
                    # the worker shouldn't know the current global step except received the message from parameter server
                    self.async_fetch_step()
                    # the only way every worker know which step they're currently on is to check the cur step variable
                    updated = self.update_step() # ~ updates ctr of step +1 or something from the PS
                    if (not updated) and (not first):
                        # wait here unitl enter next step
                        continue
                    # the real start point of this iteration
                    iter_start_time = time.time()
                    first = False
                    should_enter_next = False
                    logger.info("W_DETOX: Rank of this node: {}, Current step: {}".format(self.rank, self.cur_step))

                    fetch_weight_start_time = time.time()
                    self.async_fetch_weights_bcast()
                    fetch_weight_duration = time.time() - fetch_weight_start_time

                    # ~ test
                    # if self._group_num == 0: # ~ for one group
                    # np.save('DETOX_worker_'+str(self.rank)+'_group_'+str(self._group_num)+'_X_batch', 
                        # X_batch[(self.batch_size//len(self._group_list))*self._group_num:(self.batch_size//len(self._group_list))*(self._group_num+1),...].detach().numpy())
                    # np.save('DETOX_worker_'+str(self.rank)+'_group_'+str(self._group_num)+'_y_batch', 
                        # y_batch[(self.batch_size//len(self._group_list))*self._group_num:(self.batch_size//len(self._group_list))*(self._group_num+1)].detach().numpy())

                    self.network.train()
                    self.optimizer.zero_grad()
                    # forward step
                    forward_start_time = time.time()
                    logits = self.network(X_batch)

                    loss = self.criterion(logits, y_batch)
                    
                    # ~ test
                    # logits = self.network(X_batch[(self.batch_size//len(self._group_list))*self._group_num:(self.batch_size//len(self._group_list))*(self._group_num+1),...])
                    # loss = self.criterion(logits, y_batch[(self.batch_size//len(self._group_list))*self._group_num:(self.batch_size//len(self._group_list))*(self._group_num+1)])
                        
                    forward_duration = time.time()-forward_start_time
                    
                    # ~ test
                    # if self._group_num == 0: # ~ for one group
                    # np.save('DETOX_worker_'+str(self.rank)+'_group_'+str(self._group_num)+'_logits_step'+str(self.cur_step), logits.detach().numpy())

                    # backward step
                    backward_start_time = time.time()
                    loss.backward()
                    backward_duration = time.time() - backward_start_time
                    computation_time = forward_duration + backward_duration

                    grads = [p.grad.detach().numpy().astype(float_type) for p in self.network.parameters()]
                    
                    # ~ test
                    # save gradient of a worker for layer lay
                    # lay = 0
                    # if self.rank == 1:
                        # np.save('DETOX_worker'+str(self.rank)+'_grads'+str(lay), grads[lay])
                        
                    # ~ test
                    # if self.rank == 1:                            
                        # for i in range(len(grads)):
                            # logger.info("DEBUG_W_DETOX: grads[i]: {}, {}".format(type(grads[i]), grads[i].shape))

                    prec1, prec5 = accuracy(logits.detach(), train_label_batch.long(), topk=(1, 5))
                    
                    # ~ test
                    # prec1, prec5 = accuracy(logits.detach(), train_label_batch[(self.batch_size//len(self._group_list))*self._group_num:(self.batch_size//len(self._group_list))*(self._group_num+1)].long(), topk=(1, 5))
                    
                    # in current setting each group cotains k workers, we let each worker calculate k same batches
                    c_start = time.time()
                    self._send_grads(grads)
                    c_duration = time.time() - c_start

                    logger.info('W_DETOX: draco_lite_worker: {}, Step: {}, Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.4f}, Time Cost: {:.4f}, Comp: {:.4f}, Comm: {:.4f}, Prec@1: {}, Prec@5: {}'.format(self.rank,
                         self.cur_step, num_epoch, batch_idx * self.batch_size, len(train_loader.dataset), 
                            (100. * (batch_idx * self.batch_size) / len(train_loader.dataset)), loss.item(), time.time()-iter_start_time, computation_time, c_duration+fetch_weight_duration, prec1.numpy()[0], prec5.numpy()[0]))
                    if self.cur_step%self._eval_freq == 0 and self.rank==1:
                        #self._save_model(file_path=self._generate_model_path())
                        self._save_model(file_path=self._generate_model_path())
                        
                    break # ~ breaks here for everyone after one step of training (current batch)

    def _send_grads(self, grads):
        # ~ MPI tag for these transmissions is 88+(layer index), e.g., for Lenet 88,89,...,95
        # test
        # if self.rank == 1:
            # logger.info("DEBUG_W_DETOX: length of grads: {}".format(len(grads)))
            # for i, grad in enumerate(grads):
                # _compressed_grad = compress(grad)  
                # logger.info("DEBUG_W_DETOX: worker: {}, _compressed_grad: {} {} {}".format(self.rank, type(_compressed_grad), len(_compressed_grad), getsizeof(_compressed_grad)))
        
        req_send_check = []
        #for i, grad in enumerate(reversed(grads)):
        for i, grad in enumerate(grads): # ~ for each layer
            # ~ test
            # if self.rank == 1:
                # if i == 0: # for first layer
                    # logger.info("DEBUG_W_DETOX: Worker {} sending gradients for layer {}: {}, {}, {}".format(self.rank, i, type(grad), grad.shape, grad))
                    
            if len(req_send_check) != 0: # ~ wait on previous request before sending the current one
                req_send_check[-1].wait()
            if self._lis_simulation == "simulate": # ~ ALIE attack will be "simulated" at the PS, all workers send honest gradients
                if self._compress_grad=='compress':
                    if self._update_mode=="sign-sgd":
                        # signSGD worker side
                        grad = np.sign(grad).astype(np.int8)
                    _compressed_grad = compress(grad)
                    req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                else:
                    req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+i)
                req_send_check.append(req_isend)
            else:
                if self.rank in self._fail_workers[self.cur_step]:
                    simulation_grad = err_simulation(grad, self._err_mode)
                    if self._update_mode=="sign-sgd":
                        # signSGD worker side
                        simulation_grad = np.sign(simulation_grad).astype(np.int8)
                    if self._compress_grad=='compress':
                        _compressed_grad = compress(simulation_grad)
                        req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                    else:
                        req_isend = self.comm.Isend([simulation_grad, MPI.DOUBLE], dest=0, tag=88+i)
                    req_send_check.append(req_isend)
                else:
                    if self._compress_grad=='compress':
                        if self._update_mode=="sign-sgd":
                            # signSGD worker side
                            grad = np.sign(grad).astype(np.int8)
                        _compressed_grad = compress(grad)
                        req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                    else:
                        req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+i)
                    req_send_check.append(req_isend)
        req_send_check[-1].wait()


    '''
    def _send_grads(self, grads):
        req_send_check = []
        #for i, grad in enumerate(reversed(grads)):
        for i, grad in enumerate(grads):
            if len(req_send_check) != 0:
                req_send_check[-1].wait()
            if self.rank in self._fail_workers[self.cur_step]:
                simulation_grad = err_simulation(grad, self._err_mode)
                if self._compress_grad=='compress':
                    _compressed_grad = compress(simulation_grad)
                    req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                else:
                    req_isend = self.comm.Isend([simulation_grad, MPI.DOUBLE], dest=0, tag=88+i)
                req_send_check.append(req_isend)
            else:
                if self._compress_grad=='compress':
                    _compressed_grad = compress(grad)
                    req_isend = self.comm.isend(_compressed_grad, dest=0, tag=88+i)
                else:
                    req_isend = self.comm.Isend([grad, MPI.DOUBLE], dest=0, tag=88+i)
                req_send_check.append(req_isend)
        req_send_check[-1].wait()
    '''