from .utils import *  # ~ will import models from "model_ops", ...
from .baseline_master import SyncReplicasMaster_NN

import logging # ~ for comments on logging see "distributed_nn.py"
import torch.optim as optim

from joblib import Parallel, delayed
from functools import reduce

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ~ DEBUG: this will traceback numpy warnings and crash instead of just warning
warnings.simplefilter('error')

class DracoLiteMaster(SyncReplicasMaster_NN):
    def __init__(self, comm, **kwargs):
        '''master node here, no rank needed since the rank will always be 0 for master node'''
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.cur_step = STEP_START_
        self.lr = kwargs['learning_rate']
        self.momentum = kwargs['momentum']
        self.network_config = kwargs['network']
        self.comm_type = kwargs['comm_method']

        self._num_grad_to_collect = self.world_size - 1
        self.num_workers = self.world_size-1
        # used to aggregate tmp gradients, the length is the same as # of fc layer 
        self._grad_aggregate_buffer = []
        self._coded_grads_buffer = {}
        self._model_shapes = []
        self._first_grad_received = False # ~ never used
        self._eval_freq = kwargs['eval_freq']
        self._train_dir = kwargs['train_dir']
        self._update_mode = kwargs['update_mode']
        self._max_steps = kwargs['max_steps'] # ~ max-steps argument of run_pytorch
        self._group_list = kwargs['group_list'] # ~ DETOX: dictionary, key: group ID from {0,...,K/r-1}, value: workers (ranks) in that group (list)
        self._compress_grad = kwargs['compress_grad']
        self._checkpoint_step = kwargs['checkpoint_step']
        self._group_size = len(self._group_list[0]) # ~ == r
        self._bucket_size = kwargs['bucket_size']
        self._lis_simulation = kwargs['lis_simulation']
        self._s = kwargs['worker_fail'] # ~ == q
        self._device = kwargs['device']

        ######## LR scheduling related ############
        self.gamma = kwargs['gamma']
        self.lr_step = kwargs['lr_step']
        # self.gamma = 0.99
        # self.gamma = 0.98
        # self.lr_step = 100000000000000000
        # self.lr_step = 2*self._eval_freq
        # self.lr_step = 10
        ###########################################
        
        self._fail_workers = kwargs['adversaries'] # ~ the same set of Byzantines will be used in ALIE (not a new random one)

    def build_model(self):
        # ~ test
        # torch.manual_seed(428)
        
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
            
        if self._checkpoint_step != 0:
            file_path = self._train_dir + "model_step_" + str(self._checkpoint_step)
            self._load_model(file_path)
            self.cur_step = int(self._checkpoint_step)+1

        # assign a gradient accumulator to collect gradients from workers
        self.grad_accumulator = GradientAccumulator(self.network, self.world_size-1, mode=self._compress_grad) # ~ passes the number of workers as the number of gradients to accumulate and the model so that it knows the dimensions
        
        # ~ test
        # for i in range(len(self.grad_accumulator.gradient_aggregator)): # ~ for each  layer
            # for arr in self.grad_accumulator.gradient_aggregator[i]: # ~ for each worker's np.ndarray or Python bytearray() after decompression
                # logger.info("DEBUG_PS_DETOX: self.grad_accumulator.gradient_aggregator[i][j] of master: {}, {}".format(type(arr), len(arr)))
            
        self.init_model_shapes()
        self.optimizer = SGDModified(self.network.parameters(), lr=self.lr, momentum=self.momentum)
        #self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)

        self.network.to(self._device)

    def init_model_shapes(self):
        tmp_aggregate_buffer = []
        self._model_param_counter = 0
        for param_idx, param in enumerate(self.network.parameters()):
            shape = param.size()
            num_params = reduce((lambda x, y: x * y), shape)
            self._model_param_counter += num_params

            self._model_shapes.append(shape)
            self._grad_aggregate_buffer.append(np.zeros(shape, dtype=float_type))
            tmp_aggregate_buffer.append(np.zeros(shape, dtype=float_type))

        #if self._update_mode == "maj_vote" or self._update_mode == "draco_lite":
        for k, v in self._group_list.items():
            for i, l in enumerate(v):
                if k not in self._coded_grads_buffer.keys():
                    self._coded_grads_buffer[k] = [copy.deepcopy(tmp_aggregate_buffer)]
                else:
                    self._coded_grads_buffer[k].append(copy.deepcopy(tmp_aggregate_buffer))

        # buffer setted up for draco-lite aggregations
        self._sub_grad_size = int(self.num_workers/self._group_size)
        self._draco_lite_aggregation_buffer = np.zeros((self._sub_grad_size, self._model_param_counter), dtype=float_type)

    def start(self):
        # ~ test
        # for x in self.network.parameters():
            # logger.info("DEBUG_PS_DETOX: network.parameters()[i] of master: {}, {}".format(type(x), x.shape))
            
        # the first step we need to do here is to sync fetch the inital worl_step from the parameter server
        # we still need to make sure value fetched from ps is 1
        self.async_bcast_step() # ~ (baseline) updates step and sends it to workers, matches worker's sync_fetch_step()

        # fake test here:
        for i in range(self._checkpoint_step+1, self._max_steps+1): # ~ global steps loop across all epochs
            # switch back to training mode
            self.network.train() # ~ this only sets the model to training mode, it does not actually train anything
            self._first_grad_received = False # ~ never used
            enough_gradients_received = False

            logger.info("PS_DETOX: Master node is entering step: {}".format(i))
            self.async_bcast_step() # ~ similar to above async_bcast_step() ...
            
            # ~ test
            # logger.info("DEBUG_PS_DETOX: PS finished async_bcast_step() for step {}".format(i))

            self.async_bcast_layer_weights_bcast()
            
            # set the gradient fetch step and gather the request
            gradient_fetch_requests=self.async_fetch_gradient_start()
            # wait for enough gradients to be aggregated:
            while not enough_gradients_received: # ~ wait for any one worker to send a gradient
                status = MPI.Status()
                if self._compress_grad == "None": # ~ received_msg: np.ndarray
                    MPI.Request.Waitany(requests=gradient_fetch_requests, status=status)
                elif self._compress_grad == "compress": # ~ received_msg: received Python object, first (unassigned) value "_" is the MPI index of handle that completed (integer), here in [0,1,...,K*(no. of layers)-1] but no need to use it since the requests are associated with worker ranks/layer indices
                    _, received_msg=MPI.Request.waitany(requests=gradient_fetch_requests, status=status)
                    
                    # ~ test
                    # if status.source == 1: ~ worker with rank 1
                        # x, received_msg=MPI.Request.waitany(requests=gradient_fetch_requests, status=status)
                        # logger.info("DEBUG_PS_DETOX: MPI.Request.waitany index handle: {}".format(x))
                        # logger.info("DEBUG_PS_DETOX: received_msg: {} {}".format(type(received_msg), len(received_msg)))
                    
                    received_grad=decompress(received_msg)

                if status.tag-88 in self.grad_accumulator.model_index_range: # ~ checks if received layer index is within bounds of the receive buffer (list)
                    if not self._first_grad_received: # ~ never used
                        self._first_grad_received=True
                        grad_gather_start_time = time.time()

                    layer_index = status.tag-88

                    if self._compress_grad == "None":
                        received_grad=self.grad_accumulator.gradient_aggregator[layer_index][status.source-1]
                        
                    # ~ test
                    # this debug checks whether for MPI.Request.waitany() the MPI received buffer and the returned value are the same data in the case of "compress" mode
                    # else:
                        # if status.source == 1: # ~ worker with rank 1
                            # logger.info("DEBUG_PS_DETOX: received_msg: {} {}".format(type(received_msg), len(received_msg)))
                            # received_msg_other = self.grad_accumulator.gradient_aggregator[layer_index][status.source-1]
                            # logger.info("DEBUG_PS_DETOX: received_msg_other: {} {}".format(type(received_msg_other), len(received_msg_other)))
                            # if layer_index == 0: # ~ the received_msg is contained in received_msg_other
                                # logger.info("DEBUG_PS_DETOX: received_msg: {}".format(received_msg))
                                # logger.info("DEBUG_PS_DETOX: received_msg_other: {}".format( received_msg_other))
                    
                    # do gradient shape check here
                    assert (received_grad.shape == self._model_shapes[layer_index])
                    
                    # ~ test
                    # if status.source == 1: # ~ worker with rank 1
                        # if layer_index == 0:
                            # logger.info("DEBUG_PS_DETOX: received_grad from {} for layer {}: {} {} {}".format(status.source, layer_index, type(received_grad), received_grad.shape, received_grad))

                    # aggregate the gradient
                    if self.grad_accumulator.gradient_aggregate_counter[layer_index] <= self._num_grad_to_collect:
                        self.aggregate_gradient(received_grad, layer_index, status.source)

                    self.grad_accumulator.gradient_aggregate_counter[layer_index] += 1
                
                enough_gradients_received = True
                for j in self.grad_accumulator.gradient_aggregate_counter:
                    enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)


            ################### "A Little is enough" attack simulation on the PS"#########################
            if self._lis_simulation == "simulate":
                self._LIE_attack_simulation()
            else:
                pass            

            method_start = time.time()
            #self._grad_aggregate_buffer = draco_lite_aggregation(self._coded_grads_buffer, self._bucket_size, self.network, self._grad_aggregate_buffer)
            self._draco_lite_aggregation()
            method_duration = time.time() - method_start

            update_start = time.time()
            # update using SGD method
            self._model_update()
            
            # ~ test
            # for layer_idx, layer in enumerate(self.network.parameters()):
                # lay = layer.detach().numpy().astype(float_type)
                # np.save('DETOX_master_model_after_step_lay'+str(layer_idx), lay)

            update_duration = time.time() - update_start
            # reset essential elements
            self.meset_grad_buffer()
            self.grad_accumulator.meset_everything()

            logger.info("PS_DETOX: Master Step: {}, Method Time Cost: {}, Update Time Cost: {}".format(self.cur_step, method_duration, update_duration))
            
            #for param_group in self.optimizer.param_groups:
            #    break
            logger.info("PS_DETOX: Real time Lr: {}".format([param_group['lr'] for param_group in self.optimizer.param_groups]))   

            self.cur_step += 1
            
            # ~ test
            # break

    # ~ just stores each received gradient in self._coded_grads_buffer
    def aggregate_gradient(self, gradient, layer_idx, source):
        '''
        keep in mind the gradient here is wrapped gradient, which means it contains `W` and `b`
        '''
        #if self._update_mode == "normal":
        #    self._grad_aggregate_buffer[layer_idx] += gradient
        #elif self._update_mode == "maj_vote" or self._update_mode == "draco_lite":
        # under development, stay tunned
        for k, v in self._group_list.items():
            if source in v:
                assert self._coded_grads_buffer[k][v.index(source)][layer_idx].shape == gradient.shape
                self._coded_grads_buffer[k][v.index(source)][layer_idx] = gradient
                
                # ~ test
                # lay = 0 # layer index
                # if source == 1: # worker with rank 1
                    # if layer_idx == lay:
                        # np.save('DETOX_PS_worker'+str(source)+'_grads'+str(lay)+'_group'+str(k), self._coded_grads_buffer[k][v.index(source)][layer_idx])
                
                # ~ test
                # np.save('DETOX_PS_worker'+str(source)+'_grads'+str(layer_idx)+'_group'+str(k), self._coded_grads_buffer[k][v.index(source)][layer_idx])

    def _model_update(self):
        if self.cur_step % self.lr_step == 0:
            self.optimizer.lr_update(updated_lr=(self.lr * self.gamma ** (self.cur_step // self.lr_step)))
        self.optimizer.step(grads=self._grad_aggregate_buffer, mode="draco_lite")

    # ~ never used
    def _get_geo_median(self, bucket_grads):
        geo_median = np.array(hd.geomedian(np.array(bucket_grads), axis=0))
        return geo_median

    # ~ never used
    # elementwise maj vote among gradients gathered by PS
    def _elemwise_median(self, bucket_grads):
        elem_median = np.median(np.array(bucket_grads), axis=0)
        return elem_median

    # ~ never used
    # single-core version
    def _draco_lite_aggregation_single_thread(self):
        majority_grads = self._grad_majority_vote()
        # n-layers, r-groups: then majority should be in r * n:
        for j, p in enumerate(self.network.parameters()):
            layer_majority_grads = np.array([mg[j] for mg in majority_grads])
            indices = np.arange(len(layer_majority_grads))
            np.random.shuffle(indices)
            random_indicies = np.split(indices, self._bucket_size)
            for buckets in random_indicies:
                bucket_grads = np.take(layer_majority_grads, buckets, axis=0)
                #geo_median = self._get_geo_median(bucket_grads)
                elem_median = self._elemwise_median(bucket_grads)
                self._grad_aggregate_buffer[j] += elem_median.reshape(p.size())
        self._grad_aggregate_buffer = [x/float(len(self._group_list)) for x in self._grad_aggregate_buffer]

    # multi-core optimized verion
    def _draco_lite_aggregation(self):
        self._grad_majority_vote()

        # optimization objectives:
        # i) get rid of concated_grads and making it predefined
        # ii) use fancy slicing instead of the second for loop to get rid of grad_transformer (done, need to be further tested)
        # need to double check
        if self._update_mode == "coord-median":
            # Below, you need to one method, say "method x" and uncomment the corresponding lines labeled "method x"
            
            # bucketing stage
            
            indices = np.arange(self._sub_grad_size) # ~ 1D np.ndarray [0,...,K/r-1]
            
            # ~ in-place random shuffling, changes assignment of votes to buckets
            # needs to be commented for reproducibility if you do median-of-means, for just mean or median it should have no effect
            np.random.seed()
            np.random.shuffle(indices) # ~ comment for reproducibility
            
            # ~ method 2 (see below for method 1): works for buckets of potentially unequal sizes
            grad_transformer = self._draco_lite_aggregation_buffer[indices, :]
            buckets = np.array_split(grad_transformer, self._bucket_size, axis=0) # ~ returns list, won't raise exception if the split does not result in an equal division
            grad_mean = [np.mean(x, axis=0) for x in buckets]
            aggr_res = np.median(np.array(grad_mean), axis=0)
            aggr_res = np.squeeze(aggr_res) # ~ removes extra dimension of size 1 to match shape with that of method 1
            
            # ~ method 1 (all code below except debugging code should not be used if method 2 is uncommented)
            # splits "indices" 1D np.ndarray into self._bucket_size lists of equal-sized 1-D np.ndarrays of length (K/r)/self._bucket_size == no. of buckets.
            # Then, it concatenates them as rows to a 2D np.ndarray of shape: self._bucket_size x ((K/r)/self._bucket_size == no. of buckets)
            # random_indicies = np.array(np.split(indices, self._bucket_size))

            # ~ this assigns the rows of self._draco_lite_aggregation_buffer (1st stage majority votes, one per file) 
            # to a 3D np.ndarray of shape: (self._bucket_size) x (no. of buckets) x (total no. of "flattened" model parameters) 
            # The [i,j,:] slice is equal to self._draco_lite_aggregation_buffer[random_indicies[i,j]], i.e., 
            # it is matched with the shuffled "random_indicies", this is assigning majority votes to buckets
            # grad_transformer = self._draco_lite_aggregation_buffer[random_indicies, :]
            # num_buckets = grad_transformer.shape[0] # ~ never used

            # aggregation stage
            # ~ this should be wrong since it's aggregating across buckets (axis = 1) and then across those averages (axis = 0)
            # I think it should be axis = 0 and then axis = 1. But if you think of "--bucket-size" argument as no. of buckets, it is OK.
            
            # ~ average across dimension 1, i.e., computes mean within all buckets, 
            # output is of shape (self._bucket_size) x (total no. of "flattened" model parameters)
            # grad_mean = np.mean(grad_transformer, axis=1)
            
            # ~ test
            # np.save('DETOX_grad_mean', grad_mean)
            # if self.cur_step%self._eval_freq == 0:
                # if self._s == 0: # ~ clean
                    # np.save('DETOX_grad_mean_clean_step'+str(self.cur_step), grad_mean)
                # else: # ~ adversarial
                    # np.save('DETOX_grad_mean_byz_step'+str(self.cur_step)+'_q'+str(self._s), grad_mean)
            
            # ~ median across dimension 0, i.e., computes median of all bucket averages, 
            # output is of shape (total no. of "flattened" model parameters) x 1
            # aggr_res = np.median(grad_mean, axis=0)
            
            # ~ test
            # np.save('DETOX_aggr_res', aggr_res)
            # np.save('DETOX_grad_transformer', grad_transformer)
            # np.save('DETOX_draco_lite_aggregation_buffer', self._draco_lite_aggregation_buffer)
            # if self.cur_step%self._eval_freq == 0:
                # if self._s == 0: # ~ clean
                    # np.save('DETOX_aggr_res_clean_step'+str(self.cur_step), aggr_res)
                # else: # ~ adversarial
                    # np.save('DETOX_aggr_res_byz_step'+str(self.cur_step)+'_q'+str(self._s), aggr_res)
            
            # ~ test
            # logger.info("DEBUG_PS_DETOX: random_indicies: {} {}".format(type(random_indicies), random_indicies.shape))
            # logger.info("DEBUG_PS_DETOX: grad_transformer: {} {}".format(type(grad_transformer), grad_transformer.shape))
            # logger.info("DEBUG_PS_DETOX: grad_mean: {} {}".format(type(grad_mean), grad_mean.shape))
            # logger.info("DEBUG_PS_DETOX: aggr_res: {} {}".format(type(aggr_res), aggr_res.shape))
            
        elif self._update_mode == "sign-sgd":
            # Below, you need to one method, say "method x" and uncomment the corresponding lines labeled "method x"
            
            #logger.info("Draco-lite aggregation buffer shape: {}".format(self._draco_lite_aggregation_buffer.shape))
            indices = np.arange(self._sub_grad_size)
            np.random.seed()
            np.random.shuffle(indices) # ~ comment for reproducibility

            # ~ method 2 (see coord-median for comments): works for buckets of potentially unequal sizes
            grad_transformer = self._draco_lite_aggregation_buffer[indices, :]
            buckets = np.array_split(grad_transformer, self._bucket_size, axis=0)
            tempt_aggr_res = [np.sign(np.sum(x, axis=0)) for x in buckets]
            aggr_res = np.sign(np.sum(np.array(tempt_aggr_res), axis=0))
            aggr_res = np.squeeze(aggr_res)
            
            # ~ method 1 (all code below except debugging code should not be used if method 2 is uncommented)
            # random_indicies = np.array(np.split(indices, self._bucket_size))

            #logger.info("random indicies: {}".format(random_indicies))
            # grad_transformer = self._draco_lite_aggregation_buffer[random_indicies, :]

            # aggregation stage
            # the first maj vote stage
            # tempt_aggr_res = np.sign(np.sum(grad_transformer, axis=1))
            
            # the second maj vote stage
            # aggr_res = np.sign(np.sum(tempt_aggr_res, axis=0))
            
            # ~ test
            # np.save('DETOX_tempt_aggr_res', tempt_aggr_res)
            # np.save('DETOX_aggr_res', aggr_res)
        
        elif self._update_mode == "bulyan":
            def __bulyan(grad_list, nb_in_score, bulyan_inds): # ~ see byzshield_master for comments
                '''
                Method introduced by: https://arxiv.org/abs/1703.02757
                '''
                neighbor_distances = []
                for i, g_i in enumerate(grad_list):
                    distance = []
                    for j in range(i+1, len(grad_list)):
                        if i != j:
                            g_j = grad_list[j]
                            distance.append(float(np.linalg.norm(g_i-g_j)**2))
                    neighbor_distances.append(distance)

                scores = []
                for i, g_i in enumerate(grad_list):
                    dists = []
                    for j, g_j in enumerate(grad_list):
                        if j == i:
                            continue
                        if j < i:
                            dists.append(neighbor_distances[j][i - j - 1])
                        else:
                            dists.append(neighbor_distances[i][j - i - 1])
                    # alternative to topk in pytorch and tensorflow
                    topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
                    scores.append(sum(np.take(dists, topk_ind)))

                topk_ind = np.argpartition(scores, nb_in_score)[:nb_in_score]
                selected_grads = np.array(grad_list)[topk_ind, :]

                # starting the second stage bulyan step
                # let's start to parallelize this step: the strategy is parallel([nb_in_score, d/k], ..., [nb_in_score, d/k])
                grad_dim = selected_grads.shape[1]
                
                # ~ test
                # logger.info("DEBUG_PS_DETOX: Bulyan grad_dim: {}".format(grad_dim))

                temp_selected_grads = []
                num_pieces = 8
                segment_size = int(grad_dim / num_pieces)
                sement_counter = 0
                for i in range(num_pieces-1):
                    temp_selected_grads.append(selected_grads[:, sement_counter:sement_counter+segment_size])
                    sement_counter += segment_size
                temp_selected_grads.append(selected_grads[:, sement_counter:])

                temp_sorted_selected_grads = Parallel(n_jobs=-1, prefer="threads")(delayed(np.sort)(g, axis=0) for g in temp_selected_grads)
                sorted_selected_grads = np.concatenate(temp_sorted_selected_grads, axis=1)
                
                bulyan_selected_grads = sorted_selected_grads[bulyan_inds, :]
                aggregated_grad = np.mean(bulyan_selected_grads, axis=0)
                return aggregated_grad

            # figure out bulyan statistics
            # where `effective_s` denotes the worst case number of Byzantine workers after the majority voting stage # ~ "groups" not workers
            effective_s = math.floor(self._s / math.ceil(self._group_size/2)) # ~ max. no. of distorted groups, same as q/((r+1)/2) for odd r (easy proof)
            
            # ~ test
            # logger.info("DEBUG_PS_DETOX: Bulyan effective_s: {}".format(effective_s))
                
            nb_in_score = self._draco_lite_aggregation_buffer.shape[0] - effective_s - 2
            pivot = int(nb_in_score/2)
            bulyan_s = nb_in_score - 2 * effective_s

            if nb_in_score % 2 == 0:
                # even case
                bulyan_inds = [(pivot- 1 - i) for i in range(1, int((bulyan_s-2)/2)+1)] + [pivot-1, pivot] + [(pivot + i) for i in range(1, int((bulyan_s-2)/2)+1)]
            else:
                # odd case
                bulyan_inds = [(pivot - i) for i in range(1, int((bulyan_s-1)/2)+1)] + [pivot] + [(pivot + i) for i in range(1, int((bulyan_s-1)/2)+1)]

            aggr_res = __bulyan(self._draco_lite_aggregation_buffer, nb_in_score, bulyan_inds)

        elif self._update_mode == "multi-krum":
            def __krum(grad_list, s, num_workers):
                '''
                Method introduced by: https://arxiv.org/abs/1703.02757
                '''
                neighbor_distances = []
                for i, g_i in enumerate(grad_list):
                    distance = []
                    for j in range(i+1, len(grad_list)):
                        if i != j:
                            g_j = grad_list[j]
                            distance.append(float(np.linalg.norm(g_i-g_j)**2))
                    neighbor_distances.append(distance)

                # compute scores
                nb_in_score = num_workers - s - 2
                scores = []
                for i, g_i in enumerate(grad_list):
                    dists = []
                    for j, g_j in enumerate(grad_list):
                        if j == i:
                            continue
                        if j < i:
                            dists.append(neighbor_distances[j][i - j - 1])
                        else:
                            dists.append(neighbor_distances[i][j - i - 1])
                    # alternative to topk in pytorch and tensorflow
                    topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
                    scores.append(sum(np.take(dists, topk_ind)))

                topk_ind = np.argpartition(scores, nb_in_score)[:nb_in_score]
                
                # ~ test
                # logger.info("DEBUG_PS_DETOX: topk_ind: {}".format(topk_ind))
                
                aggregated_grad = np.mean(np.array(grad_list)[topk_ind, :], axis=0)
                return aggregated_grad

            effective_s = math.floor(self._s / math.ceil(self._group_size/2)) # ~ max. no. of distorted groups, same as q/((r+1)/2) for odd r (easy proof)
            
            # ~ test
            # logger.info("DEBUG_PS_DETOX: Multi-Krum effective_s: {}".format(effective_s))
            
            # hard coded for now
            #temp_aggr_res = []
            #for i in range(2):
            #    temp_grads = grad_transformer[i]
            #    temp = __krum(temp_grads, effective_s, temp_grads.shape[0])
            #    temp_aggr_res.append(temp)
            #aggr_res_temp = np.concatenate(temp_aggr_res, axis=0)

            # with bucketing:
            # bucketing stage in multi-krum backened draco-lite
            indices = np.arange(self._sub_grad_size)
            np.random.seed()
            np.random.shuffle(indices) # ~ comment for reproducibility

            ## Note that this part is hard coded currently
            # random_indicies = np.array_split(indices, 2) # ~ this splits the 15 groups of DETOX paper into 2 groups of sizes [8,7] s.t. Multi-Krum minimum requirements are satisfied
            random_indicies = np.array_split(indices, self._bucket_size)
            
            # ~ test
            # logger.info("DEBUG_PS_DETOX: random_indicies: {}".format(random_indicies))
            
            grad_transformer = np.array([self._draco_lite_aggregation_buffer[rand_inds, :] for rand_inds in random_indicies])
            # aggr_res_temp = Parallel(n_jobs=-1, prefer="threads")(delayed(__krum)(grad_transformer[i], effective_s, grad_transformer[i].shape[0]) for i in range(2)) # ~ for DETOX paper figures only
            aggr_res_temp = Parallel(n_jobs=-1, prefer="threads")(delayed(__krum)(grad_transformer[i], effective_s, grad_transformer[i].shape[0]) for i in range(self._bucket_size))
            
            # ~ test
            # np.save('DETOX_aggr_res_temp', aggr_res_temp)
            
            aggr_res = np.mean(np.array(aggr_res_temp), axis=0)

            # without bucketing:
            #aggr_res = __krum(self._draco_lite_aggregation_buffer, effective_s, self._draco_lite_aggregation_buffer.shape[0])

        index_pointer = 0
        for j, p in enumerate(self.network.parameters()):
            grad_size = reduce((lambda x, y: x * y), p.size())
            drac_lite_median = aggr_res[index_pointer:index_pointer+grad_size]
            index_pointer += grad_size
            self._grad_aggregate_buffer[j] = drac_lite_median.reshape(self._grad_aggregate_buffer[j].shape)
            
            # ~ test
            # np.save('DETOX_grad_aggregate_buffer_lay'+str(j), self._grad_aggregate_buffer[j])


    def _grad_majority_vote(self):
        for k, v in self._coded_grads_buffer.items():
            index_pointer = 0
            for j, p in enumerate(self.network.parameters()):
                grad_size = reduce((lambda x, y: x * y), p.size())
                _maj_counter = 0

                for i, elem in enumerate(v):
                
                    # ~ test
                    # v: list of ndarray of shape...
                    # logger.info("DEBUG_PS_DETOX: Master grad type for worker ctr = {}: {}, subtype: {}, shape: {}, dtype: {}".format(i, type(v[0]), type(v[0][0]), v[0][0].shape, v[0][0].dtype))
                    
                    # ~ test
                    # for LENET only
                    # logger.info("DEBUG_PS_DETOX: Master grad value for worker ctr = {}: value: {}".format(i, v[0][0][0][0][0]))
                
                    if _maj_counter == 0:
                        _maj_grad = elem[j]
                        _maj_counter = 1
                    elif np.array_equal(elem[j], _maj_grad):
                        _maj_counter += 1
                    else:
                        _maj_counter -= 1
                        
                    # ~ test
                    # if k == 0: # ~ k-th group
                    # np.save('DETOX_coded_grads_buffer_groupInd_'+str(k)+'_workerInGroupInd_'+str(i)+'_lay'+str(j), elem[j])
                        
                # ~ test
                # if k == 0: # ~ k-th group
                # np.save('DETOX_draco_lite_aggregation_buffer_groupInd_'+str(k)+'_lay'+str(j), _maj_grad)
                        
                # ~ Kostas test, remove this
                # assert 0 == 1
                
                try:
                    assert self._grad_aggregate_buffer[j].shape == _maj_grad.shape
                except AssertionError as e:
                    warnings.warn("Gradient shapes incompatible, deprecated! ")

                self._draco_lite_aggregation_buffer[k, index_pointer:index_pointer+grad_size] = _maj_grad.reshape(-1)
                index_pointer += grad_size

    def meset_grad_buffer(self):
        for i in range(len(self._grad_aggregate_buffer)):
            self._grad_aggregate_buffer[i] = np.zeros(self._grad_aggregate_buffer[i].shape, dtype=float_type)

        self._draco_lite_aggregation_buffer = np.zeros((self._sub_grad_size, self._model_param_counter), dtype=float_type)

    def _LIE_attack_simulation(self):
        """
        Simulating the attack method in: https://arxiv.org/abs/1902.06156
        (Otheriwse, 2-round communication strategy needs to be implemented for each iteration)
        The process is farily simple: (p_mal)_i = mu_i + z^{max} * simga_i, for i in {1, ..., d}
        """
        self.__z = 1.0
        # dummpy_adversarial_nodes = np.random.choice(self.num_workers, self._s, replace=False)
        
        # ~ option 2: choice of Byzantines won't change for ALIE
        dummpy_adversarial_nodes = self._fail_workers[self.cur_step]
        
        # ~ test
        # logger.info("DEBUG_PS_BYZ: ALIE byzantines: {}".format(dummpy_adversarial_nodes))
        
        for j, _ in enumerate(self.network.parameters()):
            # aggregate coded gradients from different groups together, this is stupid, try to avoid this
            tempt_grads = []
            for k, v in self._coded_grads_buffer.items():
                for i, elem in enumerate(v):
                    tempt_grads.append(elem[j])
            mu, sigma = np.mean(tempt_grads, axis=0), np.std(tempt_grads, axis=0)

            for adv_index in dummpy_adversarial_nodes:
                for k, v in self._coded_grads_buffer.items():
                    if adv_index in self._group_list[k]:
                        _mal_grad = mu + self.__z * sigma
                        _relative_index_in_group = self._group_list[k].index(adv_index)
                        assert self._coded_grads_buffer[k][_relative_index_in_group][j].shape == _mal_grad.shape
                        self._coded_grads_buffer[k][_relative_index_in_group][j] =  mu + self.__z * sigma