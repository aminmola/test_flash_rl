import copy
import timeit
import clientFL.Client as Client
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import torch.nn as nn
import RL.DQL as DQL
import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score
from torch.nn import functional as F

class Server_dueling(object):
    
    def __init__(self, num_clients, global_model, dict_clients, loss_fct, B, dataset_test, learning_rate, momentum, clients_info):
        """

        Parameters:
        - num_clients: The number of clients in the system.
        - global_model: The global model.
        - dict_clients: A dictionary containing information about each client.
        - loss_fct: The loss function used for training.
        - B: The size of the batch.
        - dataset_test: The test dataset used for evaluation.
        - learning_rate: The learning rate for training.
        - momentum: The momentum parameter for training.
        - clients_info: Information about the clients for simulation purposes.
        """
        # The number of clients in the system
        self.N = num_clients
        # The global model (our goal)
        self.model = global_model 
        # A list containing the @ of each client
        self.list_clients =  [] 
        # The size of the batch 
        self.B = B
        # The test dataset that helps us evaluate each round
        self.dataset_test = dataset_test
        # Test Dataloader
        self.testdataloader = DataLoader(self.dataset_test, batch_size= self.B)
        # The distribution of clients in a dict
        self.dict_clients = dict_clients
        # The loss function
        self.loss_function = copy.deepcopy(loss_fct)
        # The number of parameters
        self.number_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # RL reward normalizers / EMAs
        # Keep running mean/var for normalization of reward components
        self._norm_states = {
            'acc': {'m':0.0, 'v':1.0},
            'loss': {'m':0.0, 'v':1.0},
            'div': {'m':0.0, 'v':1.0},
            'lat': {'m':0.0, 'v':1.0},
            'diversity': {'m':0.0, 'v':1.0}
        }
        self._ema_reward = 0.0
        self._ema_acc = 0.0
        self._ema_f1 = 0.0
        self._ema_alpha = 0.01          # EMA smoothing
        self._gamma_rl = 0.99           # RL discount for potential shaping

        # Reward weights (can be tuned)
        self._rw_alpha = 1.0    # accuracy/F1 gain
        self._rw_beta  = 0.5    # loss gain
        self._rw_gamma = 0.3    # distance utility (per-client)
        self._rw_delta = 0.1    # latency penalty (per-client)
        self._rw_zeta  = 0.2    # diversity bonus (global)

        # Distance temperature
        self._dist_k = 5.0
        # Soft-sign scale for gains
        self._soft_sigma = 1e-3
        # Potential-based shaping scale
        self._phi_eta = 0.5

        # Call a function to create the clients (simulation)
        self.clients_info = clients_info
        self.create_clients(learning_rate, momentum)

    # -------------- helpers for reward shaping --------------
    def _ema_update(self, prev, x, alpha):
        return (1.0 - alpha) * prev + alpha * x

    def _norm_update(self, key, x):
        st = self._norm_states[key]
        m = st['m'] = 0.99 * st['m'] + 0.01 * x
        v = st['v'] = 0.99 * st['v'] + 0.01 * (x - m) ** 2
        return (x - m) / (np.sqrt(v) + 1e-6)

    def _pairwise_cosine_distances(self, vectors):
        # vectors: list/array of 1D arrays (same length)
        if len(vectors) <= 1:
            return 0.0, np.zeros(len(vectors))
        X = np.stack(vectors, axis=0).astype(np.float64)
        # normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Xn = X / norms
        S = Xn @ Xn.T  # cosine similarity matrix
        D = 1.0 - S    # cosine distance
        # mean pairwise distance (upper triangle without diag)
        n = D.shape[0]
        mean_pair = D[np.triu_indices(n, k=1)].mean()
        # per-vector average distance to others
        per_vec = (D.sum(axis=1) - np.diag(D)) / np.maximum(n - 1, 1)
        return float(mean_pair), per_vec

    def _build_rewards_dense(
        self,
        metric_gain,             # scalar (acc or f1 gain)
        loss_gain,               # scalar
        divergence_per_client,   # array shape [m_selected]
        latency_per_client,      # array shape [m_selected]
        weight_vectors           # list of flattened local weights for selected clients
    ):
        # Normalize scalar parts
        acc_n  = self._norm_update('acc',  metric_gain)
        loss_n = self._norm_update('loss', loss_gain)

        # Soft sign for shaping from metric gain
        soft_sign = np.tanh(metric_gain / (self._soft_sigma + 1e-8))

        # Per-client soft utility from divergence (bounded, non-saturating)
        # divergence is non-negative; higher is worse -> smaller u_i
        u_i = 1.0 / (1.0 + self._dist_k * np.maximum(divergence_per_client, 0.0))

        # Diversity (global) and per-client diversity (optional)
        diversity_global, _ = self._pairwise_cosine_distances(weight_vectors)
        diversity_n = self._norm_update('diversity', diversity_global)

        # Normalize per-client latency and divergence for stability
        # Here divergence_n is for logging consistency (not used directly since u_i already bounded)
        lat_n = np.array([self._norm_update('lat', float(l)) for l in latency_per_client], dtype=np.float64)

        # Combine (per-client rewards)
        # r_i = α*acc + β*loss + γ*(soft_sign*(u_i-0.5)) + ζ*diversity - δ*lat
        r_vec = (
            self._rw_alpha * acc_n
            + self._rw_beta  * loss_n
            + self._rw_gamma * (soft_sign * (u_i - 0.5))
            + self._rw_zeta  * diversity_n
            - self._rw_delta * lat_n
        ).astype(np.float64)

        # Clip to [-1,1] to keep TD targets stable
        r_vec = np.clip(r_vec, -1.0, 1.0)

        # Baseline (advantage-style): subtract EMA of recent rewards
        mean_r = float(np.mean(r_vec)) if r_vec.size > 0 else 0.0
        self._ema_reward = self._ema_update(self._ema_reward, mean_r, self._ema_alpha)
        r_vec = r_vec - self._ema_reward

        return r_vec

    # --------------------------------------------------------

    def create_clients(self, learning_rate, momentum):
        '''
        Create client objects based on the provided information and add them to the list.

        Parameters:
        - learning_rate: The learning rate for training clients.
        - momentum: The momentum parameter for training clients.
        '''
        cpt = 0
        for client_name in self.dict_clients.keys():
            # Create a client object with the specified parameters
            client = Client.Client(
                self.clients_info[cpt][0],  # client name
                self.dict_clients[client_name],  # client information
                copy.deepcopy(self.model),  # copy of the global model
                self.clients_info[cpt][2],  # number of cores
                self.clients_info[cpt][3],  # frequency distribution
                self.clients_info[cpt][4],  # bandwidth distribution
                copy.deepcopy(self.loss_function),  # copy of the loss function
                self.B,  # size of the batch
                learning_rate,  # learning rate
                momentum  # momentum
            )
            # Append the client to the list
            self.list_clients.append(client)
            cpt += 1


    def weight_scalling_factor(self, client, active_clients):
        '''
        Determine the factor assigned to a given client.

        Parameters:
        - client: The client for which the factor is determined.
        - active_clients: List of active clients.

        Returns:
        - The scaling factor for the client.
        '''
        # Calculate the total training data points across clients for this round
        global_count = sum([client_obj.get_size() for client_obj in active_clients])
        # Get the total number of data points held by a client
        local_count = client.get_size()

        return local_count / global_count

    def scale_model_weights(self, weight, scalar):
        '''
        Multiply the local parameters of each client by its factor.

        Parameters:
        - weight: The local model weights.
        - scalar: The scaling factor.

        Returns:
        - Scaled local model weights.
        '''
        w_scaled = copy.deepcopy(weight)

        for k in weight.keys():
            w_scaled[k] = scalar * w_scaled[k]

        return w_scaled

    def sum_scaled_weights(self, scaled_weight_list):
        '''
        Aggregate different parameters.

        Parameters:
        - scaled_weight_list: List of scaled models weights.

        Returns:
        - Aggregated weights.
        '''
        w_avg = copy.deepcopy(scaled_weight_list[0])

        for k in w_avg.keys():
            tmp = torch.zeros_like(scaled_weight_list[0][k], dtype=torch.float32)

            for i in range(len(scaled_weight_list)):
                tmp += scaled_weight_list[i][k]

            w_avg[k].copy_(tmp)

        return w_avg
    
    
    def flatten(self, weight):
        '''
        Flatten the parameters of a model.

        Parameters:
        - weight: The model weights.

        Returns:
        - Flattened model weights.
        '''
        weight_flatten = []

        for param in weight.values():
            weight_flatten.append(np.array(param).reshape(-1))

        weight_flatten = [item for sublist in weight_flatten for item in sublist]

        return weight_flatten

    def flatten_state(self, state_list):
        '''
        Flatten a list of states.

        Parameters:
        - state_list: List of states where each state is represented as a list of sublists.

        Returns:
        - Flattened list of states as a torch.Tensor.
        '''
        result_list = []
        max_length = max(len(sublist) for sublist in state_list)  # Find the maximum length of sublists

        for i in range(max_length):
            for sublist in state_list:
                if i < len(sublist):
                    element = sublist[i]
                    if isinstance(element, list):
                        result_list.extend(element)
                    else:
                        result_list.append(element)
        
        return torch.Tensor(result_list)

    
    def select_active_clients_random(self, comm_round, C):
        '''
        Randomly select a fraction of active clients for a training round.

        Parameters:
        - comm_round: The communication round.
        - C: Fraction of active clients.

        Returns:
        - List of indices representing the selected active clients.
        '''
        client_index = np.arange(0, len(self.list_clients))
        
        m = int(max(C * self.N, 1))  # max between c*k and 1

        active_clients = random.sample(list(client_index), k=m)  # Select a fraction of clients
        
        return active_clients

        
    def global_train(self, comms_round, C, E, mu, lamb, rep_init, batch_size, verbose_test=1, verbos=0, type_data="others", init_weights=None, init_commround=None, reputation_init=None):
        '''
        Initiate the global training process for federated learning.

        Parameters:
        - comms_round: Number of communication rounds.
        - C: Fraction of active clients in each round.
        - E: Number of local training iterations.
        - mu: FedProx parameter.
        - lamb: coaf of past contribtions in the reputation value.
        - rep_init: Initial reputation value for clients.
        - batch_size: Size of the RL batch.
        - verbose_test: Whether to print test results (1 for yes, 0 for no).
        - verbos: Verbosity level for training logs.
        - type_data: Type of data used for training (e.g., "others" or "Fall").
        - init_weights: Initial weights for the global model.
        - init_commround: Initial communication round.
        - reputation_init: Initial reputation values for clients.

        Returns:
        - Dictionary containing training results.
        '''
        if type_data == "Fall":
            return self.global_train_fall(comms_round, C, E, mu, lamb, rep_init, batch_size, verbose_test, verbos, type_data, init_weights, init_commround, reputation_init)
        else:
            return self.global_train_others(comms_round, C, E, mu, lamb, rep_init, batch_size, verbose_test, verbos, type_data, init_weights, init_commround, reputation_init)


    def global_train_others(self, comms_round, C, E, mu, lamb, rep_init, batch_size, verbose_test = 1, verbos = 0, type_data = "others", 
                            init_weights = None, init_commround = None, reputation_init = None):
        '''
        Federated learning training process for a scenario involving multiple clients for fall detection.

        Parameters:
        - comms_round: Number of communication rounds.
        - C: Fraction of active clients in each round.
        - E: Number of local training iterations.
        - mu: FedProx parameter.
        - lamb: coaf of past contribtions in the reputation value.
        - rep_init: Initial reputation value for clients.
        - batch_size: Size of the RL batch.
        - verbose_test: Whether to print test results (1 for yes, 0 for no).
        - verbos: Verbosity level for training logs.
        - type_data: Type of data used for training (e.g., "others" or "Fall").
        - init_weights: Initial weights for the global model.
        - init_commround: Initial communication round.
        - reputation_init: Initial reputation values for clients.

        Returns:
        - Dictionary containing training results.
        '''  
        
        m = int(max(C*self.N, 1))
        
        rounds = []
        accuarcy = []
        reputation_list = []
        rewards = []
        loss = []
        
        time_rounds = []
        time_rounds_sum = []
        best_model_weights = {}
        best_accuracy = 0
    

        num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
       # Initialize the first state
        weight_list_for_iteration = []
        numbersamples_for_iteration = []
        numbercores_for_iteration = []
        frequencies_for_iteration = []
        bandwidth_for_iteration = []
        
        
        # Append weight of global model
        max_latency = 0
        min_latency = 100000000
        
        if (init_weights != None) :
            reputation_clients_t = reputation_init
            reputation_list.append(copy.deepcopy(reputation_clients_t))
                                   
            #print('modèle préentrainner')
            self.model.load_state_dict(copy.deepcopy(init_weights))
            
            comm_start = init_commround
            
            for client in self.list_clients :
                client.set_weights(copy.deepcopy(init_weights))
            
            # For each client perform one epoch of SGD to get the weights
            for client in self.list_clients:
                frequency_client = random.choice(client.frequency)
                bandwidth_client = random.choice(client.bandwidth)
                
                latency_min = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * max(client.frequency)) + (self.number_parameters * 64)/(1000000 * max(client.bandwidth))
                latency_max = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * min(client.frequency)) + (self.number_parameters * 64)/(1000000 * min(client.bandwidth))

                if (latency_max > max_latency ):
                    max_latency = latency_max 
                    
                if (latency_min < min_latency):
                    min_latency = latency_min
            
                client_w_for_first_iteration = client.get_model()
                
                weight_list_for_iteration.append(self.flatten(client_w_for_first_iteration))
                numbersamples_for_iteration.append(client.get_size())
                numbercores_for_iteration.append(client.numbercores)
                frequencies_for_iteration.append(frequency_client)
                bandwidth_for_iteration.append(bandwidth_client)
         
        else :
            reputation_clients_t = np.full(self.N, rep_init)
            reputation_list.append(copy.deepcopy(reputation_clients_t))
            comm_start = 0
            # For each client perform one epoch of SGD to get the weights
            for client in self.list_clients:
                frequency_client = random.choice(client.frequency)
                bandwidth_client = random.choice(client.bandwidth)
                
                latency_min = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * max(client.frequency)) + (self.number_parameters * 64)/(1000000 * max(client.bandwidth))
                latency_max = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * min(client.frequency)) + (self.number_parameters * 64)/(1000000 * min(client.bandwidth))

                if (latency_max > max_latency ):
                    max_latency = latency_max 
                    
                if (latency_min < min_latency):
                    min_latency = latency_min
            
                client_w_for_first_iteration = client.train(self.model.state_dict(), 1, mu, type_data, verbos)
                
                weight_list_for_iteration.append(self.flatten(client_w_for_first_iteration))
                numbersamples_for_iteration.append(client.get_size())
                numbercores_for_iteration.append(client.numbercores)
                frequencies_for_iteration.append(frequency_client)
                bandwidth_for_iteration.append(bandwidth_client)
            
        # Apply PCA
        
        pca = PCA(n_components = len(self.list_clients))
  
        # get the weight with PCA
        weight_list_for_iteration_pca = pca.fit_transform(weight_list_for_iteration)
    
        state_list = []

        for cpt in range(0, len(self.list_clients)):
            client_state = []
            
            client_state.append(list(weight_list_for_iteration_pca[cpt]))
            client_state.append(numbersamples_for_iteration[cpt])
            client_state.append(numbercores_for_iteration[cpt])
            client_state.append(frequencies_for_iteration[cpt])
            client_state.append(bandwidth_for_iteration[cpt])
            
            state_list.append(client_state)  
    
        # State is a concatenation of the different reduced weights
        state = self.flatten_state(state_list)
        
        # init dql
        if (init_weights != None) :
            dql = DQL.DQL(len(state), len(self.list_clients), batch_size, flag = True)
        else:
            dql = DQL.DQL(len(state), len(self.list_clients), batch_size)

        Accuracy_global_pervoius, loss_test_prev = self.test(type_data)
        # keep EMAs for potential-based shaping
        self._ema_acc = self._ema_update(self._ema_acc, float(Accuracy_global_pervoius), self._ema_alpha)
        
        self.model.train()
        
        list_loss_DQL = []
       
        # For each communication round 
        for comm_round in  range(comm_start, comms_round):
            rounds.append(comm_round+1)
            
            if (verbos == 1):
                print("*************************************************************************************")
                print("Communication round n : ", comm_round + 1)
                
            # Receive the parameters of the global model (first we will have the initialized parameters)
            global_weights = self.model.state_dict()
            
            # Verify if we need to update the target network
            if (comm_round + 1) % dql.update_rate == 0:
                dql.update_target_network()
            
            
            if (comm_round == 0):
                #random selection
                active_clients_index = self.select_active_clients_random(comm_round, C)
                active_clients_index.sort()
            else:
                # epslion greedy
                active_clients_index = dql.multiaction_selection(state, C, comm_round, mode = "Mode1")
                active_clients_index.sort()
                
            # List to collect the parameters of the model * weight of each client
            scaled_local_weight_list = []
            
            # list of active client objetcs
            active_clients = [self.list_clients[i] for i in active_clients_index]

            weight_local_clients = []

            time_roundt = []
            # For each selected customer
            for client_index in active_clients_index:
                if (verbos == 1):
                     print("Entrainnement locale du client : ", client.client_name)
                
                # Training on local client data
                client_w = self.list_clients[client_index].train(global_weights, E, mu, type_data, verbos)
                
                # Append the local model weight
                wl = self.flatten(client_w)
                weight_local_clients.append(np.array(wl, dtype=np.float64))

                # MAJ des weights PCA
                state_list[client_index][0] =  list((pca.transform(np.array(self.flatten(copy.deepcopy(client_w))).reshape(1, -1)))[0])
                
                # Avoir les parametres du modele locale *  factor
                client_scaling_factor = self.weight_scalling_factor(self.list_clients[client_index], active_clients) #nk/n'
                
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor) #weights*factor
                scaled_local_weight_list.append(client_scaled_weight) #enregistrement dans la liste

                frequency_client = random.choice(self.list_clients[client_index].frequency)
                bandwidth_client = random.choice(self.list_clients[client_index].bandwidth)
                
                latency_client = (self.list_clients[client_index].get_size()*64*40*20)/(self.list_clients[client_index].numbercores * 1000000 * frequency_client) + (self.number_parameters * 64)/(1000000 * bandwidth_client)
            
                state_list[client_index][3] = frequency_client
                state_list[client_index][4] = bandwidth_client
                time_roundt.append(latency_client)

            time_rounds.append(max(time_roundt))
            time_rounds_sum.append(sum(time_roundt))
            
            # The average of the different points received
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)

            # Copy the weights in the global model
            self.model.load_state_dict(average_weights)

            Accuray_global_t, loss_test = self.test(type_data)
        
            # Test the global model each round
            if (verbose_test == 1):
                print("Training round n :", (comm_round+1),", Test accuarcy : ", round(float(Accuray_global_t)*100, 2) ,", Test loss :", round(float(loss_test), 2))
                print("*************************************************************************************")
                
                
            if (Accuray_global_t > best_accuracy):
                best_accuracy = Accuray_global_t
                best_model_weights = copy.deepcopy(average_weights)
                
                
            accuarcy.append(Accuray_global_t)
            loss.append(loss_test)
         
            # Next state
            next_state = self.flatten_state(state_list)
            
            # Action representation
            action = np.zeros(len(self.list_clients))
            action[active_clients_index] = 1

            # --------- Dense, normalized reward (per-client) ----------
            # Normalized distance between local and aggregated/global
            # Use element-wise relative change proxy as in your original code
            eps = 1e-12
            flat_global_after = np.array(self.flatten(self.model.state_dict()), dtype=np.float64)
            flat_avg          = np.array(self.flatten(average_weights), dtype=np.float64)

            # compute per-client divergence
            # normalized_distance_i = mean( |w_i - w_global| / (|w_avg| + eps) )
            # then take absolute to ensure non-negativity
            norm_den = np.abs(flat_avg) + eps
            divergences = []
            for wl in weight_local_clients:
                d = np.mean(np.abs((wl - flat_global_after) / norm_den))
                divergences.append(d)
            divergences = np.array(divergences, dtype=np.float64)

            # Normalize latencies to [0,1]
            max_latency = max(max(time_roundt), 1e-9)
            min_latency = min(time_roundt)
            lat_norm = (np.array(time_roundt, dtype=np.float64) - min_latency) / (max(max_latency - min_latency, 1e-12))

            # Gains
            acc_gain  = float(Accuray_global_t) - float(Accuracy_global_pervoius)
            loss_gain = -(float(loss_test) - float(loss_test_prev))

            # Potential-based shaping on accuracy EMA
            ema_prev = self._ema_acc
            self._ema_acc = self._ema_update(self._ema_acc, float(Accuray_global_t), self._ema_alpha)
            ema_curr = self._ema_acc
            r_shaping = self._phi_eta * (self._gamma_rl * ema_curr - ema_prev)

            # Per-client dense rewards
            reward_vec = self._build_rewards_dense(
                metric_gain=acc_gain,
                loss_gain=loss_gain,
                divergence_per_client=divergences,
                latency_per_client=lat_norm,
                weight_vectors=weight_local_clients
            )
            # Add potential-based shaping equally to all selected
            reward_vec = np.clip(reward_vec + r_shaping, -1.0, 1.0)

            # --------- Reputation update (kept for continuity) ----------
            # You can keep reputation tracking as auxiliary signal
            reputation_clients_t[active_clients_index] = (
                (1 - lamb) * reputation_clients_t[active_clients_index] 
                + lamb * ( (1.0 / (1.0 + self._dist_k * divergences)) - lat_norm )
            )
            reputation_list.append(copy.deepcopy(reputation_clients_t))

            # what the DQL expects as reward (vector over selected clients)
            reward = reward_vec.astype(np.float64)
            rewards.append(reward)

            # Store transition
            if (comm_round == comms_round - 1):
                dql.store_transistion(state, action, reward, next_state, done = True)
            else:
                dql.store_transistion(state, action, reward, next_state, done = False)
                
            # Update trackers
            state = copy.deepcopy(next_state)
            Accuracy_global_pervoius = Accuray_global_t
            loss_test_prev = loss_test

            loss_dql = dql.train(comm_round, mode = "Mode1")
            list_loss_DQL.append(loss_dql)
            
        for client in self.list_clients:
            client.delete_model()
            
        dict_result = {
            "Best_model_weights": best_model_weights,
            "Accuracy": accuarcy,
            "Loss" : loss,
            "Timeurounds" : time_rounds,
            "Timesum" : time_rounds_sum,
            "Reputation" : reputation_list,
            "Rewards" : rewards,
            "LossDQL" : list_loss_DQL
        }

        return dict_result
    
    def global_train_fall(self, comms_round, C, E, mu, lamb, rep_init, batch_size, verbose_test = 1, verbos = 0, type_data = "others",
                          init_weights = None, init_commround = None, reputation_init = None):
        '''
        Federated learning training process for a scenario involving multiple clients.

        Parameters:
        - comms_round: Number of communication rounds.
        - C: Fraction of active clients in each round.
        - E: Number of local training iterations.
        - mu: FedProx parameter.
        - lamb: coaf of past contribtions in the reputation value.
        - rep_init: Initial reputation value for clients.
        - batch_size: Size of the RL batch.
        - verbose_test: Whether to print test results (1 for yes, 0 for no).
        - verbos: Verbosity level for training logs.
        - type_data: Type of data used for training (e.g., "others" or "Fall").
        - init_weights: Initial weights for the global model.
        - init_commround: Initial communication round.
        - reputation_init: Initial reputation values for clients.

        Returns:
        - Dictionary containing training results.
        '''  
        
        m = int(max(C*self.N, 1))
        
        rounds = []
        accuarcy = []
        recall = []
        precision = []
        loss = []
        f1score = []
        reputation_list = []
        rewards = []
        
        time_rounds = []
        time_rounds_sum = []
        best_model_weights = {}
        best_f1score = 0
        
        reputation_clients_t = np.full(self.N, rep_init)
        reputation_list.append(copy.deepcopy(reputation_clients_t))

        num_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
       # Initialize the first state
        weight_list_for_iteration = []
        numbersamples_for_iteration = []
        numbercores_for_iteration = []
        frequencies_for_iteration = []
        bandwidth_for_iteration = []
        list_loss_DQL = []
        # Append weight of global model
        max_latency = 0
        min_latency = 100000000
        
        
        if (init_weights != None) :
            reputation_clients_t = reputation_init
            reputation_list.append(copy.deepcopy(reputation_clients_t))
                                   
            self.model.load_state_dict(copy.deepcopy(init_weights))
            
            comm_start = init_commround
            
            for client in self.list_clients :
                client.set_weights(copy.deepcopy(init_weights))
            
            # For each client perform one epoch of SGD to get the weights
            for client in self.list_clients:
                frequency_client = random.choice(client.frequency)
                bandwidth_client = random.choice(client.bandwidth)
                
                latency_min = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * max(client.frequency)) + (self.number_parameters * 64)/(1000000 * max(client.bandwidth))
                latency_max = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * min(client.frequency)) + (self.number_parameters * 64)/(1000000 * min(client.bandwidth))

                if (latency_max > max_latency ):
                    max_latency = latency_max 
                    
                if (latency_min < min_latency):
                    min_latency = latency_min
            
                client_w_for_first_iteration = client.get_model()
                
                weight_list_for_iteration.append(self.flatten(client_w_for_first_iteration))
                numbersamples_for_iteration.append(client.get_size())
                numbercores_for_iteration.append(client.numbercores)
                frequencies_for_iteration.append(frequency_client)
                bandwidth_for_iteration.append(bandwidth_client)
         
        else :
            reputation_clients_t = np.full(self.N, rep_init)
            reputation_list.append(copy.deepcopy(reputation_clients_t))
            comm_start = 0
            # For each client perform one epoch of SGD to get the weights
            for client in self.list_clients:
                frequency_client = random.choice(client.frequency)
                bandwidth_client = random.choice(client.bandwidth)
                
                latency_min = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * max(client.frequency)) + (self.number_parameters * 64)/(1000000 * max(client.bandwidth))
                latency_max = (client.get_size()*64*40*20)/(client.numbercores * 1000000 * min(client.frequency)) + (self.number_parameters * 64)/(1000000 * min(client.bandwidth))

                if (latency_max > max_latency ):
                    max_latency = latency_max 
                    
                if (latency_min < min_latency):
                    min_latency = latency_min
            
                client_w_for_first_iteration = client.train(self.model.state_dict(), 1, mu, type_data, verbos)
                
                weight_list_for_iteration.append(self.flatten(client_w_for_first_iteration))
                numbersamples_for_iteration.append(client.get_size())
                numbercores_for_iteration.append(client.numbercores)
                frequencies_for_iteration.append(frequency_client)
                bandwidth_for_iteration.append(bandwidth_client)
        
        pca = PCA(n_components = len(self.list_clients))
  
        # get the weight with PCA
        weight_list_for_iteration_pca = pca.fit_transform(weight_list_for_iteration)

        state_list = []
        
        for cpt in range(0, len(self.list_clients)):
            client_state = []
            
            client_state.append(list(weight_list_for_iteration_pca[cpt]))
            client_state.append(numbersamples_for_iteration[cpt])
            client_state.append(numbercores_for_iteration[cpt])
            client_state.append(frequencies_for_iteration[cpt])
            client_state.append(bandwidth_for_iteration[cpt])
            
            state_list.append(client_state)  
    
        # State is a concatenation of the different reduced weights
        state = self.flatten_state(state_list)
        
        # init dql
        if (init_weights != None) :
            dql = DQL.DQL(len(state), len(self.list_clients), batch_size, flag = True)
        else:
            dql = DQL.DQL(len(state), len(self.list_clients), batch_size)

        _, F1score_global_pervoius, _, _, loss_test_prev = self.test(type_data)

        # initialize EMA for F1
        self._ema_f1 = self._ema_update(self._ema_f1, float(F1score_global_pervoius), self._ema_alpha)

        client_selected = []
        self.model.train()
        
        # For each communication round 
        for comm_round in  tqdm(range(comm_start, comms_round)):
            rounds.append(comm_round+1)
            
            if (verbos == 1):
                print("*************************************************************************************")
                print("Communication round n : ", comm_round + 1)
                
            # Receive the parameters of the global model (first we will have the initialized parameters)
            global_weights = self.model.state_dict()
            
            # Verify if we need to update the target network
            if (comm_round + 1) % dql.update_rate == 0:
                dql.update_target_network()
            
            
            if (comm_round == 0):
                #random selection
                active_clients_index = self.select_active_clients_random(comm_round, C)
                active_clients_index.sort()
            else:
                # epslion greedy
                active_clients_index = dql.multiaction_selection(state, C, comm_round, mode = "Mode1")
                active_clients_index.sort()
             
            if (comm_round == 500 ):
               active_clients_index = [1, 11, 27, 31, 47]   
               
            if (comm_round == 501 ):
               active_clients_index = [1, 2, 24, 39, 44]

            client_selected.append(active_clients_index)

            scaled_local_weight_list = []
            active_clients = [self.list_clients[i] for i in active_clients_index]
            weight_local_clients = []
            time_roundt = []

            for client_index in active_clients_index:
                if (verbos == 1):
                     print("Entrainnement locale du client : ", client.client_name)
                
                client_w = self.list_clients[client_index].train(global_weights, E, mu, type_data, verbos)
                
                wl = self.flatten(client_w)
                weight_local_clients.append(np.array(wl, dtype=np.float64))

                state_list[client_index][0] =  list((pca.transform(np.array(self.flatten(copy.deepcopy(client_w))).reshape(1, -1)))[0])
                
                client_scaling_factor = self.weight_scalling_factor(self.list_clients[client_index], active_clients)
                
                client_scaled_weight = self.scale_model_weights(client_w, client_scaling_factor)
                scaled_local_weight_list.append(client_scaled_weight)

                frequency_client = random.choice(self.list_clients[client_index].frequency)
                bandwidth_client = random.choice(self.list_clients[client_index].bandwidth)
                
                latency_client = (self.list_clients[client_index].get_size()*64*40*20)/(self.list_clients[client_index].numbercores * 1000000 * frequency_client) + (self.number_parameters * 64)/(1000000 * bandwidth_client)
            
                state_list[client_index][3] = frequency_client
                state_list[client_index][4] = bandwidth_client
                time_roundt.append(latency_client)

            time_rounds.append(max(time_roundt))
            time_rounds_sum.append(sum(time_roundt))
            
            average_weights = self.sum_scaled_weights(scaled_local_weight_list)
            self.model.load_state_dict(average_weights)

            acc_test, F1score_global_t, recall_test, precision_test, loss_test = self.test(type_data)
        
            if (verbose_test == 1):
                print("Training round n :", (comm_round+1),", Test accuarcy : ", round(acc_test*100, 2) ,", Test F1_score :", round(F1score_global_t*100, 2)," Test loss :", round(loss_test, 2))
                print("Test Recall : ", round(recall_test*100, 2), "Test Precision : ", round(precision_test*100, 2))
                print("*************************************************************************************")
                
            if (F1score_global_t > best_f1score):
                best_f1score = F1score_global_t
                best_model_weights = copy.deepcopy(average_weights)
                
                
            accuarcy.append(acc_test)
            f1score.append(F1score_global_t)
            recall.append(recall_test)
            precision.append(precision_test)
            loss.append(loss_test)
         
            next_state = self.flatten_state(state_list)
            
            action = np.zeros(len(self.list_clients))
            action[active_clients_index] = 1

            # --------- Dense, normalized reward (per-client) ----------
            eps = 1e-12
            flat_global_after = np.array(self.flatten(self.model.state_dict()), dtype=np.float64)
            flat_avg          = np.array(self.flatten(average_weights), dtype=np.float64)
            norm_den = np.abs(flat_avg) + eps

            divergences = []
            for wl in weight_local_clients:
                d = np.mean(np.abs((wl - flat_global_after) / norm_den))
                divergences.append(d)
            divergences = np.array(divergences, dtype=np.float64)

            max_latency = max(max(time_roundt), 1e-9)
            min_latency = min(time_roundt)
            lat_norm = (np.array(time_roundt, dtype=np.float64) - min_latency) / (max(max_latency - min_latency, 1e-12))

            f1_gain   = float(F1score_global_t) - float(F1score_global_pervoius)
            loss_gain = -(float(loss_test) - float(loss_test_prev))

            # Potential-based shaping on F1 EMA
            ema_prev = self._ema_f1
            self._ema_f1 = self._ema_update(self._ema_f1, float(F1score_global_t), self._ema_alpha)
            ema_curr = self._ema_f1
            r_shaping = self._phi_eta * (self._gamma_rl * ema_curr - ema_prev)

            reward_vec = self._build_rewards_dense(
                metric_gain=f1_gain,
                loss_gain=loss_gain,
                divergence_per_client=divergences,
                latency_per_client=lat_norm,
                weight_vectors=weight_local_clients
            )
            reward_vec = np.clip(reward_vec + r_shaping, -1.0, 1.0)

            # --------- Reputation update (auxiliary) ----------
            reputation_clients_t[active_clients_index] = (
                (1 - lamb) * reputation_clients_t[active_clients_index] 
                + lamb * ( (1.0 / (1.0 + self._dist_k * divergences)) - lat_norm )
            )
            reputation_list.append(copy.deepcopy(reputation_clients_t))
            
            reward = reward_vec.astype(np.float64)
            rewards.append(reward)
            
            if (comm_round == comms_round - 1):
                dql.store_transistion(state, action, reward, next_state, done = True)
            else:
                dql.store_transistion(state, action, reward, next_state, done = False)
                
            state = copy.deepcopy(next_state)
            F1score_global_pervoius = F1score_global_t
            loss_test_prev = loss_test

            loss_dql = dql.train(comm_round, mode = "Mode1")
            list_loss_DQL.append(loss_dql)
            
        for client in self.list_clients:
            client.delete_model()
            
        dict_result = {
            "Best_model_weights": best_model_weights,
            "Accuracy": accuarcy,
            "F1_score": f1score,
            "Recall" : recall,
            "Precision" : precision,
            "Loss" : loss,
            "Timeurounds" : time_rounds,
            "Timesum" : time_rounds_sum,
            "Reputation" : reputation_list,
            "Rewards" : rewards,
            "LossDQL" : list_loss_DQL,
            "Client_selected" : client_selected
        }

        return dict_result
    

   
    def test(self, type_data):
        """
        Test the global model on the specified type of data.

        Parameters:
        - type_data: Type of data, e.g., "Fall" or "others".

        Returns:
        - The test accuracy and loss.
        """
        if type_data == "Fall":
            return self.test_falldetection()
        else:
            return self.test_others()

    
    def test_others(self):
        """
        Evaluate the global model with the test dataset for a task other than fall detection.

        Returns:
        - The accuracy and loss on the test dataset.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Testing
        test_loss = 0
        correct = 0

        # Iterate through the test dataset
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.testdataloader):
                log_probs = self.model(data)
                # Sum up batch loss
                test_loss += torch.nn.functional.cross_entropy(log_probs, target, reduction='sum').item()

                # Get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

            # Calculate average test loss and accuracy
            test_loss /= len(self.testdataloader.dataset)
            accuracy = correct / len(self.testdataloader.dataset)

        return accuracy, test_loss

    
    def test_falldetection(self):
        """
        Evaluate the global model on a test dataset for fall detection.

        Returns:
        - Accuracy, F1 score, recall, precision, and epoch loss on the fall detection test dataset.
        """
        epoch_loss = 0
        correct, total = 0, 0
        targets = list()
        preds = list()
        acc = 0

        # Set the model to evaluation mode
        self.model.eval()

        for idx, (X_batch, y_batch) in enumerate(self.testdataloader):
            batchsize = X_batch.shape[0]

            # Forward pass
            out = self.model(X_batch.float())
            loss = self.loss_function(out.contiguous(), y_batch.contiguous())

            # Predictions and metrics
            pred = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_batch.size(0)
            correct += (pred == y_batch).sum().item()

            targets += list(y_batch)
            preds += list(pred)

            epoch_loss += (loss.detach().item() / batchsize)

        # Calculate accuracy and additional metrics
        acc = correct / total
        f1score = f1_score(targets, preds, zero_division=1)
        recall = recall_score(targets, preds, zero_division=1)
        precision = precision_score(targets, preds, zero_division=1)

        return acc, f1score, recall, precision, epoch_loss
