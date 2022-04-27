import torch
import numpy as np
import numpy.random as rd
import torch.nn as nn
import torch.nn.functional as F

import os, sys

"""
Deep Rewiring in Pytorch
Based on :
"Deep Rewiring: Training very sparse deep networks"
Guillaume Bellec, David Kappel, Wolfgang Maass, Robert Legenstein
ICLR 2018
(https://arxiv.org/abs/1711.05136)
https://github.com/guillaumeBellec/deep_rewiring
"""

class Sparse_Net(nn.Module):
    def __init__(self, dims, sparsity_list):
        super().__init__()
        self.thetas = torch.nn.ParameterList()
        self.weight = torch.nn.ParameterList()
        self.signs = torch.nn.ParameterList()
        self.sparsity_list = sparsity_list
        self.nb_non_zero_list = [int(d1*d2*p) for (d1, d2, p) in zip(dims[:-1], dims[1:], sparsity_list)]
        for d1, d2, nb_non_zero in zip(dims[:-1], dims[1:], self.nb_non_zero_list) :
            w, w_sign, th, _ = weight_sampler_strict_number(d1, d2, nb_non_zero)
            self.thetas.append(th)
            self.weight.append(w)
            self.signs.append(w_sign)
            assert (w == w_sign*th*(th>0)).all()
        
    @property
    def nb_active(self) : 
        return [(t>0).sum() for t in self.thetas]

    @property
    def grad_active(self) : 
        return [(t.grad!=0).sum() for t in self.thetas]

    @property
    def grad_and_con_active(self) : 
        return [((t>0)*(t.grad!=0)).sum() for t in self.thetas]
            
    def forward(self, x) : 
        if len(x.shape)>3 : 
            x = x.flatten(start_dim=1)
        for i, (th, sign) in enumerate(zip(self.thetas, self.signs)) : 
            w = th*sign*(th>0)
            x = torch.matmul(x, w)
            x = F.relu(x)
            if i == len(self.thetas) :
                x = F.log_softmax(x, dim=1)
        
        return x
    
class Sparse_Net_Soft(nn.Module):
    def __init__(self, dims, p0s, l1, noise, lr, clip_vals=None):
        super().__init__()
        self.thetas = torch.nn.ParameterList()
        self.weight = torch.nn.ParameterList()
        self.signs = torch.nn.ParameterList()
        self.th_clips = []

        for i, (d1, d2, p0) in enumerate(zip(dims[:-1], dims[1:], p0s)) :
            w, w_sign, th, _, th_clip = weight_sampler_with_clip(d1, d2, p0, l1, noise, lr, clip_vals[i])
            self.thetas.append(th)
            self.weight.append(w)
            self.signs.append(w_sign)
            th_clip = clip_vals[i] if clip_vals[i] is not None else th_clip
            self.th_clips.append(torch.tensor(th_clip))

            assert (w == w_sign*th*(th>0)).all()
            
         
    @property
    def nb_active(self) : 
        return [(t>0).sum() for t in self.thetas]

    def forward(self, x) : 
        if len(x.shape)>3 : 
            x = x.flatten(start_dim=1)
        for i, (th, sign) in enumerate(zip(self.thetas, self.signs)) : 
            w = th*sign*(th>0)
            x = torch.matmul(x, w)
            x = F.relu(x)
            if i == len(self.thetas) :
                x = F.log_softmax(x, dim=1)
        
        return x

class Sparse_Connect(nn.Module):
    """
    Sparse network to be trained with DeepR, used as connections in a global model
    Args : 
        dims : dimensions of the network
        sparsity_list : sparsities of the different layers
    """
    def __init__(self, dims, sparsity_list):
        super().__init__()
        self.thetas = torch.nn.ParameterList()
        self.weight = torch.nn.ParameterList()
        self.signs = torch.nn.ParameterList()
        self.sparsity_list = sparsity_list
        self.out_features = dims[-1]
        self.bias = None
        self.nb_non_zero_list = [int(d1*d2*p) for (d1, d2, p) in zip(dims[:-1], dims[1:], sparsity_list)]
        for d1, d2, nb_non_zero in zip(dims[:-1], dims[1:], self.nb_non_zero_list) :
            w, w_sign, th, _ = weight_sampler_strict_number(d1, d2, nb_non_zero)
            self.thetas.append(th)
            self.weight.append(w)
            self.signs.append(w_sign)
            assert (w == w_sign*th*(th>0)).all()

        self.is_sparse_connect = True
        self.is_deepR_connect = True
            
    def forward(self, x, relu=False) : 
        if type(x) is tuple : 
            x = x[0]
        if len(x.shape)>3 : 
            print(x.shape)
            #x = x.transpose(1, 2).flatten(start_dim=2)
        for i, (th, sign) in enumerate(zip(self.thetas, self.signs)) : 
        
            w = th*sign*(th>0)
            #x = F.linear(x, w)
            x = torch.matmul(x, w)
            if relu : 
                x = F.relu(x)
        return x
               
def weight_sampler_strict_number(n_in, n_out, nb_non_zero, dtype=torch.float32):
    '''
    Returns a weight matrix and its underlying, variables, and sign matrices needed for rewiring.
    Args : 
        n_in, n_out : number of input and output neurons
        nb_non_zero : number of active weight
    '''

    w_0 = rd.randn(n_in,n_out) / np.sqrt(n_in) # initial weight values

    # Gererate the random mask
    is_con_0 = np.zeros((n_in,n_out),dtype=bool)
    ind_in = rd.choice(np.arange(n_in),size=nb_non_zero)
    ind_out = rd.choice(np.arange(n_out),size=nb_non_zero)
    is_con_0[ind_in,ind_out] = True

    # Generate random signs
    sign_0 = np.sign(rd.randn(n_in,n_out))
    
    # Define the torch matrices
    """ 
    th = torch.nn.Parameter(torch.tensor(np.abs(w_0) * is_con_0, dtype=dtype).T)
    w_sign = torch.nn.Parameter(torch.tensor(sign_0, dtype=dtype).T, requires_grad=False)
    is_connected = (th>0).byte()
    w = torch.nn.Parameter(torch.where(is_connected, w_sign * th, torch.zeros((n_in, n_out), dtype=dtype).T), requires_grad=False)
    """
    
    th = torch.nn.Parameter(torch.tensor(np.abs(w_0) * is_con_0, dtype=dtype))
    w_sign = torch.nn.Parameter(torch.tensor(sign_0, dtype=dtype), requires_grad=False)
    is_connected = (th>0).bool()
    w = torch.nn.Parameter(torch.where(is_connected, w_sign * th, torch.zeros((n_in, n_out), dtype=dtype)), requires_grad=False)
   
    return w, w_sign, th, is_connected

def weight_sampler_with_clip(n_in, n_out, p0, l1, noise, lr, th_clip=None, dtype=torch.float32):

    # Sign of the weights
    sign_0 = np.sign(rd.randn(n_in,n_out))

    # Compute a rule of thumb to find a good clipping value
    assert(noise > 0)
    T = lr*noise**2 / (2)
    beta = l1 / T
    p_positive = p0
    p_negative = 1 - p_positive
    if th_clip is None : 
        th_clip = - p_negative / (p_positive * beta)
        #th_clip = -3*l1

    # initil variable
    th_0 = np.where(rd.rand(n_in, n_out) < p0, rd.rand(n_in, n_out) / np.sqrt(n_in),
                    rd.rand(n_in, n_out) * th_clip)

    # Define the variables
    th =  torch.nn.Parameter(torch.tensor(th_0, dtype=dtype))
    w_sign = torch.nn.Parameter(torch.tensor(sign_0, dtype=dtype), requires_grad=False)
    is_connected = (th>0).bool()
    w = torch.nn.Parameter(torch.where(is_connected, w_sign * th, torch.zeros((n_in, n_out), dtype=dtype)), requires_grad=False)

    return w, w_sign, th, is_connected, th_clip

def assert_connection_number(theta, targeted_number):
    '''
    Function to check during the simulation if the number of connection in well defined after each simulation.
    '''
    th = theta.data
    is_con = (th>0).int()
    nb_is_con = torch.sum(is_con).data.item()
    assert_is_con = np.abs(nb_is_con - targeted_number) <= 0
    return assert_is_con

def rewiring(theta, target_nb_connection, epsilon=1e-10):
    '''
    The rewiring operation to use after each iteration.
    Args : 
        theta: parameter to be updated
        target_nb_connection : number of connections needed to be active
    '''
        
    device = theta.device
    th = theta.data
    is_con = th>0

    n_connected = is_con.sum()
    nb_reconnect = target_nb_connection - n_connected
    nb_reconnect = torch.maximum(nb_reconnect, torch.tensor(0))
        
    reconnect_candidate_coord = torch.where(~is_con.flatten())[0].int()
    reconnect_sample_id = reconnect_candidate_coord.cpu().data.numpy()
    rd.shuffle(reconnect_sample_id)
    reconnect_sample_id = reconnect_sample_id[:nb_reconnect]

    # Apply the rewiring
    reconnect_vals = (torch.ones(size=[nb_reconnect])*epsilon).to(device)
    th.flatten()[reconnect_sample_id] = reconnect_vals

    assert assert_connection_number(theta, target_nb_connection), f'Target number of connection is {target_nb_connection} but theta has {(th>0).sum()}'
    
    return nb_reconnect.cpu().data.item()
    
def rewiring_global(theta, nb_reconnect, epsilon=1e-12):
    '''
    The rewiring operation to use after each iteration. This version allows for a global rewiring without a theta-specific target number of connections
    Args : 
        theta: parameter to be updated
        nb_reconnect : number of connections needed to be actived, 
    '''
    device = theta.device
    th = theta.data
    is_con = th>0
    
    nb_reconnect = torch.maximum(nb_reconnect, torch.tensor(0))
    reconnect_candidate_coord = torch.where(~is_con.flatten())[0].int()
    reconnect_sample_id = reconnect_candidate_coord.cpu().data.numpy()
    rd.shuffle(reconnect_sample_id)
    reconnect_sample_id = reconnect_sample_id[:nb_reconnect]
    
    # Apply the rewiring
    reconnect_vals = (torch.ones(size=[nb_reconnect])*epsilon).to(device)
    th.flatten()[reconnect_sample_id] = reconnect_vals
       
    return nb_reconnect.cpu().data.item()

def rewiring_soft(theta, th_clip):
    '''
    The rewiring operation to use after each iteration.
    Args : 
        theta: parameter to be updated
        target_nb_connection : number of connections needed to be active
    '''
    device = theta.device
    th = theta.data
    is_con = th>0
        
    reconnect_candidate_coord = torch.where(~is_con.flatten())[0].int()
    reconnect_sample_id = reconnect_candidate_coord.cpu().data.numpy()

    #clip_max = torch.vmap(lambda th : torch.maximum(th, th_clip).type(th.dtype))
    clip_max = lambda th : torch.max(th, (torch.ones_like(th)*th_clip).to(th.device))
    th.flatten()[reconnect_sample_id] = clip_max(th.flatten()[reconnect_sample_id])
    
def sample_matrix_specific_reconnection_number_for_global_fixed_connectivity(theta_list, ps):
    """
    Compute number of activation of connections needed in a global manner, to allow global rewiring
    Args : 
        theta_list : all parameters to be rewired
        ps : sparsity of connections list
    """
    theta_vals = [theta.data for theta in theta_list]
    nb_possible_connections_list = [torch.prod(torch.tensor(theta.shape), dtype=torch.float32)*p for theta, p in zip(theta_list, ps)]
    max_total_connections = torch.sum(torch.tensor(nb_possible_connections_list), dtype=torch.int32)
    sampling_probs = torch.tensor([nb_possible_connections / max_total_connections \
                          for nb_possible_connections in nb_possible_connections_list])
    
    def nb_connected(theta_val):
        return torch.sum((theta_val>0).int())
        
    total_connected = torch.sum(torch.tensor([nb_connected(theta) for theta in theta_vals]))
    nb_reconnect = torch.max(torch.tensor(0), max_total_connections-total_connected)
    if nb_reconnect>0 : 
        sample_split = torch.distributions.categorical.Categorical(probs=sampling_probs).sample(torch.Size([nb_reconnect.item()]))
        is_class_i_list = [(sample_split == i).int() for i in range(len(theta_list))]
        counts = [torch.sum(is_class_i, dtype=torch.int32) for is_class_i in is_class_i_list]
    else : 
        counts = [torch.tensor(0) for _ in theta_list]

    return counts
       
def apply_grad(theta, params, soft=False, cost_fn=None, tag='0-0'):
    """
    Update parameter's gradient and compute random-walk step as well as regularizing step.
    """
    device = theta.device
    lr = params['lr']
    l1 = params['l1']

    #is_con = lambda theta : (theta>0)
    is_con = theta > 0

    if theta.grad is not None :

        #theta.grad *= is_con

        gdnoise = params['gdnoise']
        noise_update = (torch.randn_like(theta)*gdnoise).to(device)
        if not soft : noise_update *= is_con
        theta.data += noise_update*lr
        theta.data -= l1*lr*is_con
        if cost_fn is not None : 
            theta.data -= lr*cost_fn(theta)*is_con*1e-4

        assert  not ((theta.grad !=0)*(~is_con)).any(), 'Dormant connection has gradient'

    elif theta.requires_grad : 
        #print(f'{tag} : None Grad \r')
        ""

    return is_con.sum()
     
def step_connections(model, optimizer_connections, global_rewire, thetas_list, sparsity_list, deepR_params_dict) : 
    """
    Training step for the sparse connections of a global model
    Args : 
        model : model comporting connections to be trained
        global_rewire : wheter to rewire in a global manner under a global fixed, or a per-parameter fixed connectivity
        theta_list : all parameters to be rewired
        sparsity_list : sparsity of connections list 
        deepR_params_dict : parameters of the DeepR algorithm
    """
    #Apply gradient for sparse connections
    for tag, connect in zip(model.connections.keys(), model.connections.values()) : 
        if type(connect) is Sparse_Connect : 
            for theta in connect.thetas : 
                apply_grad(theta, deepR_params_dict, tag=tag)
    
    optimizer_connections.step()
    #Rewire sparse connections
    nb_new_con = 0
    if not global_rewire : 
        for connect in model.connections.values() :    
            if type(connect) is Sparse_Connect : 
                for theta, nb_connect in zip(connect.thetas, connect.nb_non_zero_list) : 
                    if theta.requires_grad : 
                        ''
                        nb_new_con += rewiring(theta, nb_connect)
                    else : 
                        nb_new_con += 0

    else : 
        nb_reconnections = sample_matrix_specific_reconnection_number_for_global_fixed_connectivity(thetas_list, sparsity_list)
        for connect, nb_reconnect in zip(model.connections.values(), nb_reconnections) : 
            if type(connect) is Sparse_Connect : 
                if theta.requires_grad : 
                    nb_new_con += rewiring_global(connect.thetas[0], nb_reconnect)
                else : 
                    nb_new_con += 0
                
    #print(nb_new_con)
    return nb_new_con