
import sys
import os 
sys.path.append(os.getcwd())
import logging
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, RMSprop, AdamW
from pactl.logging import set_logging, wandb, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.train_utils import eval_model
from pactl.nn import create_model, create_intrinsic_model, FcNet3, ConvNet3
from pactl.bounds.get_bound_from_chk_v2 import my_quantize_vector, eval_acc_and_bound, compute_quantization, total_nll_bits, quantize, compute_quantization_all_tasks
from pactl.bounds.quantize_global_meta_param import quantize_global
from pactl.nn.projectors import CombinedRDKronFiLM
from experiments.compute_bound import compute_bound
import copy
from copy import deepcopy
from functools import partial


class MetaSAID(nn.Module):
  def __init__(self, train_loader,test_loader,intrinsic_dim,model_name=None, base_width=None, is_said = True,is_layered = None,
                   seed=None, intrinsic_mode=None, projector = None, global_levels=20, all_train_task_levels=20,
                   cfg_path=None, transfer=False, device_id=None, log_dir=None, levels = 20, use_kmeans = True, misc_extra_bits = 7, quant_epochs = 0, quant_lr = 0.01,
                   quantize_type = 'default', is_lambda_param=True, hyper_dim = None, is_last_layer_only=False):
        super().__init__()

        data_iter = iter(train_loader[0]['train'])
        inputs, labels = next(data_iter)  
        in_chans = inputs.shape[1]
        num_classes = len(set(labels.numpy())) if labels.ndim == 1 else labels.shape[1]

        self.net_args = {'model_name': model_name, 'num_classes': num_classes, 'in_chans' : in_chans, 'base_width': base_width,
                   'seed': seed, 'intrinsic_dim': intrinsic_dim, 'intrinsic_mode': intrinsic_mode,
                   'cfg_path' : cfg_path, 'transfer' : transfer,'device_id' : device_id, 'log_dir' : log_dir}

        quant_types = quantize_type.split("|")
        self.compression_args = {'levels': levels, 'global_levels': global_levels, 'use_kmeans': use_kmeans, 'misc_extra_bits': misc_extra_bits, 'epochs': quant_epochs, 'lr': quant_lr,
                   'global_quantize_type': quant_types[0], 'quantize_type': quant_types[1], 'all_train_task_levels': all_train_task_levels}
        

        self.intrinsic_net_args = {'is_said' : True, 'ckpt_path':None, 'intrinsic_mode':intrinsic_mode, 'intrinsic_dim':intrinsic_dim, 'seed':seed,
          'device':device_id, 'is_layered' : is_layered, 'is_said_meta' : True, 'is_lambda_param' : is_lambda_param, 'is_last_layer_only':is_last_layer_only}

        if model_name == 'FcNet3':
            self.base_net = FcNet3(input_shape=(1, 28, 28), output_dim=10)
        elif model_name == 'ConvNet3':
            self.base_net = ConvNet3(input_shape=(1, 28, 28), output_dim=10)
        else:
            self.base_net = create_model(**self.net_args)

        net = create_intrinsic_model(copy.deepcopy(self.base_net), **self.intrinsic_net_args)
        self.D = net.D
        self.said_size = net.said_size  if is_lambda_param else 0   

        hyper_dimension = intrinsic_dim if hyper_dim is None else hyper_dim
        self.d = hyper_dimension - self.said_size 
        self.global_param = nn.Parameter(torch.ones(self.d)/self.d)
        
        #TODO: add other projector types
        projector = partial(CombinedRDKronFiLM, seed=seed+1,is_layered = True)
        self.global_P = projector(self.D, self.d, self.global_param, 'global_param')
        #self.global_param.requires_grad = False  ##not train

        self.train_loader= train_loader
        self.test_loader = test_loader
        self.num_train_tasks = len(train_loader)
        self.num_test_tasks = len(test_loader)
        self.num_batches = len(self.train_loader[0]['train'])

        #TODO: num_classes=train_data.num_classes ? , in_chans=train_data[0][0].size(0) ?
        self.nets = [create_intrinsic_model(copy.deepcopy(self.base_net), **self.intrinsic_net_args, global_param = self.global_param, global_P = self.global_P).to(device_id) for i in range(self.num_train_tasks)]
        self.test_nets = None
  
  #assuming train loader is an array [num_tasks][]
  def meta_train(self, criterion, optimizer, lr, device_id, epochs, log_dir, weight_decay=5e-4):
    
    params = [self.global_param]

    for i in range(self.num_train_tasks):
      params += list(self.nets[i].parameters())
    
    
    #TODO: add other optimizers beside adam
    optim = Adam(params, lr=lr, weight_decay=weight_decay)
    

    for i in range(self.num_train_tasks):
        self.nets[i].train()
    
    for e in tqdm(range(epochs)):
      train_loader_iters = [iter(self.train_loader[t]['train']) for t in range(self.num_train_tasks)]
      for i in range(self.num_batches):
        sum_loss = 0
        optim.zero_grad()

        for t in range(self.num_train_tasks):
          (X, Y) = next(train_loader_iters[t])
          X, Y = X.to(device_id), Y.to(device_id)
          f_hat = self.nets[t](X)
          loss = criterion(f_hat, Y)
          sum_loss += loss

        sum_loss.backward()
        optim.step()
    
    
    sum_train_acc = 0
    sum_test_acc = 0
    for t in range(self.num_train_tasks):
      train_acc = self.meta_eval(self.nets[t], self.train_loader[t]['train'], criterion, device_id=device_id)['acc']
      sum_train_acc += train_acc
      test_acc = self.meta_eval(self.nets[t], self.train_loader[t]['test'], criterion, device_id=device_id)['acc']
      sum_test_acc  += test_acc
  
    
    
    print("average train accuracy on train data : ",sum_train_acc/self.num_train_tasks)
    print("average test accuracy on train data: ",sum_test_acc/self.num_train_tasks)

    return [sum_train_acc/self.num_train_tasks, sum_test_acc/self.num_train_tasks]



  def train_eval_test_tasks(self,lr,device_id,criterion,epochs, weight_decay=5e-4):
    #device,epochs,lr
    #for each task, build a model, train , eval
    
    quantized_global_vec, quantized_global_message_len = quantize_global(self.compression_args['global_quantize_type'], self, self.compression_args['global_levels'],self.net_args['device_id'], self.compression_args['epochs'], self.compression_args['lr'], self.compression_args['use_kmeans'])
    
    self.global_param = deepcopy(torch.tensor(quantized_global_vec,dtype=torch.float16).float().to(device_id))
    self.global_param.requires_grad = False
    print("global message_len: ", quantized_global_message_len)    

    sum_train_acc_quantized_train = 0
    sum_test_acc_quantized_train = 0
    sum_task_message_lens = 0

    if self.compression_args['quantize_type']=='default_all_tasks':
      sum_task_message_lens = self.compress_all_tasks()
    else:
      for t in range(self.num_train_tasks):
        ###quantization
        message_len = self.compress(self.compression_args['quantize_type'], self.nets[t], self.compression_args['levels'], self.net_args['device_id'], self.train_loader[t]['train'], self.compression_args['epochs'], self.compression_args['lr'], self.compression_args['use_kmeans'])  
        sum_task_message_lens+= message_len

    for t in range(self.num_train_tasks):
      train_acc_quantized = self.meta_eval(self.nets[t], self.train_loader[t]['train'], criterion, device_id=device_id)['acc']
      test_acc_quantized = self.meta_eval(self.nets[t], self.train_loader[t]['test'], criterion, device_id=device_id)['acc']

      sum_train_acc_quantized_train += train_acc_quantized
      sum_test_acc_quantized_train += test_acc_quantized

    print("average train accuracy on train data after quantization: ",sum_train_acc_quantized_train/self.num_train_tasks)
    print("average test accuracy on train data after quantization: ",sum_test_acc_quantized_train/self.num_train_tasks)
    samples_per_train = len(self.train_loader[0]['train'].dataset)
    global_part,task_spec_part = compute_bound(m = samples_per_train, n = self.num_train_tasks, global_message_len=quantized_global_message_len, sum_task_message_lens=sum_task_message_lens)
    print("sum_task_message_lens: ",sum_task_message_lens)
    print("global part bound: ",global_part)
    print("task specific part bound: ", task_spec_part)
    print("empirical error: ",1-(sum_train_acc_quantized_train/self.num_train_tasks))
    print("total bound: ", global_part+task_spec_part+1-(sum_train_acc_quantized_train/self.num_train_tasks))
    print("multi_task_bound: ",1-(sum_train_acc_quantized_train/self.num_train_tasks)+task_spec_part)

    '''
    sum_train_acc = 0
    sum_test_acc = 0
    sum_train_acc_quantized = 0
    sum_test_acc_quantized = 0

    self.test_nets = [create_intrinsic_model(copy.deepcopy(self.base_net), **self.intrinsic_net_args, global_param =  self.global_param, global_P = self.global_P).to(device_id) for i in range(self.num_test_tasks)]
    test_task_bounds = []

    for t in tqdm(range(self.num_test_tasks)):

      model = self.test_nets[t]   
      model.train()
      optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

      for e in range(epochs):
        for i, (X,Y) in enumerate(self.test_loader[t]['train']):
          X, Y = X.to(device_id), Y.to(device_id)
          optim.zero_grad()

          f_hat = model(X)
          loss = criterion(f_hat, Y)

          loss.backward()
          optim.step()

      
      train_acc = self.meta_eval(model, self.test_loader[t]['train'], criterion, device_id=device_id)['acc']
      test_acc = self.meta_eval(model, self.test_loader[t]['test'], criterion, device_id=device_id)['acc']

      sum_train_acc += train_acc
      sum_test_acc += test_acc
 
      ###quantization
      message_len = self.compress(self.compression_args['quantize_type'], model, self.compression_args['levels'], self.net_args['device_id'], self.test_loader[t]['train'], self.compression_args['epochs'], self.compression_args['lr'], self.compression_args['use_kmeans'])  

      train_acc_quantized = self.meta_eval(model, self.test_loader[t]['train'], criterion, device_id=device_id)['acc']
      test_acc_quantized = self.meta_eval(model, self.test_loader[t]['test'], criterion, device_id=device_id)['acc']

      num_samples = len(self.test_loader[t]['train'].dataset)
      glob_part, task_part = compute_bound(m=num_samples, n=1, global_message_len = quantized_global_message_len, sum_task_message_lens=message_len)
      test_task_bounds.append(glob_part+task_part+1-train_acc_quantized)

      sum_train_acc_quantized += train_acc_quantized
      sum_test_acc_quantized += test_acc_quantized

      
    print("average train accuracy on test data: ",sum_train_acc/self.num_test_tasks)
    print("average test accuracy on test data: ",sum_test_acc/self.num_test_tasks)

    print("average train accuracy on test data after quantization: ",sum_train_acc_quantized/self.num_test_tasks)
    print("average test accuracy on test data after quantization: ",sum_test_acc_quantized/self.num_test_tasks)
    '''
    return 

  @torch.no_grad()
  def meta_eval(self,model, data_loader, criterion=None, device_id=None):
    model.eval()
    N = len(data_loader.dataset)
    N_acc = 0
    
    for i, (X, Y) in enumerate(data_loader):
      X, Y = X.to(device_id), Y.to(device_id)
      logits = model(X)
      N_acc += (logits.argmax(dim=-1) == Y).sum()

    metrics = {'acc': N_acc.item() / N}

    return metrics
    
  def compress(self, quant_type, net, levels, device, train_loader, epochs, lr, use_kmeans):
    # subspace-params_lambdas_global-lambdas
    if quant_type == 'default_all_tasks':
      quant_type = 'default_float8_float8' if net.is_lambda_param else 'default'

    if not net.is_lambda_param:
      quantized_vecs, message_len = quantize(quant_type, net, ['subspace_params'], levels, device, train_loader, epochs, lr, use_kmeans)
      net.subspace_params.data = torch.tensor(quantized_vecs[0],dtype=torch.float16).float().to(device)  
      return message_len

    types = quant_type.split("_")
    message_len = 0
  
    quantized_vecs, len = quantize(types[0], net, ['subspace_params'], levels, device, train_loader, epochs, lr, use_kmeans)
    net.subspace_params.data = torch.tensor(quantized_vecs[0],dtype=torch.float16).float().to(device)  
    message_len += len  
    
    if types[1]=='together':
      quantized_vecs, len = quantize(types[2], net, ['lambdas', 'global_lambdas'], levels, device, train_loader, epochs, lr, use_kmeans)
      net.lambdas.data = torch.tensor(quantized_vecs[0],dtype=torch.float16).float().to(device)
      net.global_lambdas.data = torch.tensor(quantized_vecs[1],dtype=torch.float16).float().to(device) 
      message_len += len
    else:
      quantized_vecs, len = quantize(types[1], net, ['lambdas'], levels, device, train_loader, epochs, lr, use_kmeans)
      net.lambdas.data = torch.tensor(quantized_vecs[0],dtype=torch.float16).float().to(device) 
      message_len += len
      quantized_vecs, len = quantize(types[2], net, ['global_lambdas'], levels, device, train_loader, epochs, lr, use_kmeans)
      net.global_lambdas.data = torch.tensor(quantized_vecs[0],dtype=torch.float16).float().to(device) 
      message_len += len

    return message_len

  def compress_all_tasks(self):
    #if is_lambda_param: lambda_global,lambda_local,subspace_params else: SUBSPACE_PARAMS
    sum_task_message_len = 0
    #subspace_params
    quantized_vecs, message_len = compute_quantization_all_tasks(self, 'subspace_params', self.compression_args['all_train_task_levels'], self.net_args['device_id'], self.compression_args['epochs'], self.compression_args['lr'], self.compression_args['use_kmeans'])
    sum_task_message_len+= message_len

    for i in range(self.num_train_tasks):
      if quantized_vecs[i] is not None:
        self.nets[i].subspace_params.data = torch.tensor(quantized_vecs[i],dtype=torch.float16).float().to(self.net_args['device_id'])
      else:
        print("error: quantized value is none!")

    
    if self.intrinsic_net_args['is_lambda_param']:
      #lambdas
      quantized_vecs, message_len = compute_quantization_all_tasks(self, 'lambdas', self.compression_args['levels'], self.net_args['device_id'], self.compression_args['epochs'], self.compression_args['lr'], self.compression_args['use_kmeans'])
      sum_task_message_len+= message_len

      for i in range(self.num_train_tasks):
        if quantized_vecs[i] is not None:
          self.nets[i].lambdas.data = torch.tensor(quantized_vecs[i],dtype=torch.float16).float().to(self.net_args['device_id'])
        else:
          print("error: quantized value is none!")

      #global_lambdas
      quantized_vecs, message_len = compute_quantization_all_tasks(self, 'global_lambdas', self.compression_args['levels'], self.net_args['device_id'], self.compression_args['epochs'], self.compression_args['lr'], self.compression_args['use_kmeans'])
      sum_task_message_len+= message_len

      for i in range(self.num_train_tasks):
        if quantized_vecs[i] is not None:
          self.nets[i].global_lambdas.data = torch.tensor(quantized_vecs[i],dtype=torch.float16).float().to(self.net_args['device_id'])
        else:
          print("error: quantized value is none!")
    

    return sum_task_message_len
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  
  
  
  
  
  
  

  
  
  
  
  
  
  
                                                   
  
    