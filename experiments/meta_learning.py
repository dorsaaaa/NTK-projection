
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
from pactl.nn import create_model, create_intrinsic_model, FcNet3
from pactl.bounds.get_bound_from_chk_v2 import my_quantize_vector, eval_acc_and_bound, compute_quantization, total_nll_bits
from pactl.bounds.quantize_global_meta_param import compute_meta_quantization
import copy
from copy import deepcopy


class Meta(nn.Module):
  def __init__(self, train_loader,test_loader,intrinsic_dim,model_name=None, num_classes=None, in_chans=None, base_width=None, is_said = None,is_layered = None,
                   seed=None, intrinsic_mode=None,
                   cfg_path=None, transfer=False, device_id=None, log_dir=None, levels = 20, use_kmeans = True, misc_extra_bits = 7, quant_epochs = 0, quant_lr = 0.01,
                   quantize_type = 'default'):
        super().__init__()
        #TODO: num classes, inchans
        self.net_args = {'model_name': model_name, 'num_classes': 10, 'in_chans' : 1, 'base_width': base_width,
                   'seed': seed, 'intrinsic_dim': intrinsic_dim, 'intrinsic_mode': intrinsic_mode,
                   'cfg_path' : cfg_path, 'transfer' : transfer,'device_id' : device_id, 'log_dir' : log_dir}

        self.compression_args = {'levels': levels, 'use_kmeans': use_kmeans, 'misc_extra_bits': misc_extra_bits, 'epochs': quant_epochs, 'lr': quant_lr,
                   'quantize_type': quantize_type}
        

        self.intrinsic_net_args = {'is_said' : is_said, 'ckpt_path':None, 'intrinsic_mode':intrinsic_mode, 'intrinsic_dim':intrinsic_dim, 'seed':seed,
          'device':device_id, 'is_layered' : is_layered}

        if model_name == 'FcNet3':
            self.base_net = FcNet3(input_shape=(1, 28, 28), output_dim=10)
        else:
            self.base_net = create_model(**self.net_args)

        if is_said:
          net = create_intrinsic_model(copy.deepcopy(self.base_net), **self.intrinsic_net_args, global_param = None)
          intrinsic_dim -= net.said_size

        self.d = intrinsic_dim
        self.global_param = nn.Parameter(torch.zeros(self.d))

        self.train_loader= train_loader
        self.test_loader = test_loader
        self.num_train_tasks = len(train_loader)
        self.num_test_tasks = len(test_loader)
        self.num_batches = len(self.train_loader[0]['train'])

        #TODO: num_classes=train_data.num_classes ? , in_chans=train_data[0][0].size(0) ?
        self.nets = [create_intrinsic_model(copy.deepcopy(self.base_net), **self.intrinsic_net_args, global_param = self.global_param).to(device_id) for i in range(self.num_train_tasks)]
        
  
  #assuming train loader is an array [num_tasks][]
  def meta_train(self, criterion, optimizer, lr, device_id, epochs, log_dir):
    
    params = [self.global_param]

    for i in range(self.num_train_tasks):
      params += list(self.nets[i].parameters())
    
    
    #TODO: add other optimizers beside adam
    optim = Adam(params, lr=lr)
    

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
      print("train_acc,test_acc",train_acc,test_acc)
    
    
    
    #print(f"after evaling global model: {self.global_param}")
    print("average train accuracy on train data : ",sum_train_acc/self.num_train_tasks)
    print("average test accuracy on train data: ",sum_test_acc/self.num_train_tasks)

    return [sum_train_acc/self.num_train_tasks, sum_test_acc/self.num_train_tasks]



  def train_eval_test_tasks(self,lr,device_id,criterion,epochs):
    #device,epochs,lr
    self.global_param.requires_grad = False
    quantized_global_vec, quantized_global_message_len = compute_meta_quantization(self, self.compression_args['levels'],self.net_args['device_id'], self.compression_args['epochs'], self.compression_args['lr'], self.compression_args['use_kmeans'])
    
    self.global_param.data = deepcopy(torch.tensor(quantized_global_vec).float().to(device_id))
    

    sum_train_acc_quantized_train = 0
    sum_test_acc_quantized_train = 0
    for t in range(self.num_train_tasks):
      ###quantization
      quantized_vec, message_len = compute_quantization(self.nets[t], self.compression_args['levels'], device_id, self.train_loader[t]['train'],
           self.compression_args['epochs'], self.compression_args['lr'], self.compression_args['use_kmeans'],is_seperate = False)  

      if quantized_vec is not None:
        self.nets[t].subspace_params.data = torch.tensor(quantized_vec).float().to(device_id)    

      train_acc_quantized = self.meta_eval(self.nets[t], self.train_loader[t]['train'], criterion, device_id=device_id)['acc']
      test_acc_quantized = self.meta_eval(self.nets[t], self.train_loader[t]['test'], criterion, device_id=device_id)['acc']
      #print(train_acc_quantized,test_acc_quantized)

      sum_train_acc_quantized_train += train_acc_quantized
      sum_test_acc_quantized_train += test_acc_quantized

    print("average train accuracy on train data after quantization: ",sum_train_acc_quantized_train/self.num_train_tasks)
    print("average test accuracy on train data after quantization: ",sum_test_acc_quantized_train/self.num_train_tasks)

    sum_train_acc = 0
    sum_test_acc = 0
    sum_train_acc_quantized = 0
    sum_test_acc_quantized = 0

    nets = [create_intrinsic_model(copy.deepcopy(self.base_net), **self.intrinsic_net_args, global_param =  self.global_param).to(device_id) for i in range(self.num_test_tasks)]


    for t in tqdm(range(self.num_test_tasks)):

      model = nets[t]   
      model.train()
      optim = Adam(model.parameters(), lr=lr)

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
      quantized_vec, message_len = compute_quantization(model, self.compression_args['levels'], device_id, self.test_loader[t]['train'],
           self.compression_args['epochs'], self.compression_args['lr'], self.compression_args['use_kmeans'],is_seperate = False)  

      '''
      if self.net_args['is_said']:
        said_quantized_vec, said_message_len = my_quantize_vector(model.subspace_params[model.d:].cpu().data.numpy(),'float8') 
        quantized_vec = np.append(quantized_vec,said_quantized_vec)
      '''

      if quantized_vec is not None:
        model.subspace_params.data = torch.tensor(quantized_vec).float().to(device_id)    

      train_acc_quantized = self.meta_eval(model, self.test_loader[t]['train'], criterion, device_id=device_id)['acc']
      test_acc_quantized = self.meta_eval(model, self.test_loader[t]['test'], criterion, device_id=device_id)['acc']

      sum_train_acc_quantized += train_acc_quantized
      sum_test_acc_quantized += test_acc_quantized
      
    print("average train accuracy on test data: ",sum_train_acc/self.num_test_tasks)
    print("average test accuracy on test data: ",sum_test_acc/self.num_test_tasks)

    print("average train accuracy on test data after quantization: ",sum_train_acc_quantized/self.num_test_tasks)
    print("average test accuracy on test data after quantization: ",sum_test_acc_quantized/self.num_test_tasks)

    return [sum_train_acc_quantized_train/self.num_train_tasks, sum_test_acc_quantized_train/self.num_train_tasks, sum_train_acc/self.num_test_tasks, sum_test_acc/self.num_test_tasks , sum_train_acc_quantized/self.num_test_tasks, sum_test_acc_quantized/self.num_test_tasks]

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
    

  def compress_model(self, model, train_loader, test_loader, train_acc, test_acc, quantized_global_vec, global_message_len, levels, misc_extra_bits, epochs, device, lr, use_kmeans):

    is_seperate = True
    '''
    if is_said and said_quantize_type == 'default':
        is_seperate = False
    '''
    quantized_vec, message_len = compute_quantization(model, levels, device, train_loader, epochs,lr, use_kmeans,is_seperate = is_seperate)
    '''
    if is_said and said_quantize_type != 'default':
        said_quantized_vec, said_message_len = my_quantize_vector(model.subspace_params[model.d:].cpu().data.numpy(),said_quantize_type) 
        message_len += said_message_len
        quantized_vec = np.append(quantized_vec,said_quantized_vec)
    '''
    
    try:
        if quantized_vec is not None and quantized_global_vec is not None:
            model.subspace_params.data = torch.tensor(quantized_vec).float().to(device)   
            model.global_param.data = torch.tensor(quantized_global_vec).float().to(device)        
        else:
            aux = torch.zeros_like(model.subspace_params.data).float().to(device)
            model.subspace_params.data = aux
    except AttributeError:
        logging.warning("Quantization vector was not updated.")
    
    
    # self delimiting message takes up additional 2 log(l) bits
    message_len += global_message_len
    prefix_message_len = message_len + 2 * np.log2(message_len) if message_len > 0 else 0
    train_nll_bits = total_nll_bits(model, train_loader, device=device, distributed=False)

    raw_output = eval_acc_and_bound(model=model, trainloader=train_loader, testloader=test_loader,
                                    prefix_message_len=prefix_message_len, device=device,
                                    misc_extra_bits=misc_extra_bits, quantized_vec=quantized_vec,
                                    posterior_scale=0.01, use_robust_adj=False,
                                    log_dir=self.net_args['log_dir'], distributed=False)
    raw_output = {f'raw_{name}': value for name, value in raw_output.items()}
    

    return {
        # **output,
        **raw_output,
        'train_nll_bits': train_nll_bits,
        'prefix_message_len': prefix_message_len,
        'train_err_100': (1. - train_acc) * 100,
        'test_err_100': (1 - test_acc) * 100
    }
    
  def train_eval(self,lr,device_id,criterion,epochs,loader):
    #for each task, build a model, train , eval
    self.global_param.requires_grad = False

    sum_train_acc= 0
    sum_test_acc = 0

    nets = [create_intrinsic_model(copy.deepcopy(self.base_net), **self.intrinsic_net_args, global_param = self.global_param).to(device_id) for i in range(len(loader))]

    for t in tqdm(range(len(loader))):
      model = self.nets[t]
      model.train()
      optim = Adam(model.parameters(), lr=lr)
      for e in range(epochs):
        for i, (X,Y) in enumerate(loader[t]['train']):
          X, Y = X.to(device_id), Y.to(device_id)
          optim.zero_grad()

          f_hat = model(X)
          loss = criterion(f_hat, Y)

          loss.backward()
          optim.step()

      #print(f'subspace params: {model.subspace_params[:self.d]}')
      #print(f'landas: {model.subspace_params[self.d:]}')

      train_acc = self.meta_eval(model, loader[t]['train'], criterion, device_id=device_id)['acc']
      test_acc = self.meta_eval(model, loader[t]['test'], criterion, device_id=device_id)['acc']

      sum_train_acc += train_acc
      sum_test_acc += test_acc

      
    print("average train accuracy on data: ",sum_train_acc/len(loader))
    print("average test accuracy on data: ",sum_test_acc/len(loader))

    return
    
