import sys
import os 
sys.path.append(os.getcwd())
import logging
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, RMSprop, AdamW
from pactl.distributed import maybe_launch_distributed
import numpy as np
from copy import deepcopy

from experiments.compute_bound import compute_bound
from pactl.logging import set_logging, wandb, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
import dill
from joblib import dump, load

from ELLA_SAID.meta_learner import ELLA_SAID
from data.Data_gen import data_gen
import json

def save_params(meta_learner, filename):
  dict = {'global_lambdas': meta_learner.global_lambdas}
  for t in range(meta_learner.num_train_tasks):
    dict['lambdas_train_task_'+str(t)] = meta_learner.nets[t].subspace_params[meta_learner.nets[t].d:]
    dict['alphas_train_task_'+str(t)] = meta_learner.nets[t].alphas

  for t in range(meta_learner.num_test_tasks):
    dict['lambdas_test_task_'+str(t)] = meta_learner.test_nets[t].subspace_params[meta_learner.test_nets[t].d:]
    dict['alphas_test_task_'+str(t)] = meta_learner.test_nets[t].alphas


  torch.save(dict, filename)
  print(f"parameters saved to {filename}")




def main(seed=137, device_id=0, distributed=False, data_dir=None, log_dir=None,
        train_subset=1, indices_path=None, label_noise=0, num_workers=2,
         cfg_path=None, transfer=False, model_name='FcNet3', base_width=8,
         batch_size=64, optimizer='adam', lr=1e-2, momentum=.9, weight_decay=5e-4, epochs=2, global_levels=20, lambdas_levels=3, is_lambda_global=False,
         intrinsic_dim=600, intrinsic_mode='filmrdkron',is_said = False,is_layered = False, warmup_epochs=0, warmup_lr=.1, all_train_task_levels=20,
         levels = 5, use_kmeans = True, misc_extra_bits = 7, quant_epochs = 30, quant_lr = 0.0001, quantize_type = 'default',  is_lambda_param=True, is_last_layer_only=False,
         basis_num = 5, data_seed=42, hyper_dim = None, data_transform = 'Shuffled_Pixels',n_train_tasks=10, n_test_tasks=20, samples_per_train=600, samples_per_test=600,
         dataset='MNIST', N_Way=40):

  random_seed_all(seed)
  train_loader, test_loader = data_gen(seed=data_seed, dataset=dataset, num_classes=N_Way, data_transform = data_transform, n_train_tasks=n_train_tasks, n_test_tasks = n_test_tasks, samples_per_train=samples_per_train, samples_per_test=samples_per_test)
  


  print("loaded data successfully")
  
  
  criterion = nn.CrossEntropyLoss()
  

  #meta learning part
  meta_learner = ELLA_SAID(train_loader, test_loader, intrinsic_dim, basis_num = basis_num, model_name=model_name, base_width=base_width,
                   seed=seed, intrinsic_mode=intrinsic_mode,cfg_path=cfg_path, transfer=transfer, device_id=device_id, log_dir=log_dir, is_lambda_global=is_lambda_global, lambdas_levels=lambdas_levels,
                   levels = levels, use_kmeans = use_kmeans, misc_extra_bits = misc_extra_bits, quant_epochs = quant_epochs, quant_lr = quant_lr, all_train_task_levels=all_train_task_levels,
                   quantize_type = quantize_type, hyper_dim = hyper_dim, global_levels=global_levels, is_lambda_param = is_lambda_param, is_last_layer_only=is_last_layer_only)
  
  
  meta_learner.meta_train(criterion, optimizer, lr, device_id, epochs, log_dir, weight_decay=weight_decay)

  
  '''
  #save model
  filename = 'ELLA_SAID/global_params/meta_learner'+str(global_levels)+'.txt'
  dict={}
  param = meta_learner.global_param.detach().cpu().numpy()
  dict['global_param']= param
  dict['global_lambdas']= meta_learner.global_lambdas.detach().cpu().numpy()
  dict['task_lambdas']= meta_learner.task_lambdas.detach().cpu().numpy()

  for i in range(meta_learner.num_train_tasks):
    dict['local_param_'+str(i)]= meta_learner.nets[i].subspace_params.detach().cpu().numpy()
    dict['local_alphas_'+str(i)]= meta_learner.nets[i].alphas.detach().cpu().numpy()
    dict['local_lambdas_'+str(i)]= meta_learner.nets[i].lambdas.detach().cpu().numpy()
    dict['local_global_lambdas_'+str(i)]= meta_learner.nets[i].global_lambdas.detach().cpu().numpy()
  text_dict = {key: value.tolist() for key, value in dict.items()}
  # Save the dictionary as a JSON in a .txt file
  with open(filename, 'w') as f:
      json.dump(text_dict, f, indent=4)
  print(f"Parameters loaded and saved to {filename}")
  '''

  meta_learner.train_eval_test_tasks(lr,device_id,criterion,epochs, weight_decay=weight_decay)
  '''
  #save model
  filename = 'ELLA_SAID/global_params/meta_learner_after_quant'+str(global_levels)+'.txt'
  dict={}
  param = meta_learner.global_param.detach().cpu().numpy()
  dict['global_param']= param
  dict['global_lambdas']= meta_learner.global_lambdas.detach().cpu().numpy()
  dict['task_lambdas']= meta_learner.task_lambdas.detach().cpu().numpy()
  for i in range(meta_learner.num_train_tasks):
    dict['local_param_'+str(i)]= meta_learner.nets[i].subspace_params.detach().cpu().numpy()
    dict['local_alphas_'+str(i)]= meta_learner.nets[i].alphas.detach().cpu().numpy()
    dict['local_lambdas_'+str(i)]= meta_learner.nets[i].lambdas.detach().cpu().numpy()
    dict['local_global_lambdas_'+str(i)]= meta_learner.nets[i].global_lambdas.detach().cpu().numpy()
  text_dict = {key: value.tolist() for key, value in dict.items()}
  # Save the dictionary as a JSON in a .txt file
  with open(filename, 'w') as f:
      json.dump(text_dict, f, indent=4)
  print(f"Parameters saved to {filename}") 
  '''


  

  return 
  
  
  


def entrypoint(log_dir=None, **kwargs):
  world_size, rank, device_id = maybe_launch_distributed()
  
  torch.backends.cudnn.benchmark = True
  torch.cuda.set_device(device_id)

  ## Only setup logging from one process (rank = 0).
  log_dir = set_logging(log_dir=log_dir) if rank == 0 else None
  if rank == 0:
    logging.info(f'Working with {world_size} process(es).')

  results = main(**kwargs, log_dir=log_dir, distributed=False, device_id=device_id)

  if rank == 0:
    finish_logging()


if __name__ == '__main__':
  import fire
  fire.Fire(entrypoint)
