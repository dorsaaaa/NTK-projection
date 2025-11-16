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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from experiments.compute_bound import compute_bound
import numpy as np
from copy import deepcopy


from pactl.distributed import maybe_launch_distributed
from pactl.logging import set_logging, wandb, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.train_utils import eval_model
from pactl.nn import create_model
from pactl.optim.third_party.functional_warm_up import LinearWarmupScheduler
from pactl.optim.schedulers import construct_stable_cosine
from pactl.optim.schedulers import construct_warm_stable_cosine
from experiments.meta_learning import Meta
from data.Data_gen import data_gen


from experiments.MetaSAID import MetaSAID


def save_params(meta_learner, filename):
  dict = {'global_lambdas': meta_learner.global_param[meta_learner.d:]}
  for t in range(meta_learner.num_train_tasks):
    dict['lambdas_train_task_'+str(t)] = meta_learner.nets[t].subspace_params[meta_learner.nets[t].d:]

  for t in range(meta_learner.num_test_tasks):
    dict['lambdas_test_task_'+str(t)] = meta_learner.test_nets[t].subspace_params[meta_learner.test_nets[t].d:]


  torch.save(dict, filename)
  print(f"parameters saved to {filename}")


def main(seed=137, device_id=0, distributed=False, data_dir=None, log_dir=None,
        train_subset=1, indices_path=None, label_noise=0, num_workers=2,
         cfg_path=None, transfer=False, model_name='FcNet3', base_width=8, global_levels=20,
         batch_size=64, optimizer='adam', lr=1e-2, momentum=.9, weight_decay=5e-4, epochs=2, is_lambda_param=True,
         intrinsic_dim=600, intrinsic_mode='filmrdkron',is_said = False,is_layered = False, warmup_epochs=0, warmup_lr=.1, all_train_task_levels=20,
         levels = 10, use_kmeans = True, misc_extra_bits = 7, quant_epochs = 30, quant_lr = 0.0001, quantize_type = 'default', hyper_dim=None, is_last_layer_only=False,
         data_seed=42, data_transform = 'Shuffled_Pixels', n_train_tasks=10, n_test_tasks = 20, samples_per_train=600, samples_per_test=600):

  train_loader, test_loader = data_gen(seed = data_seed, data_transform = data_transform, n_train_tasks=n_train_tasks, n_test_tasks = n_test_tasks, samples_per_train=samples_per_train, samples_per_test=samples_per_test)
  random_seed_all(seed)
  
  '''
  for batch_idx, (data, labels) in enumerate(train_loader[0]['train']):
    print(f"Batch {batch_idx} Labels: {labels}")
    # Print only the first batch of labels for readability
    break
  '''
  #print("loaded data successfully")
  
  
  criterion = nn.CrossEntropyLoss()
  '''
  if optimizer == 'sgd':
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)
  elif optimizer == 'ssc':
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optim_scheduler = construct_stable_cosine(
            optimizer=optimizer, lr_max=lr, lr_min=lr/100., epochs=(100, epochs - 100))
  elif optimizer == 'wsc':
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optim_scheduler = construct_warm_stable_cosine(
            optimizer=optimizer, lrs=(lr/100., lr, lr/10.),
            epochs=(5, 75, epochs - 80))
  elif optimizer == 'awsc':
    optimizer = Adam(net.parameters(), lr=lr)
    optim_scheduler = construct_warm_stable_cosine(
            optimizer=optimizer, lrs=(lr/100., lr, lr/10.),
            epochs=(5, 75, epochs - 80))
  elif optimizer == 'adam':
    optimizer = Adam(net.parameters(), lr=lr)
    optim_scheduler = None
  elif optimizer == 'rmsprop':
    optimizer = RMSprop(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
  elif optimizer == 'adamw':
    optimizer = AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100.)
  elif optimizer == 'sgd_cos':
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100.)
  elif optimizer == 'sgd_cos10':
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10.)
  elif optimizer == 'sgd_only':
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optim_scheduler = None
  else:
    raise NotImplementedError

  if warmup_epochs > 0:
    optim_scheduler = LinearWarmupScheduler(optimizer,
      warm_epochs=[warmup_epochs], lr_goal=[warmup_lr], scheduler_after=[optim_scheduler])
  '''


  '''
  #meta learning part
  meta_learner = Meta(train_loader, test_loader, intrinsic_dim, model_name=model_name, base_width=base_width, is_said = is_said,is_layered = is_layered,
                   seed=seed, intrinsic_mode=intrinsic_mode,cfg_path=cfg_path, transfer=transfer, device_id=device_id, log_dir=log_dir,
                   levels = levels, use_kmeans = use_kmeans, misc_extra_bits = misc_extra_bits, quant_epochs = quant_epochs, quant_lr = quant_lr,
                   quantize_type = quantize_type)
  '''
  
  
  meta_learner = MetaSAID(train_loader, test_loader, intrinsic_dim, model_name=model_name, base_width=base_width, is_said = is_said,is_layered = is_layered, all_train_task_levels= all_train_task_levels,
                 seed=seed, intrinsic_mode=intrinsic_mode,cfg_path=cfg_path, transfer=transfer, device_id=device_id, log_dir=log_dir, hyper_dim=hyper_dim, is_last_layer_only = is_last_layer_only,
                 levels = levels, use_kmeans = use_kmeans, misc_extra_bits = misc_extra_bits, quant_epochs = quant_epochs, quant_lr = quant_lr,
                 quantize_type = quantize_type, is_lambda_param = is_lambda_param, global_levels=global_levels)
  
  
  meta_learner.meta_train(criterion, optimizer, lr, device_id, epochs, log_dir, weight_decay=weight_decay)

  meta_learner.train_eval_test_tasks(lr,device_id,criterion,epochs, weight_decay=weight_decay)
  
  '''
  #save global _P
  filename = 'experiments/META_SAID_global_params_quant.txt'
  param = meta_learner.global_param.detach().cpu().numpy()
  np.savetxt(filename, param, fmt='%.6f')  
  print(f"Parameter saved to {filename}")
  '''


  #check if the implementation was right
  #print("train loader")
  #meta_learner.train_eval(lr,device_id,criterion,epochs,train_loader)

  #print("test loader")
  #meta_learner.train_eval(lr,device_id,criterion,epochs,test_loader)

  return 
  
  
  


def entrypoint(log_dir=None, **kwargs):
  world_size, rank, device_id = maybe_launch_distributed()

  torch.backends.cudnn.benchmark = True
  torch.cuda.set_device(device_id)

  ## Only setup logging from one process (rank = 0).
  log_dir = set_logging(log_dir=log_dir) if rank == 0 else None
  if rank == 0:
    logging.info(f'Working with {world_size} process(es).')

  results = main(**kwargs, log_dir=log_dir, distributed=(world_size > 1), device_id=device_id)

  if rank == 0:
    finish_logging()


if __name__ == '__main__':
  import fire
  fire.Fire(entrypoint)
