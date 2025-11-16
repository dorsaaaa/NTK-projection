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
import matplotlib.pyplot as plt


from pactl.logging import set_logging, wandb, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from experiments.compute_bound import compute_bound

from ELLA_Subspace.meta_learner import Meta_ELLA
from data.Data_gen import data_gen
import numpy as np
import json

from data.dataset import gen_random_loaders
from data.Data_Path import get_data_path
from data.folktables_dataset import get_folktables_dataset
from data.products_dataset import get_products_dataset


def main(seed=137, device_id=0, distributed=False, data_dir=None, log_dir=None,
        train_subset=1, indices_path=None, label_noise=0, num_workers=2,
         cfg_path=None, transfer=False, model_name='FcNet3', base_width=8, K_Shot_MetaTrain=15, K_Shot_MetaTest=15,
         batch_size=64, optimizer='adam', lr=1e-2, momentum=.9, weight_decay=5e-4, epochs=2, global_levels=20, all_train_task_levels = 20, Transfer=False,
         intrinsic_dim=600, intrinsic_mode='filmrdkron',is_said = False,is_layered = False, warmup_epochs=0, warmup_lr=.1, classes_per_user=6, do_mapping=True,
         levels = 5, use_kmeans = True, misc_extra_bits = 15, quant_epochs = 30, quant_lr = 0.0001, quantize_type = 'default',  dataset='MNIST', N_Way=10,use_aid=False,
         basis_num = 5, data_seed=42, data_transform = 'Shuffled_Pixels',n_train_tasks=20, n_test_tasks=0, samples_per_train=600, samples_per_test=600):

  
  if dataset == 'MNIST':
    train_loader, test_loader = data_gen(seed=data_seed, dataset=dataset, data_transform = data_transform, n_train_tasks=n_train_tasks, n_test_tasks = n_test_tasks, samples_per_train=samples_per_train, samples_per_test=samples_per_test)
  elif dataset == 'CIFAR10':
    misc_extra_bits += 20
    train_loader, test_loader = gen_random_loaders(data_name='cifar10', data_path=get_data_path(), num_users=n_train_tasks+n_test_tasks, num_train_users=n_train_tasks, bz=128, partition_type='by_class', classes_per_user=classes_per_user, seed=data_seed, do_mapping=do_mapping)
  elif dataset == 'CIFAR100':
    misc_extra_bits += 20
    train_loader, test_loader = gen_random_loaders(data_name='cifar100', data_path=get_data_path(), num_users=n_train_tasks + n_test_tasks, num_train_users=n_train_tasks, bz=128, partition_type='by_class', classes_per_user=classes_per_user, seed=data_seed, do_mapping=do_mapping)
  elif dataset =='folktables':
    train_loader, test_loader = get_folktables_dataset(num_train_tasks=n_train_tasks, num_test_tasks=n_test_tasks, num_samples=samples_per_train, seed=data_seed)
  elif dataset =='products':
    train_loader, test_loader = get_products_dataset(num_train_tasks=n_train_tasks, num_test_tasks=n_test_tasks, num_samples=samples_per_train, seed=data_seed)   
  
  
  random_seed_all(seed)

  print("loaded data successfully")
  '''
  data_iter = iter(test_loader[0]['test'])
  images, labels = next(data_iter)
  
  fig, axes = plt.subplots(1, 20, figsize=(15, 2))
  for i in range(20):  
      ax = axes[i]
      image = images[i].squeeze().numpy() 
      ax.imshow(image, cmap='gray')
      ax.set_title(f"{labels[i].item()}")
      ax.axis('off')

  plt.savefig('Permute_Labels2.png', format='png')
  print("saved pic")
  '''
  criterion = nn.CrossEntropyLoss()
  
  #meta learning part
  meta_learner = Meta_ELLA(train_loader, test_loader, intrinsic_dim, basis_num = basis_num, model_name=model_name, base_width=base_width, Transfer=Transfer,
                   seed=seed, intrinsic_mode=intrinsic_mode,cfg_path=cfg_path, transfer=transfer, device_id=device_id, log_dir=log_dir, classes_per_user=classes_per_user,
                   levels = levels, use_kmeans = use_kmeans, misc_extra_bits = misc_extra_bits, quant_epochs = quant_epochs, quant_lr = quant_lr,
                   quantize_type = quantize_type, global_levels=global_levels, all_train_task_levels = all_train_task_levels, do_mapping=do_mapping)
  
  
  

  meta_learner.meta_train(criterion, optimizer, lr, device_id, epochs, log_dir, weight_decay=weight_decay)
  
  meta_learner.train_eval_test_tasks(lr,device_id,criterion,epochs, weight_decay=weight_decay, use_aid=use_aid,)

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
