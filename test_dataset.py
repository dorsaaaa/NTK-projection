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

from data.Data_gen import data_gen
import numpy as np
import json

from data.dataset import gen_random_loaders
from data.Data_Path import get_data_path


def main(seed=137, device_id=0, distributed=False, data_dir=None, log_dir=None,
        train_subset=1, indices_path=None, label_noise=0, num_workers=2,
         cfg_path=None, transfer=False, model_name='FcNet3', base_width=8,
         batch_size=64, optimizer='adam', lr=1e-2, momentum=.9, weight_decay=5e-4, epochs=2, global_levels=20, all_train_task_levels = 20,
         intrinsic_dim=600, intrinsic_mode='filmrdkron',is_said = False,is_layered = False, warmup_epochs=0, warmup_lr=.1, 
         levels = 5, use_kmeans = True, misc_extra_bits = 7, quant_epochs = 30, quant_lr = 0.0001, quantize_type = 'default', classes_per_user = 2,
         basis_num = 5, data_seed=42, data_transform = 'Shuffled_Pixels',n_train_tasks=10, n_test_tasks=20, samples_per_train=600, samples_per_test=600):

  
  random_seed_all(seed)

  #train_loader, test_loader = data_gen(seed=data_seed, dataset = 'MNIST', n_train_tasks=n_train_tasks, n_test_tasks = n_test_tasks, samples_per_train=samples_per_train, samples_per_test=samples_per_test)

  train_loader, test_loader = gen_random_loaders(data_name='cifar100',data_path=get_data_path(),num_users=n_train_tasks+n_test_tasks,num_train_users=n_train_tasks,bz=128,partition_type='by_class',classes_per_user=classes_per_user,)
  print("loaded data successfully")
  
  #print(train_loader[0]['train'].dataset)
  #print(test_loader)
  
  for i in range(n_train_tasks):
    print(len(train_loader[i]['train'].dataset),len(train_loader[i]['test'].dataset))
  for i in range(n_test_tasks):
    print(len(test_loader[i]['train'].dataset),len(test_loader[i]['test'].dataset))
  
  
  
  for i in range(1):
    t1 = iter(train_loader[i]['train'])
    images, labels = next(t1)
    print(labels)
  
  '''
  t1 = iter(train_loaders[0])
  images, labels = next(t1)
  print(len(labels))
  t1 = iter(val_loaders[0])
  images, labels = next(t1)
  print(len(labels))
  t1 = iter(test_loaders[0])
  images, labels = next(t1)
  print(len(labels))
  '''
  
  '''
  fig, axes = plt.subplots(1, 20, figsize=(15, 2))
  for i in range(20):  
      ax = axes[i]
      image = images[i].permute(1, 2, 0).numpy() 
      ax.imshow(image)
      ax.set_title(f"{np.argmax(labels[i])}")
      ax.axis('off')

  plt.savefig('cifar_test2.png', format='png')
  print("saved pic")
  '''
  

  return 
  
  
  


def entrypoint(log_dir=None, **kwargs):
  world_size, rank, device_id = maybe_launch_distributed()

  torch.backends.cudnn.benchmark = True
  torch.cuda.set_device(device_id)

  results = main(**kwargs, log_dir=log_dir, distributed=(world_size > 1), device_id=device_id)



if __name__ == '__main__':
  import fire
  fire.Fire(entrypoint)
