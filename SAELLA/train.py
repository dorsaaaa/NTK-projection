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
import pickle

from pactl.logging import set_logging, wandb, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from experiments.compute_bound import compute_bound

from SAELLA.meta_learner import SAELLA
from data.Data_gen import data_gen




def main(seed=137, device_id=7, distributed=False, data_dir=None, log_dir=None,
        train_subset=1, indices_path=None, label_noise=0, num_workers=2,
         cfg_path=None, transfer=False, model_name='FcNet3', base_width=8,
         batch_size=64, optimizer='adam', lr=1e-2, momentum=.9, weight_decay=5e-4, epochs=2, global_levels=20, all_train_task_levels = 20
         intrinsic_dim=600, intrinsic_mode='filmrdkron',is_said = False,is_layered = False, warmup_epochs=0, warmup_lr=.1, is_lambda_global=True,
         levels = 5, use_kmeans = True, misc_extra_bits = 7, quant_epochs = 30, quant_lr = 0.0001, quantize_type = 'default|default_all_tasks', is_lambda_binary=False, is_base_different=False,
         basis_num = 5, data_seed=42, data_transform = 'Shuffled_Pixels',n_train_tasks=10, n_test_tasks=20, samples_per_train=600, samples_per_test=600):

  train_loader, test_loader = data_gen(seed=data_seed, data_transform = data_transform, n_train_tasks=n_train_tasks, n_test_tasks = n_test_tasks, samples_per_train=samples_per_train, samples_per_test=samples_per_test)
  random_seed_all(seed)

  print("loaded data successfully")
  '''
  data_iter = iter(train_loader[0]['train'])
  images, labels = next(data_iter)
  
  fig, axes = plt.subplots(1, 20, figsize=(15, 2))
  for i in range(20):  
      ax = axes[i]
      image = images[i].squeeze().numpy() 
      ax.imshow(image, cmap='gray')
      ax.set_title(f"{labels[i].item()}")
      ax.axis('off')

  plt.savefig('Permute_Labels.png', format='png')
  print("saved pic")
  '''
  criterion = nn.CrossEntropyLoss()
  

  #meta learning part
  meta_learner = SAELLA(train_loader, test_loader, intrinsic_dim, basis_num = basis_num, model_name=model_name, base_width=base_width, is_base_different=is_base_different,
                   seed=seed, intrinsic_mode=intrinsic_mode,cfg_path=cfg_path, transfer=transfer, device_id=device_id, log_dir=log_dir, is_lambda_global = is_lambda_global, is_lambda_binary=is_lambda_binary,
                   levels = levels, use_kmeans = use_kmeans, misc_extra_bits = misc_extra_bits, quant_epochs = quant_epochs, quant_lr = quant_lr,
                   quantize_type = quantize_type, global_levels=global_levels, all_train_task_levels = all_train_task_levels)
  
  #print(meta_learner.global_P)
  meta_learner.meta_train(criterion, optimizer, lr, device_id, epochs, log_dir, weight_decay=weight_decay)
  
  '''
  #save model
  filename = 'SAELLA/meta_learner.txt'
  dict={}
  param = meta_learner.global_param.detach().cpu().numpy()
  dict['global_param']= param
  dict['global_lambdas']=meta_learner.global_lambdas.detach().cpu().numpy()
  for i in range(meta_learner.num_train_tasks):
    dict['local_param_'+str(i)]= meta_learner.nets[i].subspace_params.detach().cpu().numpy()
  text_dict = {key: value.tolist() for key, value in dict.items()}
  # Save the dictionary as a JSON in a .txt file
  with open(filename, 'w') as f:
      json.dump(text_dict, f, indent=4)
        
  '''
  meta_learner.train_eval_test_tasks(lr,device_id,criterion,epochs, weight_decay=weight_decay)
  '''
  #save model
  filename = 'SAELLA/meta_learner_after_quant.txt'
  dict={}
  param = meta_learner.global_param.detach().cpu().numpy()
  dict['global_param']= param
  dict['global_lambdas']=meta_learner.global_lambdas.detach().cpu().numpy()
  for i in range(meta_learner.num_train_tasks):
    dict['local_param_'+str(i)]= meta_learner.nets[i].subspace_params.detach().cpu().numpy()
  text_dict = {key: value.tolist() for key, value in dict.items()}
  # Save the dictionary as a JSON in a .txt file
  with open(filename, 'w') as f:
      json.dump(text_dict, f, indent=4)
  print(f"Parameters reloaded and saved to {filename}")
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

  results = main(**kwargs, log_dir=log_dir, distributed=(world_size > 1), device_id=device_id)

  if rank == 0:
    finish_logging()


if __name__ == '__main__':
  import fire
  fire.Fire(entrypoint)
