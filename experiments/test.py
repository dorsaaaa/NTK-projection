import sys
import os 
sys.path.append(os.getcwd())
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, RMSprop, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler


from pactl.distributed import maybe_launch_distributed
from pactl.logging import set_logging, wandb, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.nn import create_model
from pactl.optim.third_party.functional_warm_up import LinearWarmupScheduler
from pactl.optim.schedulers import construct_stable_cosine
from pactl.optim.schedulers import construct_warm_stable_cosine
from experiments.meta_learning import Meta
from data.Data_gen import data_gen
from pactl.nn.projectors import create_intrinsic_model
from pactl.nn import FcNet3,ConvNet3,CNNTarget, MLPModel, FCNTK, CNTK
from pactl.bounds.get_bound_from_chk_v2 import my_quantize_vector, eval_acc_and_bound, compute_quantization, total_nll_bits, quantize
from experiments.compute_bound import compute_bound_single_task

from data.dataset import gen_random_loaders
from data.Data_Path import get_data_path
from data.folktables_dataset import get_folktables_dataset
from data.products_dataset import get_products_dataset

def eval_model(model, data_loader, device_id, partition = None):
  model.eval()
  N = len(data_loader.dataset)
  N_acc = 0
  
  for i, (X, Y) in enumerate(data_loader):
    
    X, Y = X.to(device_id), Y.to(device_id)
    logits = model(X)
    if partition is not None :
      mask = logits[:, partition]
      max_indices_partition = mask.argmax(dim=-1)
      max_indices = torch.tensor(partition, device=device_id)[max_indices_partition]
    else:
      max_indices = logits.argmax(dim=-1)
    
    N_acc += (max_indices == Y.argmax(dim=-1)).sum() if Y.ndim > 1 else (max_indices == Y).sum()
  metrics = {'acc': N_acc.item() / N}
  return metrics


def main(seed=137, device_id=5, distributed=False, data_dir=None, log_dir=None,
        train_subset=1, indices_path=None, label_noise=0, num_workers=2,
         cfg_path=None, transfer=False, model_name='FcNet3', base_width=8,
         batch_size=64, optimizer='adam', lr=1e-3, momentum=.9, weight_decay=5e-4, epochs=2,dataset='MNIST', N_Way=10,
         intrinsic_dim=3500, intrinsic_mode='filmrdkron',is_said = False,is_layered = False, warmup_epochs=0, warmup_lr=.1, do_mapping=True,
         levels = 20, use_kmeans = True, misc_extra_bits = 7, quant_epochs = 30, quant_lr = 0.0001, quantize_type = 'default', classes_per_user=6,
        samples_per_train = 600, samples_per_test = 600, data_seed=42, n_train_tasks = 20, n_test_tasks = 0, data_transform = 'Shuffled_Pixels'):

    
    if dataset == 'MNIST':
      train_loaders, test_loaders = data_gen(seed=data_seed, dataset=dataset, data_transform = data_transform, n_train_tasks=n_train_tasks, n_test_tasks = n_test_tasks, samples_per_train=samples_per_train, samples_per_test=samples_per_test)
    elif dataset == 'CIFAR10':
      train_loaders, test_loaders = gen_random_loaders(data_name='cifar10', data_path=get_data_path(), num_users=n_train_tasks + n_test_tasks, num_train_users=n_train_tasks, bz=batch_size, partition_type='by_class', classes_per_user=classes_per_user, seed=data_seed, do_mapping=do_mapping)
    elif dataset == 'CIFAR100':
      train_loaders, test_loaders = gen_random_loaders(data_name='cifar100', data_path=get_data_path(), num_users=n_train_tasks + n_test_tasks, num_train_users=n_train_tasks, bz=batch_size, partition_type='by_class', classes_per_user=classes_per_user, seed=data_seed, do_mapping=do_mapping)
    elif dataset =='folktables':
      train_loaders, test_loaders = get_folktables_dataset(num_train_tasks=n_train_tasks, num_test_tasks=n_test_tasks, num_samples=samples_per_train, seed=data_seed)
    elif dataset =='products':
      train_loaders, test_loaders = get_products_dataset(num_train_tasks=n_train_tasks, num_test_tasks=n_test_tasks, num_samples=samples_per_train, seed=data_seed)

    random_seed_all(seed)
    print("loaded data successfully")

    train, test = train_loaders[0]['train'],train_loaders[0]['test']
    print(len(train.dataset), len(test.dataset))
    data_iter = iter(train)
    inputs, labels = next(data_iter)  
    in_chans = inputs.shape[1]
    num_classes = len(set(labels.numpy())) if labels.ndim == 1 else labels.shape[1]
    print(in_chans, num_classes)
    

    avg_train_acc, avg_test_acc = 0, 0

    for t in range(n_train_tasks):
      train_loader, test_loader = train_loaders[t]['train'],train_loaders[t]['test']
      #create model
      if model_name == 'FcNet3':
        base_net = FcNet3(input_shape=(1, 28, 28), output_dim=10)
      elif model_name == 'ConvNet3':
        base_net = ConvNet3(input_shape=(1, 28, 28), output_dim=10)
      elif model_name =='CNNTarget':
        base_net = CNNTarget(in_channels=in_chans, out_dim=num_classes)  
      elif model_name =='MLPModel':
        base_net = MLPModel(input_dim=in_chans, out_dim=num_classes)
      elif model_name =='FCNTL':
        base_net = FCNTK(in_dim=in_chans, hidden_dim=8192, num_classes=num_classes)
      elif model_name =='CNTK':
        base_net = CNTK(in_chans=in_chans, base_width=512, num_classes=num_classes )
      else:
      #TODO: 
        base_net = create_model(model_name=model_name, num_classes=num_classes, in_chans=in_chans, base_width=base_width,
                       seed=seed, intrinsic_dim=intrinsic_dim, intrinsic_mode=intrinsic_mode,
                       cfg_path=cfg_path, transfer=transfer, device_id=device_id, log_dir=log_dir)

      net = base_net.to(device_id)
      partition = train_loaders[t]['partitions'] if not do_mapping else None

      
      optim = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
      criterion = nn.CrossEntropyLoss()
      #train model
      net.train()
      for e in tqdm(range(epochs)):     
          for i, (X, Y) in enumerate(train_loader):

              X, Y = X.to(device_id), Y.to(device_id)
              optim.zero_grad()

              f_hat = net(X)
              loss = criterion(f_hat[:,partition], Y[:,partition]) if partition is not None else criterion(f_hat, Y)

              loss.backward()
              optim.step()

      train_metrics = eval_model(net, train_loader, device_id=device_id, partition=partition)['acc']
      test_metrics = eval_model(net, test_loader, device_id=device_id, partition=partition)['acc']
      avg_train_acc += train_metrics
      avg_test_acc += test_metrics

    print("average train accuracy: ",avg_train_acc/n_train_tasks)
    print("average test accuracy: ",avg_test_acc/n_train_tasks)

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


